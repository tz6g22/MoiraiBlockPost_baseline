from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from safetensors.torch import load_file
import torch
from transformers import AutoTokenizer

from src.baselines.modeling import BaselineDecoderLayer, BaselineQwen3ForCausalLM
from src.baselines.training import (
    _checkpoint_identity,
    _configure_execution,
    frozen_backbone_hash,
    pseudo_query_hash,
    query_parameter_names,
    validate_baseline_config,
)
from src.common import load_yaml, sha256_json
from src.data.format_tasks import (
    encode_prompt_only,
    format_task_target,
    load_local_split,
    load_manifest,
)
from src.data.prepare_post_data import _entry, _ordered_unique
from src.distributed.fsdp_utils import (
    DistributedContext,
    barrier,
    broadcast_object,
    destroy_distributed,
    init_distributed,
    wrap_qwen3_fsdp,
)


EVALUATION_CASES = 10
MAXIMUM_NEW_TOKENS = 256
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")


def _extract_number(text: str) -> str:
    candidate = text.rsplit("####", 1)[1] if "####" in text else text
    matches = _NUMBER_PATTERN.findall(candidate)
    if not matches:
        return ""
    value = matches[-1].replace(",", "").replace(" ", "").rstrip(".")
    return value[1:] if value.startswith("+") else value


def _numeric_accuracy(prediction: str, gold: str) -> float:
    predicted_number = _extract_number(prediction)
    gold_number = _extract_number(gold)
    if not predicted_number or not gold_number:
        return 0.0
    try:
        return float(Decimal(predicted_number) == Decimal(gold_number))
    except InvalidOperation:
        return 0.0


def _select_svamp_cases(data_config: dict, manifest: list[dict]):
    source = data_config["sources"]["svamp"]
    split = str(source["official_split"])
    dataset = load_local_split(source["local_path"], split)
    ordered = _ordered_unique(
        dataset,
        task="math",
        source=source,
        split=split,
        seed=int(data_config["split_seed"]),
    )
    used_ids = {str(record["stable_id"]) for record in manifest}
    used_content = {str(record["content_sha256"]) for record in manifest}
    selected: list[dict] = []
    for _key, row_index, row in ordered:
        record = _entry(
            task="math",
            source=source,
            split=split,
            row_index=row_index,
            row=row,
            assigned_split="baseline_svamp_eval_10",
            seed=int(data_config["split_seed"]),
        )
        if (
            record["stable_id"] in used_ids
            or record["content_sha256"] in used_content
        ):
            continue
        selected.append(record)
        used_ids.add(record["stable_id"])
        used_content.add(record["content_sha256"])
        if len(selected) == EVALUATION_CASES:
            break
    if len(selected) != EVALUATION_CASES:
        raise RuntimeError("Could not select 10 unused SVAMP evaluation cases")
    return selected, dataset, source["field_mapping"]


def _load_trained_baseline(
    *,
    baseline_type: str,
    task: str,
    config: dict,
    context: DistributedContext,
):
    if task not in {"math", "multihop"}:
        raise ValueError(f"Unsupported baseline evaluation task: {task}")
    output_dir = Path(config["output_root"])
    manifest_path = output_dir / "manifest.json"
    query_path = output_dir / "query.safetensors"
    if not manifest_path.is_file() or not query_path.is_file():
        raise FileNotFoundError(
            f"Trained {task} baseline is missing under {output_dir}"
        )
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_per_task = int(config["training_cases_per_task"])
    expected_total = expected_per_task * len(config["training_task_order"])
    if (
        run_manifest.get("status") != "PASS"
        or run_manifest.get("baseline_type") != baseline_type
        or run_manifest.get("tasks") != ["math", "multihop"]
        or run_manifest.get("shared_query_across_tasks") is not True
        or run_manifest.get("training_case_count_by_task", {}).get(task)
        != expected_per_task
        or run_manifest.get("unique_case_count") != expected_total
        or run_manifest.get("repeat_count") != 0
    ):
        raise ValueError(f"Baseline training manifest is not a valid {task} run")

    checkpoint = Path(config["base_checkpoint"])
    base_hash, _ = broadcast_object(
        _checkpoint_identity(checkpoint) if context.is_rank0 else None,
        context,
    )
    if run_manifest.get("base_checkpoint_hash") != base_hash:
        raise ValueError("Baseline evaluation base checkpoint hash differs from training")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    model = BaselineQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _configure_execution(model, baseline_type=baseline_type, config=config)
    if frozen_backbone_hash(model) != run_manifest.get("backbone_after_hash"):
        raise ValueError("Baseline backbone differs from the frozen training backbone")

    query_state = load_file(query_path, device="cpu")
    expected_names = set(query_parameter_names(model))
    if set(query_state) != expected_names:
        raise ValueError("Saved baseline query parameter set is incomplete")
    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in sorted(expected_names):
            named_parameters[name].copy_(
                query_state[name].to(
                    device=named_parameters[name].device,
                    dtype=named_parameters[name].dtype,
                )
            )
    if pseudo_query_hash(model) != run_manifest.get("query_after_hash"):
        raise ValueError("Loaded baseline query hash differs from training manifest")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(BaselineDecoderLayer,),
    )
    return model, tokenizer, run_manifest


@torch.no_grad()
def _greedy_generate(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    eos_token_id: int,
    maximum_new_tokens: int = MAXIMUM_NEW_TOKENS,
) -> torch.LongTensor:
    generated = input_ids
    mask = attention_mask
    prompt_length = generated.shape[1]
    for _ in range(maximum_new_tokens):
        output = model(
            input_ids=generated,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
        mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        if bool(torch.all(next_token == eos_token_id)):
            break
    return generated[:, prompt_length:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=("full", "fixed"))
    args = parser.parse_args()
    context = init_distributed()

    baseline_type = (
        "full_attnres" if args.baseline == "full" else "fixed_block_attnres"
    )
    config_path = Path("configs/baselines") / f"{args.baseline}.yaml"
    config = load_yaml(config_path)
    validate_baseline_config(config, expected_type=baseline_type)
    model, tokenizer, training_manifest = _load_trained_baseline(
        baseline_type=baseline_type,
        task="math",
        config=config,
        context=context,
    )

    data_config = load_yaml("configs/data.yaml")
    split_manifest = load_manifest(data_config["output_dir"] + "/splits.json")
    records, dataset, mapping = _select_svamp_cases(data_config, split_manifest)
    predictions: list[dict] = []
    correct = 0
    for index, record in enumerate(records, start=1):
        row = dataset[int(record["row_index"])]
        prompt = encode_prompt_only(
            tokenizer,
            task="math",
            row=row,
            field_mapping=mapping,
            stable_id=record["stable_id"],
            max_length=2048 - MAXIMUM_NEW_TOKENS,
        )
        generated_ids = _greedy_generate(
            model,
            prompt.input_ids.unsqueeze(0).to(context.device),
            prompt.attention_mask.unsqueeze(0).to(context.device),
            eos_token_id=tokenizer.eos_token_id,
        )
        prediction = tokenizer.decode(
            generated_ids[0].cpu(),
            skip_special_tokens=True,
        )
        gold = format_task_target("math", row, mapping)
        accuracy = _numeric_accuracy(prediction, gold)
        correct += int(accuracy)
        result = {
            "case": index,
            "stable_id": record["stable_id"],
            "row_index": int(record["row_index"]),
            "question": row[mapping["question"]],
            "gold": gold,
            "prediction": prediction,
            "accuracy": accuracy,
        }
        predictions.append(result)
        if context.is_rank0:
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    output_dir = Path(config["output_root"]) / "evaluation_svamp_10"
    if context.is_rank0:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "predictions.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in predictions
            ),
            encoding="utf-8",
        )
    result_payload = {
        "baseline_type": baseline_type,
        "task": "math",
        "dataset": "svamp",
        "evaluation_cases": EVALUATION_CASES,
        "correct": correct,
        "accuracy": correct / EVALUATION_CASES,
        "maximum_new_tokens": MAXIMUM_NEW_TOKENS,
        "selection": "seed42 ordered, excluding all manifest ID/content hashes",
        "stable_ids": [record["stable_id"] for record in records],
        "evaluation_selection_sha256": sha256_json(
            [record["stable_id"] for record in records]
        ),
        "training_query_after_hash": training_manifest["query_after_hash"],
        "training_case_selection_sha256": training_manifest["case_selection_sha256"],
    }
    if context.is_rank0:
        (output_dir / "evaluation_results.json").write_text(
            json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result_payload, ensure_ascii=False, indent=2, sort_keys=True))
    barrier(context)
    destroy_distributed(context)


if __name__ == "__main__":
    main()
