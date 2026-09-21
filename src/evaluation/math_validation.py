from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.common import load_yaml, sha256_file
from src.data.format_tasks import (
    TargetCausalExample,
    canonical_content_sha256,
    canonical_stable_id,
    dataset_pool_sources,
    encode_prompt_target,
    format_task_prompt,
    format_task_target,
    load_local_split,
    load_manifest,
    nested_value,
)


def default_math_validation_manifest(repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / "outputs/formal_retrain/shared/math_validation_manifest.json"


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_json(value: Any) -> str:
    return _hash_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def _hash_lines(values: Sequence[str]) -> str:
    return _hash_text("".join(f"{value}\n" for value in values))


def _source_row(
    record: Mapping[str, Any],
    *,
    data_config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dataset_name = str(record["dataset"])
    official_split = str(record["official_split"])
    sources = [
        source
        for source in dataset_pool_sources(dict(data_config), dataset_name)
        if str(source["official_split"]) == official_split
    ]
    if len(sources) != 1:
        raise RuntimeError(
            f"CANONICAL_MATH_VALIDATION_SOURCE_AMBIGUOUS: {dataset_name}/{official_split}"
        )
    source = sources[0]
    local_path = Path(str(source["local_path"]))
    if not local_path.is_absolute():
        local_path = repo_root / local_path
    dataset = load_local_split(local_path, official_split)
    row_index = int(record["row_index"])
    if row_index < 0 or row_index >= len(dataset):
        raise RuntimeError(f"CANONICAL_MATH_VALIDATION_ROW_OUT_OF_RANGE: {record['stable_id']}")
    row = dataset[row_index]
    if canonical_stable_id(dataset_name, official_split, row, source["field_mapping"]) != str(record["stable_id"]):
        raise RuntimeError(f"CANONICAL_MATH_VALIDATION_STABLE_ID_MISMATCH: {record['stable_id']}")
    if canonical_content_sha256(dataset_name, row, source["field_mapping"]) != str(record["content_sha256"]):
        raise RuntimeError(f"CANONICAL_MATH_VALIDATION_CONTENT_HASH_MISMATCH: {record['stable_id']}")
    return row, source["field_mapping"], source


def _validate_manifest(
    manifest_path: str | Path,
    *,
    data_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical Math validation manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "canonical_math_validation_v1":
        raise RuntimeError("CANONICAL_MATH_VALIDATION_SCHEMA_MISMATCH")
    records = payload.get("records")
    if not isinstance(records, list) or int(payload.get("count", -1)) != len(records) or len(records) != 32:
        raise RuntimeError("CANONICAL_MATH_VALIDATION_COUNT_MISMATCH")
    stable_ids = [str(record.get("stable_id", "")) for record in records]
    content_hashes = [str(record.get("content_sha256", "")) for record in records]
    if any(not value for value in stable_ids + content_hashes) or len(set(stable_ids)) != len(stable_ids):
        raise RuntimeError("CANONICAL_MATH_VALIDATION_DUPLICATE_IDENTITY")
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError("CANONICAL_MATH_VALIDATION_DUPLICATE_CONTENT")
    for record in records:
        if (
            record.get("task", "math") != "math"
            or record.get("dataset") != "gsm8k"
            or record.get("assigned_split") != "stage3_adapter_val"
            or record.get("official_split") != "train"
        ):
            raise RuntimeError("CANONICAL_MATH_VALIDATION_RECORD_MISMATCH")
    if [int(record.get("order_index", -1)) for record in records] != list(range(len(records))):
        raise RuntimeError("CANONICAL_MATH_VALIDATION_ORDER_MISMATCH")
    if payload.get("stable_id_sha256") != _hash_lines(stable_ids):
        raise RuntimeError("CANONICAL_MATH_VALIDATION_ID_HASH_MISMATCH")
    if payload.get("content_sha256") != _hash_lines(content_hashes):
        raise RuntimeError("CANONICAL_MATH_VALIDATION_CONTENT_HASH_MISMATCH")
    if data_manifest_path is not None:
        actual = sha256_file(data_manifest_path)
        if actual != payload.get("data_manifest_sha256"):
            raise RuntimeError("CANONICAL_MATH_VALIDATION_DATA_MANIFEST_MISMATCH")
    return payload


def build_canonical_math_validation_manifest(
    *,
    shared_manifest_path: str | Path,
    data_config_path: str | Path,
    tokenizer,
    output_path: str | Path,
    count: int = 32,
    repo_root: str | Path | None = None,
) -> Path:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    shared_path = Path(shared_manifest_path)
    if not shared_path.is_absolute():
        shared_path = root / shared_path
    config_path = Path(data_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    data_config = load_yaml(config_path)
    records = load_manifest(shared_path)
    selected = sorted(
        (
            record
            for record in records
            if record.get("task") == "math"
            and record.get("dataset") == "gsm8k"
            and record.get("assigned_split") == "stage3_adapter_val"
        ),
        key=lambda record: str(record["stable_id"]),
    )[: int(count)]
    if len(selected) != int(count):
        raise RuntimeError(f"CANONICAL_MATH_VALIDATION_NOT_ENOUGH_ROWS: {len(selected)}")
    forbidden_stable_ids = {
        str(record.get("stable_id"))
        for record in records
        if record.get("assigned_split") in {"stage2_discovery", "stage3_adapter_train"}
    }
    forbidden_content_hashes = {
        str(record.get("content_sha256", ""))
        for record in records
        if record.get("assigned_split") in {"stage2_discovery", "stage3_adapter_train"}
    }
    for record in selected:
        if str(record["stable_id"]) in forbidden_stable_ids:
            raise RuntimeError(f"CANONICAL_MATH_VALIDATION_STABLE_ID_LEAKAGE: {record['stable_id']}")
        if str(record.get("content_sha256", "")) in forbidden_content_hashes:
            raise RuntimeError(f"CANONICAL_MATH_VALIDATION_LEAKAGE: {record['stable_id']}")
        _source_row(record, data_config=data_config, repo_root=root)

    output_records: list[dict[str, Any]] = []
    for order_index, record in enumerate(selected):
        row, mapping, source = _source_row(record, data_config=data_config, repo_root=root)
        prompt = format_task_prompt("math", row, mapping)
        target = format_task_target("math", row, mapping)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False) + [int(tokenizer.eos_token_id)]
        output_records.append(
            {
                "task": "math",
                "order_index": order_index,
                "stable_id": str(record["stable_id"]),
                "content_sha256": str(record["content_sha256"]),
                "dataset": str(record["dataset"]),
                "source": str(record["dataset"]),
                "assigned_split": str(record["assigned_split"]),
                "split": str(record["assigned_split"]),
                "official_split": str(record["official_split"]),
                "row_index": int(record["row_index"]),
                "question_sha256": _hash_text(str(nested_value(row, str(mapping["question"]))),),
                "prompt_sha256": _hash_text(prompt),
                "target_sha256": _hash_text(target),
                "prompt_token_sha256": _hash_json(prompt_ids),
                "target_token_sha256": _hash_json(target_ids),
            }
        )
    stable_ids = [record["stable_id"] for record in output_records]
    content_hashes = [record["content_sha256"] for record in output_records]
    payload = {
        "schema": "canonical_math_validation_v1",
        "task": "math",
        "count": len(output_records),
        "order": "stable_id_ascending",
        "data_manifest": str(shared_path),
        "data_manifest_sha256": sha256_file(shared_path),
        "stable_id_sha256": _hash_lines(stable_ids),
        "content_sha256": _hash_lines(content_hashes),
        "records": output_records,
    }
    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_manifest(output, data_manifest_path=shared_path)
    return output


def load_canonical_math_validation_examples(
    *,
    manifest_path: str | Path,
    data_manifest_path: str | Path,
    data_config_path: str | Path,
    tokenizer,
    max_length: int,
    repo_root: str | Path | None = None,
) -> tuple[TargetCausalExample, ...]:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    payload = _validate_manifest(manifest_path, data_manifest_path=data_manifest_path)
    config_path = Path(data_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    data_config = load_yaml(config_path)
    examples: list[TargetCausalExample] = []
    for record in payload["records"]:
        row, mapping, _source = _source_row(record, data_config=data_config, repo_root=root)
        example = encode_prompt_target(
            tokenizer,
            task="math",
            row=row,
            field_mapping=mapping,
            stable_id=str(record["stable_id"]),
            max_length=int(max_length),
        )
        prompt_ids = tokenizer.encode(format_task_prompt("math", row, mapping), add_special_tokens=False)
        target_ids = tokenizer.encode(format_task_target("math", row, mapping), add_special_tokens=False)
        target_ids.append(int(tokenizer.eos_token_id))
        if _hash_json(prompt_ids) != record.get("prompt_token_sha256") or _hash_json(target_ids) != record.get("target_token_sha256"):
            raise RuntimeError(f"CANONICAL_MATH_VALIDATION_TOKEN_HASH_MISMATCH: {record['stable_id']}")
        examples.append(example)
    return tuple(examples)


def evaluate_causal_lm(
    model,
    examples: Sequence[Any],
    *,
    pad_token_id: int,
    device: torch.device,
    distributed_context: Any | None = None,
    return_per_example: bool = False,
) -> dict[str, Any]:
    was_training = bool(model.training)
    model.eval()
    if isinstance(distributed_context, Mapping):
        rank = int(distributed_context.get("rank", 0))
        world_size = int(distributed_context.get("world_size", 1))
        is_distributed = bool(distributed_context.get("distributed", False))
    else:
        rank = int(getattr(distributed_context, "rank", 0)) if distributed_context is not None else 0
        world_size = int(getattr(distributed_context, "world_size", 1)) if distributed_context is not None else 1
        is_distributed = bool(getattr(distributed_context, "distributed", False)) if distributed_context is not None else False
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    token_count = torch.zeros((), dtype=torch.float32, device=device)
    per_example: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, example in enumerate(examples):
            if index % world_size != rank:
                continue
            labels = example.labels.clone()
            target_mask = getattr(example, "target_mask", None)
            if target_mask is not None:
                labels = labels.masked_fill(~target_mask, -100)
            input_ids = example.input_ids.unsqueeze(0).to(device)
            attention_mask = example.attention_mask.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            valid = labels != -100
            if not bool(valid.any()):
                continue
            example_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            example_tokens = int(valid.sum().item())
            loss_sum += example_sum.float()
            token_count += example_tokens
            if return_per_example:
                per_example.append(
                    {
                        "stable_id": str(example.stable_id),
                        "loss": float((example_sum / example_tokens).cpu()),
                        "tokens": example_tokens,
                    }
                )
            del logits
    if distributed_context is not None and is_distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    if int(token_count.item()) <= 0:
        raise RuntimeError("MATH_VALIDATION_HAS_NO_SUPERVISED_TOKENS")
    value = loss_sum / token_count
    if not torch.isfinite(value):
        raise FloatingPointError("MATH_VALIDATION_LOSS_NAN_OR_INF")
    if was_training:
        model.train()
    return {
        "loss": float(value.cpu()),
        "tokens": int(token_count.item()),
        "per_example": per_example if return_per_example else None,
    }
