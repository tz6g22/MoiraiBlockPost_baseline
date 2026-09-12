from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer

from src.baselines.modeling import BaselineDecoderLayer, BaselineQwen3ForCausalLM
from src.common import load_yaml, sha256_file, sha256_json, tokenizer_sha256
from src.data.format_tasks import (
    TASK_TO_SOURCE,
    collate_target_examples,
    encode_prompt_target,
    load_dataset_pool,
    load_manifest,
)
from src.distributed.fsdp_utils import (
    DistributedContext,
    all_reduce_sum,
    barrier,
    broadcast_object,
    clip_grad_norm,
    destroy_distributed,
    init_distributed,
    selected_parameter_sha256,
    selected_parameter_state,
    trainable_parameter_names,
    wrap_qwen3_fsdp,
)


BaselineType = Literal["full_attnres", "fixed_block_attnres"]
TASKS = ("math", "multihop")
FIXED_PARTITION = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]


def query_parameter_names(model) -> tuple[str, ...]:
    names = tuple(
        sorted(name for name, _ in model.named_parameters() if "pseudo_query" in name)
    )
    if not names:
        raise RuntimeError("Baseline model has no pseudo-query parameters")
    return names


def freeze_except_pseudo_query(model) -> tuple[str, ...]:
    query_names = query_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in query_names)
    trainable = tuple(
        sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    if trainable != query_names:
        raise RuntimeError("Baseline trainable parameters are not query-only")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name not in query_names
    ):
        raise RuntimeError("Baseline backbone contains a trainable parameter")
    return query_names


def pseudo_query_hash(model) -> str:
    names = query_parameter_names(model)
    payload = {
        name: dict(model.named_parameters())[name].detach().float().cpu().tolist()
        for name in names
    }
    return sha256_json(payload)


def frozen_backbone_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if "pseudo_query" in name:
            continue
        value = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_baseline_config(
    config: dict[str, Any],
    *,
    expected_type: BaselineType,
) -> None:
    expected = {
        "baseline_type": expected_type,
        "seed": 42,
        "precision": "bf16",
        "use_cache": False,
        "training_source_split": "stage3_adapter_train",
        "selection_order": "stable_id",
        "training_cases_per_task": 200,
        "training_task_order": ["math", "multihop"],
        "training_passes": 1,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "gradient_clip_norm": 1.0,
        "scheduler": "constant",
        "maximum_sequence_length": 2048,
        "expected_num_transformer_blocks": 40,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"Baseline config mismatch for {key}: expected {value!r}, "
                f"got {config.get(key)!r}"
            )
    if expected_type == "full_attnres":
        if config.get("execution_mode") != "full":
            raise ValueError("Full baseline must use execution_mode=full")
        if config.get("fixed_partition") is not None:
            raise ValueError("Full baseline must not define a partition")
    else:
        if config.get("execution_mode") != "fixed":
            raise ValueError("Fixed baseline must use execution_mode=fixed")
        if config.get("fixed_partition") != FIXED_PARTITION:
            raise ValueError("Fixed baseline partition must contain ten four-layer blocks")
    for key in (
        "base_checkpoint",
        "data_manifest",
        "data_config",
        "output_root",
    ):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"Baseline config is missing {key}")


def _checkpoint_identity(checkpoint: Path) -> tuple[str, dict[str, Any]]:
    weight_files = sorted(checkpoint.glob("model*.safetensors"))
    if len(weight_files) != 1:
        raise RuntimeError("Baseline requires exactly one base model weight file")
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Baseline base checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(weight_files[0])
    if manifest.get("model_weights_sha256") != actual_hash:
        raise ValueError("Baseline base checkpoint weight hash mismatch")
    if int(manifest.get("num_hidden_layers", -1)) != 40:
        raise ValueError("Baseline base checkpoint must contain 40 Transformer blocks")
    if not manifest.get("q_full_parameter_names") or not manifest.get("q_full_sha256"):
        raise ValueError("Baseline base checkpoint query identity is missing")
    return actual_hash, manifest


def select_training_records(
    *,
    task: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"Unknown baseline task: {task}")
    records = load_manifest(config["data_manifest"])
    data_config = load_yaml(config["data_config"])
    source_key = TASK_TO_SOURCE[task]
    source = data_config["sources"][source_key]
    candidates = [
        record
        for record in records
        if record.get("assigned_split") == config["training_source_split"]
        and record.get("task") == task
        and record.get("dataset") == source["dataset_name"]
    ]
    candidates.sort(key=lambda record: str(record["stable_id"]))
    count = int(config["training_cases_per_task"])
    if len(candidates) < count:
        raise RuntimeError(
            f"Baseline {task} needs {count} training cases, found {len(candidates)}"
        )
    selected = candidates[:count]
    stable_ids = [str(record["stable_id"]) for record in selected]
    content_hashes = [str(record["content_sha256"]) for record in selected]
    if len(set(stable_ids)) != count or len(set(content_hashes)) != count:
        raise RuntimeError("Baseline selection is not unique")
    forbidden_splits = {
        "stage3_adapter_val",
        "probe_train",
        "probe_val",
        "stage4_final_eval",
    }
    selected_ids = set(stable_ids)
    selected_content = set(content_hashes)
    for record in records:
        if record.get("assigned_split") not in forbidden_splits:
            continue
        if (
            str(record.get("stable_id")) in selected_ids
            or str(record.get("content_sha256")) in selected_content
        ):
            raise RuntimeError("Baseline training selection leaks into a forbidden split")
    return selected, source


def load_training_examples(
    *,
    task: str,
    config: dict[str, Any],
    tokenizer,
) -> tuple[tuple[Any, ...], list[str]]:
    records, source = select_training_records(task=task, config=config)
    pool = load_dataset_pool(load_yaml(config["data_config"]), source["dataset_name"])
    examples = []
    for record in records:
        dataset, field_mapping = pool[str(record["official_split"])]
        examples.append(
            encode_prompt_target(
                tokenizer,
                task=task,
                row=dataset[int(record["row_index"])],
                field_mapping=field_mapping,
                stable_id=str(record["stable_id"]),
                max_length=int(config["maximum_sequence_length"]),
            )
        )
    examples = tuple(examples)
    stable_ids = [example.stable_id for example in examples]
    expected_cases = int(config["training_cases_per_task"])
    if len(examples) != expected_cases or len(set(stable_ids)) != expected_cases:
        raise RuntimeError(
            f"Baseline must encode exactly {expected_cases} unique cases"
        )
    return examples, stable_ids


def _configure_execution(
    model,
    *,
    baseline_type: BaselineType,
    config: dict[str, Any],
) -> None:
    if int(model.config.num_hidden_layers) != int(
        config["expected_num_transformer_blocks"]
    ):
        raise ValueError("Baseline model depth does not match its config")
    if baseline_type == "full_attnres":
        model.config.baseline_execution = "full"
        model.config.baseline_partition = None
    else:
        partition = list(config["fixed_partition"])
        if partition != FIXED_PARTITION or sum(partition) != 40:
            raise ValueError("Fixed baseline partition is not the required 40-layer split")
        model.config.baseline_execution = "fixed"
        model.config.baseline_partition = partition
    model.config.use_cache = False


def _load_model_and_data(
    *,
    baseline_type: BaselineType,
    config: dict[str, Any],
    context: DistributedContext,
):
    checkpoint = Path(config["base_checkpoint"])
    base_hash, checkpoint_manifest = broadcast_object(
        _checkpoint_identity(checkpoint) if context.is_rank0 else None,
        context,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer_sha256(tokenizer) != checkpoint_manifest.get("tokenizer_sha256"):
        raise ValueError("Baseline tokenizer hash mismatch")
    examples_by_task: dict[str, tuple[Any, ...]] = {}
    stable_ids_by_task: dict[str, list[str]] = {}
    for task in config["training_task_order"]:
        examples, stable_ids = load_training_examples(
            task=task,
            config=config,
            tokenizer=tokenizer,
        )
        examples_by_task[task] = examples
        stable_ids_by_task[task] = stable_ids
    model = BaselineQwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if context.device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    _configure_execution(
        model,
        baseline_type=baseline_type,
        config=config,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    trainable_names = freeze_except_pseudo_query(model)
    if list(trainable_names) != checkpoint_manifest["q_full_parameter_names"]:
        raise ValueError("Baseline query parameter names differ from the base checkpoint")
    query_before = broadcast_object(
        pseudo_query_hash(model) if context.is_rank0 else None,
        context,
    )
    if query_before != checkpoint_manifest["q_full_sha256"]:
        raise ValueError("Baseline query initialization differs from the base checkpoint")
    backbone_before = broadcast_object(
        frozen_backbone_hash(model) if context.is_rank0 else None,
        context,
    )
    model = wrap_qwen3_fsdp(
        model,
        context,
        decoder_layer_classes=(BaselineDecoderLayer,),
    )
    return (
        model,
        tokenizer,
        examples_by_task,
        stable_ids_by_task,
        trainable_names,
        query_before,
        base_hash,
        backbone_before,
    )


def _target_token_count(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


def _single_example_batch(example, tokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    batch = collate_target_examples([example], pad_token_id=tokenizer.pad_token_id)
    target_mask = batch.pop("target_mask")
    labels = batch["labels"].masked_fill(~target_mask, -100)
    batch["labels"] = labels
    return {key: value.to(device) for key, value in batch.items()}


def check_only(
    *,
    baseline_type: BaselineType,
    config: dict[str, Any],
) -> dict[str, Any]:
    context = init_distributed(require_cuda=False)
    (
        model,
        tokenizer,
        examples_by_task,
        stable_ids_by_task,
        trainable_names,
        query_before,
        base_hash,
        _backbone_before,
    ) = _load_model_and_data(
        baseline_type=baseline_type,
        config=config,
        context=context,
    )
    model.eval()
    first_case_losses: dict[str, float] = {}
    for task in config["training_task_order"]:
        batch = _single_example_batch(
            examples_by_task[task][0], tokenizer, context.device
        )
        with torch.no_grad(), torch.autocast(
            device_type=context.device.type,
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda",
        ):
            loss = model(**batch, use_cache=False).loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(
                f"Baseline check-only {task} forward produced an invalid loss"
            )
        first_case_losses[task] = float(loss.detach().cpu())
        del batch, loss
    all_stable_ids = [
        stable_id
        for task in config["training_task_order"]
        for stable_id in stable_ids_by_task[task]
    ]
    result = {
        "status": "PASS",
        "baseline_type": baseline_type,
        "tasks": list(config["training_task_order"]),
        "device": str(context.device),
        "base_checkpoint_hash": base_hash,
        "num_transformer_blocks": int(model.config.num_hidden_layers),
        "fixed_partition": (
            list(model.config.baseline_partition)
            if model.config.baseline_partition is not None
            else None
        ),
        "training_case_count": len(all_stable_ids),
        "training_case_count_by_task": {
            task: len(examples_by_task[task])
            for task in config["training_task_order"]
        },
        "unique_case_count": len(set(all_stable_ids)),
        "repeat_count": len(all_stable_ids) - len(set(all_stable_ids)),
        "case_selection_sha256": sha256_json(all_stable_ids),
        "case_selection_sha256_by_task": {
            task: sha256_json(stable_ids_by_task[task])
            for task in config["training_task_order"]
        },
        "trainable_parameter_names": list(trainable_names),
        "all_non_query_parameters_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if "pseudo_query" not in name
        ),
        "query_before_hash": query_before,
        "first_case_loss_by_task": first_case_losses,
    }
    del model
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    is_rank0 = context.is_rank0
    barrier(context)
    destroy_distributed(context)
    return result if is_rank0 else {}


def train(
    *,
    baseline_type: BaselineType,
    config: dict[str, Any],
) -> dict[str, Any]:
    context = init_distributed()
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    (
        model,
        tokenizer,
        examples_by_task,
        stable_ids_by_task,
        trainable_names,
        query_before,
        base_hash,
        backbone_before,
    ) = _load_model_and_data(
        baseline_type=baseline_type,
        config=config,
        context=context,
    )
    output_dir = Path(config["output_root"])
    output_error = None
    if context.is_rank0:
        if output_dir.exists() and any(output_dir.iterdir()):
            output_error = f"Refusing to overwrite baseline output: {output_dir}"
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
    output_error = broadcast_object(output_error, context)
    if output_error is not None:
        raise FileExistsError(output_error)
    resolved_config = {
        **config,
        "tasks": list(config["training_task_order"]),
        "output_dir": str(output_dir),
        "resolved_training_stable_ids_by_task": stable_ids_by_task,
        "case_selection_sha256_by_task": {
            task: sha256_json(stable_ids_by_task[task])
            for task in config["training_task_order"]
        },
    }
    if context.is_rank0:
        (output_dir / "config.json").write_text(
            json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["eps"]),
        weight_decay=float(config["weight_decay"]),
    )
    if set(trainable_parameter_names(model)) != set(trainable_names):
        raise RuntimeError("Baseline optimizer does not contain exactly pseudo-query")

    model.train()
    metrics_path = output_dir / "training_metrics.jsonl"
    processed_ids: list[str] = []
    processed_ids_by_task: dict[str, list[str]] = {
        task: [] for task in config["training_task_order"]
    }
    total_input_tokens = 0
    total_target_tokens = 0
    losses: list[float] = []
    positive_finite_gradient_seen = False
    metrics_handle = metrics_path.open("x", encoding="utf-8") if context.is_rank0 else None
    try:
        training_stream = [
            (task, example)
            for task in config["training_task_order"]
            for example in examples_by_task[task]
        ]
        for step, (task, example) in enumerate(training_stream, start=1):
            if example.stable_id in processed_ids:
                raise RuntimeError("Baseline attempted to repeat a training case")
            batch = _single_example_batch(example, tokenizer, context.device)
            labels = batch.pop("labels")
            active = context.rank == (step - 1) % context.world_size
            input_tokens = int(batch["attention_mask"].sum().item()) if active else 0
            target_tokens = _target_token_count(labels) if active else 0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(**batch, use_cache=False)
            local_loss = (
                F.cross_entropy(
                    output.logits.float().reshape(-1, output.logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                if active
                else output.logits.float().sum() * 0.0
            )
            global_target_tokens = int(
                all_reduce_sum(
                    torch.tensor(target_tokens, device=context.device, dtype=torch.long),
                    context,
                ).item()
            )
            if global_target_tokens <= 0:
                raise RuntimeError("Baseline training case has no target tokens")
            loss = local_loss * context.world_size / global_target_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("Baseline training loss is invalid")
            loss.backward()
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if "pseudo_query" not in name
            ):
                raise RuntimeError("A frozen baseline parameter received a gradient")
            gradients = [
                parameter.grad.detach().float()
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("Baseline query gradients are missing or invalid")
            gradient_norm = float(
                clip_grad_norm(
                    model,
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    float(config["gradient_clip_norm"]),
                ).detach().float().cpu()
            )
            if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
                raise FloatingPointError("Baseline query gradient norm is not positive")
            positive_finite_gradient_seen = True
            optimizer.step()
            loss_value = float(
                all_reduce_sum(local_loss.detach(), context).cpu()
            ) / global_target_tokens
            input_tokens = int(
                all_reduce_sum(
                    torch.tensor(input_tokens, device=context.device, dtype=torch.long),
                    context,
                ).item()
            )
            target_tokens = global_target_tokens
            processed_ids.append(example.stable_id)
            processed_ids_by_task[task].append(example.stable_id)
            total_input_tokens += input_tokens
            total_target_tokens += target_tokens
            losses.append(loss_value)
            if metrics_handle is not None:
                metrics_handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "task": task,
                            "stable_id": example.stable_id,
                            "loss": loss_value,
                            "gradient_norm": gradient_norm,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "nonpadding_input_tokens": input_tokens,
                            "target_tokens": target_tokens,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                metrics_handle.flush()
            del batch, output, loss, gradients
    finally:
        if metrics_handle is not None:
            metrics_handle.close()

    expected_ids = [
        stable_id
        for task in config["training_task_order"]
        for stable_id in stable_ids_by_task[task]
    ]
    if processed_ids != expected_ids:
        raise RuntimeError("Baseline did not process its shared-query stream exactly once")
    expected_per_task = int(config["training_cases_per_task"])
    expected_total = expected_per_task * len(config["training_task_order"])
    if len(processed_ids) != expected_total or len(set(processed_ids)) != expected_total:
        raise RuntimeError("Baseline one-pass case count verification failed")
    if any(
        len(processed_ids_by_task[task]) != expected_per_task
        for task in TASKS
    ):
        raise RuntimeError("Baseline per-task one-pass count verification failed")
    query_state = selected_parameter_state(
        model,
        context,
        lambda name, _parameter: "pseudo_query" in name,
    )
    query_after = None
    if context.is_rank0:
        assert query_state is not None
        query_after = sha256_json(
            {name: value.detach().float().cpu().tolist() for name, value in query_state.items()}
        )
    query_after = broadcast_object(query_after, context)
    backbone_after = selected_parameter_sha256(
        model,
        context,
        lambda name, _parameter: "pseudo_query" not in name,
    )
    if backbone_before != backbone_after:
        raise RuntimeError("Baseline frozen backbone changed during training")
    if query_before == query_after:
        raise RuntimeError("Baseline pseudo-query did not change during training")
    if not positive_finite_gradient_seen:
        raise RuntimeError("Baseline never observed a positive finite query gradient")
    query_path = output_dir / "query.safetensors"
    if context.is_rank0:
        assert query_state is not None
        save_file(query_state, query_path)
    barrier(context)
    manifest = {
        "status": "PASS",
        "baseline_type": baseline_type,
        "tasks": list(config["training_task_order"]),
        "shared_query_across_tasks": True,
        "base_checkpoint_hash": base_hash,
        "training_case_count": len(processed_ids),
        "training_case_count_by_task": {
            task: len(processed_ids_by_task[task])
            for task in config["training_task_order"]
        },
        "unique_case_count": len(set(processed_ids)),
        "repeat_count": len(processed_ids) - len(set(processed_ids)),
        "training_passes": 1,
        "optimizer_steps": len(processed_ids),
        "processed_nonpadding_input_tokens": total_input_tokens,
        "trained_target_tokens": total_target_tokens,
        "trainable_parameter_names": list(trainable_names),
        "query_initialization": "base_checkpoint_kimi_attnres",
        "query_before_hash": query_before,
        "query_after_hash": query_after,
        "query_file_sha256": sha256_file(query_path),
        "backbone_before_hash": backbone_before,
        "backbone_after_hash": backbone_after,
        "positive_finite_query_gradient_seen": positive_finite_gradient_seen,
        "case_selection_sha256": sha256_json(processed_ids),
        "case_selection_sha256_by_task": {
            task: sha256_json(processed_ids_by_task[task])
            for task in config["training_task_order"]
        },
        "training_stable_ids_by_task": processed_ids_by_task,
        "mean_training_loss": sum(losses) / len(losses),
        "final_training_loss": losses[-1],
        "execution_mode": config["execution_mode"],
        "fixed_partition": config.get("fixed_partition"),
    }
    if not (
        manifest["backbone_before_hash"] == manifest["backbone_after_hash"]
        and manifest["query_before_hash"] != manifest["query_after_hash"]
        and manifest["unique_case_count"] == expected_total
        and manifest["repeat_count"] == 0
    ):
        raise RuntimeError("Baseline completion manifest verification failed")
    if context.is_rank0:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    is_rank0 = context.is_rank0
    barrier(context)
    destroy_distributed(context)
    return manifest if is_rank0 else {}


def run_cli(
    *,
    baseline_type: BaselineType,
    default_config: str,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", choices=("all",))
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_baseline_config(config, expected_type=baseline_type)
    result = (
        check_only(
            baseline_type=baseline_type,
            config=config,
        )
        if args.check_only
        else train(
            baseline_type=baseline_type,
            config=config,
        )
    )
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
