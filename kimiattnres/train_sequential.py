from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
import random
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer

from .checkpoint import load_checkpoint, save_checkpoint
from .data import Example, collate, load_training_examples, load_validation_examples
from .modeling_qwen3_kimiattnres import (
    Qwen3KimiDecoderLayer,
    convert_pretrained_qwen3,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve(root: Path, raw: str | Path) -> Path:
    value = Path(raw)
    return value if value.is_absolute() else (root / value).resolve()


def _load_config(path: str | Path) -> dict[str, Any]:
    from .data import load_yaml

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = _root() / config_path
    return load_yaml(config_path)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _parameter_groups(model, config: dict[str, Any]):
    optimizer_config = config["training"]["optimizer"]
    backbone, query, other_attnres = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "pseudo_query" in name:
            query.append(parameter)
        elif "key_norm" in name:
            other_attnres.append(parameter)
        else:
            backbone.append(parameter)
    if any("alpha" in name for name, _ in model.named_parameters()):
        raise RuntimeError("Native Kimi baseline must not define Alpha/gate parameters")
    if not backbone or not query or not other_attnres:
        raise RuntimeError("Native Kimi requires trainable backbone, pseudo-query, and RMSNorm parameters")
    groups = [
        {
            "params": backbone,
            "lr": float(optimizer_config["backbone_lr"]),
            "weight_decay": float(optimizer_config["backbone_weight_decay"]),
        },
        {
            "params": query,
            "lr": float(optimizer_config.get("query_lr", optimizer_config["attnres_lr"])),
            "weight_decay": float(optimizer_config["attnres_weight_decay"]),
        },
    ]
    groups.append(
        {
            "params": other_attnres,
            "lr": float(optimizer_config["attnres_lr"]),
            "weight_decay": float(optimizer_config["attnres_weight_decay"]),
        }
    )
    return groups


def _init_distributed() -> dict[str, Any] | None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Kimi multi-process training requires CUDA")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("Kimi distributed environment does not match process group")
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": torch.device("cuda", local_rank),
    }


def _is_rank0(context: dict[str, Any] | None) -> bool:
    return context is None or context["rank"] == 0


def _barrier(context: dict[str, Any] | None) -> None:
    if context is not None:
        dist.barrier()


def _root_model(model):
    return model.module if isinstance(model, FSDP) else model


def _canonical_parameter_name(name: str) -> str:
    return ".".join(
        component
        for component in name.split(".")
        if component != "_fsdp_wrapped_module"
    )


def _wrap_fsdp(model, context: dict[str, Any]):
    expected = tuple(
        sorted(
            _canonical_parameter_name(name)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    )
    policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3KimiDecoderLayer},
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        use_orig_params=True,
        limit_all_gathers=True,
        sync_module_states=True,
        device_id=context["device"],
    )
    actual = tuple(
        sorted(
            _canonical_parameter_name(name)
            for name, parameter in wrapped.named_parameters()
            if parameter.requires_grad
        )
    )
    if actual != expected:
        raise RuntimeError("Kimi FSDP changed the trainable parameter set")
    return wrapped


def _clip_grad_norm(model, max_norm: float) -> torch.Tensor:
    if isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def _snapshot_group(model, marker: str) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if marker in name
    }


def _group_changed(
    model,
    before: dict[str, torch.Tensor],
    context: dict[str, Any] | None,
) -> bool:
    local_changed = any(
        name not in before or not torch.equal(before[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if name in before
    )
    if context is None:
        return local_changed
    flag = torch.tensor(int(local_changed), dtype=torch.int32, device=context["device"])
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _group_l2_norm(
    model,
    marker: str,
    *,
    gradient: bool,
    context: dict[str, Any] | None,
) -> float:
    values = []
    for name, parameter in model.named_parameters():
        if marker not in name:
            continue
        value = parameter.grad if gradient else parameter
        if value is not None:
            values.append(value.detach().float().pow(2).sum())
    if values:
        total = torch.stack(values).sum()
    else:
        device = next(model.parameters()).device
        total = torch.zeros((), dtype=torch.float32, device=device)
    if context is not None:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.sqrt().cpu())


def _memory_summary(device: torch.device, context: dict[str, Any] | None):
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    local = torch.tensor(
        [
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
            free_bytes,
            total_bytes,
        ],
        dtype=torch.int64,
        device=device,
    )
    if context is None:
        rows = [local.cpu().tolist()]
    else:
        gathered = [torch.empty_like(local) for _ in range(context["world_size"])] if _is_rank0(context) else None
        dist.gather(local, gather_list=gathered, dst=0)
        if not _is_rank0(context):
            return None
        rows = [value.cpu().tolist() for value in gathered]
    return [
        {
            "rank": rank,
            "peak_allocated_bytes": int(values[0]),
            "peak_reserved_bytes": int(values[1]),
            "free_bytes_after_training": int(values[2]),
            "total_bytes": int(values[3]),
        }
        for rank, values in enumerate(rows)
    ]


class TokenCosineScheduler:
    def __init__(self, optimizer, *, maximum_tokens: int, warmup_ratio: float, min_lr_ratio: float):
        self.optimizer = optimizer
        self.maximum_tokens = int(maximum_tokens)
        self.warmup_tokens = int(self.maximum_tokens * float(warmup_ratio))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.trained_tokens = 0
        self.step(0)

    def step(self, trained_tokens: int) -> None:
        self.trained_tokens = int(trained_tokens)
        if self.trained_tokens < self.warmup_tokens:
            ratio = self.trained_tokens / max(1, self.warmup_tokens)
        else:
            progress = min(
                1.0,
                (self.trained_tokens - self.warmup_tokens)
                / max(1, self.maximum_tokens - self.warmup_tokens),
            )
            ratio = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = float(base_lr * ratio)

    def state_dict(self) -> dict[str, Any]:
        return {
            "maximum_tokens": self.maximum_tokens,
            "warmup_tokens": self.warmup_tokens,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "trained_tokens": self.trained_tokens,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, expected in (
            ("maximum_tokens", self.maximum_tokens),
            ("warmup_tokens", self.warmup_tokens),
            ("min_lr_ratio", self.min_lr_ratio),
        ):
            if state.get(key) != expected:
                raise RuntimeError(f"KIMI_SCHEDULER_MISMATCH: {key}")
        if list(state.get("base_lrs", ())) != self.base_lrs:
            raise RuntimeError("KIMI_SCHEDULER_MISMATCH: base_lrs")
        self.step(int(state["trained_tokens"]))


def _parameter_digest(model, contains: str | None = None) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if contains is not None and contains not in name:
            continue
        digest.update(name.encode())
        digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _conversion_check(original, converted, example, device: torch.device) -> dict[str, Any]:
    original.eval()
    converted.eval()
    batch = {
        "input_ids": example.input_ids.unsqueeze(0).to(device),
        "attention_mask": example.attention_mask.unsqueeze(0).to(device),
    }
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            native = original(**batch, use_cache=False).logits.float()
            kimi = converted(**batch, use_cache=False).logits.float()
    delta = (native - kimi).abs()
    result = {
        "max_abs_diff": float(delta.max().cpu()),
        "mean_abs_diff": float(delta.mean().cpu()),
    }
    if not torch.isfinite(native).all() or not torch.isfinite(kimi).all():
        raise FloatingPointError(f"KIMI_CONVERSION_NONFINITE: {result}")
    return result


def _write_metric(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


@torch.no_grad()
def _validation_loss(
    model,
    examples: list[Example] | tuple[Example, ...],
    *,
    tokenizer,
    device: torch.device,
    context: dict[str, Any] | None,
) -> float:
    model.eval()
    rank = context["rank"] if context is not None else 0
    world_size = context["world_size"] if context is not None else 1
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    token_count = torch.zeros((), dtype=torch.float32, device=device)
    for index, example in enumerate(examples):
        if index % world_size != rank:
            continue
        batch = collate([example], tokenizer.pad_token_id)
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(**inputs, use_cache=False).logits
        valid = labels != -100
        if not valid.any():
            continue
        loss_sum += F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        token_count += valid.sum()
    if context is not None:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    if token_count.item() <= 0:
        raise RuntimeError("KIMI_VALIDATION_HAS_NO_SUPERVISED_TOKENS")
    value = loss_sum / token_count
    if not torch.isfinite(value):
        raise FloatingPointError("KIMI_VALIDATION_LOSS_NAN_OR_INF")
    model.train()
    return float(value.cpu())


def _summarize_metrics(path: Path, task: str) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if path.is_file() else []
    if not rows:
        raise RuntimeError(f"KIMI_METRICS_EMPTY: {path}")
    train = [float(row["train_loss"]) for row in rows]
    validation = [row for row in rows if row.get("val_loss") is not None]
    values = [float(row["tokens_per_second"]) for row in rows if float(row["tokens_per_second"]) > 0]
    total_tokens = sum(int(row["tokens_step"]) for row in rows)
    total_seconds = sum(float(row.get("train_wall_seconds", 0.0)) for row in rows)
    peak = max(float(row.get("peak_memory_gb", 0.0)) for row in rows)
    best_val = min((float(row["val_loss"]) for row in validation), default=None)
    final_val = float(validation[-1]["val_loss"]) if validation else None
    return {
        "task": task,
        "steps": len(rows),
        "final_train_loss": train[-1],
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "best_val_ppl": float(np.exp(best_val)) if best_val is not None else None,
        "final_val_ppl": float(np.exp(final_val)) if final_val is not None else None,
        "average_tokens_per_second": total_tokens / total_seconds if total_seconds > 0 else (sum(values) / len(values) if values else 0.0),
        "peak_memory_gb": peak,
    }


def _fit_example_to_budget(example: Example, remaining_tokens: int) -> Example | None:
    if example.input_ids.numel() <= remaining_tokens:
        return example
    supervised_tokens = int((example.labels != -100).sum().item())
    if remaining_tokens <= 0 or supervised_tokens > remaining_tokens:
        return None
    return Example(
        input_ids=example.input_ids[-remaining_tokens:],
        labels=example.labels[-remaining_tokens:],
        attention_mask=example.attention_mask[-remaining_tokens:],
        stable_id=example.stable_id,
        source_key=example.source_key,
    )


def _sampling_summary(
    examples_by_task: dict[str, list[Example]],
    source_mix: dict[str, dict[str, float]],
    sampling: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for task, examples in examples_by_task.items():
        counts = Counter(example.source_key for example in examples)
        summary[task] = {
            "mode": sampling["mode"],
            "seed": int(sampling["seed"]),
            "pool_size_per_source": int(sampling["pool_size_per_source"]),
            "target_ratios": dict(source_mix[task]),
            "cycle_examples": len(examples),
            "cycle_source_counts": dict(counts),
            "first_sources": [
                example.source_key for example in examples[: len(source_mix[task])]
            ],
        }
    return summary


def _validate_config(config: dict[str, Any]) -> tuple[str, ...]:
    tasks = tuple(config["training"]["task_order"])
    if tasks != ("math", "multihop"):
        raise ValueError("Kimi baseline currently requires task_order=[math, multihop]")
    if config["use_cache"] is not False or config["dtype"] != "bfloat16":
        raise ValueError("Kimi baseline requires BF16 and use_cache=false")
    if "alpha_lr" in config["training"]["optimizer"]:
        raise ValueError("Native Kimi baseline has no Alpha learning rate")
    strategy = str(config.get("distributed", {}).get("strategy", ""))
    if strategy not in {"single_gpu", "fsdp"}:
        raise ValueError("distributed.strategy must be single_gpu or fsdp")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and strategy != "fsdp":
        raise ValueError("Kimi multi-process training requires distributed.strategy=fsdp")
    if int(config["training"]["micro_batch_size"]) != 1 or int(config["training"]["gradient_accumulation_steps"]) != 1:
        raise ValueError("Kimi baseline currently uses one example per optimizer step")
    source_mix = config["training"].get("source_mix")
    sampling = config["training"].get("sampler")
    if not isinstance(source_mix, dict) or set(source_mix) != set(tasks):
        raise ValueError("training.source_mix must cover every task exactly")
    if not isinstance(sampling, dict) or sampling.get("mode") != "fixed_ratio_round_robin":
        raise ValueError("training.sampler.mode must be fixed_ratio_round_robin")
    if int(sampling.get("pool_size_per_source", 0)) <= 0:
        raise ValueError("training.sampler.pool_size_per_source must be positive")
    if int(sampling.get("seed", -1)) != int(config["seed"]):
        raise ValueError("training.sampler.seed must equal top-level seed")
    metrics = config["training"].get("metrics", {})
    if int(metrics.get("validation_interval_steps", 0)) <= 0:
        raise ValueError("training.metrics.validation_interval_steps must be positive")
    if int(metrics.get("validation_examples_per_task", 0)) <= 0:
        raise ValueError("training.metrics.validation_examples_per_task must be positive")
    for task in tasks:
        ratios = source_mix[task]
        if not isinstance(ratios, dict) or not ratios:
            raise ValueError(f"training.source_mix[{task}] must contain at least one source")
        values = [float(value) for value in ratios.values()]
        if abs(sum(values) - 1.0) > 1.0e-8 or any(value <= 0 for value in values):
            raise ValueError(f"training.source_mix[{task}] ratios must sum to 1")
        expected = 1.0 / len(values)
        if any(abs(value - expected) > 1.0e-8 for value in values):
            raise ValueError(f"training.source_mix[{task}] must use equal ratios")
    if config["mode"] == "block":
        sizes = config.get("block_sizes")
        if sizes != [4, 4, 4, 4, 4, 4, 4]:
            raise ValueError("Block baseline requires explicit Qwen3-1.7B block_sizes=[4]*7")
    elif config["mode"] == "full":
        if config.get("block_sizes") is not None:
            raise ValueError("Full baseline must not define block_sizes")
    else:
        raise ValueError("mode must be block or full")
    return tasks


def run(
    config: dict[str, Any],
    *,
    resume: str | None = None,
    max_steps: int | None = None,
    output_override: str | None = None,
    steps_per_task: int | None = None,
    stop_after_task: str | None = None,
) -> dict[str, Any]:
    root = _root()
    tasks = _validate_config(config)
    _seed(int(config["seed"]))
    distributed_context = _init_distributed()
    device = (
        distributed_context["device"]
        if distributed_context is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    checkpoint = _resolve(root, config["base_checkpoint"])
    original, model, conversion = convert_pretrained_qwen3(
        checkpoint,
        mode=config["mode"],
        block_sizes=config.get("block_sizes"),
        dtype=dtype,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    examples_by_task = load_training_examples(
        repo_root=root,
        data_manifest=config["data_manifest"],
        data_config=config["data_config"],
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=int(config["max_sequence_length"]),
        source_mix=config["training"]["source_mix"],
        sampling=config["training"]["sampler"],
    )
    examples_by_task = {task: list(examples) for task, examples in examples_by_task.items()}
    validation_examples_by_task = load_validation_examples(
        repo_root=root,
        data_manifest=config["data_manifest"],
        data_config=config["data_config"],
        tokenizer=tokenizer,
        tasks=tasks,
        max_length=int(config["max_sequence_length"]),
        source_mix=config["training"]["source_mix"],
        examples_per_task=int(config["training"]["metrics"]["validation_examples_per_task"]),
    )
    sampling_summary = _sampling_summary(
        examples_by_task,
        config["training"]["source_mix"],
        config["training"]["sampler"],
    )
    if _is_rank0(distributed_context):
        print(json.dumps({"sampling": sampling_summary}, sort_keys=True), flush=True)
    if steps_per_task is not None and steps_per_task <= 0:
        raise ValueError("steps_per_task must be positive")
    if stop_after_task is not None and stop_after_task not in tasks:
        raise ValueError(f"stop_after_task must be one of {tasks}")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    original.to(device)
    model.to(device)
    if _is_rank0(distributed_context):
        original.to(device)
        model.to(device)
        conversion_check = _conversion_check(original, model, examples_by_task[tasks[0]][0], device)
        model.cpu()
    else:
        conversion_check = None
    del original
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if distributed_context is not None:
        payload = [conversion_check]
        dist.broadcast_object_list(payload, src=0)
        conversion_check = payload[0]
        _barrier(distributed_context)
        model = _wrap_fsdp(model, distributed_context)
    else:
        model.to(device)
    model.train()
    root_model = _root_model(model)
    root_model.config.use_cache = False
    root_model.gradient_checkpointing = bool(config.get("gradient_checkpointing", True))
    root_model.model.gradient_checkpointing = root_model.gradient_checkpointing
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    groups = _parameter_groups(model, config)
    optimizer_config = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(
        groups,
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
    )
    budgets = {task: int(config["training"]["token_budget"][task]) for task in tasks}
    scheduler_config = config["training"]["scheduler"]
    scheduler = TokenCosineScheduler(
        optimizer,
        maximum_tokens=sum(budgets.values()),
        warmup_ratio=float(scheduler_config["warmup_ratio"]),
        min_lr_ratio=float(scheduler_config["min_lr_ratio"]),
    )
    initial_query = _snapshot_group(model, "pseudo_query")
    initial_key_norm = _snapshot_group(model, "key_norm")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = _resolve(root, output_override or config["output_dir"])
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"Refusing to overwrite Kimi output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    metric_paths = {task: output / f"metrics_{task}.jsonl" for task in tasks}
    summary_path = output / "metrics_summary.json"
    state = {
        "current_task_index": 0,
        "global_step": 0,
        "task_steps": {task: 0 for task in tasks},
        "task_tokens": {task: 0 for task in tasks},
        "global_tokens": 0,
        "positions": {task: 0 for task in tasks},
        "epochs": {task: 0 for task in tasks},
    }
    if resume:
        metadata = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected={
                "mode": config["mode"],
                "block_sizes": config.get("block_sizes"),
                "task_order": list(tasks),
                "base_checkpoint": str(checkpoint),
                "source_mix": config["training"]["source_mix"],
                "sampler": config["training"]["sampler"],
            },
        )
        state.update(metadata["progress"])
    state.setdefault("stage_training_wall_seconds", {task: 0.0 for task in tasks})
    state.setdefault("global_training_wall_seconds", 0.0)
    state.setdefault("stage_peak_memory_bytes", {task: 0 for task in tasks})
    state.setdefault("current_stage_peak_memory_bytes", 0)
    window: dict[str, deque[float]] = {task: deque(maxlen=int(config["training"]["metrics_window"])) for task in tasks}
    for task_index in range(int(state["current_task_index"]), len(tasks)):
        task = tasks[task_index]
        examples = examples_by_task[task]
        task_steps_at_start = int(state["task_steps"][task])
        while int(state["task_tokens"][task]) < budgets[task]:
            if max_steps is not None and int(state["global_step"]) >= max_steps:
                break
            if steps_per_task is not None and int(state["task_steps"][task]) - task_steps_at_start >= steps_per_task:
                break
            remaining_tokens = budgets[task] - int(state["task_tokens"][task])
            example = None
            for _ in range(len(examples)):
                position = int(state["positions"][task]) % len(examples)
                state["positions"][task] = position + 1
                example = _fit_example_to_budget(examples[position], remaining_tokens)
                if example is not None:
                    break
            if example is None:
                break
            batch = collate([example], tokenizer.pad_token_id)
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            labels = batch["labels"].to(device)
            tokens_step = int(inputs["attention_mask"].sum().item())
            if tokens_step > remaining_tokens:
                raise RuntimeError("KIMI_TOKEN_BUDGET_EXCEEDED_AFTER_TRUNCATION")
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**inputs, use_cache=False).logits
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss = loss_sum / max(1, int((labels != -100).sum().item()))
            if not torch.isfinite(loss):
                raise FloatingPointError("KIMI_LOSS_NAN_OR_INF")
            reported_loss_sum = loss_sum.detach().float()
            reported_tokens = torch.tensor(
                float((labels != -100).sum().item()), device=device, dtype=torch.float32
            )
            if distributed_context is not None:
                dist.all_reduce(reported_loss_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(reported_tokens, op=dist.ReduceOp.SUM)
            reported_loss = reported_loss_sum / reported_tokens.clamp_min(1.0)
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError("KIMI_GRADIENT_NAN_OR_INF")
            query_grad_norm = _group_l2_norm(
                model, "pseudo_query", gradient=True, context=distributed_context
            )
            clipped_norm = _clip_grad_norm(
                model,
                float(config["training"]["max_grad_norm"]),
            )
            if not torch.isfinite(clipped_norm):
                raise FloatingPointError("KIMI_CLIPPED_GRADIENT_NAN_OR_INF")
            optimizer.step()
            step_wall_seconds = max(0.0, time.perf_counter() - step_started)
            query_norm = _group_l2_norm(
                model, "pseudo_query", gradient=False, context=distributed_context
            )
            state["task_tokens"][task] += tokens_step
            state["global_tokens"] += tokens_step
            state["task_steps"][task] += 1
            state["global_step"] += 1
            scheduler.step(int(state["global_tokens"]))
            state["stage_training_wall_seconds"][task] += step_wall_seconds
            state["global_training_wall_seconds"] += step_wall_seconds
            if device.type == "cuda":
                state["current_stage_peak_memory_bytes"] = max(
                    int(state["current_stage_peak_memory_bytes"]),
                    int(torch.cuda.max_memory_allocated(device)),
                )
            window[task].append(float(reported_loss.cpu()))
            metrics_config = config["training"]["metrics"]
            val_loss = None
            if int(state["global_step"]) % int(metrics_config["validation_interval_steps"]) == 0:
                val_loss = _validation_loss(
                    model,
                    validation_examples_by_task[task],
                    tokenizer=tokenizer,
                    device=device,
                    context=distributed_context,
                )
            peak_memory_gb = (
                float(state["current_stage_peak_memory_bytes"]) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            )
            tokens_per_second = tokens_step / max(step_wall_seconds, 1.0e-9)
            metric = {
                "method": f"kimi_{config['mode']}",
                "task": task,
                "stage": task,
                "dataset": example.source_key,
                "mode": config["mode"],
                "global_step": int(state["global_step"]),
                "task_step": int(state["task_steps"][task]),
                "tokens_seen": int(state["global_tokens"]),
                "train_loss": float(reported_loss.cpu()),
                "raw_loss": float(reported_loss.cpu()),
                "val_loss": val_loss,
                "val_ppl": float(np.exp(val_loss)) if val_loss is not None else None,
                "smoothed_loss": float(sum(window[task]) / len(window[task])),
                "lr_backbone": float(optimizer.param_groups[0]["lr"]),
                "lr_query": float(optimizer.param_groups[1]["lr"]),
                "lr_attnres": float(optimizer.param_groups[2]["lr"]),
                "query_norm": query_norm,
                "query_grad_norm": query_grad_norm,
                "tokens_step": tokens_step,
                "task_tokens": int(state["task_tokens"][task]),
                "global_tokens": int(state["global_tokens"]),
                "tokens_per_second": tokens_per_second,
                "train_wall_seconds": step_wall_seconds,
                "peak_memory_gb": peak_memory_gb,
            }
            if _is_rank0(distributed_context):
                _write_metric(metrics_path, metric)
                _write_metric(metric_paths[task], metric)
        stage_limit_reached = (
            steps_per_task is not None
            and int(state["task_steps"][task]) - task_steps_at_start >= steps_per_task
        )
        stage_stop_requested = stop_after_task == task
        if int(state["task_tokens"][task]) >= budgets[task] or stage_limit_reached or stage_stop_requested:
            state["current_task_index"] = task_index + 1
            if task_index + 1 < len(tasks):
                next_task = tasks[task_index + 1]
                state["current_task"] = next_task
                save_name = f"after_{task}"
            else:
                state["current_task"] = None
                save_name = "final"
            state["stage_peak_memory_bytes"][task] = int(state["current_stage_peak_memory_bytes"])
            state["current_stage_peak_memory_bytes"] = 0
            metadata = {
                "schema": "kimiattnres_native_v1",
                "mode": config["mode"],
                "block_sizes": config.get("block_sizes"),
                "task_order": list(tasks),
                "base_checkpoint": str(checkpoint),
                "source_mix": config["training"]["source_mix"],
                "sampler": config["training"]["sampler"],
                "progress": state,
                "conversion": conversion,
                "conversion_check": conversion_check,
                "parameter_hashes": {
                    "backbone": _parameter_digest(model),
                    "query": _parameter_digest(model, "pseudo_query"),
                    "key_norm": _parameter_digest(model, "key_norm"),
                },
            }
            for checkpoint_name in (save_name, "latest"):
                save_checkpoint(
                    output / checkpoint_name,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=metadata,
                    tokenizer=tokenizer,
                )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
        if stage_stop_requested:
            break
        if max_steps is not None and int(state["global_step"]) >= max_steps:
            break
    if device.type == "cuda":
        state["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        state["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        state["memory_by_rank"] = _memory_summary(device, distributed_context)
    state["query_updated"] = _group_changed(model, initial_query, distributed_context)
    state["key_norm_updated"] = _group_changed(model, initial_key_norm, distributed_context)
    if state["current_task_index"] < len(tasks):
        task = tasks[int(state["current_task_index"])]
        state["stage_peak_memory_bytes"][task] = max(
            int(state["stage_peak_memory_bytes"].get(task, 0)),
            int(state["current_stage_peak_memory_bytes"]),
        )
    stage_summaries = (
        {task: _summarize_metrics(metric_paths[task], task) for task in tasks}
        if _is_rank0(distributed_context)
        else {}
    )
    overall_peak = max(
        [int(value) for value in state["stage_peak_memory_bytes"].values()]
        + [int(state.get("peak_allocated_bytes", 0))]
    )
    if _is_rank0(distributed_context):
        summary_metrics = {
            "method": f"kimi_{config['mode']}",
            "math": stage_summaries.get("math"),
            "multihop": stage_summaries.get("multihop"),
            "overall": {
                "total_tokens": int(state["global_tokens"]),
                "total_walltime": float(state["global_training_wall_seconds"]),
                "max_peak_memory_gb": overall_peak / (1024 ** 3),
            },
        }
        summary_path.write_text(json.dumps(summary_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS",
        "mode": config["mode"],
        "device": str(device),
        "conversion_check": conversion_check,
        "conversion": conversion,
        "progress": state,
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "backbone_parameter_count": sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and not any(token in name for token in ("pseudo_query", "key_norm"))),
        "attnres_parameter_count": sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and any(token in name for token in ("pseudo_query", "key_norm"))),
        "query_updated": state["query_updated"],
        "key_norm_updated": state["key_norm_updated"],
        "memory_by_rank": state.get("memory_by_rank"),
        "sampling": sampling_summary,
        "metrics_path": str(metrics_path),
        "metrics_summary": stage_summaries,
        "stage_peak_memory_bytes": state["stage_peak_memory_bytes"],
        "metrics_summary": stage_summaries,
        "metrics_path": str(metrics_path),
        "stage_peak_memory_bytes": state["stage_peak_memory_bytes"],
    }
    if _is_rank0(distributed_context):
        (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _barrier(distributed_context)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--steps-per-task", type=int)
    parser.add_argument("--stop-after-task")
    args = parser.parse_args()
    summary = run(_load_config(args.config), resume=args.resume, max_steps=args.max_steps, output_override=args.output_dir, steps_per_task=args.steps_per_task, stop_after_task=args.stop_after_task)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(json.dumps(summary, indent=2, sort_keys=True))
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
