from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from safetensors.torch import load_model, save_model


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _root_model(model):
    return model.module if isinstance(model, FSDP) else model


def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    metadata: dict[str, Any],
    tokenizer=None,
) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    is_fsdp = isinstance(model, FSDP)
    is_rank0 = _is_rank0()
    if is_fsdp:
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_state = model.state_dict()
        if is_rank0:
            torch.save(model_state, root / "model_state.pt")
        optimizer_state = FSDP.full_optim_state_dict(
            model,
            optimizer,
            rank0_only=True,
        )
        if is_rank0:
            torch.save(optimizer_state, root / "optimizer.pt")
    else:
        if is_rank0:
            save_model(model, str(root / "model.safetensors"))
            torch.save(optimizer.state_dict(), root / "optimizer.pt")
    _barrier()
    if is_rank0:
        model_config = _root_model(model).config
        model_config.to_json_file(root / "config.json")
        if tokenizer is not None:
            tokenizer.save_pretrained(root / "tokenizer")
        (root / "scheduler.json").write_text(
            json.dumps(scheduler.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rng_name = f"rng_rank{dist.get_rank()}.pt" if dist.is_initialized() else "rng.pt"
    torch.save(_rng_state(), root / rng_name)
    _barrier()
    if is_rank0:
        if dist.is_initialized():
            torch.save(_rng_state(), root / "rng.pt")
        manifest = dict(metadata)
        manifest["model_file"] = "model_state.pt" if is_fsdp else "model.safetensors"
        manifest["distributed_world_size"] = dist.get_world_size() if is_fsdp else 1
        (root / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    _barrier()
    return root


def load_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    expected: dict[str, Any],
) -> dict[str, Any]:
    root = Path(path)
    manifest_path = root / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Explicit Kimi checkpoint manifest is missing: {manifest_path}")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"KIMI_CHECKPOINT_MISMATCH: {key}: expected {value!r}, got {metadata.get(key)!r}")
    if isinstance(model, FSDP):
        if metadata.get("model_file") != "model_state.pt":
            raise RuntimeError("KIMI_CHECKPOINT_MISMATCH: expected FSDP model state")
        model_state = torch.load(root / "model_state.pt", map_location="cpu", weights_only=False)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            incompatible = model.load_state_dict(model_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"KIMI_CHECKPOINT_MISMATCH: model state {incompatible}")
        optimizer_state = (
            torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False)
            if _is_rank0()
            else None
        )
        optimizer.load_state_dict(
            FSDP.scatter_full_optim_state_dict(optimizer_state, model, optim=optimizer)
        )
    else:
        if metadata.get("model_file", "model.safetensors") != "model.safetensors":
            raise RuntimeError("KIMI_CHECKPOINT_MISMATCH: expected single-process model state")
        load_model(model, str(root / "model.safetensors"), strict=True)
        optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(json.loads((root / "scheduler.json").read_text(encoding="utf-8")))
    rng_name = (
        root / f"rng_rank{dist.get_rank()}.pt"
        if dist.is_initialized() and (root / f"rng_rank{dist.get_rank()}.pt").is_file()
        else root / "rng.pt"
    )
    _restore_rng(torch.load(rng_name, map_location="cpu", weights_only=False))
    _barrier()
    return metadata
