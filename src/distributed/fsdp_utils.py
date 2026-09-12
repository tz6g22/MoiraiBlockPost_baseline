from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


ParameterPredicate = Callable[[str, nn.Parameter], bool]


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def init_distributed(*, require_cuda: bool = True) -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank < 0 or local_rank < 0 or world_size <= 0 or rank >= world_size:
        raise RuntimeError(
            f"Invalid distributed environment: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This stage requires CUDA")
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds the visible CUDA device count "
                f"({torch.cuda.device_count()})"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        if device.type != "cuda":
            raise RuntimeError("Multi-process execution requires NCCL and CUDA")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("Process-group identity differs from torchrun environment")
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def destroy_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_object(value, context: DistributedContext, *, src: int = 0):
    if not context.distributed:
        return value
    payload = [value if context.rank == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def all_reduce_sum(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if context.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def canonical_parameter_name(name: str) -> str:
    return ".".join(
        component
        for component in name.split(".")
        if component != "_fsdp_wrapped_module"
    )


def named_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    return tuple(
        (canonical_parameter_name(name), parameter)
        for name, parameter in model.named_parameters()
    )


def trainable_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        sorted(name for name, parameter in named_parameters(model) if parameter.requires_grad)
    )


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for _, parameter in named_parameters(model):
        if parameter.requires_grad:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    return trainable, frozen


def wrap_qwen3_fsdp(
    model: nn.Module,
    context: DistributedContext,
    *,
    decoder_layer_classes: Iterable[type[nn.Module]],
    sync_module_states: bool = True,
    device_id: torch.device | None = None,
) -> nn.Module:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    expected_trainable = trainable_parameter_names(model)
    if not context.distributed:
        wrapped = model.to(context.device)
    else:
        layer_classes = {Qwen3DecoderLayer, *decoder_layer_classes}
        if not any(isinstance(module, tuple(layer_classes)) for module in model.modules()):
            raise RuntimeError("No configured Qwen3 decoder-layer unit exists in the model")
        policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=layer_classes,
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
            sync_module_states=sync_module_states,
            device_id=device_id,
        )
    actual_trainable = trainable_parameter_names(wrapped)
    if actual_trainable != expected_trainable:
        raise RuntimeError(
            "FSDP changed the trainable parameter set: "
            f"before={expected_trainable}, after={actual_trainable}"
        )
    return wrapped


def root_module(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, FSDP) else model


def clip_grad_norm(
    model: nn.Module,
    parameters: Iterable[nn.Parameter],
    max_norm: float,
) -> torch.Tensor:
    if isinstance(model, FSDP):
        return model.clip_grad_norm_(max_norm)
    return torch.nn.utils.clip_grad_norm_(list(parameters), max_norm=max_norm)


def _fsdp_parameter_owners(
    model: nn.Module,
) -> tuple[tuple[FSDP, tuple[tuple[str, nn.Parameter], ...]], ...]:
    all_named = named_parameters(model)
    units = tuple(module for module in model.modules() if isinstance(module, FSDP))
    owners: list[tuple[FSDP, tuple[tuple[str, nn.Parameter], ...]]] = []
    for unit in units:
        descendant_ids: set[int] = set()
        for descendant in unit.module.modules():
            if isinstance(descendant, FSDP):
                descendant_ids.update(id(parameter) for parameter in descendant.parameters())
        unit_ids = {
            id(parameter) for parameter in unit.module.parameters()
        } - descendant_ids
        owners.append(
            (
                unit,
                tuple(
                    (name, parameter)
                    for name, parameter in all_named
                    if id(parameter) in unit_ids
                ),
            )
        )
    owned_ids = {id(parameter) for _, values in owners for _, parameter in values}
    expected_ids = {id(parameter) for _, parameter in all_named}
    if owned_ids != expected_ids:
        raise RuntimeError("Unable to assign every parameter to one FSDP unit")
    return tuple(owners)


def selected_parameter_state(
    model: nn.Module,
    context: DistributedContext,
    predicate: ParameterPredicate,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(model, FSDP):
        return {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in named_parameters(model)
            if predicate(name, parameter)
        }
    state: dict[str, torch.Tensor] | None = {} if context.is_rank0 else None
    for unit, owned in _fsdp_parameter_owners(model):
        selected = tuple((name, parameter) for name, parameter in owned if predicate(name, parameter))
        if not selected:
            continue
        with FSDP.summon_full_params(
            unit,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        ):
            if context.is_rank0:
                assert state is not None
                for name, parameter in selected:
                    state[name] = parameter.detach().cpu().contiguous().clone()
    return state


def load_selected_parameter_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    predicate: ParameterPredicate,
) -> None:
    expected = {name for name, parameter in named_parameters(model) if predicate(name, parameter)}
    if set(state) != expected:
        raise ValueError(
            f"Parameter state keys differ: missing={sorted(expected - set(state))}, "
            f"unexpected={sorted(set(state) - expected)}"
        )
    if not isinstance(model, FSDP):
        current = dict(named_parameters(model))
        with torch.no_grad():
            for name, value in state.items():
                current[name].copy_(value.to(current[name].device, current[name].dtype))
        return
    for unit, owned in _fsdp_parameter_owners(model):
        selected = tuple((name, parameter) for name, parameter in owned if predicate(name, parameter))
        if not selected:
            continue
        with FSDP.summon_full_params(
            unit,
            recurse=False,
            writeback=True,
            rank0_only=False,
            offload_to_cpu=False,
        ):
            with torch.no_grad():
                for name, parameter in selected:
                    parameter.copy_(state[name].to(parameter.device, parameter.dtype))


def selected_parameter_sha256(
    model: nn.Module,
    context: DistributedContext,
    predicate: ParameterPredicate,
) -> str:
    value: str | None = None
    if not isinstance(model, FSDP):
        digest = hashlib.sha256()
        for name, parameter in sorted(named_parameters(model)):
            if not predicate(name, parameter):
                continue
            tensor = parameter.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    owners = _fsdp_parameter_owners(model)
    owner_by_parameter = {
        id(parameter): unit
        for unit, values in owners
        for _, parameter in values
    }
    selected = sorted(
        (name, parameter, owner_by_parameter[id(parameter)])
        for name, parameter in named_parameters(model)
        if predicate(name, parameter)
    )
    digest = hashlib.sha256() if context.is_rank0 else None
    index = 0
    while index < len(selected):
        unit = selected[index][2]
        stop = index + 1
        while stop < len(selected) and selected[stop][2] is unit:
            stop += 1
        with FSDP.summon_full_params(
            unit,
            recurse=False,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        ):
            if context.is_rank0:
                assert digest is not None
                for name, parameter, _ in selected[index:stop]:
                    tensor = parameter.detach().cpu().contiguous()
                    digest.update(name.encode("utf-8"))
                    digest.update(str(tensor.dtype).encode("ascii"))
                    digest.update(str(tuple(tensor.shape)).encode("ascii"))
                    digest.update(tensor.view(torch.uint8).numpy().tobytes())
        index = stop
    if context.is_rank0:
        assert digest is not None
        value = digest.hexdigest()
    return broadcast_object(value, context)


def full_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context: DistributedContext,
) -> dict | None:
    if isinstance(model, FSDP):
        state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
        return state if context.is_rank0 else None
    return optimizer.state_dict()


def load_full_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: dict | None,
) -> None:
    if isinstance(model, FSDP):
        sharded = FSDP.scatter_full_optim_state_dict(
            state,
            model,
            optim=optimizer,
        )
        optimizer.load_state_dict(sharded)
    else:
        if state is None:
            raise ValueError("Single-process optimizer state is missing")
        optimizer.load_state_dict(state)


def rank0_mkdir(path: str | Path, context: DistributedContext) -> None:
    if context.is_rank0:
        Path(path).mkdir(parents=True, exist_ok=True)
    barrier(context)


def peak_memory_bytes(context: DistributedContext) -> list[int] | None:
    value = torch.tensor(
        torch.cuda.max_memory_allocated(context.device),
        dtype=torch.long,
        device=context.device,
    )
    if not context.distributed:
        return [int(value.item())]
    gathered = [torch.zeros_like(value) for _ in range(context.world_size)] if context.is_rank0 else None
    dist.gather(value, gather_list=gathered, dst=0)
    if not context.is_rank0:
        return None
    assert gathered is not None
    return [int(item.item()) for item in gathered]
