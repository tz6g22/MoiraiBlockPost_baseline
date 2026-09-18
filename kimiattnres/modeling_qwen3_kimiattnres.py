from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


Mode = Literal["block", "full"]


class Qwen3KimiAttnResConfig(Qwen3Config):
    model_type = "qwen3_kimi_attnres"

    def __init__(
        self,
        *,
        kimi_mode: Mode = "block",
        block_sizes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if kimi_mode not in {"block", "full"}:
            raise ValueError("kimi_mode must be 'block' or 'full'")
        self.kimi_mode = kimi_mode
        self.block_sizes = list(block_sizes) if block_sizes is not None else None


def _factory_compat(factory, **kwargs):
    signature = inspect.signature(factory)
    return factory(**{key: value for key, value in kwargs.items() if key in signature.parameters})


def depth_attention(
    sources: tuple[torch.Tensor, ...],
    pseudo_query: torch.Tensor,
    key_norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head Kimi depth attention; softmax is over source/depth."""
    if not sources:
        raise ValueError("Kimi depth attention requires at least one source")
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError("Kimi depth sources must have identical shapes")
    values = torch.stack(sources, dim=0)
    keys = key_norm(values)
    logits = torch.einsum("d,nbtd->nbt", pseudo_query.float(), keys.float())
    weights = torch.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", weights.to(values.dtype), values), weights


class Qwen3KimiDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3KimiAttnResConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type = config.layer_types[layer_idx]

        # Per-site Kimi parameters are one shared parameter set for the whole
        # Math -> Multi-hop run. There are no task banks in this baseline.
        self.attn_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.attn_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.mlp_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        completed_sources: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Cache | None,
        is_block_boundary: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor, ...], bool]:
        attn_sources = completed_sources + ((partial_block,) if partial_block is not None else ())
        routed_attn, _ = depth_attention(
            attn_sources, self.attn_pseudo_query, self.attn_key_norm
        )
        attn_output, _ = self.self_attn(
            hidden_states=self.input_layernorm(routed_attn),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_after_attn = routed_attn + attn_output
        partial_after_attn = (
            attn_output if partial_block is None else partial_block + attn_output
        )

        if self._mode == "full":
            mlp_sources = completed_sources + (hidden_after_attn,)
        else:
            mlp_sources = completed_sources + (partial_after_attn,)
        routed_mlp, _ = depth_attention(
            mlp_sources, self.mlp_pseudo_query, self.mlp_key_norm
        )
        mlp_output = self.mlp(self.post_attention_layernorm(routed_mlp))
        hidden_after_mlp = routed_mlp + mlp_output

        if self._mode == "full":
            # Full AttnRes keeps each post-sublayer cumulative hidden state.
            return (
                hidden_after_mlp,
                None,
                completed_sources + (hidden_after_attn, hidden_after_mlp),
                False,
            )

        if is_block_boundary:
            if partial_after_attn is None:
                raise RuntimeError("Block boundary has no partial residual")
            block_summary = partial_after_attn + mlp_output
            return (
                hidden_after_mlp,
                None,
                completed_sources + (block_summary,),
                True,
            )
        return hidden_after_mlp, partial_after_attn + mlp_output, completed_sources, False

    @property
    def _mode(self) -> Mode:
        return self._kimi_mode


class Qwen3KimiAttnResModel(Qwen3PreTrainedModel):
    config_class = Qwen3KimiAttnResConfig

    def __init__(self, config: Qwen3KimiAttnResConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3KimiDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        for layer in self.layers:
            layer._kimi_mode = config.kimi_mode
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_key_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types
        self.post_init()
        for layer in self.layers:
            layer.attn_pseudo_query.data.zero_()
            layer.mlp_pseudo_query.data.zero_()
        self.final_pseudo_query.data.zero_()

    def _validate_block_sizes(self) -> tuple[int, ...]:
        sizes = self.config.block_sizes
        if self.config.kimi_mode == "full":
            if sizes is not None:
                raise ValueError("Full Kimi AttnRes must not define block_sizes")
            return ()
        if not sizes or any(int(size) <= 0 for size in sizes):
            raise ValueError("Block Kimi AttnRes requires positive block_sizes")
        if sum(int(size) for size in sizes) != int(self.config.num_hidden_layers):
            raise ValueError("block_sizes must cover num_hidden_layers exactly")
        return tuple(int(size) for size in sizes)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        return_kimi_debug: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache:
            raise ValueError("Kimi baseline requires use_cache=False")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if not isinstance(causal_masks := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_masks = {
                "full_attention": _factory_compat(create_causal_mask, **mask_kwargs)
            }
            if self.has_sliding_layers:
                causal_masks["sliding_attention"] = _factory_compat(
                    create_sliding_window_causal_mask, **mask_kwargs
                )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        block_sizes = self._validate_block_sizes()
        boundary_layers: set[int] = set()
        if block_sizes:
            cursor = 0
            for size in block_sizes:
                cursor += size
                boundary_layers.add(cursor - 1)

        hidden = inputs_embeds
        completed_sources: tuple[torch.Tensor, ...] = (inputs_embeds,)
        partial_block: torch.Tensor | None = None
        commits: list[int] = []
        history_lengths: list[int] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_args = (
                hidden,
                completed_sources,
                partial_block,
                causal_masks[layer.attention_type],
                position_ids,
                cache_position,
                position_embeddings,
                past_key_values,
                layer_idx in boundary_layers,
            )
            if self.gradient_checkpointing and self.training:
                hidden, partial_block, completed_sources, committed = checkpoint(
                    layer,
                    *layer_args,
                    use_reentrant=False,
                )
            else:
                hidden, partial_block, completed_sources, committed = layer(*layer_args)
            if committed:
                commits.append(layer_idx)
            history_lengths.append(len(completed_sources))

        routed_final, _ = depth_attention(
            completed_sources,
            self.final_pseudo_query,
            self.final_key_norm,
        )
        output = BaseModelOutputWithPast(
            last_hidden_state=self.norm(routed_final),
            past_key_values=None,
        )
        if return_kimi_debug:
            output.kimi_debug = {
                "mode": self.config.kimi_mode,
                "boundary_layer_indices": sorted(boundary_layers),
                "completed_commit_layer_indices": commits,
                "history_lengths": history_lengths,
                "final_history_length": len(completed_sources),
            }
        return output


class Qwen3KimiAttnResForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = Qwen3KimiAttnResConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Qwen3KimiAttnResConfig) -> None:
        super().__init__(config)
        self.model = Qwen3KimiAttnResModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        for layer in self.model.layers:
            layer.attn_pseudo_query.data.zero_()
            layer.mlp_pseudo_query.data.zero_()
        self.model.final_pseudo_query.data.zero_()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        return_kimi_debug: bool = False,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            return_kimi_debug=return_kimi_debug,
            **kwargs,
        )
        hidden = outputs.last_hidden_state
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:, :]
        logits = self.lm_head(hidden)
        output = CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
        if return_kimi_debug and hasattr(outputs, "kimi_debug"):
            output.kimi_debug = outputs.kimi_debug
        if labels is not None:
            if logits.shape[:2] != labels.shape:
                raise ValueError("labels must align with logits")
            output.loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return output


def _set_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    setattr(parent, leaf, nn.Parameter(value))


def _materialize_extra_parameters(
    model: Qwen3KimiAttnResForCausalLM,
    names: set[str],
    dtype: torch.dtype,
) -> None:
    for name in names:
        parameter = dict(model.named_parameters())[name]
        if parameter.device.type != "meta":
            continue
        fill = 1.0 if "key_norm" in name else 0.0
        _set_parameter(
            model,
            name,
            torch.full(tuple(parameter.shape), fill, dtype=dtype, device="cpu"),
        )
    for name, buffer in tuple(model.named_buffers()):
        if buffer.device.type != "meta":
            continue
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if name.endswith("rotary_emb.inv_freq"):
            value = Qwen3RotaryEmbedding(config=model.config).inv_freq
        else:
            value = torch.zeros(tuple(buffer.shape), dtype=buffer.dtype)
        parent.register_buffer(leaf, value, persistent=False)


def config_from_pretrained(
    config: Qwen3Config,
    *,
    mode: Mode,
    block_sizes: list[int] | None,
) -> Qwen3KimiAttnResConfig:
    payload = config.to_dict()
    for key in ("architectures", "model_type", "transformers_version", "_name_or_path"):
        payload.pop(key, None)
    payload.update({"kimi_mode": mode, "block_sizes": block_sizes, "use_cache": False})
    return Qwen3KimiAttnResConfig(**payload)


def convert_pretrained_qwen3(
    checkpoint: str | Path,
    *,
    mode: Mode,
    block_sizes: list[int] | None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Qwen3ForCausalLM, Qwen3KimiAttnResForCausalLM, dict[str, list[str]]]:
    original = Qwen3ForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    converted_config = config_from_pretrained(
        original.config, mode=mode, block_sizes=block_sizes
    )
    with torch.device("meta"):
        converted = Qwen3KimiAttnResForCausalLM(converted_config)
    original_state = original.state_dict()
    native_keys = set(original_state)
    incompatible = converted.load_state_dict(original_state, strict=False, assign=True)
    allowed_missing = {
        name
        for name in incompatible.missing_keys
        if any(token in name for token in ("pseudo_query", "key_norm"))
    }
    unexpected_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Qwen3 to Kimi conversion mismatch: "
            f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
        )
    _materialize_extra_parameters(converted, allowed_missing, dtype)
    converted.config.use_cache = False
    return original, converted, {
        "missing_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": sorted(incompatible.unexpected_keys),
        "allowed_new_parameters": sorted(allowed_missing),
    }
