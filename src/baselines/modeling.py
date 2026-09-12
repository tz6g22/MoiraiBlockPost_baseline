from __future__ import annotations

import inspect
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


BaselineExecution = Literal["full", "fixed"]


class BaselineQwen3Config(Qwen3Config):
    model_type = "qwen3_attnres"

    def __init__(
        self,
        *,
        baseline_execution: BaselineExecution = "full",
        baseline_partition: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if baseline_execution not in {"full", "fixed"}:
            raise ValueError("baseline_execution must be 'full' or 'fixed'")
        self.baseline_execution = baseline_execution
        self.baseline_partition = baseline_partition


def _mask_factory_compat(factory, **kwargs):
    signature = inspect.signature(factory)
    filtered = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return factory(**filtered)


def attnres_aggregate(
    sources: tuple[torch.Tensor, ...],
    pseudo_query: torch.Tensor,
    key_norm: nn.Module,
) -> torch.Tensor:
    if not sources:
        raise ValueError("Baseline AttnRes requires at least one source")
    shape = sources[0].shape
    if any(source.shape != shape for source in sources):
        raise ValueError("Baseline AttnRes source shapes differ")
    if pseudo_query.shape != (shape[-1],):
        raise ValueError("Baseline pseudo-query must have shape [hidden_size]")
    values = torch.stack(sources, dim=0)
    keys = key_norm(values)
    scores = torch.einsum("d,nbtd->nbt", pseudo_query.float(), keys.float())
    weights = scores.softmax(dim=0).to(values.dtype)
    return torch.einsum("nbt,nbtd->btd", weights, values)


class BaselineDecoderLayer(nn.Module):
    def __init__(self, config: BaselineQwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.attention_type = config.layer_types[layer_idx]
        self.attn_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.attn_key_norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.mlp_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.mlp_key_norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        completed_sources: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor,
        past_key_values: Cache | None,
        cache_position: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before_attention = completed_sources
        if partial_block is not None:
            before_attention += (partial_block,)
        z_attn = attnres_aggregate(
            before_attention,
            self.attn_pseudo_query,
            self.attn_key_norm,
        )
        attention_output, _ = self.self_attn(
            hidden_states=self.input_layernorm(z_attn),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        partial_after_attention = (
            attention_output
            if partial_block is None
            else partial_block + attention_output
        )
        z_mlp = attnres_aggregate(
            completed_sources + (partial_after_attention,),
            self.mlp_pseudo_query,
            self.mlp_key_norm,
        )
        mlp_output = self.mlp(self.post_attention_layernorm(z_mlp))
        return partial_after_attention + mlp_output, attention_output, mlp_output


class BaselineQwen3Model(Qwen3PreTrainedModel):
    config_class = BaselineQwen3Config

    def __init__(self, config: BaselineQwen3Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [BaselineDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.final_pseudo_query = nn.Parameter(torch.zeros(config.hidden_size))
        self.final_key_norm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.post_init()

    def _fixed_boundary_ends(self) -> frozenset[int]:
        if self.config.baseline_execution == "full":
            if self.config.baseline_partition is not None:
                raise ValueError("Full AttnRes baseline must not define a partition")
            return frozenset()
        lengths = self.config.baseline_partition
        expected = [4] * (self.config.num_hidden_layers // 4)
        if self.config.num_hidden_layers % 4 != 0 or lengths != expected:
            raise ValueError("Fixed baseline partition must use four-layer blocks")
        if sum(lengths) != self.config.num_hidden_layers:
            raise ValueError("Fixed baseline partition does not cover the backbone")
        ends: list[int] = []
        cursor = 0
        for length in lengths:
            cursor += length
            ends.append(cursor - 1)
        return frozenset(ends)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if use_cache:
            raise ValueError("Baseline AttnRes requires use_cache=False")
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
                "full_attention": _mask_factory_compat(
                    create_causal_mask,
                    **mask_kwargs,
                )
            }
            if self.has_sliding_layers:
                causal_masks["sliding_attention"] = _mask_factory_compat(
                    create_sliding_window_causal_mask,
                    **mask_kwargs,
                )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        boundary_ends = self._fixed_boundary_ends()
        completed_sources: tuple[torch.Tensor, ...] = (inputs_embeds,)
        partial_block: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            layer_args = (
                completed_sources,
                partial_block,
                causal_masks[layer.attention_type],
                position_ids,
                past_key_values,
                cache_position,
                position_embeddings,
            )
            if self.gradient_checkpointing and self.training:
                next_partial, attention_output, mlp_output = checkpoint(
                    layer,
                    *layer_args,
                    use_reentrant=False,
                )
            else:
                next_partial, attention_output, mlp_output = layer(*layer_args)
            if self.config.baseline_execution == "full":
                completed_sources += (attention_output, mlp_output)
                partial_block = None
            else:
                partial_block = next_partial
                if layer_idx in boundary_ends:
                    completed_sources += (partial_block,)
                    partial_block = None
        if partial_block is not None:
            raise RuntimeError("Fixed baseline ended with an unfinished block")
        z_final = attnres_aggregate(
            completed_sources,
            self.final_pseudo_query,
            self.final_key_norm,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=self.norm(z_final),
            past_key_values=None,
        )


class BaselineQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    config_class = BaselineQwen3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BaselineQwen3Config) -> None:
        super().__init__(config)
        self.model = BaselineQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

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
            **kwargs,
        )
        hidden_for_logits = (
            outputs.last_hidden_state[:, -logits_to_keep:, :]
            if logits_to_keep > 0
            else outputs.last_hidden_state
        )
        logits = self.lm_head(hidden_for_logits)
        loss = None
        if labels is not None:
            if logits_to_keep > 0:
                raise ValueError("logits_to_keep is only valid when labels are absent")
            if logits.shape[:2] != labels.shape:
                raise ValueError("Baseline labels must be pre-shifted and align with logits")
            loss = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
        )
