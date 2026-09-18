from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kimiattnres.modeling_qwen3_kimiattnres import (
    Qwen3KimiAttnResConfig,
    Qwen3KimiAttnResForCausalLM,
    convert_pretrained_qwen3,
)
from kimiattnres.data import _validation_source_map


def _config(
    *,
    mode: str = "block",
    block_sizes: list[int] | None = None,
    num_layers: int = 4,
) -> Qwen3KimiAttnResConfig:
    return Qwen3KimiAttnResConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * num_layers,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
        kimi_mode=mode,
        block_sizes=block_sizes,
    )


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def test_block_commits_explicit_fixed_boundaries() -> None:
    model = Qwen3KimiAttnResForCausalLM(_config(block_sizes=[2, 2])).eval()
    with torch.no_grad():
        output = model(**_batch(), use_cache=False, return_kimi_debug=True)
    assert output.kimi_debug["boundary_layer_indices"] == [1, 3]
    assert output.kimi_debug["completed_commit_layer_indices"] == [1, 3]
    assert output.kimi_debug["final_history_length"] == 3


def test_qwen3_1_7b_fixed_block_boundaries() -> None:
    model = Qwen3KimiAttnResForCausalLM(
        _config(block_sizes=[4, 4, 4, 4, 4, 4, 4], num_layers=28)
    ).eval()
    with torch.no_grad():
        output = model(**_batch(), use_cache=False, return_kimi_debug=True)
    assert output.kimi_debug["boundary_layer_indices"] == [3, 7, 11, 15, 19, 23, 27]
    assert output.kimi_debug["completed_commit_layer_indices"] == [3, 7, 11, 15, 19, 23, 27]
    assert output.kimi_debug["final_history_length"] == 8


def test_full_keeps_attention_and_mlp_history_entries() -> None:
    model = Qwen3KimiAttnResForCausalLM(_config(mode="full")).eval()
    with torch.no_grad():
        output = model(**_batch(), use_cache=False, return_kimi_debug=True)
    assert output.kimi_debug["completed_commit_layer_indices"] == []
    assert output.kimi_debug["final_history_length"] == 1 + 2 * 4


def test_pretrained_conversion_only_adds_kimi_parameters(tmp_path) -> None:
    native_config = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )
    native = Qwen3ForCausalLM(native_config)
    checkpoint = tmp_path / "native"
    native.save_pretrained(checkpoint, safe_serialization=True)
    original, converted, report = convert_pretrained_qwen3(
        checkpoint,
        mode="block",
        block_sizes=[2, 2],
        dtype=torch.float32,
    )
    assert report["unexpected_keys"] == []
    assert all(
        "pseudo_query" in key or "key_norm" in key
        for key in report["missing_keys"]
    )
    assert all("alpha" not in key for key in report["missing_keys"])
    assert all("alpha" not in key for key in converted.state_dict())
    with torch.no_grad():
        native_logits = original(**_batch(), use_cache=False).logits
        kimi_logits = converted(**_batch(), use_cache=False).logits
    assert torch.isfinite(kimi_logits).all()
    assert not torch.equal(native_logits, kimi_logits)


def test_native_query_and_rmsnorm_receive_updates() -> None:
    torch.manual_seed(7)
    model = Qwen3KimiAttnResForCausalLM(_config(block_sizes=[2, 2])).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
    before_query = model.model.layers[1].attn_pseudo_query.detach().clone()
    before_key_norm = model.model.layers[1].attn_key_norm.weight.detach().clone()
    labels = torch.tensor([[4, 5, 6, 7, 8]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(**_batch(), labels=labels, use_cache=False)
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        optimizer.step()
    assert not torch.equal(before_query, model.model.layers[1].attn_pseudo_query)
    assert not torch.equal(before_key_norm, model.model.layers[1].attn_key_norm.weight)


def test_validation_source_overrides_training_split() -> None:
    config = {
        "sources": {
            "clutrr": {"dataset_name": "clutrr", "official_split": "train"},
        },
        "validation_sources": {
            "multihop": {
                "dataset_name": "clutrr",
                "official_split": "validation",
            },
        },
    }
    resolved = _validation_source_map(config, ("clutrr",))
    assert resolved["clutrr"]["official_split"] == "validation"
