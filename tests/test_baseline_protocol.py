from __future__ import annotations

from pathlib import Path

import torch

import src.baselines.modeling as baseline_modeling
from src.baselines.modeling import (
    BaselineQwen3Config,
    BaselineQwen3Model,
    attnres_aggregate,
)
from src.baselines.training import (
    FIXED_PARTITION,
    freeze_except_pseudo_query,
    select_training_records,
    validate_baseline_config,
)
from src.common import load_yaml, sha256_json


def test_independent_baseline_configs_are_exactly_one_pass() -> None:
    full = load_yaml("configs/baselines/full.yaml")
    fixed = load_yaml("configs/baselines/fixed.yaml")
    validate_baseline_config(full, expected_type="full_attnres")
    validate_baseline_config(fixed, expected_type="fixed_block_attnres")
    for config in (full, fixed):
        assert config["training_cases_per_task"] == 200
        assert config["training_task_order"] == ["math", "multihop"]
        assert config["training_passes"] == 1
        assert config["micro_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 1
        assert config["training_source_split"] == "stage3_adapter_train"
        assert config["selection_order"] == "stable_id"
    assert full["fixed_partition"] is None
    assert fixed["fixed_partition"] == FIXED_PARTITION
    assert sum(fixed["fixed_partition"]) == 40


def test_each_baseline_has_one_shared_query_output() -> None:
    for path in ("src/baselines/train_full.py", "src/baselines/train_fixed.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert "run_cli" in source
    training = Path("src/baselines/training.py").read_text(encoding="utf-8")
    assert 'choices=("all",)' in training
    assert 'output_dir = Path(config["output_root"])' in training
    assert '"shared_query_across_tasks": True' in training


def test_full_and_fixed_select_identical_unique_task_cases() -> None:
    full = load_yaml("configs/baselines/full.yaml")
    fixed = load_yaml("configs/baselines/fixed.yaml")
    for task in ("math", "multihop"):
        full_records, _ = select_training_records(task=task, config=full)
        fixed_records, _ = select_training_records(task=task, config=fixed)
        full_ids = [record["stable_id"] for record in full_records]
        fixed_ids = [record["stable_id"] for record in fixed_records]
        assert full_ids == sorted(full_ids)
        assert full_ids == fixed_ids
        assert len(full_ids) == len(set(full_ids)) == 200
        assert sha256_json(full_ids) == sha256_json(fixed_ids)
        assert {
            record["assigned_split"] for record in full_records
        } == {"stage3_adapter_train"}


def test_freeze_check_is_strictly_pseudo_query_only() -> None:
    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 4)
            self.attn_pseudo_query = torch.nn.Parameter(torch.zeros(4))
            self.mlp = torch.nn.Linear(4, 4)

    model = Toy()
    names = freeze_except_pseudo_query(model)
    assert names == ("attn_pseudo_query",)
    assert model.attn_pseudo_query.requires_grad
    assert not model.embedding.weight.requires_grad
    assert not model.mlp.weight.requires_grad
    assert not model.mlp.bias.requires_grad


def test_attnres_aggregate_matches_kimi_equation() -> None:
    sources = (
        torch.tensor([[[1.0, 2.0, 3.0]]]),
        torch.tensor([[[4.0, 6.0, 8.0]]]),
        torch.tensor([[[2.0, 5.0, 9.0]]]),
    )
    query = torch.tensor([0.5, -0.25, 0.75])
    norm = torch.nn.RMSNorm(3, eps=1.0e-6)

    actual = attnres_aggregate(sources, query, norm)
    values = torch.stack(sources, dim=0)
    keys = norm(values)
    expected_weights = torch.einsum("d,nbtd->nbt", query, keys).softmax(dim=0)
    expected = torch.einsum("nbt,nbtd->btd", expected_weights, values)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        attnres_aggregate(sources, torch.zeros_like(query), norm),
        values.mean(dim=0),
    )


def test_logits_to_keep_preserves_last_token_logits() -> None:
    model = _tiny_28_layer_model("fixed")
    causal_lm = baseline_modeling.BaselineQwen3ForCausalLM(model.config).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        full = causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        last = causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        ).logits
    assert last.shape == (1, 1, model.config.vocab_size)
    torch.testing.assert_close(last, full[:, -1:, :])


def _tiny_28_layer_model(execution: str) -> BaselineQwen3Model:
    return BaselineQwen3Model(
        BaselineQwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=28,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            layer_types=["full_attention"] * 28,
            baseline_execution=execution,
            baseline_partition=None if execution == "full" else [4] * 7,
            use_cache=False,
        )
    ).eval()


def test_full_and_fixed_source_histories_match_kimi_semantics(monkeypatch) -> None:
    original = baseline_modeling.attnres_aggregate
    source_counts: list[int] = []

    def recording_aggregate(sources, pseudo_query, key_norm):
        source_counts.append(len(sources))
        return original(sources, pseudo_query, key_norm)

    monkeypatch.setattr(baseline_modeling, "attnres_aggregate", recording_aggregate)
    inputs = torch.randn(1, 2, 8)
    attention_mask = torch.ones(1, 2, dtype=torch.long)

    with torch.no_grad():
        _tiny_28_layer_model("full")(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
        )
    expected_full: list[int] = []
    for transformer_block in range(28):
        expected_full.extend(
            [1 + 2 * transformer_block, 2 + 2 * transformer_block]
        )
    expected_full.append(57)
    assert source_counts == expected_full

    source_counts.clear()
    with torch.no_grad():
        fixed_model = _tiny_28_layer_model("fixed")
        fixed_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            use_cache=False,
        )
    expected_fixed: list[int] = []
    for transformer_block in range(28):
        completed = 1 + transformer_block // 4
        partial = int(transformer_block % 4 != 0)
        expected_fixed.extend([completed + partial, completed + 1])
    expected_fixed.append(8)
    assert source_counts == expected_fixed
    assert fixed_model._fixed_boundary_ends() == frozenset(
        {3, 7, 11, 15, 19, 23, 27}
    )


def test_baseline_code_is_not_wired_into_main_pipeline() -> None:
    baseline_model = Path("src/baselines/modeling.py").read_text(encoding="utf-8")
    baseline_training = Path("src/baselines/training.py").read_text(encoding="utf-8")
    for forbidden_import in (
        "src.adapter",
        "src.discovery",
        "src.modeling.full_attnres",
        "src.modeling.partition",
        "src.probe",
    ):
        assert forbidden_import not in baseline_model
        assert forbidden_import not in baseline_training
