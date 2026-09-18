# Kimi Attention Residuals Baselines

This directory is an independent Block AttnRes / Full AttnRes baseline. It
does not import the TA-IBAR/MoiraiBlock model or trainer and writes only under
`kimiattnres/outputs/`.

## Local audit used for the configs

- Checkpoint: `artifacts/models/Qwen3-1.7B`.
- Native architecture read from the checkpoint: 28 layers, hidden size 2048,
  intermediate size 6144, 16 attention heads, 8 KV heads, head size 128,
  BF16 weights.
- Training manifest: `kimiattnres/configs/qwen3_1.7b_1m_manifest.jsonl`.
- Training split: `stage3_adapter_train` records selected from the local
  source snapshots after the existing discovery/validation/probe exclusions.
  Math uses `GSM8K` and Multi-hop uses `CLUTRR`; each task has a deterministic
  single-source pool that may be reused after exhaustion.
- Formatting: the local `Question: ...\nAnswer:` and multi-hop context/story
  prompts are reproduced in `data.py`; target tokens plus EOS are supervised,
  prompt and padding labels are `-100`.
- Formal token budgets: 500,000 non-padding input tokens per task, 1,000,000
  total. The
  existing formal config uses micro-batch 1, accumulation 1, BF16, FP32 loss
  accumulation, `use_cache=false`, gradient checkpointing, AdamW, backbone LR
  `5e-6`, AttnRes LR `3e-5`, betas `(0.9, 0.95)`, backbone weight decay
  `0.1`, AttnRes weight decay `0`, cosine warmup `0.03`, and clip norm `1.0`.
- The existing formal trainer keeps one shared optimizer/scheduler and one
  Backbone through Math then Multi-hop. This independent runner follows that
  parameter continuity. It does not import the formal trainer. The data plan
  is consumed by the configured deterministic mixed sampler.
- The existing formal entry has an FSDP path. This independent implementation
  uses the same FULL_SHARD/BF16/per-decoder-layer checkpointing path and has
  been exercised with two local GPUs for the short Block and Full smoke runs.

## Model semantics

The converter loads native Qwen3 weights with `strict=False`, then rejects any
missing native Qwen parameter or unexpected key. Only new Kimi pseudo-query and
RMSNorm parameters may be missing. Every pseudo-query is zero-initialized, so
the initial depth routing is uniform; there is no Alpha or residual gate.

Both modes use one shared per-site parameter set across Math and Multi-hop:
there are no task-specific query banks. Routing is single-head depth softmax
over stacked sources after RMSNorm of the keys.

Block mode uses the explicit Qwen3-1.7B partition:

```text
[4, 4, 4, 4, 4, 4, 4]
```

Full mode has no block partition and keeps embedding plus each post-Attention
and post-MLP cumulative source. Block mode commits completed cumulative block
residuals at layer indices `3, 7, 11, 15, 19, 23, 27`.

## Commands

CPU/unit tests:

```bash
PYTHONPATH=. .venv/bin/python -m pytest kimiattnres/tests -q
```

GPU smoke, three Math steps, explicit checkpoint/reload, then three Multi-hop
steps:

```bash
bash kimiattnres/scripts/smoke_block.sh
bash kimiattnres/scripts/smoke_full.sh
```

Formal training commands exist in `train_block.sh` and `train_full.sh`, but
they are not run by the implementation step.
