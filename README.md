# Qwen3 AttnRes Baselines

This repository contains the independent baselines used for the MoiraiBlock
post-training experiments:

- `full_attnres`: Full AttnRes with one shared pseudo-query bank;
- `fixed_block_attnres`: Kimi-compatible fixed four-layer blocks with one shared
  pseudo-query bank.

The implementation is intentionally separate from the task-adaptive pipeline.
No model weights, datasets, checkpoints, training outputs, or caches are
included in this repository.

## Repository layout

- `src/baselines/`: baseline model, training, and SVAMP evaluation entrypoints;
- `kimiattnres/`: current Native Kimi Block/Full implementation for Qwen3-1.7B,
  including the shared Math -> Multi-hop sequential trainer and FSDP launcher;
- `configs/baselines/`: full and fixed baseline configurations;
- `configs/data.yaml`: dataset source and manifest schema, with local paths that
  must be supplied by the execution environment;
- `tests/`: baseline protocol tests that do not require model weights.

## Environment

Create an environment with the pinned dependencies in `requirements.txt`.
Provide the local Qwen3 checkpoint and prepared dataset artifacts referenced by
the configuration files. These artifacts are deliberately external to the
repository.

## Commands

Validate a baseline configuration and local checkpoint/data inputs:

```bash
torchrun --standalone --nproc_per_node=1 -m src.baselines.train_full \
  --check-only --config configs/baselines/full.yaml
torchrun --standalone --nproc_per_node=1 -m src.baselines.train_fixed \
  --check-only --config configs/baselines/fixed.yaml
```

Run the protocol tests that only exercise configuration and model semantics:

```bash
pytest tests/test_baseline_protocol.py \
  -k 'not full_and_fixed_select_identical_unique_task_cases'
```

Training and evaluation write only to the configured `outputs/` paths, which
are ignored by Git.

The Native Kimi implementation is validated separately from the older
`src/baselines/` compatibility entrypoints. Its local checkpoint and prepared
manifest are intentionally external; use the configuration files under
`kimiattnres/configs/` with paths supplied by the execution environment.
