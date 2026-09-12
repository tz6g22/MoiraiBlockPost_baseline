from src.baselines.training import run_cli


if __name__ == "__main__":
    run_cli(
        baseline_type="fixed_block_attnres",
        default_config="configs/baselines/fixed.yaml",
    )
