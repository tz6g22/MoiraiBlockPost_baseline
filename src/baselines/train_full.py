from src.baselines.training import run_cli


if __name__ == "__main__":
    run_cli(
        baseline_type="full_attnres",
        default_config="configs/baselines/full.yaml",
    )
