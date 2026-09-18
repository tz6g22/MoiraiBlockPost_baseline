from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from kimiattnres.data import load_training_examples, load_yaml


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    configs = [
        ROOT / "kimiattnres/configs/qwen3_1.7b_block.yaml",
        ROOT / "kimiattnres/configs/qwen3_1.7b_full.yaml",
    ]
    reports = []
    for config_path in configs:
        config = load_yaml(config_path)
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / config["base_checkpoint"], local_files_only=True, use_fast=True
        )
        source_mix = config["training"]["source_mix"]
        sampling = config["training"]["sampler"]
        examples_by_task = load_training_examples(
            repo_root=ROOT,
            data_manifest=config["data_manifest"],
            data_config=config["data_config"],
            tokenizer=tokenizer,
            tasks=tuple(config["training"]["task_order"]),
            max_length=int(config["max_sequence_length"]),
            source_mix=source_mix,
            sampling=sampling,
        )
        task_report = {}
        for task, examples in examples_by_task.items():
            budget = int(config["training"]["token_budget"][task])
            counts = Counter()
            tokens = defaultdict(int)
            position = 0
            total = 0
            while total < budget:
                example = examples[position % len(examples)]
                position += 1
                used = min(int(example.input_ids.numel()), budget - total)
                source = example.source_key
                counts[source] += 1
                tokens[source] += used
                total += used
            task_report[task] = {
                "budget": budget,
                "cycle_examples": len(examples),
                "first_sources": [example.source_key for example in examples[:4]],
                "cycle_counts": dict(Counter(example.source_key for example in examples)),
                "simulated_draw_counts": dict(counts),
                "simulated_token_counts": dict(tokens),
                "simulated_total_tokens": total,
                "simulated_sample_ratios": {
                    source: counts[source] / sum(counts.values())
                    for source in source_mix[task]
                },
                "target_ratios": source_mix[task],
            }
        reports.append({"config": str(config_path.relative_to(ROOT)), "tasks": task_report})
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
