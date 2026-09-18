from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from src.data.format_tasks import (
    canonical_content_sha256,
    canonical_stable_id,
    load_local_split,
)
from kimiattnres.data import load_yaml


ROOT = Path(__file__).resolve().parents[2]
SEED = 42
POOL_SIZE = 1024
OUTPUT = ROOT / "kimiattnres/configs/qwen3_1.7b_1m_manifest.jsonl"
DATA_CONFIG = ROOT / "configs/data_qwen3_1.7b.yaml"
OLD_MANIFEST = ROOT / "outputs/data/qwen3_1.7b/splits.json"

SOURCE_MIX = {
    "math": ("svamp", "gsm8k", "math_train", "openmathinstruct2"),
    "multihop": ("clutrr", "musique", "hotpotqa", "2wikimultihopqa"),
}
FORBIDDEN_SPLITS = {
    "stage2_discovery",
    "stage3_adapter_val",
    "probe_train",
    "probe_val",
    "stage4_final_eval",
}


def order_key(seed: int, dataset: str, row_index: int) -> str:
    return hashlib.sha256(f"{seed}:{dataset}:{row_index}".encode()).hexdigest()


def main() -> None:
    data_config = load_yaml(DATA_CONFIG)
    forbidden_ids: set[str] = set()
    forbidden_hashes: set[str] = set()
    if OLD_MANIFEST.is_file():
        for line in OLD_MANIFEST.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("assigned_split") in FORBIDDEN_SPLITS:
                forbidden_ids.add(str(record["stable_id"]))
                forbidden_hashes.add(str(record["content_sha256"]))

    entries: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selected_hashes: set[str] = set()
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for task, source_keys in SOURCE_MIX.items():
        for source_key in source_keys:
            source = data_config["sources"][source_key]
            split = str(source["official_split"])
            dataset_name = str(source["dataset_name"])
            dataset = load_local_split(ROOT / source["local_path"], split)
            indices = list(range(len(dataset)))
            random.Random(f"{SEED}:{task}:{source_key}").shuffle(indices)
            selected = 0
            for row_index in indices:
                row = dataset[row_index]
                stable_id = canonical_stable_id(
                    dataset_name, split, row, source["field_mapping"]
                )
                content_hash = canonical_content_sha256(
                    dataset_name, row, source["field_mapping"]
                )
                if stable_id in forbidden_ids or content_hash in forbidden_hashes:
                    continue
                if stable_id in selected_ids or content_hash in selected_hashes:
                    continue
                entries.append(
                    {
                        "assigned_split": "stage3_adapter_train",
                        "content_sha256": content_hash,
                        "dataset": dataset_name,
                        "dataset_revision": source["revision"],
                        "official_split": split,
                        "row_index": row_index,
                        "sampling_seed": SEED,
                        "source_key": source_key,
                        "source_pool_index": selected,
                        "split_key": order_key(SEED, dataset_name, row_index),
                        "stable_id": stable_id,
                        "task": task,
                    }
                )
                selected_ids.add(stable_id)
                selected_hashes.add(content_hash)
                selected += 1
                if selected == POOL_SIZE:
                    break
            if selected == 0:
                raise RuntimeError(f"No eligible rows for {task}/{source_key}")
            counts[task][source_key] = selected

    entries.sort(key=lambda item: (str(item["task"]), int(item["source_pool_index"]), str(item["source_key"])))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"manifest": str(OUTPUT), "records": len(entries), "counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
