from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def normalize_stage_overlap_pairs(
    pairs: Iterable[Iterable[str]],
) -> frozenset[frozenset[str]]:
    normalized: set[frozenset[str]] = set()
    for pair in pairs:
        values = tuple(str(value) for value in pair)
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError("Each allowed stage overlap must contain two distinct stages")
        normalized.add(frozenset(values))
    return frozenset(normalized)


def audit_manifest(
    path: str | Path,
    *,
    allowed_cross_stage_reuse: Iterable[Iterable[str]] = (),
) -> dict:
    allowed_pairs = normalize_stage_overlap_pairs(allowed_cross_stage_reuse)
    split_ids: dict[str, set[str]] = defaultdict(set)
    split_content: dict[str, set[str]] = defaultdict(set)
    duplicate_ids: list[dict[str, str]] = []
    duplicate_content: list[dict[str, str]] = []
    allowed_id_counts: dict[str, int] = defaultdict(int)
    allowed_content_counts: dict[str, int] = defaultdict(int)

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            for key in {
                "stable_id",
                "dataset",
                "dataset_revision",
                "official_split",
                "assigned_split",
                "content_sha256",
            }:
                if key not in row:
                    raise ValueError(f"Manifest line {line_number} is missing {key}")
            split_name = row["assigned_split"]
            stable_id = row["stable_id"]
            content_hash = row["content_sha256"]
            for other_split, ids in split_ids.items():
                if stable_id not in ids:
                    continue
                pair = frozenset((other_split, split_name))
                if other_split != split_name and pair in allowed_pairs:
                    key = "<->".join(sorted(pair))
                    allowed_id_counts[key] += 1
                else:
                    duplicate_ids.append(
                        {"stable_id": stable_id, "a": other_split, "b": split_name}
                    )
            for other_split, hashes in split_content.items():
                if content_hash not in hashes:
                    continue
                pair = frozenset((other_split, split_name))
                if other_split != split_name and pair in allowed_pairs:
                    key = "<->".join(sorted(pair))
                    allowed_content_counts[key] += 1
                else:
                    duplicate_content.append(
                        {
                            "content_sha256": content_hash,
                            "a": other_split,
                            "b": split_name,
                        }
                    )
            split_ids[split_name].add(stable_id)
            split_content[split_name].add(content_hash)

    return {
        "status": "PASS" if not duplicate_ids and not duplicate_content else "FAILED",
        "allowed_cross_stage_reuse": [
            sorted(pair) for pair in sorted(allowed_pairs, key=lambda value: sorted(value))
        ],
        "allowed_id_intersection_counts": dict(sorted(allowed_id_counts.items())),
        "allowed_content_intersection_counts": dict(
            sorted(allowed_content_counts.items())
        ),
        "id_intersections": duplicate_ids,
        "content_hash_intersections": duplicate_content,
        "split_unique_counts": {
            split_name: len(ids) for split_name, ids in sorted(split_ids.items())
        },
    }
