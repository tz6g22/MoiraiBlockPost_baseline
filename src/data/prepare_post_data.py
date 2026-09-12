from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from src.common import canonical_json, load_yaml
from src.data.format_tasks import (
    TASK_TO_SOURCE,
    canonical_stable_id,
    canonical_content_sha256,
    load_local_split,
    nested_value,
    task_split_count,
)
from src.data.leakage_audit import audit_manifest, normalize_stage_overlap_pairs


def _stable_id(
    *,
    dataset_name: str,
    split: str,
    row: dict[str, Any],
    mapping: dict[str, Any],
) -> str:
    return canonical_stable_id(dataset_name, split, row, mapping)


def _order_key(seed: int, stable_id: str) -> str:
    return hashlib.sha256(f"{seed}::{stable_id}".encode("utf-8")).hexdigest()


def _load_source(source: dict[str, Any]):
    path = Path(source["local_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"Required original-protocol dataset is missing: {path}. "
            "Dataset substitution is disabled."
        )
    return load_local_split(path, str(source["official_split"]))


def _entry(
    *,
    task: str,
    source: dict[str, Any],
    split: str,
    row_index: int,
    row: dict[str, Any],
    assigned_split: str,
    seed: int,
) -> dict[str, Any]:
    dataset_name = str(source["dataset_name"])
    mapping = source["field_mapping"]
    stable_id = _stable_id(
        dataset_name=dataset_name,
        split=split,
        row=row,
        mapping=mapping,
    )
    return {
        "stable_id": stable_id,
        "dataset": dataset_name,
        "dataset_revision": source["revision"],
        "official_split": split,
        "row_index": row_index,
        "content_sha256": canonical_content_sha256(
            str(source["dataset_name"]),
            row,
            mapping,
        ),
        "split_key": _order_key(seed, stable_id),
        "assigned_split": assigned_split,
        "task": task,
    }


def _ordered_unique_pool(
    data_config: dict[str, Any],
    *,
    task: str,
    dataset_name: str,
    seed: int,
    section_names: tuple[str, ...] = ("sources",),
) -> list[tuple[str, dict[str, Any], str, int, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any], str, int, dict[str, Any]]] = []
    seen: set[str] = set()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for section_name in section_names:
        for source in data_config.get(section_name, {}).values():
            if source.get("dataset_name") == dataset_name:
                key = (str(source["local_path"]), str(source["official_split"]))
                candidates[key] = source
    if not candidates:
        raise ValueError(
            f"No {dataset_name} source is configured in sections {section_names}"
        )
    for source in (candidates[key] for key in sorted(candidates)):
        split = str(source["official_split"])
        dataset = _load_source(source)
        for row_index in range(len(dataset)):
            row = dataset[row_index]
            stable_id = _stable_id(
                dataset_name=str(source["dataset_name"]),
                split=split,
                row=row,
                mapping=source["field_mapping"],
            )
            if stable_id in seen:
                continue
            seen.add(stable_id)
            rows.append(
                (_order_key(seed, stable_id), source, split, row_index, row)
            )
    rows.sort(key=lambda value: value[0])
    return rows


def _ordered_unique_task_pool(
    data_config: dict[str, Any],
    *,
    task: str,
    source_keys: tuple[str, ...],
    seed: int,
    section_names: tuple[str, ...],
) -> list[tuple[str, dict[str, Any], str, int, dict[str, Any]]]:
    """Build one deterministic pool from several task-local source keys."""
    if not source_keys:
        raise ValueError(f"{task} has no configured source keys")
    rows: list[tuple[str, dict[str, Any], str, int, dict[str, Any]]] = []
    for source_key in source_keys:
        source = data_config.get("sources", {}).get(source_key)
        if source is None:
            raise ValueError(
                f"{task} source key {source_key!r} is not configured in sources"
            )
        rows.extend(
            _ordered_unique_pool(
                data_config,
                task=task,
                dataset_name=str(source["dataset_name"]),
                seed=seed,
                section_names=section_names,
            )
        )
    rows.sort(key=lambda value: value[0])
    return rows


def _stage_source_sections(
    data_config: dict[str, Any],
    *,
    task: str,
    dataset_name: str,
    stage: str,
) -> tuple[str, ...]:
    """Map each stage to an isolated source section.

    Evaluation must never draw from the generic training pool.  A task without
    a dedicated validation/probe source intentionally falls back to the
    training source section; the global assignment audit still makes those
    rows disjoint from earlier stages.
    """
    del task
    if stage == "stage4_final_eval":
        sections = ("evaluation_sources",)
    elif stage == "stage3_adapter_val":
        sections = (
            ("validation_sources",)
            if any(
                source.get("dataset_name") == dataset_name
                for source in data_config.get("validation_sources", {}).values()
            )
            else ("sources",)
        )
    elif stage in {"probe_train", "probe_val"}:
        sections = (
            "probe_sources",
            "sources",
        )
    else:
        sections = ("sources",)
    return sections


def _ordered_unique(
    dataset,
    *,
    task: str,
    source: dict[str, Any],
    split: str,
    seed: int,
) -> list[tuple[str, int, dict[str, Any]]]:
    del task
    rows: list[tuple[str, int, dict[str, Any]]] = []
    seen: set[str] = set()
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        stable_id = _stable_id(
            dataset_name=str(source["dataset_name"]),
            split=split,
            row=row,
            mapping=source["field_mapping"],
        )
        if stable_id in seen:
            continue
        seen.add(stable_id)
        rows.append((_order_key(seed, stable_id), row_index, row))
    rows.sort(key=lambda value: value[0])
    return rows


def _append_assignments(
    *,
    entries: list[dict[str, Any]],
    stable_id_splits: dict[str, set[str]],
    content_hash_splits: dict[str, set[str]],
    allowed_overlap_pairs: frozenset[frozenset[str]],
    rows: list[tuple[str, dict[str, Any], str, int, dict[str, Any]]],
    assignments: tuple[tuple[str, int], ...],
    task: str,
    seed: int,
) -> None:
    for assigned_split, count in assignments:
        selected = 0
        for _, source, split, row_index, row in rows:
            if selected == count:
                break
            entry = _entry(
                task=task,
                source=source,
                split=split,
                row_index=row_index,
                row=row,
                assigned_split=assigned_split,
                seed=seed,
            )
            existing_id_splits = stable_id_splits.get(entry["stable_id"], set())
            existing_content_splits = content_hash_splits.get(
                entry["content_sha256"],
                set(),
            )
            if any(
                other == assigned_split
                or frozenset((other, assigned_split)) not in allowed_overlap_pairs
                for other in existing_id_splits | existing_content_splits
            ):
                continue
            entries.append(entry)
            stable_id_splits.setdefault(entry["stable_id"], set()).add(assigned_split)
            content_hash_splits.setdefault(entry["content_sha256"], set()).add(
                assigned_split
            )
            selected += 1
        if selected != count:
            raise RuntimeError(
                f"{task}/{assigned_split} needs {count} unique eligible rows, "
                f"found {selected}"
            )


def prepare(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    counts = config["counts"]
    seed = int(config["split_seed"])
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "splits.json"
    entries: list[dict[str, Any]] = []
    stable_id_splits: dict[str, set[str]] = {}
    content_hash_splits: dict[str, set[str]] = {}
    configured_overlap_pairs = config.get("allowed_cross_stage_reuse", [])
    allowed_overlap_pairs = normalize_stage_overlap_pairs(configured_overlap_pairs)
    report: dict[str, Any] = {"tasks": {}, "counts": counts}
    discovery_sources = config["discovery_sources"]

    for task, source_key in TASK_TO_SOURCE.items():
        source = config["sources"][source_key]
        dataset_name = str(source["dataset_name"])
        training_source_keys = tuple(
            str(value)
            for value in config.get("training_sources", {}).get(task, (source_key,))
        )
        pool_available_by_stage: dict[str, dict[str, Any]] = {}
        stage_counts = (
            (
                "stage2_discovery",
                int(discovery_sources[task].get(source_key, 0)),
            ),
            (
                "stage3_adapter_train",
                task_split_count(
                    config,
                    task=task,
                    split_name="stage3_adapter_train",
                ),
            ),
            (
                "stage3_adapter_val",
                task_split_count(
                    config,
                    task=task,
                    split_name="stage3_adapter_val",
                ),
            ),
            (
                "probe_train",
                task_split_count(config, task=task, split_name="probe_train"),
            ),
            (
                "probe_val",
                task_split_count(config, task=task, split_name="probe_val"),
            ),
        )
        for assigned_split, count in stage_counts:
            if assigned_split == "stage3_adapter_train":
                pool_rows = _ordered_unique_task_pool(
                    config,
                    task=task,
                    source_keys=training_source_keys,
                    seed=seed,
                    section_names=("sources",),
                )
            else:
                pool_rows = _ordered_unique_pool(
                    config,
                    task=task,
                    dataset_name=dataset_name,
                    seed=seed,
                    section_names=_stage_source_sections(
                        config,
                        task=task,
                        dataset_name=dataset_name,
                        stage=assigned_split,
                    ),
                )
            pool_available_by_stage[assigned_split] = {
                "count": len(pool_rows),
                "official_splits": sorted({split for _, _, split, _, _ in pool_rows}),
            }
            _append_assignments(
                entries=entries,
                stable_id_splits=stable_id_splits,
                content_hash_splits=content_hash_splits,
                allowed_overlap_pairs=allowed_overlap_pairs,
                rows=pool_rows,
                assignments=((assigned_split, count),),
                task=task,
                seed=seed,
            )

        evaluation_count = task_split_count(
            config,
            task=task,
            split_name="stage4_final_eval",
        )
        pool_rows = _ordered_unique_pool(
            config,
            task=task,
            dataset_name=dataset_name,
            seed=seed,
            section_names=("evaluation_sources",),
        )
        pool_available_by_stage["stage4_final_eval"] = {
            "count": len(pool_rows),
            "official_splits": sorted({split for _, _, split, _, _ in pool_rows}),
        }
        _append_assignments(
            entries=entries,
            stable_id_splits=stable_id_splits,
            content_hash_splits=content_hash_splits,
            allowed_overlap_pairs=allowed_overlap_pairs,
            rows=pool_rows,
            assignments=(("stage4_final_eval", evaluation_count),),
            task=task,
            seed=seed,
        )
        report["tasks"][task] = {
            "pool_available_by_stage": pool_available_by_stage,
            "selected": {
                **dict(stage_counts),
                "stage2_discovery": int(counts["stage2_discovery"][task]),
                "stage2_discovery_by_source": discovery_sources[task],
                "stage4_final_eval": evaluation_count,
            },
        }

    for task, source_counts in discovery_sources.items():
        primary_source = TASK_TO_SOURCE[task]
        for source_key, requested_count in source_counts.items():
            if source_key == primary_source:
                continue
            source = config["sources"][source_key]
            rows = _ordered_unique_pool(
                config,
                task=task,
                dataset_name=str(source["dataset_name"]),
                seed=seed,
                section_names=("sources",),
            )
            count = int(requested_count)
            if len(rows) < count:
                raise RuntimeError(
                    f"{task} discovery source {source_key} needs {count} unique rows, "
                    f"got {len(rows)}"
                )
            _append_assignments(
                entries=entries,
                stable_id_splits=stable_id_splits,
                content_hash_splits=content_hash_splits,
                allowed_overlap_pairs=allowed_overlap_pairs,
                rows=rows,
                assignments=(("stage2_discovery", count),),
                task=task,
                seed=seed,
            )

    entries.sort(key=lambda row: (row["task"], row["assigned_split"], row["split_key"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(canonical_json(entry) + "\n")
    temporary.replace(manifest_path)
    leakage = audit_manifest(
        manifest_path,
        allowed_cross_stage_reuse=configured_overlap_pairs,
    )
    if leakage["status"] != "PASS":
        raise RuntimeError("Generated post-training manifest contains data leakage")
    report.update(
        {
            "status": "PASS",
            "manifest_path": str(manifest_path),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "leakage_audit": leakage,
        }
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(prepare(parse_args().config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
