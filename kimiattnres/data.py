from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from datasets import DatasetDict, load_from_disk


@dataclass(frozen=True)
class Example:
    input_ids: torch.LongTensor
    labels: torch.LongTensor
    attention_mask: torch.LongTensor
    stable_id: str
    source_key: str = ""


def _value(row: dict[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing mapped field: {field}")
        value = value[part]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return value


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _path(root: Path, raw: str) -> Path:
    value = Path(raw)
    return value if value.is_absolute() else (root / value).resolve()


def _dataset_pool(source: dict[str, Any], root: Path):
    dataset = load_from_disk(str(_path(root, source["local_path"])))
    if isinstance(dataset, DatasetDict):
        split = str(source["official_split"])
        if split not in dataset:
            raise KeyError(f"Dataset does not contain split {split}: {source['local_path']}")
        dataset = dataset[split]
    return dataset, source["field_mapping"]


def _hotpot_context(context: Any) -> str:
    if isinstance(context, str):
        context = json.loads(context)
    if isinstance(context, dict):
        titles, sentences = context.get("title"), context.get("sentences")
        if not isinstance(titles, list) or not isinstance(sentences, list) or len(titles) != len(sentences):
            raise ValueError("Invalid HotpotQA context")
        return "\n".join(
            f"{title}: {' '.join(str(sentence) for sentence in paragraph)}"
            for title, paragraph in zip(titles, sentences)
        )
    if isinstance(context, list):
        values = []
        for paragraph in context:
            if isinstance(paragraph, dict):
                title = paragraph.get("title")
                text = paragraph.get("paragraph_text", paragraph.get("text", paragraph.get("sentences")))
            elif isinstance(paragraph, (list, tuple)) and len(paragraph) == 2:
                title, text = paragraph
            else:
                raise ValueError("Unsupported HotpotQA paragraph")
            if isinstance(text, (list, tuple)):
                text = " ".join(str(item) for item in text)
            values.append(f"{title}: {text}")
        return "\n".join(values)
    raise ValueError("Unsupported HotpotQA context")


def format_prompt(task: str, row: dict[str, Any], mapping: dict[str, Any]) -> str:
    if task == "math":
        return f"Question: {str(_value(row, mapping['question'])).strip()}\nAnswer:"
    if task != "multihop":
        raise ValueError(f"Kimi baseline only supports math/multihop, got {task}")
    if "context" in mapping:
        context = _hotpot_context(_value(row, mapping["context"]))
        question = str(_value(row, mapping["question"])).strip()
        return f"Context:\n{context}\nQuestion: {question}\nAnswer:"
    story = str(_value(row, mapping["story"])).strip()
    query = str(_value(row, mapping["query"])).strip()
    return f"Story: {story}\nQuery: {query}\nRelationship:"


def format_target(task: str, row: dict[str, Any], mapping: dict[str, Any]) -> str:
    if task == "math" or task == "multihop":
        target = str(_value(row, mapping["target"])).strip()
    else:
        raise ValueError(f"Kimi baseline only supports math/multihop, got {task}")
    if not target:
        raise ValueError(f"Empty {task} target")
    return target


def encode_example(
    tokenizer,
    *,
    task: str,
    row: dict[str, Any],
    mapping: dict[str, Any],
    stable_id: str,
    max_length: int,
    token_limit: int | None = None,
    source_key: str = "",
) -> Example:
    prompt_ids = tokenizer.encode(format_prompt(task, row, mapping), add_special_tokens=False)
    target_ids = tokenizer.encode(format_target(task, row, mapping), add_special_tokens=False)
    target_ids.append(int(tokenizer.eos_token_id))
    if len(target_ids) >= max_length:
        raise ValueError(f"{task} target is too long for max_length={max_length}")
    if token_limit is None:
        prompt_limit = max_length - len(target_ids)
    else:
        token_limit = int(token_limit)
        target_input_tokens = len(target_ids) - 1
        if token_limit <= 0 or target_input_tokens > token_limit:
            raise ValueError(
                f"{task} token_limit={token_limit} cannot preserve target "
                f"with {target_input_tokens} input tokens"
            )
        prompt_limit = token_limit - target_input_tokens
    prompt_ids = prompt_ids[: max(0, prompt_limit)]
    tokens = prompt_ids + target_ids
    input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
    labels = torch.tensor(tokens[1:], dtype=torch.long)
    prompt_label_end = max(0, len(prompt_ids) - 1)
    labels[:prompt_label_end] = -100
    return Example(
        input_ids=input_ids,
        labels=labels,
        attention_mask=torch.ones_like(input_ids),
        stable_id=stable_id,
        source_key=source_key,
    )


def collate(examples: Sequence[Example], pad_token_id: int) -> dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("Cannot collate empty examples")
    width = max(example.input_ids.numel() for example in examples)
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    for index, example in enumerate(examples):
        size = example.input_ids.numel()
        input_ids[index, :size] = example.input_ids
        labels[index, :size] = example.labels
        attention_mask[index, :size] = example.attention_mask
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def load_training_examples(
    *,
    repo_root: str | Path,
    data_manifest: str | Path,
    data_config: str | Path,
    tokenizer,
    tasks: Sequence[str],
    max_length: int,
    source_mix: Mapping[str, Mapping[str, float]],
    sampling: Mapping[str, Any],
) -> dict[str, tuple[Example, ...]]:
    if sampling.get("mode") != "fixed_ratio_round_robin":
        raise RuntimeError(
            "KIMI_SAMPLER_MODE_UNSUPPORTED: expected fixed_ratio_round_robin"
        )
    pool_size = int(sampling.get("pool_size_per_source", 0))
    if pool_size <= 0:
        raise RuntimeError("KIMI_SAMPLER_POOL_SIZE_INVALID")
    int(sampling.get("seed", 0))
    root = Path(repo_root).resolve()
    config = load_yaml(_path(root, str(data_config)))
    records = load_manifest(_path(root, str(data_manifest)))
    result: dict[str, tuple[Example, ...]] = {}
    for task in tasks:
        ratios = source_mix.get(task)
        if not isinstance(ratios, Mapping) or not ratios:
            raise RuntimeError(f"Missing source mix for {task}")
        source_keys = tuple(str(value) for value in ratios)
        expected_ratio = 1.0 / len(source_keys)
        if any(
            abs(float(ratios[key]) - expected_ratio) > 1.0e-8
            for key in source_keys
        ):
            raise RuntimeError(f"{task} source mix must be equal-ratio")
        source_by_dataset: dict[str, dict[str, Any]] = {}
        source_key_by_dataset: dict[str, str] = {}
        for source_key in source_keys:
            source = config.get("sources", {}).get(source_key)
            if source is None:
                raise RuntimeError(
                    f"{task} source mix references unknown source {source_key!r}"
                )
            dataset_name = str(source["dataset_name"])
            if dataset_name in source_by_dataset:
                raise RuntimeError(
                    f"{task} source mix maps multiple sources to {dataset_name!r}"
                )
            source_by_dataset[dataset_name] = source
            source_key_by_dataset[dataset_name] = source_key
        selected = [
            record for record in records
            if record.get("task") == task
            and record.get("assigned_split") == "stage3_adapter_train"
            and record.get("dataset") in source_by_dataset
        ]
        if not selected:
            raise RuntimeError(f"No stage3_adapter_train records for {task}")
        for record in selected:
            source = source_by_dataset[str(record["dataset"])]
            if str(record.get("official_split")) != str(source["official_split"]):
                raise RuntimeError(
                    f"{task} record uses split {record.get('official_split')!r} "
                    f"for source {record['dataset']!r}, expected {source['official_split']!r}"
                )
            if str(record.get("dataset_revision")) != str(source.get("revision")):
                raise RuntimeError(
                    f"{task} record revision mismatch for {record['dataset']!r}"
                )
        stable_ids = {str(record["stable_id"]) for record in selected}
        content_hashes = {str(record.get("content_sha256", "")) for record in selected}
        if len(stable_ids) != len(selected) or len(content_hashes) != len(selected):
            raise RuntimeError(f"Duplicate training record identity for {task}")
        forbidden = {
            "stage2_discovery",
            "stage3_adapter_val",
            "probe_train",
            "probe_val",
            "stage4_final_eval",
        }
        if any(
            record.get("assigned_split") in forbidden
            and (
                str(record.get("stable_id")) in stable_ids
                or str(record.get("content_sha256", "")) in content_hashes
            )
            for record in records
        ):
            raise RuntimeError(f"Training data overlaps a forbidden split for {task}")
        selected_by_dataset: dict[str, list[dict[str, Any]]] = {
            dataset: [] for dataset in source_by_dataset
        }
        for record in selected:
            selected_by_dataset[str(record["dataset"])].append(record)
        source_pools: list[list[Example]] = []
        for dataset_name in source_by_dataset:
            source_records = selected_by_dataset[dataset_name]
            if not source_records:
                raise RuntimeError(
                    f"KIMI_MIX_SOURCE_EMPTY: {task}/{dataset_name}"
                )
            source_records.sort(
                key=lambda record: (
                    int(record.get("source_pool_index", 0)),
                    str(record.get("split_key", "")),
                    str(record["stable_id"]),
                )
            )
            source_records = source_records[:pool_size]
            dataset, mapping = _dataset_pool(source_by_dataset[dataset_name], root)
            source_pools.append(
                [
                    encode_example(
                        tokenizer,
                        task=task,
                        row=dataset[int(record["row_index"])],
                        mapping=mapping,
                        stable_id=str(record["stable_id"]),
                        max_length=max_length,
                        token_limit=record.get("token_limit"),
                        source_key=source_key_by_dataset[dataset_name],
                    )
                    for record in source_records
                ]
            )
        cycle_length = max(len(pool) for pool in source_pools)
        examples = [
            source_pools[source_index][cycle_index % len(source_pools[source_index])]
            for cycle_index in range(cycle_length)
            for source_index in range(len(source_pools))
        ]
        result[task] = tuple(examples)
    return result


def _validation_source_map(
    config: Mapping[str, Any],
    source_keys: Sequence[str],
) -> dict[str, dict[str, Any]]:
    training_sources = {
        str(config["sources"][key]["dataset_name"]): config["sources"][key]
        for key in source_keys
    }
    sources = dict(training_sources)
    for validation_source in config.get("validation_sources", {}).values():
        if not isinstance(validation_source, Mapping):
            continue
        dataset_name = str(validation_source.get("dataset_name", ""))
        if dataset_name in training_sources:
            sources[dataset_name] = validation_source
    return sources


def load_validation_examples(
    *,
    repo_root: str | Path,
    data_manifest: str | Path,
    data_config: str | Path,
    tokenizer,
    tasks: Sequence[str],
    max_length: int,
    source_mix: Mapping[str, Mapping[str, float]],
    examples_per_task: int,
) -> dict[str, tuple[Example, ...]]:
    """Load the same fixed validation split used by the formal protocol."""
    if examples_per_task <= 0:
        raise ValueError("examples_per_task must be positive")
    root = Path(repo_root).resolve()
    config = load_yaml(_path(root, str(data_config)))
    records = load_manifest(_path(root, str(data_manifest)))
    result: dict[str, tuple[Example, ...]] = {}
    for task in tasks:
        source_keys = tuple(str(value) for value in source_mix[task])
        sources = _validation_source_map(config, source_keys)
        selected = sorted(
            (
                record for record in records
                if record.get("task") == task
                and record.get("assigned_split") == "stage3_adapter_val"
                and str(record.get("dataset")) in sources
            ),
            key=lambda record: str(record["stable_id"]),
        )[:examples_per_task]
        if not selected:
            raise RuntimeError(f"No stage3_adapter_val records for {task}")
        examples: list[Example] = []
        for record in selected:
            source = sources[str(record["dataset"])]
            if str(record.get("official_split")) != str(source["official_split"]):
                raise RuntimeError(f"Validation split mismatch for {task}/{record['dataset']}")
            dataset, mapping = _dataset_pool(source, root)
            examples.append(
                encode_example(
                    tokenizer,
                    task=task,
                    row=dataset[int(record["row_index"])],
                    mapping=mapping,
                    stable_id=str(record["stable_id"]),
                    max_length=max_length,
                    token_limit=record.get("token_limit"),
                    source_key=str(record["dataset"]),
                )
            )
        result[task] = tuple(examples)
    return result
