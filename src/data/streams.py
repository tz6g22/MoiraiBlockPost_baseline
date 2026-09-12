from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.data.format_tasks import (
    PromptOnlyExample,
    TargetCausalExample,
    encode_prompt_only,
    encode_prompt_target,
)


@dataclass(frozen=True)
class ManifestTaskRows:
    task: str
    records: tuple[dict[str, Any], ...]
    dataset: Any
    field_mapping: dict[str, Any]

    def target_examples(
        self,
        tokenizer,
        *,
        max_length: int = 2048,
    ) -> tuple[TargetCausalExample, ...]:
        return tuple(
            encode_prompt_target(
                tokenizer,
                task=self.task,
                row=self.dataset[int(record["row_index"])],
                field_mapping=self.field_mapping,
                stable_id=str(record["stable_id"]),
                max_length=max_length,
            )
            for record in self.records
        )

    def prompt_examples(
        self,
        tokenizer,
        *,
        max_length: int = 2048,
    ) -> tuple[PromptOnlyExample, ...]:
        return tuple(
            encode_prompt_only(
                tokenizer,
                task=self.task,
                row=self.dataset[int(record["row_index"])],
                field_mapping=self.field_mapping,
                stable_id=str(record["stable_id"]),
                max_length=max_length,
            )
            for record in self.records
        )
