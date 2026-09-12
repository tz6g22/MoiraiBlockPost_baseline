from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_sha256(tokenizer) -> str:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("MoiraiBlock requires a fast tokenizer for stable hashing")
    payload = {
        "backend": json.loads(tokenizer.backend_tokenizer.to_str()),
        "special_tokens_map": {
            key: str(value)
            for key, value in sorted(tokenizer.special_tokens_map.items())
        },
        "vocab_size": len(tokenizer),
    }
    return sha256_json(payload)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return value


def config_sha256(path: str | Path) -> str:
    return sha256_json(load_yaml(path))


def require_keys(mapping: dict[str, Any], keys: set[str], *, context: str) -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")
