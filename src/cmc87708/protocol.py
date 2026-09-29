from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

FORBIDDEN_FIELDS = {
    "label", "label_raw", "attack", "attack_type", "attack_subtype",
    "ground_truth", "groundtruth", "true_label", "target_label", "y", "edge_y",
}

# Run/audit metadata is stored outside the evidence packet. These fields must never be
# serialized into the payload shown to the LLM, because they are not security evidence.
AUDIT_METADATA_FIELDS = {
    "packet_hash", "prompt_hash", "output_hash", "native_output_hash",
    "record_key", "audit_pass", "audit_failures", "fallback_used",
}

STRUCTURAL_FIELDS = (
    "src_in_degree", "src_out_degree", "dst_in_degree", "dst_out_degree",
    "repeated_edge_count", "src_unique_neighbors", "dst_unique_neighbors",
)


def normalize_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _find_named_fields(value: Any, forbidden: set[str], path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if normalize_key(key) in forbidden:
                hits.append(child_path)
            hits.extend(_find_named_fields(child, forbidden, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for i, child in enumerate(value):
            hits.extend(_find_named_fields(child, forbidden, f"{path}[{i}]"))
    return hits


def find_forbidden_fields(value: Any, path: str = "$") -> list[str]:
    return _find_named_fields(value, FORBIDDEN_FIELDS, path)


def find_audit_metadata_fields(value: Any, path: str = "$") -> list[str]:
    return _find_named_fields(value, AUDIT_METADATA_FIELDS, path)


def assert_no_ground_truth(value: Any, context: str = "LLM-facing payload") -> None:
    hits = find_forbidden_fields(value)
    if hits:
        raise AssertionError(f"{context} exposes forbidden field(s): {', '.join(sorted(hits))}")


def assert_no_audit_metadata(value: Any, context: str = "LLM-facing payload") -> None:
    hits = find_audit_metadata_fields(value)
    if hits:
        raise AssertionError(f"{context} exposes audit-only field(s): {', '.join(sorted(hits))}")


def assert_llm_facing_packet(value: Any, context: str = "LLM-facing packet") -> None:
    assert_no_ground_truth(value, context)
    assert_no_audit_metadata(value, context)


def assert_structural_allowlist(graph: Mapping[str, Any]) -> None:
    unexpected = sorted(set(map(str, graph.keys())) - set(STRUCTURAL_FIELDS))
    if unexpected:
        raise AssertionError("Unexpected graph fields: " + ", ".join(unexpected))


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()
