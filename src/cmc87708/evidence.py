from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .protocol import STRUCTURAL_FIELDS, assert_no_ground_truth, assert_structural_allowlist

CASE_ANCHOR_FIELDS = (
    "dataset", "split", "window_id", "graph_id", "edge_id",
)
TARGET_FLOW_FIELDS = (
    "src_ip", "dst_ip", "src_port", "dst_port", "protocol",
    "src_node_id", "dst_node_id", "src_endpoint_alias", "dst_endpoint_alias",
)
SIDECAR_FIELDS = CASE_ANCHOR_FIELDS + TARGET_FLOW_FIELDS
FLOW_FIELDS = ("src_ip", "dst_ip", "src_port", "dst_port", "protocol")
DETECTOR_FIELDS = ("model_name", "prediction", "score", "threshold")


def load_selected_sidecar_rows(
    sidecar_csv: str | Path,
    targets: pd.DataFrame,
    dataset_name: str,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Load only selected target rows from a potentially multi-GB sidecar CSV.

    Ground-truth columns are not included in `usecols`.
    """
    sidecar_csv = Path(sidecar_csv)
    header = pd.read_csv(sidecar_csv, nrows=0)
    available = set(header.columns)
    required = {"split", "window_id", "edge_id", "src_node_id", "dst_node_id"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Sidecar missing required columns: {missing}")

    usecols = [c for c in SIDECAR_FIELDS if c in available]
    selected_keys = set(zip(targets["window_id"].astype(str), targets["edge_id"].astype(str)))
    found: list[pd.DataFrame] = []
    found_keys: set[tuple[str, str]] = set()

    for chunk in pd.read_csv(sidecar_csv, usecols=usecols, chunksize=chunksize):
        keys = list(zip(chunk["window_id"].astype(str), chunk["edge_id"].astype(str)))
        mask = [k in selected_keys for k in keys]
        if any(mask):
            hit = chunk.loc[mask].copy()
            found.append(hit)
            found_keys.update(zip(hit["window_id"].astype(str), hit["edge_id"].astype(str)))
        if len(found_keys) == len(selected_keys):
            break

    if not found:
        raise ValueError(f"No selected target rows found in sidecar: {sidecar_csv}")
    result = pd.concat(found, ignore_index=True)
    if "dataset" not in result.columns:
        result["dataset"] = dataset_name
    if "split" not in result.columns:
        result["split"] = "test"

    missing_keys = selected_keys - set(zip(result["window_id"].astype(str), result["edge_id"].astype(str)))
    if missing_keys:
        raise ValueError(f"Missing {len(missing_keys)} selected target rows in sidecar")
    return result


def add_label_free_structural_context_streaming(
    sidecar_csv: str | Path,
    targets: pd.DataFrame,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Compute label-free structural statistics by streaming the sidecar once.

    Only window/node/pair keys that occur in the sampled targets are retained. No Label,
    Attack, detector prediction, or detector score is read or consulted.
    """
    sidecar_csv = Path(sidecar_csv)
    target_node_keys: set[tuple[str, Any]] = set()
    target_pair_keys: set[tuple[str, Any, Any]] = set()
    for row in targets.itertuples(index=False):
        w = str(getattr(row, "window_id"))
        src = getattr(row, "src_node_id")
        dst = getattr(row, "dst_node_id")
        target_node_keys.add((w, src))
        target_node_keys.add((w, dst))
        target_pair_keys.add((w, src, dst))

    in_degree: dict[tuple[str, Any], int] = defaultdict(int)
    out_degree: dict[tuple[str, Any], int] = defaultdict(int)
    pair_count: dict[tuple[str, Any, Any], int] = defaultdict(int)
    neighbors: dict[tuple[str, Any], set[Any]] = defaultdict(set)

    usecols = ["window_id", "src_node_id", "dst_node_id"]
    for chunk in pd.read_csv(sidecar_csv, usecols=usecols, chunksize=chunksize):
        for window, src, dst in chunk.itertuples(index=False, name=None):
            w = str(window)
            src_key = (w, src)
            dst_key = (w, dst)
            if src_key in target_node_keys:
                out_degree[src_key] += 1
                neighbors[src_key].add(dst)
            if dst_key in target_node_keys:
                in_degree[dst_key] += 1
                neighbors[dst_key].add(src)
            pair_key = (w, src, dst)
            if pair_key in target_pair_keys:
                pair_count[pair_key] += 1

    result = targets.copy()
    result["src_out_degree"] = [out_degree[(str(w), s)] for w, s in zip(result["window_id"], result["src_node_id"])]
    result["src_in_degree"] = [in_degree[(str(w), s)] for w, s in zip(result["window_id"], result["src_node_id"])]
    result["dst_out_degree"] = [out_degree[(str(w), d)] for w, d in zip(result["window_id"], result["dst_node_id"])]
    result["dst_in_degree"] = [in_degree[(str(w), d)] for w, d in zip(result["window_id"], result["dst_node_id"])]
    result["repeated_edge_count"] = [pair_count[(str(w), s, d)] for w, s, d in zip(result["window_id"], result["src_node_id"], result["dst_node_id"])]
    result["src_unique_neighbors"] = [len(neighbors[(str(w), s)]) for w, s in zip(result["window_id"], result["src_node_id"])]
    result["dst_unique_neighbors"] = [len(neighbors[(str(w), d)]) for w, d in zip(result["window_id"], result["dst_node_id"])]
    return result


def add_label_free_structural_context(sidecar: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """In-memory helper retained for unit tests and small datasets."""
    out_degree = sidecar.groupby(["window_id", "src_node_id"]).size().rename("src_out_degree")
    in_degree = sidecar.groupby(["window_id", "dst_node_id"]).size().rename("dst_in_degree")
    pair_count = sidecar.groupby(["window_id", "src_node_id", "dst_node_id"]).size().rename("repeated_edge_count")
    forward = sidecar[["window_id", "src_node_id", "dst_node_id"]].rename(columns={"src_node_id":"node_id","dst_node_id":"neighbor_id"})
    reverse = sidecar[["window_id", "src_node_id", "dst_node_id"]].rename(columns={"dst_node_id":"node_id","src_node_id":"neighbor_id"})
    neighbor_count = (
        pd.concat([forward, reverse], ignore_index=True)
        .drop_duplicates(["window_id", "node_id", "neighbor_id"])
        .groupby(["window_id", "node_id"]).size().rename("unique_neighbors")
    )
    result = targets.copy()
    src_keys = pd.MultiIndex.from_frame(result[["window_id", "src_node_id"]])
    dst_keys = pd.MultiIndex.from_frame(result[["window_id", "dst_node_id"]])
    pair_keys = pd.MultiIndex.from_frame(result[["window_id", "src_node_id", "dst_node_id"]])
    result["src_out_degree"] = out_degree.reindex(src_keys).fillna(0).to_numpy(dtype=int)
    result["src_in_degree"] = in_degree.reindex(src_keys).fillna(0).to_numpy(dtype=int)
    result["dst_out_degree"] = out_degree.reindex(dst_keys).fillna(0).to_numpy(dtype=int)
    result["dst_in_degree"] = in_degree.reindex(dst_keys).fillna(0).to_numpy(dtype=int)
    result["repeated_edge_count"] = pair_count.reindex(pair_keys).fillna(0).to_numpy(dtype=int)
    result["src_unique_neighbors"] = neighbor_count.reindex(src_keys).fillna(0).to_numpy(dtype=int)
    result["dst_unique_neighbors"] = neighbor_count.reindex(dst_keys).fillna(0).to_numpy(dtype=int)
    return result


def _visible_case_anchor(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in CASE_ANCHOR_FIELDS if k in record and pd.notna(record[k])}


def _visible_target_flow(record: dict[str, Any]) -> dict[str, Any]:
    fields = CASE_ANCHOR_FIELDS + TARGET_FLOW_FIELDS
    return {k: record[k] for k in fields if k in record and pd.notna(record[k])}


def build_packet(record: dict[str, Any], state: str) -> dict[str, Any]:
    # LLM-only is a true evidence-gap control: case anchor only. It receives no target-flow
    # metadata, graph-local statistics, or detector-state fields.
    target = _visible_case_anchor(record) if state == "llm_only" else _visible_target_flow(record)
    flow = {k: record[k] for k in FLOW_FIELDS if k in record and pd.notna(record[k])}
    graph = {k: record[k] for k in STRUCTURAL_FIELDS if k in record and pd.notna(record[k])}
    detector = {k: record[k] for k in DETECTOR_FIELDS if k in record and pd.notna(record[k])}
    if "score" in detector and "threshold" in detector:
        detector["score_margin"] = float(detector["score"]) - float(detector["threshold"])
    assert_structural_allowlist(graph)

    if state == "llm_only":
        flow, graph, detector = {}, {}, {}
    elif state == "flow":
        graph, detector = {}, {}
    elif state == "graph":
        detector = {}
    elif state != "detector":
        raise ValueError(f"Unknown state: {state}")

    packet = {
        "dataset": record.get("dataset"),
        "state": state,
        "target": target,
        "flow_evidence": flow,
        "graph_evidence": graph,
        "detector_state": detector,
        "traceability": {
            "split": record.get("split"),
            "window_id": record.get("window_id"),
            "edge_id": record.get("edge_id"),
        },
    }
    assert_no_ground_truth(packet, "R2 packet")
    return packet
