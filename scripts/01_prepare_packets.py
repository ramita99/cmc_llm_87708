from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from cmc87708.evidence import (
    add_label_free_structural_context_streaming,
    build_packet,
    load_selected_sidecar_rows,
)
from cmc87708.io import read_table
from cmc87708.protocol import FORBIDDEN_FIELDS, assert_no_ground_truth, normalize_key, stable_hash
from cmc87708.sampling import sample_window_stratified_uniform

JOIN_KEYS = ["dataset", "split", "window_id", "edge_id"]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def target_ids(df: pd.DataFrame) -> list[tuple[str, str]]:
    return list(zip(df["window_id"].astype(str), df["edge_id"].astype(str)))


def drop_ground_truth_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped = [c for c in df.columns if normalize_key(c) in FORBIDDEN_FIELDS]
    return df.drop(columns=dropped, errors="ignore"), dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/primary_final.yaml")
    ap.add_argument("--datasets", default="configs/datasets.example.yaml")
    args = ap.parse_args()

    exp = load_yaml(args.experiment)
    data_cfg = load_yaml(args.datasets)
    out = Path(exp["output_root"])
    (out / "packets").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    run_manifest = []
    for ds in data_cfg["datasets"]:
        name = ds["name"]
        detector = read_table(ds["detector"])
        if "dataset" not in detector.columns:
            detector["dataset"] = name
        detector = detector[detector["split"].astype(str) == "test"].copy()

        # Stored detector artifacts may contain offline ground truth. Remove all such columns
        # before constructing the CMC 87708 sampling frame.
        detector_sampling, dropped_gt_columns = drop_ground_truth_columns(detector)

        sampled = sample_window_stratified_uniform(
            detector_sampling, int(exp["targets_per_dataset"]), int(exp["sample_seed"])
        )
        sampled_repeat = sample_window_stratified_uniform(
            detector_sampling, int(exp["targets_per_dataset"]), int(exp["sample_seed"])
        )
        gate_b_pass = target_ids(sampled) == target_ids(sampled_repeat)
        if not gate_b_pass:
            raise AssertionError(f"Gate B failed for {name}: repeated fixed-seed sampling changed target IDs")

        side_targets = load_selected_sidecar_rows(ds["sidecar"], sampled, name)
        detector_fields = [c for c in [
            "dataset", "split", "window_id", "edge_id", "model_name",
            "prediction", "score", "threshold", "feature_condition", "seed",
            "sampling_seed", "sampling_stratum", "selection_order",
        ] if c in sampled.columns]
        targets = side_targets.merge(
            sampled[detector_fields], on=JOIN_KEYS, how="inner", validate="one_to_one"
        )
        if len(targets) != len(sampled):
            raise AssertionError(
                f"Selected-target sidecar join mismatch for {name}: {len(targets)} != {len(sampled)}"
            )

        targets = add_label_free_structural_context_streaming(ds["sidecar"], targets)

        packets = []
        packet_hashes = []
        for _, row in targets.iterrows():
            record = row.dropna().to_dict()
            for state in exp["states"]:
                packet = build_packet(record, state)
                assert_no_ground_truth(packet)
                packet_hashes.append(stable_hash(packet))
                packets.append(packet)

        # Gate A: every complete LLM-facing packet must be free of ground-truth fields.
        for packet in packets:
            assert_no_ground_truth(packet, f"Gate A packet {name}")
        gate_a_pass = True

        packet_path = out / "packets" / f"{name}.jsonl"
        with packet_path.open("w", encoding="utf-8") as f:
            for p in packets:
                # Do not put audit hashes inside the packet itself: the serialized packet is
                # exactly what downstream prompting is allowed to expose to the LLM.
                f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")

        manifest_cols = [c for c in [
            "dataset", "split", "window_id", "edge_id", "prediction", "score", "threshold",
            "model_name", "feature_condition", "seed",
            "sampling_seed", "sampling_stratum", "selection_order"
        ] if c in targets.columns]
        target_manifest = targets[manifest_cols].copy().sort_values(["window_id", "edge_id"]).reset_index(drop=True)
        target_path = out / "manifests" / f"target_manifest_{name}.csv"
        target_manifest.to_csv(target_path, index=False)

        duplicate_targets = int(target_manifest.duplicated(["window_id", "edge_id"]).sum())
        if duplicate_targets:
            raise AssertionError(f"Duplicate sampled target IDs for {name}: {duplicate_targets}")

        run_manifest.append({
            "dataset": name,
            "eligible_test_edges": len(detector_sampling),
            "selected_targets": len(targets),
            "packet_records": len(packets),
            "target_manifest_hash": stable_hash(target_manifest.to_dict(orient="records")),
            "packet_bundle_hash": stable_hash(packet_hashes),
            "sampling_seed": exp["sample_seed"],
            "ground_truth_columns_removed_before_sampling": ";".join(dropped_gt_columns),
            "ground_truth_used_for_sampling": False,
            "ground_truth_exposed_to_llm": False,
            "gate_a_no_ground_truth_exposure": gate_a_pass,
            "gate_b_sampling_reproducible": gate_b_pass,
            "duplicate_target_ids": duplicate_targets,
            "sidecar_processing": "streamed_target_lookup_and_structural_scan",
        })

    report = pd.DataFrame(run_manifest)
    report.to_csv(out / "manifests" / "run_manifest.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
