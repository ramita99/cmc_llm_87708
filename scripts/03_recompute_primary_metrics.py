from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from cmc87708.evaluation import protected_field_metrics
from cmc87708.evaluation_v2 import AUDIT_RULE_VERSION, unsupported_claim_flags


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prefixed(prefix: str, values: dict) -> dict:
    return {f"{prefix}{k}": v for k, v in values.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/r2_full_frozen.yaml")
    args = ap.parse_args()

    exp = load_yaml(args.experiment)
    root = Path(exp["output_root"])
    numeric_tolerance = float(exp.get("numeric_tolerance", 1e-4))
    rows = []

    for method_dir in sorted((root / "outputs").glob("*")):
        if not method_dir.is_dir():
            continue
        for file in sorted(method_dir.glob("*.jsonl")):
            for rec in load_jsonl(file):
                packet = rec["packet"]
                final_text = rec["text"]
                native_text = rec.get("native_text", final_text)

                row = {
                    "audit_rule_version": AUDIT_RULE_VERSION,
                    "dataset": rec["dataset"],
                    "state": rec["state"],
                    "method": rec["method"],
                    "window_id": packet["traceability"].get("window_id"),
                    "edge_id": packet["traceability"].get("edge_id"),
                    "packet_hash": rec.get("packet_hash"),
                    "prompt_hash": rec.get("prompt_hash"),
                    "native_output_hash": rec.get("native_output_hash", rec.get("output_hash")),
                    "output_hash": rec.get("output_hash"),
                    "output_length_chars": len(final_text),
                    "native_output_length_chars": len(native_text),
                    "sentence_count": max(1, final_text.count(".") + final_text.count("!") + final_text.count("?")),
                    "native_sentence_count": max(1, native_text.count(".") + native_text.count("!") + native_text.count("?")),
                    "generated_tokens": rec.get("generated_tokens"),
                    "terminated_by_eos": rec.get("terminated_by_eos"),
                    "truncated_at_max_tokens": bool(rec.get("truncated_at_max_tokens", False)),
                    "audit_pass_generation": bool(rec.get("audit_pass", True)),
                    "audit_failures_generation": ";".join(rec.get("audit_failures", [])),
                    "fallback_used": bool(rec.get("fallback_used", False)),
                    "fallback_reason": rec.get("fallback_reason"),
                }

                row.update(protected_field_metrics(packet, final_text, numeric_tolerance=numeric_tolerance))
                row.update(unsupported_claim_flags(final_text))

                row.update(prefixed(
                    "native_",
                    protected_field_metrics(packet, native_text, numeric_tolerance=numeric_tolerance),
                ))
                row.update(prefixed("native_", unsupported_claim_flags(native_text)))
                rows.append(row)

    df = pd.DataFrame(rows)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    df.to_csv(results / "record_metrics_audit_v2.csv", index=False)

    summary = (
        df.groupby(["dataset", "state", "method"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(results / "summary_metrics_audit_v2.csv", index=False)
    print(f"Audit rule version: {AUDIT_RULE_VERSION}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
