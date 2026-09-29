from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROTECTED = [
    ("prediction", "native_detector_prediction_ok"),
    ("score", "native_detector_score_ok"),
    ("threshold", "native_detector_threshold_ok"),
    ("score_margin", "native_detector_score_margin_ok"),
]


def export_table8(model_label: str, root: Path) -> pd.DataFrame:
    path = root / "results" / "record_metrics_audit_v2.csv"
    df = pd.read_csv(path)
    df = df[df["state"].eq("detector") & df["method"].isin(["basic_prompt", "proposed_source_gated"])].copy()
    rows = []
    for (dataset, method), g in df.groupby(["dataset", "method"]):
        row = {
            "model": model_label,
            "dataset": dataset,
            "method": "Basic" if method == "basic_prompt" else "Source-gated native",
            "n": len(g),
        }
        vals = []
        for paper_name, col in PROTECTED:
            if col not in g:
                raise KeyError(f"Missing {col} in {path}")
            value = float(pd.to_numeric(g[col], errors="coerce").mean())
            row[f"{paper_name}_rate"] = value
            vals.append(value)
        row["mean_protected_detector_field_rate"] = sum(vals) / len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def export_table11(contextual_root: Path) -> pd.DataFrame:
    path = contextual_root / "results" / "contextual_record_metrics.csv"
    df = pd.read_csv(path)
    rows = []
    for (model_key, dataset, scope), g in df.groupby(["model_key", "dataset", "scope"]):
        row = {"model": model_key, "dataset": dataset, "scope": scope}
        for policy in ["basic", "bounded"]:
            x = g[g["policy"].eq(policy)]
            row[f"{policy}_rate"] = float(pd.to_numeric(x["unsupported_security_extrapolation"], errors="coerce").mean())
            row[f"{policy}_n"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-root", required=True)
    ap.add_argument("--qwen-root", required=True)
    ap.add_argument("--contextual-root", required=True)
    ap.add_argument("--output", default="paper_exports")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    table8 = pd.concat([
        export_table8("Gemma-2-9B-IT", Path(args.gemma_root)),
        export_table8("Qwen2.5-7B-Instruct", Path(args.qwen_root)),
    ], ignore_index=True)
    table8.to_csv(out / "table8_protected_detector_state.csv", index=False)

    table11 = export_table11(Path(args.contextual_root))
    table11.to_csv(out / "table11_unsupported_security_extrapolation.csv", index=False)

    # AlignScore scripts already write manuscript-ready summaries/statistics.
    print("Exported:")
    print(out / "table8_protected_detector_state.csv")
    print(out / "table11_unsupported_security_extrapolation.csv")


if __name__ == "__main__":
    main()
