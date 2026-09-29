from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from cmc87708.evaluation import mcnemar_exact, paired_risk_difference


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paired_continuous(a: pd.Series, b: pd.Series, reps: int, seed: int) -> dict[str, float]:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {
            "mean_difference": np.nan,
            "median_difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "wilcoxon_p": np.nan,
            "n": 0,
        }
    diff = x - y
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        boot[i] = np.median(diff[rng.integers(0, n, n)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    try:
        p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return {
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "wilcoxon_p": p,
        "n": int(n),
    }


def apply_holm(df: pd.DataFrame, p_col: str, out_col: str) -> pd.DataFrame:
    df[out_col] = np.nan
    mask = pd.to_numeric(df[p_col], errors="coerce").notna()
    if mask.any():
        df.loc[mask, out_col] = multipletests(df.loc[mask, p_col].astype(float), method="holm")[1]
    return df


def binary_compare(
    proposed: pd.DataFrame,
    basic: pd.DataFrame,
    common,
    metrics: list[str],
    reps: int,
    seed: int,
    label: str,
    dataset: str,
) -> list[dict]:
    rows = []
    for metric in metrics:
        if metric not in proposed.columns or metric not in basic.columns:
            continue
        a = proposed.loc[common, metric]
        b = basic.loc[common, metric]
        rd = paired_risk_difference(a, b, reps=reps, seed=seed)
        mc = mcnemar_exact(a, b)
        rows.append({
            "comparison": label,
            "dataset": dataset,
            "metric": metric,
            **rd,
            **mc,
        })
    return rows


def continuous_compare(
    proposed: pd.DataFrame,
    basic: pd.DataFrame,
    common,
    metrics: list[str],
    reps: int,
    seed: int,
    label: str,
    dataset: str,
) -> list[dict]:
    rows = []
    for metric in metrics:
        if metric not in proposed.columns or metric not in basic.columns:
            continue
        stats = paired_continuous(proposed.loc[common, metric], basic.loc[common, metric], reps, seed)
        rows.append({
            "comparison": label,
            "dataset": dataset,
            "metric": metric,
            **stats,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="configs/r2_full_frozen.yaml")
    args = ap.parse_args()

    exp = load_yaml(args.experiment)
    root = Path(exp["output_root"])
    df = pd.read_csv(root / "results" / "record_metrics_audit_v2.csv")
    full = df[df["state"] == exp["full_state_for_method_comparison"]].copy()
    reps = int(exp["bootstrap_replicates"])
    seed = int(exp["sample_seed"])
    key = ["dataset", "window_id", "edge_id"]

    # Hypothesis families intentionally exclude:
    # - audit_pass_generation: semantics differ between Basic and Proposed.
    # - fallback_used: workflow diagnostic only; Basic never uses fallback.
    # - unsupported_detector_attribution: exact alias of unsupported_causal_attribution.
    native_binary = [
        "native_target_src_ip_ok",
        "native_target_dst_ip_ok",
        "native_target_src_port_ok",
        "native_target_dst_port_ok",
        "native_target_protocol_ok",
        "native_detector_prediction_ok",
        "native_detector_score_ok",
        "native_detector_threshold_ok",
        "native_detector_score_margin_ok",
        "native_unsupported_attack_category",
        "native_unsupported_causal_attribution",
        "native_unsupported_intent",
        "native_unauthorized_remediation",
        "truncated_at_max_tokens",
    ]
    native_continuous = [
        "native_output_length_chars",
        "native_sentence_count",
        "generated_tokens",
        "native_required_field_completeness",
    ]

    workflow_binary = [
        "target_src_ip_ok",
        "target_dst_ip_ok",
        "target_src_port_ok",
        "target_dst_port_ok",
        "target_protocol_ok",
        "detector_prediction_ok",
        "detector_score_ok",
        "detector_threshold_ok",
        "detector_score_margin_ok",
        "unsupported_attack_category",
        "unsupported_causal_attribution",
        "unsupported_intent",
        "unauthorized_remediation",
    ]
    workflow_continuous = [
        "output_length_chars",
        "sentence_count",
        "required_field_completeness",
    ]

    native_b, native_c, workflow_b, workflow_c = [], [], [], []

    for dataset, g in full.groupby("dataset"):
        proposed = g[g["method"] == "proposed_source_gated"].set_index(key)
        basic = g[g["method"] == "basic_prompt"].set_index(key)
        common = proposed.index.intersection(basic.index)
        if len(common) == 0:
            continue

        native_b += binary_compare(
            proposed, basic, common, native_binary, reps, seed,
            "proposed_native_vs_basic_native", dataset,
        )
        native_c += continuous_compare(
            proposed, basic, common, native_continuous, reps, seed,
            "proposed_native_vs_basic_native", dataset,
        )
        workflow_b += binary_compare(
            proposed, basic, common, workflow_binary, reps, seed,
            "proposed_final_vs_basic_native", dataset,
        )
        workflow_c += continuous_compare(
            proposed, basic, common, workflow_continuous, reps, seed,
            "proposed_final_vs_basic_native", dataset,
        )

    results = root / "results"

    nbin = pd.DataFrame(native_b)
    if not nbin.empty:
        nbin = apply_holm(nbin, "p_value", "p_value_holm")
    nbin.to_csv(results / "paired_native_binary_statistics_v2.csv", index=False)

    ncon = pd.DataFrame(native_c)
    if not ncon.empty:
        ncon = apply_holm(ncon, "wilcoxon_p", "wilcoxon_p_holm")
    ncon.to_csv(results / "paired_native_continuous_statistics_v2.csv", index=False)

    wbin = pd.DataFrame(workflow_b)
    if not wbin.empty:
        wbin = apply_holm(wbin, "p_value", "p_value_holm")
    wbin.to_csv(results / "paired_workflow_binary_statistics_v2.csv", index=False)

    wcon = pd.DataFrame(workflow_c)
    if not wcon.empty:
        wcon = apply_holm(wcon, "wilcoxon_p", "wilcoxon_p_holm")
    wcon.to_csv(results / "paired_workflow_continuous_statistics_v2.csv", index=False)

    print("\n[NATIVE VS NATIVE — BINARY]")
    print(nbin.to_string(index=False) if not nbin.empty else "No comparisons.")
    print("\n[NATIVE VS NATIVE — CONTINUOUS]")
    print(ncon.to_string(index=False) if not ncon.empty else "No comparisons.")
    print("\n[FINAL WORKFLOW VS BASIC — BINARY]")
    print(wbin.to_string(index=False) if not wbin.empty else "No comparisons.")
    print("\n[FINAL WORKFLOW VS BASIC — CONTINUOUS]")
    print(wcon.to_string(index=False) if not wcon.empty else "No comparisons.")


if __name__ == "__main__":
    main()
