from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cmc87708.alignscore_compat import (
    ALIGNSCORE_BACKBONE,
    ALIGNSCORE_MODE,
    AlignScoreLargeNLI,
    context_hash,
    packet_to_alignscore_context,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


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
    boot = np.empty(reps, dtype=float)
    n = len(diff)
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


def collect_records(root: Path) -> list[dict]:
    rows: list[dict] = []
    for method in ["basic_prompt", "proposed_source_gated"]:
        method_dir = root / "outputs" / method
        if not method_dir.exists():
            raise FileNotFoundError(method_dir)

        for path in sorted(method_dir.glob("*.jsonl")):
            for rec in load_jsonl(path):
                if rec.get("state") != "detector":
                    continue

                packet = rec["packet"]
                context = packet_to_alignscore_context(packet)
                native_text = rec.get("native_text", rec["text"])
                final_text = rec["text"]

                base = {
                    "dataset": rec["dataset"],
                    "state": rec["state"],
                    "method": rec["method"],
                    "window_id": packet["traceability"].get("window_id"),
                    "edge_id": packet["traceability"].get("edge_id"),
                    "packet_hash": rec.get("packet_hash"),
                    "context_hash": context_hash(context),
                    "truncated_at_max_tokens": bool(rec.get("truncated_at_max_tokens", False)),
                    "fallback_used": bool(rec.get("fallback_used", False)),
                }
                rows.append({
                    **base,
                    "view": "native",
                    "context": context,
                    "text": native_text,
                    "text_hash": sha256_text(native_text),
                })
                if method == "proposed_source_gated":
                    rows.append({
                        **base,
                        "view": "final",
                        "context": context,
                        "text": final_text,
                        "text_hash": sha256_text(final_text),
                    })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bootstrap-replicates", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    rows = collect_records(root)
    df = pd.DataFrame(rows)

    # Score each unique context/text pair once. This avoids duplicate computation
    # when Proposed native output passed the audit and is identical to final output.
    unique = (
        df[["context_hash", "text_hash", "context", "text"]]
        .drop_duplicates(["context_hash", "text_hash"])
        .reset_index(drop=True)
    )

    scorer = AlignScoreLargeNLI(
        ckpt_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        backbone=ALIGNSCORE_BACKBONE,
    )

    scores = scorer.score(unique["context"].tolist(), unique["text"].tolist())
    unique["alignscore"] = scores

    score_map = {
        (r.context_hash, r.text_hash): float(r.alignscore)
        for r in unique.itertuples(index=False)
    }
    df["alignscore"] = [
        score_map[(c, t)]
        for c, t in zip(df["context_hash"], df["text_hash"])
    ]
    df["evaluator"] = "AlignScore-large"
    df["evaluation_mode"] = ALIGNSCORE_MODE
    df["run_label"] = args.label

    # Do not expose packet/context or generated text in the compact numeric result.
    compact_cols = [
        "run_label", "dataset", "state", "method", "view",
        "window_id", "edge_id", "packet_hash", "context_hash", "text_hash",
        "alignscore", "truncated_at_max_tokens", "fallback_used",
        "evaluator", "evaluation_mode",
    ]
    compact = df[compact_cols].copy()
    compact.to_csv(results / f"alignscore_records_{args.label}.csv", index=False)

    summary = (
        compact.groupby(["run_label", "dataset", "method", "view"], dropna=False)
        .agg(
            n=("alignscore", "size"),
            mean_alignscore=("alignscore", "mean"),
            median_alignscore=("alignscore", "median"),
            std_alignscore=("alignscore", "std"),
            truncation_rate=("truncated_at_max_tokens", "mean"),
            fallback_rate=("fallback_used", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(results / f"alignscore_summary_{args.label}.csv", index=False)

    # Paired comparisons are intentionally separated:
    # (A) prompt effect: Proposed native vs Basic native
    # (B) workflow effect: Proposed final vs Basic native
    key = ["dataset", "window_id", "edge_id"]

    # Gate C for this evaluator: Basic and Proposed must refer to the exact same
    # detector-state packet for each paired edge.
    native_only = compact[compact["view"] == "native"]
    for dataset in sorted(native_only["dataset"].unique()):
        g = native_only[native_only["dataset"] == dataset]
        b_hash = g[g["method"] == "basic_prompt"].set_index(key)["packet_hash"]
        p_hash = g[g["method"] == "proposed_source_gated"].set_index(key)["packet_hash"]
        common_hash = b_hash.index.intersection(p_hash.index)
        if len(common_hash) != len(b_hash) or len(common_hash) != len(p_hash):
            raise AssertionError(f"{dataset}: incomplete Basic/Proposed AlignScore pairing")
        mismatches = int((b_hash.loc[common_hash].astype(str) != p_hash.loc[common_hash].astype(str)).sum())
        if mismatches:
            raise AssertionError(f"{dataset}: {mismatches} packet-hash mismatches")

    stats_rows: list[dict] = []
    for dataset in sorted(compact["dataset"].unique()):
        g = compact[compact["dataset"] == dataset]
        basic = g[
            (g["method"] == "basic_prompt") & (g["view"] == "native")
        ].set_index(key)["alignscore"]
        p_native = g[
            (g["method"] == "proposed_source_gated") & (g["view"] == "native")
        ].set_index(key)["alignscore"]
        p_final = g[
            (g["method"] == "proposed_source_gated") & (g["view"] == "final")
        ].set_index(key)["alignscore"]

        for comparison, proposed in [
            ("proposed_native_vs_basic_native", p_native),
            ("proposed_final_vs_basic_native", p_final),
        ]:
            common = basic.index.intersection(proposed.index)
            if len(common) == 0:
                continue
            stats = paired_continuous(
                proposed.loc[common],
                basic.loc[common],
                reps=args.bootstrap_replicates,
                seed=args.seed,
            )
            stats_rows.append({
                "run_label": args.label,
                "dataset": dataset,
                "comparison": comparison,
                "metric": "alignscore",
                **stats,
            })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(results / f"alignscore_paired_statistics_{args.label}.csv", index=False)

    print("\n[ALIGNSCORE SUMMARY]")
    print(summary.to_string(index=False))
    print("\n[PAIRED ALIGNSCORE STATISTICS]")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
