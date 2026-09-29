from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from cmc87708.alignscore_compat import (
    ALIGNSCORE_BACKBONE,
    ALIGNSCORE_MODE,
    AlignScoreLargeNLI,
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def contextual_to_text(context: dict) -> str:
    lines = []
    flow = context.get("target_flow_metadata", {})
    labels = {
        "src_ip": "Source IP",
        "dst_ip": "Destination IP",
        "src_port": "Source port",
        "dst_port": "Destination port",
        "protocol": "Protocol",
    }
    for key, label in labels.items():
        if key in flow:
            lines.append(f"{label} is {fmt(flow[key])}.")

    graph = context.get("graph_local_context", {})
    graph_labels = {
        "src_in_degree": "Source in-degree",
        "src_out_degree": "Source out-degree",
        "dst_in_degree": "Destination in-degree",
        "dst_out_degree": "Destination out-degree",
        "repeated_edge_count": "Repeated edge count",
        "src_unique_neighbors": "Source unique-neighbor count",
        "dst_unique_neighbors": "Destination unique-neighbor count",
    }
    for key, label in graph_labels.items():
        if key in graph:
            lines.append(f"{label} is {fmt(graph[key])}.")
    return " ".join(lines)


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def paired_continuous(a: pd.Series, b: pd.Series, reps: int, seed: int) -> dict:
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"mean_difference": np.nan, "median_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "wilcoxon_p": np.nan, "n": 0}
    diff = x - y
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(reps)])
    try:
        p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        p = 1.0
    return {
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "ci_low": float(np.quantile(boot, .025)),
        "ci_high": float(np.quantile(boot, .975)),
        "wilcoxon_p": p,
        "n": int(len(diff)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc_87708_contextual_final.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg["output_root"])
    reps = int(cfg["bootstrap_replicates"])
    seed = int(cfg["sample_seed"])
    rows = []

    for model_key in cfg["models"]:
        for scope in cfg["scopes"]:
            for policy in cfg["policies"]:
                for dataset in cfg["expected_datasets"]:
                    path = root / model_key / "outputs" / scope / policy / f"{dataset}.jsonl"
                    for rec in load_jsonl(path):
                        context_text = contextual_to_text(rec["context"])
                        rows.append({
                            "model_key": model_key,
                            "dataset": dataset,
                            "scope": scope,
                            "policy": policy,
                            "window_id": rec["window_id"],
                            "edge_id": rec["edge_id"],
                            "context_hash": rec["context_hash"],
                            "context_text": context_text,
                            "text": rec["text"],
                            "text_hash": hash_text(rec["text"]),
                        })

    df = pd.DataFrame(rows)
    unique = df[["context_hash", "text_hash", "context_text", "text"]].drop_duplicates(["context_hash", "text_hash"]).reset_index(drop=True)

    scorer = AlignScoreLargeNLI(
        ckpt_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        backbone=ALIGNSCORE_BACKBONE,
    )
    unique["alignscore"] = scorer.score(unique["context_text"].tolist(), unique["text"].tolist())
    score_map = {(r.context_hash, r.text_hash): float(r.alignscore) for r in unique.itertuples(index=False)}
    df["alignscore"] = [score_map[(c, t)] for c, t in zip(df["context_hash"], df["text_hash"])]

    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = df.drop(columns=["context_text", "text"])
    compact["evaluator"] = "AlignScore-large"
    compact["evaluation_mode"] = ALIGNSCORE_MODE
    compact.to_csv(out_dir / "contextual_alignscore_records.csv", index=False)

    summary = (
        compact.groupby(["model_key", "dataset", "scope", "policy"], dropna=False)
        .agg(n=("alignscore", "size"), mean_alignscore=("alignscore", "mean"), median_alignscore=("alignscore", "median"), std_alignscore=("alignscore", "std"))
        .reset_index()
    )
    summary.to_csv(out_dir / "contextual_alignscore_summary.csv", index=False)

    stats_rows = []
    for (model_key, dataset, scope), g in compact.groupby(["model_key", "dataset", "scope"]):
        basic = g[g.policy == "basic"].set_index(["window_id", "edge_id"])["alignscore"]
        bounded = g[g.policy == "bounded"].set_index(["window_id", "edge_id"])["alignscore"]
        common = basic.index.intersection(bounded.index)
        stats_rows.append({
            "model_key": model_key,
            "dataset": dataset,
            "scope": scope,
            **paired_continuous(bounded.loc[common], basic.loc[common], reps, seed),
        })

    stats = pd.DataFrame(stats_rows)
    stats["p_holm"] = np.nan
    for model_key, idx in stats.groupby("model_key").groups.items():
        stats.loc[idx, "p_holm"] = multipletests(stats.loc[idx, "wilcoxon_p"].astype(float), method="holm")[1]
    stats.to_csv(out_dir / "contextual_alignscore_paired_statistics.csv", index=False)

    print(summary.to_string(index=False))
    print("\n[PAIRED ALIGNSCORE]\n", stats.to_string(index=False))


if __name__ == "__main__":
    main()
