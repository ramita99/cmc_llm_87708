from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

ATTACK_TERMS = (
    r"ddos", r"dos attack", r"ransomware", r"botnet", r"port[ -]?scan(?:ning)?",
    r"brute[ -]?force", r"malware", r"phishing", r"sql injection", r"exploit(?:ation)?",
    r"command[- ]and[- ]control", r"c2\b", r"credential stuffing", r"lateral movement",
)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", text or "") if s.strip()]


def attack_context_is_safe(sentence: str, matched_term: str) -> bool:
    lower = sentence.lower()
    term = re.escape(matched_term.lower())
    safe = (
        rf"(?:insufficient|not enough)\s+(?:evidence|information).{{0,80}}(?:to\s+)?(?:infer|determine|confirm|establish|support|attribute|classif\w*|conclude).{{0,40}}{term}",
        rf"(?:cannot|can't|can not|do not|don't|does not|should not|must not|unable to).{{0,70}}(?:infer|determine|confirm|establish|attribute|classif\w*|conclude).{{0,40}}{term}",
        rf"(?:investigat\w*|check\w*|review\w*|rule out|monitor\w*).{{0,80}}(?:for|whether|signs? of|evidence of).{{0,40}}{term}",
        rf"(?:no|insufficient)\s+(?:evidence|information).{{0,60}}(?:of|for|supporting).{{0,40}}{term}",
    )
    return any(re.search(p, lower) for p in safe)


def remediation_context_is_safe(sentence: str) -> bool:
    lower = sentence.lower()
    action = r"(?:block\w*|isolat\w*|quarantin\w*|reset\w*|disabl\w*|drop\w*|denylist\w*|blacklist\w*|revok\w*|terminat\w*)"
    safe = (
        rf"\b(?:do not|don't|does not|should not|must not|avoid|refrain from|not authorized to)\b.{{0,50}}\b{action}\b",
        r"\bno remediation action\b.{0,50}\b(?:authorized|recommended|supported)\b",
    )
    return any(re.search(p, lower) for p in safe)


def unsupported_security_components(text: str) -> dict[str, int]:
    attack = causal = intent = remediation = 0
    attack_re = re.compile("|".join(f"(?:{t})" for t in ATTACK_TERMS), re.I)
    assertion = re.compile(r"\b(?:potential|possible|likely|suspected|appears?|indicates?|suggests?|consistent with|evidence of|signs? of|classified as|identified as|is|are|was|were)\b", re.I)
    graph_terms = re.compile(r"\b(?:in[- ]?degree|out[- ]?degree|graph context|graph evidence|repeated edge|unique neighbors?|central|high[- ]traffic|popular target)\b", re.I)
    causal_markers = re.compile(r"\b(?:because|due to|therefore|thus|hence|suggests?|indicates?|causes?|explains?|raises? suspicion|supports? the|could indicate|may indicate)\b", re.I)
    security_conclusion = re.compile(r"\b(?:malicious|suspicious|anomalous|attack|threat|compromise|intrusion|detector|flagged|classification)\b", re.I)
    intent_re = re.compile(r"\b(?:attacker|adversary|threat actor|source|host|endpoint)\b.{0,60}\b(?:intends?|attempts?|attempting|tries?|trying|aims?|seeks?|wants?)\b", re.I)
    remediation_re = re.compile(r"\b(?:block\w*|isolat\w*|quarantin\w*|reset\w*|disabl\w*|drop\w*|denylist\w*|blacklist\w*|revok\w*|terminat\w*)\b", re.I)

    for s in sentences(text):
        for m in attack_re.finditer(s):
            if not attack_context_is_safe(s, m.group(0)) and (assertion.search(s) or "attempt" in s.lower()):
                attack = 1
                break
        if graph_terms.search(s) and causal_markers.search(s) and security_conclusion.search(s):
            causal = 1
        if intent_re.search(s):
            intent = 1
        if remediation_re.search(s) and not remediation_context_is_safe(s):
            remediation = 1

    return {
        "unsupported_attack_type": attack,
        "unsupported_causal_security": causal,
        "unsupported_intent": intent,
        "unsupported_remediation": remediation,
        "unsupported_security_extrapolation": int(any([attack, causal, intent, remediation])),
    }


def _canon_protocol(value) -> set[str]:
    raw = str(value).strip().lower()
    if raw.endswith(".0"):
        raw = raw[:-2]
    aliases = {
        "1": {"1", "icmp"},
        "6": {"6", "tcp"},
        "17": {"17", "udp"},
        "58": {"58", "icmpv6", "icmp6"},
    }
    for key, vals in aliases.items():
        if raw == key or raw in vals:
            return vals
    return {raw}


def _first_group(pattern: str, text: str):
    m = re.search(pattern, text, flags=re.I)
    return None if m is None else m.group(1)


def context_fact_fidelity(context: dict, text: str) -> dict[str, float]:
    """Conservative contradiction checks for facts that are explicitly verbalized.

    Omission is allowed for concise contextual notes. A contradiction is recorded only
    when a recognizable source/destination IP, role-specific port, or protocol statement
    is present and disagrees with the supplied context.
    """
    text = text or ""
    flow = context.get("target_flow_metadata", {})
    contradiction = 0

    # Explicit role-labelled forms.
    labelled = {
        "src_ip": r"(?:source|src)\s*(?:ip|address)?\s*(?:is|=|:)?\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
        "dst_ip": r"(?:destination|dst)\s*(?:ip|address)?\s*(?:is|=|:)?\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
        "src_port": r"(?:source|src)\s*port\s*(?:is|=|:)?\s*(\d+)",
        "dst_port": r"(?:destination|dst)\s*port\s*(?:is|=|:)?\s*(\d+)",
    }
    for field, pattern in labelled.items():
        if field in flow:
            observed = _first_group(pattern, text)
            if observed is not None and str(observed) != str(flow[field]):
                contradiction = 1

    # Common compact narrative form: "between SRC and DST on ports SPORT->DPORT".
    pair = re.search(
        r"between\s+([0-9]{1,3}(?:\.[0-9]{1,3}){3})\s+and\s+"
        r"([0-9]{1,3}(?:\.[0-9]{1,3}){3}).{0,80}?"
        r"ports?\s+(\d+)\s*(?:->|to)\s*(\d+)",
        text,
        flags=re.I,
    )
    if pair:
        expected = [flow.get("src_ip"), flow.get("dst_ip"), flow.get("src_port"), flow.get("dst_port")]
        observed = list(pair.groups())
        for obs, exp in zip(observed, expected):
            if exp is not None and str(obs) != str(exp):
                contradiction = 1

    if "protocol" in flow:
        protocol_observed = _first_group(
            r"protocol\s*(?:is|=|:)?\s*([A-Za-z0-9]+)",
            text,
        )
        if protocol_observed is None:
            protocol_observed = _first_group(
                r"\b(TCP|UDP|ICMPv?6?|\d+)\s+(?:connection|flow|communication|traffic)\b",
                text,
            )
        if protocol_observed is not None:
            if protocol_observed.lower() not in _canon_protocol(flow["protocol"]):
                contradiction = 1

    graph = context.get("graph_local_context", {})
    graph_labels = {
        "src_in_degree": r"(?:source|src)\s+in[- ]?degree",
        "src_out_degree": r"(?:source|src)\s+out[- ]?degree",
        "dst_in_degree": r"(?:destination|dst)\s+in[- ]?degree",
        "dst_out_degree": r"(?:destination|dst)\s+out[- ]?degree",
        "repeated_edge_count": r"repeated[- _]?edge(?:\s+count)?",
        "src_unique_neighbors": r"(?:source|src)\s+unique[- ]?neighbor(?:\s+count)?",
        "dst_unique_neighbors": r"(?:destination|dst)\s+unique[- ]?neighbor(?:\s+count)?",
    }
    graph_value_use = 0
    for field, value in graph.items():
        label = graph_labels.get(field)
        if label and re.search(label + r".{0,30}?" + re.escape(str(value)), text, flags=re.I):
            graph_value_use = 1
            break

    return {
        "contextual_flow_fact_contradiction": float(contradiction),
        "explicit_graph_value_use": float(graph_value_use) if graph else np.nan,
    }


def paired_binary(a: pd.Series, b: pd.Series, reps: int, seed: int) -> dict:
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    diff = x - y
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(reps)])
    b01 = int(((x == 0) & (y == 1)).sum())
    b10 = int(((x == 1) & (y == 0)).sum())
    discordant = b01 + b10
    p = 1.0 if discordant == 0 else float(binomtest(min(b01, b10), discordant, 0.5).pvalue)
    return {
        "risk_difference": float(diff.mean()),
        "ci_low": float(np.quantile(boot, .025)),
        "ci_high": float(np.quantile(boot, .975)),
        "mcnemar_p": p,
        "n": len(diff),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cmc_87708_contextual_final.yaml")
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
                    records = load_jsonl(path)
                    if len(records) != int(cfg["targets_per_dataset"]):
                        raise AssertionError(f"{path}: expected {cfg['targets_per_dataset']}, got {len(records)}")
                    for r in records:
                        metrics = unsupported_security_components(r["text"])
                        metrics.update(context_fact_fidelity(r["context"], r["text"]))
                        rows.append({
                            "model_key": model_key,
                            "dataset": dataset,
                            "scope": scope,
                            "policy": policy,
                            "window_id": r["window_id"],
                            "edge_id": r["edge_id"],
                            "context_hash": r["context_hash"],
                            "output_hash": r["output_hash"],
                            "truncated_at_max_tokens": bool(r["truncated_at_max_tokens"]),
                            **metrics,
                        })

    df = pd.DataFrame(rows)
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(result_dir / "contextual_record_metrics.csv", index=False)

    # Pairing gate: Basic and Bounded must use byte-identical context within model/dataset/scope/target.
    key = ["model_key", "dataset", "scope", "window_id", "edge_id"]
    basic = df[df.policy == "basic"].set_index(key)
    bounded = df[df.policy == "bounded"].set_index(key)
    common = basic.index.intersection(bounded.index)
    if len(common) != len(basic) or len(common) != len(bounded):
        raise AssertionError("Incomplete Basic/Bounded pairing")
    mismatches = (basic.loc[common, "context_hash"].astype(str) != bounded.loc[common, "context_hash"].astype(str)).sum()
    if mismatches:
        raise AssertionError(f"{mismatches} context-hash mismatches")

    stats = []
    for (model_key, dataset, scope), g in df.groupby(["model_key", "dataset", "scope"]):
        b = g[g.policy == "basic"].set_index(["window_id", "edge_id"])
        d = g[g.policy == "bounded"].set_index(["window_id", "edge_id"])
        idx = b.index.intersection(d.index)
        for metric in ["unsupported_security_extrapolation", "contextual_flow_fact_contradiction", "explicit_graph_value_use"]:
            if metric not in b or metric not in d:
                continue
            if b.loc[idx, metric].isna().all() or d.loc[idx, metric].isna().all():
                continue
            st = paired_binary(d.loc[idx, metric], b.loc[idx, metric], reps, seed)
            stats.append({"model_key": model_key, "dataset": dataset, "scope": scope, "metric": metric, **st})

    stats_df = pd.DataFrame(stats)
    if not stats_df.empty:
        stats_df["p_holm"] = np.nan
        for (model_key, metric), idx in stats_df.groupby(["model_key", "metric"]).groups.items():
            vals = stats_df.loc[idx, "mcnemar_p"].astype(float)
            stats_df.loc[idx, "p_holm"] = multipletests(vals, method="holm")[1]
    stats_df.to_csv(result_dir / "contextual_paired_binary_statistics.csv", index=False)

    summary = (
        df.groupby(["model_key", "dataset", "scope", "policy"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(result_dir / "contextual_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\n[PAIRED STATISTICS]\n", stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
