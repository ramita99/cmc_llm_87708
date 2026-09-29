from __future__ import annotations

import hashlib
import pandas as pd


def allocate_window_quotas(counts: pd.Series, n: int) -> dict[str, int]:
    counts = counts[counts > 0].astype(int)
    raw = counts / counts.sum() * n
    quota = raw.astype(int)
    remainder = n - int(quota.sum())
    for key in (raw - quota).sort_values(ascending=False).index[:remainder]:
        quota.loc[key] += 1
    return {str(k): int(v) for k, v in quota.items()}


def sample_window_stratified_uniform(eligible: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(eligible) < n:
        raise ValueError(f"Only {len(eligible)} eligible rows for requested n={n}")
    if "window_id" not in eligible.columns or "edge_id" not in eligible.columns:
        raise ValueError("eligible must contain window_id and edge_id")
    quotas = allocate_window_quotas(eligible.groupby("window_id").size(), n)
    parts = []
    for window, q in quotas.items():
        group = eligible[eligible["window_id"].astype(str) == window]
        local_seed = int(hashlib.sha256(f"{seed}|{window}".encode()).hexdigest()[:8], 16)
        parts.append(group.sample(n=min(q, len(group)), random_state=local_seed))
    selected = pd.concat(parts, ignore_index=True)
    if len(selected) < n:
        used = set(zip(selected["window_id"].astype(str), selected["edge_id"].astype(str)))
        mask = [(str(w), str(e)) not in used for w, e in zip(eligible["window_id"], eligible["edge_id"])]
        remainder = eligible.loc[mask]
        selected = pd.concat([selected, remainder.sample(n=n-len(selected), random_state=seed)], ignore_index=True)
    selected = selected.sort_values(["window_id", "edge_id"]).head(n).reset_index(drop=True)
    selected["sampling_seed"] = seed
    selected["sampling_stratum"] = selected["window_id"].astype(str)
    selected["selection_order"] = range(1, len(selected)+1)
    return selected
