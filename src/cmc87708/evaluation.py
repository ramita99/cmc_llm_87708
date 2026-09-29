from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROTOCOL_ALIASES = {
    "1": {"1", "icmp"},
    "6": {"6", "tcp"},
    "17": {"17", "udp"},
    "58": {"58", "icmpv6", "icmp6"},
}

ATTACK_TERMS = (
    r"ddos", r"dos attack", r"ransomware", r"botnet", r"port[ -]?scan(?:ning)?",
    r"brute[ -]?force", r"malware", r"phishing", r"sql injection", r"exploit(?:ation)?",
    r"command[- ]and[- ]control", r"c2\b", r"credential stuffing", r"lateral movement",
)


def _contains_token(text: str, value: Any) -> bool:
    if value is None:
        return True
    token = str(value)
    pattern = rf"(?<![A-Za-z0-9_-]){re.escape(token)}(?![A-Za-z0-9_-])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _numbers_close(observed: float, expected: float, tolerance: float) -> bool:
    return math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance)


def _label_value_pattern(label_group: str, value_pattern: str) -> str:
    return rf"(?:{label_group})\*{{0,2}}\s*(?:(?:[:=])\*{{0,2}}|of\b)?\s*{value_pattern}"


def _field_numeric_match(text: str, labels: tuple[str, ...], value: Any, tolerance: float) -> bool:
    try:
        expected = float(value)
    except (TypeError, ValueError):
        return False
    label_group = "|".join(re.escape(label) for label in labels)
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    pattern = _label_value_pattern(label_group, number)
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        try:
            if _numbers_close(float(match.group(1)), expected, tolerance):
                return True
        except ValueError:
            continue
    return False


def _prediction_match(text: str, value: Any) -> bool:
    token = re.escape(str(value))
    label_group = r"prediction|predicted\s+label"
    pattern = _label_value_pattern(label_group, rf"({token})(?!\d)")
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _canonical_protocol(value: Any) -> set[str]:
    raw = str(value).strip().lower()
    if raw.endswith(".0"):
        raw = raw[:-2]
    for key, aliases in PROTOCOL_ALIASES.items():
        if raw == key or raw in aliases:
            return aliases
    return {raw}


def _protocol_match(text: str, value: Any) -> bool:
    aliases = sorted(_canonical_protocol(value), key=len, reverse=True)
    alt = "|".join(re.escape(a) for a in aliases)
    pattern = _label_value_pattern(r"protocol", rf"(?:{alt})(?![A-Za-z0-9])")
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def protected_field_metrics(packet: dict[str, Any], text: str, numeric_tolerance: float = 1e-4) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    target = packet.get("target", {})
    detector = packet.get("detector_state", {})

    for field in ["src_ip", "dst_ip", "src_port", "dst_port"]:
        if field in target:
            metrics[f"target_{field}_ok"] = int(_contains_token(text, target[field]))
    if "protocol" in target:
        metrics["target_protocol_ok"] = int(_protocol_match(text, target["protocol"]))

    if "prediction" in detector:
        metrics["detector_prediction_ok"] = int(_prediction_match(text, detector["prediction"]))
    if "score" in detector:
        metrics["detector_score_ok"] = int(
            _field_numeric_match(text, ("detector score", "score"), detector["score"], numeric_tolerance)
        )
    if "threshold" in detector:
        metrics["detector_threshold_ok"] = int(
            _field_numeric_match(text, ("threshold",), detector["threshold"], numeric_tolerance)
        )
    if "score_margin" in detector:
        metrics["detector_score_margin_ok"] = int(
            _field_numeric_match(text, ("score margin", "score_margin", "margin"), detector["score_margin"], numeric_tolerance)
        )

    eligible = [v for k, v in metrics.items() if k.endswith("_ok")]
    metrics["required_field_completeness"] = float(np.mean(eligible)) if eligible else np.nan
    return metrics


def protected_detector_audit(packet: dict[str, Any], text: str, numeric_tolerance: float = 1e-4) -> dict[str, Any]:
    detector = packet.get("detector_state", {})
    if not detector:
        return {"pass": True, "failures": [], "metrics": {}}
    metrics = protected_field_metrics(packet, text, numeric_tolerance=numeric_tolerance)
    names = {
        "prediction": "detector_prediction_ok",
        "score": "detector_score_ok",
        "threshold": "detector_threshold_ok",
        "score_margin": "detector_score_margin_ok",
    }
    failures = [field for field, metric in names.items() if field in detector and metrics.get(metric) != 1]
    return {"pass": not failures, "failures": failures, "metrics": metrics}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if s.strip()]


def _is_negated_or_investigative(sentence: str, term: str) -> bool:
    lower = sentence.lower()
    safe_patterns = (
        r"insufficient (?:evidence|information).{0,60}" + term,
        r"cannot (?:determine|confirm|infer).{0,60}" + term,
        r"do not (?:infer|conclude|determine).{0,60}" + term,
        r"(?:investigate|check|review|look|search|test|rule out|screen|monitor)\w* .{0,50}(?:for|whether).{0,30}" + term,
    )
    return any(re.search(p, lower) for p in safe_patterns)


def unsupported_claim_flags(text: str) -> dict[str, int]:
    """Conservative separated diagnostics for unsupported free-text claims."""
    sentences = _sentences(text)
    attack_assertion = 0
    causal_attribution = 0
    unsupported_intent = 0
    unauthorized_remediation = 0

    attack_regex = re.compile("|".join(f"(?:{t})" for t in ATTACK_TERMS), re.IGNORECASE)
    assertion_markers = re.compile(
        r"\b(?:potential|possible|likely|suspected|appears?|indicates?|suggests?|consistent with|"
        r"evidence of|signs? of|classified as|identified as|is|are|was|were)\b",
        re.IGNORECASE,
    )
    graph_terms = re.compile(
        r"\b(?:in[- ]?degree|out[- ]?degree|graph context|graph evidence|repeated edge|"
        r"unique neighbors?|central|high[- ]traffic|popular target)\b",
        re.IGNORECASE,
    )
    causal_markers = re.compile(
        r"\b(?:because|due to|therefore|thus|hence|suggests?|indicates?|causes?|explains?|"
        r"raises? suspicion|supports? the|strengthens? (?:the )?suspicion|could indicate|may indicate)\b",
        re.IGNORECASE,
    )
    security_conclusion = re.compile(
        r"\b(?:malicious|suspicious|anomalous|attack|threat|compromise|intrusion|detector|flagged|classification)\b",
        re.IGNORECASE,
    )
    intent_patterns = re.compile(
        r"\b(?:attacker|adversary|threat actor|source|host|endpoint)\b.{0,60}"
        r"\b(?:intends?|attempts?|attempting|tries?|trying|aims?|seeks?|wants?)\b",
        re.IGNORECASE,
    )
    remediation_patterns = re.compile(
        r"\b(?:block(?:ing)?|isolate|isolating|quarantine|reset (?:the )?credentials?|"
        r"disable (?:the )?(?:account|host|service)|drop (?:the )?(?:traffic|connection)|"
        r"denylist|blacklist|revoke (?:the )?(?:token|credential|access)|terminate (?:the )?connection)\b",
        re.IGNORECASE,
    )
    remediation_negation = re.compile(
        r"\b(?:do not|not recommended to|avoid)\b.{0,30}\b(?:block|isolate|quarantine|reset|disable|drop|revoke|terminate)\b",
        re.IGNORECASE,
    )

    for sentence in sentences:
        lower = sentence.lower()
        for m in attack_regex.finditer(sentence):
            term = re.escape(m.group(0).lower())
            if _is_negated_or_investigative(sentence, term):
                continue
            if assertion_markers.search(sentence) or "attempt" in lower or sentence.startswith("#"):
                attack_assertion = 1
                break

        # Reported graph statistics are allowed. Flag only when structural evidence is used
        # to support a security conclusion or detector attribution that the packet does not provide.
        if graph_terms.search(sentence) and causal_markers.search(sentence) and security_conclusion.search(sentence):
            causal_attribution = 1

        if intent_patterns.search(sentence):
            unsupported_intent = 1

        if remediation_patterns.search(sentence) and not remediation_negation.search(sentence):
            unauthorized_remediation = 1

    return {
        "unsupported_attack_category": attack_assertion,
        "unsupported_causal_attribution": causal_attribution,
        "unsupported_detector_attribution": causal_attribution,
        "unsupported_intent": unsupported_intent,
        "unauthorized_remediation": unauthorized_remediation,
    }


def paired_risk_difference(a: pd.Series, b: pd.Series, reps: int = 10000, seed: int = 7) -> dict[str, float]:
    x = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return {"risk_difference": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    diff = x - y
    rng = np.random.default_rng(seed)
    boot = np.empty(reps, dtype=float)
    n = len(diff)
    for i in range(reps):
        boot[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"risk_difference": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def mcnemar_exact(a: pd.Series, b: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = ~(x.isna() | y.isna())
    x = x[mask].astype(int)
    y = y[mask].astype(int)
    b01 = int(((x == 0) & (y == 1)).sum())
    b10 = int(((x == 1) & (y == 0)).sum())
    discordant = b01 + b10
    p = 1.0 if discordant == 0 else binomtest(min(b01, b10), discordant, 0.5, alternative="two-sided").pvalue
    return {"b01": b01, "b10": b10, "p_value": float(p)}
