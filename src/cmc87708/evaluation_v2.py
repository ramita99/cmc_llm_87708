from __future__ import annotations

import re
from typing import Any

from cmc87708.evaluation import (
    mcnemar_exact,
    paired_risk_difference,
    protected_detector_audit,
    protected_field_metrics,
)

AUDIT_RULE_VERSION = "CMC-R2-AUDIT-V2-2026-09-18"

ATTACK_TERMS = (
    r"ddos", r"dos attack", r"ransomware", r"botnet", r"port[ -]?scan(?:ning)?",
    r"brute[ -]?force", r"malware", r"phishing", r"sql injection", r"exploit(?:ation)?",
    r"command[- ]and[- ]control", r"c2\b", r"credential stuffing", r"lateral movement",
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", text) if s.strip()]


def _attack_context_is_safe(sentence: str, matched_term: str) -> bool:
    """Return True when an attack term is explicitly negated, bounded, or investigative.

    V2 fixes two false-positive classes found by controlled perturbation:
    (1) 'evidence is insufficient to infer <attack>' and related evidence-limit wording;
    (2) investigative wording such as 'review ... for signs of <attack> before concluding'.
    """
    lower = sentence.lower()
    term = re.escape(matched_term.lower())
    safe_patterns = (
        rf"(?:insufficient|not enough)\s+(?:evidence|information).{{0,80}}(?:to\s+)?"
        rf"(?:infer|determine|confirm|establish|support|attribute|classif\w*|conclude).{{0,40}}{term}",

        rf"(?:evidence|information).{{0,20}}(?:is|are|remains?)?\s*(?:insufficient|inadequate).{{0,60}}"
        rf"(?:to\s+)?(?:infer|determine|confirm|establish|support|attribute|classif\w*|conclude).{{0,40}}{term}",

        rf"(?:cannot|can't|can not|do not|don't|does not|should not|must not|unable to).{{0,70}}"
        rf"(?:infer|determine|confirm|establish|attribute|classif\w*|conclude).{{0,40}}{term}",

        rf"(?:investigat\w*|check\w*|review\w*|look\w*|search\w*|test\w*|rule out|screen\w*|monitor\w*).{{0,80}}"
        rf"(?:for|whether|signs? of|evidence of).{{0,40}}{term}",

        rf"(?:no|insufficient)\s+(?:evidence|information).{{0,60}}(?:of|for|supporting).{{0,40}}{term}",

        rf"{term}.{{0,50}}(?:cannot|can't|can not|is not|was not|has not been).{{0,40}}"
        rf"(?:established|confirmed|determined|inferred|supported|classified)",
    )
    return any(re.search(pattern, lower) for pattern in safe_patterns)


def _remediation_context_is_safe(sentence: str) -> bool:
    """Return True when remediation wording is explicitly negated or unauthorized.

    V2 uses action stems so forms such as 'isolating' are covered by the same
    negation rule as 'isolate'.
    """
    lower = sentence.lower()
    action = (
        r"(?:block\w*|isolat\w*|quarantin\w*|reset\w*|disabl\w*|drop\w*|"
        r"denylist\w*|blacklist\w*|revok\w*|terminat\w*)"
    )
    safe_patterns = (
        rf"\b(?:do not|don't|does not|should not|must not|avoid|refrain from|"
        rf"not recommended to|not authorized to)\b.{{0,50}}\b{action}\b",

        rf"\b{action}\b.{{0,50}}\b(?:should not|must not|is not|are not|was not|were not)\b.{{0,30}}"
        rf"\b(?:performed|applied|used|taken|recommended|authorized)\b",

        r"\bno remediation action\b.{0,50}\b(?:authorized|recommended|supported)\b",
    )
    return any(re.search(pattern, lower) for pattern in safe_patterns)


def unsupported_claim_flags(text: str) -> dict[str, int]:
    """Conservative unsupported-claim diagnostics, version 2.

    The detector/field fidelity logic is unchanged from the frozen experiment.
    Only free-text unsupported-claim handling is amended to reduce false
    positives for explicit evidence-limit and negated-remediation language.
    """
    sentences = _sentences(text)
    attack_assertion = 0
    causal_attribution = 0
    unsupported_intent = 0
    unauthorized_remediation = 0

    attack_regex = re.compile("|".join(f"(?:{term})" for term in ATTACK_TERMS), re.IGNORECASE)
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
        r"\b(?:block\w*|isolat\w*|quarantin\w*|reset\w*|disabl\w*|drop\w*|"
        r"denylist\w*|blacklist\w*|revok\w*|terminat\w*)\b",
        re.IGNORECASE,
    )

    for sentence in sentences:
        lower = sentence.lower()

        for match in attack_regex.finditer(sentence):
            if _attack_context_is_safe(sentence, match.group(0)):
                continue
            if assertion_markers.search(sentence) or "attempt" in lower or sentence.startswith("#"):
                attack_assertion = 1
                break

        # Plain graph facts are allowed; only graph-to-security causal language is flagged.
        if graph_terms.search(sentence) and causal_markers.search(sentence) and security_conclusion.search(sentence):
            causal_attribution = 1

        if intent_patterns.search(sentence):
            unsupported_intent = 1

        if remediation_patterns.search(sentence) and not _remediation_context_is_safe(sentence):
            unauthorized_remediation = 1

    return {
        "unsupported_attack_category": attack_assertion,
        "unsupported_causal_attribution": causal_attribution,
        # Kept as an alias for backward-compatible tables; do not count as a separate
        # hypothesis in V2 multiple-testing correction.
        "unsupported_detector_attribution": causal_attribution,
        "unsupported_intent": unsupported_intent,
        "unauthorized_remediation": unauthorized_remediation,
    }
