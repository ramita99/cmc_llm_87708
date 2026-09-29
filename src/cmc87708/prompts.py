from __future__ import annotations

import json
from typing import Any

from .protocol import assert_llm_facing_packet

PROTECTED_RULES = """Use only the supplied evidence. Preserve identifiers and numeric detector fields exactly. Do not infer attack category, intent, hidden causes, or remediation that is not present in the packet. If evidence is insufficient, say so explicitly."""


def proposed_prompt(packet: dict[str, Any]) -> str:
    assert_llm_facing_packet(packet, "proposed prompt packet")
    return (
        "You are generating a concise analyst-facing NetFlow triage note.\n"
        + PROTECTED_RULES
        + "\n\nAUTHORIZED EVIDENCE PACKET:\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
    )


def basic_prompt(packet: dict[str, Any]) -> str:
    assert_llm_facing_packet(packet, "basic prompt packet")
    return (
        "Using the following NetFlow, graph-context, and detector information, "
        "write a concise triage note for a security analyst.\n\n"
        + json.dumps(packet, ensure_ascii=False, indent=2)
    )


def deterministic_template(packet: dict[str, Any]) -> str:
    assert_llm_facing_packet(packet, "template packet")
    t = packet.get("target", {})
    parts = [
        f"Target edge {t.get('edge_id')} in window {t.get('window_id')}: "
        f"{t.get('src_ip', t.get('src_endpoint_alias', t.get('src_node_id')))} -> "
        f"{t.get('dst_ip', t.get('dst_endpoint_alias', t.get('dst_node_id')))}."
    ]
    for name, key in [("Flow evidence", "flow_evidence"), ("Graph context", "graph_evidence"), ("Detector state", "detector_state")]:
        values = packet.get(key, {})
        if values:
            parts.append(name + ": " + ", ".join(f"{k}={v}" for k, v in values.items()) + ".")
    if not (packet.get("flow_evidence") or packet.get("graph_evidence") or packet.get("detector_state")):
        parts.append("No flow, graph, or detector-state evidence is exposed in this condition.")
    return " ".join(parts)
