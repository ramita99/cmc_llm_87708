from cmc87708.evaluation import protected_detector_audit, protected_field_metrics
from cmc87708.evaluation_v2 import unsupported_claim_flags
from cmc87708.prompts import deterministic_template


def sample_packet(protocol=6):
    return {
        "dataset": "demo",
        "state": "detector",
        "target": {
            "dataset": "demo",
            "split": "test",
            "window_id": "win_1",
            "edge_id": "e_1",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": protocol,
        },
        "flow_evidence": {
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": 12345,
            "dst_port": 80,
            "protocol": protocol,
        },
        "graph_evidence": {},
        "detector_state": {
            "model_name": "demo-model",
            "prediction": 1,
            "score": 0.8123,
            "threshold": 0.5,
            "score_margin": 0.3123,
        },
        "traceability": {"split": "test", "window_id": "win_1", "edge_id": "e_1"},
    }


def test_protocol_canonicalization():
    assert protected_field_metrics(sample_packet(6), "Protocol: TCP")["target_protocol_ok"] == 1
    assert protected_field_metrics(sample_packet(17), "Protocol: UDP")["target_protocol_ok"] == 1


def test_deterministic_template_has_full_exact_fidelity():
    packet = sample_packet()
    metrics = protected_field_metrics(packet, deterministic_template(packet))
    assert all(v == 1 for k, v in metrics.items() if k.endswith("_ok"))


def test_detector_audit_detects_missing_margin():
    packet = sample_packet()
    audit = protected_detector_audit(
        packet,
        "Prediction: 1. Detector Score: 0.8123. Threshold: 0.5. Protocol: TCP.",
    )
    assert audit["pass"] is False
    assert "score_margin" in audit["failures"]


def test_unsupported_attack_and_remediation_diagnostics():
    assert unsupported_claim_flags("Potential SSH brute-force attempt.")["unsupported_attack_category"] == 1
    assert unsupported_claim_flags("Investigate whether malware indicators are present.")["unsupported_attack_category"] == 0
    assert unsupported_claim_flags("Block the connection and isolate the endpoint.")["unauthorized_remediation"] == 1
    assert unsupported_claim_flags("Do not block or isolate the host based on this evidence.")["unauthorized_remediation"] == 0
