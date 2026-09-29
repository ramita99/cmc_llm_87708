import pytest

from cmc87708.protocol import (
    assert_llm_facing_packet,
    assert_no_ground_truth,
    assert_structural_allowlist,
    stable_hash,
)


def test_nested_ground_truth_is_blocked():
    with pytest.raises(AssertionError):
        assert_no_ground_truth({"packet": {"detector": {"Label": 1}}})


def test_safe_packet_passes():
    assert_no_ground_truth({"packet": {"detector": {"prediction": 1, "score": 0.9}}})


def test_structural_allowlist_rejects_unapproved_field():
    with pytest.raises(AssertionError):
        assert_structural_allowlist({"src_in_degree": 1, "suspicious_1hop_edge_count": 3})


def test_hash_is_stable_to_mapping_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_audit_metadata_is_blocked_from_llm_payload():
    with pytest.raises(AssertionError):
        assert_llm_facing_packet({"dataset": "x", "packet_hash": "abc"})


def test_normal_detector_state_packet_passes_guards():
    assert_llm_facing_packet({
        "dataset": "x",
        "state": "detector",
        "target": {"edge_id": "e1"},
        "detector_state": {"prediction": 0, "score": 0.1, "threshold": 0.5},
    })
