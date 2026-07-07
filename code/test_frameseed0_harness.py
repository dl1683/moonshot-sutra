"""Tests for the FRAMESEED-0 B27 audit harness.

These are audit-contract tests only. They do not run learner optimization or
hidden-family performance evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from frameseed0_harness import (
    ABSORPTION_PRECEDENCE,
    BASELINE_NAMES,
    TERMINAL_TOKENS,
    BlindPacketConstructor,
    BudgetLedger,
    Packet,
    TokenEvidence,
    _all_signal_gates,
    assign_terminal_token,
    audit_baseline_parity,
    audit_budget_recomputation,
    audit_constructor_provenance,
    audit_packet_serialization,
    audit_sabotage_control,
    generate_sibling_worlds,
    generate_world,
    make_baseline_views,
    make_public_transcript,
    packet_bit_length,
    run_generator_mi_audit,
    run_golden_token_controls,
    run_preimplementation_audit,
    split_rngs,
    TaskBundle,
)


PUBLIC_SEED = "FRAMESEED0_TEST_SEED"


def _packet_fixture():
    world = generate_world(PUBLIC_SEED, 16, 0, "test_gate").world
    transcript = make_public_transcript(world)
    rng = split_rngs(PUBLIC_SEED, f"packet:{world.world_id}")["packet_construction"]
    packet = BlindPacketConstructor().construct(transcript, rng)
    return world, transcript, packet


def test_world_generator_uses_split_streams_and_decisive_intervention():
    generated = generate_world(PUBLIC_SEED, 16, 3, "unit")
    purposes = {record.purpose for record in generated.rng_records}
    assert purposes == set([
        "world_structure",
        "names",
        "orientations",
        "hidden_queries",
        "packet_construction",
        "learner_tie_breaks",
        "baseline_tie_breaks",
        "ablations",
    ])
    assert generated.world.decisive_intervention_exists()


def test_blind_constructor_packet_has_clean_provenance_and_serialization():
    _, transcript, packet = _packet_fixture()
    provenance = audit_constructor_provenance(packet, transcript)
    serialization = audit_packet_serialization(packet)
    assert packet.constructor_mode == "blind"
    assert provenance.passed
    assert serialization.passed
    assert packet.declared_bits == packet_bit_length(packet)
    assert any(entry.entry_type == "representation_patch" for entry in packet.entries)


def test_support_swap_sabotage_is_detected():
    _, transcript, packet = _packet_fixture()
    report = audit_sabotage_control(packet, transcript)
    assert report.passed
    details = report.findings[0].details
    assert details["passed"] is False


def test_baseline_parity_passes_and_denied_packet_field_fails():
    world, _, packet = _packet_fixture()
    siblings = generate_sibling_worlds(PUBLIC_SEED, world, 2)
    bundle = TaskBundle(world.world_id, tuple(s.world_id for s in siblings))

    views = make_baseline_views(packet, bundle, query_budget=100)
    assert audit_baseline_parity(views).passed

    denied = make_baseline_views(packet, bundle, query_budget=100, denied_fields={"l2_cegis": ["entries"]})
    denied_report = audit_baseline_parity(denied)
    assert not denied_report.passed
    assert any(f.check_id == "BASELINE_PACKET_HASH_PARITY" and not f.passed for f in denied_report.findings)


def test_budget_recomputation_detects_tampered_packet_bits():
    _, _, packet = _packet_fixture()
    assert audit_budget_recomputation(packet, BudgetLedger(packet_bits=packet_bit_length(packet))).passed
    tampered = BudgetLedger(packet_bits=packet_bit_length(packet) + 8)
    report = audit_budget_recomputation(packet, tampered)
    assert not report.passed
    assert any(f.check_id == "BUDGET_PACKET_BITS_MATCH" and not f.passed for f in report.findings)


def test_generator_mi_audit_passes_at_precommitted_size():
    report = run_generator_mi_audit(PUBLIC_SEED, sample_count=10000, threshold=0.05)
    assert report.passed
    assert report.metrics["worst_metric"] <= 0.05


def test_token_precedence_void_boolean_rep_prior_negative_absorptions_signal():
    assert assign_terminal_token(_all_signal_gates(smuggling_detected=True)) == TERMINAL_TOKENS["void"]
    assert assign_terminal_token(_all_signal_gates(boolean_escape_satisfied=False)) == TERMINAL_TOKENS["boolean_trap"]
    assert assign_terminal_token(_all_signal_gates(representation_noncontainment_passed=False)) == TERMINAL_TOKENS["representation_prior"]
    assert assign_terminal_token(TokenEvidence(l3_full_threshold_passed=False)) == TERMINAL_TOKENS["negative"]
    assert assign_terminal_token(_all_signal_gates(baseline_absorptions={"teaching_dimension": True, "rag": True})) == TERMINAL_TOKENS["teaching_dimension"]
    assert assign_terminal_token(_all_signal_gates(baseline_absorptions={"library_learning": True, "nuisance_oracle": True})) == TERMINAL_TOKENS["library_learning"]
    assert assign_terminal_token(_all_signal_gates()) == TERMINAL_TOKENS["signal"]


def test_golden_token_controls_all_pass():
    report = run_golden_token_controls()
    assert report.passed
    assert set(report.metrics.values()) >= {TERMINAL_TOKENS["void"], TERMINAL_TOKENS["negative"], TERMINAL_TOKENS["signal"]}


def test_top_level_audit_declares_no_performance_runs():
    report = run_preimplementation_audit(PUBLIC_SEED, dry_run_worlds=400, mi_threshold=0.25)
    assert report.passed
    assert report.metrics["no_performance_runs"] is True
    assert report.metrics["hidden_hfa_reported"] is False


def test_declared_names_are_complete_for_control_surface():
    assert "l2_cegis" in BASELINE_NAMES
    assert "library_learning" in BASELINE_NAMES
    assert ABSORPTION_PRECEDENCE[0] == "teaching_dimension"