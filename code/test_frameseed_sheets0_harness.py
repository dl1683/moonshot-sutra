"""Tests for the FRAMESEED-SHEETS-0 B30 public audit harness.

Audit-contract tests only. They do not run hidden evaluation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from frameseed_sheets0_harness import (
    BASELINE_NAMES,
    DOMAIN_ABSORPTION_PRECEDENCE,
    TERMINAL_TOKENS,
    BlindTypedPacketConstructor,
    BudgetLedger,
    TokenEvidence,
    _all_signal_gates,
    assign_terminal_token,
    audit_baseline_parity,
    audit_budget_recomputation,
    audit_constructor_provenance,
    audit_cost_split,
    audit_domain_baseline_roster,
    audit_packet_serialization,
    audit_world,
    generate_sibling_worlds,
    generate_world,
    make_baseline_views,
    make_budget_ledger,
    make_public_transcript,
    packet_bit_length,
    run_generator_leakage_audit,
    run_golden_token_controls,
    run_preimplementation_audit,
    split_rngs,
    TaskBundle,
)

PUBLIC_SEED = "FRAMESEED_SHEETS0_TEST_SEED"


def _packet_fixture():
    world = generate_world(PUBLIC_SEED, 16, 0, "test_gate").world
    transcript = make_public_transcript(world)
    rng = split_rngs(PUBLIC_SEED, f"packet:{world.world_id}")["packet_construction"]
    packet = BlindTypedPacketConstructor().construct(transcript, rng)
    return world, transcript, packet


def test_world_generator_decoys_outputs_and_split_rngs():
    generated = generate_world(PUBLIC_SEED, 16, 3, "unit")
    assert {r.purpose for r in generated.rng_records} == set([
        "world_structure", "schema_names", "row_order", "display_names",
        "unit_choices", "constraints", "packet_construction",
        "learner_tie_breaks", "baseline_tie_breaks", "ablations", "hidden_queries",
    ])
    assert audit_world(generated.world).passed
    assert generated.world.non_boolean_output_fraction() >= 0.50


def test_blind_constructor_packet_has_provenance_and_budget_split():
    _, transcript, packet = _packet_fixture()
    ledger = make_budget_ledger(packet)
    assert packet.constructor_mode == "blind"
    assert audit_constructor_provenance(packet, transcript).passed
    assert audit_packet_serialization(packet).passed
    assert audit_budget_recomputation(packet, ledger).passed
    assert audit_cost_split(ledger).passed
    assert packet.declared_bits == packet_bit_length(packet)
    assert any(e.entry_type == "composition_gate" for e in packet.entries)


def test_baseline_parity_detects_denied_packet_fields():
    world, _, packet = _packet_fixture()
    siblings = generate_sibling_worlds(PUBLIC_SEED, world, 3)
    bundle = TaskBundle(world.world_id, tuple(s.world_id for s in siblings))
    assert audit_baseline_parity(make_baseline_views(packet, bundle, query_budget=10)).passed
    denied = make_baseline_views(packet, bundle, query_budget=10, denied_fields={"relational_algebra": ["entries"]})
    report = audit_baseline_parity(denied)
    assert not report.passed
    assert any(f.check_id == "BASELINE_PACKET_HASH_PARITY" and not f.passed for f in report.findings)


def test_domain_baseline_roster_and_leakage_smoke_pass():
    assert audit_domain_baseline_roster().passed
    report = run_generator_leakage_audit(PUBLIC_SEED, sample_count=400, threshold=0.25)
    assert report.passed
    assert report.metrics["worst_metric"] <= 0.25


def test_token_precedence_domain_absorber_before_negative_and_generic_after_l3():
    assert assign_terminal_token(_all_signal_gates(smuggling_detected=True)) == TERMINAL_TOKENS["void"]
    assert assign_terminal_token(_all_signal_gates(parser_prior_absorbed=True)) == TERMINAL_TOKENS["parser_prior"]
    assert assign_terminal_token(TokenEvidence(l3_full_threshold_passed=False, non_boolean_output_floor_passed=True, domain_absorptions={"relational_algebra": True})) == TERMINAL_TOKENS["relational_algebra"]
    assert assign_terminal_token(TokenEvidence(l3_full_threshold_passed=False, generic_absorptions={"teaching_dimension": True})) == TERMINAL_TOKENS["negative"]
    assert assign_terminal_token(_all_signal_gates(generic_absorptions={"teaching_dimension": True})) == TERMINAL_TOKENS["teaching_dimension"]
    assert assign_terminal_token(_all_signal_gates()) == TERMINAL_TOKENS["signal"]


def test_golden_controls_and_top_level_public_audit():
    assert run_golden_token_controls().passed
    report = run_preimplementation_audit(PUBLIC_SEED, dry_run_worlds=400, leakage_threshold=0.25)
    assert report.passed
    assert report.metrics["no_performance_runs"] is True
    assert report.metrics["hidden_hfa_reported"] is False


def test_declared_names_cover_q37_absorbers():
    assert "relational_algebra" in BASELINE_NAMES
    assert "unit_system" in BASELINE_NAMES
    assert "entity_resolution" in BASELINE_NAMES
    assert "pbe_prose" in BASELINE_NAMES
    assert "data_wrangling" in BASELINE_NAMES
    assert "typed_cegis_exact" in BASELINE_NAMES
    assert DOMAIN_ABSORPTION_PRECEDENCE[0] == "relational_algebra"