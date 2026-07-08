"""Tests for the WGD-0 B36 pre-hidden audit harness.

Audit-contract tests only. They do not open a hidden seed or run WGD signal
measurement.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from wgd0_harness import (
    ABSORPTION_PRECEDENCE,
    REQUIRED_ABSORBERS,
    TERMINAL_TOKENS,
    BlindWGDPacketConstructor,
    CostLedger,
    GrammarIR,
    Packet,
    TokenEvidence,
    _all_signal_gates,
    assign_terminal_token,
    audit_baseline_parity,
    audit_cost_ledger,
    audit_grammar_ir,
    audit_native_absorber_capability_witnesses,
    audit_no_signal_measurement,
    audit_packet_serialization,
    count_nonduplicate_reduced_siblings,
    default_human_substrate_ledger,
    generate_sibling_worlds,
    generate_world,
    make_baseline_views,
    make_cost_ledger,
    make_public_transcript,
    make_smoke_grammar_ir,
    packet_bit_length,
    run_golden_token_controls,
    run_native_absorber_witnesses,
    run_preimplementation_audit,
    split_rngs,
    TaskBundle,
)

PUBLIC_SEED = "WGD0_TEST_SEED"
SMOKE_SEED = "WGD0_TEST_SMOKE_SEED"


def _fixture():
    world = generate_world(PUBLIC_SEED, 16, 0, "test_gate").world
    transcript = make_public_transcript(world)
    rng = split_rngs(PUBLIC_SEED, f"packet:{world.world_id}")["packet_construction"]
    packet = BlindWGDPacketConstructor().construct(transcript, rng)
    grammar = make_smoke_grammar_ir(transcript)
    return world, transcript, packet, grammar


def test_world_generator_split_streams_and_public_transcript_clean():
    generated = generate_world(PUBLIC_SEED, 16, 3, "unit")
    assert {record.purpose for record in generated.rng_records} == set([
        "world_structure", "opaque_ids", "surface_permutation", "value_generation",
        "public_transcript", "packet_construction", "baseline_tie_breaks",
        "calibration_worlds", "leakage_audits", "ablations", "hidden_queries",
    ])
    transcript = make_public_transcript(generated.world)
    assert transcript.schema["schema_version"] == "wgd0-public-substrate-v1"
    assert all(trace.feedback in {"ACCEPTED", "REJECTED", "UNSAFE", "AMBIGUOUS", "WRONG"} for trace in transcript.traces)


def test_packet_grammar_and_cost_audits_pass():
    _, transcript, packet, grammar = _fixture()
    human = default_human_substrate_ledger(packet)
    ledger = make_cost_ledger(packet, grammar, human)
    assert packet.declared_bits == packet_bit_length(packet)
    assert audit_packet_serialization(packet).passed
    assert audit_grammar_ir(grammar, transcript).passed
    assert audit_cost_ledger(ledger, human).passed
    assert ledger.total_cost_substrate_charged >= ledger.total_cost_substrate_free


def test_grammar_ir_rejects_solver_smuggling():
    _, transcript, _, grammar = _fixture()
    bad_node = grammar.nodes[0]
    smuggled = type(bad_node)(bad_node.node_id, bad_node.node_type, {"code": "lambda x: hidden_label"}, bad_node.provenance, bad_node.cost_category, bad_node.declared_bits)
    bad = GrammarIR(grammar.ir_version, (smuggled,) + grammar.nodes[1:], True, False)
    report = audit_grammar_ir(bad, transcript)
    assert not report.passed
    assert any(f.check_id == "GRAMMAR_IR_NO_SOLVER_PAYLOADS" and not f.passed for f in report.findings)


def test_baseline_parity_passes_and_denied_fields_fail():
    world, _, packet, _ = _fixture()
    siblings = generate_sibling_worlds(SMOKE_SEED, world, 3)
    bundle = TaskBundle(world.world_id, tuple(s.world_id for s in siblings))
    assert audit_baseline_parity(make_baseline_views(packet, bundle, query_budget=0)).passed
    denied = make_baseline_views(packet, bundle, query_budget=0, denied_fields={"cegis": ["entries"]})
    report = audit_baseline_parity(denied)
    assert not report.passed
    assert any(f.check_id == "BASELINE_PACKET_HASH_PARITY" and not f.passed for f in report.findings)


def test_native_absorbers_are_complete_and_competent():
    results = run_native_absorber_witnesses()
    report = audit_native_absorber_capability_witnesses(results)
    assert {r.absorber_name for r in results} == set(REQUIRED_ABSORBERS)
    assert report.passed
    assert all(r.passed for r in results)
    assert "proxy_absorber" not in {r.status for r in results}


def test_sibling_clone_resistance_counts_three_nonduplicates():
    world = generate_world(PUBLIC_SEED, 16, 0, "test_gate").world
    siblings = generate_sibling_worlds(SMOKE_SEED, world, 3)
    count, details = count_nonduplicate_reduced_siblings(world, siblings)
    assert count == 3
    assert all(item["shared_field_ids"] == 0 for item in details)


def test_token_precedence_voids_absorbers_inconclusive_negative_signal():
    assert assign_terminal_token(_all_signal_gates(post_hidden_mutation=True)) == TERMINAL_TOKENS["void_post_hidden_mutation"]
    assert assign_terminal_token(_all_signal_gates(baseline_parity_failure=True)) == TERMINAL_TOKENS["void_baseline_parity"]
    assert assign_terminal_token(_all_signal_gates(absorptions={"operation_ontology": True, "pbe": True})) == TERMINAL_TOKENS["operation_ontology"]
    assert assign_terminal_token(_all_signal_gates(absorptions={"pbe": True, "cegis": True})) == TERMINAL_TOKENS["pbe"]
    assert assign_terminal_token(_all_signal_gates(baseline_not_native=True)) == TERMINAL_TOKENS["inconclusive_baselines"]
    assert assign_terminal_token(TokenEvidence(functional_gates_passed=False)) == TERMINAL_TOKENS["negative"]
    assert assign_terminal_token(_all_signal_gates()) == TERMINAL_TOKENS["signal"]
    assert ABSORPTION_PRECEDENCE[0] == "operation_ontology"


def test_no_signal_measurement_and_top_level_smoke():
    assert audit_no_signal_measurement({"hidden_seed_opened": False, "hidden_hfa_reported": False, "wgd_signal_measured": False}).passed
    assert run_golden_token_controls().passed
    report = run_preimplementation_audit(PUBLIC_SEED, SMOKE_SEED, dry_run_worlds=120, leakage_threshold=0.5)
    assert report.passed
    assert report.metrics["hidden_seed_opened"] is False
    assert report.metrics["hidden_hfa_reported"] is False
    assert report.metrics["wgd_signal_measured"] is False
    assert report.metrics["baseline_calibration_only"] is True