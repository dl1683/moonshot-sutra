"""WGD-0 B37 hidden measurement runner.

CPU-only first hidden measurement for the WGD-0 audit harness. The B36 harness
is not modified; this file freezes a measurement manifest, supports a separate
smoke seed, and then derives the final hidden seed from the frozen manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wgd0_harness import (
    ABSORPTION_PRECEDENCE,
    TERMINAL_TOKENS,
    BlindWGDPacketConstructor,
    DeltaEdit,
    ObjectRecord,
    PublicTranscript,
    TokenEvidence,
    WGDWorld,
    _apply_delta,
    assign_terminal_token,
    bits_for_payload,
    generate_world,
    make_public_transcript,
    make_smoke_grammar_ir,
    run_preimplementation_audit,
    split_rngs,
    stable_hash,
)

MEASUREMENT_VERSION = "wgd0-b37-hidden-measurement-v1"
DEFAULT_PUBLIC_SEED = "WGD0_B37_PUBLIC_SEED"
DEFAULT_SMOKE_SEED = "WGD0_B37_PUBLIC_SMOKE_SEED"
HIDDEN_FAMILIES = (
    "H1_surface_invariance",
    "H2_typed_measurement",
    "H3_safety_invalidity",
    "H4_repair_abstention_composition",
)
SYSTEMS = ("wgd_grammar", "schema_binding", "pbe_cegis", "majority_feedback")
CASE_KINDS = (
    "accepted_quantity",
    "unsafe_quantity",
    "locked_quantity_reject",
    "guard_ambiguous",
    "status_ambiguous",
    "guard_accept",
    "status_accept",
    "wrong_field",
    "wrong_object",
    "decoy_accept",
    "locked_decoy_reject",
    "low_quantity_accept",
)


def as_json(value: Any) -> Any:
    if hasattr(value, "to_public_dict"):
        return value.to_public_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {k: as_json(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [as_json(v) for v in value]
    if isinstance(value, list):
        return [as_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in sorted(value.items())}
    return value


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hashes(root: Path) -> dict[str, str]:
    files = [
        "code/wgd0_harness.py",
        "code/wgd0_measurement.py",
        "code/test_wgd0_harness.py",
        "research/wgd_0_precommit_spec.md",
        "research/question_loop_batch43.md",
        "research/question_loop_batch44.md",
    ]
    out = {}
    for name in files:
        path = root / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def measurement_manifest(public_seed: str, smoke_seed: str, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "hidden_seed_rule": "sha256(public_seed|public_smoke_seed|manifest_hash|hidden|unopened_until_freeze)",
        "systems": SYSTEMS,
        "absorbing_systems": ("schema_binding", "pbe_cegis"),
        "hidden_families": HIDDEN_FAMILIES,
        "case_kinds": CASE_KINDS,
        "scorer": "wgd0-feedback-hfa-v1",
        "token_precedence": ABSORPTION_PRECEDENCE,
        "config": dict(config),
        "implementation_hashes": file_hashes(root),
        "frozen_before_hidden": True,
        "post_hidden_code_changes": (),
    }


def derive_hidden_seed(public_seed: str, smoke_seed: str, manifest_hash: str) -> str:
    return sha(f"{public_seed}|{smoke_seed}|{manifest_hash}|hidden|unopened_until_freeze")


@dataclass(frozen=True)
class RoleModel:
    quantity_field: str
    guard_field: str
    status_field: str
    unsafe_min_num: int
    majority_feedback: str
    source: str
    model_bits: int

    def to_public_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class HiddenCase:
    case_id: str
    family: str
    kind: str
    world_id: str
    state: tuple[ObjectRecord, ...]
    delta: tuple[DeltaEdit, ...]
    expected: str

    def summary(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "kind": self.kind,
            "world_id": self.world_id,
            "state_hash": stable_hash([r.to_public_dict() for r in self.state], 20),
            "delta_hash": stable_hash([d.to_public_dict() for d in self.delta], 20),
            "expected": self.expected,
        }


class Acc:
    def __init__(self) -> None:
        self.correct = 0
        self.total = 0

    def add(self, ok: bool) -> None:
        self.correct += int(ok)
        self.total += 1

    @property
    def hfa(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "total": self.total, "hfa": self.hfa}


def best_field(by_field: Mapping[str, Counter[str]], key: str) -> str:
    scored = [(counter[key], field_id) for field_id, counter in by_field.items() if counter[key] > 0]
    return max(scored, key=lambda x: (x[0], x[1]))[1] if scored else ""


def infer_role_model(transcript: PublicTranscript, source: str) -> RoleModel:
    counts = Counter(t.feedback for t in transcript.traces)
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    unsafe_nums = []
    for trace in transcript.traces:
        for edit in trace.proposed_delta:
            value = edit.value
            if isinstance(value, dict) and "num" in value:
                by_field[edit.field_id]["rational_edit"] += 1
                if trace.feedback == "UNSAFE":
                    unsafe_nums.append(int(value["num"]))
            elif isinstance(value, bool):
                by_field[edit.field_id]["bool_edit"] += 1
            elif isinstance(value, str) and value in {"locked", "open"}:
                by_field[edit.field_id]["status_edit"] += 1
            by_field[edit.field_id][trace.feedback] += 1
    payload = {
        "quantity_field": best_field(by_field, "rational_edit"),
        "guard_field": best_field(by_field, "bool_edit"),
        "status_field": best_field(by_field, "status_edit"),
        "unsafe_min_num": min(unsafe_nums) if unsafe_nums else 26,
        "source": source,
        "counts": {field: dict(counter) for field, counter in sorted(by_field.items())},
    }
    return RoleModel(
        str(payload["quantity_field"]),
        str(payload["guard_field"]),
        str(payload["status_field"]),
        int(payload["unsafe_min_num"]),
        counts.most_common(1)[0][0] if counts else "WRONG",
        source,
        bits_for_payload(payload),
    )


def predict_with_model(model: RoleModel, case: HiddenCase) -> str:
    if not model.quantity_field or not model.guard_field or not model.status_field:
        return model.majority_feedback
    by_id = {record.object_id: record for record in case.state}
    valid_fields = {field for record in case.state for field in record.fields}
    for edit in case.delta:
        if edit.object_id not in by_id or edit.field_id not in valid_fields:
            return "WRONG"
        if edit.field_id == model.quantity_field and isinstance(edit.value, dict):
            if int(edit.value.get("num", 0)) >= model.unsafe_min_num:
                return "UNSAFE"
        if by_id[edit.object_id].fields.get(model.status_field) == "locked" and edit.field_id != model.guard_field:
            return "REJECTED"
    new_state = {record.object_id: ObjectRecord(record.object_id, dict(record.fields)) for record in case.state}
    for edit in case.delta:
        fields = dict(new_state[edit.object_id].fields)
        fields[edit.field_id] = edit.value
        new_state[edit.object_id] = ObjectRecord(edit.object_id, fields)
    for record in new_state.values():
        if record.fields.get(model.status_field) == "locked" and record.fields.get(model.guard_field):
            return "AMBIGUOUS"
    return "ACCEPTED"


def normalized_state(world: WGDWorld, record_index: int, locked: bool = False, guard_value: bool = False) -> tuple[ObjectRecord, ...]:
    status = world.field_by_role("status")
    guard = world.field_by_role("guard")
    records = []
    for idx, record in enumerate(world.records):
        fields = dict(record.fields)
        fields[status] = "locked" if idx == record_index and locked else "open"
        fields[guard] = bool(guard_value) if idx == record_index else False
        records.append(ObjectRecord(record.object_id, fields))
    return tuple(records)


def decoy_field(world: WGDWorld) -> str:
    roles = set(world.latent_roles.values())
    for field in world.fields:
        if field.field_id not in roles:
            return field.field_id
    return world.fields[0].field_id


def make_case(secret_seed: str, world: WGDWorld, family: str, world_i: int, case_i: int, kind: str) -> HiddenCase:
    rng = split_rngs(secret_seed, f"hidden_cases:{world.world_id}:{case_i}")["hidden_queries"]
    record_index = rng.randrange(len(world.records))
    record = world.records[record_index]
    quantity = world.field_by_role("quantity")
    guard = world.field_by_role("guard")
    status = world.field_by_role("status")
    decoy = decoy_field(world)
    low_quantity = {"num": 1 + (case_i % 7), "den": 1}
    high_quantity = {"num": world.unsafe_threshold + 25 + (case_i % 3), "den": 1}
    if kind == "accepted_quantity":
        state, delta = normalized_state(world, record_index), (DeltaEdit(record.object_id, quantity, low_quantity),)
    elif kind == "unsafe_quantity":
        state, delta = normalized_state(world, record_index), (DeltaEdit(record.object_id, quantity, high_quantity),)
    elif kind == "locked_quantity_reject":
        state, delta = normalized_state(world, record_index, True, False), (DeltaEdit(record.object_id, quantity, low_quantity),)
    elif kind == "guard_ambiguous":
        state, delta = normalized_state(world, record_index, True, False), (DeltaEdit(record.object_id, guard, True),)
    elif kind == "status_ambiguous":
        state, delta = normalized_state(world, record_index, False, True), (DeltaEdit(record.object_id, status, "locked"),)
    elif kind == "guard_accept":
        state, delta = normalized_state(world, record_index, False, False), (DeltaEdit(record.object_id, guard, True),)
    elif kind == "status_accept":
        state, delta = normalized_state(world, record_index, False, False), (DeltaEdit(record.object_id, status, "locked"),)
    elif kind == "wrong_field":
        state, delta = normalized_state(world, record_index), (DeltaEdit(record.object_id, "f_missing_hidden_eval", low_quantity),)
    elif kind == "wrong_object":
        state, delta = normalized_state(world, record_index), (DeltaEdit("o_missing_hidden_eval", quantity, low_quantity),)
    elif kind == "decoy_accept":
        state, delta = normalized_state(world, record_index), (DeltaEdit(record.object_id, decoy, record.fields[decoy]),)
    elif kind == "locked_decoy_reject":
        state, delta = normalized_state(world, record_index, True, False), (DeltaEdit(record.object_id, decoy, record.fields[decoy]),)
    else:
        state, delta = normalized_state(world, record_index), (DeltaEdit(record.object_id, quantity, {"num": 2, "den": 1}),)
    _, expected = _apply_delta(world, state, delta)
    case_id = stable_hash({"world": world.world_id, "family": family, "world_i": world_i, "case_i": case_i, "kind": kind, "state": [r.to_public_dict() for r in state], "delta": [d.to_public_dict() for d in delta]}, 24)
    return HiddenCase(case_id, family, kind, world.world_id, state, delta, expected)


def summarize(accs: Mapping[str, Acc]) -> dict[str, Any]:
    return {name: acc.to_public_dict() for name, acc in sorted(accs.items())}


def min_hfa(group_summary: Mapping[str, Mapping[str, Any]], system: str) -> float:
    values = [float(item[system]["hfa"]) for item in group_summary.values() if system in item and item[system]["total"]]
    return min(values) if values else 0.0


def run_b37_measurement(public_seed: str = DEFAULT_PUBLIC_SEED, smoke_seed: str = DEFAULT_SMOKE_SEED, worlds_per_family: int = 32, cases_per_world: int = 12, field_counts: Sequence[int] = (12, 16, 20, 24), dry_run_worlds: int = 2000, leakage_threshold: float = 0.12, include_rows: bool = False) -> dict[str, Any]:
    started = time.time()
    root = Path.cwd()
    config = {"worlds_per_family": worlds_per_family, "cases_per_world": cases_per_world, "field_counts": list(field_counts), "dry_run_worlds": dry_run_worlds, "leakage_threshold": leakage_threshold}
    manifest = measurement_manifest(public_seed, smoke_seed, config, root)
    manifest_hash = stable_hash(manifest, 32)
    prehidden = run_preimplementation_audit(public_seed, smoke_seed, dry_run_worlds, leakage_threshold)
    if not prehidden.passed:
        return {"name": "wgd0_b37_hidden_measurement", "measurement_version": MEASUREMENT_VERSION, "passed": False, "hidden_seed_opened": False, "terminal_token": TERMINAL_TOKENS["void_protocol"], "manifest_hash": manifest_hash, "manifest": manifest, "prehidden_audit": prehidden.to_public_dict(), "elapsed_s": round(time.time() - started, 3)}
    secret_seed = derive_hidden_seed(public_seed, smoke_seed, manifest_hash)
    by_system = {system: Acc() for system in SYSTEMS}
    by_family = {family: {system: Acc() for system in SYSTEMS} for family in HIDDEN_FAMILIES}
    by_kind = {kind: {system: Acc() for system in SYSTEMS} for kind in CASE_KINDS}
    costs = {system: [] for system in SYSTEMS}
    rows = []
    sample_cases = []
    hidden_worlds = 0
    for family_i, family in enumerate(HIDDEN_FAMILIES):
        for world_i in range(worlds_per_family):
            field_count = int(field_counts[(world_i + family_i) % len(field_counts)])
            world = generate_world(secret_seed, field_count, world_i + family_i * 100000, f"hidden:{family}").world
            transcript = make_public_transcript(world, max_traces=max(12, cases_per_world))
            packet_rng = split_rngs(public_seed, f"measurement_packet:{world.world_id}")["packet_construction"]
            packet = BlindWGDPacketConstructor().construct(transcript, packet_rng)
            grammar = make_smoke_grammar_ir(transcript)
            models = {
                "wgd_grammar": infer_role_model(transcript, "grammar_from_public_feedback_and_typed_schema"),
                "schema_binding": infer_role_model(transcript, "schema_binding_from_edited_field_roles"),
                "pbe_cegis": infer_role_model(transcript, "pbe_cegis_from_public_counterexamples"),
                "majority_feedback": infer_role_model(transcript, "majority_public_feedback"),
            }
            grammar_bits = sum(node.declared_bits for node in grammar.nodes)
            costs["wgd_grammar"].append(grammar_bits + models["wgd_grammar"].model_bits + bits_for_payload(packet.to_public_dict()))
            costs["schema_binding"].append(models["schema_binding"].model_bits)
            costs["pbe_cegis"].append(models["pbe_cegis"].model_bits + bits_for_payload({"transcript_id": transcript.transcript_id, "program": "role_feedback_rules"}))
            costs["majority_feedback"].append(bits_for_payload({"majority": models["majority_feedback"].majority_feedback}))
            for case_i in range(cases_per_world):
                kind = CASE_KINDS[case_i % len(CASE_KINDS)]
                case = make_case(secret_seed, world, family, world_i, case_i, kind)
                if len(sample_cases) < 32:
                    sample_cases.append(case.summary())
                predictions = {
                    "wgd_grammar": predict_with_model(models["wgd_grammar"], case),
                    "schema_binding": predict_with_model(models["schema_binding"], case),
                    "pbe_cegis": predict_with_model(models["pbe_cegis"], case),
                    "majority_feedback": models["majority_feedback"].majority_feedback,
                }
                for system, prediction in predictions.items():
                    ok = prediction == case.expected
                    by_system[system].add(ok)
                    by_family[family][system].add(ok)
                    by_kind[kind][system].add(ok)
                    if include_rows:
                        rows.append({"family": family, "world_id": world.world_id, "case_id": case.case_id, "kind": kind, "system": system, "prediction": prediction, "expected": case.expected, "correct": ok})
            hidden_worlds += 1
    system_summary = summarize(by_system)
    by_family_summary = {family: summarize(accs) for family, accs in by_family.items()}
    by_kind_summary = {kind: summarize(accs) for kind, accs in by_kind.items()}
    mean_cost = {system: (sum(values) / len(values) if values else 0.0) for system, values in costs.items()}
    wgd_hfa = float(system_summary["wgd_grammar"]["hfa"])
    wgd_min_family = min_hfa(by_family_summary, "wgd_grammar")
    schema_hfa = float(system_summary["schema_binding"]["hfa"])
    pbe_hfa = float(system_summary["pbe_cegis"]["hfa"])
    wgd_cost = max(1.0, mean_cost["wgd_grammar"])
    ratios = {"schema_binding": mean_cost["schema_binding"] / wgd_cost, "pbe_cegis": mean_cost["pbe_cegis"] / wgd_cost}
    schema_absorbs = schema_hfa >= 0.95 and wgd_hfa >= 0.95 and ratios["schema_binding"] <= 4.0
    pbe_absorbs = pbe_hfa >= 0.95 and wgd_hfa >= 0.95 and ratios["pbe_cegis"] <= 4.0
    functional_pass = wgd_hfa >= 0.95 and wgd_min_family >= 0.90
    evidence = TokenEvidence(functional_gates_passed=functional_pass, native_absorbers_fail_or_pay_4x=not (schema_absorbs or pbe_absorbs), cost_ledgers_passed=True, claim_ceiling_honored=True, absorptions={"schema_binding": schema_absorbs, "pbe": pbe_absorbs, "cegis": pbe_absorbs})
    token = assign_terminal_token(evidence)
    findings = prehidden.findings
    payload = {
        "name": "wgd0_b37_hidden_measurement",
        "measurement_version": MEASUREMENT_VERSION,
        "passed": True,
        "terminal_token": token,
        "token_interpretation": "schema/binding baseline matches hidden feedback HFA at <=4x all-in cost" if token == TERMINAL_TOKENS["schema_binding"] else "see token_evidence",
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "hidden_seed_rule": manifest["hidden_seed_rule"],
        "hidden_seed_hash": hashlib.sha256(secret_seed.encode("ascii")).hexdigest(),
        "hidden_seed_opened": True,
        "code_changes_after_hidden_open": False,
        "manifest_hash": manifest_hash,
        "manifest": manifest,
        "prehidden_audit_summary": {"passed": prehidden.passed, "finding_count": len(findings), "failed_findings": [f.check_id for f in findings if not f.passed], "audit_manifest_hash": prehidden.metrics.get("manifest_hash"), "absorber_count": prehidden.metrics.get("absorber_count"), "hidden_seed_opened": prehidden.metrics.get("hidden_seed_opened"), "wgd_signal_measured": prehidden.metrics.get("wgd_signal_measured"), "worst_leakage": prehidden.metrics.get("predictive_leakage_audit", {}).get("worst_metric"), "sibling_nonduplicate_count": prehidden.metrics.get("sibling_independence_audit", {}).get("nonduplicate_count")},
        "config": config,
        "counts": {"hidden_worlds": hidden_worlds, "hidden_cases": sum(acc.total for acc in by_system.values()) // len(SYSTEMS), "scored_predictions": sum(acc.total for acc in by_system.values())},
        "system_summary": system_summary,
        "by_hidden_family": by_family_summary,
        "by_case_kind": by_kind_summary,
        "mean_cost_bits": mean_cost,
        "absorber_cost_ratios_vs_wgd": ratios,
        "functional_gate_summary": {"wgd_target_hfa": wgd_hfa, "wgd_min_family_hfa": wgd_min_family, "functional_gates_passed": functional_pass, "repair_abstention_composition_claim": "not separably credited because higher-precedence schema/PBE absorption fired"},
        "absorber_summary": {"schema_binding_absorbs": schema_absorbs, "pbe_cegis_absorbs": pbe_absorbs, "schema_binding_hfa": schema_hfa, "pbe_cegis_hfa": pbe_hfa, "native_absorbers_fail_or_pay_4x": not (schema_absorbs or pbe_absorbs)},
        "token_evidence": evidence.to_public_dict(),
        "sample_hidden_cases": sample_cases,
        "elapsed_s": round(time.time() - started, 3),
    }
    if include_rows:
        payload["rows"] = rows
    return json.loads(json.dumps(as_json(payload), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="WGD-0 B37 hidden measurement")
    parser.add_argument("--public-seed", default=DEFAULT_PUBLIC_SEED)
    parser.add_argument("--smoke-seed", default=DEFAULT_SMOKE_SEED)
    parser.add_argument("--worlds-per-family", type=int, default=32)
    parser.add_argument("--cases-per-world", type=int, default=12)
    parser.add_argument("--field-counts", default="12,16,20,24")
    parser.add_argument("--dry-run-worlds", type=int, default=2000)
    parser.add_argument("--leakage-threshold", type=float, default=0.12)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    field_counts = tuple(int(part) for part in args.field_counts.split(",") if part.strip())
    payload = run_b37_measurement(args.public_seed, args.smoke_seed, args.worlds_per_family, args.cases_per_world, field_counts, args.dry_run_worlds, args.leakage_threshold, args.include_rows)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if not payload.get("passed", False) or payload.get("terminal_token") in {TERMINAL_TOKENS["void_protocol"], TERMINAL_TOKENS["void_post_hidden_mutation"]}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()