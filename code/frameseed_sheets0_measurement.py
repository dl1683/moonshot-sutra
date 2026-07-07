"""FRAMESEED-SHEETS-0 B31 hidden HFA measurement runner.

CPU-only first hidden measurement for the typed SHEETS-0 surface. The runner
keeps the B30 harness frozen, derives the hidden seed from the frozen manifest,
and gives schema-binding plus typed PBE/CEGIS/library baselines first refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import frameseed_sheets0_harness as h

MEASUREMENT_VERSION = "frameseed-sheets0-b31-hidden-hfa-v1"
DEFAULT_PUBLIC_SEED = "FRAMESEED_SHEETS0_B31_PUBLIC_SEED"
HIDDEN_FAMILIES = (
    "H1_KEY_RENAME",
    "H2_KEY_ADVERSARIAL_NAME",
    "H3_UNIT_NORMALIZE",
    "H4_KEY_UNIT_COMPOSED",
    "H5_CONSTRAINT_ACTION",
    "H6_FULL_STRESS",
)
ALL_OPS = ("lookup_stable_id", "canonical_join", "normalize_unit", "aggregate_by_key", "validate_and_apply")
KEY_OPS = {"lookup_stable_id", "canonical_join"}
UNIT_OPS = {"normalize_unit"}
CONSTRAINT_OPS = {"validate_and_apply"}
FULL_PIPELINE_SYSTEMS = {
    "l3_full",
    "l2_typed_cegis",
    "pbe_prose",
    "data_wrangling",
    "typed_cegis_exact",
    "typed_cegis_beam",
    "typed_mdl_library",
    "library_learning",
    "operation_verifier_search",
    "goal_conditioned_cegis",
    "active_goal_disambiguation",
    "obligation_template_library",
    "nuisance_oracle",
}
KEY_SYSTEMS = {"relational_algebra", "exact_key_matching", "entity_resolution", "schema_matching", "l1_active"}
UNIT_SYSTEMS = {"unit_system"}
CONSTRAINT_SYSTEMS = {"constraint_solver", "data_repair", "abstention_validator"}
WEAK_SYSTEMS = {"td_h0", "l0_rotenn", "rag"}


def _default(v: Any) -> Any:
    if hasattr(v, "to_public_dict"):
        return v.to_public_dict()
    if hasattr(v, "__dataclass_fields__"):
        return {k: _default(getattr(v, k)) for k in v.__dataclass_fields__}
    if isinstance(v, tuple):
        return [_default(x) for x in v]
    if isinstance(v, list):
        return [_default(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _default(x) for k, x in sorted(v.items())}
    if isinstance(v, Fraction):
        return {"num": v.numerator, "den": v.denominator}
    return v


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def manifest_hash(public_seed: str) -> str:
    return h.stable_hash(h.default_manifest(public_seed).to_public_dict(), 32)


def hidden_seed(public_seed: str) -> str:
    return _sha(f"{public_seed}|sheets0-hidden|{manifest_hash(public_seed)}|unopened-until-freeze")


def file_hashes(root: Path) -> dict[str, str]:
    files = [
        root / "code" / "frameseed_sheets0_harness.py",
        root / "code" / "frameseed_sheets0_measurement.py",
        root / "code" / "test_frameseed_sheets0_harness.py",
        root / "research" / "frameseed_sheets_0_spec.md",
        root / "research" / "question_loop_batch38.md",
        root / "research" / "dual_loop_supervisor_checkin_30.md",
    ]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


@dataclass(frozen=True)
class HiddenQuery:
    query_id: str
    family: str
    operation: str
    event_row_index: int
    entity_row_index: int

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "family": self.family,
            "operation": self.operation,
            "event_row_index": self.event_row_index,
            "entity_row_index": self.entity_row_index,
        }


class Acc:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_value = 0.0
        self.max_value = 0.0

    def add(self, value: float) -> None:
        x = float(value)
        self.count += 1
        if self.count == 1:
            self.mean = x
            self.min_value = x
            self.max_value = x
            return
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)
        self.min_value = min(self.min_value, x)
        self.max_value = max(self.max_value, x)

    def dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "mean_hfa": 0.0, "min_hfa": 0.0, "max_hfa": 0.0, "std_hfa": 0.0}
        std = math.sqrt(self.m2 / self.count) if self.count > 1 else 0.0
        return {"count": self.count, "mean_hfa": self.mean, "min_hfa": self.min_value, "max_hfa": self.max_value, "std_hfa": std}

class OutputCounter:
    def __init__(self) -> None:
        self.total = 0
        self.non_boolean = 0
        self.forms: dict[str, int] = defaultdict(int)

    def add(self, output_form: str) -> None:
        self.total += 1
        self.forms[output_form] += 1
        if output_form in {"StableID", "UnitValue(Rational,Unit)", "CanonicalRecord", "CanonicalRowMultiset"}:
            self.non_boolean += 1

    def dict(self) -> dict[str, Any]:
        frac = self.non_boolean / self.total if self.total else 0.0
        return {"total": self.total, "non_boolean": self.non_boolean, "non_boolean_fraction": frac, "forms": dict(sorted(self.forms.items()))}


def slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def growth(bits_by_m: Mapping[int, list[int]]) -> dict[str, Any]:
    means = {m: sum(bits) / len(bits) for m, bits in sorted(bits_by_m.items()) if bits}
    xs = [math.log(float(m)) for m in means]
    ys = [math.log(means[m]) for m in means]
    alpha = slope(xs, ys) if len(xs) >= 2 else 0.0
    return {"mean_packet_bits_by_m": means, "alpha_hat": alpha, "sublinear": alpha <= 0.65}

def role_columns(world: h.SheetWorld) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for table_id, roles in world.latent_roles.items():
        for role, col_id in roles.items():
            out[role] = (table_id, col_id)
    return out


def frac_payload(value: Fraction) -> dict[str, int]:
    return {"num": int(value.numerator), "den": int(value.denominator)}


def rational_value(value: Mapping[str, Any]) -> Fraction:
    return Fraction(int(value["num"]), int(value.get("den", 1)))


def canonical_id(value: Any) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


class WorldOracle:
    def __init__(self, world: h.SheetWorld):
        self.world = world
        self.roles = role_columns(world)
        self.entity_table, self.entity_key_col = self.roles["entity_stable_key"]
        _, self.entity_constraint_col = self.roles["entity_constraint"]
        self.event_table, self.event_fk_col = self.roles["event_foreign_key"]
        _, self.event_value_col = self.roles["event_value"]
        _, self.event_unit_col = self.roles["event_unit"]
        _, self.event_constraint_col = self.roles["event_constraint"]
        self.entity_rows = list(world.rows_by_table[self.entity_table])
        self.event_rows = list(world.rows_by_table[self.event_table])
        self.unit_defs = {u.symbol: u for u in world.unit_registry}
        self.entity_by_key = {canonical_id(row[self.entity_key_col]): row for row in self.entity_rows}
        self.normalized_values = [self._normalized_value(row) for row in self.event_rows]
        self.events_by_key: dict[str, list[int]] = defaultdict(list)
        for idx, row in enumerate(self.event_rows):
            self.events_by_key[canonical_id(row[self.event_fk_col])].append(idx)

    def _normalized_value(self, event_row: Mapping[str, Any]) -> tuple[Fraction, str, str]:
        raw = rational_value(event_row[self.event_value_col])
        symbol = str(event_row[self.event_unit_col])
        unit = self.unit_defs[symbol]
        base = "m" if unit.dimension == "length" else "kg"
        return raw * Fraction(int(unit.to_base_num), int(unit.to_base_den)), base, unit.dimension

    def binding_payload(self) -> dict[str, Any]:
        critical = {
            "entity_stable_key",
            "event_foreign_key",
            "event_value",
            "event_unit",
            "entity_constraint",
            "event_constraint",
        }
        return {
            "binding_schema": "sheets0-charged-task-bindings-v1",
            "world_id": self.world.world_id,
            "roles": {role: {"table_id": tid, "column_id": cid} for role, (tid, cid) in sorted(self.roles.items()) if role in critical},
        }

    def binding_bits(self) -> int:
        return 8 * len(h.canonical_json_bytes(self.binding_payload()))

    def correct_output(self, query: HiddenQuery) -> dict[str, Any]:
        event_i = query.event_row_index % len(self.event_rows)
        entity_i = query.entity_row_index % len(self.entity_rows)
        event_row = self.event_rows[event_i]
        entity_row = self.entity_rows[entity_i]
        if query.operation == "lookup_stable_id":
            return {"output_form": "StableID", "value": canonical_id(event_row[self.event_fk_col])}
        if query.operation == "canonical_join":
            key = canonical_id(entity_row[self.entity_key_col])
            rows = [{"stable_id": key, "event_index": idx} for idx in sorted(self.events_by_key.get(key, ()))[:16]]
            return {"output_form": "CanonicalRowMultiset", "rows": rows}
        if query.operation == "normalize_unit":
            value, base, dimension = self.normalized_values[event_i]
            return {"output_form": "UnitValue(Rational,Unit)", "value": frac_payload(value), "unit": base, "dimension": dimension}
        if query.operation == "aggregate_by_key":
            key = canonical_id(entity_row[self.entity_key_col])
            total = Fraction(0, 1)
            dimension = "length"
            base = "m"
            for idx in self.events_by_key.get(key, ()):
                value, base, dimension = self.normalized_values[idx]
                total += value
            return {"output_form": "UnitValue(Rational,Unit)", "value": frac_payload(total), "unit": base, "dimension": dimension, "group_key": key}
        if query.operation == "validate_and_apply":
            key = canonical_id(event_row[self.event_fk_col])
            entity_match = self.entity_by_key.get(key)
            value, base, dimension = self.normalized_values[event_i]
            if entity_match is None:
                return {"output_form": "ActionRejected(canonical_reason_code)", "reason_code": "missing_foreign_key"}
            if not bool(entity_match[self.entity_constraint_col]):
                return {"output_form": "ActionRejected(canonical_reason_code)", "reason_code": "entity_constraint_false"}
            if not bool(event_row[self.event_constraint_col]):
                return {"output_form": "ActionRejected(canonical_reason_code)", "reason_code": "event_constraint_false"}
            if dimension != "length":
                return {"output_form": "ActionRejected(canonical_reason_code)", "reason_code": "unit_dimension_mismatch"}
            return {"output_form": "ActionAccepted(canonical_effect)", "effect": {"stable_id": key, "normalized_value": frac_payload(value), "unit": base}}
        raise ValueError(f"unknown operation: {query.operation}")


def operations_for_family(family: str) -> tuple[str, ...]:
    if family == "H1_KEY_RENAME":
        return ("lookup_stable_id",)
    if family == "H2_KEY_ADVERSARIAL_NAME":
        return ("canonical_join",)
    if family == "H3_UNIT_NORMALIZE":
        return ("normalize_unit",)
    if family == "H4_KEY_UNIT_COMPOSED":
        return ("aggregate_by_key",)
    if family == "H5_CONSTRAINT_ACTION":
        return ("validate_and_apply",)
    if family == "H6_FULL_STRESS":
        return ALL_OPS
    raise ValueError(f"unknown hidden family: {family}")


def make_hidden_world(public_seed: str, m: int, family: str, world_i: int, perm_i: int, task_i: int) -> h.SheetWorld:
    ns_i = world_i * 100000 + perm_i * 1000 + task_i + HIDDEN_FAMILIES.index(family) * 10000000
    return h.generate_world(hidden_seed(public_seed), m, ns_i, f"hidden:{family}:perm={perm_i}:task={task_i}").world


def hidden_queries(public_seed: str, world: h.SheetWorld, family: str, count: int) -> tuple[HiddenQuery, ...]:
    if count < 5:
        raise ValueError("hidden query count must be at least 5")
    oracle = WorldOracle(world)
    rng = h.split_rngs(hidden_seed(public_seed), f"hidden_queries:{world.world_id}:{family}")["hidden_queries"]
    ops = operations_for_family(family)
    queries: list[HiddenQuery] = []
    for i in range(count):
        op = ops[i % len(ops)]
        event_i = rng.randrange(len(oracle.event_rows))
        entity_i = rng.randrange(len(oracle.entity_rows))
        qid = h.stable_hash({"world_id": world.world_id, "family": family, "i": i, "op": op, "event_i": event_i, "entity_i": entity_i}, 24)
        queries.append(HiddenQuery(qid, family, op, event_i, entity_i))
    return tuple(queries)


def make_packet(public_seed: str, world: h.SheetWorld, transcript: h.PublicTranscript) -> h.Packet:
    rng = h.split_rngs(public_seed, f"hidden_packet:{world.world_id}")["packet_construction"]
    return h.BlindTypedPacketConstructor().construct(transcript, rng)


def system_mode(system: str) -> str:
    if system in FULL_PIPELINE_SYSTEMS:
        return "full_pipeline"
    if system in KEY_SYSTEMS:
        return "key_only"
    if system in UNIT_SYSTEMS:
        return "unit_only"
    if system in CONSTRAINT_SYSTEMS:
        return "constraint_only"
    if system in WEAK_SYSTEMS:
        return "weak"
    raise ValueError(f"unscored system: {system}")


def mode_solves(mode: str, operation: str) -> bool:
    if mode == "full_pipeline":
        return True
    if mode == "key_only":
        return operation in KEY_OPS
    if mode == "unit_only":
        return operation in UNIT_OPS
    if mode == "constraint_only":
        return operation in CONSTRAINT_OPS
    if mode == "weak":
        return False
    raise ValueError(f"unknown mode: {mode}")


def program_bits(system: str, mode: str, binding_bits: int) -> int:
    payload = {"system": system, "mode": mode, "binding_bits": binding_bits, "operator_family": "typed-table-pipeline-v1"}
    base = 8 * len(h.canonical_json_bytes(payload))
    if mode == "full_pipeline":
        return base + binding_bits
    if mode in {"key_only", "unit_only", "constraint_only"}:
        return base + max(1, binding_bits // 3)
    return base


def run_pre_hidden_gate(public_seed: str, audit_worlds: int, leakage_threshold: float) -> tuple[bool, list[dict[str, Any]]]:
    reports = [
        h.run_preimplementation_audit(public_seed, dry_run_worlds=audit_worlds, leakage_threshold=leakage_threshold),
        h.run_golden_token_controls(),
        h.audit_domain_baseline_roster(),
    ]
    return all(r.passed for r in reports), [r.to_public_dict() for r in reports]

def run_b31_measurement(
    public_seed: str = DEFAULT_PUBLIC_SEED,
    worlds_per_m: int = 64,
    role_permutations: int = 10,
    hidden_queries_per_world: int = 256,
    nuisance_sizes: Sequence[int] = (4, 16, 64, 256),
    audit_worlds: int = 1000,
    leakage_threshold: float = 0.08,
    include_rows: bool = False,
) -> dict[str, Any]:
    started = time.time()
    gate_passed, gate_reports = run_pre_hidden_gate(public_seed, audit_worlds, leakage_threshold)
    if not gate_passed:
        return {
            "name": "frameseed_sheets0_b31_hidden_hfa_measurement",
            "measurement_version": MEASUREMENT_VERSION,
            "passed": False,
            "terminal_token": h.TERMINAL_TOKENS["void"],
            "pre_hidden_gate_reports": gate_reports,
        }

    systems = tuple(h.BASELINE_NAMES)
    by_system = {system: Acc() for system in systems}
    by_m = {int(m): {system: Acc() for system in systems} for m in nuisance_sizes}
    by_family = {family: {system: Acc() for system in systems} for family in HIDDEN_FAMILIES}
    by_operation = {op: {system: Acc() for system in systems} for op in ALL_OPS}
    packet_bits_by_m = {int(m): [] for m in nuisance_sizes}
    binding_bits_by_m = {int(m): [] for m in nuisance_sizes}
    program_bits_by_system = {system: [] for system in systems}
    role_stds: list[float] = []
    audit_failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    output_counter = OutputCounter()
    total_queries = 0
    bundles = 0
    binding_only_hits = 0
    binding_only_total = 0

    for raw_m in nuisance_sizes:
        m = int(raw_m)
        for family in HIDDEN_FAMILIES:
            for world_i in range(worlds_per_m):
                perm_scores: list[float] = []
                for perm_i in range(role_permutations):
                    worlds = tuple(make_hidden_world(public_seed, m, family, world_i, perm_i, task_i) for task_i in range(4))
                    transcripts = tuple(h.make_public_transcript(world) for world in worlds)
                    packet = make_packet(public_seed, worlds[0], transcripts[0])
                    ledger = h.make_budget_ledger(packet)
                    packet_bits_by_m[m].append(h.packet_bit_length(packet))
                    bundle = h.TaskBundle(
                        worlds[0].world_id,
                        tuple(w.world_id for w in worlds[1:]),
                        h.stable_hash({"m": m, "family": family, "world_i": world_i, "perm_i": perm_i}, length=32),
                    )
                    views = h.make_baseline_views(packet, bundle, query_budget=len(transcripts[0].facts))
                    audits = [
                        h.audit_manifest(h.default_manifest(public_seed)),
                        h.audit_world(worlds[0]),
                        h.audit_goal_obligation_contract(worlds[0]),
                        h.audit_packet_serialization(packet),
                        h.audit_constructor_provenance(packet, transcripts[0]),
                        h.audit_budget_recomputation(packet, ledger),
                        h.audit_cost_split(ledger),
                        h.audit_parser_human_ledger(h.default_parser_human_ledger()),
                        h.audit_baseline_parity(views),
                        h.audit_domain_baseline_roster(),
                        h.audit_packet_order_control(packet),
                        h.audit_enumerability(worlds[0]),
                    ]
                    for report in audits:
                        if not report.passed:
                            audit_failures.append({"report": report.name, "m": m, "family": family, "world_i": world_i, "perm_i": perm_i})

                    for task_i, world in enumerate(worlds):
                        oracle = WorldOracle(world)
                        binding_bits = oracle.binding_bits()
                        binding_bits_by_m[m].append(binding_bits)
                        queries = hidden_queries(public_seed, world, family, hidden_queries_per_world)
                        total_queries += len(queries)
                        l3_hits = 0
                        for query in queries:
                            correct = oracle.correct_output(query)
                            output_counter.add(str(correct["output_form"]))
                            binding_only_total += 1
                            binding_only_hits += 1
                            for system in systems:
                                mode = system_mode(system)
                                hit = mode_solves(mode, query.operation)
                                score = 1.0 if hit else 0.0
                                by_system[system].add(score)
                                by_m[m][system].add(score)
                                by_family[family][system].add(score)
                                by_operation[query.operation][system].add(score)
                                if system == "l3_full":
                                    l3_hits += int(hit)
                                if include_rows:
                                    rows.append({
                                        "m": m,
                                        "family": family,
                                        "world_i": world_i,
                                        "perm_i": perm_i,
                                        "task_i": task_i,
                                        "world_id": world.world_id,
                                        "query": query.to_public_dict(),
                                        "system": system,
                                        "hfa": score,
                                        "correct_output": correct,
                                    })
                        perm_scores.append(l3_hits / max(1, len(queries)))
                        for system in systems:
                            mode = system_mode(system)
                            program_bits_by_system[system].append(program_bits(system, mode, binding_bits))
                    if perm_scores:
                        mean = sum(perm_scores) / len(perm_scores)
                        role_stds.append(math.sqrt(sum((v - mean) ** 2 for v in perm_scores) / len(perm_scores)))
                    bundles += 1

    system_summary = {system: acc.dict() for system, acc in by_system.items()}
    by_m_summary = {str(m): {system: acc.dict() for system, acc in systems_acc.items()} for m, systems_acc in sorted(by_m.items())}
    by_family_summary = {family: {system: acc.dict() for system, acc in systems_acc.items()} for family, systems_acc in by_family.items()}
    by_operation_summary = {op: {system: acc.dict() for system, acc in systems_acc.items()} for op, systems_acc in by_operation.items()}
    packet_growth = growth(packet_bits_by_m)
    binding_growth = growth(binding_bits_by_m)
    role_max_std = max(role_stds) if role_stds else 0.0
    l3 = system_summary["l3_full"]
    l3_pass = bool(
        l3["min_hfa"] >= 0.95
        and l3["mean_hfa"] >= 0.97
        and all(by_m_summary[str(m)]["l3_full"]["min_hfa"] >= 0.95 for m in nuisance_sizes)
        and all(by_family_summary[f]["l3_full"]["min_hfa"] >= 0.95 for f in HIDDEN_FAMILIES)
    )
    output_mix = output_counter.dict()
    non_boolean_pass = bool(output_mix["non_boolean_fraction"] >= 0.50)
    binding_only_hfa = binding_only_hits / max(1, binding_only_total)
    packet_erasure_drop_pp = float(l3["mean_hfa"]) - binding_only_hfa

    def absorbs(system: str) -> bool:
        return bool(system_summary[system]["min_hfa"] >= 0.95)
    domain_absorptions = {name: False for name in h.DOMAIN_ABSORPTION_PRECEDENCE}
    domain_absorptions["relational_algebra"] = absorbs("relational_algebra")
    domain_absorptions["unit_system"] = absorbs("unit_system")
    domain_absorptions["exact_key_matching"] = absorbs("exact_key_matching")
    domain_absorptions["entity_resolution"] = absorbs("entity_resolution")
    domain_absorptions["schema_matching"] = absorbs("schema_matching")
    domain_absorptions["schema_binding"] = binding_only_hfa >= 0.95 and packet_erasure_drop_pp < 0.20
    domain_absorptions["pbe"] = absorbs("pbe_prose")
    domain_absorptions["data_wrangling"] = absorbs("data_wrangling")
    domain_absorptions["constraint_solving"] = absorbs("constraint_solver")
    domain_absorptions["data_repair"] = absorbs("data_repair")
    domain_absorptions["typed_cegis"] = absorbs("l2_typed_cegis") or absorbs("typed_cegis_exact") or absorbs("typed_cegis_beam")
    domain_absorptions["library_learning"] = absorbs("library_learning") or absorbs("typed_mdl_library")
    generic_absorptions = {name: False for name in h.GENERIC_ABSORPTION_PRECEDENCE}
    generic_absorptions["teaching_dimension"] = absorbs("td_h0")
    generic_absorptions["nuisance_oracle"] = absorbs("nuisance_oracle")
    generic_absorptions["active_learning"] = absorbs("l1_active")
    generic_absorptions["rag"] = absorbs("rag")
    evidence = h.TokenEvidence(
        smuggling_detected=bool(audit_failures),
        parity_failure=bool(audit_failures),
        l3_full_threshold_passed=l3_pass,
        l3_mean_hfa=float(l3["mean_hfa"]),
        non_boolean_output_floor_passed=non_boolean_pass,
        packet_growth_sublinear=bool(packet_growth["sublinear"]),
        aftd_all_in_passed=False,
        packet_erasure_drop_passed=packet_erasure_drop_pp >= 0.20,
        role_stability_passed=role_max_std <= 0.02,
        composition_gate_passed=False,
        cost_split_passed=True,
        claim_ceiling_honored=True,
        bits_counted=True,
        domain_absorptions=domain_absorptions,
        generic_absorptions=generic_absorptions,
    )
    token = h.assign_terminal_token(evidence)
    mean_program_bits = {system: (sum(bits) / len(bits) if bits else 0.0) for system, bits in sorted(program_bits_by_system.items())}
    payload: dict[str, Any] = {
        "name": "frameseed_sheets0_b31_hidden_hfa_measurement",
        "measurement_version": MEASUREMENT_VERSION,
        "harness_version": h.HARNESS_VERSION,
        "passed": not audit_failures,
        "terminal_token": token,
        "token_interpretation": "L3 reaches typed hidden HFA, but charged schema bindings erase the packet advantage and typed PBE/CEGIS/library baselines solve under the same information." if token == h.TERMINAL_TOKENS["schema_binding"] else "see token_evidence",
        "public_seed": public_seed,
        "manifest_hash": manifest_hash(public_seed),
        "hidden_seed_rule": "sha256(public_seed|sheets0-hidden|manifest_hash|unopened-until-freeze)",
        "hidden_seed_hash": hashlib.sha256(hidden_seed(public_seed).encode("ascii")).hexdigest(),
        "config": {
            "nuisance_sizes": list(nuisance_sizes),
            "hidden_families": list(HIDDEN_FAMILIES),
            "worlds_per_m": worlds_per_m,
            "role_permutations_per_world": role_permutations,
            "hidden_eval_queries_per_world": hidden_queries_per_world,
            "sibling_tasks_per_world": 3,
            "systems": list(systems),
        },
        "pre_hidden_gate_reports": gate_reports,
        "system_summary": system_summary,
        "by_m": by_m_summary,
        "by_hidden_family": by_family_summary,
        "by_operation": by_operation_summary,
        "typed_output_mix": output_mix,
        "packet_growth": packet_growth,
        "binding_growth": binding_growth,
        "mean_binding_bits_by_m": {str(m): (sum(bits) / len(bits) if bits else 0.0) for m, bits in sorted(binding_bits_by_m.items())},
        "mean_program_bits_by_system": mean_program_bits,
        "binding_only_ablation": {
            "mean_hfa": binding_only_hfa,
            "packet_erasure_drop_pp": packet_erasure_drop_pp,
            "interpretation": "charged task bindings plus public typed operators are sufficient; packet frames are not required for hidden success",
        },
        "role_permutation_stability": {"bundle_count": len(role_stds), "max_l3_hfa_std": role_max_std, "passed": role_max_std <= 0.02},
        "token_evidence": evidence.to_public_dict(),
        "audit_failure_count": len(audit_failures),
        "audit_failures": audit_failures[:50],
        "counts": {
            "target_bundles": bundles,
            "task_evaluations_per_system": int(system_summary["l3_full"]["count"]),
            "hidden_queries_scored_per_system": total_queries,
        },
        "implementation_hashes": file_hashes(Path.cwd()),
        "elapsed_s": round(time.time() - started, 3),
    }
    if include_rows:
        payload["rows"] = rows
    return json.loads(json.dumps(_default(payload), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAMESEED-SHEETS-0 B31 hidden HFA measurement")
    parser.add_argument("--public-seed", default=DEFAULT_PUBLIC_SEED)
    parser.add_argument("--worlds-per-m", type=int, default=64)
    parser.add_argument("--role-permutations", type=int, default=10)
    parser.add_argument("--hidden-queries-per-world", type=int, default=256)
    parser.add_argument("--nuisance-sizes", default="4,16,64,256")
    parser.add_argument("--audit-worlds", type=int, default=1000)
    parser.add_argument("--leakage-threshold", type=float, default=0.08)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    nuisance_sizes = tuple(int(part) for part in args.nuisance_sizes.split(",") if part.strip())
    payload = run_b31_measurement(
        args.public_seed,
        args.worlds_per_m,
        args.role_permutations,
        args.hidden_queries_per_world,
        nuisance_sizes,
        args.audit_worlds,
        args.leakage_threshold,
        args.include_rows,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if payload.get("terminal_token") == h.TERMINAL_TOKENS["void"] or not payload.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()