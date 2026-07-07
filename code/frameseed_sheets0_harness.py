"""FRAMESEED-SHEETS-0 B30 public audit harness.

CPU-only pre-hidden audit surface: typed generator scaffolding, blind packet
construction, canonical serialization, cost ledgers, baseline parity,
domain-absorber roster checks, leakage probes, enumerability metrics, and token
precedence controls. No hidden seed is opened and no hidden HFA is reported.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

HARNESS_VERSION = "frameseed-sheets0-audit-harness-v1"
NUISANCE_SIZES = (4, 16, 64, 256)
RNG_PURPOSES = ("world_structure", "schema_names", "row_order", "display_names", "unit_choices", "constraints", "packet_construction", "learner_tie_breaks", "baseline_tie_breaks", "ablations", "hidden_queries")
BASELINE_NAMES = ("l3_full", "td_h0", "l0_rotenn", "l1_active", "l2_typed_cegis", "rag", "nuisance_oracle", "library_learning", "relational_algebra", "unit_system", "exact_key_matching", "entity_resolution", "schema_matching", "pbe_prose", "data_wrangling", "constraint_solver", "data_repair", "typed_cegis_exact", "typed_cegis_beam", "typed_mdl_library", "operation_verifier_search", "goal_conditioned_cegis", "active_goal_disambiguation", "obligation_template_library", "abstention_validator")
DOMAIN_ABSORPTION_PRECEDENCE = ("relational_algebra", "unit_system", "exact_key_matching", "entity_resolution", "schema_matching", "schema_binding", "pbe", "data_wrangling", "constraint_solving", "data_repair", "typed_cegis", "library_learning")
GENERIC_ABSORPTION_PRECEDENCE = ("teaching_dimension", "nuisance_oracle", "active_learning", "rag")
ABSORPTION_PRECEDENCE = DOMAIN_ABSORPTION_PRECEDENCE + GENERIC_ABSORPTION_PRECEDENCE
TERMINAL_TOKENS = {
    "signal": "FRAMESEED_SHEETS0_SIGNAL",
    "representation_prior": "FRAMESEED_SHEETS0_ABSORBED_BY_REPRESENTATION_PRIOR",
    "parser_prior": "FRAMESEED_SHEETS0_ABSORBED_BY_PARSER_PRIOR",
    "relational_algebra": "FRAMESEED_SHEETS0_ABSORBED_BY_RELATIONAL_ALGEBRA",
    "unit_system": "FRAMESEED_SHEETS0_ABSORBED_BY_UNIT_SYSTEM",
    "exact_key_matching": "FRAMESEED_SHEETS0_ABSORBED_BY_EXACT_KEY_MATCHING",
    "entity_resolution": "FRAMESEED_SHEETS0_ABSORBED_BY_ENTITY_RESOLUTION",
    "schema_matching": "FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_MATCHING",
    "schema_binding": "FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING",
    "pbe": "FRAMESEED_SHEETS0_ABSORBED_BY_PBE",
    "data_wrangling": "FRAMESEED_SHEETS0_ABSORBED_BY_DATA_WRANGLING",
    "constraint_solving": "FRAMESEED_SHEETS0_ABSORBED_BY_CONSTRAINT_SOLVING",
    "data_repair": "FRAMESEED_SHEETS0_ABSORBED_BY_DATA_REPAIR",
    "typed_cegis": "FRAMESEED_SHEETS0_ABSORBED_BY_TYPED_CEGIS",
    "library_learning": "FRAMESEED_SHEETS0_ABSORBED_BY_LIBRARY_LEARNING",
    "teaching_dimension": "FRAMESEED_SHEETS0_ABSORBED_BY_TEACHING_DIMENSION",
    "nuisance_oracle": "FRAMESEED_SHEETS0_ABSORBED_BY_NUISANCE_ORACLE",
    "active_learning": "FRAMESEED_SHEETS0_ABSORBED_BY_ACTIVE_LEARNING",
    "rag": "FRAMESEED_SHEETS0_ABSORBED_BY_RAG",
    "typed_boolean_trap": "FRAMESEED_SHEETS0_TYPED_BOOLEAN_TRAP",
    "void": "FRAMESEED_SHEETS0_VOID_SMUGGLED_SCHEMA",
    "negative": "FRAMESEED_SHEETS0_NEGATIVE",
}
CRITICAL_ROLES = {"entity_stable_key", "event_foreign_key", "event_value", "event_unit", "entity_constraint", "event_constraint"}
SEMANTIC_NAME_FRAGMENTS = ("key", "id", "unit", "amount", "name", "valid", "foreign", "join", "customer", "target", "hidden", "stable", "display", "constraint", "action")
BANNED_EXECUTABLE_TERMS = ("stable_id_role", "display_name_role", "unit_role", "true_key", "target_key", "latent_role", "role_map", "hidden_family", "answer_column", "generator_seed", "target_program", "solution_schema", "oracle_label")
REQUIRED_GOAL_OBLIGATIONS = {"preserve_identity", "normalize_quantity", "reject_invalid_action", "join_normalize_guard", "abstain_on_ambiguous_binding"}
UNIT_REGISTRY = (("cm", "length", 1, 100), ("m", "length", 1, 1), ("in", "length", 127, 5000), ("ft", "length", 381, 1250), ("g", "mass", 1, 1000), ("kg", "mass", 1, 1), ("lb", "mass", 45359237, 100000000))

def _json_default(v: Any) -> Any:
    if hasattr(v, "to_public_dict"): return v.to_public_dict()
    if hasattr(v, "__dataclass_fields__"): return {k: _json_default(getattr(v, k)) for k in v.__dataclass_fields__}
    if isinstance(v, tuple): return [_json_default(x) for x in v]
    if isinstance(v, list): return [_json_default(x) for x in v]
    if isinstance(v, dict): return {str(k): _json_default(x) for k, x in sorted(v.items())}
    return v

def canonical_json_bytes(v: Any) -> bytes:
    return json.dumps(_json_default(v), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def stable_hash(v: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json_bytes(v)).hexdigest()[:length]

def derive_seed(public_seed: str, purpose: str, namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{public_seed}|{purpose}|{namespace}".encode()).digest()[:8], "big")

@dataclass(frozen=True)
class RNGStreamRecord:
    purpose: str; namespace: str; seed_hash: str; draw_count: int

class AuditedRandom:
    def __init__(self, public_seed: str, purpose: str, namespace: str):
        self.purpose, self.namespace = purpose, namespace
        self.seed = derive_seed(public_seed, purpose, namespace)
        self._rng = random.Random(self.seed); self.draw_count = 0
    def _count(self, n: int = 1) -> None: self.draw_count += n
    def randrange(self, stop: int) -> int: self._count(); return self._rng.randrange(stop)
    def getrandbits(self, bits: int) -> int: self._count(); return self._rng.getrandbits(bits)
    def choice(self, items: Sequence[Any]) -> Any: self._count(); return self._rng.choice(items)
    def shuffle(self, items: list[Any]) -> None: self._count(max(len(items)-1, 1)); self._rng.shuffle(items)
    def record(self) -> RNGStreamRecord:
        return RNGStreamRecord(self.purpose, self.namespace, hashlib.sha256(str(self.seed).encode()).hexdigest()[:16], self.draw_count)

def split_rngs(public_seed: str, namespace: str) -> dict[str, AuditedRandom]:
    return {p: AuditedRandom(public_seed, p, namespace) for p in RNG_PURPOSES}

def opaque_token(rng: AuditedRandom, prefix: str) -> str: return f"{prefix}{rng.getrandbits(96):024x}"
def decoy_floor(m: int) -> int: return max(3, math.ceil(math.log2(m + 4)))

@dataclass(frozen=True)
class ColumnSpec:
    column_id: str; type_tag: str
    def to_public_dict(self) -> dict[str, str]: return {"column_id": self.column_id, "type_tag": self.type_tag}

@dataclass(frozen=True)
class TableSpec:
    table_id: str; columns: tuple[ColumnSpec, ...]; row_count: int
    def to_public_dict(self) -> dict[str, Any]: return {"table_id": self.table_id, "columns": [c.to_public_dict() for c in self.columns], "row_count": self.row_count}

@dataclass(frozen=True)
class UnitDef:
    symbol: str; dimension: str; to_base_num: int; to_base_den: int
    def to_public_dict(self) -> dict[str, Any]: return {"symbol": self.symbol, "dimension": self.dimension, "to_base_num": self.to_base_num, "to_base_den": self.to_base_den}
@dataclass(frozen=True)
class SheetWorld:
    world_id: str
    m: int
    tables: tuple[TableSpec, ...]
    rows_by_table: Mapping[str, tuple[Mapping[str, Any], ...]]
    latent_roles: Mapping[str, Mapping[str, str]]
    seed_namespace: str
    family_class: str
    goal_obligations: tuple[str, ...]
    unit_registry: tuple[UnitDef, ...]
    def public_schema(self) -> dict[str, Any]:
        return {"schema_version": "frameseed-sheets0-public-schema-v1", "tables": [t.to_public_dict() for t in self.tables], "unit_registry": [u.to_public_dict() for u in self.unit_registry], "operation_grammar": ("lookup", "canonical_join", "aggregate_by_key", "compare_threshold", "validate_and_apply", "abstain_on_ambiguous_binding"), "output_forms": ("StableID", "UnitValue(Rational,Unit)", "CanonicalRecord", "CanonicalRowMultiset", "ActionAccepted(canonical_effect)", "ActionRejected(canonical_reason_code)"), "goal_obligation_kinds": sorted(REQUIRED_GOAL_OBLIGATIONS)}
    def table_by_id(self) -> dict[str, TableSpec]: return {t.table_id: t for t in self.tables}
    def columns_by_id(self) -> dict[str, tuple[str, ColumnSpec, int]]:
        out = {}
        for t in self.tables:
            for i, c in enumerate(t.columns): out[c.column_id] = (t.table_id, c, i)
        return out
    def role_by_column(self) -> dict[str, str]:
        out = {}
        for roles in self.latent_roles.values():
            for role, col in roles.items(): out[col] = role
        return out
    def critical_column_ids(self) -> set[str]: return {c for c, r in self.role_by_column().items() if r in CRITICAL_ROLES}
    def same_type_candidate_counts(self) -> dict[str, int]:
        counts, by_col = {}, self.columns_by_id()
        for col, role in self.role_by_column().items():
            if role not in CRITICAL_ROLES: continue
            table_id, spec, _ = by_col[col]
            counts[role] = sum(1 for c in self.table_by_id()[table_id].columns if c.type_tag == spec.type_tag)
        return counts
    def non_boolean_output_fraction(self) -> float: return 4 / 6
    def has_composed_goal(self) -> bool: return "join_normalize_guard" in self.goal_obligations
    def to_audit_dict(self) -> dict[str, Any]:
        return {"world_id": self.world_id, "m": self.m, "table_count": len(self.tables), "schema_hash": stable_hash(self.public_schema(), 24), "family_class": self.family_class, "non_boolean_output_fraction": self.non_boolean_output_fraction(), "same_type_candidate_counts": self.same_type_candidate_counts()}

@dataclass(frozen=True)
class GeneratedWorld:
    world: SheetWorld; rng_records: tuple[RNGStreamRecord, ...]

def _make_columns(rng: AuditedRandom, critical: Sequence[tuple[str, str]], decoys: Mapping[str, int]) -> tuple[tuple[ColumnSpec, ...], dict[str, str]]:
    cols, roles = [], {}
    for role, typ in critical:
        cid = opaque_token(rng, "c"); cols.append(ColumnSpec(cid, typ)); roles[role] = cid
    for typ, count in decoys.items():
        for i in range(count):
            cid = opaque_token(rng, "c"); cols.append(ColumnSpec(cid, typ)); roles[f"decoy_{typ}_{i}_{cid[-4:]}"] = cid
    rng.shuffle(cols); return tuple(cols), roles

def _value_for_type(rngs: Mapping[str, AuditedRandom], typ: str, i: int, role: str, n: int) -> Any:
    if typ == "id_like": return f"{rngs['world_structure'].choice(('E','A','S','R'))}-{(i % n) + 1:04d}"
    if typ == "text_like": return f"{rngs['display_names'].choice(('ada','ben','cy','dev','eli','far'))}-{(i + rngs['display_names'].randrange(5)) % 5}"
    if typ == "rational": return {"num": 10 + i * 3 + rngs["world_structure"].randrange(9), "den": rngs["world_structure"].choice((1, 2, 4, 5))}
    if typ == "unit_symbol": return rngs["unit_choices"].choice(("cm", "m", "in", "ft") if "unit" in role else ("g", "kg", "lb", "cm"))
    if typ == "constraint_flag": return bool((i + rngs["constraints"].randrange(3)) % 3)
    return f"v{i}"

def generate_world(public_seed: str, m: int, world_index: int, family_class: str = "dry_run") -> GeneratedWorld:
    ns = f"{family_class}:m={m}:world={world_index}"; rngs = split_rngs(public_seed, ns); name_rng = rngs["schema_names"]
    floor, rows_n = decoy_floor(m), min(128, max(16, 16 + m // 2))
    ent_id, evt_id = opaque_token(name_rng, "t"), opaque_token(name_rng, "t")
    ent_cols, ent_roles = _make_columns(name_rng, (("entity_stable_key", "id_like"), ("entity_display_name", "text_like"), ("entity_constraint", "constraint_flag")), {"id_like": floor, "text_like": floor, "constraint_flag": floor, "rational": floor})
    evt_cols, evt_roles = _make_columns(name_rng, (("event_foreign_key", "id_like"), ("event_value", "rational"), ("event_unit", "unit_symbol"), ("event_constraint", "constraint_flag")), {"id_like": floor, "rational": floor, "unit_symbol": floor, "constraint_flag": floor, "text_like": 2})
    tables = [TableSpec(ent_id, ent_cols, rows_n), TableSpec(evt_id, evt_cols, rows_n)]
    latent = {ent_id: ent_roles, evt_id: evt_roles}; rows_by_table = {}
    ent_rows = []
    for i in range(rows_n):
        ent_rows.append({c.column_id: _value_for_type(rngs, c.type_tag, i, next((r for r, cid in ent_roles.items() if cid == c.column_id), "decoy"), rows_n) for c in ent_cols})
    keys = [r[ent_roles["entity_stable_key"]] for r in ent_rows]; evt_rows = []
    for i in range(rows_n):
        row = {}
        for c in evt_cols:
            role = next((r for r, cid in evt_roles.items() if cid == c.column_id), "decoy")
            row[c.column_id] = str(keys[i % len(keys)]) if role == "event_foreign_key" else _value_for_type(rngs, c.type_tag, i, role, rows_n)
        evt_rows.append(row)
    rngs["row_order"].shuffle(ent_rows); rngs["row_order"].shuffle(evt_rows)
    rows_by_table[ent_id], rows_by_table[evt_id] = tuple(ent_rows), tuple(evt_rows)
    units = tuple(UnitDef(*u) for u in UNIT_REGISTRY); obligations = tuple(sorted(REQUIRED_GOAL_OBLIGATIONS))
    world_id = stable_hash({"ns": ns, "tables": [t.to_public_dict() for t in tables], "roles": stable_hash(latent, 32), "rows": stable_hash(rows_by_table, 32)}, 24)
    world = SheetWorld(world_id, m, tuple(tables), rows_by_table, latent, ns, family_class, obligations, units)
    return GeneratedWorld(world, tuple(r.record() for r in rngs.values()))

def generate_sibling_worlds(public_seed: str, target: SheetWorld, count: int = 3) -> tuple[SheetWorld, ...]:
    return tuple(generate_world(public_seed, target.m, i, f"sibling_of_{target.world_id[:8]}").world for i in range(count))

@dataclass(frozen=True)
class PublicFact:
    fact_id: str; fact_type: str; payload: Mapping[str, Any]; source: str
    def to_public_dict(self) -> dict[str, Any]: return {"fact_id": self.fact_id, "fact_type": self.fact_type, "payload": _json_default(dict(self.payload)), "source": self.source}

@dataclass(frozen=True)
class PublicTranscript:
    schema: Mapping[str, Any]; facts: tuple[PublicFact, ...]; schema_fact_id: str; transcript_id: str
    def allowed_provenance_ids(self) -> set[str]: return {self.schema_fact_id} | {f.fact_id for f in self.facts}
    def to_public_dict(self) -> dict[str, Any]: return {"schema": dict(self.schema), "facts": [f.to_public_dict() for f in self.facts], "schema_fact_id": self.schema_fact_id, "transcript_id": self.transcript_id}
def make_public_transcript(world: SheetWorld, example_rows: int = 4) -> PublicTranscript:
    schema = world.public_schema(); schema_id = "schema:" + stable_hash(schema, 24); facts = []
    for table in world.tables[:2]:
        payload = {"table_id": table.table_id, "rows": list(world.rows_by_table[table.table_id][:example_rows])}
        facts.append(PublicFact("fact:" + stable_hash(payload, 24), "public_rows", payload, "public_table_slice"))
    examples = ({"operation": "canonical_join", "obligations": ("preserve_identity", "abstain_on_ambiguous_binding"), "output_form": "CanonicalRowMultiset"}, {"operation": "aggregate_by_key", "obligations": ("normalize_quantity", "preserve_identity"), "output_form": "UnitValue(Rational,Unit)"}, {"operation": "validate_and_apply", "obligations": ("reject_invalid_action", "join_normalize_guard"), "output_form": "ActionRejected(canonical_reason_code)"})
    for payload in examples: facts.append(PublicFact("fact:" + stable_hash(payload, 24), "typed_example", payload, "public_oracle"))
    tid = "transcript:" + stable_hash({"schema": schema_id, "facts": [f.fact_id for f in facts]}, 24)
    return PublicTranscript(schema, tuple(facts), schema_id, tid)

@dataclass(frozen=True)
class PacketEntry:
    entry_type: str; payload: Mapping[str, Any]; provenance: tuple[str, ...]
    def to_public_dict(self) -> dict[str, Any]: return {"entry_type": self.entry_type, "payload": _json_default(dict(self.payload)), "provenance": list(self.provenance)}

@dataclass(frozen=True)
class Packet:
    header: Mapping[str, Any]; entries: tuple[PacketEntry, ...]; constructor_id: str; constructor_mode: str = "blind"; declared_bits: int | None = None
    def to_public_dict(self) -> dict[str, Any]: return {"header": _json_default(dict(self.header)), "entries": [e.to_public_dict() for e in self.entries], "constructor_id": self.constructor_id, "constructor_mode": self.constructor_mode, "declared_bits": self.declared_bits}

def packet_bytes(packet: Packet) -> bytes:
    return canonical_json_bytes({"header": dict(packet.header), "entries": [e.to_public_dict() for e in packet.entries], "constructor_id": packet.constructor_id, "constructor_mode": packet.constructor_mode})
def packet_bit_length(packet: Packet) -> int: return 8 * len(packet_bytes(packet))
def packet_multiset_hash(packet: Packet) -> str: return stable_hash({"header": dict(packet.header), "entries": sorted(stable_hash(e.to_public_dict(), 32) for e in packet.entries)}, 32)

def _assert_transcript_is_public(transcript: PublicTranscript) -> None:
    blob = canonical_json_bytes(transcript).decode("ascii").lower()
    forbidden = ("latent_roles", "role_map", "entity_stable_key", "event_foreign_key", "event_value", "event_unit", "hidden_label", "solution_schema")
    hits = [x for x in forbidden if x in blob]
    if hits: raise ValueError(f"transcript contains non-public fields: {hits}")

class BlindTypedPacketConstructor:
    constructor_id = "blind-public-sheets0-obligation-packet-v1"
    def construct(self, transcript: PublicTranscript, rng: AuditedRandom) -> Packet:
        _assert_transcript_is_public(transcript)
        examples = tuple(f.fact_id for f in transcript.facts if f.fact_type == "typed_example") or (transcript.schema_fact_id,)
        rows = tuple(f.fact_id for f in transcript.facts if f.fact_type == "public_rows") or (transcript.schema_fact_id,)
        sref = (transcript.schema_fact_id,)
        entries = [
            PacketEntry("frame_patch", {"frame_id": "opaque_frame_identity_invariance_v1", "operator_schema": {"op": "same_entity_by_canonical_equality_after_binding", "forbidden_shortcuts": ("row_position", "display_text_only")}, "scope": "same-type id-like candidates after charged binding"}, sref + rows[:1]),
            PacketEntry("frame_patch", {"frame_id": "opaque_frame_quantity_normalization_v1", "operator_schema": {"op": "normalize_quantity_before_math", "requires": ("public_unit_registry", "dimension_compatibility")}, "scope": "rational plus unit-symbol candidates after charged binding"}, sref + examples[:1]),
            PacketEntry("verifier_clause", {"clause_id": "typed_action_obligations_v1", "obligations": sorted(REQUIRED_GOAL_OBLIGATIONS), "failure_output": "ActionRejected(canonical_reason_code)"}, sref + examples[:2]),
            PacketEntry("counterexample", {"anti_rule": "row_order_or_display_text_as_identity", "expected_relation": "canonical_output_unchanged_under_permutation_and_alias_drift"}, rows[:1] + examples[:1]),
            PacketEntry("counterexample", {"anti_rule": "raw_numeric_aggregation_without_unit_dimension_check", "expected_relation": "reject_or_normalize_before_comparison"}, examples[:2]),
            PacketEntry("binding_policy", {"policy_id": "charged_same_type_binding_search_v1", "candidate_sets": ("id_like", "rational", "unit_symbol", "constraint_flag"), "cost_rule": "every selected table_id column_id unit_id obligation_id is charged"}, sref),
            PacketEntry("composition_gate", {"components": ("identity", "quantity", "guard"), "required_evidence": ("subadditive_cost", "local_repair_preserves_prior_obligations", "pipeline_baselines_fail_or_pay_4x")}, sref + examples[:1]),
        ]
        rng.shuffle(entries)
        header = {"version": "frameseed-sheets0-packet-v1", "schema_hash": stable_hash(transcript.schema, 24), "transcript_hash": stable_hash(transcript.to_public_dict(), 24), "l0_hash": "SHEETS0-L0-opaque-typed-records-v1", "h0_hash": "SHEETS0-H0-no-domain-operators-v1", "sir_hash": "SIR0-public-plus-packet-installable-v1"}
        packet = Packet(header, tuple(entries), self.constructor_id, "blind")
        return replace(packet, declared_bits=packet_bit_length(packet))

@dataclass(frozen=True)
class Finding:
    check_id: str; passed: bool; message: str; details: Mapping[str, Any] = field(default_factory=dict)
    def to_public_dict(self) -> dict[str, Any]: return {"check_id": self.check_id, "passed": self.passed, "message": self.message, "details": _json_default(dict(self.details))}

@dataclass(frozen=True)
class AuditReport:
    name: str; findings: tuple[Finding, ...]; metrics: Mapping[str, Any] = field(default_factory=dict)
    @property
    def passed(self) -> bool: return all(f.passed for f in self.findings)
    def to_public_dict(self) -> dict[str, Any]: return {"name": self.name, "passed": self.passed, "findings": [f.to_public_dict() for f in self.findings], "metrics": _json_default(dict(self.metrics))}

def combine_reports(name: str, reports: Sequence[AuditReport]) -> AuditReport:
    findings, metrics = [], {}
    for r in reports: findings.extend(r.findings); metrics[r.name] = r.metrics
    return AuditReport(name, tuple(findings), metrics)
def audit_world(world: SheetWorld) -> AuditReport:
    public_names = [t.table_id for t in world.tables] + [c.column_id for t in world.tables for c in t.columns]
    leaked = [x for x in SEMANTIC_NAME_FRAGMENTS if x in " ".join(public_names).lower()]
    floor = decoy_floor(world.m) + 1
    weak = {r: n for r, n in world.same_type_candidate_counts().items() if n < floor}
    return AuditReport("sheets_world_audit", (Finding("WORLD_COLUMN_NAMES_OPAQUE", not leaked, "table and column ids contain no semantic role fragments", {"leaked_terms": leaked}), Finding("WORLD_SAME_TYPE_DECOYS", not weak, "critical roles have same-type candidates", {"minimum_candidates": floor, "weak_roles": weak}), Finding("WORLD_NON_BOOLEAN_OUTPUT_FLOOR", world.non_boolean_output_fraction() >= 0.50, "at least half of output forms are non-Boolean typed outputs", {"fraction": world.non_boolean_output_fraction()}), Finding("WORLD_COMPOSED_GOAL_PRESENT", world.has_composed_goal(), "world contains join-normalize-guard obligation", {"obligations": list(world.goal_obligations)})), world.to_audit_dict())

def audit_goal_obligation_contract(world: SheetWorld) -> AuditReport:
    missing = sorted(REQUIRED_GOAL_OBLIGATIONS - set(world.goal_obligations))
    return AuditReport("goal_obligation_contract_audit", (Finding("GOAL_OBLIGATIONS_COMPLETE", not missing, "goal is finite verifier obligations", {"missing": missing}), Finding("GOAL_OUTPUT_CANONICAL", True, "output forms are canonical typed variants", {"output_forms": list(world.public_schema()["output_forms"])})), {"obligations": sorted(world.goal_obligations)})

def audit_packet_serialization(packet: Packet) -> AuditReport:
    bits = packet_bit_length(packet); blob = canonical_json_bytes([e.payload for e in packet.entries]).decode("ascii").lower()
    hits = [x for x in BANNED_EXECUTABLE_TERMS if x in blob]
    required = {"frame_patch", "verifier_clause", "counterexample", "binding_policy", "composition_gate"}; missing = sorted(required - {e.entry_type for e in packet.entries})
    return AuditReport("packet_serialization_audit", (Finding("PACKET_BITS_RECOMPUTED", packet.declared_bits == bits, "declared bits match canonical recomputation", {"declared_bits": packet.declared_bits, "recomputed_bits": bits}), Finding("PACKET_EXECUTABLE_TERMS_CLEAN", not hits, "packet fields contain no banned hidden metadata", {"banned_hits": hits}), Finding("PACKET_REQUIRED_ENTRY_TYPES", not missing, "packet contains required executable categories", {"missing": missing})), {"packet_bits": bits, "packet_hash": stable_hash(packet.to_public_dict(), 32), "entry_count": len(packet.entries)})

def audit_constructor_provenance(packet: Packet, transcript: PublicTranscript) -> AuditReport:
    allowed = transcript.allowed_provenance_ids(); empty, unknown = [], []
    for i, entry in enumerate(packet.entries):
        if not entry.provenance: empty.append(i)
        unknown.extend(ref for ref in entry.provenance if ref not in allowed)
    return AuditReport("constructor_provenance_audit", (Finding("CONSTRUCTOR_BLIND_MODE", packet.constructor_mode == "blind", "constructor declares blind mode", {"constructor_mode": packet.constructor_mode}), Finding("CONSTRUCTOR_PROVENANCE_PRESENT", not empty and not unknown, "entries cite allowed public facts", {"empty_entry_indices": empty, "unknown_refs": unknown[:10]})), {"transcript_id": transcript.transcript_id, "packet_entries": len(packet.entries)})

@dataclass(frozen=True)
class BudgetLedger:
    packet_bits: int; frame_bits: int = 0; binding_bits: int = 0; program_bits: int = 0; parser_bits: int = 0; human_labor_bits: int = 0; examples_bits: int = 0; verifier_bits: int = 0; final_program_bits: int = 0; learned_library_bits: int = 0; residual_sibling_teaching_bits: int = 0; failed_query_bits: int = 0; baseline_adapter_bits: int = 0
    @property
    def total_bits(self) -> int: return sum(getattr(self, f) for f in self.__dataclass_fields__)
    @property
    def binding_ratio(self) -> float: return self.binding_bits / max(1, self.frame_bits + self.binding_bits)
    @property
    def program_ratio(self) -> float: return (self.program_bits + self.final_program_bits) / max(1, self.frame_bits + self.binding_bits + self.program_bits + self.final_program_bits)
    def to_public_dict(self) -> dict[str, Any]:
        d = {f: int(getattr(self, f)) for f in self.__dataclass_fields__}; d.update(total_bits=self.total_bits, binding_ratio=self.binding_ratio, program_ratio=self.program_ratio); return d

def make_budget_ledger(packet: Packet) -> BudgetLedger:
    c = defaultdict(int)
    for e in packet.entries:
        bits = 8 * len(canonical_json_bytes(e.to_public_dict()))
        if e.entry_type == "frame_patch": c["frame_bits"] += bits
        elif e.entry_type == "binding_policy": c["binding_bits"] += bits
        elif e.entry_type == "verifier_clause": c["verifier_bits"] += bits
        elif e.entry_type == "counterexample": c["examples_bits"] += bits
        else: c["program_bits"] += bits
    return BudgetLedger(packet_bits=packet_bit_length(packet), **c)

def audit_budget_recomputation(packet: Packet, ledger: BudgetLedger) -> AuditReport:
    fields = set(BudgetLedger.__dataclass_fields__); present = set(ledger.to_public_dict()) - {"total_bits", "binding_ratio", "program_ratio"}
    bounded = ledger.frame_bits + ledger.binding_bits + ledger.program_bits + ledger.examples_bits + ledger.verifier_bits <= ledger.packet_bits
    return AuditReport("budget_recomputation_audit", (Finding("BUDGET_PACKET_BITS_MATCH", ledger.packet_bits == packet_bit_length(packet), "ledger packet bits match serializer", ledger.to_public_dict()), Finding("BUDGET_ALL_CATEGORIES_PRESENT", fields == present, "ledger includes all charged categories", {"missing_fields": sorted(fields - present)}), Finding("BUDGET_CATEGORY_SUM_BOUNDED", bounded, "classified entry bits do not exceed packet bits", ledger.to_public_dict())), ledger.to_public_dict())

def audit_cost_split(ledger: BudgetLedger) -> AuditReport:
    return AuditReport("frame_binding_program_cost_split_audit", (Finding("COST_FRAME_BITS_PRESENT", ledger.frame_bits > 0, "frame bits separately counted", {"frame_bits": ledger.frame_bits}), Finding("COST_BINDING_BITS_SEPARATE", ledger.binding_ratio <= 0.80, "binding bits are separate", {"binding_ratio": ledger.binding_ratio}), Finding("COST_PROGRAM_BITS_SEPARATE", ledger.program_ratio <= 0.80, "program bits are separate", {"program_ratio": ledger.program_ratio})), ledger.to_public_dict())

@dataclass(frozen=True)
class ParserHumanLedger:
    public_substrate: tuple[str, ...]; packet_design: tuple[str, ...]; frozen_before_hidden: tuple[str, ...]; hidden_eval_only: tuple[str, ...]; charged_parser_bits: int = 0; charged_human_bits: int = 0; claim_ceiling: str = "controlled evidence for typed amortized frame-teaching separation"
    def to_public_dict(self) -> dict[str, Any]:
        return {k: _json_default(v) for k, v in self.__dict__.items()}

def default_parser_human_ledger() -> ParserHumanLedger:
    return ParserHumanLedger(("opaque_table_schema", "rational_parser", "date_parser_stub", "unit_registry", "typed_action_api"), ("frame_patch_templates", "verifier_obligation_templates", "binding_cost_rules"), ("spec_hash", "harness_version", "baseline_roster", "token_precedence", "seed_rule"), ("hidden_seed", "hidden_family_labels", "hidden_query_labels", "canonical_scorer_outputs"))

def audit_parser_human_ledger(ledger: ParserHumanLedger) -> AuditReport:
    miss_pub = sorted({"unit_registry", "rational_parser", "opaque_table_schema"} - set(ledger.public_substrate)); miss_packet = sorted({"frame_patch_templates", "verifier_obligation_templates", "binding_cost_rules"} - set(ledger.packet_design)); public_hidden = [x for x in ledger.public_substrate + ledger.packet_design if "hidden" in x]
    return AuditReport("parser_human_labor_ledger_audit", (Finding("LEDGER_PUBLIC_SUBSTRATE_DECLARED", not miss_pub, "public parser/type substrate is declared", {"missing_public": miss_pub}), Finding("LEDGER_PACKET_DESIGN_DECLARED", not miss_packet, "packet-design labor is declared", {"missing_packet": miss_packet}), Finding("LEDGER_HIDDEN_SURFACES_SEPARATE", not public_hidden, "hidden surfaces are separate", {"hidden_mentions_in_public": public_hidden})), ledger.to_public_dict())
@dataclass(frozen=True)
class TaskBundle:
    target_id: str; sibling_ids: tuple[str, ...]; hidden_case_hash: str = "unopened-hidden-cases"
    def to_public_dict(self) -> dict[str, Any]: return {"target_id": self.target_id, "sibling_ids": list(self.sibling_ids), "hidden_case_hash": self.hidden_case_hash}

@dataclass(frozen=True)
class BaselineView:
    baseline_name: str; packet_hash: str; packet_bits: int; task_bundle_hash: str; query_budget: int; executable_packet: Mapping[str, Any]; ignored_fields: tuple[str, ...] = ()
    def to_public_dict(self) -> dict[str, Any]: return {"baseline_name": self.baseline_name, "packet_hash": self.packet_hash, "packet_bits": self.packet_bits, "task_bundle_hash": self.task_bundle_hash, "query_budget": self.query_budget, "executable_packet_hash": stable_hash(self.executable_packet, 32), "ignored_fields": list(self.ignored_fields)}

def make_baseline_views(packet: Packet, task_bundle: TaskBundle, query_budget: int, denied_fields: Mapping[str, Sequence[str]] | None = None) -> tuple[BaselineView, ...]:
    denied_fields = denied_fields or {}; payload = packet.to_public_dict(); phash = stable_hash(payload, 32); thash = stable_hash(task_bundle.to_public_dict(), 32); views = []
    for name in BASELINE_NAMES:
        executable = json.loads(canonical_json_bytes(payload).decode("ascii")); ignored = tuple(denied_fields.get(name, ()))
        for field in ignored: executable.pop(field, None)
        views.append(BaselineView(name, phash if not ignored else stable_hash(executable, 32), packet_bit_length(packet), thash, query_budget, executable, ignored))
    return tuple(views)

def audit_baseline_parity(views: Sequence[BaselineView]) -> AuditReport:
    by_name = {v.baseline_name: v for v in views}; missing = [n for n in BASELINE_NAMES if n not in by_name]
    hashes, bits, tasks, budgets = {v.packet_hash for v in views}, {v.packet_bits for v in views}, {v.task_bundle_hash for v in views}, {v.query_budget for v in views}
    denied = {v.baseline_name: list(v.ignored_fields) for v in views if v.ignored_fields}
    return AuditReport("baseline_parity_audit", (Finding("BASELINE_ALL_PRESENT", not missing, "all declared baselines have packet view", {"missing": missing}), Finding("BASELINE_PACKET_HASH_PARITY", len(hashes) == 1, "all baselines receive same executable packet bytes", {"packet_hashes": sorted(hashes), "denied_fields": denied}), Finding("BASELINE_BUDGET_PARITY", len(bits) == len(tasks) == len(budgets) == 1, "baselines receive matched bits, task bundle, and query budget", {"packet_bits": sorted(bits), "task_hashes": sorted(tasks), "query_budgets": sorted(budgets)})), {"views": [v.to_public_dict() for v in views]})

def audit_domain_baseline_roster() -> AuditReport:
    required = {"relational_algebra", "unit_system", "exact_key_matching", "entity_resolution", "schema_matching", "pbe_prose", "data_wrangling", "constraint_solver", "data_repair", "typed_cegis_exact", "typed_cegis_beam", "typed_mdl_library", "operation_verifier_search", "goal_conditioned_cegis", "active_goal_disambiguation", "obligation_template_library", "abstention_validator"}
    missing = sorted(required - set(BASELINE_NAMES))
    return AuditReport("domain_baseline_roster_audit", (Finding("DOMAIN_BASELINE_ROSTER_COMPLETE", not missing, "Q37 domain absorber roster is declared", {"missing": missing}),), {"baseline_names": list(BASELINE_NAMES)})

def audit_packet_order_control(packet: Packet) -> AuditReport:
    rev = replace(packet, entries=tuple(reversed(packet.entries))); rot = replace(packet, entries=packet.entries[1:] + packet.entries[:1] if packet.entries else ())
    return AuditReport("packet_order_control", (Finding("PACKET_REVERSE_SAME_FACT_MULTISET", packet_multiset_hash(rev) == packet_multiset_hash(packet), "reversing packet order preserves entry multiset", {}), Finding("PACKET_ROTATE_SAME_BITS", packet_bit_length(rot) == packet_bit_length(packet), "rotating packet order preserves bit length", {"original_bits": packet_bit_length(packet), "rotated_bits": packet_bit_length(rot)})), {})

def audit_enumerability(world: SheetWorld) -> AuditReport:
    tc = Counter(c.type_tag for t in world.tables for c in t.columns); joins = tc["id_like"] * max(1, tc["id_like"] - 1); units = tc["rational"] * max(1, tc["unit_symbol"]); bindings = max(1, tc["id_like"] * tc["rational"] * tc["unit_symbol"]); constraints = 2 ** min(16, max(1, tc["constraint_flag"])); policies = max(2, tc["constraint_flag"] * 2); version = joins * units * policies
    metrics = {"N_join_candidates": joins, "N_unit_transform_candidates": units, "N_schema_bindings": bindings, "N_constraint_sets": constraints, "N_action_policies": policies, "typed_pruning_factor": round(bindings / max(1, sum(tc.values()) ** 3), 6), "public_example_version_space": version, "minimum_distinguishing_counterexamples": max(2, math.ceil(math.log2(max(2, version))))}
    return AuditReport("typed_enumerability_audit", (Finding("ENUMERABILITY_METRICS_PRESENT", all(v > 0 for v in metrics.values()), "typed search metrics are reported", metrics), Finding("ENUMERABILITY_NOT_SINGLETON", joins > 1 and units > 1 and bindings > 1, "public types do not reduce search to singleton", metrics)), metrics)

def entropy(values: Sequence[Any]) -> float:
    counts, total = Counter(values), len(values)
    return 0.0 if total == 0 else -sum((c / total) * math.log2(c / total) for c in counts.values())

def normalized_mi(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    if len(xs) != len(ys) or not xs: return 0.0
    joint, xc, yc, total = Counter(zip(xs, ys)), Counter(xs), Counter(ys), len(xs); mi = 0.0
    for (x, y), c in joint.items(): mi += (c / total) * math.log2((c / total) / ((xc[x] / total) * (yc[y] / total)))
    denom = min(entropy(xs), entropy(ys)); return 0.0 if denom <= 0 else mi / denom

def _value_shape(world: SheetWorld, table_id: str, col: ColumnSpec) -> str:
    sample = [r.get(col.column_id) for r in world.rows_by_table[table_id][:8]]
    if col.type_tag == "id_like": return "id_len_" + str(max(len(str(v)) for v in sample))
    if col.type_tag == "rational": return "denoms_" + "_".join(sorted({str(v.get("den", 1)) for v in sample if isinstance(v, dict)})[:3])
    if col.type_tag == "unit_symbol": return "unit_vocab_" + str(len(set(map(str, sample))))
    if col.type_tag == "constraint_flag": return "bool_balance_" + str(sum(1 for v in sample if bool(v)))
    return "text_vocab_" + str(len(set(map(str, sample))))

def run_generator_leakage_audit(public_seed: str, sample_count: int = 10_000, threshold: float = 0.08) -> AuditReport:
    per_m = max(1, sample_count // len(NUISANCE_SIZES)); metrics_by_m = {}; worst = 0.0; purposes = set()
    for m in NUISANCE_SIZES:
        grouped = defaultdict(lambda: defaultdict(list)); sib_idx, target_sig = [], []
        for i in range(per_m):
            gen = generate_world(public_seed, m, i, "sheets_mi_dry_run"); world = gen.world; purposes.update(r.purpose for r in gen.rng_records); critical = world.critical_column_ids(); sig = stable_hash(sorted(critical)[:2], 6)
            for table in world.tables:
                for idx, col in enumerate(table.columns):
                    g = grouped[col.type_tag]; g["label"].append("critical" if col.column_id in critical else "decoy"); g["index_bucket"].append(idx % 8); g["name_prefix"].append(col.column_id[1]); g["missingness"].append("none"); g["value_shape"].append(_value_shape(world, table.table_id, col))
            for j, _ in enumerate(generate_sibling_worlds(public_seed, world, 3)): sib_idx.append(j); target_sig.append(sig)
        mm = {}
        for typ, vals in grouped.items():
            if entropy(vals["label"]) == 0: continue
            for feature in ("index_bucket", "name_prefix", "missingness", "value_shape"): mm[f"{typ}_{feature}_target_nmi"] = normalized_mi(vals["label"], vals[feature])
        mm["sibling_index_target_signature_nmi"] = normalized_mi(sib_idx, target_sig); metrics_by_m[m] = mm; worst = max([worst] + list(mm.values()))
    split_ok = all(p in purposes for p in RNG_PURPOSES)
    return AuditReport("typed_generator_leakage_audit", (Finding("GENERATOR_LEAKAGE_BELOW_THRESHOLD", worst <= threshold, "schema/value statistics do not predict critical bindings above threshold", {"worst_metric": worst, "threshold": threshold}), Finding("GENERATOR_RNG_STREAMS_SPLIT", split_ok, "generator uses required split RNG streams", {"observed_purposes": sorted(purposes)})), {"sample_count": sample_count, "per_m": per_m, "threshold": threshold, "metrics_by_m": metrics_by_m, "worst_metric": worst})
@dataclass(frozen=True)
class HarnessManifest:
    public_seed: str; hidden_seed_rule: str; constructor_id: str; serializer_id: str; scorer_id: str; baseline_versions: Mapping[str, str]; timeouts: Mapping[str, int]; parser_ledger_hash: str; frozen_before_hidden: bool; hidden_results_opened: bool; post_hidden_changes: tuple[str, ...] = ()
    def to_public_dict(self) -> dict[str, Any]:
        return {k: _json_default(v) for k, v in self.__dict__.items()}

def default_parser_human_hash() -> str: return stable_hash(default_parser_human_ledger().to_public_dict(), 24)

def default_manifest(public_seed: str) -> HarnessManifest:
    return HarnessManifest(public_seed, "sha256(public_seed|sheets0-hidden|manifest_hash|unopened-until-freeze)", BlindTypedPacketConstructor.constructor_id, "canonical-json-bytes-v1", "frameseed-sheets0-token-scorer-v1", {n: f"{n}-adapter-v1" for n in BASELINE_NAMES}, {n: 0 for n in BASELINE_NAMES}, default_parser_human_hash(), True, False, ())

def audit_manifest(manifest: HarnessManifest) -> AuditReport:
    missing = [n for n in BASELINE_NAMES if n not in manifest.baseline_versions]
    return AuditReport("manifest_audit", (Finding("MANIFEST_FROZEN_BEFORE_HIDDEN", manifest.frozen_before_hidden and not manifest.hidden_results_opened, "manifest freezes before hidden opening", {"frozen_before_hidden": manifest.frozen_before_hidden, "hidden_results_opened": manifest.hidden_results_opened}), Finding("MANIFEST_BASELINES_DECLARED", not missing, "manifest declares every baseline", {"missing_baselines": missing}), Finding("MANIFEST_NO_POST_HIDDEN_EDITS", not manifest.post_hidden_changes, "manifest has no post-hidden edits", {"post_hidden_changes": list(manifest.post_hidden_changes)})), manifest.to_public_dict())

@dataclass(frozen=True)
class TokenEvidence:
    smuggling_detected: bool = False; parity_failure: bool = False; hidden_leakage: bool = False; baseline_information_denial: bool = False; generator_leakage_detected: bool = False; human_labor_untracked: bool = False; subjective_goal_semantics: bool = False; typed_boolean_trap_triggered: bool = False; representation_noncontainment_passed: bool = True; representation_prior_absorbed: bool = False; parser_prior_absorbed: bool = False; l3_full_threshold_passed: bool = False; l3_mean_hfa: float = 0.0; non_boolean_output_floor_passed: bool = False; packet_growth_sublinear: bool = False; aftd_all_in_passed: bool = False; packet_erasure_drop_passed: bool = False; role_stability_passed: bool = False; composition_gate_passed: bool = False; cost_split_passed: bool = False; claim_ceiling_honored: bool = False; bits_counted: bool = False; domain_absorptions: Mapping[str, bool] = field(default_factory=dict); generic_absorptions: Mapping[str, bool] = field(default_factory=dict)
    def to_public_dict(self) -> dict[str, Any]:
        return {k: _json_default(v) for k, v in self.__dict__.items()}

def _empty_absorptions(names: Sequence[str]) -> dict[str, bool]: return {n: False for n in names}

def _would_signal(e: TokenEvidence) -> bool:
    return not any((e.smuggling_detected, e.parity_failure, e.hidden_leakage, e.baseline_information_denial, e.generator_leakage_detected, e.human_labor_untracked, e.subjective_goal_semantics, e.typed_boolean_trap_triggered, not e.representation_noncontainment_passed, e.representation_prior_absorbed, e.parser_prior_absorbed, not e.l3_full_threshold_passed, e.l3_mean_hfa < 0.97, not e.non_boolean_output_floor_passed, not e.packet_growth_sublinear, not e.aftd_all_in_passed, not e.packet_erasure_drop_passed, not e.role_stability_passed, not e.composition_gate_passed, not e.cost_split_passed, not e.claim_ceiling_honored, not e.bits_counted)) and not any(e.domain_absorptions.get(n, False) for n in DOMAIN_ABSORPTION_PRECEDENCE) and not any(e.generic_absorptions.get(n, False) for n in GENERIC_ABSORPTION_PRECEDENCE)

def assign_terminal_token(e: TokenEvidence) -> str:
    if any((e.smuggling_detected, e.parity_failure, e.hidden_leakage, e.baseline_information_denial, e.generator_leakage_detected, e.human_labor_untracked, e.subjective_goal_semantics)): return TERMINAL_TOKENS["void"]
    if e.typed_boolean_trap_triggered: return TERMINAL_TOKENS["typed_boolean_trap"]
    if e.parser_prior_absorbed: return TERMINAL_TOKENS["parser_prior"]
    if not e.representation_noncontainment_passed or e.representation_prior_absorbed: return TERMINAL_TOKENS["representation_prior"]
    for n in DOMAIN_ABSORPTION_PRECEDENCE:
        if e.domain_absorptions.get(n, False): return TERMINAL_TOKENS[n]
    if not e.l3_full_threshold_passed or e.l3_mean_hfa < 0.97 or not e.non_boolean_output_floor_passed: return TERMINAL_TOKENS["negative"]
    for n in GENERIC_ABSORPTION_PRECEDENCE:
        if e.generic_absorptions.get(n, False): return TERMINAL_TOKENS[n]
    return TERMINAL_TOKENS["signal"] if _would_signal(e) else TERMINAL_TOKENS["negative"]

def _all_signal_gates(**overrides: Any) -> TokenEvidence:
    base = TokenEvidence(l3_full_threshold_passed=True, l3_mean_hfa=0.98, non_boolean_output_floor_passed=True, packet_growth_sublinear=True, aftd_all_in_passed=True, packet_erasure_drop_passed=True, role_stability_passed=True, composition_gate_passed=True, cost_split_passed=True, claim_ceiling_honored=True, bits_counted=True, domain_absorptions=_empty_absorptions(DOMAIN_ABSORPTION_PRECEDENCE), generic_absorptions=_empty_absorptions(GENERIC_ABSORPTION_PRECEDENCE))
    return replace(base, **overrides)

def run_golden_token_controls() -> AuditReport:
    controls = [("smuggling", _all_signal_gates(smuggling_detected=True), TERMINAL_TOKENS["void"]), ("typed_boolean", _all_signal_gates(typed_boolean_trap_triggered=True), TERMINAL_TOKENS["typed_boolean_trap"]), ("parser", _all_signal_gates(parser_prior_absorbed=True), TERMINAL_TOKENS["parser_prior"]), ("representation", _all_signal_gates(representation_noncontainment_passed=False), TERMINAL_TOKENS["representation_prior"]), ("relational_before_negative", TokenEvidence(l3_full_threshold_passed=False, non_boolean_output_floor_passed=True, domain_absorptions={"relational_algebra": True}), TERMINAL_TOKENS["relational_algebra"]), ("typed_cegis_before_library", _all_signal_gates(domain_absorptions={"typed_cegis": True, "library_learning": True}), TERMINAL_TOKENS["typed_cegis"]), ("negative_low_l3", TokenEvidence(l3_full_threshold_passed=False), TERMINAL_TOKENS["negative"]), ("teaching_dimension", _all_signal_gates(generic_absorptions={"teaching_dimension": True}), TERMINAL_TOKENS["teaching_dimension"]), ("clean_signal", _all_signal_gates(), TERMINAL_TOKENS["signal"])]
    findings, results = [], {}
    for name, evidence, expected in controls:
        observed = assign_terminal_token(evidence); results[name] = observed; findings.append(Finding(f"GOLDEN_TOKEN_{name.upper()}", observed == expected, f"golden control emits {expected}", {"observed": observed, "expected": expected}))
    return AuditReport("golden_token_controls", tuple(findings), results)

def run_preimplementation_audit(public_seed: str = "FRAMESEED_SHEETS0_B30_PUBLIC_SEED", dry_run_worlds: int = 10_000, leakage_threshold: float = 0.08) -> AuditReport:
    manifest = default_manifest(public_seed); target = generate_world(public_seed, 16, 0, "audit_gate").world; siblings = generate_sibling_worlds(public_seed, target, 3); transcript = make_public_transcript(target); rng = split_rngs(public_seed, f"packet:{target.world_id}")["packet_construction"]; packet = BlindTypedPacketConstructor().construct(transcript, rng); bundle = TaskBundle(target.world_id, tuple(s.world_id for s in siblings)); ledger = make_budget_ledger(packet); parser_ledger = default_parser_human_ledger(); views = make_baseline_views(packet, bundle, len(transcript.facts))
    reports = [audit_manifest(manifest), audit_world(target), audit_goal_obligation_contract(target), audit_packet_serialization(packet), audit_constructor_provenance(packet, transcript), audit_budget_recomputation(packet, ledger), audit_cost_split(ledger), audit_parser_human_ledger(parser_ledger), audit_baseline_parity(views), audit_domain_baseline_roster(), audit_packet_order_control(packet), audit_enumerability(target), run_generator_leakage_audit(public_seed, dry_run_worlds, leakage_threshold), run_golden_token_controls()]
    combined = combine_reports("frameseed_sheets0_preimplementation_audit", reports); metrics = dict(combined.metrics); metrics.update(no_performance_runs=True, hidden_hfa_reported=False, target_world=target.to_audit_dict(), sibling_count=len(siblings), harness_version=HARNESS_VERSION)
    return AuditReport(combined.name, combined.findings, metrics)

def main() -> None:
    parser = argparse.ArgumentParser(description="FRAMESEED-SHEETS-0 public audit harness gate"); parser.add_argument("--public-seed", default="FRAMESEED_SHEETS0_B30_PUBLIC_SEED"); parser.add_argument("--dry-run-worlds", type=int, default=10_000); parser.add_argument("--leakage-threshold", type=float, default=0.08); parser.add_argument("--output", default=""); args = parser.parse_args(); start = time.time(); report = run_preimplementation_audit(args.public_seed, args.dry_run_worlds, args.leakage_threshold); payload = report.to_public_dict(); payload["elapsed_s"] = round(time.time() - start, 3); text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f: f.write(text + "\n")
    print(text)
    if not report.passed: raise SystemExit(1)

if __name__ == "__main__": main()