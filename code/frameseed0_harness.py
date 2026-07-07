"""FRAMESEED-0 pre-implementation audit harness.

Harness-first scope only: audited RNG streams, dry-run world generation, blind
packet construction with provenance, canonical serialization and budget checks,
baseline parity, smuggling controls, generator MI audits, and terminal-token
golden controls. This module does not optimize L3, run hidden HFA, or report a
learned-performance signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

HARNESS_VERSION = "frameseed0-audit-harness-v1"
NUISANCE_SIZES = (4, 16, 64, 256)
RNG_PURPOSES = (
    "world_structure",
    "names",
    "orientations",
    "hidden_queries",
    "packet_construction",
    "learner_tie_breaks",
    "baseline_tie_breaks",
    "ablations",
)
BASELINE_NAMES = (
    "l3_full",
    "td_h0",
    "l0_rotenn",
    "l1_active",
    "l2_cegis",
    "rag",
    "nuisance_oracle",
    "library_learning",
)
ABSORPTION_PRECEDENCE = (
    "teaching_dimension",
    "library_learning",
    "nuisance_oracle",
    "cegis",
    "active_learning",
    "rag",
)
TERMINAL_TOKENS = {
    "signal": "FRAMESEED_T3R_SIGNAL",
    "teaching_dimension": "FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION",
    "representation_prior": "FRAMESEED_T3_ABSORBED_BY_REPRESENTATION_PRIOR",
    "nuisance_oracle": "FRAMESEED_T3_ABSORBED_BY_NUISANCE_ORACLE",
    "library_learning": "FRAMESEED_T3_ABSORBED_BY_LIBRARY_LEARNING",
    "active_learning": "FRAMESEED_T3_ABSORBED_BY_ACTIVE_LEARNING",
    "cegis": "FRAMESEED_T3_ABSORBED_BY_CEGIS",
    "rag": "FRAMESEED_T3_ABSORBED_BY_RAG",
    "boolean_trap": "FRAMESEED_T3_BOOLEAN_TRAP",
    "void": "FRAMESEED_T3_VOID_SMUGGLED_FRAME",
    "negative": "FRAMESEED_T3_NEGATIVE",
}
BANNED_EXECUTABLE_TERMS = (
    "causal",
    "spurious",
    "nuisance",
    "alias",
    "true_role",
    "target_kernel",
    "hidden_family",
    "family_id",
    "rho",
    "beta",
    "pi",
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_public_dict"):
        return value.to_public_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_default(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_json_default(v) for v in value]
    if isinstance(value, list):
        return [_json_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_default(v) for k, v in sorted(value.items())}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_default(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def stable_hash(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()[:length]


def derive_seed(public_seed: str, purpose: str, namespace: str) -> int:
    digest = hashlib.sha256(f"{public_seed}|{purpose}|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class RNGStreamRecord:
    purpose: str
    namespace: str
    seed_hash: str
    draw_count: int


class AuditedRandom:
    def __init__(self, public_seed: str, purpose: str, namespace: str):
        self.purpose = purpose
        self.namespace = namespace
        self.seed = derive_seed(public_seed, purpose, namespace)
        self._rng = random.Random(self.seed)
        self.draw_count = 0

    def _count(self, n: int = 1) -> None:
        self.draw_count += n

    def randrange(self, stop: int) -> int:
        self._count()
        return self._rng.randrange(stop)

    def getrandbits(self, bits: int) -> int:
        self._count()
        return self._rng.getrandbits(bits)

    def choice(self, items: Sequence[Any]) -> Any:
        self._count()
        return self._rng.choice(items)

    def shuffle(self, items: list[Any]) -> None:
        self._count(max(len(items) - 1, 1))
        self._rng.shuffle(items)

    def record(self) -> RNGStreamRecord:
        seed_hash = hashlib.sha256(str(self.seed).encode("ascii")).hexdigest()[:16]
        return RNGStreamRecord(self.purpose, self.namespace, seed_hash, self.draw_count)


def split_rngs(public_seed: str, namespace: str) -> dict[str, AuditedRandom]:
    return {purpose: AuditedRandom(public_seed, purpose, namespace) for purpose in RNG_PURPOSES}


def _pair_index(a: int, b: int) -> int:
    return (int(a) << 1) | int(b)


def _pair_bits(index: int) -> tuple[int, int]:
    return ((index >> 1) & 1, index & 1)


def _kernel_depends_on_both(table: tuple[int, int, int, int]) -> bool:
    depends_on_a = any(table[_pair_index(0, b)] != table[_pair_index(1, b)] for b in (0, 1))
    depends_on_b = any(table[_pair_index(a, 0)] != table[_pair_index(a, 1)] for a in (0, 1))
    return len(set(table)) > 1 and depends_on_a and depends_on_b


ADMITTED_KERNELS = tuple(
    table
    for i in range(16)
    for table in ((i & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1),)
    if _kernel_depends_on_both(table)
)


def _kernel_id(table: tuple[int, int, int, int]) -> str:
    return "K" + "".join(str(bit) for bit in table)


def _role_category(role: str) -> str:
    if role.startswith("c"):
        return "effect"
    if role.startswith("s"):
        return "paired"
    return "background"


@dataclass(frozen=True)
class Edit:
    slot: int
    value: int

    def to_public_dict(self) -> dict[str, int]:
        return {"slot": self.slot, "value": self.value}


@dataclass(frozen=True)
class Query:
    observation: tuple[int, ...]
    edits: tuple[Edit, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {"observation": list(self.observation), "edits": [edit.to_public_dict() for edit in self.edits]}


@dataclass(frozen=True)
class World:
    world_id: str
    m: int
    kernel: tuple[int, int, int, int]
    rho: tuple[int, int, int, int]
    orientations: tuple[int, ...]
    slot_to_role: tuple[str, ...]
    names: tuple[str, ...]
    seed_namespace: str
    family_class: str = "dry_run"

    @property
    def d(self) -> int:
        return self.m + 4

    @property
    def kernel_id(self) -> str:
        return _kernel_id(self.kernel)

    def role_to_slot(self) -> dict[str, int]:
        return {role: idx for idx, role in enumerate(self.slot_to_role)}

    def public_schema(self) -> dict[str, Any]:
        return {
            "schema_version": "frameseed0-public-schema-v1",
            "slot_count": self.d,
            "slot_type": "bit",
            "intervention_grammar": ("none", "set_one", "compose"),
            "query_grammar": "surface_observation_plus_set_edits",
            "label_type": "bit",
            "slot_names": list(self.names),
        }

    def label_from_effect_bits(self, a: int, b: int) -> int:
        return int(self.kernel[_pair_index(a, b)])

    def make_observation(self, c0: int, c1: int, background_bits: Sequence[int] | None = None) -> tuple[int, ...]:
        if background_bits is None:
            background_bits = [0] * self.m
        if len(background_bits) != self.m:
            raise ValueError(f"expected {self.m} background bits")
        s0, s1 = _pair_bits(self.rho[_pair_index(c0, c1)])
        latent = {"c0": int(c0), "c1": int(c1), "s0": int(s0), "s1": int(s1)}
        for i, bit in enumerate(background_bits):
            latent[f"n{i}"] = int(bit)
        surface = [0] * self.d
        for slot, role in enumerate(self.slot_to_role):
            surface[slot] = latent[role] ^ self.orientations[slot]
        return tuple(surface)

    def label_query(self, query: Query) -> int:
        obs = list(query.observation)
        edits = {edit.slot: edit.value for edit in query.edits}
        role_to_slot = self.role_to_slot()
        values = {}
        for role in ("c0", "c1"):
            slot = role_to_slot[role]
            values[role] = int(edits.get(slot, obs[slot])) ^ self.orientations[slot]
        return self.label_from_effect_bits(values["c0"], values["c1"])

    def decisive_intervention_exists(self) -> bool:
        role_to_slot = self.role_to_slot()
        c_slot = role_to_slot["c0"]
        p_slot = role_to_slot["s0"]
        for c0 in (0, 1):
            for c1 in (0, 1):
                obs = self.make_observation(c0, c1, [0] * self.m)
                base = self.label_query(Query(obs))
                c_edit = Query(obs, (Edit(c_slot, 1 - obs[c_slot]),))
                p_edit = Query(obs, (Edit(p_slot, 1 - obs[p_slot]),))
                if self.label_query(c_edit) != base and self.label_query(p_edit) == base:
                    return True
        return False

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "m": self.m,
            "d": self.d,
            "kernel_id": self.kernel_id,
            "family_class": self.family_class,
            "schema_hash": stable_hash(self.public_schema()),
        }


@dataclass(frozen=True)
class GeneratedWorld:
    world: World
    rng_records: tuple[RNGStreamRecord, ...]


def generate_world(public_seed: str, m: int, world_index: int, family_class: str = "dry_run") -> GeneratedWorld:
    namespace = f"{family_class}:m={m}:world={world_index}"
    rngs = split_rngs(public_seed, namespace)
    structure_rng = rngs["world_structure"]
    name_rng = rngs["names"]
    orientation_rng = rngs["orientations"]
    kernel = structure_rng.choice(ADMITTED_KERNELS)
    rho = list(range(4))
    structure_rng.shuffle(rho)
    slot_to_role = ["c0", "c1", "s0", "s1"] + [f"n{i}" for i in range(m)]
    structure_rng.shuffle(slot_to_role)
    orientations = tuple(orientation_rng.randrange(2) for _ in range(m + 4))
    names = tuple(f"x{name_rng.getrandbits(96):024x}" for _ in range(m + 4))
    world_id = stable_hash(
        {"namespace": namespace, "kernel": kernel, "rho": rho, "slot_to_role": slot_to_role, "orientations": orientations, "names": names},
        length=20,
    )
    world = World(world_id, m, tuple(kernel), tuple(rho), orientations, tuple(slot_to_role), names, namespace, family_class)
    return GeneratedWorld(world, tuple(r.record() for r in rngs.values()))


def generate_sibling_worlds(public_seed: str, target: World, count: int = 2) -> tuple[World, ...]:
    siblings = []
    for idx in range(count):
        sibling = generate_world(public_seed, target.m, idx, f"sibling_of_{target.world_id[:8]}").world
        if sibling.kernel == target.kernel:
            kernel_index = (ADMITTED_KERNELS.index(sibling.kernel) + 1) % len(ADMITTED_KERNELS)
            sibling = replace(sibling, kernel=ADMITTED_KERNELS[kernel_index])
        siblings.append(sibling)
    return tuple(siblings)
@dataclass(frozen=True)
class PublicFact:
    fact_id: str
    query: Query
    label: int
    source: str

    def to_public_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "query": self.query.to_public_dict(), "label": self.label, "source": self.source}


@dataclass(frozen=True)
class PublicTranscript:
    schema: Mapping[str, Any]
    facts: tuple[PublicFact, ...]
    schema_fact_id: str
    transcript_id: str

    def allowed_provenance_ids(self) -> set[str]:
        return {self.schema_fact_id} | {fact.fact_id for fact in self.facts}

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema": dict(self.schema),
            "schema_fact_id": self.schema_fact_id,
            "facts": [fact.to_public_dict() for fact in self.facts],
            "transcript_id": self.transcript_id,
        }


def make_public_transcript(world: World, rounds_per_cell: int = 2) -> PublicTranscript:
    facts: list[PublicFact] = []
    schema = world.public_schema()
    schema_fact_id = "schema:" + stable_hash(schema, length=20)
    for c0 in (0, 1):
        for c1 in (0, 1):
            for round_idx in range(rounds_per_cell):
                background = [((i + c0 + 2 * c1 + round_idx) % 2) for i in range(world.m)]
                obs = world.make_observation(c0, c1, background)
                base_query = Query(obs)
                base_label = world.label_query(base_query)
                base_fact = PublicFact(
                    "fact:" + stable_hash({"query": base_query, "label": base_label, "kind": "base"}, length=24),
                    base_query,
                    base_label,
                    "public_oracle",
                )
                facts.append(base_fact)
                for slot, bit in enumerate(obs):
                    edit_query = Query(obs, (Edit(slot, 1 - bit),))
                    edit_label = world.label_query(edit_query)
                    payload = {"query": edit_query, "label": edit_label, "kind": "single_set"}
                    facts.append(PublicFact("fact:" + stable_hash(payload, length=24), edit_query, edit_label, "public_oracle"))
    transcript_id = "transcript:" + stable_hash({"schema": schema_fact_id, "facts": [f.fact_id for f in facts]}, length=24)
    return PublicTranscript(schema, tuple(facts), schema_fact_id, transcript_id)


@dataclass(frozen=True)
class PacketEntry:
    entry_type: str
    payload: Mapping[str, Any]
    provenance: tuple[str, ...]

    def to_public_dict(self) -> dict[str, Any]:
        return {"entry_type": self.entry_type, "payload": _json_default(dict(self.payload)), "provenance": list(self.provenance)}


@dataclass(frozen=True)
class Packet:
    header: Mapping[str, Any]
    entries: tuple[PacketEntry, ...]
    constructor_id: str
    constructor_mode: str = "blind"
    declared_bits: int | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "header": _json_default(dict(self.header)),
            "entries": [entry.to_public_dict() for entry in self.entries],
            "constructor_id": self.constructor_id,
            "constructor_mode": self.constructor_mode,
            "declared_bits": self.declared_bits,
        }


def packet_bytes(packet: Packet) -> bytes:
    return canonical_json_bytes({"header": dict(packet.header), "entries": [e.to_public_dict() for e in packet.entries], "constructor_id": packet.constructor_id, "constructor_mode": packet.constructor_mode})


def packet_bit_length(packet: Packet) -> int:
    return 8 * len(packet_bytes(packet))


def packet_multiset_hash(packet: Packet) -> str:
    entries = sorted(stable_hash(entry.to_public_dict(), length=32) for entry in packet.entries)
    return stable_hash({"header": dict(packet.header), "entries": entries}, length=32)


def _assert_transcript_is_public(transcript: PublicTranscript) -> None:
    payload = canonical_json_bytes(transcript).decode("ascii").lower()
    forbidden = ("slot_to_role", "role_to_slot", "kernel_id", "rho", "beta", "pi", "seed_namespace", "hidden_label")
    found = [term for term in forbidden if term in payload]
    if found:
        raise ValueError(f"transcript contains non-public fields: {found}")


class BlindPacketConstructor:
    constructor_id = "blind-public-transcript-effect-support-v1"

    def construct(self, transcript: PublicTranscript, rng: AuditedRandom) -> Packet:
        _assert_transcript_is_public(transcript)
        base_labels = {fact.query.observation: fact.label for fact in transcript.facts if not fact.query.edits}
        effect_counts: dict[int, int] = defaultdict(int)
        effect_provenance: dict[int, list[str]] = defaultdict(list)
        inert_slots: set[int] = set(range(int(transcript.schema["slot_count"])))
        for fact in transcript.facts:
            if len(fact.query.edits) != 1:
                continue
            before = base_labels.get(fact.query.observation)
            if before is None:
                continue
            slot = fact.query.edits[0].slot
            if fact.label != before:
                effect_counts[slot] += 1
                effect_provenance[slot].append(fact.fact_id)
                inert_slots.discard(slot)
        support_slots = sorted(effect_counts, key=lambda s: (-effect_counts[s], s))[:2]
        support_provenance = tuple(fid for slot in support_slots for fid in effect_provenance.get(slot, [])[:2]) or (transcript.schema_fact_id,)
        entries: list[PacketEntry] = []
        example_by_pair: dict[tuple[int, int], PublicFact] = {}
        if len(support_slots) == 2:
            for fact in transcript.facts:
                if fact.query.edits:
                    continue
                pair = (fact.query.observation[support_slots[0]], fact.query.observation[support_slots[1]])
                example_by_pair.setdefault(pair, fact)
            for pair, fact in sorted(example_by_pair.items()):
                entries.append(PacketEntry("example", {"obs_mask": [[support_slots[0], pair[0]], [support_slots[1], pair[1]]], "intervention": [], "label": fact.label}, (fact.fact_id,)))
        for slot in support_slots:
            for fact_id in effect_provenance.get(slot, [])[:1]:
                fact = next(f for f in transcript.facts if f.fact_id == fact_id)
                entries.append(PacketEntry(
                    "intervention_example",
                    {"base_mask": [[idx, bit] for idx, bit in enumerate(fact.query.observation) if idx in support_slots], "edit": fact.query.edits[0].to_public_dict(), "label_before": base_labels[fact.query.observation], "label_after": fact.label},
                    (fact_id,),
                ))
        inert_list = sorted(inert_slots)
        if len(inert_list) >= 2 and support_slots:
            entries.append(PacketEntry("counterexample", {"candidate_program_ast": {"op": "truth_table2", "slots": inert_list[:2], "table": [0, 1, 1, 0]}, "query_kind": "single_set_check", "expected_label_source": "public_fact"}, support_provenance[:2]))
        if support_slots:
            entries.append(PacketEntry("invariant", {"transform_schema": {"op": "set_any", "slot_set": {"op": "set_complement", "slots": list(support_slots)}, "bit": "either"}, "context_schema": "listed_masks", "relation": "output_unchanged"}, support_provenance))
            entries.append(PacketEntry("representation_patch", {"kind": "intervention_generator", "ast_or_schema": {"op": "paired_edit_effect_support_v1", "support_slots": list(support_slots)}, "declared_cost": "canonical_packet_bits", "admissibility_scope": {"slot_count": transcript.schema["slot_count"], "grammar": "single_set_edits"}}, support_provenance))
            entries.append(PacketEntry("verifier_clause", {"clause_id": "finite_effect_support_consistency", "finite_scope": "listed_public_facts", "required_relation": "agree_with_examples_and_invariants"}, support_provenance))
        rng.shuffle(entries)
        header = {"version": "frameseed0-packet-v1", "schema_hash": stable_hash(transcript.schema, length=24), "surface_slot_count": transcript.schema["slot_count"], "l0_hash": "L0-min-public-v1", "h0_hash": "H0-no-representation-patch-v1"}
        packet = Packet(header, tuple(entries), self.constructor_id, "blind")
        return replace(packet, declared_bits=packet_bit_length(packet))
@dataclass(frozen=True)
class Finding:
    check_id: str
    passed: bool
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "passed": self.passed, "message": self.message, "details": _json_default(dict(self.details))}


@dataclass(frozen=True)
class AuditReport:
    name: str
    findings: tuple[Finding, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(finding.passed for finding in self.findings)

    def to_public_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "findings": [f.to_public_dict() for f in self.findings], "metrics": _json_default(dict(self.metrics))}


def combine_reports(name: str, reports: Sequence[AuditReport]) -> AuditReport:
    findings: list[Finding] = []
    metrics: dict[str, Any] = {}
    for report in reports:
        findings.extend(report.findings)
        metrics[report.name] = report.metrics
    return AuditReport(name, tuple(findings), metrics)


def audit_world(world: World) -> AuditReport:
    lower_names = " ".join(world.names).lower()
    leaked_terms = [term for term in BANNED_EXECUTABLE_TERMS if term in lower_names]
    return AuditReport("world_audit", (
        Finding("WORLD_DECISIVE_INTERVENTION", world.decisive_intervention_exists(), "world has a decisive public intervention", world.to_audit_dict()),
        Finding("WORLD_NAME_NO_ROLE_WORDS", not leaked_terms, "surface names contain no banned role words", {"leaked_terms": leaked_terms}),
    ), world.to_audit_dict())


def _base_label_map(transcript: PublicTranscript) -> dict[tuple[int, ...], int]:
    return {fact.query.observation: fact.label for fact in transcript.facts if not fact.query.edits}


def _supported_slots_from_transcript(transcript: PublicTranscript) -> dict[int, set[str]]:
    base_labels = _base_label_map(transcript)
    support: dict[int, set[str]] = defaultdict(set)
    for fact in transcript.facts:
        if len(fact.query.edits) != 1:
            continue
        before = base_labels.get(fact.query.observation)
        if before is not None and fact.label != before:
            support[fact.query.edits[0].slot].add(fact.fact_id)
    return support


def list_packet_atoms(packet: Packet) -> dict[str, Any]:
    slots: list[int] = []
    opcodes: list[str] = []

    def walk(value: Any, key: str = "") -> None:
        if key == "slot" and isinstance(value, int):
            slots.append(value)
        if key == "slots" and isinstance(value, list):
            slots.extend(int(v) for v in value if isinstance(v, int))
        if key in {"entry_type", "op", "kind", "relation"} and isinstance(value, str):
            opcodes.append(value)
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)

    walk(packet.to_public_dict())
    return {"slot_ids": sorted(set(slots)), "opcodes": sorted(set(opcodes))}


def audit_packet_serialization(packet: Packet) -> AuditReport:
    recomputed_bits = packet_bit_length(packet)
    executable_blob = canonical_json_bytes([entry.to_public_dict()["payload"] for entry in packet.entries]).decode("ascii").lower()
    banned_hits = [term for term in BANNED_EXECUTABLE_TERMS if term in executable_blob]
    return AuditReport("packet_serialization_audit", (
        Finding("PACKET_BITS_RECOMPUTED", packet.declared_bits == recomputed_bits, "declared packet bits match canonical serializer recomputation", {"declared_bits": packet.declared_bits, "recomputed_bits": recomputed_bits}),
        Finding("PACKET_EXECUTABLE_TERMS_CLEAN", not banned_hits, "executable packet fields contain no banned role or hidden metadata terms", {"banned_hits": banned_hits}),
    ), {"packet_bits": recomputed_bits, "packet_hash": stable_hash(packet.to_public_dict(), length=32), "atoms": list_packet_atoms(packet), "entry_count": len(packet.entries)})


def audit_constructor_provenance(packet: Packet, transcript: PublicTranscript) -> AuditReport:
    allowed = transcript.allowed_provenance_ids()
    support_by_slot = _supported_slots_from_transcript(transcript)
    unknown_refs: list[str] = []
    empty_refs: list[int] = []
    for idx, entry in enumerate(packet.entries):
        if not entry.provenance:
            empty_refs.append(idx)
        unknown_refs.extend(ref for ref in entry.provenance if ref not in allowed)
    unsupported_slots: list[int] = []
    for entry in packet.entries:
        if entry.entry_type != "representation_patch":
            continue
        schema = dict(entry.payload).get("ast_or_schema", {})
        for slot in schema.get("support_slots", []):
            if not support_by_slot.get(int(slot), set()).intersection(entry.provenance):
                unsupported_slots.append(int(slot))
    return AuditReport("constructor_provenance_audit", (
        Finding("CONSTRUCTOR_BLIND_MODE", packet.constructor_mode == "blind", "packet constructor declares blind mode", {"constructor_mode": packet.constructor_mode}),
        Finding("CONSTRUCTOR_PROVENANCE_PRESENT", not empty_refs and not unknown_refs, "every packet entry cites allowed public transcript facts", {"empty_entry_indices": empty_refs, "unknown_refs": unknown_refs[:10]}),
        Finding("CONSTRUCTOR_SUPPORT_FROM_PROVENANCE", not unsupported_slots, "representation-patch support slots are justified by label-changing public facts", {"unsupported_slots": unsupported_slots}),
    ), {"transcript_id": transcript.transcript_id, "packet_entries": len(packet.entries)})


def make_support_swap_sabotage(packet: Packet, transcript: PublicTranscript) -> Packet:
    support = _supported_slots_from_transcript(transcript)
    inert = sorted(set(range(int(transcript.schema["slot_count"]))) - set(support))
    if len(inert) < 2:
        raise ValueError("not enough inert slots for sabotage")
    new_entries: list[PacketEntry] = []
    for entry in packet.entries:
        payload = json.loads(canonical_json_bytes(entry.payload).decode("ascii"))
        if entry.entry_type == "representation_patch" and "support_slots" in payload.get("ast_or_schema", {}):
            old_len = len(payload["ast_or_schema"]["support_slots"])
            payload["ast_or_schema"]["support_slots"] = inert[:old_len]
        new_entries.append(PacketEntry(entry.entry_type, payload, entry.provenance))
    sabotaged = replace(packet, entries=tuple(new_entries), declared_bits=None)
    return replace(sabotaged, declared_bits=packet_bit_length(sabotaged))


def audit_sabotage_control(packet: Packet, transcript: PublicTranscript) -> AuditReport:
    sabotaged = make_support_swap_sabotage(packet, transcript)
    provenance_report = audit_constructor_provenance(sabotaged, transcript)
    return AuditReport("sabotage_control", (
        Finding("SABOTAGE_SUPPORT_SWAP_DETECTED", not provenance_report.passed, "swapping support slots to inert slots is rejected", provenance_report.to_public_dict()),
    ), {"sabotaged_packet_hash": stable_hash(sabotaged.to_public_dict(), length=32)})


@dataclass(frozen=True)
class BudgetLedger:
    packet_bits: int
    oracle_query_bits: int = 0
    oracle_answer_bits: int = 0
    final_program_bits: int = 0
    learned_library_bits: int = 0
    residual_sibling_teaching_bits: int = 0
    failed_query_bits: int = 0
    verifier_expansion_bits: int = 0

    @property
    def total_bits(self) -> int:
        return sum(getattr(self, field_name) for field_name in self.__dataclass_fields__)

    def to_public_dict(self) -> dict[str, int]:
        data = {field_name: int(getattr(self, field_name)) for field_name in self.__dataclass_fields__}
        data["total_bits"] = self.total_bits
        return data


def audit_budget_recomputation(packet: Packet, ledger: BudgetLedger) -> AuditReport:
    recomputed_packet_bits = packet_bit_length(packet)
    required_fields = set(BudgetLedger.__dataclass_fields__.keys())
    present_fields = set(ledger.to_public_dict().keys()) - {"total_bits"}
    return AuditReport("budget_recomputation_audit", (
        Finding("BUDGET_PACKET_BITS_MATCH", ledger.packet_bits == recomputed_packet_bits, "budget ledger packet bits match canonical packet serializer", {"ledger_packet_bits": ledger.packet_bits, "recomputed_packet_bits": recomputed_packet_bits}),
        Finding("BUDGET_ALL_CATEGORIES_PRESENT", required_fields == present_fields, "budget ledger includes all charged categories", {"missing_fields": sorted(required_fields - present_fields)}),
    ), ledger.to_public_dict())


@dataclass(frozen=True)
class TaskBundle:
    target_id: str
    sibling_ids: tuple[str, ...]
    hidden_case_hash: str = "unopened-hidden-cases"

    def to_public_dict(self) -> dict[str, Any]:
        return {"target_id": self.target_id, "sibling_ids": list(self.sibling_ids), "hidden_case_hash": self.hidden_case_hash}


@dataclass(frozen=True)
class BaselineView:
    baseline_name: str
    packet_hash: str
    packet_bits: int
    task_bundle_hash: str
    query_budget: int
    executable_packet: Mapping[str, Any]
    ignored_fields: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {"baseline_name": self.baseline_name, "packet_hash": self.packet_hash, "packet_bits": self.packet_bits, "task_bundle_hash": self.task_bundle_hash, "query_budget": self.query_budget, "executable_packet_hash": stable_hash(self.executable_packet, length=32), "ignored_fields": list(self.ignored_fields)}


def make_baseline_views(packet: Packet, task_bundle: TaskBundle, query_budget: int, denied_fields: Mapping[str, Sequence[str]] | None = None) -> tuple[BaselineView, ...]:
    denied_fields = denied_fields or {}
    packet_payload = packet.to_public_dict()
    packet_hash = stable_hash(packet_payload, length=32)
    task_hash = stable_hash(task_bundle.to_public_dict(), length=32)
    views: list[BaselineView] = []
    for name in BASELINE_NAMES:
        executable_packet = json.loads(canonical_json_bytes(packet_payload).decode("ascii"))
        ignored = tuple(denied_fields.get(name, ()))
        for field_name in ignored:
            executable_packet.pop(field_name, None)
        view_hash = packet_hash if not ignored else stable_hash(executable_packet, length=32)
        views.append(BaselineView(name, view_hash, packet_bit_length(packet), task_hash, query_budget, executable_packet, ignored))
    return tuple(views)


def audit_baseline_parity(views: Sequence[BaselineView]) -> AuditReport:
    by_name = {view.baseline_name: view for view in views}
    missing = [name for name in BASELINE_NAMES if name not in by_name]
    hashes = {view.packet_hash for view in views}
    bits = {view.packet_bits for view in views}
    tasks = {view.task_bundle_hash for view in views}
    budgets = {view.query_budget for view in views}
    denied = {view.baseline_name: list(view.ignored_fields) for view in views if view.ignored_fields}
    return AuditReport("baseline_parity_audit", (
        Finding("BASELINE_ALL_PRESENT", not missing, "all declared baselines have a packet view", {"missing": missing}),
        Finding("BASELINE_PACKET_HASH_PARITY", len(hashes) == 1, "all baselines receive the same lossless executable packet bytes", {"packet_hashes": sorted(hashes), "denied_fields": denied}),
        Finding("BASELINE_BUDGET_PARITY", len(bits) == 1 and len(tasks) == 1 and len(budgets) == 1, "all baselines receive matched bits, task bundle, and query budget", {"packet_bits": sorted(bits), "task_hashes": sorted(tasks), "query_budgets": sorted(budgets)}),
    ), {"views": [view.to_public_dict() for view in views]})


def audit_packet_order_control(packet: Packet) -> AuditReport:
    reversed_packet = replace(packet, entries=tuple(reversed(packet.entries)))
    rotated_entries = packet.entries[1:] + packet.entries[:1] if packet.entries else ()
    rotated_packet = replace(packet, entries=rotated_entries)
    return AuditReport("packet_order_control", (
        Finding("PACKET_REVERSE_SAME_FACT_MULTISET", packet_multiset_hash(reversed_packet) == packet_multiset_hash(packet), "reversing packet order preserves entry multiset hash", {}),
        Finding("PACKET_ROTATE_SAME_BITS", packet_bit_length(rotated_packet) == packet_bit_length(packet), "rotating packet order preserves canonical bit length", {"original_bits": packet_bit_length(packet), "rotated_bits": packet_bit_length(rotated_packet)}),
    ), {})
def entropy(values: Sequence[Any]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mutual_information(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    if len(xs) != len(ys):
        raise ValueError("MI inputs must have same length")
    if not xs:
        return 0.0
    joint = Counter(zip(xs, ys))
    x_counts = Counter(xs)
    y_counts = Counter(ys)
    total = len(xs)
    mi = 0.0
    for (x, y), count in joint.items():
        pxy = count / total
        px = x_counts[x] / total
        py = y_counts[y] / total
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def normalized_mi(xs: Sequence[Any], ys: Sequence[Any]) -> float:
    denom = min(entropy(xs), entropy(ys))
    if denom <= 0:
        return 0.0
    return mutual_information(xs, ys) / denom


def run_generator_mi_audit(public_seed: str, sample_count: int = 10_000, threshold: float = 0.05) -> AuditReport:
    per_m = max(1, sample_count // len(NUISANCE_SIZES))
    metrics_by_m: dict[int, dict[str, float]] = {}
    worst_metric = 0.0
    all_rng_purposes: set[str] = set()
    for m in NUISANCE_SIZES:
        role_categories: list[str] = []
        slot_indices: list[int] = []
        name_prefixes: list[str] = []
        orientations: list[int] = []
        kernel_ids_by_slot: list[str] = []
        sibling_indices: list[int] = []
        sibling_target_buckets: list[str] = []
        for idx in range(per_m):
            generated = generate_world(public_seed, m, idx, "mi_dry_run")
            world = generated.world
            all_rng_purposes.update(record.purpose for record in generated.rng_records)
            role_to_slot = world.role_to_slot()
            target_bucket = f"{min(role_to_slot['c0'], role_to_slot['c1']) % 16}:{max(role_to_slot['c0'], role_to_slot['c1']) % 16}"
            for slot, role in enumerate(world.slot_to_role):
                role_categories.append(_role_category(role))
                slot_indices.append(slot)
                name_prefixes.append(world.names[slot][1])
                orientations.append(world.orientations[slot])
                kernel_ids_by_slot.append(world.kernel_id)
            for sib_idx, sibling in enumerate(generate_sibling_worlds(public_seed, world, count=2)):
                sibling_indices.append(sib_idx)
                sibling_target_buckets.append(target_bucket)
        m_metrics = {
            "role_slot_index_nmi": normalized_mi(role_categories, slot_indices),
            "role_name_prefix_nmi": normalized_mi(role_categories, name_prefixes),
            "role_orientation_nmi": normalized_mi(role_categories, orientations),
            "role_kernel_id_nmi": normalized_mi(role_categories, kernel_ids_by_slot),
            "sibling_id_target_role_map_nmi": normalized_mi(sibling_indices, sibling_target_buckets),
        }
        metrics_by_m[m] = m_metrics
        worst_metric = max(worst_metric, *m_metrics.values())
    split_ok = all(purpose in all_rng_purposes for purpose in RNG_PURPOSES)
    return AuditReport("generator_mi_audit", (
        Finding("GENERATOR_MI_BELOW_THRESHOLD", worst_metric <= threshold, "dry-run role/name/slot/orientation/sibling MI metrics are below threshold", {"worst_metric": worst_metric, "threshold": threshold}),
        Finding("GENERATOR_RNG_STREAMS_SPLIT", split_ok, "world generator uses all required split RNG streams", {"observed_purposes": sorted(all_rng_purposes)}),
    ), {"sample_count": sample_count, "per_m": per_m, "threshold": threshold, "metrics_by_m": metrics_by_m, "worst_metric": worst_metric})


@dataclass(frozen=True)
class HarnessManifest:
    public_seed: str
    hidden_seed_rule: str
    constructor_id: str
    serializer_id: str
    scorer_id: str
    baseline_versions: Mapping[str, str]
    timeouts: Mapping[str, int]
    oracle_api: str
    frozen_before_hidden: bool
    hidden_results_opened: bool
    post_hidden_changes: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "public_seed": self.public_seed,
            "hidden_seed_rule": self.hidden_seed_rule,
            "constructor_id": self.constructor_id,
            "serializer_id": self.serializer_id,
            "scorer_id": self.scorer_id,
            "baseline_versions": dict(self.baseline_versions),
            "timeouts": dict(self.timeouts),
            "oracle_api": self.oracle_api,
            "frozen_before_hidden": self.frozen_before_hidden,
            "hidden_results_opened": self.hidden_results_opened,
            "post_hidden_changes": list(self.post_hidden_changes),
        }


def default_manifest(public_seed: str) -> HarnessManifest:
    return HarnessManifest(
        public_seed=public_seed,
        hidden_seed_rule="sha256(public_seed|hidden|unopened-until-freeze)",
        constructor_id=BlindPacketConstructor.constructor_id,
        serializer_id="canonical-json-bytes-v1",
        scorer_id="frameseed0-token-scorer-v1",
        baseline_versions={name: f"{name}-adapter-v1" for name in BASELINE_NAMES},
        timeouts={name: 0 for name in BASELINE_NAMES},
        oracle_api="public_transcript_only_no_hidden_labels",
        frozen_before_hidden=True,
        hidden_results_opened=False,
        post_hidden_changes=(),
    )


def audit_manifest(manifest: HarnessManifest) -> AuditReport:
    missing_baselines = [name for name in BASELINE_NAMES if name not in manifest.baseline_versions]
    return AuditReport("manifest_audit", (
        Finding("MANIFEST_FROZEN_BEFORE_HIDDEN", manifest.frozen_before_hidden and not manifest.hidden_results_opened, "manifest declares code frozen before hidden results", {"frozen_before_hidden": manifest.frozen_before_hidden, "hidden_results_opened": manifest.hidden_results_opened}),
        Finding("MANIFEST_BASELINES_DECLARED", not missing_baselines, "manifest declares every baseline adapter version", {"missing_baselines": missing_baselines}),
        Finding("MANIFEST_NO_POST_HIDDEN_EDITS", not manifest.post_hidden_changes, "manifest has no post-hidden edits", {"post_hidden_changes": list(manifest.post_hidden_changes)}),
    ), manifest.to_public_dict())


@dataclass(frozen=True)
class TokenEvidence:
    smuggling_detected: bool = False
    parity_failure: bool = False
    hidden_leakage: bool = False
    baseline_information_denial: bool = False
    boolean_trap_triggered: bool = False
    representation_noncontainment_passed: bool = True
    representation_prior_absorbed: bool = False
    l3_full_threshold_passed: bool = False
    l3_mean_hfa: float = 0.0
    packet_growth_sublinear: bool = False
    aftd_passed: bool = False
    ablation_drop_passed: bool = False
    role_stability_passed: bool = False
    bits_counted: bool = False
    boolean_escape_satisfied: bool = True
    baseline_absorptions: Mapping[str, bool] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "smuggling_detected": self.smuggling_detected,
            "parity_failure": self.parity_failure,
            "hidden_leakage": self.hidden_leakage,
            "baseline_information_denial": self.baseline_information_denial,
            "boolean_trap_triggered": self.boolean_trap_triggered,
            "representation_noncontainment_passed": self.representation_noncontainment_passed,
            "representation_prior_absorbed": self.representation_prior_absorbed,
            "l3_full_threshold_passed": self.l3_full_threshold_passed,
            "l3_mean_hfa": self.l3_mean_hfa,
            "packet_growth_sublinear": self.packet_growth_sublinear,
            "aftd_passed": self.aftd_passed,
            "ablation_drop_passed": self.ablation_drop_passed,
            "role_stability_passed": self.role_stability_passed,
            "bits_counted": self.bits_counted,
            "boolean_escape_satisfied": self.boolean_escape_satisfied,
            "baseline_absorptions": dict(self.baseline_absorptions),
        }


def _would_signal_except_boolean(evidence: TokenEvidence) -> bool:
    absorbers_clear = not any(evidence.baseline_absorptions.get(name, False) for name in ABSORPTION_PRECEDENCE)
    return (
        not evidence.smuggling_detected
        and not evidence.parity_failure
        and not evidence.hidden_leakage
        and not evidence.baseline_information_denial
        and evidence.representation_noncontainment_passed
        and not evidence.representation_prior_absorbed
        and evidence.l3_full_threshold_passed
        and evidence.l3_mean_hfa >= 0.97
        and evidence.packet_growth_sublinear
        and evidence.aftd_passed
        and evidence.ablation_drop_passed
        and evidence.role_stability_passed
        and evidence.bits_counted
        and absorbers_clear
    )


def assign_terminal_token(evidence: TokenEvidence) -> str:
    if evidence.smuggling_detected or evidence.parity_failure or evidence.hidden_leakage or evidence.baseline_information_denial:
        return TERMINAL_TOKENS["void"]
    if evidence.boolean_trap_triggered or (_would_signal_except_boolean(evidence) and not evidence.boolean_escape_satisfied):
        return TERMINAL_TOKENS["boolean_trap"]
    if not evidence.representation_noncontainment_passed or evidence.representation_prior_absorbed:
        return TERMINAL_TOKENS["representation_prior"]
    if not evidence.l3_full_threshold_passed:
        return TERMINAL_TOKENS["negative"]
    for absorber in ABSORPTION_PRECEDENCE:
        if evidence.baseline_absorptions.get(absorber, False):
            return TERMINAL_TOKENS[absorber]
    if _would_signal_except_boolean(evidence) and evidence.boolean_escape_satisfied:
        return TERMINAL_TOKENS["signal"]
    return TERMINAL_TOKENS["negative"]


def _all_signal_gates(**overrides: Any) -> TokenEvidence:
    base = TokenEvidence(
        l3_full_threshold_passed=True,
        l3_mean_hfa=0.98,
        packet_growth_sublinear=True,
        aftd_passed=True,
        ablation_drop_passed=True,
        role_stability_passed=True,
        bits_counted=True,
        boolean_escape_satisfied=True,
        baseline_absorptions={name: False for name in ABSORPTION_PRECEDENCE},
    )
    return replace(base, **overrides)


def run_golden_token_controls() -> AuditReport:
    controls: list[tuple[str, TokenEvidence, str]] = [
        ("smuggling_precedence", _all_signal_gates(smuggling_detected=True), TERMINAL_TOKENS["void"]),
        ("boolean_trap_precedence", _all_signal_gates(boolean_escape_satisfied=False), TERMINAL_TOKENS["boolean_trap"]),
        ("representation_prior", _all_signal_gates(representation_noncontainment_passed=False), TERMINAL_TOKENS["representation_prior"]),
        ("golden_negative_low_l3", TokenEvidence(l3_full_threshold_passed=False, baseline_absorptions={name: False for name in ABSORPTION_PRECEDENCE}), TERMINAL_TOKENS["negative"]),
        ("teaching_dimension_absorption", _all_signal_gates(baseline_absorptions={"teaching_dimension": True}), TERMINAL_TOKENS["teaching_dimension"]),
        ("library_learning_absorption", _all_signal_gates(baseline_absorptions={"library_learning": True}), TERMINAL_TOKENS["library_learning"]),
        ("nuisance_oracle_absorption", _all_signal_gates(baseline_absorptions={"nuisance_oracle": True}), TERMINAL_TOKENS["nuisance_oracle"]),
        ("cegis_before_active_and_rag", _all_signal_gates(baseline_absorptions={"cegis": True, "active_learning": True, "rag": True}), TERMINAL_TOKENS["cegis"]),
        ("clean_signal_shape", _all_signal_gates(), TERMINAL_TOKENS["signal"]),
    ]
    findings: list[Finding] = []
    results: dict[str, str] = {}
    for name, evidence, expected in controls:
        observed = assign_terminal_token(evidence)
        results[name] = observed
        findings.append(Finding(f"GOLDEN_TOKEN_{name.upper()}", observed == expected, f"golden control emits {expected}", {"observed": observed, "expected": expected}))
    return AuditReport("golden_token_controls", tuple(findings), results)


def run_preimplementation_audit(public_seed: str = "FRAMESEED0_B27_PUBLIC_SEED", dry_run_worlds: int = 10_000, mi_threshold: float = 0.05) -> AuditReport:
    manifest = default_manifest(public_seed)
    target = generate_world(public_seed, 16, 0, "audit_gate").world
    siblings = generate_sibling_worlds(public_seed, target, 2)
    transcript = make_public_transcript(target)
    packet_rng = split_rngs(public_seed, f"packet:{target.world_id}")["packet_construction"]
    packet = BlindPacketConstructor().construct(transcript, packet_rng)
    task_bundle = TaskBundle(target.world_id, tuple(s.world_id for s in siblings))
    ledger = BudgetLedger(packet_bits=packet_bit_length(packet))
    views = make_baseline_views(packet, task_bundle, query_budget=len(transcript.facts))
    reports = [
        audit_manifest(manifest),
        audit_world(target),
        audit_packet_serialization(packet),
        audit_constructor_provenance(packet, transcript),
        audit_budget_recomputation(packet, ledger),
        audit_baseline_parity(views),
        audit_packet_order_control(packet),
        audit_sabotage_control(packet, transcript),
        run_generator_mi_audit(public_seed, sample_count=dry_run_worlds, threshold=mi_threshold),
        run_golden_token_controls(),
    ]
    combined = combine_reports("frameseed0_preimplementation_audit", reports)
    metrics = dict(combined.metrics)
    metrics["no_performance_runs"] = True
    metrics["hidden_hfa_reported"] = False
    metrics["target_world"] = target.to_audit_dict()
    metrics["sibling_count"] = len(siblings)
    return AuditReport(combined.name, combined.findings, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAMESEED-0 audit harness gate")
    parser.add_argument("--public-seed", default="FRAMESEED0_B27_PUBLIC_SEED")
    parser.add_argument("--dry-run-worlds", type=int, default=10_000)
    parser.add_argument("--mi-threshold", type=float, default=0.05)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    started = time.time()
    report = run_preimplementation_audit(args.public_seed, args.dry_run_worlds, args.mi_threshold)
    payload = report.to_public_dict()
    payload["elapsed_s"] = round(time.time() - started, 3)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()