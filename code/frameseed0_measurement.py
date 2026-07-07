"""FRAMESEED-0 B28 hidden HFA measurement runner.

CPU-only first hidden measurement. The runner deliberately gives exact finite
teaching/search and oracle baselines the chance to absorb the Boolean result.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from frameseed0_harness import (
    ADMITTED_KERNELS,
    BlindPacketConstructor,
    BudgetLedger,
    Edit,
    Packet,
    PublicTranscript,
    Query,
    TERMINAL_TOKENS,
    TaskBundle,
    TokenEvidence,
    World,
    assign_terminal_token,
    audit_baseline_parity,
    audit_budget_recomputation,
    audit_constructor_provenance,
    audit_manifest,
    audit_packet_order_control,
    audit_packet_serialization,
    audit_sabotage_control,
    audit_world,
    canonical_json_bytes,
    default_manifest,
    generate_world,
    make_baseline_views,
    make_public_transcript,
    packet_bit_length,
    run_generator_mi_audit,
    run_golden_token_controls,
    split_rngs,
    stable_hash,
)

MEASUREMENT_VERSION = "frameseed0-b28-hidden-hfa-v1"
DEFAULT_PUBLIC_SEED = "FRAMESEED0_B28_PUBLIC_SEED"
HIDDEN_FAMILIES = ("H1_identity", "H2_nonidentity", "H3_oriented", "H4_composed")
SYSTEMS = (
    "l3_full",
    "td_h0",
    "l0_rotenn",
    "l1_active",
    "l2_cegis",
    "rag",
    "nuisance_oracle",
    "library_learning",
)


def _default(value: Any) -> Any:
    if hasattr(value, "to_public_dict"):
        return value.to_public_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _default(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_default(v) for v in value]
    if isinstance(value, list):
        return [_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _default(v) for k, v in sorted(value.items())}
    return value


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hidden_seed(public_seed: str) -> str:
    return _sha(f"{public_seed}|hidden|unopened-until-freeze")


def file_hashes(root: Path) -> dict[str, str]:
    files = [
        root / "code" / "frameseed0_harness.py",
        root / "code" / "frameseed0_measurement.py",
        root / "code" / "test_frameseed0_harness.py",
        root / "research" / "frameseed_0_precommit_spec.md",
    ]
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.exists()}


def _kernel_name(kernel: tuple[int, int, int, int]) -> str:
    return "K" + "".join(str(bit) for bit in kernel)


def _kernel_order(public_seed: str) -> tuple[tuple[int, int, int, int], ...]:
    kernels = list(ADMITTED_KERNELS)
    split_rngs(public_seed, "kernel_split")["world_structure"].shuffle(kernels)
    return tuple(kernels)


def hidden_kernel_split(public_seed: str) -> dict[str, list[str]]:
    kernels = _kernel_order(public_seed)
    return {"seen": [_kernel_name(k) for k in kernels[:4]], "hidden": [_kernel_name(k) for k in kernels[4:]]}


def _hidden_kernels(public_seed: str) -> tuple[tuple[int, int, int, int], ...]:
    return _kernel_order(public_seed)[4:]


def _nonidentity_rhos() -> tuple[tuple[int, int, int, int], ...]:
    return tuple(rho for rho in itertools.permutations(range(4)) if rho != (0, 1, 2, 3))


def _rho(public_seed: str, family: str, world_i: int, task_i: int) -> tuple[int, int, int, int]:
    if family == "H1_identity":
        return (0, 1, 2, 3)
    rhos = _nonidentity_rhos()
    idx = int(_sha(f"{public_seed}|rho|{family}|{world_i}|{task_i}")[:8], 16) % len(rhos)
    return rhos[idx]


def make_hidden_world(public_seed: str, m: int, family: str, world_i: int, perm_i: int, task_i: int) -> World:
    ns_i = world_i * 10000 + perm_i * 100 + task_i + HIDDEN_FAMILIES.index(family) * 1000000
    base = generate_world(hidden_seed(public_seed), m, ns_i, f"hidden:{family}").world
    kernel = _hidden_kernels(public_seed)[(world_i + task_i + HIDDEN_FAMILIES.index(family)) % len(_hidden_kernels(public_seed))]
    rho = _rho(public_seed, family, world_i, task_i)
    orientations = tuple(0 for _ in base.orientations) if family in {"H1_identity", "H2_nonidentity"} else base.orientations
    world_id = stable_hash(
        {
            "family": family,
            "m": m,
            "world_i": world_i,
            "perm_i": perm_i,
            "task_i": task_i,
            "kernel": kernel,
            "rho": rho,
            "orientations": orientations,
            "slot_to_role": base.slot_to_role,
            "names": base.names,
        },
        length=20,
    )
    return World(world_id, m, kernel, rho, orientations, base.slot_to_role, base.names, f"hidden:{family}:m={m}:w={world_i}:p={perm_i}:t={task_i}", family)


def _edited(query: Query, slot: int) -> int:
    for edit in query.edits:
        if edit.slot == slot:
            return int(edit.value)
    return int(query.observation[slot])


@dataclass(frozen=True)
class Program:
    system: str
    slots: tuple[int, int]
    table: Mapping[str, int]
    source: str
    bits: int
    fallback: int = 0

    def predict(self, query: Query) -> int:
        key = f"{_edited(query, self.slots[0])}{_edited(query, self.slots[1])}"
        return int(self.table.get(key, self.fallback))

    def to_public_dict(self) -> dict[str, Any]:
        return {"system": self.system, "slots": list(self.slots), "table": dict(self.table), "source": self.source, "bits": self.bits, "fallback": self.fallback}


def _base_labels(transcript: PublicTranscript) -> dict[tuple[int, ...], int]:
    return {fact.query.observation: fact.label for fact in transcript.facts if not fact.query.edits}


def support_slots(transcript: PublicTranscript) -> tuple[int, int]:
    base = _base_labels(transcript)
    counts: dict[int, int] = defaultdict(int)
    for fact in transcript.facts:
        if len(fact.query.edits) != 1:
            continue
        before = base.get(fact.query.observation)
        if before is not None and fact.label != before:
            counts[fact.query.edits[0].slot] += 1
    if len(counts) < 2:
        raise ValueError("public transcript did not expose two support slots")
    pair = sorted(counts, key=lambda slot: (-counts[slot], slot))[:2]
    return int(pair[0]), int(pair[1])


def table_from_transcript(transcript: PublicTranscript, slots: tuple[int, int]) -> dict[str, int]:
    table: dict[str, int] = {}
    labels: list[int] = []
    for fact in transcript.facts:
        if fact.query.edits:
            continue
        labels.append(fact.label)
        key = f"{fact.query.observation[slots[0]]}{fact.query.observation[slots[1]]}"
        table.setdefault(key, fact.label)
    fallback = 1 if sum(labels) > len(labels) / 2 else 0
    for key in ("00", "01", "10", "11"):
        table.setdefault(key, fallback)
    return table


def program_bits(system: str, slots: tuple[int, int], table: Mapping[str, int], source: str) -> int:
    return 8 * len(canonical_json_bytes({"system": system, "op": "truth_table2", "slots": slots, "table": table, "source": source}))


def exact_program(transcript: PublicTranscript, system: str, source: str) -> Program:
    slots = support_slots(transcript)
    table = table_from_transcript(transcript, slots)
    fallback = 1 if sum(table.values()) > len(table) / 2 else 0
    return Program(system, slots, table, source, program_bits(system, slots, table, source), fallback)


def oracle_program(world: World, transcript: PublicTranscript) -> Program:
    r2s = world.role_to_slot()
    slots = (r2s["c0"], r2s["c1"])
    table = table_from_transcript(transcript, slots)
    return Program("nuisance_oracle", slots, table, "oracle_causal_mask", program_bits("nuisance_oracle", slots, table, "oracle_causal_mask"), 1 if sum(table.values()) > len(table) / 2 else 0)


def packet_program(packet: Packet) -> Program:
    slots: tuple[int, int] | None = None
    for entry in packet.entries:
        if entry.entry_type == "representation_patch":
            schema = dict(dict(entry.payload).get("ast_or_schema", {}))
            raw = tuple(int(slot) for slot in schema.get("support_slots", ()))
            if len(raw) == 2:
                slots = (raw[0], raw[1])
                break
    if slots is None:
        raise ValueError("packet lacks two-slot representation patch")
    table: dict[str, int] = {}
    labels: list[int] = []
    for entry in packet.entries:
        if entry.entry_type != "example":
            continue
        payload = dict(entry.payload)
        values = {int(slot): int(bit) for slot, bit in payload.get("obs_mask", ())}
        key = f"{values.get(slots[0], 0)}{values.get(slots[1], 0)}"
        label = int(payload.get("label", 0))
        labels.append(label)
        table[key] = label
    fallback = 1 if labels and sum(labels) > len(labels) / 2 else 0
    for key in ("00", "01", "10", "11"):
        table.setdefault(key, fallback)
    return Program("rag", slots, table, "target_packet_direct", program_bits("rag", slots, table, "target_packet_direct"), fallback)


def make_packet(public_seed: str, world: World, transcript: PublicTranscript) -> Packet:
    rng = split_rngs(public_seed, f"hidden_packet:{world.world_id}")["packet_construction"]
    return BlindPacketConstructor().construct(transcript, rng)


def hidden_queries(public_seed: str, world: World, count: int) -> tuple[Query, ...]:
    if count < 5:
        raise ValueError("hidden query count must be at least 5")
    rng = split_rngs(hidden_seed(public_seed), f"hidden_queries:{world.world_id}")["hidden_queries"]
    r2s = world.role_to_slot()
    kinds = ("none", "causal", "alias", "nuisance", "composed")
    queries: list[Query] = []
    for i in range(count):
        c0 = rng.randrange(2)
        c1 = rng.randrange(2)
        obs = world.make_observation(c0, c1, [rng.randrange(2) for _ in range(world.m)])
        edits: list[Edit] = []
        kind = kinds[i % len(kinds)]
        if kind == "causal":
            slot = r2s[rng.choice(("c0", "c1"))]
            edits.append(Edit(slot, 1 - obs[slot]))
        elif kind == "alias":
            slot = r2s[rng.choice(("s0", "s1"))]
            edits.append(Edit(slot, 1 - obs[slot]))
        elif kind == "nuisance":
            slot = r2s[f"n{rng.randrange(world.m)}"] if world.m else r2s["s0"]
            edits.append(Edit(slot, 1 - obs[slot]))
        elif kind == "composed":
            slot_a = r2s[rng.choice(("c0", "c1"))]
            role_b = f"n{rng.randrange(world.m)}" if world.m and rng.randrange(2) == 0 else rng.choice(("s0", "s1"))
            slot_b = r2s[role_b]
            edits.append(Edit(slot_a, 1 - obs[slot_a]))
            if slot_b != slot_a:
                edits.append(Edit(slot_b, 1 - obs[slot_b]))
        queries.append(Query(obs, tuple(edits)))
    return tuple(queries)


def hfa(world: World, program: Program, queries: Sequence[Query]) -> float:
    return sum(1 for q in queries if program.predict(q) == world.label_query(q)) / len(queries)


class Acc:
    def __init__(self) -> None:
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def dict(self) -> dict[str, float | int]:
        if not self.values:
            return {"count": 0, "mean_hfa": 0.0, "min_hfa": 0.0, "max_hfa": 0.0, "std_hfa": 0.0}
        mean = sum(self.values) / len(self.values)
        std = math.sqrt(sum((v - mean) ** 2 for v in self.values) / len(self.values)) if len(self.values) > 1 else 0.0
        return {"count": len(self.values), "mean_hfa": mean, "min_hfa": min(self.values), "max_hfa": max(self.values), "std_hfa": std}


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
    return {"mean_packet_bits_by_m": means, "alpha_hat": alpha, "sublinear": alpha < 1.0}


def run_b28_measurement(public_seed: str = DEFAULT_PUBLIC_SEED, worlds_per_m: int = 64, role_permutations: int = 10, hidden_queries_per_world: int = 512, nuisance_sizes: Sequence[int] = (4, 16, 64, 256), include_rows: bool = False) -> dict[str, Any]:
    started = time.time()
    manifest = default_manifest(public_seed)
    harness_reports = [audit_manifest(manifest), run_generator_mi_audit(public_seed, 10000, 0.05), run_golden_token_controls()]
    if not all(report.passed for report in harness_reports):
        return {"name": "frameseed0_b28_hidden_hfa_measurement", "measurement_version": MEASUREMENT_VERSION, "passed": False, "terminal_token": TERMINAL_TOKENS["void"], "harness_reports": [r.to_public_dict() for r in harness_reports]}

    by_system = {system: Acc() for system in SYSTEMS}
    by_m = {int(m): {system: Acc() for system in SYSTEMS} for m in nuisance_sizes}
    by_family = {family: {system: Acc() for system in SYSTEMS} for family in HIDDEN_FAMILIES}
    packet_bits_by_m = {int(m): [] for m in nuisance_sizes}
    residual_bits_by_m = {int(m): [] for m in nuisance_sizes}
    role_stds: list[float] = []
    audit_failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    total_queries = 0
    bundles = 0

    for m in nuisance_sizes:
        m = int(m)
        for family in HIDDEN_FAMILIES:
            for world_i in range(worlds_per_m):
                perm_values: list[float] = []
                for perm_i in range(role_permutations):
                    worlds = tuple(make_hidden_world(public_seed, m, family, world_i, perm_i, task_i) for task_i in (0, 1, 2))
                    transcripts = tuple(make_public_transcript(world) for world in worlds)
                    packet = make_packet(public_seed, worlds[0], transcripts[0])
                    ledger_bits = packet_bit_length(packet)
                    packet_bits_by_m[m].append(ledger_bits)
                    residual_bits = sum(8 * len(canonical_json_bytes(t)) for t in transcripts[1:])
                    residual_bits_by_m[m].append(residual_bits)
                    bundle = TaskBundle(worlds[0].world_id, tuple(w.world_id for w in worlds[1:]), stable_hash({"m": m, "family": family, "world_i": world_i, "perm_i": perm_i}, length=32))
                    views = make_baseline_views(packet, bundle, query_budget=len(transcripts[0].facts))
                    audits = [
                        audit_world(worlds[0]),
                        audit_packet_serialization(packet),
                        audit_constructor_provenance(packet, transcripts[0]),
                        audit_budget_recomputation(packet, BudgetLedger(packet_bits=ledger_bits)),
                        audit_baseline_parity(views),
                        audit_packet_order_control(packet),
                        audit_sabotage_control(packet, transcripts[0]),
                    ]
                    for report in audits:
                        if not report.passed:
                            audit_failures.append({"report": report.name, "m": m, "family": family, "world_i": world_i, "perm_i": perm_i})
                    packet_rag = packet_program(packet)
                    for task_i, (world, transcript) in enumerate(zip(worlds, transcripts)):
                        queries = hidden_queries(public_seed, world, hidden_queries_per_world)
                        total_queries += len(queries)
                        programs = {
                            "l3_full": exact_program(transcript, "l3_full", "frame_patch_plus_task_public_transcript"),
                            "td_h0": exact_program(transcript, "td_h0", "exact_finite_two_slot_teaching_search"),
                            "l0_rotenn": exact_program(transcript, "l0_rotenn", "rote_public_fact_proxy"),
                            "l1_active": exact_program(transcript, "l1_active", "active_two_slot_version_space"),
                            "l2_cegis": exact_program(transcript, "l2_cegis", "cegis_two_slot_truth_table"),
                            "rag": packet_rag if task_i == 0 else exact_program(transcript, "rag", "same_public_packet_plus_task_records"),
                            "nuisance_oracle": oracle_program(world, transcript),
                            "library_learning": exact_program(transcript, "library_learning", "shared_two_slot_macro_proxy"),
                        }
                        for system, program in programs.items():
                            score = hfa(world, program, queries)
                            by_system[system].add(score)
                            by_m[m][system].add(score)
                            by_family[family][system].add(score)
                            if system == "l3_full":
                                perm_values.append(score)
                            if include_rows:
                                rows.append({"m": m, "family": family, "world_i": world_i, "perm_i": perm_i, "task_i": task_i, "task_id": world.world_id, "system": system, "hfa": score, "program_bits": program.bits, "packet_bits": ledger_bits, "residual_sibling_bits": residual_bits if task_i else 0})
                    if perm_values:
                        mean = sum(perm_values) / len(perm_values)
                        role_stds.append(math.sqrt(sum((v - mean) ** 2 for v in perm_values) / len(perm_values)))
                    bundles += 1

    system_summary = {system: acc.dict() for system, acc in by_system.items()}
    by_m_summary = {str(m): {system: acc.dict() for system, acc in systems.items()} for m, systems in sorted(by_m.items())}
    by_family_summary = {family: {system: acc.dict() for system, acc in systems.items()} for family, systems in by_family.items()}
    packet_growth = growth(packet_bits_by_m)
    role_max_std = max(role_stds) if role_stds else 0.0
    l3 = system_summary["l3_full"]
    l3_pass = bool(l3["min_hfa"] >= 0.95 and l3["mean_hfa"] >= 0.97 and all(by_m_summary[str(m)]["l3_full"]["min_hfa"] >= 0.95 for m in nuisance_sizes) and all(by_family_summary[f]["l3_full"]["min_hfa"] >= 0.95 for f in HIDDEN_FAMILIES))
    absorptions = {
        "teaching_dimension": bool(system_summary["td_h0"]["min_hfa"] >= 0.95),
        "library_learning": bool(system_summary["library_learning"]["min_hfa"] >= 0.95),
        "nuisance_oracle": bool(system_summary["nuisance_oracle"]["min_hfa"] >= 0.95),
        "cegis": bool(system_summary["l2_cegis"]["min_hfa"] >= 0.95),
        "active_learning": bool(system_summary["l1_active"]["min_hfa"] >= 0.95),
        "rag": bool(system_summary["rag"]["min_hfa"] >= 0.95),
    }
    evidence = TokenEvidence(
        smuggling_detected=bool(audit_failures),
        parity_failure=bool(audit_failures),
        l3_full_threshold_passed=l3_pass,
        l3_mean_hfa=float(l3["mean_hfa"]),
        packet_growth_sublinear=bool(packet_growth["sublinear"]),
        aftd_passed=False,
        ablation_drop_passed=False,
        role_stability_passed=role_max_std <= 0.02,
        bits_counted=True,
        boolean_escape_satisfied=True,
        baseline_absorptions=absorptions,
    )
    token = assign_terminal_token(evidence)
    payload: dict[str, Any] = {
        "name": "frameseed0_b28_hidden_hfa_measurement",
        "measurement_version": MEASUREMENT_VERSION,
        "passed": not audit_failures,
        "terminal_token": token,
        "token_interpretation": "hidden HFA threshold passed, but exact finite teaching/search baselines absorb before any T3-R signal" if token == TERMINAL_TOKENS["teaching_dimension"] else "see token_evidence",
        "public_seed": public_seed,
        "hidden_seed_rule": "sha256(public_seed|hidden|unopened-until-freeze)",
        "hidden_seed_hash": hashlib.sha256(hidden_seed(public_seed).encode("ascii")).hexdigest(),
        "kernel_split": hidden_kernel_split(public_seed),
        "config": {"nuisance_sizes": list(nuisance_sizes), "hidden_families": list(HIDDEN_FAMILIES), "worlds_per_m": worlds_per_m, "role_permutations_per_world": role_permutations, "hidden_eval_queries_per_world": hidden_queries_per_world, "sibling_tasks_per_world": 2},
        "harness_reports": [r.to_public_dict() for r in harness_reports],
        "system_summary": system_summary,
        "by_m": by_m_summary,
        "by_hidden_family": by_family_summary,
        "packet_growth": packet_growth,
        "residual_bits_by_m": {str(m): {"mean_residual_sibling_bits": (sum(bits) / len(bits) if bits else 0.0), "count": len(bits)} for m, bits in sorted(residual_bits_by_m.items())},
        "role_permutation_stability": {"bundle_count": len(role_stds), "max_l3_hfa_std": role_max_std, "passed": role_max_std <= 0.02},
        "token_evidence": evidence.to_public_dict(),
        "audit_failure_count": len(audit_failures),
        "audit_failures": audit_failures[:50],
        "counts": {"target_bundles": bundles, "task_evaluations": sum(acc.dict()["count"] for acc in by_system.values()) // len(SYSTEMS), "hidden_queries_scored_per_system": total_queries},
        "implementation_hashes": file_hashes(Path.cwd()),
        "elapsed_s": round(time.time() - started, 3),
    }
    if include_rows:
        payload["rows"] = rows
    return json.loads(json.dumps(_default(payload), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAMESEED-0 B28 hidden HFA measurement")
    parser.add_argument("--public-seed", default=DEFAULT_PUBLIC_SEED)
    parser.add_argument("--worlds-per-m", type=int, default=64)
    parser.add_argument("--role-permutations", type=int, default=10)
    parser.add_argument("--hidden-queries-per-world", type=int, default=512)
    parser.add_argument("--nuisance-sizes", default="4,16,64,256")
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    nuisance_sizes = tuple(int(part) for part in args.nuisance_sizes.split(",") if part.strip())
    payload = run_b28_measurement(args.public_seed, args.worlds_per_m, args.role_permutations, args.hidden_queries_per_world, nuisance_sizes, args.include_rows)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if payload.get("terminal_token") == TERMINAL_TOKENS["void"] or not payload.get("passed", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()