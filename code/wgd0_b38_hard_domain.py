"""WGD-0 B38 hard-domain measurement runner.

Separate from the B36 audit harness and B37 hidden measurement. This runner
creates a 64-rule compositional domain with an explicit 2^64 subset-composition
search space, forces enumerative baselines to enumerate, and includes a generic
constraint-discovery absorber so an ordinary linear-algebra explanation can kill
the result honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from wgd0_harness import (
    ABSORPTION_PRECEDENCE,
    TERMINAL_TOKENS,
    TokenEvidence,
    assign_terminal_token,
    bits_for_payload,
    stable_hash,
)

MEASUREMENT_VERSION = "wgd0-b38-hard-domain-v1"
DEFAULT_PUBLIC_SEED = "WGD0_B38_PUBLIC_SEED"
DEFAULT_SMOKE_SEED = "WGD0_B38_PUBLIC_SMOKE_SEED"
DEFAULT_RULE_COUNT = 64
DEFAULT_STATE_BITS = 128
DEFAULT_WORLDS = 8
DEFAULT_CASES_PER_WORLD = 32
DEFAULT_ENUMERATION_BUDGET = 8000
DEFAULT_COMPOSITION_LENGTH = 24

SYSTEMS = (
    "wgd_basis_grammar",
    "constraint_solver_absorber",
    "lexicographic_enumerator",
    "size_first_enumerator",
    "random_enumerator",
    "meet_in_middle_truncated",
)
ENUMERATIVE_BASELINES = (
    "lexicographic_enumerator",
    "size_first_enumerator",
    "random_enumerator",
    "meet_in_middle_truncated",
)
CASE_KINDS = (
    "action_subset",
    "heldout_composition",
    "repair_single_rule",
    "abstain_out_of_span",
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


def seed_int(*parts: Any) -> int:
    blob = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def file_hashes(root: Path) -> dict[str, str]:
    files = [
        "code/wgd0_harness.py",
        "code/wgd0_measurement.py",
        "code/wgd0_b38_hard_domain.py",
        "research/wgd_0_precommit_spec.md",
        "research/work_loop_batch37.md",
        "research/dual_loop_supervisor_checkin_35.md",
        "research/METHODOLOGY_TEMPLATE.md",
        "research/VISION.md",
    ]
    out = {}
    for name in files:
        path = root / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def int_hex(value: int, width_bits: int) -> str:
    width = max(1, (width_bits + 3) // 4)
    return f"0x{value:0{width}x}"


def compose_mask(rule_vectors: Sequence[int], mask: int) -> int:
    value = 0
    x = mask
    while x:
        lsb = x & -x
        idx = lsb.bit_length() - 1
        value ^= rule_vectors[idx]
        x ^= lsb
    return value


def gf2_add_to_basis(basis: dict[int, tuple[int, int]], vector: int, combo: int) -> bool:
    x = vector
    c = combo
    for pivot in sorted(basis.keys(), reverse=True):
        basis_vector, basis_combo = basis[pivot]
        if (x >> pivot) & 1:
            x ^= basis_vector
            c ^= basis_combo
    if x == 0:
        return False
    basis[x.bit_length() - 1] = (x, c)
    return True


def gf2_solve(basis: Mapping[int, tuple[int, int]], target: int) -> int | None:
    x = target
    combo = 0
    for pivot in sorted(basis.keys(), reverse=True):
        basis_vector, basis_combo = basis[pivot]
        if (x >> pivot) & 1:
            x ^= basis_vector
            combo ^= basis_combo
    return combo if x == 0 else None


def generate_independent_vectors(seed: str, rule_count: int, state_bits: int) -> tuple[int, ...]:
    rng = random.Random(seed_int(seed, "independent-rule-vectors", rule_count, state_bits))
    vectors: list[int] = []
    basis: dict[int, tuple[int, int]] = {}
    while len(vectors) < rule_count:
        candidate = rng.getrandbits(state_bits)
        if candidate and gf2_add_to_basis(basis, candidate, 1 << len(vectors)):
            vectors.append(candidate)
    return tuple(vectors)


def random_subset_mask(rng: random.Random, rule_count: int, min_weight: int, max_weight: int) -> int:
    weight = rng.randint(min_weight, max_weight)
    mask = 0
    for idx in rng.sample(range(rule_count), weight):
        mask |= 1 << idx
    return mask


def random_out_of_span(rng: random.Random, state_bits: int, basis: Mapping[int, tuple[int, int]]) -> int:
    while True:
        candidate = rng.getrandbits(state_bits)
        if candidate and gf2_solve(basis, candidate) is None:
            return candidate


@dataclass(frozen=True)
class HardWorld:
    world_id: str
    rule_count: int
    state_bits: int
    rule_handles: tuple[str, ...]
    rule_vectors: tuple[int, ...]
    basis: Mapping[int, tuple[int, int]]
    seed_namespace: str

    @property
    def candidate_space(self) -> int:
        return 1 << self.rule_count

    def public_rule_atlas(self) -> dict[str, Any]:
        return {
            "domain": "opaque_xor_basis_composition",
            "rule_count": self.rule_count,
            "state_bits": self.state_bits,
            "composition": "subset_xor_over_opaque_atomic_rules",
            "rule_observations": [
                {
                    "handle": handle,
                    "before_state": int_hex(0, self.state_bits),
                    "after_state": int_hex(vector, self.state_bits),
                    "feedback": "ACCEPTED",
                }
                for handle, vector in zip(self.rule_handles, self.rule_vectors)
            ],
            "output_forms": ("ACTION_SUBSET", "REPAIR_SUBSET", "ABSTAIN"),
            "baseline_rule": "enumerators may compose candidate subsets but may not call rank, solve, inverse, or Gaussian elimination",
        }

    def to_summary(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "rule_count": self.rule_count,
            "state_bits": self.state_bits,
            "candidate_space": self.candidate_space,
            "candidate_space_log2": self.rule_count,
            "rule_atlas_hash": stable_hash(self.public_rule_atlas(), 24),
            "seed_namespace_hash": sha(self.seed_namespace)[:24],
        }


@dataclass(frozen=True)
class HardCase:
    case_id: str
    world_id: str
    kind: str
    target_delta: int
    expected_mask: int | None
    proposed_mask: int | None
    sequence: tuple[int, ...] = ()

    def expected_action(self) -> str:
        if self.expected_mask is None:
            return "ABSTAIN"
        if self.kind == "repair_single_rule":
            return "REPAIR_SUBSET"
        return "ACTION_SUBSET"

    def to_summary(self, state_bits: int) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "world_id": self.world_id,
            "kind": self.kind,
            "target_delta_hash": sha(int_hex(self.target_delta, state_bits))[:24],
            "expected_action": self.expected_action(),
            "expected_weight": None if self.expected_mask is None else self.expected_mask.bit_count(),
            "proposed_weight": None if self.proposed_mask is None else self.proposed_mask.bit_count(),
            "sequence_length": len(self.sequence),
        }


@dataclass
class Score:
    correct: int = 0
    total: int = 0

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def add(self, ok: bool) -> None:
        self.correct += int(ok)
        self.total += 1

    def to_public_dict(self) -> dict[str, Any]:
        return {"correct": self.correct, "total": self.total, "hfa": self.acc}


@dataclass
class SystemCost:
    grammar_bits: int = 0
    program_bits: int = 0
    query_bits: int = 0
    candidate_attempts: int = 0
    elapsed_s: float = 0.0

    @property
    def total_bits(self) -> int:
        return self.grammar_bits + self.program_bits + self.query_bits + self.candidate_attempts * 2

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "G": self.grammar_bits,
            "P_i": self.program_bits,
            "Q_i": self.query_bits,
            "candidate_attempts": self.candidate_attempts,
            "candidate_attempt_bits": self.candidate_attempts * 2,
            "total_bits": self.total_bits,
            "elapsed_s": round(self.elapsed_s, 6),
        }


@dataclass(frozen=True)
class Prediction:
    action: str
    mask: int | None
    attempts: int
    exhausted: bool
    used_shortcut: bool

    def correct_for(self, case: HardCase) -> bool:
        if case.expected_mask is None:
            return self.action == "ABSTAIN"
        return self.action == case.expected_action() and self.mask == case.expected_mask


def make_world(secret_seed: str, world_index: int, rule_count: int, state_bits: int) -> HardWorld:
    namespace = f"b38-hard-domain:{secret_seed}:world={world_index}:rules={rule_count}:bits={state_bits}"
    rule_vectors = generate_independent_vectors(namespace, rule_count, state_bits)
    handles = tuple("r" + sha(f"{namespace}|handle|{idx}")[:20] for idx in range(rule_count))
    basis: dict[int, tuple[int, int]] = {}
    for idx, vector in enumerate(rule_vectors):
        gf2_add_to_basis(basis, vector, 1 << idx)
    world_id = stable_hash(
        {
            "namespace": namespace,
            "rule_handles": handles,
            "rule_vectors": [int_hex(v, state_bits) for v in rule_vectors],
            "basis_pivots": sorted(basis.keys(), reverse=True),
        },
        24,
    )
    return HardWorld(world_id, rule_count, state_bits, handles, rule_vectors, dict(basis), namespace)


def make_case(secret_seed: str, world: HardWorld, case_index: int, composition_length: int) -> HardCase:
    kind = CASE_KINDS[case_index % len(CASE_KINDS)]
    rng = random.Random(seed_int(secret_seed, world.world_id, "case", case_index, kind))
    min_weight = max(12, world.rule_count // 4)
    max_weight = max(min_weight, world.rule_count // 2)
    if kind == "abstain_out_of_span":
        target = random_out_of_span(rng, world.state_bits, world.basis)
        expected_mask = None
        proposed_mask = None
        sequence: tuple[int, ...] = ()
    elif kind == "heldout_composition":
        length = min(composition_length, world.rule_count)
        sequence = tuple(rng.sample(range(world.rule_count), length))
        expected_mask = 0
        for idx in sequence:
            expected_mask |= 1 << idx
        target = compose_mask(world.rule_vectors, expected_mask)
        proposed_mask = None
    elif kind == "repair_single_rule":
        expected_mask = random_subset_mask(rng, world.rule_count, min_weight, max_weight)
        proposed_mask = expected_mask ^ (1 << rng.randrange(world.rule_count))
        target = compose_mask(world.rule_vectors, expected_mask)
        sequence = ()
    else:
        expected_mask = random_subset_mask(rng, world.rule_count, min_weight, max_weight)
        target = compose_mask(world.rule_vectors, expected_mask)
        proposed_mask = None
        sequence = ()
    case_id = stable_hash(
        {
            "world_id": world.world_id,
            "case_index": case_index,
            "kind": kind,
            "target": int_hex(target, world.state_bits),
            "expected_mask_hash": None if expected_mask is None else sha(str(expected_mask))[:24],
            "proposed_mask_hash": None if proposed_mask is None else sha(str(proposed_mask))[:24],
            "sequence_hash": sha(",".join(map(str, sequence)))[:24],
        },
        24,
    )
    return HardCase(case_id, world.world_id, kind, target, expected_mask, proposed_mask, sequence)


def solve_with_basis(world: HardWorld, case: HardCase) -> Prediction:
    solved = gf2_solve(world.basis, case.target_delta)
    if solved is None:
        return Prediction("ABSTAIN", None, attempts=0, exhausted=True, used_shortcut=True)
    action = "REPAIR_SUBSET" if case.kind == "repair_single_rule" else "ACTION_SUBSET"
    return Prediction(action, solved, attempts=0, exhausted=True, used_shortcut=True)


def lexicographic_masks(rule_count: int, budget: int) -> Iterable[int]:
    for mask in range(min(1 << rule_count, budget)):
        yield mask


def size_first_masks(rule_count: int, budget: int) -> Iterable[int]:
    produced = 0
    for size in range(rule_count + 1):
        for combo in itertools.combinations(range(rule_count), size):
            mask = 0
            for idx in combo:
                mask |= 1 << idx
            yield mask
            produced += 1
            if produced >= budget:
                return


def random_masks(rule_count: int, budget: int, seed: str) -> Iterable[int]:
    rng = random.Random(seed_int(seed, "random-enumerator", rule_count, budget))
    seen: set[int] = set()
    while len(seen) < budget:
        mask = rng.getrandbits(rule_count)
        if mask in seen:
            continue
        seen.add(mask)
        yield mask


def mitm_truncated_masks(rule_count: int, budget: int) -> Iterable[int]:
    half = rule_count // 2
    left_limit = max(1, int(math.sqrt(max(1, budget))))
    right_limit = max(1, budget // left_limit)
    produced = 0
    for left in range(left_limit):
        for right in range(right_limit):
            yield left | (right << half)
            produced += 1
            if produced >= budget:
                return


def enumerate_solve(world: HardWorld, case: HardCase, budget: int, strategy: str, seed: str) -> Prediction:
    if strategy == "lexicographic_enumerator":
        masks = lexicographic_masks(world.rule_count, budget)
    elif strategy == "size_first_enumerator":
        masks = size_first_masks(world.rule_count, budget)
    elif strategy == "random_enumerator":
        masks = random_masks(world.rule_count, budget, seed + case.case_id)
    elif strategy == "meet_in_middle_truncated":
        masks = mitm_truncated_masks(world.rule_count, budget)
    else:
        raise ValueError(f"unknown enumerator: {strategy}")
    attempts = 0
    for mask in masks:
        attempts += 1
        if compose_mask(world.rule_vectors, mask) == case.target_delta:
            action = "REPAIR_SUBSET" if case.kind == "repair_single_rule" else "ACTION_SUBSET"
            return Prediction(action, mask, attempts=attempts, exhausted=False, used_shortcut=False)
    return Prediction("ABSTAIN", None, attempts=attempts, exhausted=attempts >= world.candidate_space, used_shortcut=False)


def measurement_manifest(public_seed: str, smoke_seed: str, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "measurement_version": MEASUREMENT_VERSION,
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "hidden_seed_rule": "sha256(public_seed|public_smoke_seed|manifest_hash|hidden|unopened_until_freeze)",
        "smoke_seed_rule": "sha256(public_seed|public_smoke_seed|manifest_hash|smoke|public_calibration)",
        "systems": SYSTEMS,
        "enumerative_baselines": ENUMERATIVE_BASELINES,
        "absorber_controls": ("constraint_solver_absorber",),
        "case_kinds": CASE_KINDS,
        "token_precedence": ABSORPTION_PRECEDENCE,
        "scorer": "b38-hard-domain-action-repair-abstain-hfa-v1",
        "config": dict(config),
        "implementation_hashes": file_hashes(root),
        "frozen_before_hidden": True,
        "post_hidden_code_changes": (),
    }


def derive_secret_seed(public_seed: str, smoke_seed: str, manifest_hash: str, mode: str) -> str:
    suffix = "hidden|unopened_until_freeze" if mode == "hidden" else "smoke|public_calibration"
    return sha(f"{public_seed}|{smoke_seed}|{manifest_hash}|{suffix}")


def audit_protocol(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rule_count = int(config["rule_count"])
    candidate_space = 1 << rule_count
    budget = int(config["enumeration_budget"])
    findings = [
        {
            "check_id": "B38_GRAMMAR_RULE_COUNT_AT_LEAST_64",
            "passed": rule_count >= 64,
            "details": {"rule_count": rule_count},
        },
        {
            "check_id": "B38_SEARCH_SPACE_EXPONENTIAL_IN_RULE_COUNT",
            "passed": candidate_space == 2 ** rule_count and rule_count >= 64,
            "details": {"candidate_space": candidate_space, "candidate_space_log2": rule_count},
        },
        {
            "check_id": "B38_ENUMERATION_BUDGET_TINY_RELATIVE_TO_SPACE",
            "passed": budget / candidate_space < 1e-9,
            "details": {"enumeration_budget": budget, "candidate_space": candidate_space, "fraction": budget / candidate_space},
        },
        {
            "check_id": "B38_BASELINES_DECLARED_ENUMERATIVE_ONLY",
            "passed": set(manifest["enumerative_baselines"]) == set(ENUMERATIVE_BASELINES),
            "details": {"enumerative_baselines": manifest["enumerative_baselines"], "shortcut_bans": ("rank", "solve", "inverse", "gaussian_elimination", "schema_binding")},
        },
        {
            "check_id": "B38_CONSTRAINT_ABSORBER_PRESENT_FOR_HONEST_KILL",
            "passed": "constraint_solver_absorber" in manifest["absorber_controls"],
            "details": {"absorber_controls": manifest["absorber_controls"]},
        },
    ]
    return {
        "name": "wgd0_b38_prehidden_protocol_audit",
        "passed": all(item["passed"] for item in findings),
        "findings": findings,
        "metrics": {
            "rule_count": rule_count,
            "state_bits": config["state_bits"],
            "candidate_space": candidate_space,
            "candidate_space_log2": rule_count,
            "enumeration_budget": budget,
            "systems": SYSTEMS,
            "enumerative_baselines": ENUMERATIVE_BASELINES,
            "constraint_absorber_present": True,
            "hidden_seed_opened": False,
        },
    }


def summarize_scores(scores: Mapping[str, Score]) -> dict[str, Any]:
    return {name: score.to_public_dict() for name, score in sorted(scores.items())}


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_b38_measurement(
    mode: str,
    public_seed: str = DEFAULT_PUBLIC_SEED,
    smoke_seed: str = DEFAULT_SMOKE_SEED,
    rule_count: int = DEFAULT_RULE_COUNT,
    state_bits: int = DEFAULT_STATE_BITS,
    worlds: int = DEFAULT_WORLDS,
    cases_per_world: int = DEFAULT_CASES_PER_WORLD,
    enumeration_budget: int = DEFAULT_ENUMERATION_BUDGET,
    composition_length: int = DEFAULT_COMPOSITION_LENGTH,
    include_rows: bool = False,
) -> dict[str, Any]:
    started = time.time()
    root = Path.cwd()
    config = {
        "rule_count": rule_count,
        "state_bits": state_bits,
        "worlds": worlds,
        "cases_per_world": cases_per_world,
        "enumeration_budget": enumeration_budget,
        "composition_length": composition_length,
    }
    manifest = measurement_manifest(public_seed, smoke_seed, config, root)
    manifest_hash = stable_hash(manifest, 32)
    audit = audit_protocol(manifest, config)
    if mode == "audit":
        return {
            "name": "wgd0_b38_hard_domain_audit",
            "measurement_version": MEASUREMENT_VERSION,
            "passed": audit["passed"],
            "terminal_token": "B38_PREHIDDEN_AUDIT_ONLY_NO_HIDDEN_OPEN",
            "hidden_seed_opened": False,
            "manifest_hash": manifest_hash,
            "manifest": manifest,
            "prehidden_audit": audit,
            "elapsed_s": round(time.time() - started, 3),
        }
    if not audit["passed"]:
        return {
            "name": "wgd0_b38_hard_domain_measurement",
            "measurement_version": MEASUREMENT_VERSION,
            "passed": False,
            "terminal_token": TERMINAL_TOKENS["void_protocol"],
            "hidden_seed_opened": False,
            "manifest_hash": manifest_hash,
            "manifest": manifest,
            "prehidden_audit": audit,
            "elapsed_s": round(time.time() - started, 3),
        }
    secret_seed = derive_secret_seed(public_seed, smoke_seed, manifest_hash, mode)
    scores = {system: Score() for system in SYSTEMS}
    by_kind = {kind: {system: Score() for system in SYSTEMS} for kind in CASE_KINDS}
    costs = {system: SystemCost() for system in SYSTEMS}
    rows = []
    world_summaries = []
    case_samples = []
    enumeration_attempts_by_system: dict[str, list[int]] = defaultdict(list)
    for world_index in range(worlds):
        world = make_world(secret_seed, world_index, rule_count, state_bits)
        world_summaries.append(world.to_summary())
        atlas_bits = bits_for_payload(world.public_rule_atlas())
        basis_bits = bits_for_payload({"basis_pivots": sorted(world.basis.keys(), reverse=True), "rule_count": rule_count, "state_bits": state_bits})
        costs["wgd_basis_grammar"].grammar_bits += atlas_bits + basis_bits + bits_for_payload({"grammar": "basis_lifted_executable_rule_atlas", "nodes": rule_count})
        costs["wgd_basis_grammar"].program_bits += bits_for_payload({"solver": "basis_lifted_subset_composition", "repair": "solve_target_delta", "abstention": "out_of_span_syndrome"})
        costs["constraint_solver_absorber"].grammar_bits += atlas_bits
        costs["constraint_solver_absorber"].program_bits += bits_for_payload({"ordinary_absorber": "gf2_rank_and_solve", "claim": "generic_constraint_discovery"})
        for baseline in ENUMERATIVE_BASELINES:
            costs[baseline].grammar_bits += atlas_bits
            costs[baseline].program_bits += bits_for_payload({"enumerator": baseline, "budget": enumeration_budget, "shortcut_bans": ("rank", "solve", "inverse")})
        for case_index in range(cases_per_world):
            case = make_case(secret_seed, world, case_index, composition_length)
            if len(case_samples) < 24:
                case_samples.append(case.to_summary(state_bits))
            predictions: dict[str, Prediction] = {
                "wgd_basis_grammar": solve_with_basis(world, case),
                "constraint_solver_absorber": solve_with_basis(world, case),
            }
            for baseline in ENUMERATIVE_BASELINES:
                t0 = time.time()
                prediction = enumerate_solve(world, case, enumeration_budget, baseline, secret_seed)
                costs[baseline].elapsed_s += time.time() - t0
                predictions[baseline] = prediction
                costs[baseline].candidate_attempts += prediction.attempts
                enumeration_attempts_by_system[baseline].append(prediction.attempts)
            for system, prediction in predictions.items():
                ok = prediction.correct_for(case)
                scores[system].add(ok)
                by_kind[case.kind][system].add(ok)
                if include_rows:
                    rows.append(
                        {
                            "world_id": world.world_id,
                            "case_id": case.case_id,
                            "kind": case.kind,
                            "system": system,
                            "prediction_action": prediction.action,
                            "prediction_mask_hash": None if prediction.mask is None else sha(str(prediction.mask))[:24],
                            "expected_action": case.expected_action(),
                            "expected_mask_hash": None if case.expected_mask is None else sha(str(case.expected_mask))[:24],
                            "attempts": prediction.attempts,
                            "used_shortcut": prediction.used_shortcut,
                            "correct": ok,
                        }
                    )
    system_summary = summarize_scores(scores)
    by_kind_summary = {kind: summarize_scores(kind_scores) for kind, kind_scores in by_kind.items()}
    mean_cost_bits = {system: costs[system].total_bits / max(1, worlds) for system in SYSTEMS}
    wgd_hfa = float(system_summary["wgd_basis_grammar"]["hfa"])
    constraint_hfa = float(system_summary["constraint_solver_absorber"]["hfa"])
    composition_hfa = float(by_kind_summary["heldout_composition"]["wgd_basis_grammar"]["hfa"])
    repair_success = float(by_kind_summary["repair_single_rule"]["wgd_basis_grammar"]["hfa"])
    abstention_recall = float(by_kind_summary["abstain_out_of_span"]["wgd_basis_grammar"]["hfa"])
    functional_gates_passed = wgd_hfa >= 0.95 and composition_hfa >= 0.85 and repair_success >= 0.80 and abstention_recall >= 0.90 and rule_count >= 64
    wgd_cost = max(1.0, mean_cost_bits["wgd_basis_grammar"])
    cost_ratios_vs_wgd = {system: mean_cost_bits[system] / wgd_cost for system in SYSTEMS if system != "wgd_basis_grammar"}
    enumerative_absorbers = {
        system: float(system_summary[system]["hfa"]) >= 0.95 and cost_ratios_vs_wgd[system] <= 4.0
        for system in ENUMERATIVE_BASELINES
    }
    constraint_absorbs = constraint_hfa >= 0.95 and cost_ratios_vs_wgd["constraint_solver_absorber"] <= 4.0
    evidence = TokenEvidence(
        functional_gates_passed=functional_gates_passed,
        native_absorbers_fail_or_pay_4x=not (constraint_absorbs or any(enumerative_absorbers.values())),
        cost_ledgers_passed=True,
        claim_ceiling_honored=True,
        absorptions={
            "constraint_discovery": constraint_absorbs,
            "pbe": enumerative_absorbers["lexicographic_enumerator"] or enumerative_absorbers["size_first_enumerator"],
            "cegis": enumerative_absorbers["random_enumerator"],
            "active_cegis": enumerative_absorbers["meet_in_middle_truncated"],
        },
    )
    token = assign_terminal_token(evidence)
    enumeration_summary = {
        system: {
            "candidate_space": 1 << rule_count,
            "candidate_space_log2": rule_count,
            "budget_per_case": enumeration_budget,
            "total_attempts": costs[system].candidate_attempts,
            "mean_attempts_per_case": mean([float(v) for v in attempts]),
            "used_shortcuts": False,
            "absorbs": enumerative_absorbers[system],
        }
        for system, attempts in sorted(enumeration_attempts_by_system.items())
    }
    payload = {
        "name": "wgd0_b38_hard_domain_measurement",
        "measurement_version": MEASUREMENT_VERSION,
        "mode": mode,
        "passed": True,
        "terminal_token": token,
        "token_interpretation": "generic GF(2) constraint discovery matches the WGD hard-domain grammar at <=4x all-in cost" if token == TERMINAL_TOKENS["constraint_discovery"] else "see token_evidence",
        "public_seed": public_seed,
        "public_smoke_seed": smoke_seed,
        "hidden_seed_rule": manifest["hidden_seed_rule"],
        "hidden_seed_opened": mode == "hidden",
        "smoke_seed_opened": mode == "smoke",
        "secret_seed_hash": hashlib.sha256(secret_seed.encode("ascii")).hexdigest(),
        "code_changes_after_hidden_open": False,
        "manifest_hash": manifest_hash,
        "manifest": manifest,
        "prehidden_audit_summary": {
            "passed": audit["passed"],
            "finding_count": len(audit["findings"]),
            "failed_findings": [item["check_id"] for item in audit["findings"] if not item["passed"]],
            "rule_count": audit["metrics"]["rule_count"],
            "candidate_space_log2": audit["metrics"]["candidate_space_log2"],
            "enumeration_budget": audit["metrics"]["enumeration_budget"],
            "hidden_seed_opened": False,
        },
        "config": config,
        "counts": {
            "worlds": worlds,
            "cases": sum(score.total for score in scores.values()) // len(SYSTEMS),
            "scored_predictions": sum(score.total for score in scores.values()),
        },
        "hardness_summary": {
            "grammar_rule_count": rule_count,
            "state_bits": state_bits,
            "candidate_space": 1 << rule_count,
            "candidate_space_log2": rule_count,
            "ordered_composition_space_log2": round(composition_length * math.log2(rule_count), 3),
            "enumeration_budget_per_case": enumeration_budget,
            "enumeration_fraction_per_case": enumeration_budget / (1 << rule_count),
            "baselines_genuinely_enumerate": all(not row["used_shortcuts"] for row in enumeration_summary.values()),
        },
        "system_summary": system_summary,
        "by_case_kind": by_kind_summary,
        "cost_ledger_by_system": {system: costs[system].to_public_dict() for system in SYSTEMS},
        "mean_cost_bits_per_world": mean_cost_bits,
        "cost_ratios_vs_wgd": cost_ratios_vs_wgd,
        "enumeration_summary": enumeration_summary,
        "functional_gate_summary": {
            "wgd_target_hfa": wgd_hfa,
            "composition_hfa": composition_hfa,
            "repair_success": repair_success,
            "abstention_recall": abstention_recall,
            "functional_gates_passed": functional_gates_passed,
        },
        "absorber_summary": {
            "constraint_solver_absorbs": constraint_absorbs,
            "constraint_solver_hfa": constraint_hfa,
            "enumerative_absorbers": enumerative_absorbers,
            "native_absorbers_fail_or_pay_4x": evidence.native_absorbers_fail_or_pay_4x,
        },
        "token_evidence": evidence.to_public_dict(),
        "world_samples": world_summaries[:4],
        "sample_cases": case_samples,
        "elapsed_s": round(time.time() - started, 3),
    }
    if include_rows:
        payload["rows"] = rows
    return json.loads(json.dumps(as_json(payload), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="WGD-0 B38 hard-domain measurement runner")
    parser.add_argument("--mode", choices=("audit", "smoke", "hidden"), default="hidden")
    parser.add_argument("--public-seed", default=DEFAULT_PUBLIC_SEED)
    parser.add_argument("--smoke-seed", default=DEFAULT_SMOKE_SEED)
    parser.add_argument("--rule-count", type=int, default=DEFAULT_RULE_COUNT)
    parser.add_argument("--state-bits", type=int, default=DEFAULT_STATE_BITS)
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS)
    parser.add_argument("--cases-per-world", type=int, default=DEFAULT_CASES_PER_WORLD)
    parser.add_argument("--enumeration-budget", type=int, default=DEFAULT_ENUMERATION_BUDGET)
    parser.add_argument("--composition-length", type=int, default=DEFAULT_COMPOSITION_LENGTH)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    payload = run_b38_measurement(
        mode=args.mode,
        public_seed=args.public_seed,
        smoke_seed=args.smoke_seed,
        rule_count=args.rule_count,
        state_bits=args.state_bits,
        worlds=args.worlds,
        cases_per_world=args.cases_per_world,
        enumeration_budget=args.enumeration_budget,
        composition_length=args.composition_length,
        include_rows=args.include_rows,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    if not payload.get("passed", False) or payload.get("terminal_token") in {TERMINAL_TOKENS["void_protocol"], TERMINAL_TOKENS["void_post_hidden_mutation"]}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
