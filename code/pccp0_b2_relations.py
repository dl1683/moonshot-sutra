#!/usr/bin/env python3
"""PCCP-0 B2 metamorphic relation discovery absorption suite.

B1 asked whether changing one field preserves the output. B2 asks whether an
input transformation induces an output transformation:

    F(tau(x)) == phi(F(x))

This file intentionally uses only a tiny finite Boolean world and stdlib
Python. It is an absorption test, not a moonshot result: Relation Miner v0 is
compared against exhaustive metamorphic relation mining over the same transform
grammar T and output-relation grammar Phi.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BitTuple = Tuple[int, ...]
BINARY_DOMAIN = (0, 1)
PHIS = ("identity", "NOT")


@dataclass(frozen=True)
class Intervention:
    family: str
    c0: Optional[int] = None
    c1: Optional[int] = None


@dataclass(frozen=True)
class Case:
    world_id: str
    obs: BitTuple
    intervention: Intervention
    target: int
    split: str


@dataclass(frozen=True)
class World:
    """Role-permuted parity world: C0, C1, N0..Nm, S."""

    m: int
    permutation: Tuple[int, ...]

    @property
    def world_id(self) -> str:
        return f"B2_m={self.m}_perm={','.join(map(str, self.permutation))}"

    @property
    def n_obs(self) -> int:
        return 2 + self.m + 1

    @property
    def s_latent_index(self) -> int:
        return 2 + self.m

    @property
    def role_to_obs(self) -> Dict[str, int]:
        role_to_latent = {"C0": 0, "C1": 1, "S": self.s_latent_index}
        for j in range(self.m):
            role_to_latent[f"N{j}"] = 2 + j
        return {role: self.permutation.index(latent) for role, latent in role_to_latent.items()}

    def target_rule(self, c0: int, c1: int) -> int:
        return int(c0) ^ int(c1)

    def encode(self, c: BitTuple, n: BitTuple, s: int) -> BitTuple:
        latent = tuple(c) + tuple(n) + (int(s),)
        return tuple(latent[self.permutation[pos]] for pos in range(len(latent)))

    def factual_states(self) -> Iterable[Tuple[BitTuple, BitTuple]]:
        for c in product(BINARY_DOMAIN, repeat=2):
            for n in product(BINARY_DOMAIN, repeat=self.m):
                yield tuple(c), tuple(n)

    def seen_relation_states(self) -> Iterable[Tuple[BitTuple, BitTuple]]:
        """Partial seen slice, deliberately not closed under C0 flips."""
        for c1 in BINARY_DOMAIN:
            for n in product(BINARY_DOMAIN, repeat=self.m):
                yield (0, c1), tuple(n)

    def make_case(
        self,
        split: str,
        family: str,
        c: BitTuple,
        n: BitTuple,
        c0: Optional[int] = None,
        c1: Optional[int] = None,
        s_surface: Optional[int] = None,
    ) -> Case:
        obs_s = self.target_rule(c[0], c[1]) if s_surface is None else int(s_surface)
        obs = self.encode(c, n, obs_s)
        eff_c0 = c[0] if c0 is None else int(c0)
        eff_c1 = c[1] if c1 is None else int(c1)
        return Case(self.world_id, obs, Intervention(family, c0, c1), eff_c0 ^ eff_c1, split)

    def relation_seed_cases(self) -> List[Case]:
        return [self.make_case("seen_relation", "id", c, n) for c, n in self.seen_relation_states()]

    def v0_b2_cases(self) -> List[Case]:
        cases: List[Case] = []
        for c, n in self.seen_relation_states():
            cases.append(self.make_case("v0_b2", "id", c, n))
            for value in BINARY_DOMAIN:
                cases.append(self.make_case("v0_b2", "do_c0_seen", c, n, c0=value))
                cases.append(self.make_case("v0_b2", "do_c1_seen", c, n, c1=value))
        return cases

    def hidden_factual_cases(self) -> List[Case]:
        return [self.make_case("hidden_relation", "id", c, n) for c, n in self.factual_states()]


def make_world_with_seed(m: int, seed: int) -> World:
    perm = list(range(2 + m + 1))
    random.Random(seed).shuffle(perm)
    return World(m, tuple(perm))


def decode_latent(world: World, obs: BitTuple) -> BitTuple:
    latent = [0] * world.n_obs
    for obs_pos, latent_idx in enumerate(world.permutation):
        latent[latent_idx] = int(obs[obs_pos])
    return tuple(latent)


def target_oracle_for_obs(world: World, case: Case, obs: BitTuple) -> int:
    latent = decode_latent(world, obs)
    c0, c1 = latent[0], latent[1]
    i = case.intervention
    eff_c0 = c0 if i.c0 is None else int(i.c0)
    eff_c1 = c1 if i.c1 is None else int(i.c1)
    return eff_c0 ^ eff_c1


class Program:
    name: str

    def predict(self, case: Case) -> int:
        raise NotImplementedError

    def description(self) -> str:
        return self.name


class TrueCausalProgram(Program):
    def __init__(self, world: World):
        roles = world.role_to_obs
        self.name = "P_true"
        self.c0_pos = roles["C0"]
        self.c1_pos = roles["C1"]

    def predict(self, case: Case) -> int:
        i = case.intervention
        c0 = case.obs[self.c0_pos] if i.c0 is None else int(i.c0)
        c1 = case.obs[self.c1_pos] if i.c1 is None else int(i.c1)
        return c0 ^ c1

    def description(self) -> str:
        return f"P_true = (has_c0 ? val_c0 : x{self.c0_pos}) XOR (has_c1 ? val_c1 : x{self.c1_pos})"


class BadCovarianceProgram(Program):
    """Passes V0_B2 but has wrong covariance off the seen manifold."""

    def __init__(self, world: World):
        roles = world.role_to_obs
        self.name = "P_bad_B2"
        self.c0_pos = roles["C0"]
        self.c1_pos = roles["C1"]
        self.s_pos = roles["S"]

    def predict(self, case: Case) -> int:
        i = case.intervention
        raw_c0 = case.obs[self.c0_pos]
        raw_c1 = case.obs[self.c1_pos]
        if i.c0 is not None or i.c1 is not None:
            c0 = raw_c0 if i.c0 is None else int(i.c0)
            c1 = raw_c1 if i.c1 is None else int(i.c1)
            return c0 ^ c1
        if raw_c0 == 0:
            return raw_c1
        return case.obs[self.s_pos]

    def description(self) -> str:
        return f"P_bad_B2 = if C override then parity; else if x{self.c0_pos}=0 then x{self.c1_pos} else x{self.s_pos}"


class Verifier:
    def __init__(self, world: World, cases: Sequence[Case], name: str):
        self.world = world
        self.cases = list(cases)
        self.name = name

    def verify(self, program: Program) -> Tuple[bool, Optional[Tuple[int, Case, int]]]:
        for idx, case in enumerate(self.cases):
            actual = program.predict(case)
            if actual != case.target:
                return False, (idx, case, actual)
        return True, None

    def family_accuracy(self, program: Program) -> Dict[str, float]:
        buckets: Dict[str, List[int]] = {}
        for case in self.cases:
            buckets.setdefault(case.intervention.family, []).append(int(program.predict(case) == case.target))
        return {family: sum(vals) / len(vals) for family, vals in sorted(buckets.items())}


@dataclass(frozen=True)
class Transform:
    fields: Tuple[int, ...]

    @property
    def arity(self) -> int:
        return len(self.fields)

    def label(self) -> str:
        return "flip(" + ",".join(f"x{field}" for field in self.fields) + ")"


@dataclass(frozen=True)
class RelationScore:
    transform: Transform
    phi: str
    score: float
    matches: int
    support: int
    mdl_length: int
    mdl_score: float
    source: str

    def key(self) -> Tuple[Tuple[int, ...], str]:
        return self.transform.fields, self.phi

    def label(self) -> str:
        return f"{self.transform.label()} -> {self.phi}"


@dataclass(frozen=True)
class MiningResult:
    source: str
    clauses: List[RelationScore]
    scores: List[RelationScore]
    target_label_calls: int
    tau_phi_score_units: int
    negative_control_perfect: int


@dataclass(frozen=True)
class RandomSearchStats:
    trials: int
    samples_per_trial: int
    query_units_per_trial: int
    success_rate: float
    any_covariance_hit_rate: float
    all_required_hit_rate: float
    mean_clauses: float


def all_transforms(n_obs: int) -> List[Transform]:
    transforms = [Transform((field,)) for field in range(n_obs)]
    transforms.extend(Transform(tuple(pair)) for pair in combinations(range(n_obs), 2))
    return transforms


def single_flip_transforms(n_obs: int) -> List[Transform]:
    return [Transform((field,)) for field in range(n_obs)]


def apply_transform(obs: BitTuple, transform: Transform) -> BitTuple:
    updated = list(obs)
    for field in transform.fields:
        updated[field] = 1 - updated[field]
    return tuple(updated)


def apply_phi(phi: str, y: int) -> int:
    if phi == "identity":
        return int(y)
    if phi == "NOT":
        return 1 - int(y)
    raise ValueError(phi)


def relation_mdl_length(n_obs: int, transform: Transform, phi: str) -> int:
    return 1 + transform.arity * math.ceil(math.log2(max(2, n_obs))) + 1


def score_relation(world: World, cases: Sequence[Case], transform: Transform, phi: str, source: str) -> RelationScore:
    matches = 0
    for case in cases:
        y = target_oracle_for_obs(world, case, case.obs)
        y_tau = target_oracle_for_obs(world, case, apply_transform(case.obs, transform))
        matches += int(y_tau == apply_phi(phi, y))
    support = len(cases)
    score = matches / support if support else 1.0
    mdl_length = relation_mdl_length(world.n_obs, transform, phi)
    return RelationScore(transform, phi, score, matches, support, mdl_length, matches - mdl_length, source)


def sorted_perfect_relations(scores: Sequence[RelationScore]) -> List[RelationScore]:
    perfect = [score for score in scores if score.score == 1.0]
    return sorted(perfect, key=lambda item: (item.transform.arity, item.mdl_length, PHIS.index(item.phi), item.transform.fields))


def shuffled_transform(transform: Transform, n_obs: int, seed: int) -> Transform:
    rng = random.Random(seed)
    mapping = list(range(n_obs))
    rng.shuffle(mapping)
    mapped = tuple(sorted(mapping[field] for field in transform.fields))
    if mapped == transform.fields:
        mapped = tuple(sorted((field + 1) % n_obs for field in transform.fields))
    return Transform(mapped)


def count_negative_control_perfect(world: World, cases: Sequence[Case], clauses: Sequence[RelationScore], seed: int) -> int:
    perfect = 0
    for idx, clause in enumerate(clauses):
        tau = shuffled_transform(clause.transform, world.n_obs, seed + idx * 17)
        perfect += int(score_relation(world, cases, tau, clause.phi, "negative_control").score == 1.0)
    return perfect


def mine_relations(world: World, cases: Sequence[Case], transforms: Sequence[Transform], phis: Sequence[str], source: str, seed: int) -> MiningResult:
    scores = [score_relation(world, cases, transform, phi, source) for transform in transforms for phi in phis]
    clauses = sorted_perfect_relations(scores)
    target_label_calls = len(cases) * len(transforms)
    tau_phi_score_units = len(cases) * len(transforms) * len(phis)
    negative = count_negative_control_perfect(world, cases, clauses, seed)
    return MiningResult(source, clauses, scores, target_label_calls, tau_phi_score_units, negative)


def compile_relation_cases(base_cases: Sequence[Case], clauses: Sequence[RelationScore], split: str) -> List[Case]:
    compiled: List[Case] = []
    for clause in clauses:
        fields = "_".join(str(field) for field in clause.transform.fields)
        family = f"rel_{clause.phi}_{fields}"
        for base in base_cases:
            compiled.append(Case(base.world_id, apply_transform(base.obs, clause.transform), Intervention(family), apply_phi(clause.phi, base.target), split))
    return compiled


def compile_relation_verifier(world: World, v0_cases: Sequence[Case], relation_cases: Sequence[Case], clauses: Sequence[RelationScore], name: str) -> Verifier:
    return Verifier(world, list(v0_cases) + compile_relation_cases(relation_cases, clauses, "discovered_relation"), name)


def relation_keys(clauses: Sequence[RelationScore]) -> set:
    return {clause.key() for clause in clauses}


def required_relation_specs(world: World) -> Dict[Tuple[Tuple[int, ...], str], str]:
    roles = world.role_to_obs
    c0 = roles["C0"]
    c1 = roles["C1"]
    s = roles["S"]
    return {
        ((c0,), "NOT"): "flip(C0) -> NOT(y)",
        ((c1,), "NOT"): "flip(C1) -> NOT(y)",
        (tuple(sorted((c0, c1))), "identity"): "flip(C0,C1) -> identity(y)",
        ((s,), "identity"): "flip(S) -> identity(y)",
    }


def required_clauses(world: World) -> List[RelationScore]:
    clauses: List[RelationScore] = []
    for (fields, phi), _label in required_relation_specs(world).items():
        transform = Transform(fields)
        clauses.append(RelationScore(transform, phi, 1.0, 0, 0, relation_mdl_length(world.n_obs, transform, phi), 0.0, "required_hidden"))
    return clauses


def b1_only_invariance(world: World, cases: Sequence[Case], seed: int) -> MiningResult:
    return mine_relations(world, cases, single_flip_transforms(world.n_obs), ("identity",), "b1_only_single_field_identity", seed)


def random_relation_search(world: World, relation_cases: Sequence[Case], v0_cases: Sequence[Case], true_program: Program, bad_program: Program, required: Dict[Tuple[Tuple[int, ...], str], str], trials: int, seed: int) -> RandomSearchStats:
    rng = random.Random(seed)
    transforms = all_transforms(world.n_obs)
    samples_per_trial = len(transforms) * len(PHIS)
    query_units = len(relation_cases) * samples_per_trial
    successes = 0
    any_cov = 0
    all_required = 0
    clause_counts: List[int] = []
    required_keys = set(required)
    covariance_keys = {key for key, label in required.items() if "NOT" in label}
    for _ in range(trials):
        found: Dict[Tuple[Tuple[int, ...], str], RelationScore] = {}
        for _sample in range(samples_per_trial):
            transform = rng.choice(transforms)
            phi = rng.choice(PHIS)
            score = score_relation(world, relation_cases, transform, phi, "random_relation")
            if score.score == 1.0:
                found.setdefault(score.key(), score)
        clauses = sorted_perfect_relations(found.values())
        verifier = compile_relation_verifier(world, v0_cases, relation_cases, clauses, "random_relation_v1")
        successes += int((not verifier.verify(bad_program)[0]) and verifier.verify(true_program)[0])
        keys = relation_keys(clauses)
        any_cov += int(bool(keys & covariance_keys))
        all_required += int(required_keys.issubset(keys))
        clause_counts.append(len(clauses))
    return RandomSearchStats(trials, samples_per_trial, query_units, successes / trials if trials else 0.0, any_cov / trials if trials else 0.0, all_required / trials if trials else 0.0, sum(clause_counts) / trials if trials else 0.0)


@dataclass
class B2Run:
    seed: int
    m: int
    role_to_obs: Dict[str, int]
    v0_cases: int
    relation_cases: int
    hidden_relation_cases: int
    transform_count: int
    phi_count: int
    miner: MiningResult
    exhaustive: MiningResult
    b1_only: MiningResult
    random_stats: RandomSearchStats
    v0_accepts_p_bad: bool
    v0_accepts_true: bool
    no_discovery_rejects_p_bad: bool
    miner_rejects_p_bad: bool
    miner_accepts_true: bool
    exhaustive_rejects_p_bad: bool
    exhaustive_accepts_true: bool
    b1_rejects_p_bad: bool
    b1_accepts_true: bool
    hidden_required_acc_p_bad: Dict[str, float]
    hidden_required_acc_true: Dict[str, float]
    miner_hidden_transfer_true: bool
    exhaustive_hidden_transfer_true: bool
    required_found_by_miner: Dict[str, bool]
    required_found_by_exhaustive: Dict[str, bool]
    required_found_by_b1: Dict[str, bool]
    bad_program: str
    true_program: str


def run_one(m: int, seed: int, random_trials: int = 96) -> B2Run:
    world = make_world_with_seed(m, seed)
    relation_cases = world.relation_seed_cases()
    v0_cases = world.v0_b2_cases()
    hidden_cases = world.hidden_factual_cases()
    for case in v0_cases:
        if target_oracle_for_obs(world, case, case.obs) != case.target:
            raise AssertionError("target oracle disagrees with V0_B2 case")
    true_program = TrueCausalProgram(world)
    bad_program = BadCovarianceProgram(world)
    v0 = Verifier(world, v0_cases, "V0_B2")
    transforms = all_transforms(world.n_obs)
    required = required_relation_specs(world)
    miner = mine_relations(world, relation_cases, transforms, PHIS, "relation_miner_v0", seed)
    exhaustive = mine_relations(world, relation_cases, transforms, PHIS, "exhaustive_mr", seed + 1000)
    b1 = b1_only_invariance(world, relation_cases, seed + 2000)
    miner_v = compile_relation_verifier(world, v0_cases, relation_cases, miner.clauses, "B2_V1")
    exhaustive_v = compile_relation_verifier(world, v0_cases, relation_cases, exhaustive.clauses, "exhaustive_mr_v1")
    b1_v = compile_relation_verifier(world, v0_cases, relation_cases, b1.clauses, "b1_only_v1")
    hidden_required_v = Verifier(world, compile_relation_cases(hidden_cases, required_clauses(world), "hidden_required"), "hidden_required_relations")
    miner_hidden_v = Verifier(world, compile_relation_cases(hidden_cases, miner.clauses, "hidden_miner_transfer"), "hidden_miner_transfer")
    exhaustive_hidden_v = Verifier(world, compile_relation_cases(hidden_cases, exhaustive.clauses, "hidden_exhaustive_transfer"), "hidden_exhaustive_transfer")
    random_stats = random_relation_search(world, relation_cases, v0_cases, true_program, bad_program, required, random_trials, seed + 3000)
    miner_keys = relation_keys(miner.clauses)
    exhaustive_keys = relation_keys(exhaustive.clauses)
    b1_keys = relation_keys(b1.clauses)
    return B2Run(
        seed=seed,
        m=m,
        role_to_obs=world.role_to_obs,
        v0_cases=len(v0_cases),
        relation_cases=len(relation_cases),
        hidden_relation_cases=len(hidden_cases),
        transform_count=len(transforms),
        phi_count=len(PHIS),
        miner=miner,
        exhaustive=exhaustive,
        b1_only=b1,
        random_stats=random_stats,
        v0_accepts_p_bad=v0.verify(bad_program)[0],
        v0_accepts_true=v0.verify(true_program)[0],
        no_discovery_rejects_p_bad=not v0.verify(bad_program)[0],
        miner_rejects_p_bad=not miner_v.verify(bad_program)[0],
        miner_accepts_true=miner_v.verify(true_program)[0],
        exhaustive_rejects_p_bad=not exhaustive_v.verify(bad_program)[0],
        exhaustive_accepts_true=exhaustive_v.verify(true_program)[0],
        b1_rejects_p_bad=not b1_v.verify(bad_program)[0],
        b1_accepts_true=b1_v.verify(true_program)[0],
        hidden_required_acc_p_bad=hidden_required_v.family_accuracy(bad_program),
        hidden_required_acc_true=hidden_required_v.family_accuracy(true_program),
        miner_hidden_transfer_true=miner_hidden_v.verify(true_program)[0],
        exhaustive_hidden_transfer_true=exhaustive_hidden_v.verify(true_program)[0],
        required_found_by_miner={label: key in miner_keys for key, label in required.items()},
        required_found_by_exhaustive={label: key in exhaustive_keys for key, label in required.items()},
        required_found_by_b1={label: key in b1_keys for key, label in required.items()},
        bad_program=bad_program.description(),
        true_program=true_program.description(),
    )


def run_experiment(m: int = 4, permutations: int = 8) -> List[B2Run]:
    runs: List[B2Run] = []
    seen_perms = set()
    seed = 91001
    while len(runs) < permutations:
        world = make_world_with_seed(m, seed)
        if world.permutation not in seen_perms:
            seen_perms.add(world.permutation)
            runs.append(run_one(m, seed))
        seed += 41
    return runs


def role_permutation_ok(runs: Sequence[B2Run]) -> bool:
    if not runs:
        return False
    maps = {tuple(sorted(run.role_to_obs.items())) for run in runs}
    c0_positions = {run.role_to_obs["C0"] for run in runs}
    c1_positions = {run.role_to_obs["C1"] for run in runs}
    s_positions = {run.role_to_obs["S"] for run in runs}
    return len(maps) == len(runs) and len(c0_positions) > 1 and len(c1_positions) > 1 and len(s_positions) > 1


def required_all_found(found: Dict[str, bool]) -> bool:
    return all(found.values())


def b1_only_insufficient(runs: Sequence[B2Run]) -> bool:
    return all(
        not run.required_found_by_b1["flip(C0) -> NOT(y)"]
        and not run.required_found_by_b1["flip(C1) -> NOT(y)"]
        and not run.required_found_by_b1["flip(C0,C1) -> identity(y)"]
        and run.required_found_by_b1["flip(S) -> identity(y)"]
        for run in runs
    )


def exhaustive_absorbs_b2(runs: Sequence[B2Run]) -> bool:
    return all(
        relation_keys(run.miner.clauses) == relation_keys(run.exhaustive.clauses)
        and run.miner.tau_phi_score_units == run.exhaustive.tau_phi_score_units
        and run.miner_rejects_p_bad == run.exhaustive_rejects_p_bad
        and run.miner_accepts_true == run.exhaustive_accepts_true
        and run.miner_hidden_transfer_true == run.exhaustive_hidden_transfer_true
        and required_all_found(run.required_found_by_exhaustive)
        for run in runs
    )


def b2_signal_condition(runs: Sequence[B2Run]) -> bool:
    if not runs:
        return False
    caught = all(run.v0_accepts_p_bad and run.v0_accepts_true and run.miner_rejects_p_bad and run.miner_accepts_true and required_all_found(run.required_found_by_miner) for run in runs)
    return caught and not exhaustive_absorbs_b2(runs)


def verdict(runs: Sequence[B2Run]) -> str:
    if not runs or not role_permutation_ok(runs):
        return "VOID"
    if not all(run.v0_accepts_p_bad and run.v0_accepts_true and required_all_found(run.required_found_by_miner) for run in runs):
        return "VOID"
    if exhaustive_absorbs_b2(runs):
        return "B2_DISCOVERY_ABSORBED"
    if b2_signal_condition(runs):
        return "B2_DISCOVERY_SIGNAL"
    return "VOID"


def role_by_obs_index(role_to_obs: Dict[str, int]) -> Dict[int, str]:
    return {obs_index: role for role, obs_index in role_to_obs.items()}


def summarize_required(found: Dict[str, bool]) -> str:
    short = {"flip(C0) -> NOT(y)": "C0", "flip(C1) -> NOT(y)": "C1", "flip(C0,C1) -> identity(y)": "C0C1", "flip(S) -> identity(y)": "S"}
    return ",".join(short[label] for label, ok in found.items() if ok) or "-"


def print_report(runs: Sequence[B2Run]) -> None:
    print("PCCP-0 B2 metamorphic relation discovery suite")
    print("=" * 78)
    print("World: role-permuted x0..xN with C0, C1, m nuisance bits N, and spurious S.")
    print("Target: y = C0 XOR C1. S equals y on factual observations but is not causal.")
    print("V0_B2: seen id examples plus do(C0:=v), do(C1:=v); no metamorphic clauses.")
    print("Relation seed split: id examples with latent C0=0 only; hidden transfer uses all factual states.")
    print("T: generic observed-field single flips and pair flips. Phi: identity and NOT.")
    print("Relation Miner v0: enumerate T x Phi, score exact paired labels, rank by MDL.")
    print("Exhaustive MR baseline: same T, Phi, relation data, and budget; accepts all perfect relations.")
    print()
    if runs:
        first = runs[0]
        print(f"Per-run n={first.m + 3} observed fields, |T|={first.transform_count}, |Phi|={first.phi_count}.")
        print(f"Relation examples |E|={first.relation_cases}; V0_B2 cases={first.v0_cases}.")
        print(f"Q_exhaustive score units = |E| * |T| * |Phi| = {first.exhaustive.tau_phi_score_units}; paired target-label calls = {first.exhaustive.target_label_calls}.")
        print(f"Random search gets {first.random_stats.samples_per_trial} sampled (tau,phi) pairs per trial over the same |E|.")
        print()
    header = "seed   C0@ C1@ S@  miner_req  V0_Pbad  B1_Pbad  B2_Pbad  exh_Pbad  clauses  rand_success"
    print(header)
    print("-" * len(header))
    for run in runs:
        print(f"{run.seed:<6} {run.role_to_obs['C0']:>3} {run.role_to_obs['C1']:>3} {run.role_to_obs['S']:>2}  {summarize_required(run.required_found_by_miner):<10} {'PASS' if run.v0_accepts_p_bad else 'REJECT':<8} {'REJECT' if run.b1_rejects_p_bad else 'PASS':<8} {'REJECT' if run.miner_rejects_p_bad else 'PASS':<8} {'REJECT' if run.exhaustive_rejects_p_bad else 'PASS':<9} {len(run.miner.clauses):>7} {run.random_stats.success_rate:>11.3f}")
    print()
    print("Role permutation audit:")
    for run in runs:
        print(f"  seed={run.seed}: role_to_obs={run.role_to_obs}")
    if runs:
        first = runs[0]
        audit_roles = role_by_obs_index(first.role_to_obs)
        print()
        print("First permutation required relation status:")
        for label, ok in first.required_found_by_miner.items():
            print(f"  relation_miner_v0 found {label}: {ok}")
        print()
        print("First permutation perfect relation counts:")
        single = sum(clause.transform.arity == 1 for clause in first.miner.clauses)
        pair = sum(clause.transform.arity == 2 for clause in first.miner.clauses)
        identity = sum(clause.phi == "identity" for clause in first.miner.clauses)
        not_phi = sum(clause.phi == "NOT" for clause in first.miner.clauses)
        print(f"  total={len(first.miner.clauses)} single={single} pair={pair} identity={identity} NOT={not_phi}")
        print(f"  negative-control perfect re-maps={first.miner.negative_control_perfect}")
        print()
        print("First permutation MDL-ranked single-field clauses (roles post-hoc only):")
        for clause in first.miner.clauses:
            if clause.transform.arity != 1:
                continue
            field = clause.transform.fields[0]
            role = audit_roles.get(field, "?")
            print(f"  {clause.label():<22} role={role:<2} support={clause.support:<3} mdl_len={clause.mdl_length}")
        print()
        print("First permutation hidden required-relation accuracy:")
        for family, acc in first.hidden_required_acc_p_bad.items():
            print(f"  P_bad_B2 {family}: {acc:.3f}")
        print(f"  P_true all hidden required relations pass: {all(acc == 1.0 for acc in first.hidden_required_acc_true.values())}")
        print()
        print(f"Example P_bad_B2: {first.bad_program}")
        print(f"Example P_true: {first.true_program}")
    print()
    print("Baseline parity checks:")
    print(f"  no_discovery_rejects_P_bad: {all(run.no_discovery_rejects_p_bad for run in runs)}")
    print(f"  relation_miner_rejects_P_bad: {all(run.miner_rejects_p_bad for run in runs)}")
    print(f"  exhaustive_mr_rejects_P_bad: {all(run.exhaustive_rejects_p_bad for run in runs)}")
    print(f"  relation_miner_accepts_true: {all(run.miner_accepts_true for run in runs)}")
    print(f"  exhaustive_mr_accepts_true: {all(run.exhaustive_accepts_true for run in runs)}")
    print(f"  relation_miner_hidden_transfer_true: {all(run.miner_hidden_transfer_true for run in runs)}")
    print(f"  exhaustive_mr_hidden_transfer_true: {all(run.exhaustive_hidden_transfer_true for run in runs)}")
    print(f"  miner_equals_exhaustive_clause_set: {all(relation_keys(run.miner.clauses) == relation_keys(run.exhaustive.clauses) for run in runs)}")
    print(f"  b1_only_insufficient_for_covariance: {b1_only_insufficient(runs)}")
    print()
    print("Smuggling audit:")
    print("  1. Relation miner sees role labels? No. It sees x0..xN, T, Phi, examples, and paired labels only.")
    print("  2. Is T target-specific? No. T is generic single observed-field flips plus pair flips.")
    print("  3. Is Phi target-specific? No. Phi is the generic binary-output grammar {identity, NOT}.")
    print("  4. Does exhaustive get equal T, Phi, and budget? Yes. It is the same enumeration without extra ranking claims.")
    print("  5. Is P_bad hard to catch without relations? Yes for V0_B2 and B1-only on the seen relation split; hidden all-state S checks can still catch S-like shortcuts, which is reported as a limitation.")
    print(f"  6. Role permutation diverse? {role_permutation_ok(runs)}")
    print()
    print("Narrative gate:")
    print(f"  Earned verdict token: {verdict(runs)}")
    print("  Gossip summary: The laptop found the flip rule, but the exhaustive tester found the same rule for the same price.")
    print("  What this does NOT prove: It does not prove transformation-grammar discovery, unknown Phi discovery, NTB-0 superiority, B3 decomposition, or open-world verifier discovery.")
    if exhaustive_absorbs_b2(runs):
        print("  Is exhaustive metamorphic relation mining sufficient here? Yes. It matches catch rate, transfer, false-positive behavior, and query budget.")
    else:
        print("  Is exhaustive metamorphic relation mining sufficient here? No on this run.")
    print(f"  Does B1-only discovery miss covariance relations? {b1_only_insufficient(runs)}")
    print(f"B2_DISCOVERY_SIGNAL_CONDITION: {b2_signal_condition(runs)}")
    print(f"EXHAUSTIVE_METAMORPHIC_MINING_ABSORBS: {exhaustive_absorbs_b2(runs)}")
    print(f"B1_ONLY_INSUFFICIENT: {b1_only_insufficient(runs)}")
    print(f"B2_VERDICT_TOKEN: {verdict(runs)}")


def main() -> None:
    runs = run_experiment()
    print_report(runs)


if __name__ == "__main__":
    main()
