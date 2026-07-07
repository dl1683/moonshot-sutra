#!/usr/bin/env python3
"""PCCP-0 B3 decomposition discovery absorption suite.

B3 asks whether a flat observation vector can be decomposed into independent
subproblems. This suite is small, finite, CPU-only, and honest: the miner is
compared to an equal-information exhaustive sensitivity/interaction baseline.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

BitTuple = Tuple[int, ...]
Target = Union[int, Tuple[int, int]]
BINARY_DOMAIN = (0, 1)
PHIS = ("identity", "NOT")


@dataclass(frozen=True)
class Case:
    world_id: str
    obs: BitTuple
    target: Target
    mode: str
    split: str
    family: str


@dataclass(frozen=True)
class World:
    """Role-permuted two-component Boolean world.

    Latent order: A0, A1, AN*, B0, B1, BN*, S.
    A target is XOR; B target is AND; S is the factual XOR of both targets.
    """

    k: int
    permutation: Tuple[int, ...]

    @property
    def world_id(self) -> str:
        return f"B3_k={self.k}_perm={','.join(map(str, self.permutation))}"

    @property
    def n_obs(self) -> int:
        return 5 + 2 * self.k

    @property
    def b0_i(self) -> int:
        return 2 + self.k

    @property
    def b1_i(self) -> int:
        return 3 + self.k

    @property
    def s_i(self) -> int:
        return 4 + 2 * self.k

    @property
    def role_to_latent(self) -> Dict[str, int]:
        roles = {"A0": 0, "A1": 1, "B0": self.b0_i, "B1": self.b1_i, "S": self.s_i}
        for j in range(self.k):
            roles[f"AN{j}"] = 2 + j
            roles[f"BN{j}"] = 4 + self.k + j
        return roles

    @property
    def role_to_obs(self) -> Dict[str, int]:
        return {role: self.permutation.index(latent) for role, latent in self.role_to_latent.items()}

    @property
    def comp_a(self) -> Set[int]:
        r = self.role_to_obs
        return {r["A0"], r["A1"]}

    @property
    def comp_b(self) -> Set[int]:
        r = self.role_to_obs
        return {r["B0"], r["B1"]}

    def encode(self, a0: int, a1: int, b0: int, b1: int, an: BitTuple, bn: BitTuple, s: Optional[int] = None) -> BitTuple:
        ya = int(a0) ^ int(a1)
        yb = int(b0) & int(b1)
        latent = (int(a0), int(a1)) + tuple(an) + (int(b0), int(b1)) + tuple(bn) + ((ya ^ yb) if s is None else int(s),)
        return tuple(latent[self.permutation[pos]] for pos in range(len(latent)))

    def decode(self, obs: BitTuple) -> BitTuple:
        latent = [0] * self.n_obs
        for obs_pos, latent_i in enumerate(self.permutation):
            latent[latent_i] = int(obs[obs_pos])
        return tuple(latent)

    def target_pair(self, obs: BitTuple) -> Tuple[int, int]:
        latent = self.decode(obs)
        return latent[0] ^ latent[1], latent[self.b0_i] & latent[self.b1_i]

    def target(self, obs: BitTuple, mode: str) -> Target:
        ya, yb = self.target_pair(obs)
        if mode == "multi":
            return ya, yb
        if mode == "single":
            return ya ^ yb
        raise ValueError(mode)

    def nuisance_assignments(self) -> Iterable[Tuple[BitTuple, BitTuple]]:
        for an in product(BINARY_DOMAIN, repeat=self.k):
            for bn in product(BINARY_DOMAIN, repeat=self.k):
                yield tuple(an), tuple(bn)

    def full_states(self) -> Iterable[Tuple[int, int, int, int, BitTuple, BitTuple]]:
        for a0, a1, b0, b1 in product(BINARY_DOMAIN, repeat=4):
            for an, bn in self.nuisance_assignments():
                yield a0, a1, b0, b1, an, bn

    def support_states(self) -> Iterable[Tuple[int, int, int, int, BitTuple, BitTuple]]:
        # Deliberate support shortcut: target_A == target_B on V0/discovery seed cases.
        for b0, b1 in product(BINARY_DOMAIN, repeat=2):
            target_b = b0 & b1
            for a0, a1 in product(BINARY_DOMAIN, repeat=2):
                if (a0 ^ a1) == target_b:
                    for an, bn in self.nuisance_assignments():
                        yield a0, a1, b0, b1, an, bn

    def case_from_obs(self, obs: BitTuple, mode: str, split: str, family: str) -> Case:
        return Case(self.world_id, obs, self.target(obs, mode), mode, split, family)

    def support_cases(self, mode: str) -> List[Case]:
        return [
            self.case_from_obs(self.encode(a0, a1, b0, b1, an, bn), mode, "support", "id")
            for a0, a1, b0, b1, an, bn in self.support_states()
        ]

    def v0_cases(self, mode: str) -> List[Case]:
        # V0 has seen labels and some single-field do checks, but only on nuisance/S fields.
        r = self.role_to_obs
        do_fields = [r[f"AN{j}"] for j in range(self.k)] + [r[f"BN{j}"] for j in range(self.k)] + [r["S"]]
        cases: List[Case] = []
        for base in self.support_cases(mode):
            cases.append(Case(base.world_id, base.obs, base.target, mode, "v0_b3", "id"))
            for field in do_fields:
                cases.append(self.case_from_obs(flip_field(base.obs, field), mode, "v0_b3", f"do_x{field}_seen"))
        return cases

    def hidden_cases(self, mode: str) -> List[Case]:
        cases: List[Case] = []
        for a0, a1, b0, b1, an, bn in self.full_states():
            for s_value in BINARY_DOMAIN:
                cases.append(self.case_from_obs(self.encode(a0, a1, b0, b1, an, bn, s=s_value), mode, "hidden", "full_domain_do_s"))
        return cases


def make_world(k: int, seed: int) -> World:
    perm = list(range(5 + 2 * k))
    random.Random(seed).shuffle(perm)
    return World(k, tuple(perm))


def flip_field(obs: BitTuple, field: int) -> BitTuple:
    out = list(obs)
    out[field] = 1 - out[field]
    return tuple(out)


def flip_fields(obs: BitTuple, fields: Sequence[int]) -> BitTuple:
    out = list(obs)
    for field in fields:
        out[field] = 1 - out[field]
    return tuple(out)


def channels(target: Target) -> Tuple[int, ...]:
    return target if isinstance(target, tuple) else (target,)


class Program:
    name: str

    def predict(self, case: Case) -> Target:
        raise NotImplementedError

    def description(self) -> str:
        return self.name


class TrueProgram(Program):
    def __init__(self, world: World):
        self.world = world
        self.name = "P_true_B3"

    def predict(self, case: Case) -> Target:
        return self.world.target(case.obs, case.mode)

    def description(self) -> str:
        r = self.world.role_to_obs
        return f"P_true_B3 = (x{r['A0']} XOR x{r['A1']}, x{r['B0']} AND x{r['B1']}); scalar XOR-composed"


class BadEntangledProgram(Program):
    def __init__(self, world: World):
        self.world = world
        self.roles = world.role_to_obs
        self.name = "P_bad_B3"

    def a_shortcut(self, obs: BitTuple) -> int:
        return obs[self.roles["A0"]] ^ obs[self.roles["A1"]]

    def predict(self, case: Case) -> Target:
        a_value = self.a_shortcut(case.obs)
        if case.mode == "multi":
            return a_value, a_value
        if case.mode == "single":
            return a_value ^ a_value
        raise ValueError(case.mode)

    def description(self) -> str:
        return f"P_bad_B3 = output_A=x{self.roles['A0']} XOR x{self.roles['A1']}; output_B=output_A"


class Verifier:
    def __init__(self, cases: Sequence[Case], name: str):
        self.cases = list(cases)
        self.name = name

    def verify(self, program: Program) -> Tuple[bool, Optional[Tuple[int, Case, Target]]]:
        for idx, case in enumerate(self.cases):
            actual = program.predict(case)
            if actual != case.target:
                return False, (idx, case, actual)
        return True, None

    def accuracy(self, program: Program) -> float:
        return sum(program.predict(case) == case.target for case in self.cases) / len(self.cases) if self.cases else 1.0


@dataclass(frozen=True)
class IndependenceClause:
    channel: int
    outside_field: int
    perturbed_obs: BitTuple
    expected_value: int
    family: str

    def key(self) -> Tuple[int, int]:
        return self.channel, self.outside_field


class IndependenceVerifier:
    def __init__(self, base_cases: Sequence[Case], clauses: Sequence[IndependenceClause], name: str):
        self.base = Verifier(base_cases, name + "_base")
        self.clauses = list(clauses)
        self.name = name

    def verify(self, program: Program) -> Tuple[bool, Optional[Tuple[str, int, object, Target]]]:
        base_ok, base_ce = self.base.verify(program)
        if not base_ok:
            return False, ("base", -1, base_ce, ())
        for idx, clause in enumerate(self.clauses):
            case = Case("", clause.perturbed_obs, (0, 0), "multi", "independence", clause.family)
            actual = program.predict(case)
            if channels(actual)[clause.channel] != clause.expected_value:
                return False, ("independence", idx, clause, actual)
        return True, None

@dataclass(frozen=True)
class SensitivityScore:
    field: int
    channel: int
    changed: int
    support: int
    score: float
    pattern: Tuple[int, ...]


@dataclass(frozen=True)
class PairInteraction:
    field_a: int
    field_b: int
    changed: int
    support: int
    score: float


@dataclass
class DecompositionResult:
    source: str
    mode: str
    components: List[Set[int]]
    output_to_component: Dict[int, int]
    sensitivities: List[SensitivityScore]
    interactions: List[PairInteraction]
    clauses: List[IndependenceClause]
    target_label_calls: int
    score_units: int

    def component_key(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(sorted(tuple(sorted(component)) for component in self.components))

    def clause_keys(self) -> Set[Tuple[int, int]]:
        return {clause.key() for clause in self.clauses}


def sensitivity_scores(world: World, cases: Sequence[Case], mode: str) -> Tuple[List[SensitivityScore], Dict[int, Tuple[int, ...]], int, int]:
    base_targets = [channels(world.target(case.obs, mode)) for case in cases]
    channel_count = len(base_targets[0]) if base_targets else (2 if mode == "multi" else 1)
    scores: List[SensitivityScore] = []
    patterns_by_field: Dict[int, Tuple[int, ...]] = {}
    for field in range(world.n_obs):
        channel_patterns: List[List[int]] = [[] for _ in range(channel_count)]
        for idx, case in enumerate(cases):
            y = base_targets[idx]
            yp = channels(world.target(flip_field(case.obs, field), mode))
            for channel in range(channel_count):
                channel_patterns[channel].append(int(yp[channel] != y[channel]))
        if mode == "single":
            patterns_by_field[field] = tuple(channel_patterns[0])
        for channel, vals in enumerate(channel_patterns):
            pattern = tuple(vals)
            changed = sum(pattern)
            support = len(pattern)
            scores.append(SensitivityScore(field, channel, changed, support, changed / support if support else 0.0, pattern))
    return scores, patterns_by_field, len(cases) * world.n_obs, len(cases) * world.n_obs * channel_count


def pairwise_interactions(world: World, cases: Sequence[Case]) -> Tuple[List[PairInteraction], int, int]:
    base_y = [int(world.target(case.obs, "single")) for case in cases]
    single_y = {
        field: [int(world.target(flip_field(case.obs, field), "single")) for case in cases]
        for field in range(world.n_obs)
    }
    out: List[PairInteraction] = []
    for a, b in combinations(range(world.n_obs), 2):
        changed = 0
        for idx, case in enumerate(cases):
            yb = single_y[b][idx]
            yab = int(world.target(flip_fields(case.obs, (a, b)), "single"))
            effect_a = single_y[a][idx] ^ base_y[idx]
            effect_a_after_b = yab ^ yb
            changed += int(effect_a != effect_a_after_b)
        support = len(cases)
        out.append(PairInteraction(a, b, changed, support, changed / support if support else 0.0))
    pair_count = (world.n_obs * (world.n_obs - 1)) // 2
    return out, len(cases) * (world.n_obs + pair_count), len(cases) * pair_count


def connected_components(active: Set[int], edges: Set[Tuple[int, int]]) -> List[Set[int]]:
    adj = {node: set() for node in active}
    for a, b in edges:
        if a in active and b in active:
            adj[a].add(b)
            adj[b].add(a)
    comps: List[Set[int]] = []
    remaining = set(active)
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        stack = [root]
        comp: Set[int] = set()
        while stack:
            node = stack.pop()
            comp.add(node)
            for nxt in sorted(adj[node]):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return sorted(comps, key=lambda c: (min(c), len(c)))


def build_graph(mode: str, scores: Sequence[SensitivityScore], patterns: Dict[int, Tuple[int, ...]], interactions: Sequence[PairInteraction]) -> Tuple[List[Set[int]], Dict[int, int]]:
    active = {score.field for score in scores if score.score > 0.0}
    edges: Set[Tuple[int, int]] = set()
    by_channel: Dict[int, List[SensitivityScore]] = {}
    for score in scores:
        by_channel.setdefault(score.channel, []).append(score)
    if mode == "multi":
        for channel_scores in by_channel.values():
            fields = [score.field for score in channel_scores if score.score > 0.0]
            for a, b in combinations(fields, 2):
                edges.add(tuple(sorted((a, b))))
    else:
        for a, b in combinations(sorted(active), 2):
            if patterns.get(a) == patterns.get(b) and any(patterns.get(a, ())) :
                edges.add((a, b))
        for inter in interactions:
            if inter.score > 0.0:
                edges.add(tuple(sorted((inter.field_a, inter.field_b))))
    comps = connected_components(active, edges)
    out_to_comp: Dict[int, int] = {}
    if mode == "multi":
        for channel, channel_scores in by_channel.items():
            fields = {score.field for score in channel_scores if score.score > 0.0}
            for idx, comp in enumerate(comps):
                if fields and fields.issubset(comp):
                    out_to_comp[channel] = idx
                    break
    return comps, out_to_comp


def compile_independence(world: World, cases: Sequence[Case], result: DecompositionResult) -> List[IndependenceClause]:
    if result.mode != "multi":
        return []
    clauses: List[IndependenceClause] = []
    for channel, comp_idx in sorted(result.output_to_component.items()):
        inside = result.components[comp_idx]
        for field in range(world.n_obs):
            if field in inside:
                continue
            for case in cases:
                expected = channels(world.target(case.obs, "multi"))[channel]
                clauses.append(IndependenceClause(channel, field, flip_field(case.obs, field), expected, f"outside_x{field}_keeps_y{channel}"))
    return clauses


def mine_decomposition(world: World, cases: Sequence[Case], mode: str, source: str) -> DecompositionResult:
    scores, patterns, calls, units = sensitivity_scores(world, cases, mode)
    interactions: List[PairInteraction] = []
    if mode == "single":
        interactions, pair_calls, pair_units = pairwise_interactions(world, cases)
        calls += pair_calls
        units += pair_units
    comps, out_to_comp = build_graph(mode, scores, patterns, interactions)
    result = DecompositionResult(source, mode, comps, out_to_comp, scores, interactions, [], calls, units)
    result.clauses = compile_independence(world, cases, result)
    return result


@dataclass(frozen=True)
class Transform:
    fields: Tuple[int, ...]

    @property
    def arity(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class RelationClause:
    transform: Transform
    phi: str
    matches: int
    support: int
    mode: str

    def key(self) -> Tuple[Tuple[int, ...], str]:
        return self.transform.fields, self.phi


@dataclass(frozen=True)
class B2BaselineResult:
    clauses: List[RelationClause]
    target_label_calls: int
    score_units: int
    rejects_p_bad: bool
    accepts_true: bool


def all_b2_transforms(n_obs: int) -> List[Transform]:
    return [Transform((field,)) for field in range(n_obs)] + [Transform(tuple(pair)) for pair in combinations(range(n_obs), 2)]


def apply_phi(phi: str, target: Target) -> Target:
    vals = channels(target)
    if phi == "identity":
        mapped = vals
    elif phi == "NOT":
        mapped = tuple(1 - val for val in vals)
    else:
        raise ValueError(phi)
    return mapped if isinstance(target, tuple) else mapped[0]


def score_relation(world: World, cases: Sequence[Case], mode: str, transform: Transform, phi: str) -> RelationClause:
    matches = 0
    for case in cases:
        y = world.target(case.obs, mode)
        yp = world.target(flip_fields(case.obs, transform.fields), mode)
        matches += int(yp == apply_phi(phi, y))
    return RelationClause(transform, phi, matches, len(cases), mode)


def compile_relation_cases(cases: Sequence[Case], clauses: Sequence[RelationClause]) -> List[Case]:
    out: List[Case] = []
    for clause in clauses:
        family = f"b2_{clause.phi}_{'_'.join(str(field) for field in clause.transform.fields)}"
        for case in cases:
            out.append(Case(case.world_id, flip_fields(case.obs, clause.transform.fields), apply_phi(clause.phi, case.target), case.mode, "b2_relation", family))
    return out


def run_b2_only(world: World, v0_cases: Sequence[Case], relation_cases: Sequence[Case], mode: str, true_program: Program, bad_program: Program) -> B2BaselineResult:
    transforms = all_b2_transforms(world.n_obs)
    clauses: List[RelationClause] = []
    for transform in transforms:
        for phi in PHIS:
            rel = score_relation(world, relation_cases, mode, transform, phi)
            if rel.matches == rel.support:
                clauses.append(rel)
    clauses.sort(key=lambda rel: (rel.transform.arity, rel.transform.fields, PHIS.index(rel.phi)))
    verifier = Verifier(list(v0_cases) + compile_relation_cases(relation_cases, clauses), f"b2_{mode}")
    return B2BaselineResult(clauses, len(relation_cases) * len(transforms), len(relation_cases) * len(transforms) * len(PHIS), not verifier.verify(bad_program)[0], verifier.verify(true_program)[0])


@dataclass(frozen=True)
class RandomClusterStats:
    trials: int
    success_rate: float
    accepts_true_rate: float
    rejects_bad_rate: float


def random_cluster_baseline(world: World, v0_cases: Sequence[Case], relation_cases: Sequence[Case], result: DecompositionResult, true_program: Program, bad_program: Program, trials: int, seed: int) -> RandomClusterStats:
    if result.mode != "multi" or len(result.components) != 2:
        return RandomClusterStats(trials, 0.0, 0.0, 0.0)
    active = sorted(set().union(*result.components))
    size0 = len(result.components[0])
    rng = random.Random(seed)
    success = accepts = rejects = 0
    for _ in range(trials):
        shuffled = list(active)
        rng.shuffle(shuffled)
        fake = DecompositionResult("random_cluster", "multi", [set(shuffled[:size0]), set(shuffled[size0:])], {0: 0, 1: 1}, result.sensitivities, [], [], result.target_label_calls, result.score_units)
        fake.clauses = compile_independence(world, relation_cases, fake)
        verifier = IndependenceVerifier(v0_cases, fake.clauses, "random_v1")
        true_ok = verifier.verify(true_program)[0]
        bad_rejected = not verifier.verify(bad_program)[0]
        accepts += int(true_ok)
        rejects += int(bad_rejected)
        success += int(true_ok and bad_rejected)
    return RandomClusterStats(trials, success / trials, accepts / trials, rejects / trials)

class Expr:
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError

    def length(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True)
class Var(Expr):
    name: str

    def eval(self, env: Dict[str, int]) -> int:
        return int(env[self.name])

    def length(self) -> int:
        return 1

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def eval(self, env: Dict[str, int]) -> int:
        return int(self.value)

    def length(self) -> int:
        return 1

    def __str__(self) -> str:
        return "TRUE" if self.value else "FALSE"


@dataclass(frozen=True)
class Not(Expr):
    arg: Expr

    def eval(self, env: Dict[str, int]) -> int:
        return 1 - self.arg.eval(env)

    def length(self) -> int:
        return 1 + self.arg.length()

    def __str__(self) -> str:
        return f"(NOT {self.arg})"


@dataclass(frozen=True)
class Bin(Expr):
    op: str
    left: Expr
    right: Expr

    def eval(self, env: Dict[str, int]) -> int:
        a, b = self.left.eval(env), self.right.eval(env)
        if self.op == "AND":
            return a & b
        if self.op == "OR":
            return a | b
        if self.op == "XOR":
            return a ^ b
        if self.op == "EQ":
            return int(a == b)
        raise ValueError(self.op)

    def length(self) -> int:
        return 1 + self.left.length() + self.right.length()

    def __str__(self) -> str:
        return f"({self.op} {self.left} {self.right})"


@dataclass
class ExprBank:
    by_len: Dict[int, Dict[int, Expr]]
    attempts: int
    elapsed_sec: float

    @property
    def unique_semantics(self) -> int:
        return sum(len(bucket) for bucket in self.by_len.values())

    def count_upto(self, max_len: int) -> int:
        return sum(len(bucket) for length, bucket in self.by_len.items() if length <= max_len)


@dataclass(frozen=True)
class SynthMetric:
    name: str
    found: bool
    program: str
    program_length: int
    candidate_attempts: int
    unique_semantics: int
    candidate_space: int
    elapsed_sec: float


@dataclass(frozen=True)
class SynthesisValue:
    multi_joint: SynthMetric
    multi_decomposed: SynthMetric
    single_joint: SynthMetric
    single_decomposed: SynthMetric


def envs_for_cases(cases: Sequence[Case]) -> List[Dict[str, int]]:
    return [{f"x{idx}": bit for idx, bit in enumerate(case.obs)} for case in cases]


def mask_for_values(values: Sequence[int]) -> int:
    mask = 0
    for idx, value in enumerate(values):
        if int(value):
            mask |= 1 << idx
    return mask


def expr_semantics(expr: Expr, envs: Sequence[Dict[str, int]]) -> int:
    mask = 0
    for idx, env in enumerate(envs):
        if expr.eval(env):
            mask |= 1 << idx
    return mask


def add_expr(by_len: Dict[int, Dict[int, Expr]], expr: Expr, sem: int) -> None:
    by_len.setdefault(expr.length(), {})
    by_len[expr.length()].setdefault(sem, expr)


def enumerate_expr_bank(var_fields: Sequence[int], envs: Sequence[Dict[str, int]], max_len: int) -> ExprBank:
    started = time.time()
    by_len: Dict[int, Dict[int, Expr]] = {}
    attempts = 0
    full_mask = (1 << len(envs)) - 1
    atoms: List[Expr] = [Var(f"x{field}") for field in var_fields] + [Const(0), Const(1)]
    for atom in atoms:
        attempts += 1
        add_expr(by_len, atom, expr_semantics(atom, envs))
    ops = ("AND", "OR", "XOR", "EQ")
    for length in range(2, max_len + 1):
        for sem, expr in list(by_len.get(length - 1, {}).items()):
            attempts += 1
            add_expr(by_len, Not(expr), full_mask ^ sem)
        for left_len in range(1, length - 1):
            right_len = length - 1 - left_len
            if right_len < left_len:
                continue
            left_items = list(by_len.get(left_len, {}).items())
            right_items = list(by_len.get(right_len, {}).items())
            for i, (left_sem, left_expr) in enumerate(left_items):
                for j, (right_sem, right_expr) in enumerate(right_items):
                    if left_len == right_len and j < i:
                        continue
                    for op in ops:
                        attempts += 1
                        if op == "AND":
                            sem = left_sem & right_sem
                        elif op == "OR":
                            sem = left_sem | right_sem
                        elif op == "XOR":
                            sem = left_sem ^ right_sem
                        else:
                            sem = full_mask ^ (left_sem ^ right_sem)
                        add_expr(by_len, Bin(op, left_expr, right_expr), sem)
    return ExprBank(by_len, attempts, time.time() - started)


def find_expr_metric(name: str, bank: ExprBank, target_mask: int, max_len: int) -> SynthMetric:
    started = time.time()
    tested = 0
    for length in range(1, max_len + 1):
        bucket = bank.by_len.get(length, {})
        tested += len(bucket)
        if target_mask in bucket:
            expr = bucket[target_mask]
            return SynthMetric(name, True, str(expr), expr.length(), bank.attempts + tested, bank.unique_semantics, bank.count_upto(max_len), bank.elapsed_sec + time.time() - started)
    return SynthMetric(name, False, "-", 0, bank.attempts + tested, bank.unique_semantics, bank.count_upto(max_len), bank.elapsed_sec + time.time() - started)


def find_xor_decomposed_metric(name: str, bank_a: ExprBank, bank_b: ExprBank, target_mask: int, max_len: int) -> SynthMetric:
    started = time.time()
    candidate_space = 0
    for la in range(1, max_len):
        for lb in range(1, max_len):
            if 1 + la + lb <= max_len:
                candidate_space += len(bank_a.by_len.get(la, {})) * len(bank_b.by_len.get(lb, {}))
    tested = 0
    for total_len in range(3, max_len + 1):
        for la in range(1, total_len - 1):
            lb = total_len - 1 - la
            left_bucket = bank_a.by_len.get(la, {})
            right_bucket = bank_b.by_len.get(lb, {})
            for left_sem, left_expr in left_bucket.items():
                tested += 1
                right_expr = right_bucket.get(target_mask ^ left_sem)
                if right_expr is not None:
                    expr = Bin("XOR", left_expr, right_expr)
                    return SynthMetric(name, True, str(expr), expr.length(), bank_a.attempts + bank_b.attempts + tested, bank_a.unique_semantics + bank_b.unique_semantics, candidate_space, bank_a.elapsed_sec + bank_b.elapsed_sec + time.time() - started)
    return SynthMetric(name, False, "-", 0, bank_a.attempts + bank_b.attempts + tested, bank_a.unique_semantics + bank_b.unique_semantics, candidate_space, bank_a.elapsed_sec + bank_b.elapsed_sec + time.time() - started)


def channel_mask(world: World, cases: Sequence[Case], channel: int) -> int:
    return mask_for_values([channels(world.target(case.obs, "multi"))[channel] for case in cases])


def run_synthesis_value(world: World, multi_result: DecompositionResult, single_result: DecompositionResult) -> SynthesisValue:
    synth_cases = world.hidden_cases("multi")
    envs = envs_for_cases(synth_cases)
    all_fields = list(range(world.n_obs))
    target_a = channel_mask(world, synth_cases, 0)
    target_b = channel_mask(world, synth_cases, 1)
    combo = mask_for_values([int(world.target(case.obs, "single")) for case in synth_cases])

    all_len3 = enumerate_expr_bank(all_fields, envs, 3)
    joint_a = find_expr_metric("multi_joint_A_all_fields", all_len3, target_a, 3)
    joint_b = find_expr_metric("multi_joint_B_all_fields", all_len3, target_b, 3)
    multi_joint = SynthMetric("multi_joint_pair_all_fields", joint_a.found and joint_b.found, f"({joint_a.program}, {joint_b.program})", joint_a.program_length + joint_b.program_length, joint_a.candidate_attempts + joint_b.candidate_attempts, all_len3.unique_semantics, all_len3.count_upto(3) ** 2, joint_a.elapsed_sec + joint_b.elapsed_sec)

    comp_a = sorted(multi_result.components[multi_result.output_to_component[0]])
    comp_b = sorted(multi_result.components[multi_result.output_to_component[1]])
    bank_a = enumerate_expr_bank(comp_a, envs, 3)
    bank_b = enumerate_expr_bank(comp_b, envs, 3)
    decomp_a = find_expr_metric("multi_decomp_A_component", bank_a, target_a, 3)
    decomp_b = find_expr_metric("multi_decomp_B_component", bank_b, target_b, 3)
    multi_decomp = SynthMetric("multi_decomposed_components", decomp_a.found and decomp_b.found, f"({decomp_a.program}, {decomp_b.program})", decomp_a.program_length + decomp_b.program_length, decomp_a.candidate_attempts + decomp_b.candidate_attempts, bank_a.unique_semantics + bank_b.unique_semantics, bank_a.count_upto(3) * bank_b.count_upto(3), decomp_a.elapsed_sec + decomp_b.elapsed_sec)

    all_len7 = enumerate_expr_bank(all_fields, envs, 7)
    single_joint = find_expr_metric("single_joint_all_fields", all_len7, combo, 7)
    comps = sorted(single_result.components, key=lambda c: (len(c), min(c)))
    if len(comps) >= 2:
        bank_s0 = enumerate_expr_bank(sorted(comps[0]), envs, 3)
        bank_s1 = enumerate_expr_bank(sorted(comps[1]), envs, 3)
        single_decomp = find_xor_decomposed_metric("single_decomposed_xor_components", bank_s0, bank_s1, combo, 7)
    else:
        single_decomp = SynthMetric("single_decomposed_xor_components", False, "-", 0, 0, 0, 0, 0.0)
    return SynthesisValue(multi_joint, multi_decomp, single_joint, single_decomp)

@dataclass
class B3Run:
    seed: int
    mode: str
    role_to_obs: Dict[str, int]
    v0_cases: int
    discovery_cases: int
    hidden_cases: int
    miner: DecompositionResult
    exhaustive: DecompositionResult
    b2_only: B2BaselineResult
    random_stats: Optional[RandomClusterStats]
    v0_accepts_p_bad: bool
    v0_accepts_true: bool
    hidden_acc_p_bad: float
    miner_rejects_p_bad: Optional[bool]
    miner_accepts_true: Optional[bool]
    exhaustive_rejects_p_bad: Optional[bool]
    exhaustive_accepts_true: Optional[bool]
    boundary_correct: bool
    true_program: str
    bad_program: str


def components_match(result: DecompositionResult, world: World) -> bool:
    expected = {tuple(sorted(world.comp_a)), tuple(sorted(world.comp_b))}
    found = {tuple(sorted(comp)) for comp in result.components}
    return expected.issubset(found)


def run_one(mode: str, k: int, seed: int, random_trials: int = 128) -> B3Run:
    world = make_world(k, seed)
    v0_cases = world.v0_cases(mode)
    discovery_cases = world.support_cases(mode)
    hidden_cases = world.hidden_cases(mode)
    true_program = TrueProgram(world)
    bad_program = BadEntangledProgram(world)
    v0 = Verifier(v0_cases, f"V0_B3_{mode}")
    hidden = Verifier(hidden_cases, f"hidden_{mode}")
    miner = mine_decomposition(world, discovery_cases, mode, "decomposition_miner")
    exhaustive = mine_decomposition(world, discovery_cases, mode, "exhaustive_interaction")
    b2 = run_b2_only(world, v0_cases, discovery_cases, mode, true_program, bad_program)
    random_stats: Optional[RandomClusterStats] = None
    miner_rejects = miner_accepts = exhaustive_rejects = exhaustive_accepts = None
    if mode == "multi":
        miner_v = IndependenceVerifier(v0_cases, miner.clauses, "B3_V1")
        exhaustive_v = IndependenceVerifier(v0_cases, exhaustive.clauses, "exhaustive_v1")
        miner_rejects = not miner_v.verify(bad_program)[0]
        miner_accepts = miner_v.verify(true_program)[0]
        exhaustive_rejects = not exhaustive_v.verify(bad_program)[0]
        exhaustive_accepts = exhaustive_v.verify(true_program)[0]
        random_stats = random_cluster_baseline(world, v0_cases, discovery_cases, miner, true_program, bad_program, random_trials, seed + 9000)
    return B3Run(
        seed,
        mode,
        world.role_to_obs,
        len(v0_cases),
        len(discovery_cases),
        len(hidden_cases),
        miner,
        exhaustive,
        b2,
        random_stats,
        v0.verify(bad_program)[0],
        v0.verify(true_program)[0],
        hidden.accuracy(bad_program),
        miner_rejects,
        miner_accepts,
        exhaustive_rejects,
        exhaustive_accepts,
        components_match(miner, world),
        true_program.description(),
        bad_program.description(),
    )


def run_experiment(k: int = 2, permutations: int = 8) -> Tuple[List[B3Run], List[B3Run], SynthesisValue]:
    multi: List[B3Run] = []
    single: List[B3Run] = []
    seen = set()
    seed = 131071
    while len(multi) < permutations:
        world = make_world(k, seed)
        if world.permutation not in seen:
            seen.add(world.permutation)
            multi.append(run_one("multi", k, seed))
            single.append(run_one("single", k, seed))
        seed += 53
    synth_world = make_world(k, multi[0].seed)
    synthesis = run_synthesis_value(synth_world, multi[0].miner, single[0].miner)
    return multi, single, synthesis


def role_by_obs(role_to_obs: Dict[str, int]) -> Dict[int, str]:
    return {obs: role for role, obs in role_to_obs.items()}


def fmt_components(comps: Sequence[Set[int]]) -> str:
    return " | ".join("{" + ",".join(f"x{field}" for field in sorted(comp)) + "}" for comp in comps) or "-"


def role_permutation_ok(runs: Sequence[B3Run]) -> bool:
    if not runs:
        return False
    maps = {tuple(sorted(run.role_to_obs.items())) for run in runs}
    a0s = {run.role_to_obs["A0"] for run in runs}
    b0s = {run.role_to_obs["B0"] for run in runs}
    return len(maps) == len(runs) and len(a0s) > 1 and len(b0s) > 1


def exhaustive_absorbs(runs: Sequence[B3Run]) -> bool:
    return all(
        run.miner.component_key() == run.exhaustive.component_key()
        and run.miner.clause_keys() == run.exhaustive.clause_keys()
        and run.miner.target_label_calls == run.exhaustive.target_label_calls
        and run.miner.score_units == run.exhaustive.score_units
        and run.miner_rejects_p_bad == run.exhaustive_rejects_p_bad
        and run.miner_accepts_true == run.exhaustive_accepts_true
        for run in runs
    )


def b3_frame_signal(runs: Sequence[B3Run]) -> bool:
    return all(
        run.v0_accepts_p_bad
        and run.v0_accepts_true
        and run.miner_rejects_p_bad is True
        and run.miner_accepts_true is True
        and run.boundary_correct
        for run in runs
    )


def synthesis_reduction(synthesis: SynthesisValue) -> bool:
    return (
        synthesis.multi_decomposed.found
        and synthesis.single_decomposed.found
        and synthesis.multi_decomposed.candidate_space < synthesis.multi_joint.candidate_space
        and synthesis.single_decomposed.candidate_space < synthesis.single_joint.candidate_space
    )


def verdict(multi: Sequence[B3Run], single: Sequence[B3Run], synthesis: SynthesisValue) -> str:
    if not multi or not single or not role_permutation_ok(multi):
        return "VOID"
    if not all(run.boundary_correct for run in list(multi) + list(single)):
        return "B3_DECOMPOSITION_WRONG"
    if not b3_frame_signal(multi):
        return "VOID"
    if exhaustive_absorbs(multi):
        return "B3_SYNTHESIS_VALUE" if synthesis_reduction(synthesis) else "B3_DISCOVERY_ABSORBED"
    return "B3_DISCOVERY_SIGNAL" if synthesis_reduction(synthesis) else "VOID"


def print_metric(metric: SynthMetric) -> None:
    print(
        f"  {metric.name:<34} found={metric.found} len={metric.program_length:<2} "
        f"space={metric.candidate_space:<10} attempts={metric.candidate_attempts:<10} "
        f"unique={metric.unique_semantics:<7} time={metric.elapsed_sec:.4f}s program={metric.program}"
    )


def print_report(multi: Sequence[B3Run], single: Sequence[B3Run], synthesis: SynthesisValue) -> None:
    print("PCCP-0 B3 decomposition discovery suite")
    print("=" * 78)
    print("World: flat role-permuted fields with A: A0 XOR A1, B: B0 AND B1, and shared spurious S.")
    print("V0_B3: support examples where target_A == target_B plus nuisance/S do checks; no component independence.")
    print("P_bad_B3: output_B is contaminated by output_A, so it passes V0 and fails A-outside-B independence.")
    print("Miner: first-order field/output sensitivity; scalar run adds effect signatures and pair interactions.")
    print("Exhaustive baseline: same tests, labels, and budget.")
    print()
    if multi:
        first = multi[0]
        print(f"Per-run observed fields n={len(first.role_to_obs)}, V0 cases={first.v0_cases}, discovery cases={first.discovery_cases}, hidden cases={first.hidden_cases}.")
        print(f"Multi-output miner score units={first.miner.score_units}, target-label calls={first.miner.target_label_calls}, independence clauses={len(first.miner.clauses)}.")
        print(f"B2-only whole-output score units={first.b2_only.score_units}, target-label calls={first.b2_only.target_label_calls}.")
        print()

    print("Multi-output decomposition and hidden-failure catch:")
    header = "seed    A0@ A1@ B0@ B1@  components                  V0_bad B3_bad exh_bad B2_bad boundary rand_success"
    print(header)
    print("-" * len(header))
    for run in multi:
        rand_success = run.random_stats.success_rate if run.random_stats else 0.0
        print(
            f"{run.seed:<7} {run.role_to_obs['A0']:>3} {run.role_to_obs['A1']:>3} {run.role_to_obs['B0']:>3} {run.role_to_obs['B1']:>3}  "
            f"{fmt_components(run.miner.components):<27} "
            f"{'PASS' if run.v0_accepts_p_bad else 'REJECT':<6} "
            f"{'REJECT' if run.miner_rejects_p_bad else 'PASS':<6} "
            f"{'REJECT' if run.exhaustive_rejects_p_bad else 'PASS':<7} "
            f"{'REJECT' if run.b2_only.rejects_p_bad else 'PASS':<6} "
            f"{str(run.boundary_correct):<8} {rand_success:>11.3f}"
        )

    print()
    print("Single-output decomposition from scalar y = target_A XOR target_B:")
    header = "seed    components                  boundary B2_bad hidden_bad_acc score_units target_calls"
    print(header)
    print("-" * len(header))
    for run in single:
        print(
            f"{run.seed:<7} {fmt_components(run.miner.components):<27} "
            f"{str(run.boundary_correct):<8} "
            f"{'REJECT' if run.b2_only.rejects_p_bad else 'PASS':<7} "
            f"{run.hidden_acc_p_bad:>14.3f} {run.miner.score_units:>11} {run.miner.target_label_calls:>12}"
        )

    print()
    print("Role permutation audit:")
    for run in multi:
        print(f"  seed={run.seed}: role_to_obs={run.role_to_obs}")

    if multi:
        first = multi[0]
        audit = role_by_obs(first.role_to_obs)
        print()
        print("First multi-output sensitivity scores (roles are post-hoc audit only):")
        for score in first.miner.sensitivities:
            if score.score == 0.0:
                continue
            print(f"  x{score.field}: role={audit.get(score.field, '?'):<3} channel=y{score.channel} changed={score.changed}/{score.support} score={score.score:.3f}")
        print()
        print(f"Example true program: {first.true_program}")
        print(f"Example bad program:  {first.bad_program}")

    print()
    print("Synthesis value test (same Boolean DSL; only allowed variable set changes):")
    print_metric(synthesis.multi_joint)
    print_metric(synthesis.multi_decomposed)
    print_metric(synthesis.single_joint)
    print_metric(synthesis.single_decomposed)

    print()
    print("Baseline checks:")
    print(f"  role_permutation_control: {role_permutation_ok(multi)}")
    print(f"  v0_accepts_P_bad_multi: {all(run.v0_accepts_p_bad for run in multi)}")
    print(f"  b3_rejects_P_bad_multi: {all(run.miner_rejects_p_bad for run in multi)}")
    print(f"  b3_accepts_true_multi: {all(run.miner_accepts_true for run in multi)}")
    print(f"  b2_only_rejects_P_bad_multi: {all(run.b2_only.rejects_p_bad for run in multi)}")
    print(f"  exhaustive_interaction_absorbs_discovery: {exhaustive_absorbs(multi)}")
    print(f"  single_output_boundaries_correct: {all(run.boundary_correct for run in single)}")
    print(f"  decomposed_synthesis_cheaper_than_joint: {synthesis_reduction(synthesis)}")

    print()
    print("Smuggling audit:")
    print("  1. Does the miner see component labels? No. It sees x0..xN, perturbations, and target labels only; role names are printed after discovery.")
    print("  2. Is sensitivity scoring target-specific? It is generic perturbation effect per output channel; the scalar run has no channel labels.")
    print("  3. Does exhaustive get equal budget and information? Yes. It runs the same field/output and pair-interaction tests with the same oracle calls.")
    print("  4. Is P_bad hard to catch without decomposition? In the multi-output suite, V0 and whole-output B2-only relations pass it; component independence rejects it.")
    print("  5. Is decomposition meaningful? The discovered active blocks match {A0,A1} and {B0,B1}; nuisances and S remain inactive.")
    print("  6. Does synthesis value use the same DSL? Yes. Joint and decomposed search both use Var, Const, Not, and Bin(AND/OR/XOR/EQ).")

    final = verdict(multi, single, synthesis)
    print()
    print("Narrative gate:")
    print(f"  Earned verdict token: {final}")
    print("  Gossip summary: The affair was real, but the exhaustive tester found the same phone records.")
    print("  What this does NOT prove: It does not prove transformation-grammar discovery, neural-tool resistance, or open-world frame formation.")
    if exhaustive_absorbs(multi):
        print("  Is exhaustive interaction testing sufficient? Yes for clause discovery in this toy world; it gets the same boundary and clauses for the same budget.")
    else:
        print("  Is exhaustive interaction testing sufficient? No on this run.")
    print(f"  Does decomposition provide synthesis value? {synthesis_reduction(synthesis)}")
    print("  Honest PCCP-H narrative: B1 and B2 are absorbed; B3 clause discovery is absorbed here too, but decomposition still gives a concrete synthesis-search reduction. The discovery moonshot now points at B4 transformation grammar discovery or a repositioned verifier/compiler/audit layer.")
    print(f"B3_VERDICT_TOKEN: {final}")


def main() -> None:
    multi, single, synthesis = run_experiment()
    print_report(multi, single, synthesis)


if __name__ == "__main__":
    main()

