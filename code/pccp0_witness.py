#!/usr/bin/env python3
"""PCCP-0 finite witness: finite SCM, DSL, verifier, synthesis, baselines."""
from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BitTuple = Tuple[int, ...]


# ----------------------------- DSL -----------------------------------------

class Expr:
    def eval(self, env: Dict[str, int]) -> int:
        raise NotImplementedError
    def length(self) -> int:
        raise NotImplementedError
    def vars_read(self) -> Tuple[str, ...]:
        raise NotImplementedError

@dataclass(frozen=True)
class Var(Expr):
    name: str
    def eval(self, env: Dict[str, int]) -> int:
        return int(env[self.name])
    def length(self) -> int:
        return 1
    def vars_read(self) -> Tuple[str, ...]:
        return (self.name,)
    def __str__(self) -> str:
        return self.name

@dataclass(frozen=True)
class Const(Expr):
    value: int
    def eval(self, env: Dict[str, int]) -> int:
        return int(self.value)
    def length(self) -> int:
        return 1
    def vars_read(self) -> Tuple[str, ...]:
        return ()
    def __str__(self) -> str:
        return "TRUE" if self.value else "FALSE"

@dataclass(frozen=True)
class Not(Expr):
    arg: Expr
    def eval(self, env: Dict[str, int]) -> int:
        return 1 - self.arg.eval(env)
    def length(self) -> int:
        return 1 + self.arg.length()
    def vars_read(self) -> Tuple[str, ...]:
        return self.arg.vars_read()
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
    def vars_read(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.left.vars_read()) | set(self.right.vars_read())))
    def __str__(self) -> str:
        return f"({self.op} {self.left} {self.right})"

@dataclass(frozen=True)
class If(Expr):
    cond: Expr
    then_branch: Expr
    else_branch: Expr
    def eval(self, env: Dict[str, int]) -> int:
        return self.then_branch.eval(env) if self.cond.eval(env) else self.else_branch.eval(env)
    def length(self) -> int:
        return 1 + self.cond.length() + self.then_branch.length() + self.else_branch.length()
    def vars_read(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.cond.vars_read()) | set(self.then_branch.vars_read()) | set(self.else_branch.vars_read())))
    def __str__(self) -> str:
        return f"(IF {self.cond} THEN {self.then_branch} ELSE {self.else_branch})"

DSL_PRIMITIVES = (
    "Var(name): typed Boolean variable access",
    "Const(False/True): Boolean constants",
    "Not(expr): Boolean negation",
    "Bin(AND/OR/XOR/EQ,left,right): generic Boolean connectives/equality",
    "If(cond,then,else): bounded Boolean conditional",
)


# ----------------------------- finite world --------------------------------

@dataclass(frozen=True)
class Intervention:
    family: str
    c0: Optional[int] = None
    c1: Optional[int] = None
    n: Optional[BitTuple] = None
    s: Optional[int] = None

@dataclass(frozen=True)
class Case:
    world_id: str
    obs: BitTuple
    query: str
    intervention: Intervention
    target: int
    split: str
    def key(self) -> Tuple[BitTuple, Tuple[object, ...], str]:
        i = self.intervention
        return self.obs, (i.family, i.c0, i.c1, i.n, i.s), self.query

@dataclass
class World:
    m: int
    permutation: Tuple[int, ...]
    @property
    def world_id(self) -> str:
        return f"W_m={self.m}_perm={','.join(map(str, self.permutation))}"
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
    def target_rule(self, c: BitTuple) -> int:
        return c[0] ^ c[1]
    def encode(self, c: BitTuple, n: BitTuple, s: int) -> BitTuple:
        latent = tuple(c) + tuple(n) + (s,)
        return tuple(latent[self.permutation[pos]] for pos in range(len(latent)))
    def factual_states(self) -> Iterable[Tuple[BitTuple, BitTuple]]:
        for c in product((0, 1), repeat=2):
            for n in product((0, 1), repeat=self.m):
                yield tuple(c), tuple(n)
    def all_n_values(self) -> List[BitTuple]:
        return [tuple(n) for n in product((0, 1), repeat=self.m)]
    def make_case(self, split: str, family: str, c: BitTuple, n: BitTuple,
                  c0: Optional[int] = None, c1: Optional[int] = None,
                  n_surface: Optional[BitTuple] = None,
                  s_surface: Optional[int] = None) -> Case:
        obs_n = n if n_surface is None else n_surface
        obs_s = self.target_rule(c) if s_surface is None else s_surface
        obs = self.encode(c, obs_n, obs_s)
        eff_c0 = c[0] if c0 is None else c0
        eff_c1 = c[1] if c1 is None else c1
        target = eff_c0 ^ eff_c1
        return Case(self.world_id, obs, "target_parity", Intervention(family, c0, c1, n_surface, s_surface), target, split)
    def seen_cases(self) -> List[Case]:
        cases: List[Case] = []
        for c, n in self.factual_states():
            cases.append(self.make_case("seen", "id", c, n))
            for s in (0, 1):
                cases.append(self.make_case("seen", "environment_shift_seen", c, n, s_surface=s))
            for value in (0, 1):
                cases.append(self.make_case("seen", "do_c0_seen", c, n, c0=value))
                cases.append(self.make_case("seen", "do_c1_seen", c, n, c1=value))
        return cases
    def hidden_cases(self) -> List[Case]:
        cases: List[Case] = []
        n_values = self.all_n_values()
        for c, n in self.factual_states():
            for n_prime in n_values:
                cases.append(self.make_case("hidden", "do_n_hidden", c, n, n_surface=n_prime))
            for s_prime in (0, 1):
                cases.append(self.make_case("hidden", "do_s_hidden", c, n, s_surface=s_prime))
            for value in (0, 1):
                cases.append(self.make_case("hidden", "counterfactual_hold_c0_hidden", c, n, c0=value))
                cases.append(self.make_case("hidden", "counterfactual_hold_c1_hidden", c, n, c1=value))
                cases.append(self.make_case("hidden", "composition_shift_hidden", c, n, c0=value,
                                            n_surface=tuple(1 - bit for bit in n),
                                            s_surface=1 - self.target_rule(c)))
        return cases


def make_world_with_seed(m: int, seed: int) -> World:
    perm = list(range(2 + m + 1))
    random.Random(seed).shuffle(perm)
    return World(m, tuple(perm))


def make_world(m: int) -> World:
    return make_world_with_seed(m, 1729 + m)


def variable_names(m: int, n_obs: int) -> List[str]:
    names = [f"x{i}" for i in range(n_obs)]
    names += ["has_c0", "val_c0", "has_c1", "val_c1", "has_s", "val_s"]
    for j in range(m):
        names += [f"has_n{j}", f"val_n{j}"]
    return names


def env_for_case(case: Case, m: int) -> Dict[str, int]:
    env: Dict[str, int] = {f"x{i}": bit for i, bit in enumerate(case.obs)}
    i = case.intervention
    env.update({
        "has_c0": int(i.c0 is not None), "val_c0": 0 if i.c0 is None else int(i.c0),
        "has_c1": int(i.c1 is not None), "val_c1": 0 if i.c1 is None else int(i.c1),
        "has_s": int(i.s is not None), "val_s": 0 if i.s is None else int(i.s),
    })
    for j in range(m):
        env[f"has_n{j}"] = int(i.n is not None)
        env[f"val_n{j}"] = 0 if i.n is None else int(i.n[j])
    return env

# ----------------------------- exact verifier -------------------------------

class Verifier:
    def __init__(self, world: World, cases: Sequence[Case], name: str):
        self.world = world
        self.cases = list(cases)
        self.name = name
    def verify(self, program: Expr) -> Tuple[bool, Optional[Tuple[int, Case, int]]]:
        for idx, case in enumerate(self.cases):
            actual = program.eval(env_for_case(case, self.world.m))
            if actual != case.target:
                return False, (idx, case, actual)
        return True, None
    def accuracy_expr(self, program: Expr) -> float:
        if not self.cases:
            return 1.0
        return sum(program.eval(env_for_case(c, self.world.m)) == c.target for c in self.cases) / len(self.cases)
    def accuracy_predictor(self, predictor: "Predictor") -> float:
        if not self.cases:
            return 1.0
        return sum(predictor.predict(c, self.world.m) == c.target for c in self.cases) / len(self.cases)
    def family_accuracy_predictor(self, predictor: "Predictor") -> Dict[str, float]:
        buckets: Dict[str, List[int]] = {}
        for case in self.cases:
            buckets.setdefault(case.intervention.family, []).append(
                int(predictor.predict(case, self.world.m) == case.target)
            )
        return {family: sum(vals) / len(vals) for family, vals in sorted(buckets.items())}


# ----------------------------- synthesis ------------------------------------

@dataclass
class SynthResult:
    program: Expr
    length: int
    generated_semantics: int
    elapsed_sec: float


def mask_for_cases(cases: Sequence[Case]) -> int:
    mask = 0
    for idx, case in enumerate(cases):
        if case.target:
            mask |= 1 << idx
    return mask


def semantics(expr: Expr, envs: Sequence[Dict[str, int]]) -> int:
    bits = 0
    for idx, env in enumerate(envs):
        if expr.eval(env):
            bits |= 1 << idx
    return bits


def add_expr(by_len: Dict[int, Dict[int, Expr]], expr: Expr, sem: int) -> None:
    length = expr.length()
    by_len.setdefault(length, {})
    if sem not in by_len[length]:
        by_len[length][sem] = expr


def synthesize(seen: Verifier, max_len: int = 9) -> SynthResult:
    """Enumerate the declared PCCP-0 grammar in length order.

    Search grammar:
      basis := expressions of length <= 4 from atoms, Not, binary Boolean ops,
               and atom-guarded If nodes;
      program := basis OR Bin(op, basis, basis) with total length <= max_len.

    This finite grammar is generic and target-agnostic. It contains no parity,
    causal-parent, nuisance, spurious, or verifier oracle primitive.
    """
    started = time.time()
    envs = [env_for_case(case, seen.world.m) for case in seen.cases]
    full_mask = (1 << len(seen.cases)) - 1
    target = mask_for_cases(seen.cases)
    atoms: List[Expr] = [Var(name) for name in variable_names(seen.world.m, seen.world.n_obs)]
    atoms += [Const(0), Const(1)]
    atom_sem = {expr: semantics(expr, envs) for expr in atoms}
    by_len: Dict[int, Dict[int, Expr]] = {}

    for expr, sem in atom_sem.items():
        add_expr(by_len, expr, sem)
    for expr, sem in list(atom_sem.items()):
        add_expr(by_len, Not(expr), full_mask ^ sem)

    ops = ("AND", "OR", "XOR", "EQ")
    atom_items = list(atom_sem.items())
    for i, (left, left_sem) in enumerate(atom_items):
        for j, (right, right_sem) in enumerate(atom_items):
            if j < i:
                continue
            for op in ops:
                if op == "AND":
                    sem = left_sem & right_sem
                elif op == "OR":
                    sem = left_sem | right_sem
                elif op == "XOR":
                    sem = left_sem ^ right_sem
                else:
                    sem = full_mask ^ (left_sem ^ right_sem)
                add_expr(by_len, Bin(op, left, right), sem)

    for sem, expr in list(by_len.get(2, {}).items()):
        add_expr(by_len, Not(expr), full_mask ^ sem)

    for cond, cond_sem in atom_items:
        not_cond = full_mask ^ cond_sem
        for then_expr, then_sem in atom_items:
            then_part = cond_sem & then_sem
            for else_expr, else_sem in atom_items:
                add_expr(by_len, If(cond, then_expr, else_expr), then_part | (not_cond & else_sem))

    generated = sum(len(v) for v in by_len.values())
    for length in range(1, min(4, max_len) + 1):
        expr = by_len.get(length, {}).get(target)
        if expr is not None:
            return SynthResult(expr, length, generated, time.time() - started)

    for length in range(5, max_len + 1):
        for left_len in range(1, min(4, length - 2) + 1):
            right_len = length - 1 - left_len
            if right_len < 1 or right_len > 4:
                continue
            left_map, right_map = by_len.get(left_len, {}), by_len.get(right_len, {})
            if not left_map or not right_map:
                continue
            for left_sem, left_expr in left_map.items():
                right_expr = right_map.get(target ^ left_sem)
                if right_expr is not None:
                    expr = Bin("XOR", left_expr, right_expr)
                    return SynthResult(expr, expr.length(), generated, time.time() - started)
                right_expr = right_map.get(left_sem ^ (full_mask ^ target))
                if right_expr is not None:
                    expr = Bin("EQ", left_expr, right_expr)
                    return SynthResult(expr, expr.length(), generated, time.time() - started)
            if len(left_map) * len(right_map) <= 250_000:
                for left_sem, left_expr in left_map.items():
                    for right_sem, right_expr in right_map.items():
                        if (left_sem & right_sem) == target:
                            expr = Bin("AND", left_expr, right_expr)
                            return SynthResult(expr, expr.length(), generated, time.time() - started)
                        if (left_sem | right_sem) == target:
                            expr = Bin("OR", left_expr, right_expr)
                            return SynthResult(expr, expr.length(), generated, time.time() - started)
    raise RuntimeError(f"no program found up to length {max_len}")

# ----------------------------- baselines ------------------------------------

class Predictor:
    length: int
    def predict(self, case: Case, m: int) -> int:
        raise NotImplementedError

class ExprPredictor(Predictor):
    def __init__(self, expr: Expr, length: Optional[int] = None):
        self.expr = expr
        self.length = expr.length() if length is None else length
    def predict(self, case: Case, m: int) -> int:
        return self.expr.eval(env_for_case(case, m))

class LookupTable(Predictor):
    def __init__(self, cases: Sequence[Case]):
        self.table = {case.key(): case.target for case in cases}
        ones = sum(case.target for case in cases)
        self.default = int(ones * 2 >= len(cases))
        self.length = len(self.table)
    def predict(self, case: Case, m: int) -> int:
        return self.table.get(case.key(), self.default)

@dataclass
class TreeNode:
    prediction: int
    feature: Optional[str] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    def predict(self, env: Dict[str, int]) -> int:
        if self.feature is None:
            return self.prediction
        child = self.right if env[self.feature] else self.left
        return self.prediction if child is None else child.predict(env)
    def size(self) -> int:
        if self.feature is None:
            return 1
        return 1 + (0 if self.left is None else self.left.size()) + (0 if self.right is None else self.right.size())

class DecisionTree(Predictor):
    def __init__(self, cases: Sequence[Case], world: World):
        self.world = world
        self.features = variable_names(world.m, world.n_obs)
        rows = [(env_for_case(case, world.m), case.target) for case in cases]
        self.root = self._build(rows, self.features)
        self.length = self.root.size()
    def predict(self, case: Case, m: int) -> int:
        return self.root.predict(env_for_case(case, m))
    def _build(self, rows: List[Tuple[Dict[str, int], int]], features: List[str]) -> TreeNode:
        ones = sum(label for _, label in rows)
        pred = int(ones * 2 >= len(rows))
        if ones == 0 or ones == len(rows) or not features:
            return TreeNode(pred)
        base = entropy(ones, len(rows) - ones)
        best_feature, best_gain = None, -1.0
        for feature in features:
            left = [label for env, label in rows if env[feature] == 0]
            right = [label for env, label in rows if env[feature] == 1]
            if not left or not right:
                continue
            gain = base
            gain -= (len(left) / len(rows)) * entropy(sum(left), len(left) - sum(left))
            gain -= (len(right) / len(rows)) * entropy(sum(right), len(right) - sum(right))
            if gain > best_gain + 1e-12:
                best_feature, best_gain = feature, gain
        if best_feature is None:
            return TreeNode(pred)
        rest = [feature for feature in features if feature != best_feature]
        left_rows = [(env, label) for env, label in rows if env[best_feature] == 0]
        right_rows = [(env, label) for env, label in rows if env[best_feature] == 1]
        return TreeNode(pred, best_feature, self._build(left_rows, rest), self._build(right_rows, rest))


def entropy(ones: int, zeros: int) -> float:
    total = ones + zeros
    if total == 0 or ones == 0 or zeros == 0:
        return 0.0
    p, q = ones / total, zeros / total
    return -(p * math.log2(p) + q * math.log2(q))


def reconstruction_proxy_baseline(world: World) -> ExprPredictor:
    """Identity reconstructor over C+N+S plus shortest observational extractor S."""
    s_pos = world.role_to_obs["S"]
    extractor = Var(f"x{s_pos}")
    return ExprPredictor(extractor, length=world.n_obs + extractor.length())


def reconstruction_verifier_aware_control(world: World, pccp_program: Expr) -> ExprPredictor:
    return ExprPredictor(pccp_program, length=world.n_obs + pccp_program.length())


def random_expr_of_length(length: int, var_names_: Sequence[str], rng: random.Random) -> Expr:
    if length <= 1:
        return Const(rng.randint(0, 1)) if rng.random() < 0.12 else Var(rng.choice(var_names_))
    if length == 2:
        return Not(random_expr_of_length(1, var_names_, rng))
    if length >= 4 and rng.random() < 0.28:
        remaining = length - 1
        a = rng.randint(1, remaining - 2)
        b = rng.randint(1, remaining - a - 1)
        c = remaining - a - b
        return If(random_expr_of_length(a, var_names_, rng),
                  random_expr_of_length(b, var_names_, rng),
                  random_expr_of_length(c, var_names_, rng))
    left_len = rng.randint(1, length - 2)
    right_len = length - 1 - left_len
    return Bin(rng.choice(("AND", "OR", "XOR", "EQ")),
               random_expr_of_length(left_len, var_names_, rng),
               random_expr_of_length(right_len, var_names_, rng))


def random_program_baseline(world: World, length: int, seen: Verifier, hidden: Verifier,
                            samples: int = 16) -> Dict[str, float]:
    rng = random.Random(9001 + world.m)
    names = variable_names(world.m, world.n_obs)
    seen_accs, hidden_accs = [], []
    passes_seen = 0
    for _ in range(samples):
        expr = random_expr_of_length(length, names, rng)
        seen_acc = seen.accuracy_expr(expr)
        hidden_acc = hidden.accuracy_expr(expr)
        seen_accs.append(seen_acc)
        hidden_accs.append(hidden_acc)
        passes_seen += int(seen_acc == 1.0)
    return {
        "samples": samples,
        "seen_avg": sum(seen_accs) / samples,
        "hidden_avg": sum(hidden_accs) / samples,
        "hidden_best": max(hidden_accs),
        "seen_pass_rate": passes_seen / samples,
    }

# ----------------------------- FDM-0 discovery ------------------------------

BINARY_DOMAIN = (0, 1)


@dataclass(frozen=True)
class InvarianceClause:
    field_index: int
    stable_score: float
    shortcut_score: float
    support: int
    mdl_score: float
    candidate_kind: str
    def label(self) -> str:
        return f"invariant_to(x{self.field_index})"


@dataclass(frozen=True)
class FieldEffect:
    field_index: int
    stable_score: float
    shortcut_score: float
    support: int
    changed: int
    mdl_score: float


@dataclass(frozen=True)
class RandomClauseStats:
    trials: int
    query_pairs_per_trial: int
    success_rate: float
    hit_spurious_rate: float
    mean_unique_clauses: float


@dataclass
class FDMRun:
    seed: int
    m: int
    world_id: str
    role_to_obs: Dict[str, int]
    v0_cases: int
    hidden_s_cases: int
    query_pairs: int
    fdm_effects: List[FieldEffect]
    fdm_clauses: List[InvarianceClause]
    exhaustive_clauses: List[InvarianceClause]
    random_stats: RandomClauseStats
    v0_accepts_p_bad: bool
    v0_accepts_true: bool
    hidden_s_acc_p_bad: float
    fdm_rejects_p_bad: bool
    fdm_accepts_true: bool
    exhaustive_rejects_p_bad: bool
    exhaustive_accepts_true: bool
    no_discovery_rejects_p_bad: bool
    fdm_found_spurious: bool
    p_bad_program: str
    true_program: str


def decode_latent(world: World, obs: BitTuple) -> BitTuple:
    latent = [0] * world.n_obs
    for obs_pos, latent_idx in enumerate(world.permutation):
        latent[latent_idx] = int(obs[obs_pos])
    return tuple(latent)


def target_oracle_for_obs(world: World, case: Case, obs: BitTuple) -> int:
    """Grounding oracle for FDM paired traces.

    The caller supplies only a perturbed observation tuple. The oracle evaluates
    the target function for that tuple and the case's existing intervention
    descriptor. FDM-0 gets the returned label, not the latent role mapping used
    internally to generate it.
    """
    latent = decode_latent(world, obs)
    c0, c1 = latent[0], latent[1]
    i = case.intervention
    eff_c0 = c0 if i.c0 is None else int(i.c0)
    eff_c1 = c1 if i.c1 is None else int(i.c1)
    return eff_c0 ^ eff_c1


def replace_observed_field(obs: BitTuple, field_index: int, value: int) -> BitTuple:
    updated = list(obs)
    updated[field_index] = int(value)
    return tuple(updated)


def fdm_v0_cases(world: World) -> List[Case]:
    """Partial verifier V0: examples, id, and do(C:=c') only."""
    cases: List[Case] = []
    for c, n in world.factual_states():
        cases.append(world.make_case("fdm_v0", "id", c, n))
        for value in BINARY_DOMAIN:
            cases.append(world.make_case("fdm_v0", "do_c0_seen", c, n, c0=value))
            cases.append(world.make_case("fdm_v0", "do_c1_seen", c, n, c1=value))
    return cases


def fdm_hidden_s_cases(world: World) -> List[Case]:
    cases: List[Case] = []
    for c, n in world.factual_states():
        for s_value in BINARY_DOMAIN:
            cases.append(world.make_case("fdm_hidden", "do_s_hidden", c, n, s_surface=s_value))
    return cases


def true_causal_program(world: World) -> Expr:
    c0_pos = world.role_to_obs["C0"]
    c1_pos = world.role_to_obs["C1"]
    c0 = If(Var("has_c0"), Var("val_c0"), Var(f"x{c0_pos}"))
    c1 = If(Var("has_c1"), Var("val_c1"), Var(f"x{c1_pos}"))
    return Bin("XOR", c0, c1)


def spurious_shortcut_program(world: World) -> Expr:
    """Bad shortcut: treat S as the factual target and patch seen C overrides."""
    s_pos = world.role_to_obs["S"]
    c0_pos = world.role_to_obs["C0"]
    c1_pos = world.role_to_obs["C1"]
    s = Var(f"x{s_pos}")
    c0 = Var(f"x{c0_pos}")
    c1 = Var(f"x{c1_pos}")
    both_overridden = Bin("XOR", Var("val_c0"), Var("val_c1"))
    c0_overridden = Bin("XOR", Bin("XOR", s, c0), Var("val_c0"))
    c1_overridden = Bin("XOR", Bin("XOR", s, c1), Var("val_c1"))
    c1_branch = If(Var("has_c1"), c1_overridden, s)
    c0_branch = If(Var("has_c1"), both_overridden, c0_overridden)
    return If(Var("has_c0"), c0_branch, c1_branch)


def mutual_information(pairs: Sequence[Tuple[int, int]]) -> float:
    total = len(pairs)
    if total == 0:
        return 0.0
    joint = Counter(pairs)
    xs = Counter(x for x, _ in pairs)
    ys = Counter(y for _, y in pairs)
    mi = 0.0
    for (x, y), count in joint.items():
        pxy = count / total
        px = xs[x] / total
        py = ys[y] / total
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def field_effect_score(world: World, cases: Sequence[Case], field_index: int) -> FieldEffect:
    stable = 0
    total = 0
    for case in cases:
        y = target_oracle_for_obs(world, case, case.obs)
        for value in BINARY_DOMAIN:
            if value == case.obs[field_index]:
                continue
            obs_prime = replace_observed_field(case.obs, field_index, value)
            y_prime = target_oracle_for_obs(world, case, obs_prime)
            stable += int(y_prime == y)
            total += 1
    id_pairs = [(case.obs[field_index], case.target) for case in cases if case.intervention.family == "id"]
    shortcut = mutual_information(id_pairs)
    stable_score = stable / total if total else 1.0
    clause_bits = 1 + math.ceil(math.log2(max(2, world.n_obs)))
    mdl_score = stable_score * total + shortcut * max(1, len(id_pairs)) - clause_bits
    return FieldEffect(field_index, stable_score, shortcut, total, total - stable, mdl_score)


def fdm0_discover(world: World, cases: Sequence[Case], stable_threshold: float = 0.999,
                  shortcut_threshold: float = 0.50) -> Tuple[List[FieldEffect], List[InvarianceClause]]:
    effects = [field_effect_score(world, cases, j) for j in range(world.n_obs)]
    clauses: List[InvarianceClause] = []
    for effect in effects:
        if effect.stable_score >= stable_threshold:
            kind = "spurious_candidate" if effect.shortcut_score >= shortcut_threshold else "stable_invariant"
            clauses.append(InvarianceClause(effect.field_index, effect.stable_score,
                                            effect.shortcut_score, effect.support,
                                            effect.mdl_score, kind))
    clauses.sort(key=lambda clause: (-clause.mdl_score, clause.field_index))
    return effects, clauses


def exhaustive_single_field_clauses(effects: Sequence[FieldEffect],
                                    stable_threshold: float = 0.999) -> List[InvarianceClause]:
    clauses = [
        InvarianceClause(effect.field_index, effect.stable_score, effect.shortcut_score,
                         effect.support, effect.mdl_score, "exhaustive_stable")
        for effect in effects
        if effect.stable_score >= stable_threshold
    ]
    clauses.sort(key=lambda clause: (-clause.stable_score, clause.field_index))
    return clauses


def compile_invariance_verifier(world: World, base_cases: Sequence[Case],
                                clauses: Sequence[InvarianceClause], name: str) -> Verifier:
    compiled = list(base_cases)
    for clause in clauses:
        field_index = clause.field_index
        for case in base_cases:
            for value in BINARY_DOMAIN:
                if value == case.obs[field_index]:
                    continue
                obs_prime = replace_observed_field(case.obs, field_index, value)
                compiled.append(Case(case.world_id, obs_prime, case.query,
                                     case.intervention, case.target, "discovered"))
    return Verifier(world, compiled, name)


def random_clause_search_baseline(world: World, base_cases: Sequence[Case],
                                  effects: Sequence[FieldEffect], p_bad: Expr,
                                  true_program: Expr, trials: int, seed: int,
                                  stable_threshold: float = 0.999) -> RandomClauseStats:
    rng = random.Random(seed)
    successes = 0
    hit_spurious = 0
    unique_counts: List[int] = []
    spurious_field = world.role_to_obs["S"]
    for _ in range(trials):
        selected: List[int] = []
        for _scan in range(world.n_obs):
            field_index = rng.randrange(world.n_obs)
            effect = effects[field_index]
            if effect.stable_score >= stable_threshold and field_index not in selected:
                selected.append(field_index)
        clauses = [
            InvarianceClause(effects[j].field_index, effects[j].stable_score,
                             effects[j].shortcut_score, effects[j].support,
                             effects[j].mdl_score, "random_stable")
            for j in selected
        ]
        verifier = compile_invariance_verifier(world, base_cases, clauses, "random_clause_v1")
        rejects_p_bad = not verifier.verify(p_bad)[0]
        accepts_true = verifier.verify(true_program)[0]
        successes += int(rejects_p_bad and accepts_true)
        hit_spurious += int(spurious_field in selected)
        unique_counts.append(len(selected))
    query_pairs = world.n_obs * len(base_cases)
    return RandomClauseStats(
        trials=trials,
        query_pairs_per_trial=query_pairs,
        success_rate=successes / trials if trials else 0.0,
        hit_spurious_rate=hit_spurious / trials if trials else 0.0,
        mean_unique_clauses=sum(unique_counts) / trials if trials else 0.0,
    )


def run_fdm0_one(m: int, seed: int, random_trials: int = 96) -> FDMRun:
    world = make_world_with_seed(m, seed)
    v0_cases = fdm_v0_cases(world)
    for case in v0_cases:
        expected = target_oracle_for_obs(world, case, case.obs)
        if expected != case.target:
            raise AssertionError("FDM target oracle disagrees with V0 case label")
    v0 = Verifier(world, v0_cases, "FDM_V0")
    hidden_s = Verifier(world, fdm_hidden_s_cases(world), "FDM_hidden_do_s")
    p_bad = spurious_shortcut_program(world)
    true_program = true_causal_program(world)

    v0_accepts_p_bad = v0.verify(p_bad)[0]
    v0_accepts_true = v0.verify(true_program)[0]
    hidden_s_acc_p_bad = hidden_s.accuracy_expr(p_bad)

    effects, fdm_clauses = fdm0_discover(world, v0_cases)
    fdm_v1 = compile_invariance_verifier(world, v0_cases, fdm_clauses, "FDM_V1")
    fdm_rejects_p_bad = not fdm_v1.verify(p_bad)[0]
    fdm_accepts_true = fdm_v1.verify(true_program)[0]

    exhaustive_clauses = exhaustive_single_field_clauses(effects)
    exhaustive_v1 = compile_invariance_verifier(world, v0_cases, exhaustive_clauses,
                                                "exhaustive_single_field_v1")
    exhaustive_rejects_p_bad = not exhaustive_v1.verify(p_bad)[0]
    exhaustive_accepts_true = exhaustive_v1.verify(true_program)[0]

    random_stats = random_clause_search_baseline(world, v0_cases, effects, p_bad,
                                                 true_program, random_trials,
                                                 seed + 100_000)
    spurious_field = world.role_to_obs["S"]
    fdm_found_spurious = any(clause.field_index == spurious_field and
                             clause.candidate_kind == "spurious_candidate"
                             for clause in fdm_clauses)
    query_pairs = world.n_obs * len(v0_cases)
    return FDMRun(
        seed=seed,
        m=m,
        world_id=world.world_id,
        role_to_obs=world.role_to_obs,
        v0_cases=len(v0_cases),
        hidden_s_cases=len(hidden_s.cases),
        query_pairs=query_pairs,
        fdm_effects=effects,
        fdm_clauses=fdm_clauses,
        exhaustive_clauses=exhaustive_clauses,
        random_stats=random_stats,
        v0_accepts_p_bad=v0_accepts_p_bad,
        v0_accepts_true=v0_accepts_true,
        hidden_s_acc_p_bad=hidden_s_acc_p_bad,
        fdm_rejects_p_bad=fdm_rejects_p_bad,
        fdm_accepts_true=fdm_accepts_true,
        exhaustive_rejects_p_bad=exhaustive_rejects_p_bad,
        exhaustive_accepts_true=exhaustive_accepts_true,
        no_discovery_rejects_p_bad=not v0_accepts_p_bad,
        fdm_found_spurious=fdm_found_spurious,
        p_bad_program=str(p_bad),
        true_program=str(true_program),
    )


def run_fdm0_experiment(m: int = 4, permutations: int = 8) -> List[FDMRun]:
    runs: List[FDMRun] = []
    seen_perms = set()
    seed = 73_001
    while len(runs) < permutations:
        world = make_world_with_seed(m, seed)
        if world.permutation not in seen_perms:
            seen_perms.add(world.permutation)
            runs.append(run_fdm0_one(m, seed))
        seed += 37
    return runs


def fdm_frame_signal(runs: Sequence[FDMRun]) -> bool:
    return all(
        run.v0_accepts_p_bad and
        run.v0_accepts_true and
        run.hidden_s_acc_p_bad < 1.0 and
        run.fdm_rejects_p_bad and
        run.fdm_accepts_true and
        run.fdm_found_spurious
        for run in runs
    )


def fdm_absorbed_by_exhaustive(runs: Sequence[FDMRun]) -> bool:
    return all(run.exhaustive_rejects_p_bad and run.exhaustive_accepts_true for run in runs)


def fdm_role_permutation_ok(runs: Sequence[FDMRun]) -> bool:
    s_positions = {run.role_to_obs["S"] for run in runs}
    permutations = {tuple(sorted(run.role_to_obs.items())) for run in runs}
    return len(s_positions) > 1 and len(permutations) == len(runs)


def fdm_verdict(runs: Sequence[FDMRun]) -> str:
    if not runs or not fdm_role_permutation_ok(runs) or not fdm_frame_signal(runs):
        return "VOID"
    if fdm_absorbed_by_exhaustive(runs):
        return "DISCOVERY_ABSORBED"
    return "FRAME_SIGNAL"


def role_by_obs_index(role_to_obs: Dict[str, int]) -> Dict[int, str]:
    return {obs_index: role for role, obs_index in role_to_obs.items()}


def print_fdm_report(runs: Sequence[FDMRun]) -> None:
    print()
    print("FDM-0 frame discovery experiment")
    print("=" * 78)
    print("V0: id examples plus do(C0:=v) and do(C1:=v); no S/N invariance clauses.")
    print("P_bad: spurious shortcut using S as factual target, with patches for seen C overrides.")
    print("FDM input fields are only x0..xN; role maps below are post-hoc audit data.")
    print("Perturbation grammar: generic single observed-field binary replacement.")
    if runs:
        print(f"Per-run FDM query budget: {runs[0].query_pairs} paired perturbations "
              f"({runs[0].query_pairs * 2} target-label calls if base labels are recounted).")
        print(f"Random clause baseline gets {runs[0].random_stats.query_pairs_per_trial} paired "
              "perturbations per trial over the same V0 cases.")
    print()
    header = ("seed   S@  FDM_spurious  FDM_invariants        V0_Pbad  hiddenS  "
              "V1_Pbad  V1_true  exhaustive  random_success")
    print(header)
    print("-" * len(header))
    for run in runs:
        spurious = [f"x{clause.field_index}" for clause in run.fdm_clauses
                    if clause.candidate_kind == "spurious_candidate"]
        invariants = [f"x{clause.field_index}" for clause in run.fdm_clauses]
        print(f"{run.seed:<6} {run.role_to_obs['S']:>2}  "
              f"{','.join(spurious) or '-':<13} "
              f"{','.join(invariants):<21} "
              f"{'PASS' if run.v0_accepts_p_bad else 'REJECT':<8} "
              f"{run.hidden_s_acc_p_bad:>7.3f}  "
              f"{'REJECT' if run.fdm_rejects_p_bad else 'PASS':<7} "
              f"{'PASS' if run.fdm_accepts_true else 'REJECT':<7} "
              f"{'REJECT' if run.exhaustive_rejects_p_bad else 'PASS':<10} "
              f"{run.random_stats.success_rate:>6.3f}")
    print()
    print("Role permutation audit:")
    for run in runs:
        print(f"  seed={run.seed}: role_to_obs={run.role_to_obs}")
    print()
    if runs:
        first = runs[0]
        audit_roles = role_by_obs_index(first.role_to_obs)
        print("First permutation field scores (roles shown only after discovery for audit):")
        for effect in first.fdm_effects:
            role = audit_roles.get(effect.field_index, "?")
            print(f"  x{effect.field_index}: role={role:<2} stable={effect.stable_score:.3f} "
                  f"shortcut_MI={effect.shortcut_score:.3f} changed={effect.changed}/{effect.support} "
                  f"mdl={effect.mdl_score:.2f}")
        print()
        print(f"Example P_bad: {first.p_bad_program}")
        print(f"Example true causal program: {first.true_program}")
        print()
    frame_signal = fdm_frame_signal(runs)
    exhaustive_absorbs = fdm_absorbed_by_exhaustive(runs)
    print(f"FRAME_SIGNAL_CONDITION: {frame_signal}")
    print(f"EXHAUSTIVE_SINGLE_FIELD_ABSORBS: {exhaustive_absorbs}")
    print(f"ROLE_PERMUTATION_CONTROL: {fdm_role_permutation_ok(runs)}")
    print(f"FDM_VERDICT_TOKEN: {fdm_verdict(runs)}")
    print("FDM_GOSSIP_SUMMARY: The shortcut wore a fake badge, and field-toggling found the badge did not change the job.")
    print("FDM_SCOPE_LIMITS: This is B1 role discovery under a given finite field-replacement grammar and an exact target oracle; it does not prove transformation-grammar discovery, open-world verifier discovery, or novelty over spec mining.")
    if exhaustive_absorbs:
        print("EXHAUSTIVE_SUFFICIENCY: Yes. Exhaustive single-field invariance checking gets the same catch here, so FDM-0 is absorbed on this smallest demo.")
    else:
        print("EXHAUSTIVE_SUFFICIENCY: No on this run; exhaustive single-field checking did not match FDM-0.")

# ----------------------------- experiment -----------------------------------

@dataclass
class Row:
    m: int
    seen_cases: int
    hidden_cases: int
    pccp_length: int
    pccp_hidden: float
    lookup_length: int
    lookup_hidden: float
    tree_length: int
    tree_hidden: float
    recon_length: int
    recon_hidden: float
    recon_control_length: int
    recon_control_hidden: float
    random_hidden_avg: float
    random_hidden_best: float
    synth_sec: float
    program: str
    role_to_obs: Dict[str, int]
    failures: Dict[str, Dict[str, float]]


def run_one(m: int) -> Row:
    world = make_world(m)
    seen = Verifier(world, world.seen_cases(), "seen")
    hidden = Verifier(world, world.hidden_cases(), "hidden")
    synth = synthesize(seen)
    seen_pass, seen_ce = seen.verify(synth.program)
    if not seen_pass:
        raise AssertionError(f"synthesized program failed seen verifier: {seen_ce}")
    hidden_pass, _ = hidden.verify(synth.program)

    pccp = ExprPredictor(synth.program)
    lookup = LookupTable(seen.cases)
    tree = DecisionTree(seen.cases, world)
    recon = reconstruction_proxy_baseline(world)
    recon_control = reconstruction_verifier_aware_control(world, synth.program)
    rand = random_program_baseline(world, synth.length, seen, hidden)

    predictors: List[Tuple[str, Predictor]] = [
        ("lookup", lookup),
        ("decision_tree", tree),
        ("reconstruction_proxy", recon),
        ("reconstruction_verifier_aware_control", recon_control),
        ("pccp", pccp),
    ]
    failures: Dict[str, Dict[str, float]] = {}
    for name, predictor in predictors:
        fam_acc = hidden.family_accuracy_predictor(predictor)
        bad = {family: acc for family, acc in fam_acc.items() if acc < 1.0}
        if bad:
            failures[name] = bad

    return Row(
        m=m,
        seen_cases=len(seen.cases),
        hidden_cases=len(hidden.cases),
        pccp_length=synth.length,
        pccp_hidden=1.0 if hidden_pass else hidden.accuracy_expr(synth.program),
        lookup_length=lookup.length,
        lookup_hidden=hidden.accuracy_predictor(lookup),
        tree_length=tree.length,
        tree_hidden=hidden.accuracy_predictor(tree),
        recon_length=recon.length,
        recon_hidden=hidden.accuracy_predictor(recon),
        recon_control_length=recon_control.length,
        recon_control_hidden=hidden.accuracy_predictor(recon_control),
        random_hidden_avg=rand["hidden_avg"],
        random_hidden_best=rand["hidden_best"],
        synth_sec=synth.elapsed_sec,
        program=str(synth.program),
        role_to_obs=world.role_to_obs,
        failures=failures,
    )


def verdict(rows: Sequence[Row]) -> str:
    pccp_constant = len({row.pccp_length for row in rows}) == 1
    recon_grows = rows[-1].recon_length > rows[0].recon_length
    pccp_passes = all(row.pccp_hidden == 1.0 for row in rows)
    proxy_recon_fails = any(row.recon_hidden < 1.0 for row in rows)
    control_passes = all(row.recon_control_hidden == 1.0 for row in rows)
    if not pccp_passes:
        return "KILL_PCCP"
    if pccp_constant and recon_grows and proxy_recon_fails and control_passes:
        return "STRONG_PCCP"
    if pccp_constant and recon_grows:
        return "PCCP_SIGNAL"
    return "VOID"


def print_report(rows: Sequence[Row]) -> None:
    print("PCCP-0 finite witness")
    print("=" * 78)
    print("World: 2 causal bits C0,C1; m nuisance bits N; one spurious bit S.")
    print("Target: parity(C0,C1), with C overrides applied from intervention descriptors.")
    print("Seen split: id, environment_shift_seen, do_c0_seen, do_c1_seen.")
    print("Hidden split: do_n, do_s, counterfactual_hold, composition_shift.")
    print("DSL primitives:")
    for primitive in DSL_PRIMITIVES:
        print(f"  - {primitive}")
    print()
    header = ("m  seen  hidden  PCCP_len  PCCP_H  lookup_len lookup_H  tree_len tree_H  "
              "recon_len recon_H  recon+PCCP_len recon+PCCP_H  random_H_avg")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row.m:>1} {row.seen_cases:>6} {row.hidden_cases:>7} "
              f"{row.pccp_length:>9} {row.pccp_hidden:>7.3f} "
              f"{row.lookup_length:>10} {row.lookup_hidden:>8.3f} "
              f"{row.tree_length:>8} {row.tree_hidden:>6.3f} "
              f"{row.recon_length:>9} {row.recon_hidden:>7.3f} "
              f"{row.recon_control_length:>15} {row.recon_control_hidden:>13.3f} "
              f"{row.random_hidden_avg:>12.3f}")
    print()
    print("Synthesized PCCP programs and role positions:")
    for row in rows:
        print(f"  m={row.m}: len={row.pccp_length}, synth={row.synth_sec:.3f}s, "
              f"role_to_obs={row.role_to_obs}, program={row.program}")
    print()
    print("Hidden-family failures by baseline (only families with accuracy < 1.0):")
    for row in rows:
        print(f"  m={row.m}:")
        if not row.failures:
            print("    none")
        for name, bad in row.failures.items():
            bits = ", ".join(f"{family}:{acc:.3f}" for family, acc in bad.items())
            print(f"    {name}: {bits}")
    print()
    print("Interpretation:")
    print("  PCCP synthesis finds a constant-length executable rule that ignores N and S and uses typed C override fields.")
    print("  The proxy reconstruction baseline pays to reconstruct every observed bit, so its length grows with m, then its shortest observational extractor reads S and fails hidden shifts.")
    print("  The verifier-aware reconstruction control also reconstructs every bit and then bolts on the PCCP rule; it passes hidden, but is longer by the observation-reconstruction cost.")
    print("  Decision-tree behavior is reported rather than forced to fail; if it passes, that is prior-art absorption pressure, not a PCCP victory.")
    print()
    print(f"VERDICT_TOKEN: {verdict(rows)}")
    print("GOSSIP_SUMMARY: The wallpaper memorizer dragged every irrelevant bit along; the tiny causal rule kept only the switch and survived the hidden rewiring.")
    print("SCOPE_LIMITS: This proves only a finite PCCP-A witness for a human-designed world/verifier/DSL; it does not prove open-world verifier discovery, novelty over CEGIS/SyGuS, or a PCCP-H paradigm shift.")


def main() -> None:
    rows = [run_one(m) for m in range(0, 9)]
    print_report(rows)
    fdm_runs = run_fdm0_experiment()
    print_fdm_report(fdm_runs)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# SMUGGLING AUDIT
#
# 1. Does the DSL contain the answer?
#    No target-specific primitive is present. The complete primitive inventory is
#    Var, Const, Not, Bin(AND/OR/XOR/EQ), and If. XOR is a generic Boolean
#    connective, not a named parity/target/causal-selector primitive. The DSL
#    does not expose causal parent, nuisance, spurious, generator seed, hidden
#    family, or target labels as callable primitives.
#
# 2. Does the synthesizer see the hidden verifier?
#    No. run_one constructs separate Verifier objects. synthesize(seen) receives
#    only seen.cases from the seen split. hidden.verify(...) is called only after
#    the Expr is frozen. The hidden split contains do_n_hidden, do_s_hidden,
#    counterfactual_hold_*_hidden, and composition_shift_hidden.
#
# 3. Are the baselines handicapped?
#    The lookup table, decision tree, reconstruction proxy, reconstruction+PCCP
#    control, and random programs receive the same seen cases. The reconstruction
#    proxy intentionally optimizes the proxy habit: exact observation
#    reconstruction plus the shortest observational extractor. To avoid
#    overstating that baseline's failure, the report also includes a
#    verifier-aware reconstruction control; it passes hidden but is longer by the
#    cost of reconstructing C+N+S.
#
# 4. Is the world family too trivial?
#    It is tiny by design: two causal bits, m nuisance bits, one spurious bit,
#    and a parity target. A lookup table cannot solve held-out intervention
#    families, but a strong generic synthesis/CEGIS system should find the same
#    program. Therefore this is a finite witness for the separation, not a
#    moonshot-level benchmark.
#
# 5. Is the reconstruction baseline implemented fairly?
#    Two versions are reported. The proxy version is fair as a reconstruction
#    objective baseline and fails for the theorem-predicted reason: it preserves
#    surface variables and uses the spurious observational extractor. The
#    verifier-aware control is the fairness backstop: reconstruction plus the
#    correct functional decoder passes hidden, but its length grows with m while
#    PCCP length remains constant.
#
# 6. Does FDM-0 see role labels?
#    No during discovery. FDM-0 iterates over observed indices x0..xN and uses a
#    generic binary replacement grammar. role_to_obs is used to construct the
#    deliberately bad shortcut program, the true causal control program, and the
#    post-hoc audit printout. It is not used by fdm0_discover or by the random
#    and exhaustive discovery baselines to select clauses.
#
# 7. Is the perturbation grammar target-specific?
#    It is generic at the field level: replace every observed field by every
#    value in its finite domain. It is still human-supplied and narrow. This is
#    a B1 role-discovery demo, not discovery of the transformation grammar.
#
# 8. Does the discovery baseline get equal information?
#    Yes for the implemented baselines. FDM-0, random clause search, and
#    exhaustive single-field checking receive the same V0 cases, exact target
#    oracle over paired perturbations, binary field-replacement grammar, and
#    paired-query budget. Exhaustive checking gets the same result, which is
#    reported as absorption rather than hidden.
#
# 9. Is the clause grammar too narrow or too wide?
#    It is deliberately narrow: invariant_to(single observed field). That is
#    enough to catch the B22 spurious shortcut but too weak for composite
#    metamorphic relations, covariance, precondition boundaries, or open-world
#    frame formation. The narrowness makes the positive result interpretable and
#    also makes exhaustive single-field checking sufficient here.
#
# NARRATIVE GATE
#
# 1. Earned verdict tokens:
#    The original after-frame witness can still earn STRONG_PCCP for the narrow
#    finite PCCP-A length-gap result. The FDM-0 extension earns
#    DISCOVERY_ABSORBED on the precommitted B22 discovery tokens when exhaustive
#    single-field invariance checking catches P_bad under the same information.
#    If a future run breaks that condition, print_fdm_report reports the changed
#    token directly.
#
# 2. Gossip-magazine summary:
#    The shortcut wore a fake badge, and field-toggling found the badge did not
#    change the job.
#
# 3. What this does NOT prove:
#    It does not prove open-world verifier discovery, transformation-grammar
#    discovery, composite metamorphic relation discovery, novelty over Daikon /
#    metamorphic/spec-mining tools, neural-tool-agent superiority, scaling beyond
#    tiny finite Boolean worlds, or that learned verifiers are automatically
#    aligned. It proves only that a bounded active perturbation routine can add a
#    missing single-field invariance obligation in this toy world.
#
# 4. Is exhaustive single-field check sufficient?
#    Yes for this smallest B22 demo. Exhaustive single-field checking uses the
#    same perturbation oracle and finite field grammar, proposes the same S
#    invariance needed to reject P_bad, and therefore absorbs FDM-0 at this level.
