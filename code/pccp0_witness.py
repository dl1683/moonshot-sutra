#!/usr/bin/env python3
"""PCCP-0 finite witness: finite SCM, DSL, verifier, synthesis, baselines."""
from __future__ import annotations

import math
import random
import time
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


def make_world(m: int) -> World:
    perm = list(range(2 + m + 1))
    random.Random(1729 + m).shuffle(perm)
    return World(m, tuple(perm))


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
# NARRATIVE GATE
#
# 1. Earned verdict token:
#    STRONG_PCCP for the narrow finite PCCP-A length-gap witness if the run shows
#    constant PCCP length, growing reconstruction length, hidden verifier pass
#    for PCCP, proxy reconstruction failure, and verifier-aware reconstruction
#    pass. It is not MOONSHOT_PCCP or a full PCCP-H result.
#
# 2. Gossip-magazine summary:
#    The wallpaper memorizer dragged every irrelevant bit along; the tiny causal
#    rule kept only the switch and survived the hidden rewiring.
#
# 3. What this does NOT prove:
#    It does not prove verifier discovery, open-world frame formation, novelty
#    over CEGIS/SyGuS/ILP/DreamCoder/spec mining, neural-tool-agent superiority,
#    scaling beyond tiny finite Boolean worlds, or that reconstruction always
#    discards causal information.

