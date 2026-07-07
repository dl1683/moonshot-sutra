"""CPU-only weight-transplant gauntlet for Sutra/Eklavya.

Tier 1 is the exact known-gauge linear test: a two-layer linear teacher is
reparameterized by a hidden-basis gauge transform. Raw per-layer SVD should
change under a non-orthogonal gauge; function/chart-aware transplant should not.

Tier 2 and Tier 2.5 are lightweight analytic binding tests. They are not claims
about real Sutra. They exercise Procrustes, Jacobian/slot, raw-SVD, shuffled,
wrong-circuit, frequency-matched, and byte-codec controls on a nonlinear binding
operator so the next real experiment has a sharper target.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Array = np.ndarray

NAMES = ["alic", "bobx", "carl", "dana"]
COLORS = ["redx", "blue", "gren", "gold"]
ROOMS = ["rm_1", "rm_2", "rm_3", "rm_4"]
ACTIONS = ["pick", "grab", "take", "hold"]
ATTR_TO_VALUES = {"colr": COLORS, "room": ROOMS, "actn": ACTIONS}
ALL_VALUES = COLORS + ROOMS + ACTIONS
PATCH_SIZE = 4


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def rel_fro(a: Array, b: Array) -> float:
    return float(np.linalg.norm(a - b, ord="fro") / (np.linalg.norm(b, ord="fro") + 1e-12))


def cosine_flat(a: Array, b: Array) -> float:
    av = a.reshape(-1)
    bv = b.reshape(-1)
    return float(np.dot(av, bv) / ((np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-12))


def mse_on_inputs(f_hat: Array, f_true: Array, x: Array) -> float:
    diff = (f_hat - f_true) @ x
    return float(np.mean(diff * diff))


def orthonormal_rows(rng: np.random.Generator, rows: int, cols: int) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(cols, rows)))
    return q.T


def matrix_sqrt_factor(f: Array, rank: int) -> tuple[Array, Array, Array]:
    u, s, vt = np.linalg.svd(f, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vt_r = vt[:rank, :]
    sqrt_s = np.sqrt(np.maximum(s_r, 0.0))
    return u_r * sqrt_s[None, :], sqrt_s[:, None] * vt_r, s_r


@dataclass
class Tier1Config:
    seed: int = 11
    input_dim: int = 24
    hidden_dim: int = 48
    output_dim: int = 16
    transplant_rank: int = 8
    decoy_rank: int = 16
    n_calibration: int = 256
    n_test: int = 1024
    nonorth_active_scale: float = 0.18
    nonorth_decoy_scale: float = 7.0
    raw_drift_gate: float = 0.50
    exact_mse_gate: float = 1e-20
    exact_drift_gate: float = 1e-10


@dataclass
class LinearTeacher:
    w1: Array
    w2: Array

    @property
    def function(self) -> Array:
        return self.w2 @ self.w1


def make_linear_teacher(cfg: Tier1Config, rng: np.random.Generator) -> LinearTeacher:
    r = cfg.transplant_rank
    w1 = np.zeros((cfg.hidden_dim, cfg.input_dim), dtype=np.float64)
    w2 = np.zeros((cfg.output_dim, cfg.hidden_dim), dtype=np.float64)
    active_in = orthonormal_rows(rng, r, cfg.input_dim)
    active_out = rng.normal(size=(cfg.output_dim, r))
    w1[:r, :] = np.linspace(1.0, 0.65, r)[:, None] * active_in
    w2[:, :r] = active_out
    decoy_end = min(cfg.hidden_dim, r + cfg.decoy_rank)
    n_decoy = decoy_end - r
    if n_decoy > 0:
        decoy_in = orthonormal_rows(rng, n_decoy, cfg.input_dim)
        w1[r:decoy_end, :] = np.linspace(0.44, 0.28, n_decoy)[:, None] * decoy_in
    return LinearTeacher(w1=w1, w2=w2)


def gauge_teacher(teacher: LinearTeacher, a: Array) -> LinearTeacher:
    return LinearTeacher(w1=a @ teacher.w1, w2=teacher.w2 @ np.linalg.inv(a))


def make_orthogonal_gauge(cfg: Tier1Config, rng: np.random.Generator) -> Array:
    q, _ = np.linalg.qr(rng.normal(size=(cfg.hidden_dim, cfg.hidden_dim)))
    return q


def make_nonorthogonal_gauge(cfg: Tier1Config, rng: np.random.Generator) -> Array:
    r = cfg.transplant_rank
    d = cfg.hidden_dim
    scales = np.ones(d)
    scales[:r] = cfg.nonorth_active_scale
    scales[r:] = cfg.nonorth_decoy_scale
    shear = np.eye(d) + np.triu(rng.normal(scale=0.015, size=(d, d)), k=1)
    return shear @ np.diag(scales)


def raw_w1_svd_transplant(teacher: LinearTeacher, rank: int) -> Array:
    u, s, vt = np.linalg.svd(teacher.w1, full_matrices=False)
    u_r = u[:, :rank]
    return (teacher.w2 @ u_r) @ (s[:rank, None] * vt[:rank, :])


def exact_function_transplant(teacher: LinearTeacher, rank: int) -> Array:
    w2_s, w1_s, _ = matrix_sqrt_factor(teacher.function, rank)
    return w2_s @ w1_s


def chart_procrustes_transplant(teacher: LinearTeacher, reference: LinearTeacher, x_cal: Array, rank: int) -> Array:
    h_src = teacher.w1 @ x_cal
    h_ref = reference.w1 @ x_cal
    y = teacher.w2 @ h_src
    chart = h_ref @ np.linalg.pinv(h_src, rcond=1e-12)
    w1_ref = chart @ teacher.w1
    w2_ref = y @ np.linalg.pinv(h_ref, rcond=1e-12)
    w2_s, w1_s, _ = matrix_sqrt_factor(w2_ref @ w1_ref, rank)
    return w2_s @ w1_s


def random_spectrum_control(f_true: Array, rng: np.random.Generator) -> Array:
    _, s, _ = np.linalg.svd(f_true, full_matrices=False)
    q_u, _ = np.linalg.qr(rng.normal(size=(f_true.shape[0], f_true.shape[0])))
    q_v, _ = np.linalg.qr(rng.normal(size=(f_true.shape[1], f_true.shape[1])))
    diag = np.zeros_like(f_true)
    np.fill_diagonal(diag, s[: min(f_true.shape)])
    return q_u @ diag @ q_v.T


def run_tier1(cfg: Tier1Config) -> dict:
    rng = set_seed(cfg.seed)
    base = make_linear_teacher(cfg, rng)
    orth = gauge_teacher(base, make_orthogonal_gauge(cfg, rng))
    nonorth = gauge_teacher(base, make_nonorthogonal_gauge(cfg, rng))
    x_cal = rng.normal(size=(cfg.input_dim, cfg.n_calibration))
    x_test = rng.normal(size=(cfg.input_dim, cfg.n_test))
    f_true = base.function
    raw_base = raw_w1_svd_transplant(base, cfg.transplant_rank)
    raw_orth = raw_w1_svd_transplant(orth, cfg.transplant_rank)
    raw_nonorth = raw_w1_svd_transplant(nonorth, cfg.transplant_rank)
    exact_base = exact_function_transplant(base, cfg.transplant_rank)
    exact_nonorth = exact_function_transplant(nonorth, cfg.transplant_rank)
    chart_nonorth = chart_procrustes_transplant(nonorth, base, x_cal, cfg.transplant_rank)
    spectrum = random_spectrum_control(f_true, rng)
    metrics = {
        "true_rank": int(np.linalg.matrix_rank(f_true, tol=1e-10)),
        "raw_base_mse": mse_on_inputs(raw_base, f_true, x_test),
        "raw_orthogonal_mse": mse_on_inputs(raw_orth, f_true, x_test),
        "raw_nonorthogonal_mse": mse_on_inputs(raw_nonorth, f_true, x_test),
        "raw_orthogonal_drift_rel": rel_fro(raw_orth, raw_base),
        "raw_nonorthogonal_drift_rel": rel_fro(raw_nonorth, raw_base),
        "raw_nonorthogonal_cos_to_true": cosine_flat(raw_nonorth, f_true),
        "exact_base_mse": mse_on_inputs(exact_base, f_true, x_test),
        "exact_nonorthogonal_mse": mse_on_inputs(exact_nonorth, f_true, x_test),
        "exact_nonorthogonal_drift_rel": rel_fro(exact_nonorth, exact_base),
        "chart_nonorthogonal_mse": mse_on_inputs(chart_nonorth, f_true, x_test),
        "chart_nonorthogonal_drift_rel": rel_fro(chart_nonorth, exact_base),
        "random_spectrum_mse": mse_on_inputs(spectrum, f_true, x_test),
        "random_spectrum_cos_to_true": cosine_flat(spectrum, f_true),
    }
    gates = {
        "raw_svd_changes_under_nonorthogonal_gauge": metrics["raw_nonorthogonal_drift_rel"] >= cfg.raw_drift_gate,
        "raw_svd_stable_under_orthogonal_control": metrics["raw_orthogonal_drift_rel"] < 1e-10,
        "exact_transplant_works_base": metrics["exact_base_mse"] <= cfg.exact_mse_gate,
        "exact_transplant_works_nonorthogonal": metrics["exact_nonorthogonal_mse"] <= cfg.exact_mse_gate,
        "exact_transplant_invariant": metrics["exact_nonorthogonal_drift_rel"] <= cfg.exact_drift_gate,
        "chart_transplant_works_nonorthogonal": metrics["chart_nonorthogonal_mse"] <= 1e-20,
        "matched_spectrum_control_fails": metrics["random_spectrum_cos_to_true"] < 0.50,
    }
    return {"config": asdict(cfg), "metrics": metrics, "gates": gates, "pass": all(gates.values())}


@dataclass(frozen=True)
class BindingExample:
    facts: tuple[tuple[str, str, str], ...]
    query_entity: str
    query_attr: str
    correct: str
    distractors: tuple[str, ...]


@dataclass
class Tier2Config:
    seed: int = 12
    teacher_width: int = 128
    student_width: int = 64
    teacher_layers: int = 4
    student_layers: int = 4
    teacher_dim: int = 64
    student_dim: int = 32
    n_eval: int = 800
    noise: float = 0.015
    pass_gate: float = 0.95
    control_gate: float = 0.45


class BindingGeometry:
    def __init__(self, cfg: Tier2Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.keys = [(e, a) for e in NAMES for a in ATTR_TO_VALUES]
        self.key_index = {k: i for i, k in enumerate(self.keys)}
        self.value_index = {v: i for i, v in enumerate(ALL_VALUES)}
        self.k_teacher = self._norm(rng.normal(size=(len(self.keys), cfg.teacher_dim)))
        self.v_teacher = self._norm(rng.normal(size=(len(ALL_VALUES), cfg.teacher_dim)))
        self.p_key_true = self._proj(cfg.student_dim, cfg.teacher_dim, rng)
        self.p_val_true = self._proj(cfg.student_dim, cfg.teacher_dim, rng)
        self.k_student_chart = self._norm(self.k_teacher @ self.p_key_true.T + cfg.noise * rng.normal(size=(len(self.keys), cfg.student_dim)))
        self.v_student_chart = self._norm(self.v_teacher @ self.p_val_true.T + cfg.noise * rng.normal(size=(len(ALL_VALUES), cfg.student_dim)))
        self.p_key_hat = self._least_squares_map(self.k_teacher, self.k_student_chart)
        self.p_val_hat = self._least_squares_map(self.v_teacher, self.v_student_chart)
        self.random_p_key = self._proj(cfg.student_dim, cfg.teacher_dim, rng)
        self.random_p_val = self._proj(cfg.student_dim, cfg.teacher_dim, rng)

    @staticmethod
    def _norm(x: Array) -> Array:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    @staticmethod
    def _proj(out_dim: int, in_dim: int, rng: np.random.Generator) -> Array:
        q, _ = np.linalg.qr(rng.normal(size=(in_dim, out_dim)))
        return q.T

    @staticmethod
    def _least_squares_map(src: Array, dst: Array) -> Array:
        return dst.T @ np.linalg.pinv(src.T, rcond=1e-10)

    def key_t(self, entity: str, attr: str) -> Array:
        return self.k_teacher[self.key_index[(entity, attr)]]

    def val_t(self, value: str) -> Array:
        return self.v_teacher[self.value_index[value]]

    def key_s(self, entity: str, attr: str) -> Array:
        return self.k_student_chart[self.key_index[(entity, attr)]]

    def val_s(self, value: str) -> Array:
        return self.v_student_chart[self.value_index[value]]

    def map_key(self, vec: Array, random_chart: bool = False) -> Array:
        out = (self.random_p_key if random_chart else self.p_key_hat) @ vec
        return out / (np.linalg.norm(out) + 1e-12)

    def map_val(self, vec: Array, random_chart: bool = False) -> Array:
        out = (self.random_p_val if random_chart else self.p_val_hat) @ vec
        return out / (np.linalg.norm(out) + 1e-12)

    def teacher_memory(self, ex: BindingExample) -> Array:
        mem = np.zeros((self.cfg.teacher_dim, self.cfg.teacher_dim), dtype=np.float64)
        for entity, attr, value in ex.facts:
            mem += np.outer(self.key_t(entity, attr), self.val_t(value))
        return mem

    def student_slot_memory(self, ex: BindingExample, shuffled_values: bool = False, frequency_matched: bool = False, random_chart: bool = False) -> Array:
        mem = np.zeros((self.cfg.student_dim, self.cfg.student_dim), dtype=np.float64)
        facts = list(ex.facts)
        if shuffled_values:
            values = [f[2] for f in facts]
            self.rng.shuffle(values)
            facts = [(e, a, values[i]) for i, (e, a, _) in enumerate(facts)]
        if frequency_matched:
            facts = [(e, a, str(self.rng.choice(ATTR_TO_VALUES[a]))) for e, a, _ in facts]
        for entity, attr, value in facts:
            mem += np.outer(self.map_key(self.key_t(entity, attr), random_chart), self.map_val(self.val_t(value), random_chart))
        return mem

    def procrustes_operator_memory(self, ex: BindingExample, random_chart: bool = False) -> Array:
        p_k = self.random_p_key if random_chart else self.p_key_hat
        p_v = self.random_p_val if random_chart else self.p_val_hat
        return p_k @ self.teacher_memory(ex) @ p_v.T

    def jacobian_slot_memory(self, ex: BindingExample) -> Array:
        mem = np.zeros((self.cfg.student_dim, self.cfg.student_dim), dtype=np.float64)
        for entity, attr, value in ex.facts:
            slot = np.outer(self.key_t(entity, attr), self.val_t(value))
            mem += self.p_key_hat @ slot @ self.p_val_hat.T
        return mem

    def raw_svd_memory(self, ex: BindingExample, rank: int = 4) -> Array:
        mem_t = self.teacher_memory(ex)
        u, s, vt = np.linalg.svd(mem_t, full_matrices=False)
        approx = (u[:, :rank] * s[:rank][None, :]) @ vt[:rank, :]
        return approx[: self.cfg.student_dim, : self.cfg.student_dim]

    def score_teacher(self, ex: BindingExample, candidate: str) -> float:
        return float(self.key_t(ex.query_entity, ex.query_attr) @ self.teacher_memory(ex) @ self.val_t(candidate))

    def score_with_memory(self, ex: BindingExample, candidate: str, mem: Array, wrong_circuit: bool = False, byte_codec: Callable[[str, str], Array] | None = None) -> float:
        attr = ex.query_attr
        if wrong_circuit:
            attr = {"colr": "room", "room": "actn", "actn": "colr"}[attr]
        if byte_codec is None:
            q = self.key_s(ex.query_entity, attr)
            v = self.val_s(candidate)
        else:
            q = byte_codec(f"{ex.query_entity}:{attr}", "key")
            v = byte_codec(candidate, "value")
        return float(q @ mem @ v)


def generate_binding_example(rng: np.random.Generator) -> BindingExample:
    entities = list(rng.choice(NAMES, size=2, replace=False))
    facts: list[tuple[str, str, str]] = []
    bindings: dict[tuple[str, str], str] = {}
    for entity in entities:
        for attr, values in ATTR_TO_VALUES.items():
            value = str(rng.choice(values))
            facts.append((entity, attr, value))
            bindings[(entity, attr)] = value
    query_entity = str(rng.choice(entities))
    query_attr = str(rng.choice(list(ATTR_TO_VALUES)))
    correct = bindings[(query_entity, query_attr)]
    distractors = [v for v in ATTR_TO_VALUES[query_attr] if v != correct]
    rng.shuffle(distractors)
    return BindingExample(tuple(facts), query_entity, query_attr, correct, tuple(distractors[:3]))


def mcq_accuracy(scorer: Callable[[BindingExample, str], float], n_eval: int, rng: np.random.Generator) -> float:
    correct = 0
    for _ in range(n_eval):
        ex = generate_binding_example(rng)
        choices = [ex.correct] + list(ex.distractors)
        rng.shuffle(choices)
        scores = [scorer(ex, c) for c in choices]
        correct += int(choices[int(np.argmax(scores))] == ex.correct)
    return correct / n_eval


def run_tier2(cfg: Tier2Config) -> dict:
    rng = set_seed(cfg.seed)
    geom = BindingGeometry(cfg, rng)
    eval_rng = np.random.default_rng(cfg.seed + 1000)
    scorers = {
        "teacher": lambda ex, c: geom.score_teacher(ex, c),
        "procrustes_operator": lambda ex, c: geom.score_with_memory(ex, c, geom.procrustes_operator_memory(ex)),
        "jacobian_sketch": lambda ex, c: geom.score_with_memory(ex, c, geom.jacobian_slot_memory(ex)),
        "mlp_slots": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex)),
        "raw_svd_no_chart": lambda ex, c: geom.score_with_memory(ex, c, geom.raw_svd_memory(ex)),
        "shuffled_pairs_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex, shuffled_values=True)),
        "wrong_circuit_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex), wrong_circuit=True),
        "frequency_matched_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex, frequency_matched=True)),
        "random_chart_control": lambda ex, c: geom.score_with_memory(ex, c, geom.procrustes_operator_memory(ex, random_chart=True)),
    }
    metrics = {name: mcq_accuracy(fn, cfg.n_eval, np.random.default_rng(eval_rng.integers(0, 2**31 - 1))) for name, fn in scorers.items()}
    gates = {
        "teacher_solves_task": metrics["teacher"] >= 0.99,
        "procrustes_passes": metrics["procrustes_operator"] >= cfg.pass_gate,
        "jacobian_passes": metrics["jacobian_sketch"] >= cfg.pass_gate,
        "mlp_slots_passes": metrics["mlp_slots"] >= cfg.pass_gate,
        "raw_svd_fails": metrics["raw_svd_no_chart"] <= cfg.control_gate,
        "shuffled_pairs_fail": metrics["shuffled_pairs_control"] <= cfg.control_gate,
        "wrong_circuit_fails": metrics["wrong_circuit_control"] <= cfg.control_gate,
        "frequency_matched_fails": metrics["frequency_matched_control"] <= cfg.control_gate,
        "random_chart_fails": metrics["random_chart_control"] <= cfg.control_gate,
    }
    return {"config": asdict(cfg), "metrics": metrics, "gates": gates, "pass": all(gates.values())}


@dataclass
class Tier25Config:
    seed: int = 13
    n_eval: int = 800
    codec_noise: float = 0.02
    chart_noise: float = 0.0
    pass_gate: float = 0.90
    control_gate: float = 0.45


def word_to_bytes(word: str) -> tuple[int, ...]:
    bs = list(word.encode("ascii"))[:PATCH_SIZE]
    while len(bs) < PATCH_SIZE:
        bs.append(0)
    return tuple(bs)


def key_to_bytes(word: str) -> tuple[int, ...]:
    return tuple(word.encode("ascii"))


class BytePatchCodec:
    def __init__(self, geom: BindingGeometry, rng: np.random.Generator, noise: float, mode: str, chart_noise: float = 0.0):
        self.key_lookup: dict[tuple[int, ...], Array] = {}
        self.value_lookup: dict[tuple[int, ...], Array] = {}
        self.rng = rng
        self.mode = mode
        self.noise = noise
        self.chart_noise = chart_noise
        for entity in NAMES:
            for attr in ATTR_TO_VALUES:
                word = f"{entity}:{attr}"
                self.key_lookup[key_to_bytes(word)] = self._corrupt(geom.key_s(entity, attr))
        for value in ALL_VALUES:
            self.value_lookup[word_to_bytes(value)] = self._corrupt(geom.val_s(value))

    def _corrupt(self, vec: Array) -> Array:
        if self.mode == "real":
            out = vec + self.noise * self.rng.normal(size=vec.shape)
        elif self.mode == "random":
            out = self.rng.normal(size=vec.shape)
        elif self.mode == "shuffled":
            out = vec.copy()
        else:
            raise ValueError(self.mode)
        return out / (np.linalg.norm(out) + 1e-12)

    def finalize_shuffle(self) -> None:
        if self.mode != "shuffled":
            return
        key_vals = list(self.key_lookup.values())
        val_vals = list(self.value_lookup.values())
        self.rng.shuffle(key_vals)
        self.rng.shuffle(val_vals)
        for k, v in zip(list(self.key_lookup), key_vals):
            self.key_lookup[k] = v
        for k, v in zip(list(self.value_lookup), val_vals):
            self.value_lookup[k] = v

    def _maybe_corrupt_lookup(self, lookup: dict[tuple[int, ...], Array], key: tuple[int, ...]) -> Array:
        if self.mode != "real" or self.chart_noise <= 0.0:
            return lookup[key]
        if self.rng.random() >= self.chart_noise:
            return lookup[key]
        wrong_keys = [k for k in lookup if k != key]
        wrong_key = wrong_keys[int(self.rng.integers(0, len(wrong_keys)))]
        return lookup[wrong_key]

    def __call__(self, word: str, kind: str) -> Array:
        if kind == "key":
            return self._maybe_corrupt_lookup(self.key_lookup, key_to_bytes(word))
        if kind == "value":
            return self._maybe_corrupt_lookup(self.value_lookup, word_to_bytes(word))
        raise ValueError(kind)


def run_tier25(cfg: Tier25Config) -> dict:
    rng = set_seed(cfg.seed)
    geom = BindingGeometry(Tier2Config(seed=cfg.seed, n_eval=cfg.n_eval, noise=cfg.codec_noise), rng)
    real_codec = BytePatchCodec(geom, rng, cfg.codec_noise, "real", chart_noise=cfg.chart_noise)
    random_codec = BytePatchCodec(geom, rng, cfg.codec_noise, "random")
    shuffled_codec = BytePatchCodec(geom, rng, cfg.codec_noise, "shuffled")
    shuffled_codec.finalize_shuffle()
    scorers = {
        "byte_codec_chart": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex), byte_codec=real_codec),
        "random_byte_codec_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex), byte_codec=random_codec),
        "shuffled_byte_codec_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex), byte_codec=shuffled_codec),
        "wrong_circuit_with_codec_control": lambda ex, c: geom.score_with_memory(ex, c, geom.student_slot_memory(ex), wrong_circuit=True, byte_codec=real_codec),
    }
    eval_rng = np.random.default_rng(cfg.seed + 2500)
    metrics = {name: mcq_accuracy(fn, cfg.n_eval, np.random.default_rng(eval_rng.integers(0, 2**31 - 1))) for name, fn in scorers.items()}
    gates = {
        "byte_codec_chart_passes": metrics["byte_codec_chart"] >= cfg.pass_gate,
        "random_byte_codec_fails": metrics["random_byte_codec_control"] <= cfg.control_gate,
        "shuffled_byte_codec_fails": metrics["shuffled_byte_codec_control"] <= cfg.control_gate,
        "wrong_circuit_fails": metrics["wrong_circuit_with_codec_control"] <= cfg.control_gate,
    }
    return {"config": asdict(cfg), "metrics": metrics, "gates": gates, "pass": all(gates.values())}


def chart_top1_accuracy(codec: BytePatchCodec, trials: int, rng: np.random.Generator) -> dict:
    """Measure how often codec lookups retrieve their own chart entry."""
    key_keys = list(codec.key_lookup)
    value_keys = list(codec.value_lookup)
    key_matrix = np.stack([codec.key_lookup[k] for k in key_keys])
    value_matrix = np.stack([codec.value_lookup[k] for k in value_keys])
    key_correct = 0
    value_correct = 0
    for _ in range(trials):
        key_idx = int(rng.integers(0, len(key_keys)))
        key_vec = codec._maybe_corrupt_lookup(codec.key_lookup, key_keys[key_idx])
        key_pred = int(np.argmax(key_matrix @ key_vec))
        key_correct += int(key_pred == key_idx)

        value_idx = int(rng.integers(0, len(value_keys)))
        value_vec = codec._maybe_corrupt_lookup(codec.value_lookup, value_keys[value_idx])
        value_pred = int(np.argmax(value_matrix @ value_vec))
        value_correct += int(value_pred == value_idx)
    key_top1 = key_correct / max(1, trials)
    value_top1 = value_correct / max(1, trials)
    return {
        "key_top1": key_top1,
        "value_top1": value_top1,
        "combined_top1": 0.5 * (key_top1 + value_top1),
        "trials_per_kind": trials,
    }


def calibrate_chart_noise(
    target_top1: float,
    base_cfg: Tier25Config,
    probe_trials: int,
    seed: int,
) -> dict:
    """Find a chart_noise setting whose measured top-1 is closest to target."""
    best: dict | None = None
    for noise in np.linspace(0.0, 1.0, 101):
        rng = np.random.default_rng(seed + int(round(noise * 1000)))
        geom = BindingGeometry(Tier2Config(seed=base_cfg.seed, n_eval=base_cfg.n_eval, noise=base_cfg.codec_noise), rng)
        codec = BytePatchCodec(geom, rng, base_cfg.codec_noise, "real", chart_noise=float(noise))
        measured = chart_top1_accuracy(codec, probe_trials, np.random.default_rng(seed + 100000 + int(round(noise * 1000))))
        err = abs(measured["combined_top1"] - target_top1)
        candidate = {
            "target_chart_top1": target_top1,
            "chart_noise": float(noise),
            "measured_chart": measured,
            "abs_error": float(err),
        }
        if best is None or err < best["abs_error"]:
            best = candidate
    assert best is not None
    return best


def run_tier25_degradation_curve(
    cfg: Tier25Config,
    targets: list[float],
    probe_trials: int,
) -> dict:
    points = []
    for i, target in enumerate(targets):
        calibration = calibrate_chart_noise(target, cfg, probe_trials, cfg.seed + 7000 + i * 100)
        run_cfg = Tier25Config(
            seed=cfg.seed + i * 17,
            n_eval=cfg.n_eval,
            codec_noise=cfg.codec_noise,
            chart_noise=calibration["chart_noise"],
            pass_gate=cfg.pass_gate,
            control_gate=cfg.control_gate,
        )
        tier25 = run_tier25(run_cfg)
        points.append({
            **calibration,
            "transplant_accuracy": tier25["metrics"]["byte_codec_chart"],
            "control_accuracies": {
                "random_byte_codec_control": tier25["metrics"]["random_byte_codec_control"],
                "shuffled_byte_codec_control": tier25["metrics"]["shuffled_byte_codec_control"],
                "wrong_circuit_with_codec_control": tier25["metrics"]["wrong_circuit_with_codec_control"],
            },
            "tier25_pass": tier25["pass"],
        })

    p25 = min(points, key=lambda p: abs(p["measured_chart"]["combined_top1"] - 0.25))
    p50 = min(points, key=lambda p: abs(p["measured_chart"]["combined_top1"] - 0.50))
    sorted_points = sorted(points, key=lambda p: p["measured_chart"]["combined_top1"])
    viable = [p for p in sorted_points if p["transplant_accuracy"] >= 0.80]
    cliff_top1 = viable[0]["measured_chart"]["combined_top1"] if viable else None
    if p25["transplant_accuracy"] > 0.80:
        verdict = "NO_CLIFF_PHASE_1_5_VIABLE_EVEN_AT_25_CHART_TOP1"
    elif p50["transplant_accuracy"] < 0.50:
        verdict = "CLIFF_PHASE_1_5_NEEDS_ABOVE_50_CHART_TOP1"
    else:
        verdict = "SMOOTH_OR_MODERATE_DEGRADATION_PHASE_1_5_TARGET_IS_EMPIRICAL"

    return {
        "config": asdict(cfg),
        "targets": targets,
        "probe_trials": probe_trials,
        "points": points,
        "cliff_chart_top1_for_80_transplant": cliff_top1,
        "precommitted_checks": {
            "transplant_gt_80_at_chart_25": p25["transplant_accuracy"] > 0.80,
            "transplant_lt_50_at_chart_50": p50["transplant_accuracy"] < 0.50,
            "point_near_25": p25,
            "point_near_50": p50,
        },
        "verdict": verdict,
    }


def write_degradation_svg(curve: dict, path: str) -> None:
    points = sorted(curve["points"], key=lambda p: p["measured_chart"]["combined_top1"])
    width, height = 720, 460
    left, right, top, bottom = 70, 25, 35, 65
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(point: dict) -> tuple[float, float]:
        x_val = point["measured_chart"]["combined_top1"]
        y_val = point["transplant_accuracy"]
        x = left + x_val * plot_w
        y = top + (1.0 - y_val) * plot_h
        return x, y

    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(p) for p in points))
    circles = []
    labels = []
    for p in points:
        x, y = xy(p)
        chart = p["measured_chart"]["combined_top1"]
        acc = p["transplant_accuracy"]
        circles.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb" />')
        labels.append(
            f'<text x="{x + 7:.1f}" y="{y - 7:.1f}" font-size="11" fill="#111827">'
            f'{chart:.2f},{acc:.2f}</text>'
        )

    grid = []
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_w
        y = top + (1.0 - tick) * plot_h
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#e5e7eb" />')
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb" />')
        grid.append(f'<text x="{x - 10:.1f}" y="{height - bottom + 22}" font-size="12" fill="#374151">{tick:.2f}</text>')
        grid.append(f'<text x="{left - 48}" y="{y + 4:.1f}" font-size="12" fill="#374151">{tick:.2f}</text>')

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">
  <rect width=\"100%\" height=\"100%\" fill=\"white\" />
  <text x=\"{left}\" y=\"24\" font-size=\"18\" font-family=\"Arial\" fill=\"#111827\">Tier 2.5 Chart Degradation Curve</text>
  {''.join(grid)}
  <line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{height-bottom}\" stroke=\"#111827\" />
  <line x1=\"{left}\" y1=\"{height-bottom}\" x2=\"{width-right}\" y2=\"{height-bottom}\" stroke=\"#111827\" />
  <polyline points=\"{poly}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\" />
  {''.join(circles)}
  {''.join(labels)}
  <text x=\"{left + plot_w/2 - 75:.1f}\" y=\"{height - 18}\" font-size=\"14\" font-family=\"Arial\" fill=\"#111827\">Measured chart top-1</text>
  <text transform=\"translate(18 {top + plot_h/2 + 80:.1f}) rotate(-90)\" font-size=\"14\" font-family=\"Arial\" fill=\"#111827\">Transplant MCQ accuracy</text>
  <text x=\"{left}\" y=\"{height - 38}\" font-size=\"12\" font-family=\"Arial\" fill=\"#374151\">Verdict: {curve['verdict']}</text>
</svg>
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(svg, encoding="utf-8")


def run_all(args: argparse.Namespace) -> dict:
    result: dict[str, dict] = {}
    if args.tier in ("tier1", "all"):
        result["tier1"] = run_tier1(Tier1Config(seed=args.seed))
    if args.tier in ("tier2", "all"):
        result["tier2"] = run_tier2(Tier2Config(seed=args.seed + 1))
    if args.tier in ("tier25", "all"):
        result["tier25"] = run_tier25(Tier25Config(seed=args.seed + 2, n_eval=args.n_eval, codec_noise=args.codec_noise, chart_noise=args.chart_noise))
    return result


def compact_print(results: dict) -> None:
    for tier, payload in results.items():
        print(f"\n[{tier}] pass={payload['pass']}")
        print("metrics:")
        for key, value in payload["metrics"].items():
            print(f"  {key}: {value:.12g}" if isinstance(value, float) else f"  {key}: {value}")
        print("gates:")
        for key, value in payload["gates"].items():
            print(f"  {key}: {'PASS' if value else 'FAIL'}")


def compact_print_degradation_curve(curve: dict) -> None:
    print(f"\n[tier25_degradation_curve] verdict={curve['verdict']}")
    print("target chart_noise measured_chart transplant random shuffled wrong_circuit")
    for point in curve["points"]:
        controls = point["control_accuracies"]
        print(
            f"{point['target_chart_top1']:.2f} "
            f"{point['chart_noise']:.2f} "
            f"{point['measured_chart']['combined_top1']:.4f} "
            f"{point['transplant_accuracy']:.4f} "
            f"{controls['random_byte_codec_control']:.4f} "
            f"{controls['shuffled_byte_codec_control']:.4f} "
            f"{controls['wrong_circuit_with_codec_control']:.4f}"
        )
    cliff = curve["cliff_chart_top1_for_80_transplant"]
    print(f"80% transplant cliff chart_top1: {cliff if cliff is not None else 'not reached'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["tier1", "tier2", "tier25", "all"], default="all")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--n-eval", type=int, default=800)
    parser.add_argument("--codec-noise", type=float, default=0.02)
    parser.add_argument("--chart-noise", type=float, default=0.0)
    parser.add_argument("--degradation-curve", action="store_true")
    parser.add_argument("--curve-targets", type=float, nargs="+", default=[0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.25])
    parser.add_argument("--curve-probe-trials", type=int, default=2000)
    parser.add_argument("--curve-output-dir", default="C:/sutra_fast/codec_phase1.5")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    args = parser.parse_args()
    if args.degradation_curve:
        cfg = Tier25Config(seed=args.seed + 2, n_eval=args.n_eval, codec_noise=args.codec_noise)
        curve = run_tier25_degradation_curve(cfg, args.curve_targets, args.curve_probe_trials)
        if args.curve_output_dir:
            out_dir = Path(args.curve_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "toy_degradation_curve.json").write_text(json.dumps(curve, indent=2, sort_keys=True), encoding="utf-8")
            write_degradation_svg(curve, str(out_dir / "toy_degradation_curve.svg"))
        print(json.dumps(curve, indent=2, sort_keys=True) if args.json else "", end="") if args.json else compact_print_degradation_curve(curve)
        return
    results = run_all(args)
    print(json.dumps(results, indent=2, sort_keys=True) if args.json else "", end="") if args.json else compact_print(results)


if __name__ == "__main__":
    main()


