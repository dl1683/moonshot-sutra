"""E3 Functional Teacher Tomography toy experiment.

CPU-only one-shot kill test for the E3 redirect. The controlled world is a
binary candidate-ranking task with hidden counterfactual structure. Teachers are
not label oracles: one is surface-biased, and two are complementary sensors for
latent ranking bits. E3 must use source-specific teacher measurements to compile
lesson packets and train a tiny teacher-free student.

Batch 44 makes the toy hostile by admitting equal-geometry absorbers. B13 gets
the exact hidden constructor, B15 gets nuisance geometry without teacher
identity, and B10+ gets the transformation generator without teacher signals.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


PRECOMMIT = {
    "direction": "E3 Functional Teacher Tomography toy v1 hostile absorbers",
    "claim": (
        "source-specific teacher measurements can be inverted into compact "
        "counterfactual ranking lessons that transfer to a hidden transform "
        "better than raw teacher outputs and ordinary absorbers; this hostile "
        "variant separately tests whether exact tools or supplied geometry "
        "absorb the signal"
    ),
    "signal_token": "E3_TOY_HOSTILE_SIGNAL_SURVIVES_SUPPLIED_GEOMETRY",
    "kill_tokens": [
        "E3_TOY_ABSORBED_BY_EXACT_DOMAIN_TOOL",
        "E3_TOY_ABSORBED_BY_NUISANCE_ORACLE",
        "E3_TOY_ABSORBED_BY_ENHANCED_AUGMENTATION",
        "E3_TOY_ABSORBED_BY_CE_ONLY",
        "E3_TOY_ABSORBED_BY_SINGLE_TEACHER",
        "E3_TOY_ABSORBED_BY_TEACHER_AVERAGE_OR_WEIGHTED_VOTE",
        "E3_TOY_ABSORBED_BY_ACTIVE_LEARNING",
        "E3_TOY_ABSORBED_BY_AUGMENTATION",
        "E3_TOY_SHUFFLED_SENSORS_MATCH_REAL",
        "E3_TOY_NEGATIVE",
    ],
    "void_tokens": [
        "E3_TOY_VOID_NONFINITE",
        "E3_TOY_VOID_EMPTY_SPLIT",
        "E3_TOY_VOID_SUPPLIED_GEOMETRY_OR_LEAKAGE",
    ],
    "continuation_gate": {
        "beat_best_ordinary_by_pp": 5.0,
        "beat_active_by_pp": 3.0,
        "beat_best_single_by_pp": 3.0,
        "beat_shuffled_by_pp": 6.0,
        "beat_enhanced_augmentation_by_pp": 5.0,
        "beat_nuisance_oracle_by_pp": 5.0,
    },
    "absorber_tests": {
        "B13_exact_domain_tool": {
            "confirm_token": "B44_B13_CONFIRM_EXACT_TOOL_GRANTED_AND_SCORED",
            "kill_token": "B44_B13_KILL_E3_IF_EXACT_TOOL_HIDDEN_ACC_GE_E3",
            "void_token": "B44_B13_VOID_IF_TOOL_USES_TEACHER_SIGNALS",
        },
        "B15_nuisance_oracle": {
            "confirm_token": "B44_B15_CONFIRM_TEACHER_IDENTITY_RESIDUAL_IF_E3_BEATS_NUISANCE_ORACLE_BY_5PP",
            "kill_token": "B44_B15_KILL_TEACHER_IDENTITY_IF_NUISANCE_ORACLE_WITHIN_5PP_OR_BEATS_E3",
            "void_token": "B44_B15_VOID_IF_ORACLE_USES_TEACHER_ROLE_MAP_OR_HIDDEN_SET_LOOKUP",
        },
        "B10_plus_enhanced_augmentation": {
            "confirm_token": "B44_B10P_CONFIRM_TEACHER_SIGNAL_RESIDUAL_IF_E3_BEATS_TRANSFORM_AUG_BY_5PP",
            "kill_token": "B44_B10P_KILL_E3_IF_TRANSFORM_AUGMENTATION_WITHOUT_TEACHERS_WITHIN_5PP_OR_BEATS_E3",
            "void_token": "B44_B10P_VOID_IF_AUGMENTATION_USES_TEACHER_MARGINS_OR_HIDDEN_TEST_LABELS",
        },
    },
}


TEACHER_ROLES = {
    "surface_lexical": {
        "family": "surface",
        "measures": "observed shortcut x0 xor x1",
        "refuses": "nuisance correction",
        "cost": 1.0,
    },
    "semantic_z0": {
        "family": "semantic_sensor",
        "measures": "latent factor z0",
        "refuses": "final ranking alone",
        "cost": 1.0,
    },
    "verifier_z1": {
        "family": "verifier_sensor",
        "measures": "latent factor z1 and flip localization",
        "refuses": "surface style prior",
        "cost": 1.0,
    },
}


@dataclass
class World:
    x: np.ndarray
    y: np.ndarray
    z0: np.ndarray
    z1: np.ndarray
    n0: np.ndarray
    n1: np.ndarray
    distractors: np.ndarray


@dataclass
class TrainedRun:
    name: str
    hidden_acc: float
    train_size: int
    notes: dict


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


class TinyStudent:
    """Tiny linear readout over generic monomial features."""

    def __init__(self, input_dim: int, rng: np.random.Generator):
        self.w = rng.normal(0.0, 0.05, size=input_dim)
        self.b = 0.0

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(x @ self.w + self.b)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        lr: float,
        weight_decay: float,
        rng: np.random.Generator,
    ) -> None:
        if len(x) == 0:
            raise ValueError("cannot train on empty data")
        n = len(x)
        y = y.astype(np.float64)
        for _ in range(epochs):
            order = rng.permutation(n)
            xb = x[order]
            yb = y[order]
            p = self.predict_proba(xb)
            err = (p - yb) / n
            grad_w = xb.T @ err + weight_decay * self.w
            grad_b = float(err.sum())
            self.w -= lr * grad_w
            self.b -= lr * grad_b


def make_world(n_distractors: int = 4) -> World:
    rows = []
    for z0 in (0, 1):
        for z1 in (0, 1):
            for n0 in (0, 1):
                for n1 in (0, 1):
                    for d_int in range(2 ** n_distractors):
                        ds = [(d_int >> i) & 1 for i in range(n_distractors)]
                        x0 = z0 ^ n0
                        x1 = z1 ^ n1
                        y = z0 ^ z1
                        rows.append((x0, x1, n0, n1, *ds, y, z0, z1))
    arr = np.array(rows, dtype=np.int64)
    x = arr[:, : 4 + n_distractors]
    y = arr[:, 4 + n_distractors]
    z0 = arr[:, 5 + n_distractors]
    z1 = arr[:, 6 + n_distractors]
    return World(
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        z0=z0,
        z1=z1,
        n0=x[:, 2].astype(np.int64),
        n1=x[:, 3].astype(np.int64),
        distractors=x[:, 4:].astype(np.int64),
    )


def student_features(x: np.ndarray, max_degree: int = 4) -> np.ndarray:
    signed = x * 2.0 - 1.0
    parts = [signed]
    cols = range(signed.shape[1])
    for degree in range(2, max_degree + 1):
        products = []
        for combo in combinations(cols, degree):
            products.append(np.prod(signed[:, combo], axis=1))
        parts.append(np.stack(products, axis=1))
    return np.concatenate(parts, axis=1)


def teacher_margins(world: World, x: np.ndarray | None = None) -> dict[str, np.ndarray]:
    if x is None:
        x = world.x
    x_int = x.astype(np.int64)
    x0 = x_int[:, 0]
    x1 = x_int[:, 1]
    n0 = x_int[:, 2]
    n1 = x_int[:, 3]
    z0 = x0 ^ n0
    z1 = x1 ^ n1
    surface = x0 ^ x1

    # Margins are signed candidate-ranking readings. They are deliberately not
    # all final-label sensors.
    return {
        "surface_lexical": np.where(surface == 1, 2.2, -2.2).astype(np.float64),
        "semantic_z0": np.where(z0 == 1, 2.0, -2.0).astype(np.float64),
        "verifier_z1": np.where(z1 == 1, 2.0, -2.0).astype(np.float64),
    }


def margin_probs(margins: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: sigmoid(margin) for name, margin in margins.items()}


def hard_from_prob(prob: np.ndarray) -> np.ndarray:
    return (prob >= 0.5).astype(np.float64)


def packet_labels_from_roles(
    margins: dict[str, np.ndarray],
    role_map: dict[str, str],
    invert: bool = False,
) -> np.ndarray:
    semantic = hard_from_prob(sigmoid(margins[role_map["semantic"]]))
    verifier = hard_from_prob(sigmoid(margins[role_map["verifier"]]))
    labels = np.logical_xor(semantic.astype(bool), verifier.astype(bool)).astype(np.float64)
    if invert:
        labels = 1.0 - labels
    return labels


def _teacher_bits(margins: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.stack(
        [hard_from_prob(sigmoid(margins[name])).astype(np.int64) for name in names],
        axis=1,
    )


def _truth_key(bits: np.ndarray | tuple[int, ...]) -> str:
    return "".join(str(int(v)) for v in bits)


def _truth_states(degree: int) -> list[tuple[int, ...]]:
    return list(product((0, 1), repeat=degree))


def _truth_table_from_mask(degree: int, mask: int) -> dict[str, float]:
    table = {}
    for i, state in enumerate(_truth_states(degree)):
        table[_truth_key(state)] = float((mask >> i) & 1)
    return table


def _truth_table_predict(bits: np.ndarray, truth_table: dict[str, float]) -> np.ndarray:
    return np.array([truth_table[_truth_key(row)] for row in bits], dtype=np.float64)


def _expression_for_truth_table(names: list[str], truth_table: dict[str, float]) -> str:
    states = _truth_states(len(names))
    values = [int(truth_table[_truth_key(state)]) for state in states]
    if len(names) == 1:
        if values == [0, 1]:
            return names[0]
        if values == [1, 0]:
            return f"not({names[0]})"
        if values == [0, 0]:
            return "constant_0"
        if values == [1, 1]:
            return "constant_1"
    if len(names) == 2:
        a, b = names
        known = {
            (0, 1, 1, 0): f"xor({a}, {b})",
            (1, 0, 0, 1): f"xnor({a}, {b})",
            (0, 0, 0, 1): f"and({a}, {b})",
            (0, 1, 1, 1): f"or({a}, {b})",
            (1, 1, 1, 0): f"nand({a}, {b})",
            (1, 0, 0, 0): f"nor({a}, {b})",
            (0, 0, 1, 0): f"{a}_and_not_{b}",
            (0, 1, 0, 0): f"not_{a}_and_{b}",
        }
        if tuple(values) in known:
            return known[tuple(values)]
    return "truth_table(" + ",".join(names) + ")=" + "".join(str(v) for v in values)


def harden_labels(labels: np.ndarray) -> np.ndarray:
    return (labels >= 0.5).astype(np.float64)


def packet_labels_from_rule(margins: dict[str, np.ndarray], rule: dict) -> np.ndarray:
    rule_type = rule["rule_type"]
    if rule_type == "role_xor":
        return packet_labels_from_roles(margins, rule["role_map"], invert=rule["invert"])
    if rule_type == "truth_table":
        bits = _teacher_bits(margins, rule["teacher_names"])
        return _truth_table_predict(bits, rule["truth_table"])
    if rule_type == "single_teacher":
        return hard_from_prob(sigmoid(margins[rule["teacher_name"]]))
    if rule_type == "average_teachers":
        probs = margin_probs(margins)
        return np.mean(np.stack([probs[name] for name in rule["teacher_names"]]), axis=0)
    raise ValueError(f"unknown packet rule type: {rule_type}")


def _single_and_average_stats(
    margins: dict[str, np.ndarray],
    y: np.ndarray,
) -> tuple[dict[str, float], float, float]:
    probs = margin_probs(margins)
    single_accs = {
        name: float(np.mean(hard_from_prob(prob) == y))
        for name, prob in probs.items()
    }
    avg_probs = np.mean(np.stack(list(probs.values())), axis=0)
    avg_acc = float(np.mean(hard_from_prob(avg_probs) == y))
    return single_accs, avg_acc, max(max(single_accs.values()), avg_acc)


def _binary_mutual_information(feature: np.ndarray, y: np.ndarray) -> float:
    x = feature.astype(np.int64)
    yy = y.astype(np.int64)
    total = float(len(yy))
    mi = 0.0
    for xv in (0, 1):
        px = float(np.sum(x == xv)) / total
        if px == 0.0:
            continue
        for yv in (0, 1):
            py = float(np.sum(yy == yv)) / total
            pxy = float(np.sum((x == xv) & (yy == yv))) / total
            if py > 0.0 and pxy > 0.0:
                mi += pxy * math.log2(pxy / (px * py))
    return float(mi)


def _degree2_margin_features(margins: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = list(margins.keys())
    signed = {name: np.where(margins[name] >= 0.0, 1.0, -1.0) for name in names}
    parts = []
    feature_names = []
    for name in names:
        parts.append(signed[name])
        feature_names.append(f"margin:{name}")
    for a, b in combinations(names, 2):
        parts.append(signed[a] * signed[b])
        feature_names.append(f"product:{a}*{b}")
    return np.stack(parts, axis=1), feature_names


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _fit_logistic_degree2(margins: dict[str, np.ndarray], y: np.ndarray) -> dict:
    x, feature_names = _degree2_margin_features(margins)
    w = np.zeros(x.shape[1], dtype=np.float64)
    b = 0.0
    yy = y.astype(np.float64)
    lr = 0.25
    l2 = 1e-3
    for _ in range(600):
        p = sigmoid(x @ w + b)
        err = (p - yy) / len(yy)
        w -= lr * (x.T @ err + l2 * w)
        b -= lr * float(err.sum())
    probs = sigmoid(x @ w + b)
    pred = hard_from_prob(probs)
    top_idx = np.argsort(np.abs(w))[::-1][:5]
    return {
        "calibration_acc": float(np.mean(pred == yy)),
        "bias": float(b),
        "top_weights": [
            {"feature": feature_names[i], "weight": float(w[i])}
            for i in top_idx
        ],
    }


def _product_truth_table_rule(
    pair: list[str],
    negative_product_is_one: bool,
    calibration_acc: float,
) -> dict:
    table = {}
    for state in _truth_states(2):
        product_is_negative = state[0] != state[1]
        label = product_is_negative if negative_product_is_one else not product_is_negative
        table[_truth_key(state)] = float(label)
    return {
        "rule_type": "truth_table",
        "condition": "simple_margin_product",
        "selected_by": "best_single_degree2_product_feature",
        "teacher_names": pair,
        "truth_table": table,
        "degree": 2,
        "expression": _expression_for_truth_table(pair, table),
        "calibration_packet_acc": calibration_acc,
    }


def _margin_relation_diagnostics(margins: dict[str, np.ndarray], y: np.ndarray) -> dict:
    x, feature_names = _degree2_margin_features(margins)
    y_signed = y * 2.0 - 1.0
    feature_rows = []
    best_product = None
    for i, feature_name in enumerate(feature_names):
        values = x[:, i]
        positive_labels = (values > 0.0).astype(np.float64)
        negative_labels = 1.0 - positive_labels
        positive_acc = float(np.mean(positive_labels == y))
        negative_acc = float(np.mean(negative_labels == y))
        best_acc = max(positive_acc, negative_acc)
        binary_feature = positive_labels.astype(np.int64)
        row = {
            "feature": feature_name,
            "abs_corr_with_label": abs(_safe_corr(values, y_signed)),
            "mutual_information_bits": _binary_mutual_information(binary_feature, y),
            "best_threshold_acc": best_acc,
            "best_polarity": "positive_is_one" if positive_acc >= negative_acc else "negative_is_one",
        }
        feature_rows.append(row)
        if feature_name.startswith("product:"):
            names = feature_name.removeprefix("product:").split("*")
            negative_product_is_one = negative_acc > positive_acc
            candidate_rule = _product_truth_table_rule(names, negative_product_is_one, best_acc)
            candidate = {**row, "rule": candidate_rule}
            if best_product is None or candidate["best_threshold_acc"] > best_product["best_threshold_acc"]:
                best_product = candidate
    feature_rows.sort(
        key=lambda row: (
            row["best_threshold_acc"],
            row["mutual_information_bits"],
            row["abs_corr_with_label"],
        ),
        reverse=True,
    )
    return {
        "logistic_degree2": _fit_logistic_degree2(margins, y),
        "top_margin_relations": feature_rows[:8],
        "best_simple_margin_product": best_product,
    }


def _rule_calibration_acc(margins: dict[str, np.ndarray], y: np.ndarray, rule: dict) -> float:
    pred = harden_labels(packet_labels_from_rule(margins, rule))
    return float(np.mean(pred == y))


def hand_authored_packet_rule(world: World, calib_idx: np.ndarray) -> dict:
    margins = teacher_margins(world, world.x[calib_idx])
    y = world.y[calib_idx]
    roles = {"semantic": "semantic_z0", "verifier": "verifier_z1"}
    base_rule = {
        "rule_type": "role_xor",
        "condition": "hand_authored",
        "selected_by": "researcher_supplied_semantic_verifier_xor",
        "role_map": roles,
        "invert": False,
        "expression": "xor(semantic_z0, verifier_z1)",
    }
    xor_acc = _rule_calibration_acc(margins, y, base_rule)
    inverted = {**base_rule, "invert": True, "expression": "not_xor(semantic_z0, verifier_z1)"}
    not_xor_acc = _rule_calibration_acc(margins, y, inverted)
    rule = inverted if not_xor_acc > xor_acc else base_rule
    single_accs, avg_acc, best_no_comp = _single_and_average_stats(margins, y)
    rule.update({
        "calibration_packet_acc": max(xor_acc, not_xor_acc),
        "calibration_single_accs": single_accs,
        "calibration_avg_acc": avg_acc,
        "packet_value_prior": max(xor_acc, not_xor_acc) - best_no_comp,
        "predicted_lesson_type": "counterfactual_xor_ranking_packet",
        "predicted_student_gap": "absent_counterfactual_readout",
    })
    return rule


def no_composition_rule(margins: dict[str, np.ndarray], y: np.ndarray) -> dict:
    teacher_names = list(margins.keys())
    single_accs, avg_acc, _ = _single_and_average_stats(margins, y)
    candidates = []
    for name, acc in single_accs.items():
        candidates.append({
            "rule_type": "single_teacher",
            "condition": "no_composition",
            "selected_by": "best_calibration_single_teacher_or_average",
            "teacher_name": name,
            "expression": name,
            "calibration_packet_acc": acc,
        })
    candidates.append({
        "rule_type": "average_teachers",
        "condition": "no_composition",
        "selected_by": "best_calibration_single_teacher_or_average",
        "teacher_names": teacher_names,
        "expression": "average(" + ",".join(teacher_names) + ")",
        "calibration_packet_acc": avg_acc,
    })
    candidates.sort(key=lambda item: (item["calibration_packet_acc"], item["expression"]), reverse=True)
    rule = candidates[0]
    rule.update({
        "calibration_single_accs": single_accs,
        "calibration_avg_acc": avg_acc,
        "packet_value_prior": 0.0,
        "predicted_lesson_type": "none",
    })
    return rule


def random_packet_rule(
    margins: dict[str, np.ndarray],
    y: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    teacher_names = list(margins.keys())
    table = {
        _truth_key(state): float(rng.integers(0, 2))
        for state in _truth_states(len(teacher_names))
    }
    rule = {
        "rule_type": "truth_table",
        "condition": "random",
        "selected_by": "random_truth_table_over_all_teacher_margins",
        "teacher_names": teacher_names,
        "truth_table": table,
        "degree": len(teacher_names),
        "expression": _expression_for_truth_table(teacher_names, table),
    }
    single_accs, avg_acc, best_no_comp = _single_and_average_stats(margins, y)
    packet_acc = _rule_calibration_acc(margins, y, rule)
    rule.update({
        "calibration_packet_acc": packet_acc,
        "calibration_single_accs": single_accs,
        "calibration_avg_acc": avg_acc,
        "packet_value_prior": packet_acc - best_no_comp,
        "predicted_lesson_type": "random_composition_control",
    })
    return rule


def infer_packet_rule(world: World, calib_idx: np.ndarray) -> dict:
    margins = teacher_margins(world, world.x[calib_idx])
    y = world.y[calib_idx]
    teacher_names = list(margins.keys())
    single_accs, avg_acc, best_no_comp = _single_and_average_stats(margins, y)
    diagnostics = _margin_relation_diagnostics(margins, y)

    candidates = []
    for degree in (1, 2):
        for names_tuple in combinations(teacher_names, degree):
            names = list(names_tuple)
            bits = _teacher_bits(margins, names)
            observed_states = {_truth_key(row) for row in bits}
            for mask in range(2 ** (2 ** degree)):
                table = _truth_table_from_mask(degree, mask)
                pred = _truth_table_predict(bits, table)
                acc = float(np.mean(pred == y))
                output_rate = float(np.mean(pred))
                candidates.append({
                    "rule_type": "truth_table",
                    "condition": "inferred",
                    "selected_by": "exhaustive_boolean_functions_degree_le_2",
                    "teacher_names": names,
                    "truth_table": table,
                    "degree": degree,
                    "expression": _expression_for_truth_table(names, table),
                    "calibration_packet_acc": acc,
                    "state_coverage": float(len(observed_states) / (2 ** degree)),
                    "output_balance": min(output_rate, 1.0 - output_rate),
                })
    candidates.sort(
        key=lambda item: (
            item["calibration_packet_acc"],
            item["state_coverage"],
            item["output_balance"],
            item["degree"],
        ),
        reverse=True,
    )
    rule = dict(candidates[0])
    rule.update({
        "calibration_single_accs": single_accs,
        "calibration_avg_acc": avg_acc,
        "packet_value_prior": rule["calibration_packet_acc"] - best_no_comp,
        "predicted_lesson_type": "inferred_counterfactual_ranking_packet",
        "predicted_student_gap": "calibration_selected_margin_composition",
        "diagnostics": {
            **diagnostics,
            "candidate_count": len(candidates),
            "top_exhaustive_candidates": [
                {
                    "expression": item["expression"],
                    "teacher_names": item["teacher_names"],
                    "degree": item["degree"],
                    "calibration_packet_acc": item["calibration_packet_acc"],
                    "state_coverage": item["state_coverage"],
                    "output_balance": item["output_balance"],
                }
                for item in candidates[:8]
            ],
        },
    })
    return rule

def true_labels_for_x(x: np.ndarray) -> np.ndarray:
    xi = x.astype(np.int64)
    z0 = xi[:, 0] ^ xi[:, 2]
    z1 = xi[:, 1] ^ xi[:, 3]
    return (z0 ^ z1).astype(np.float64)


def transformation_batches(x: np.ndarray) -> list[tuple[str, bool, np.ndarray]]:
    batches = [("identity", False, x.copy())]
    d_start = 4
    # Irrelevant-slot invariances.
    for j in range(d_start, x.shape[1]):
        v = x.copy()
        v[:, j] = 1.0 - v[:, j]
        batches.append((f"irrelevant_slot_flip_{j}", False, v))
    # Nuisance-preserving transformations: keep latent z0/z1 fixed.
    for obs_col, nuisance_col in ((0, 2), (1, 3)):
        v = x.copy()
        v[:, obs_col] = 1.0 - v[:, obs_col]
        v[:, nuisance_col] = 1.0 - v[:, nuisance_col]
        batches.append((f"nuisance_preserving_flip_{obs_col}_{nuisance_col}", False, v))
    # True counterfactual ranking flips: flip one latent factor only.
    for obs_col in (0, 1):
        v = x.copy()
        v[:, obs_col] = 1.0 - v[:, obs_col]
        batches.append((f"single_latent_counterfactual_flip_{obs_col}", True, v))
    return batches


def transform_variants(x: np.ndarray) -> np.ndarray:
    return np.vstack([batch_x for _, _, batch_x in transformation_batches(x)])


def unique_rows(x: np.ndarray) -> np.ndarray:
    _, idx = np.unique(x.astype(np.int64), axis=0, return_index=True)
    return x[np.sort(idx)]


def unique_labeled_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    labels_by_row: dict[tuple[int, ...], float] = {}
    conflicts = 0
    for row, label in zip(x.astype(np.int64), y.astype(np.float64)):
        key = tuple(int(v) for v in row)
        label_value = float(label)
        if key in labels_by_row and labels_by_row[key] != label_value:
            conflicts += 1
            continue
        labels_by_row.setdefault(key, label_value)
    rows = np.array(list(labels_by_row.keys()), dtype=np.float64)
    labels = np.array(list(labels_by_row.values()), dtype=np.float64)
    return rows, labels, conflicts


def build_transform_augmented_examples(
    base_x: np.ndarray,
    base_y: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    xs = []
    ys = []
    transform_counts = {}
    for name, flips_label, batch_x in transformation_batches(base_x):
        xs.append(batch_x)
        ys.append(1.0 - base_y if flips_label else base_y.copy())
        transform_counts[name] = int(len(batch_x))
    augmented_x, augmented_y, conflicts = unique_labeled_rows(np.vstack(xs), np.concatenate(ys))
    if len(augmented_x) > limit:
        keep = rng.choice(len(augmented_x), size=limit, replace=False)
        augmented_x = augmented_x[keep]
        augmented_y = augmented_y[keep]
    meta = {
        "n_augmented_examples": int(len(augmented_x)),
        "label_source": "calibration_labels_plus_transformation_preserve_flip_semantics",
        "uses_teacher_signals": False,
        "transformation_conflicts": int(conflicts),
        "transform_counts": transform_counts,
    }
    return augmented_x, augmented_y, meta


def nuisance_oracle_labels(x: np.ndarray) -> np.ndarray:
    return true_labels_for_x(x)


def _rule_teacher_count(rule: dict) -> int:
    if rule["rule_type"] == "single_teacher":
        return 1
    if rule["rule_type"] == "role_xor":
        return len(set(rule["role_map"].values()))
    return len(rule.get("teacher_names", TEACHER_ROLES))


def build_packet_candidate_examples(
    world: World,
    source_idx: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    base = world.x[source_idx]
    transform_counts = {}
    transformed_batches = []
    for name, _, batch_x in transformation_batches(base):
        transform_counts[name] = int(len(batch_x))
        transformed_batches.append(batch_x)
    transformed = unique_rows(np.vstack(transformed_batches))
    candidate_count_before_limit = int(len(transformed))
    if len(transformed) > limit:
        transformed = transformed[rng.choice(len(transformed), size=limit, replace=False)]
    meta = {
        "n_packet_examples": int(len(transformed)),
        "n_candidate_examples_before_limit": candidate_count_before_limit,
        "transforms": [
            "irrelevant_slot_flip",
            "nuisance_preserving_flip",
            "single_latent_counterfactual_flip",
        ],
        "transform_counts": transform_counts,
    }
    return transformed, meta


def packet_meta_for_rule(packet_x: np.ndarray, packet_rule: dict) -> dict:
    return {
        "n_packet_examples": int(len(packet_x)),
        "composition_condition": packet_rule["condition"],
        "rule_type": packet_rule["rule_type"],
        "rule_expression": packet_rule.get("expression", "unknown"),
        "selected_by": packet_rule.get("selected_by", "unknown"),
        "calibration_packet_acc": packet_rule.get("calibration_packet_acc"),
        "packet_value_prior": packet_rule.get("packet_value_prior"),
        "teacher_measurement_cost": int(len(packet_x) * _rule_teacher_count(packet_rule)),
        "uses_teacher_signals": True,
    }


def label_packet_examples(
    world: World,
    packet_x: np.ndarray,
    packet_rule: dict,
) -> tuple[np.ndarray, dict]:
    margins = teacher_margins(world, packet_x)
    labels = packet_labels_from_rule(margins, packet_rule)
    return labels, packet_meta_for_rule(packet_x, packet_rule)


def build_e3_packet_examples(
    world: World,
    source_idx: np.ndarray,
    packet_rule: dict,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    transformed, _ = build_packet_candidate_examples(world, source_idx, limit, rng)
    labels, meta = label_packet_examples(world, transformed, packet_rule)
    return transformed, labels, meta

def train_and_eval(
    name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    hidden_x: np.ndarray,
    hidden_y: np.ndarray,
    seed: int,
    epochs: int,
    lr: float = 0.08,
) -> TrainedRun:
    rng = np.random.default_rng(seed + 1009)
    train_feat = student_features(train_x)
    hidden_feat = student_features(hidden_x)
    model = TinyStudent(train_feat.shape[1], rng=rng)
    model.fit(train_feat, train_y, epochs, lr, 1e-4, rng)
    pred = hard_from_prob(model.predict_proba(hidden_feat))
    acc = float(np.mean(pred == hidden_y))
    return TrainedRun(name=name, hidden_acc=acc, train_size=int(len(train_x)), notes={})


def combine_supervised_and_pseudo(
    calib_x: np.ndarray,
    calib_y: np.ndarray,
    pseudo_x: np.ndarray,
    pseudo_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([calib_x, pseudo_x]), np.concatenate([calib_y, pseudo_y])


def run_one_seed(seed: int, epochs: int, packet_limit: int, smoke: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    world = make_world(n_distractors=3 if smoke else 4)
    hidden_mask = (world.n0 == 1) & (world.n1 == 0)
    calibration_mask = ((world.n0 == 0) & (world.n1 == 0)) | ((world.n0 == 0) & (world.n1 == 1))
    source_mask = ~hidden_mask

    hidden_idx = np.flatnonzero(hidden_mask)
    source_idx_all = np.flatnonzero(source_mask)
    calib_candidates = np.flatnonzero(calibration_mask)
    calib_n = 16 if smoke else 32
    calib_idx = rng.choice(calib_candidates, size=calib_n, replace=False)
    pool_n = 32 if smoke else 96
    source_idx = rng.choice(source_idx_all, size=pool_n, replace=False)

    calib_x = world.x[calib_idx]
    calib_y = world.y[calib_idx]
    hidden_x = world.x[hidden_idx]
    hidden_y = world.y[hidden_idx]
    pool_x = world.x[source_idx]
    pool_margins = teacher_margins(world, pool_x)
    pool_probs = margin_probs(pool_margins)
    calib_margins = teacher_margins(world, calib_x)

    inferred_rule = infer_packet_rule(world, calib_idx)
    hand_rule = hand_authored_packet_rule(world, calib_idx)
    random_rule_rng = np.random.default_rng(seed + 424242)
    random_rule = random_packet_rule(calib_margins, calib_y, random_rule_rng)
    no_comp_rule = no_composition_rule(calib_margins, calib_y)
    condition_rules = {
        "inferred": inferred_rule,
        "hand_authored": hand_rule,
        "random": random_rule,
        "no_composition": no_comp_rule,
    }

    packet_x, packet_candidate_meta = build_packet_candidate_examples(
        world, source_idx, packet_limit, rng)
    inferred_y, inferred_meta = label_packet_examples(world, packet_x, inferred_rule)
    hand_y, hand_meta = label_packet_examples(world, packet_x, hand_rule)
    random_y, random_meta = label_packet_examples(world, packet_x, random_rule)
    no_comp_y, no_comp_meta = label_packet_examples(world, packet_x, no_comp_rule)
    condition_packet_meta = {
        "inferred": inferred_meta,
        "hand_authored": hand_meta,
        "random": random_meta,
        "no_composition": no_comp_meta,
    }

    runs: list[TrainedRun] = []
    runs.append(train_and_eval(
        "B0_CE_only_same_student", calib_x, calib_y,
        hidden_x, hidden_y, seed, epochs))

    single_runs = []
    for teacher_name, probs in pool_probs.items():
        tx, ty = combine_supervised_and_pseudo(
            calib_x, calib_y, pool_x, hard_from_prob(probs))
        single_runs.append(train_and_eval(
            f"B4_single_{teacher_name}", tx, ty, hidden_x, hidden_y,
            seed + len(single_runs) * 17, epochs))
    runs.extend(single_runs)

    avg_prob = np.mean(np.stack(list(pool_probs.values())), axis=0)
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, pool_x, avg_prob)
    runs.append(train_and_eval(
        "B5_naive_teacher_average", tx, ty, hidden_x, hidden_y,
        seed + 201, epochs))

    # Dawid-Skene style proxy: reliability-weighted vote from calibration labels.
    calib_probs = margin_probs(calib_margins)
    weights = []
    ordered_names = list(pool_probs.keys())
    for teacher_name in ordered_names:
        acc = np.mean(hard_from_prob(calib_probs[teacher_name]) == calib_y)
        weights.append(max(acc - 0.5, 0.0) + 1e-3)
    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= weights_arr.sum()
    weighted_prob = sum(weights_arr[i] * pool_probs[name] for i, name in enumerate(ordered_names))
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, pool_x, weighted_prob)
    runs.append(train_and_eval(
        "B6_weighted_vote_calibrated", tx, ty, hidden_x, hidden_y,
        seed + 301, epochs))

    disagreement = np.var(np.stack(list(pool_probs.values())), axis=0)
    active_n = min(len(pool_x), max(8, packet_limit // 3))
    active_idx = np.argsort(disagreement)[-active_n:]
    tx, ty = combine_supervised_and_pseudo(
        calib_x, calib_y, pool_x[active_idx], avg_prob[active_idx])
    runs.append(train_and_eval(
        "B9_active_hard_examples_average_label", tx, ty,
        hidden_x, hidden_y, seed + 401, epochs))

    shuffled_labels = inferred_y.copy()
    rng.shuffle(shuffled_labels)
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, shuffled_labels)
    runs.append(train_and_eval(
        "B7_shuffled_teacher_measurements", tx, ty, hidden_x, hidden_y,
        seed + 501, epochs))

    shuffled_roles = {"semantic": "surface_lexical", "verifier": "semantic_z0"}
    shuffled_identity_y = packet_labels_from_roles(
        teacher_margins(world, packet_x), shuffled_roles, invert=hand_rule["invert"])
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, shuffled_identity_y)
    runs.append(train_and_eval(
        "B8_shuffled_teacher_identity", tx, ty, hidden_x, hidden_y,
        seed + 601, epochs))

    aug_probs = np.mean(
        np.stack(list(margin_probs(teacher_margins(world, packet_x)).values())),
        axis=0,
    )
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, aug_probs)
    runs.append(train_and_eval(
        "B10_counterfactual_augmentation_no_tomography", tx, ty,
        hidden_x, hidden_y, seed + 701, epochs))

    enhanced_aug_x, enhanced_aug_y, enhanced_aug_meta = build_transform_augmented_examples(
        calib_x, calib_y, packet_limit, rng)
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, enhanced_aug_x, enhanced_aug_y)
    b10_plus = train_and_eval(
        "B10_plus_enhanced_counterfactual_augmentation", tx, ty,
        hidden_x, hidden_y, seed + 751, epochs)
    b10_plus.notes.update(enhanced_aug_meta)
    runs.append(b10_plus)

    nuisance_y = nuisance_oracle_labels(packet_x)
    nuisance_match = float(np.mean(nuisance_y == harden_labels(inferred_y))) if len(inferred_y) else float("nan")
    shared_condition_seed = seed + 851
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, nuisance_y)
    b15 = train_and_eval(
        "B15_nuisance_oracle_against_inferred_packet", tx, ty,
        hidden_x, hidden_y, shared_condition_seed, epochs)
    b15.notes.update({
        "n_oracle_labeled_examples": int(len(packet_x)),
        "oracle_knowledge": "n0_n1_nuisance_bits_and_transformation_rules_no_teacher_identity",
        "uses_teacher_signals": False,
        "matches_inferred_packet_labels": nuisance_match,
    })
    runs.append(b15)

    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, inferred_y)
    inferred_run = train_and_eval(
        "E3_inferred_composition_lesson_packets", tx, ty,
        hidden_x, hidden_y, shared_condition_seed, epochs)
    inferred_run.notes.update(inferred_meta)
    runs.append(inferred_run)

    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, hand_y)
    hand_run = train_and_eval(
        "E3_hand_authored_composition_lesson_packets", tx, ty,
        hidden_x, hidden_y, shared_condition_seed, epochs)
    hand_run.notes.update(hand_meta)
    runs.append(hand_run)

    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, random_y)
    random_run = train_and_eval(
        "B16_random_composition_function", tx, ty,
        hidden_x, hidden_y, seed + 951, epochs)
    random_run.notes.update(random_meta)
    runs.append(random_run)

    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, no_comp_y)
    no_comp_run = train_and_eval(
        "B17_no_composition_best_single_or_average", tx, ty,
        hidden_x, hidden_y, seed + 1001, epochs)
    no_comp_run.notes.update(no_comp_meta)
    runs.append(no_comp_run)

    true_oracle_acc = 1.0
    exact_tool_label = "admitted_hostile_absorber"
    runs.append(TrainedRun(
        name="B13_exact_domain_tool_hidden_constructor",
        hidden_acc=true_oracle_acc,
        train_size=0,
        notes={
            "oracle_knowledge": "hidden_constructor_n0_1_n1_0_and_true_ranking_formula",
            "uses_teacher_signals": False,
            "mode": "direct_reconstruction_no_student_training",
        },
    ))

    all_margins = teacher_margins(world, world.x)
    hidden_margins = teacher_margins(world, hidden_x)
    hand_all = harden_labels(packet_labels_from_rule(all_margins, hand_rule))
    inferred_all = harden_labels(packet_labels_from_rule(all_margins, inferred_rule))
    hand_hidden = harden_labels(packet_labels_from_rule(hidden_margins, hand_rule))
    inferred_hidden = harden_labels(packet_labels_from_rule(hidden_margins, inferred_rule))
    simple_product = inferred_rule["diagnostics"]["best_simple_margin_product"]
    simple_rule = simple_product["rule"] if simple_product else None
    simple_matches_hand = float("nan")
    simple_expression = None
    if simple_rule is not None:
        simple_all = harden_labels(packet_labels_from_rule(all_margins, simple_rule))
        simple_matches_hand = float(np.mean(simple_all == hand_all))
        simple_expression = simple_rule["expression"]

    result = {
        "seed": seed,
        "hidden_count": int(len(hidden_idx)),
        "calibration_count": int(len(calib_idx)),
        "source_pool_count": int(len(source_idx)),
        "packet_rule": inferred_rule,
        "condition_rules": condition_rules,
        "packet_meta": inferred_meta,
        "packet_candidate_meta": packet_candidate_meta,
        "condition_packet_meta": condition_packet_meta,
        "runs": {r.name: {"hidden_acc": r.hidden_acc, "train_size": r.train_size, **r.notes} for r in runs},
        "diagnostics": {
            "exact_domain_tool_hidden_acc": true_oracle_acc,
            "exact_domain_tool_label": exact_tool_label,
            "active_query_count": int(active_n),
            "weighted_vote_weights": dict(zip(ordered_names, weights_arr.tolist())),
            "inferred_matches_hand_authored_on_world": float(np.mean(inferred_all == hand_all)),
            "inferred_matches_hand_authored_on_hidden": float(np.mean(inferred_hidden == hand_hidden)),
            "inferred_matches_hand_authored_on_packet": float(np.mean(harden_labels(inferred_y) == harden_labels(hand_y))),
            "inferred_matches_b15_oracle_on_packet": nuisance_match,
            "simple_product_matches_hand_authored_on_world": simple_matches_hand,
            "simple_product_expression": simple_expression,
            "logistic_degree2_calibration_acc": inferred_rule["diagnostics"]["logistic_degree2"]["calibration_acc"],
            "logistic_degree2_top_weights": inferred_rule["diagnostics"]["logistic_degree2"]["top_weights"],
        },
    }
    return result

def _count_values(values: list[str | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "None" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize(results: list[dict]) -> dict:
    names = sorted(results[0]["runs"].keys())
    metrics = {}
    for name in names:
        vals = np.array([r["runs"][name]["hidden_acc"] for r in results], dtype=np.float64)
        metrics[name] = {
            "mean_hidden_acc": float(vals.mean()),
            "std_hidden_acc": float(vals.std()),
            "min_hidden_acc": float(vals.min()),
            "max_hidden_acc": float(vals.max()),
        }

    inferred_name = "E3_inferred_composition_lesson_packets"
    hand_name = "E3_hand_authored_composition_lesson_packets"
    random_name = "B16_random_composition_function"
    no_comp_name = "B17_no_composition_best_single_or_average"
    b15_name = "B15_nuisance_oracle_against_inferred_packet"

    inferred = metrics[inferred_name]["mean_hidden_acc"]
    hand = metrics[hand_name]["mean_hidden_acc"]
    random_comp = metrics[random_name]["mean_hidden_acc"]
    no_comp = metrics[no_comp_name]["mean_hidden_acc"]
    b15 = metrics[b15_name]["mean_hidden_acc"]
    best_single = max(v["mean_hidden_acc"] for k, v in metrics.items() if k.startswith("B4_single_"))
    avg_or_weighted = max(
        metrics["B5_naive_teacher_average"]["mean_hidden_acc"],
        metrics["B6_weighted_vote_calibrated"]["mean_hidden_acc"],
    )
    active = metrics["B9_active_hard_examples_average_label"]["mean_hidden_acc"]
    shuffled = max(
        metrics["B7_shuffled_teacher_measurements"]["mean_hidden_acc"],
        metrics["B8_shuffled_teacher_identity"]["mean_hidden_acc"],
    )
    augmentation = metrics["B10_counterfactual_augmentation_no_tomography"]["mean_hidden_acc"]
    enhanced_augmentation = metrics["B10_plus_enhanced_counterfactual_augmentation"]["mean_hidden_acc"]
    exact_tool = metrics["B13_exact_domain_tool_hidden_constructor"]["mean_hidden_acc"]
    ce = metrics["B0_CE_only_same_student"]["mean_hidden_acc"]
    ordinary = max(ce, best_single, avg_or_weighted, active, shuffled, augmentation)
    hostile_non_exact = max(ordinary, enhanced_augmentation, b15, hand, random_comp, no_comp)
    all_absorbers = max(hostile_non_exact, exact_tool)

    margins = {
        "inferred_minus_hand_authored_pp": 100.0 * (inferred - hand),
        "inferred_minus_b15_nuisance_oracle_pp": 100.0 * (inferred - b15),
        "inferred_minus_random_composition_pp": 100.0 * (inferred - random_comp),
        "inferred_minus_no_composition_pp": 100.0 * (inferred - no_comp),
        "inferred_minus_best_ordinary_pp": 100.0 * (inferred - ordinary),
        "inferred_minus_best_non_exact_absorber_pp": 100.0 * (inferred - hostile_non_exact),
        "inferred_minus_best_all_absorber_pp": 100.0 * (inferred - all_absorbers),
        "inferred_minus_ce_pp": 100.0 * (inferred - ce),
        "inferred_minus_best_single_pp": 100.0 * (inferred - best_single),
        "inferred_minus_avg_or_weighted_pp": 100.0 * (inferred - avg_or_weighted),
        "inferred_minus_active_pp": 100.0 * (inferred - active),
        "inferred_minus_shuffled_pp": 100.0 * (inferred - shuffled),
        "inferred_minus_augmentation_pp": 100.0 * (inferred - augmentation),
        "inferred_minus_b10_plus_enhanced_augmentation_pp": 100.0 * (inferred - enhanced_augmentation),
        "inferred_minus_b13_exact_domain_tool_pp": 100.0 * (inferred - exact_tool),
    }

    priors = np.array([r["packet_rule"]["packet_value_prior"] for r in results], dtype=np.float64)
    non_exact_realized = np.array([
        r["runs"][inferred_name]["hidden_acc"]
        - max(
            v["hidden_acc"]
            for k, v in r["runs"].items()
            if k != inferred_name and not k.startswith("B13_exact_domain_tool")
        )
        for r in results
    ])
    all_realized = np.array([
        r["runs"][inferred_name]["hidden_acc"]
        - max(v["hidden_acc"] for k, v in r["runs"].items() if k != inferred_name)
        for r in results
    ])
    forecast_ok_non_exact = bool(
        np.mean(priors > 0.0) >= 0.75 and np.mean(non_exact_realized > 0.0) >= 0.75
    )
    forecast_ok_all = bool(
        np.mean(priors > 0.0) >= 0.75 and np.mean(all_realized > 0.0) >= 0.75
    )

    gate = PRECOMMIT["continuation_gate"]
    absorber_precommit = PRECOMMIT["absorber_tests"]
    absorber_verdicts = {
        "B13_exact_domain_tool": {
            "token": (
                absorber_precommit["B13_exact_domain_tool"]["kill_token"]
                if exact_tool >= inferred
                else absorber_precommit["B13_exact_domain_tool"]["confirm_token"]
            ),
            "absorbed": bool(exact_tool >= inferred),
            "margin_pp": margins["inferred_minus_b13_exact_domain_tool_pp"],
        },
        "B15_nuisance_oracle_against_inferred": {
            "token": (
                absorber_precommit["B15_nuisance_oracle"]["kill_token"]
                if margins["inferred_minus_b15_nuisance_oracle_pp"] < gate["inferred_beats_b15_by_pp"]
                else absorber_precommit["B15_nuisance_oracle"]["confirm_token"]
            ),
            "absorbed": bool(
                margins["inferred_minus_b15_nuisance_oracle_pp"] < gate["inferred_beats_b15_by_pp"]
            ),
            "margin_pp": margins["inferred_minus_b15_nuisance_oracle_pp"],
        },
        "B10_plus_enhanced_augmentation": {
            "token": (
                absorber_precommit["B10_plus_enhanced_augmentation"]["kill_token"]
                if margins["inferred_minus_b10_plus_enhanced_augmentation_pp"]
                < gate["beat_enhanced_augmentation_by_pp"]
                else absorber_precommit["B10_plus_enhanced_augmentation"]["confirm_token"]
            ),
            "absorbed": bool(
                margins["inferred_minus_b10_plus_enhanced_augmentation_pp"]
                < gate["beat_enhanced_augmentation_by_pp"]
            ),
            "margin_pp": margins["inferred_minus_b10_plus_enhanced_augmentation_pp"],
        },
    }

    world_match = np.array([
        r["diagnostics"]["inferred_matches_hand_authored_on_world"]
        for r in results
    ], dtype=np.float64)
    hidden_match = np.array([
        r["diagnostics"]["inferred_matches_hand_authored_on_hidden"]
        for r in results
    ], dtype=np.float64)
    packet_match = np.array([
        r["diagnostics"]["inferred_matches_hand_authored_on_packet"]
        for r in results
    ], dtype=np.float64)
    b15_packet_match = np.array([
        r["diagnostics"]["inferred_matches_b15_oracle_on_packet"]
        for r in results
    ], dtype=np.float64)
    simple_product_match = np.array([
        r["diagnostics"]["simple_product_matches_hand_authored_on_world"]
        for r in results
    ], dtype=np.float64)
    logistic_acc = np.array([
        r["diagnostics"]["logistic_degree2_calibration_acc"]
        for r in results
    ], dtype=np.float64)
    selected_expressions = [r["packet_rule"]["expression"] for r in results]
    simple_product_expressions = [r["diagnostics"]["simple_product_expression"] for r in results]
    top_logistic_features = [
        r["diagnostics"]["logistic_degree2_top_weights"][0]["feature"]
        for r in results
    ]

    mean_world_match = float(np.nanmean(world_match))
    inferred_matches_hand = mean_world_match >= gate["inferred_matches_hand_rate"]
    inferred_transfers = (
        margins["inferred_minus_hand_authored_pp"]
        >= -gate["inferred_transfer_within_hand_pp"]
    )
    inferred_beats_b15 = margins["inferred_minus_b15_nuisance_oracle_pp"] >= gate["inferred_beats_b15_by_pp"]
    simple_product_discovers = np.nan_to_num(simple_product_match, nan=0.0) >= gate["inferred_matches_hand_rate"]
    trivial_discovery = bool(np.mean(simple_product_discovers) >= 0.75)

    if not all(math.isfinite(v["mean_hidden_acc"]) for v in metrics.values()):
        token = "E3_TOY_VOID_NONFINITE"
    elif not inferred_matches_hand or not inferred_transfers:
        token = "E3_INFERENCE_FAILS"
    elif inferred_beats_b15:
        token = PRECOMMIT["signal_token"]
    elif trivial_discovery:
        token = "E3_INFERENCE_TRIVIAL"
    else:
        token = "E3_INFERRED_MATCHES_SUPPLIED"

    secondary_token = (
        "E3_INFERRED_SIGNAL"
        if inferred_beats_b15
        else "E3_INFERRED_MATCHES_SUPPLIED"
        if inferred_matches_hand and inferred_transfers
        else "E3_INFERENCE_FAILS"
    )

    return {
        "precommit": PRECOMMIT,
        "summary_metrics": metrics,
        "condition_comparison": {
            "INFERRED_composition": metrics[inferred_name],
            "HAND_AUTHORED_composition": metrics[hand_name],
            "RANDOM_composition": metrics[random_name],
            "NO_composition_best_single_or_average": metrics[no_comp_name],
            "B15_nuisance_oracle_against_inferred": metrics[b15_name],
        },
        "margins": margins,
        "absorber_verdicts": absorber_verdicts,
        "inference_gate": {
            "mean_inferred_matches_hand_authored_on_world": mean_world_match,
            "mean_inferred_matches_hand_authored_on_hidden": float(np.nanmean(hidden_match)),
            "mean_inferred_matches_hand_authored_on_packet": float(np.nanmean(packet_match)),
            "mean_inferred_matches_b15_oracle_on_packet": float(np.nanmean(b15_packet_match)),
            "seeds_inferred_matches_hand_authored": int(np.sum(world_match >= gate["inferred_matches_hand_rate"])),
            "seeds_inferred_beats_b15_by_3pp": int(np.sum([
                r["runs"][inferred_name]["hidden_acc"] - r["runs"][b15_name]["hidden_acc"] >= 0.03
                for r in results
            ])),
            "mean_simple_product_matches_hand_authored_on_world": float(np.nanmean(simple_product_match)),
            "seeds_simple_product_discovers_rule": int(np.sum(simple_product_discovers)),
            "mean_logistic_degree2_calibration_acc": float(np.mean(logistic_acc)),
            "selected_rule_expressions": _count_values(selected_expressions),
            "simple_product_expressions": _count_values(simple_product_expressions),
            "top_logistic_features": _count_values(top_logistic_features),
            "matched_hand_authored": bool(inferred_matches_hand),
            "transfers_to_hidden": bool(inferred_transfers),
            "beats_b15_by_3pp": bool(inferred_beats_b15),
            "trivial_simple_product_discovery": bool(trivial_discovery),
            "secondary_token": secondary_token,
        },
        "packet_value_forecast": {
            "mean_prior": float(priors.mean()),
            "mean_realized_vs_best_non_exact_absorber": float(non_exact_realized.mean()),
            "mean_realized_vs_best_all_absorber": float(all_realized.mean()),
            "forecast_ok_non_exact": forecast_ok_non_exact,
            "forecast_ok_all_absorbers": forecast_ok_all,
        },
        "terminal_token": token,
    }

def run_experiment(seeds: list[int], epochs: int, packet_limit: int, smoke: bool = False) -> dict:
    results = [run_one_seed(seed, epochs, packet_limit, smoke=smoke) for seed in seeds]
    summary = summarize(results)
    summary["seeds"] = seeds
    summary["epochs"] = epochs
    summary["packet_limit"] = packet_limit
    summary["smoke"] = smoke
    summary["per_seed"] = results
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run E3 teacher tomography toy experiment")
    parser.add_argument("--smoke", action="store_true", help="fast smoke run")
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds")
    parser.add_argument("--epochs", type=int, default=450)
    parser.add_argument("--packet-limit", type=int, default=128)
    parser.add_argument("--output", default="experiments/e3_teacher_tomography_result.json")
    args = parser.parse_args()

    if args.smoke:
        seeds = [0]
        epochs = min(args.epochs, 80)
        packet_limit = min(args.packet_limit, 32)
    else:
        seeds = list(range(args.seeds))
        epochs = args.epochs
        packet_limit = args.packet_limit

    report = run_experiment(seeds, epochs, packet_limit, smoke=args.smoke)
    out = Path(args.output)
    if out.parent and not out.parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {out.parent}")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["summary_metrics"]
    print("E3 teacher tomography toy")
    print(f"terminal_token: {report['terminal_token']}")
    for name, vals in sorted(metrics.items()):
        print(f"{name}: mean_hidden_acc={vals['mean_hidden_acc']:.4f} min={vals['min_hidden_acc']:.4f}")
    print("margins_pp:")
    for key, value in report["margins"].items():
        print(f"  {key}: {value:.2f}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
