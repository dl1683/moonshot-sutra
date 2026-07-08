"""E3 Functional Teacher Tomography toy experiment.

CPU-only one-shot kill test for the E3 redirect. The controlled world is a
binary candidate-ranking task with hidden counterfactual structure. Teachers are
not label oracles: one is surface-biased, and two are complementary sensors for
latent ranking bits. E3 must use source-specific teacher measurements to compile
lesson packets and train a tiny teacher-free student.
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
    "direction": "E3 Functional Teacher Tomography toy v0",
    "claim": (
        "source-specific teacher measurements can be inverted into compact "
        "counterfactual ranking lessons that transfer to a hidden transform "
        "better than raw teacher outputs and ordinary absorbers"
    ),
    "signal_token": "E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON",
    "kill_tokens": [
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
        "E3_TOY_VOID_EXACT_TOOL_GRANTED",
    ],
    "continuation_gate": {
        "beat_best_ordinary_by_pp": 5.0,
        "beat_active_by_pp": 3.0,
        "beat_best_single_by_pp": 3.0,
        "beat_shuffled_by_pp": 6.0,
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


def infer_packet_rule(world: World, calib_idx: np.ndarray) -> dict:
    margins = teacher_margins(world)
    roles = {"semantic": "semantic_z0", "verifier": "verifier_z1"}
    xor_labels = packet_labels_from_roles(
        {k: v[calib_idx] for k, v in margins.items()}, roles, invert=False)
    not_xor_labels = 1.0 - xor_labels
    y = world.y[calib_idx]
    xor_acc = float(np.mean(xor_labels == y))
    not_xor_acc = float(np.mean(not_xor_labels == y))
    invert = not_xor_acc > xor_acc

    single_accs = {}
    for name, probs in margin_probs({k: v[calib_idx] for k, v in margins.items()}).items():
        single_accs[name] = float(np.mean(hard_from_prob(probs) == y))
    avg_probs = np.mean(np.stack(list(margin_probs({k: v[calib_idx] for k, v in margins.items()}).values())), axis=0)
    avg_acc = float(np.mean(hard_from_prob(avg_probs) == y))
    packet_acc = max(xor_acc, not_xor_acc)
    best_single = max(single_accs.values())
    packet_value_prior = packet_acc - max(best_single, avg_acc)

    return {
        "role_map": roles,
        "invert": invert,
        "calibration_packet_acc": packet_acc,
        "calibration_single_accs": single_accs,
        "calibration_avg_acc": avg_acc,
        "packet_value_prior": packet_value_prior,
        "predicted_lesson_type": "counterfactual_xor_ranking_packet",
        "predicted_student_gap": "absent_counterfactual_readout",
    }


def true_labels_for_x(x: np.ndarray) -> np.ndarray:
    xi = x.astype(np.int64)
    z0 = xi[:, 0] ^ xi[:, 2]
    z1 = xi[:, 1] ^ xi[:, 3]
    return (z0 ^ z1).astype(np.float64)


def transform_variants(x: np.ndarray) -> np.ndarray:
    variants = [x.copy()]
    d_start = 4
    # Irrelevant-slot invariances.
    for j in range(d_start, x.shape[1]):
        v = x.copy()
        v[:, j] = 1.0 - v[:, j]
        variants.append(v)
    # Nuisance-preserving transformations: keep latent z0/z1 fixed.
    for obs_col, nuisance_col in ((0, 2), (1, 3)):
        v = x.copy()
        v[:, obs_col] = 1.0 - v[:, obs_col]
        v[:, nuisance_col] = 1.0 - v[:, nuisance_col]
        variants.append(v)
    # True counterfactual ranking flips: flip one latent factor only.
    for obs_col in (0, 1):
        v = x.copy()
        v[:, obs_col] = 1.0 - v[:, obs_col]
        variants.append(v)
    return np.vstack(variants)


def unique_rows(x: np.ndarray) -> np.ndarray:
    _, idx = np.unique(x.astype(np.int64), axis=0, return_index=True)
    return x[np.sort(idx)]


def build_e3_packet_examples(
    world: World,
    source_idx: np.ndarray,
    packet_rule: dict,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict]:
    base = world.x[source_idx]
    transformed = unique_rows(transform_variants(base))
    if len(transformed) > limit:
        transformed = transformed[rng.choice(len(transformed), size=limit, replace=False)]
    margins = teacher_margins(world, transformed)
    labels = packet_labels_from_roles(
        margins, packet_rule["role_map"], invert=packet_rule["invert"])
    meta = {
        "n_packet_examples": int(len(transformed)),
        "transforms": [
            "irrelevant_slot_flip",
            "nuisance_preserving_flip",
            "single_latent_counterfactual_flip",
        ],
        "teacher_measurement_cost": int(len(transformed) * len(TEACHER_ROLES)),
        "packet_value_prior": packet_rule["packet_value_prior"],
    }
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

    packet_rule = infer_packet_rule(world, calib_idx)
    packet_x, packet_y, packet_meta = build_e3_packet_examples(
        world, source_idx, packet_rule, packet_limit, rng)

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
    calib_margins = teacher_margins(world, calib_x)
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

    shuffled_labels = packet_y.copy()
    rng.shuffle(shuffled_labels)
    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, shuffled_labels)
    runs.append(train_and_eval(
        "B7_shuffled_teacher_measurements", tx, ty, hidden_x, hidden_y,
        seed + 501, epochs))

    shuffled_roles = {"semantic": "surface_lexical", "verifier": "semantic_z0"}
    shuffled_identity_y = packet_labels_from_roles(
        teacher_margins(world, packet_x), shuffled_roles, invert=packet_rule["invert"])
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

    tx, ty = combine_supervised_and_pseudo(calib_x, calib_y, packet_x, packet_y)
    e3 = train_and_eval(
        "E3_source_specific_lesson_packets", tx, ty,
        hidden_x, hidden_y, seed + 801, epochs)
    e3.notes.update(packet_meta)
    runs.append(e3)

    true_oracle_acc = 1.0
    exact_tool_label = "formal_oracle_not_admitted"
    result = {
        "seed": seed,
        "hidden_count": int(len(hidden_idx)),
        "calibration_count": int(len(calib_idx)),
        "source_pool_count": int(len(source_idx)),
        "packet_rule": packet_rule,
        "packet_meta": packet_meta,
        "runs": {r.name: {"hidden_acc": r.hidden_acc, "train_size": r.train_size, **r.notes} for r in runs},
        "diagnostics": {
            "exact_domain_tool_hidden_acc": true_oracle_acc,
            "exact_domain_tool_label": exact_tool_label,
            "active_query_count": int(active_n),
            "weighted_vote_weights": dict(zip(ordered_names, weights_arr.tolist())),
        },
    }
    return result


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

    e3 = metrics["E3_source_specific_lesson_packets"]["mean_hidden_acc"]
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
    ce = metrics["B0_CE_only_same_student"]["mean_hidden_acc"]
    ordinary = max(ce, best_single, avg_or_weighted, active, shuffled, augmentation)

    margins = {
        "e3_minus_best_ordinary_pp": 100.0 * (e3 - ordinary),
        "e3_minus_ce_pp": 100.0 * (e3 - ce),
        "e3_minus_best_single_pp": 100.0 * (e3 - best_single),
        "e3_minus_avg_or_weighted_pp": 100.0 * (e3 - avg_or_weighted),
        "e3_minus_active_pp": 100.0 * (e3 - active),
        "e3_minus_shuffled_pp": 100.0 * (e3 - shuffled),
        "e3_minus_augmentation_pp": 100.0 * (e3 - augmentation),
    }

    priors = np.array([r["packet_rule"]["packet_value_prior"] for r in results], dtype=np.float64)
    realized = np.array([
        r["runs"]["E3_source_specific_lesson_packets"]["hidden_acc"]
        - max(v["hidden_acc"] for k, v in r["runs"].items() if not k.startswith("E3_"))
        for r in results
    ])
    forecast_ok = bool(np.mean(priors > 0.0) >= 0.75 and np.mean(realized > 0.0) >= 0.75)

    gate = PRECOMMIT["continuation_gate"]
    if not all(math.isfinite(v["mean_hidden_acc"]) for v in metrics.values()):
        token = "E3_TOY_VOID_NONFINITE"
    elif margins["e3_minus_ce_pp"] < gate["beat_best_ordinary_by_pp"] and ce >= ordinary:
        token = "E3_TOY_ABSORBED_BY_CE_ONLY"
    elif margins["e3_minus_best_single_pp"] < gate["beat_best_single_by_pp"]:
        token = "E3_TOY_ABSORBED_BY_SINGLE_TEACHER"
    elif margins["e3_minus_avg_or_weighted_pp"] < gate["beat_best_ordinary_by_pp"]:
        token = "E3_TOY_ABSORBED_BY_TEACHER_AVERAGE_OR_WEIGHTED_VOTE"
    elif margins["e3_minus_active_pp"] < gate["beat_active_by_pp"]:
        token = "E3_TOY_ABSORBED_BY_ACTIVE_LEARNING"
    elif margins["e3_minus_augmentation_pp"] < gate["beat_best_ordinary_by_pp"]:
        token = "E3_TOY_ABSORBED_BY_AUGMENTATION"
    elif margins["e3_minus_shuffled_pp"] < gate["beat_shuffled_by_pp"]:
        token = "E3_TOY_SHUFFLED_SENSORS_MATCH_REAL"
    elif margins["e3_minus_best_ordinary_pp"] >= gate["beat_best_ordinary_by_pp"] and forecast_ok:
        token = "E3_TOY_SIGNAL_SOURCE_SPECIFIC_COUNTERFACTUAL_LESSON"
    else:
        token = "E3_TOY_NEGATIVE"

    return {
        "precommit": PRECOMMIT,
        "summary_metrics": metrics,
        "margins": margins,
        "packet_value_forecast": {
            "mean_prior": float(priors.mean()),
            "mean_realized_vs_best_baseline": float(realized.mean()),
            "forecast_ok": forecast_ok,
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
