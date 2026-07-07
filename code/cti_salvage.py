"""CTI salvage analysis for existing Eklavya/B14 mechanism-control data.

Reads the B14 JSON artifact, computes rough local fine-tuning FLOPs, fits
D_proxy(C) = D_inf + k*C^(-alpha) on training loss, and writes CTI-0 tables and
plots. This script does not load models or datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kendalltau, spearmanr


CONDITION_ORDER = OrderedDict(
    [
        ("label_only", "label_only"),
        ("single_teacher_smol360", "single_teacher"),
        ("oracle_route_ceiling", "oracle"),
        ("non_oracle_confidence", "non_oracle"),
        ("random_route_control", "random"),
    ]
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def condition_items(payload: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    runs = payload["runs"]
    out: list[tuple[str, str, dict[str, Any]]] = []
    for run_name, display_name in CONDITION_ORDER.items():
        if run_name in runs:
            out.append((run_name, display_name, runs[run_name]))
    for run_name, run in runs.items():
        if run_name not in CONDITION_ORDER:
            out.append((run_name, run_name, run))
    return out


def cumulative_flops(trainable_params: int, batch_examples: int, step: int) -> int:
    return int(6 * trainable_params * batch_examples * step)


def build_curve_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, condition, run in condition_items(payload):
        training = run["training"]
        trainable_params = int(training["trainable_parameters"])
        batch_examples = int(training["batch_examples"])
        for point in training["history"]:
            step = int(point["step"])
            flops = cumulative_flops(trainable_params, batch_examples, step)
            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition,
                    "step": step,
                    "cumulative_flops": flops,
                    "cumulative_gflops": flops / 1e9,
                    "d_proxy_loss": float(point["loss"]),
                    "ce": finite_float(point.get("ce")),
                    "kl": finite_float(point.get("kl")),
                    "batch_accuracy": finite_float(point.get("batch_accuracy")),
                    "grad_norm": finite_float(point.get("grad_norm")),
                    "trainable_parameters": trainable_params,
                    "batch_examples": batch_examples,
                }
            )
    return rows


def build_final_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    benchmarks = list(payload["run"]["benchmarks"])
    for run_name, condition, run in condition_items(payload):
        training = run["training"]
        history = training["history"]
        final_step = int(history[-1]["step"])
        final_flops = cumulative_flops(
            int(training["trainable_parameters"]),
            int(training["batch_examples"]),
            final_step,
        )
        accuracies = {bench: float(run["benchmarks"][bench]["accuracy"]) for bench in benchmarks}
        margins = {
            bench: finite_float(run["benchmarks"][bench].get("mean_margin_best_wrong_minus_gold_nll"))
            for bench in benchmarks
        }
        mean_accuracy = float(np.mean(list(accuracies.values())))
        margin_values = [v for v in margins.values() if v is not None]
        row: dict[str, Any] = {
            "run_name": run_name,
            "condition": condition,
            "final_step": final_step,
            "final_cumulative_flops": final_flops,
            "final_cumulative_gflops": final_flops / 1e9,
            "mean_accuracy": mean_accuracy,
            "d_func": 1.0 - mean_accuracy,
            "mean_margin_best_wrong_minus_gold_nll": float(np.mean(margin_values)) if margin_values else None,
            "final_d_proxy_loss": float(history[-1]["loss"]),
            "final_batch_accuracy": finite_float(history[-1].get("batch_accuracy")),
        }
        for bench, value in accuracies.items():
            row[f"accuracy_{bench}"] = value
        for bench, value in margins.items():
            row[f"margin_{bench}"] = value
        rows.append(row)
    return rows


def power_model_normalized(x_norm: np.ndarray, d_inf: float, k_norm: float, alpha: float) -> np.ndarray:
    return d_inf + k_norm * np.power(x_norm, -alpha)


def fit_power_law(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = np.array([float(row["cumulative_flops"]) for row in rows], dtype=float)
    y = np.array([float(row["d_proxy_loss"]) for row in rows], dtype=float)
    c_max = float(np.max(x))
    x_norm = x / c_max
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    bounds = ([0.0, 0.0, 1e-6], [max(5.0, y_max * 2.0), max(20.0, y_max * 20.0), 5.0])
    starts = []
    for d_frac in (0.2, 0.5, 0.8, 1.0):
        d0 = max(0.0, y_min * d_frac)
        for k_frac in (0.25, 0.5, 1.0, 2.0):
            k0 = max(1e-6, (y_max - d0) * k_frac)
            for alpha0 in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
                starts.append([d0, k0, alpha0])
    best: tuple[float, np.ndarray, np.ndarray | None] | None = None
    last_error = None
    for p0 in starts:
        try:
            popt, pcov = curve_fit(
                power_model_normalized,
                x_norm,
                y,
                p0=p0,
                bounds=bounds,
                maxfev=100000,
            )
        except Exception as exc:
            last_error = str(exc)
            continue
        pred = power_model_normalized(x_norm, *popt)
        sse = float(np.sum(np.square(y - pred)))
        if best is None or sse < best[0]:
            best = (sse, popt, pcov)
    if best is None:
        return {"fit_valid": False, "fit_error": last_error or "curve_fit failed", "n_points": int(len(y))}
    sse, popt, pcov = best
    d_inf, k_norm, alpha = [float(v) for v in popt]
    pred = power_model_normalized(x_norm, d_inf, k_norm, alpha)
    residuals = y - pred
    sst = float(np.sum(np.square(y - np.mean(y))))
    perr: list[float | None] = [None, None, None]
    if pcov is not None and np.all(np.isfinite(pcov)):
        diag = np.diag(pcov)
        if np.all(diag >= 0):
            perr = [float(v) for v in np.sqrt(diag)]
    return {
        "fit_valid": True,
        "n_points": int(len(y)),
        "d_inf": d_inf,
        "k_normalized_to_final_compute": k_norm,
        "k_flops": float(k_norm * (c_max**alpha)),
        "alpha": alpha,
        "d_inf_stderr": perr[0],
        "k_normalized_stderr": perr[1],
        "alpha_stderr": perr[2],
        "compute_normalizer_flops": c_max,
        "sse": sse,
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": None if sst <= 0 else 1.0 - (sse / sst),
        "alpha_boundary_hit": bool(alpha <= 1.01e-6 or alpha >= 4.999),
    }


def build_fit_rows(curve_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = OrderedDict()
    names: dict[str, str] = {}
    for row in curve_rows:
        grouped.setdefault(row["run_name"], []).append(row)
        names[row["run_name"]] = row["condition"]
    return [{"run_name": name, "condition": names[name], **fit_power_law(rows)} for name, rows in grouped.items()]


def ranking(values: dict[str, float]) -> list[str]:
    return [name for name, _ in sorted(values.items(), key=lambda item: (item[1], item[0]))]


def values_at_step(curve_rows: list[dict[str, Any]], step: int) -> dict[str, float]:
    return {row["condition"]: float(row["d_proxy_loss"]) for row in curve_rows if int(row["step"]) == step}


def pairwise_order_accuracy(proxy_values: dict[str, float], func_values: dict[str, float]) -> dict[str, Any]:
    keys = [k for k in proxy_values if k in func_values]
    total = 0
    correct = 0
    ties = 0
    disagreements = []
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            proxy_delta = proxy_values[left] - proxy_values[right]
            func_delta = func_values[left] - func_values[right]
            if proxy_delta == 0 or func_delta == 0:
                ties += 1
                continue
            total += 1
            agrees = (proxy_delta > 0) == (func_delta > 0)
            correct += int(agrees)
            if not agrees:
                disagreements.append(
                    {
                        "left": left,
                        "right": right,
                        "proxy_left": proxy_values[left],
                        "proxy_right": proxy_values[right],
                        "d_func_left": func_values[left],
                        "d_func_right": func_values[right],
                    }
                )
    return {
        "correct_pairs": correct,
        "total_pairs": total,
        "accuracy_excluding_ties": None if total == 0 else correct / total,
        "ties_skipped": ties,
        "disagreements": disagreements,
    }


def ranking_diagnostic(curve_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    early_proxy = values_at_step(curve_rows, step)
    final_proxy = {row["condition"]: float(row["final_d_proxy_loss"]) for row in final_rows}
    d_func = {row["condition"]: float(row["d_func"]) for row in final_rows}
    common = [name for name in early_proxy if name in d_func]
    early_vec = np.array([early_proxy[name] for name in common], dtype=float)
    final_proxy_vec = np.array([final_proxy[name] for name in common], dtype=float)
    func_vec = np.array([d_func[name] for name in common], dtype=float)
    early_s = spearmanr(early_vec, func_vec)
    early_k = kendalltau(early_vec, func_vec)
    final_s = spearmanr(final_proxy_vec, func_vec)
    final_k = kendalltau(final_proxy_vec, func_vec)
    early_rank = ranking(early_proxy)
    final_proxy_rank = ranking(final_proxy)
    func_rank = ranking(d_func)
    return {
        "early_step": step,
        "proxy_at_step": early_proxy,
        "final_proxy": final_proxy,
        "d_func": d_func,
        "early_proxy_ranking_low_to_high": early_rank,
        "final_proxy_ranking_low_to_high": final_proxy_rank,
        "d_func_ranking_low_to_high": func_rank,
        "early_top1_matches_final_functional_top1": early_rank[0] == func_rank[0],
        "final_proxy_top1_matches_final_functional_top1": final_proxy_rank[0] == func_rank[0],
        "early_proxy_spearman_rho": finite_float(early_s.statistic),
        "early_proxy_spearman_p": finite_float(early_s.pvalue),
        "early_proxy_kendall_tau": finite_float(early_k.statistic),
        "early_proxy_kendall_p": finite_float(early_k.pvalue),
        "final_proxy_spearman_rho": finite_float(final_s.statistic),
        "final_proxy_spearman_p": finite_float(final_s.pvalue),
        "final_proxy_kendall_tau": finite_float(final_k.statistic),
        "final_proxy_kendall_p": finite_float(final_k.pvalue),
        "early_proxy_pairwise": pairwise_order_accuracy(early_proxy, d_func),
        "final_proxy_pairwise": pairwise_order_accuracy(final_proxy, d_func),
    }


def zero_shot_summary(payload: dict[str, Any]) -> dict[str, Any]:
    benches = list(payload["run"]["benchmarks"])
    accuracies = {bench: float(payload["zero_shot_heldout"][bench]["accuracy"]) for bench in benches}
    margins = {
        bench: finite_float(payload["zero_shot_heldout"][bench].get("mean_margin_best_wrong_minus_gold_nll"))
        for bench in benches
    }
    margin_values = [v for v in margins.values() if v is not None]
    mean_acc = float(np.mean(list(accuracies.values())))
    return {
        "mean_accuracy": mean_acc,
        "d_func": 1.0 - mean_acc,
        "accuracy_by_benchmark": accuracies,
        "mean_margin_best_wrong_minus_gold_nll": float(np.mean(margin_values)) if margin_values else None,
        "margin_by_benchmark": margins,
    }


def infer_verdict(fit_rows: list[dict[str, Any]], diag: dict[str, Any]) -> str:
    valid = [row for row in fit_rows if row.get("fit_valid")]
    if len(valid) < 3:
        return "CTI_SALVAGE_INVALID"
    alphas = [float(row["alpha"]) for row in valid]
    diverged = (not diag["early_top1_matches_final_functional_top1"]) or (
        not diag["final_proxy_top1_matches_final_functional_top1"]
    )
    if max(alphas) - min(alphas) >= 0.05 and diverged:
        return "CTI_SALVAGE_INFORMATIVE"
    return "CTI_SALVAGE_FLAT"


def plot_proxy_fits(path: Path, curve_rows: list[dict[str, Any]], fit_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_condition: dict[str, list[dict[str, Any]]] = OrderedDict()
    for row in curve_rows:
        by_condition.setdefault(row["condition"], []).append(row)
    fits = {row["condition"]: row for row in fit_rows}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for condition, rows in by_condition.items():
        xs = np.array([float(row["cumulative_flops"]) for row in rows], dtype=float)
        ys = np.array([float(row["d_proxy_loss"]) for row in rows], dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{condition} observed")
        fit = fits.get(condition, {})
        if fit.get("fit_valid"):
            x_grid = np.geomspace(float(np.min(xs)), float(np.max(xs)), 200)
            y_grid = power_model_normalized(
                x_grid / float(fit["compute_normalizer_flops"]),
                float(fit["d_inf"]),
                float(fit["k_normalized_to_final_compute"]),
                float(fit["alpha"]),
            )
            ax.plot(x_grid, y_grid, linestyle="--", linewidth=1.0, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative local fine-tuning FLOPs")
    ax.set_ylabel("D_proxy = training loss")
    ax.set_title("CTI-0 proxy compute-distortion fits from B14")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_functional(path: Path, final_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["condition"] for row in final_rows]
    values = [float(row["d_func"]) for row in final_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(labels, values, s=80)
    ax.plot(labels, values, linewidth=1.0, alpha=0.45)
    ax.set_ylabel("D_func = 1 - mean held-out accuracy")
    ax.set_title("CTI-0 final functional distortion by B14 condition")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_proxy_vs_function(path: Path, diag: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proxy = diag["proxy_at_step"]
    d_func = diag["d_func"]
    labels = [name for name in proxy if name in d_func]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter([proxy[name] for name in labels], [d_func[name] for name in labels], s=80)
    for name in labels:
        ax.annotate(name, (proxy[name], d_func[name]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"D_proxy at step {diag['early_step']}")
    ax.set_ylabel("Final D_func")
    ax.set_title("Early proxy loss vs final functional distortion")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(input_path: Path, output_dir: Path, ranking_step: int, make_plots: bool) -> dict[str, Any]:
    payload = read_json(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = build_curve_rows(payload)
    final_rows = build_final_rows(payload)
    fit_rows = build_fit_rows(curve_rows)
    diag = ranking_diagnostic(curve_rows, final_rows, ranking_step)
    verdict = infer_verdict(fit_rows, diag)
    curve_csv = output_dir / "cti0_b14_compute_proxy_curves.csv"
    final_csv = output_dir / "cti0_b14_final_functional_distortion.csv"
    fit_csv = output_dir / "cti0_b14_proxy_powerlaw_fits.csv"
    summary_json = output_dir / "cti0_salvage_summary.json"
    write_csv(curve_csv, curve_rows)
    write_csv(final_csv, final_rows)
    write_csv(fit_csv, fit_rows)
    artifacts = {
        "curve_csv": str(curve_csv),
        "final_functional_csv": str(final_csv),
        "fit_csv": str(fit_csv),
        "summary_json": str(summary_json),
    }
    if make_plots:
        proxy_plot = output_dir / "cti0_proxy_loss_powerlaw_fits.png"
        func_plot = output_dir / "cti0_functional_distortion_by_condition.png"
        proxy_func_plot = output_dir / "cti0_step30_proxy_vs_final_dfunc.png"
        plot_proxy_fits(proxy_plot, curve_rows, fit_rows)
        plot_functional(func_plot, final_rows)
        plot_proxy_vs_function(proxy_func_plot, diag)
        artifacts.update(
            {
                "proxy_fit_plot": str(proxy_plot),
                "functional_distortion_plot": str(func_plot),
                "step_proxy_vs_final_dfunc_plot": str(proxy_func_plot),
            }
        )
    alphas = [float(row["alpha"]) for row in fit_rows if row.get("fit_valid")]
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "analysis": "CTI-0 Eklavya B14 salvage measurement",
        "input_path": str(input_path),
        "verdict": verdict,
        "compute_accounting": {
            "formula": "C = 6 * trainable_parameters * batch_examples * step",
            "unit": "local fine-tuning FLOPs",
            "pretraining_compute_included": False,
            "model_loading": "none; artifact-only analysis",
        },
        "b14_run": payload.get("run", {}),
        "zero_shot_heldout": zero_shot_summary(payload),
        "conditions": final_rows,
        "proxy_powerlaw_fits": fit_rows,
        "alpha_summary": {
            "min_alpha": min(alphas) if alphas else None,
            "max_alpha": max(alphas) if alphas else None,
            "range_alpha": (max(alphas) - min(alphas)) if alphas else None,
        },
        "ranking_diagnostic": diag,
        "limitations": [
            "B14 has final held-out functional evaluation only, so D_func(C) cannot be fit as a curve.",
            "D_proxy is logged batch training loss at sparse intervals, not a held-out proxy.",
            "Every condition has the same 150-step local LoRA budget; intervention comparisons are at one final compute point.",
            "The fit uses observed proxy loss only and is diagnostic, not a CTI law validation.",
            "Held-out MCQ slices are n=48 per benchmark and train-safe, not public validation splits.",
        ],
        "artifacts": artifacts,
    }
    write_json(summary_json, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CTI-0 compute-distortion salvage data from B14 JSON.")
    parser.add_argument("--input", type=Path, default=Path("tmp_work_loop_b14/smollm2_mechanism_control.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("tmp_work_loop_b15"))
    parser.add_argument("--ranking-step", type=int, default=30)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze(args.input, args.output_dir, args.ranking_step, not args.no_plots)
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "artifacts": summary["artifacts"],
                "alpha_summary": summary["alpha_summary"],
                "early_proxy_ranking": summary["ranking_diagnostic"]["early_proxy_ranking_low_to_high"],
                "functional_ranking": summary["ranking_diagnostic"]["d_func_ranking_low_to_high"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
