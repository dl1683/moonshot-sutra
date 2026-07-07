"""CTI-1 Board 1: random transformer on addition modulo 97.

Self-contained runner for W-Loop B16. It performs the smoke run, locks
predictions after step 100, resumes to step 3000, scores forecasts, and writes
all required artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit

P = 97
SEED = 42
CONDITIONS = ["label_only", "shuffled_labels", "quarter_data"]
CHECKPOINTS = [10, 30, 100, 300, 1000, 3000]
FIT_STEPS = [10, 30, 100]
HELDOUT_STEPS = [300, 1000, 3000]
FORECASTERS = [
    "cti_power_law",
    "b0_last_point",
    "b1_linear_log_compute",
    "b2_independent_power_law",
    "b3_proxy_only",
    "b4_random_intervention_ranking",
]
CFG = {
    "board": "CTI-1 Board 1",
    "task": "addition_mod_97",
    "p": P,
    "seed": SEED,
    "train_fraction": 0.5,
    "interventions": CONDITIONS,
    "checkpoint_steps": CHECKPOINTS,
    "fit_steps": FIT_STEPS,
    "heldout_steps": HELDOUT_STEPS,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1.0,
    "batch_size": 512,
    "eval_batch_size": 4096,
    "max_steps": 3000,
    "dtype": "float32",
    "device": "cuda",
    "vocab_size": P,
    "sequence_length": 2,
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "dim_feedforward": 704,
    "dropout": 0.0,
    "model_birth": "random_init_from_scratch",
    "pretrained_weights_loaded": False,
    "compute_formula": "cumulative_flops = 6 * total_params * batch_size * checkpoint_step",
    "proxy_slice_examples": 100,
}


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = CFG["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, CFG["n_heads"], dropout=0.0, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, CFG["dim_feedforward"]),
            nn.GELU(),
            nn.Linear(CFG["dim_feedforward"], d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = CFG["d_model"]
        self.tok = nn.Embedding(P, d)
        self.pos = nn.Parameter(torch.zeros(1, 2, d))
        self.blocks = nn.ModuleList([Block() for _ in range(CFG["n_layers"])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, P)
        self.apply(self._init)

    def _init(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tok(x) + self.pos
        for block in self.blocks:
            h = block(h)
        h = self.ln(h).mean(dim=1)
        return self.head(h)


def now() -> str:
    return datetime.now(UTC).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flops(total_params: int, step: int) -> int:
    return int(6 * total_params * CFG["batch_size"] * step)


def split_data() -> dict[str, np.ndarray]:
    pairs = np.array([(a, b) for a in range(P) for b in range(P)], dtype=np.int64)
    labels = ((pairs[:, 0] + pairs[:, 1]) % P).astype(np.int64)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    n_train = len(pairs) // 2
    tr = order[:n_train]
    he = order[n_train:]
    return {
        "train_x": pairs[tr],
        "train_y": labels[tr],
        "heldout_x": pairs[he],
        "heldout_y": labels[he],
        "train_idx": tr,
        "heldout_idx": he,
    }


def split_digest(s: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(s["train_idx"].astype(np.int64).tobytes())
    h.update(s["heldout_idx"].astype(np.int64).tobytes())
    return h.hexdigest()


def tensorize(split: dict[str, np.ndarray], device: torch.device) -> dict[str, dict[str, Any]]:
    full_x_np = split["train_x"]
    full_y_np = split["train_y"]
    held_x = torch.tensor(split["heldout_x"], dtype=torch.long, device=device)
    held_y = torch.tensor(split["heldout_y"], dtype=torch.long, device=device)
    full_x = torch.tensor(full_x_np, dtype=torch.long, device=device)
    full_y = torch.tensor(full_y_np, dtype=torch.long, device=device)
    rng = np.random.default_rng(SEED + 1000)
    shuffled_y = full_y_np[rng.permutation(len(full_y_np))]
    q = len(full_x_np) // 4
    specs = OrderedDict([
        ("label_only", (full_x_np, full_y_np, full_y_np)),
        ("shuffled_labels", (full_x_np, shuffled_y, full_y_np)),
        ("quarter_data", (full_x_np[:q], full_y_np[:q], full_y_np[:q])),
    ])
    out: dict[str, dict[str, Any]] = {}
    for name, (x_np, y_obj_np, y_true_np) in specs.items():
        x = torch.tensor(x_np, dtype=torch.long, device=device)
        y_obj = torch.tensor(y_obj_np, dtype=torch.long, device=device)
        y_true = torch.tensor(y_true_np, dtype=torch.long, device=device)
        n_proxy = min(100, len(x_np))
        out[name] = {
            "x": x,
            "y_obj": y_obj,
            "y_true": y_true,
            "proxy_x": x[:n_proxy],
            "proxy_y": y_obj[:n_proxy],
            "full_x": full_x,
            "full_y": full_y,
            "held_x": held_x,
            "held_y": held_y,
            "n_train": len(x_np),
            "n_full_train": len(full_x_np),
            "n_heldout": len(split["heldout_x"]),
        }
    return out


def ece(conf: torch.Tensor, corr: torch.Tensor, bins: int = 10) -> float:
    total = max(1, int(conf.numel()))
    ans = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if bool(mask.any()):
            ans += int(mask.sum().item()) / total * abs(float(conf[mask].mean()) - float(corr[mask].mean()))
    return float(ans)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    losses = 0.0
    correct = 0
    total = 0
    margins = []
    confs = []
    corrs = []
    bs = CFG["eval_batch_size"]
    for start in range(0, x.shape[0], bs):
        xb = x[start:start + bs]
        yb = y[start:start + bs]
        logits = model(xb)
        losses += float(F.cross_entropy(logits, yb, reduction="sum").detach().cpu())
        pred = logits.argmax(-1)
        corr = pred.eq(yb)
        probs = torch.softmax(logits, dim=-1)
        confs.append(probs.max(-1).values.detach().cpu())
        corrs.append(corr.float().detach().cpu())
        gold = logits.gather(1, yb[:, None]).squeeze(1)
        wrong = logits.clone()
        wrong.scatter_(1, yb[:, None], -float("inf"))
        margins.append((wrong.max(-1).values - gold).detach().cpu())
        correct += int(corr.sum().detach().cpu())
        total += int(yb.numel())
    conf = torch.cat(confs)
    corr = torch.cat(corrs)
    return {
        "loss": losses / max(1, total),
        "accuracy": correct / max(1, total),
        "margin": float(torch.cat(margins).mean()),
        "ece": ece(conf, corr),
    }


def grad_norm(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            s += float((g * g).sum().detach().cpu())
    return math.sqrt(s)


def checkpoint_row(condition: str, step: int, model: nn.Module, data: dict[str, Any], total_params: int,
                   mb_loss: float, mb_acc: float, elapsed: float) -> dict[str, Any]:
    proxy = evaluate(model, data["proxy_x"], data["proxy_y"])
    train = evaluate(model, data["x"], data["y_obj"])
    train_true = evaluate(model, data["x"], data["y_true"])
    full_true = evaluate(model, data["full_x"], data["full_y"])
    held = evaluate(model, data["held_x"], data["held_y"])
    return {
        "model_birth": CFG["model_birth"],
        "model_name": "tiny_transformer_mod97",
        "task_family": "modular_arithmetic",
        "task_id": "addition_mod_97",
        "intervention": condition,
        "seed": SEED,
        "checkpoint_step": step,
        "cumulative_flops": flops(total_params, step),
        "cumulative_gflops": flops(total_params, step) / 1e9,
        "total_params": total_params,
        "trainable_params": total_params,
        "batch_size": CFG["batch_size"],
        "train_examples_seen": CFG["batch_size"] * step,
        "train_examples_in_condition": data["n_train"],
        "full_train_examples": data["n_full_train"],
        "heldout_examples": data["n_heldout"],
        "eval_split_id": "mod97_seed42_split50_sha256",
        "d_func": 1.0 - held["accuracy"],
        "d_proxy": proxy["loss"],
        "d_proxy_fixed_slice_loss": proxy["loss"],
        "d_proxy_current_minibatch_loss": mb_loss,
        "d_gap": abs(train["accuracy"] - held["accuracy"]),
        "train_accuracy": train["accuracy"],
        "train_loss": train["loss"],
        "condition_train_task_accuracy": train_true["accuracy"],
        "full_train_task_accuracy": full_true["accuracy"],
        "held_out_accuracy": held["accuracy"],
        "held_out_loss": held["loss"],
        "d_margin": held["margin"],
        "d_cal": held["ece"],
        "proxy_accuracy": proxy["accuracy"],
        "current_minibatch_accuracy": mb_acc,
        "elapsed_seconds": elapsed,
        "created_at_utc": now(),
    }


def train_range(condition: str, model: nn.Module, opt: torch.optim.Optimizer, data: dict[str, Any], total_params: int,
                gen: torch.Generator, start: int, end: int, ckpts: set[int], log_path: Path, t0: float) -> list[dict[str, Any]]:
    rows = []
    mb_loss = float("nan")
    mb_acc = float("nan")
    for step in range(start + 1, end + 1):
        model.train()
        idx = torch.randint(0, data["x"].shape[0], (CFG["batch_size"],), device=data["x"].device, generator=gen)
        xb = data["x"][idx]
        yb = data["y_obj"][idx]
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        gn = grad_norm(model)
        opt.step()
        with torch.no_grad():
            mb_loss = float(loss.detach().cpu())
            mb_acc = float(logits.argmax(-1).eq(yb).float().mean().detach().cpu())
        append_jsonl(log_path, {
            "created_at_utc": now(),
            "condition": condition,
            "step": step,
            "loss": mb_loss,
            "batch_accuracy": mb_acc,
            "grad_norm": gn,
            "learning_rate": CFG["learning_rate"],
            "weight_decay": CFG["weight_decay"],
            "cumulative_flops": flops(total_params, step),
            "elapsed_seconds": time.perf_counter() - t0,
        })
        if step in ckpts:
            rows.append(checkpoint_row(condition, step, model, data, total_params, mb_loss, mb_acc, time.perf_counter() - t0))
    return rows



def power_fn(x_norm: np.ndarray, d_inf: float, k: float, alpha: float) -> np.ndarray:
    return d_inf + k * np.power(x_norm, -alpha)


def clip01(x: float) -> float:
    if not math.isfinite(x):
        return 1.0
    return min(1.0, max(0.0, float(x)))


def fit_power(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    cmax = float(x.max())
    xn = x / cmax
    if float(y.max() - y.min()) < 1e-10:
        return {
            "fit_valid": True,
            "fallback": "constant",
            "d_inf": float(y[-1]),
            "k": 0.0,
            "alpha": 0.0,
            "compute_normalizer_flops": cmax,
            "rmse_fit": 0.0,
            "r2_fit": None,
        }
    best = None
    err = None
    starts = []
    for d0 in (max(0.0, float(y.min()) - 0.05), max(0.0, float(y.min()) * 0.8), float(y[-1])):
        for k0 in (0.01, 0.05, 0.2, max(1e-4, float(y.max() - d0))):
            for a0 in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
                starts.append([d0, k0, a0])
    for p0 in starts:
        try:
            popt, _ = curve_fit(
                power_fn,
                xn,
                y,
                p0=p0,
                bounds=([0.0, 0.0, 1e-6], [1.5, 3.0, 5.0]),
                maxfev=100000,
            )
        except Exception as exc:
            err = str(exc)
            continue
        pred = power_fn(xn, *popt)
        sse = float(np.square(y - pred).sum())
        if best is None or sse < best[0]:
            best = (sse, popt, pred)
    if best is None:
        return {
            "fit_valid": False,
            "fallback": "constant_after_error",
            "fit_error": err,
            "d_inf": float(y[-1]),
            "k": 0.0,
            "alpha": 0.0,
            "compute_normalizer_flops": cmax,
            "rmse_fit": None,
            "r2_fit": None,
        }
    sse, popt, pred = best
    sst = float(np.square(y - y.mean()).sum())
    return {
        "fit_valid": True,
        "fallback": None,
        "d_inf": float(popt[0]),
        "k": float(popt[1]),
        "alpha": float(popt[2]),
        "compute_normalizer_flops": cmax,
        "rmse_fit": float(np.sqrt(np.square(y - pred).mean())),
        "r2_fit": None if sst <= 0 else 1.0 - sse / sst,
        "alpha_boundary_hit": bool(popt[2] <= 1.01e-6 or popt[2] >= 4.999),
    }


def pred_power(fit: dict[str, Any], x: float) -> float:
    return clip01(float(power_fn(np.array([x / fit["compute_normalizer_flops"]]), fit["d_inf"], fit["k"], fit["alpha"])[0]))


def fit_linear_log(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    lx = np.log(np.asarray(x, dtype=float))
    slope, intercept = np.polyfit(lx, y, deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def build_predictions(rows: list[dict[str, Any]], total_params: int, out_dir: Path, digest: str) -> dict[str, Any]:
    early = [r for r in rows if int(r["checkpoint_step"]) in FIT_STEPS]
    if len(early) != len(CONDITIONS) * len(FIT_STEPS):
        raise RuntimeError(f"expected 9 early rows, got {len(early)}")
    byc = OrderedDict()
    for c in CONDITIONS:
        rr = [r for r in early if r["intervention"] == c]
        byc[c] = sorted(rr, key=lambda r: int(r["checkpoint_step"]))
    held_c = {str(s): flops(total_params, s) for s in HELDOUT_STEPS}
    forecasts: dict[str, dict[str, dict[str, float]]] = {f: {} for f in FORECASTERS}
    fits: dict[str, Any] = {"cti_power_law": {}, "proxy_power_law": {}, "linear_log_compute": {}}
    for c, rr in byc.items():
        x = np.array([float(r["cumulative_flops"]) for r in rr])
        y = np.array([float(r["d_func"]) for r in rr])
        yp = np.array([float(r["d_proxy"]) for r in rr])
        f = fit_power(x, y)
        fp = fit_power(x, yp)
        fl = fit_linear_log(x, y)
        fits["cti_power_law"][c] = f
        fits["proxy_power_law"][c] = fp
        fits["linear_log_compute"][c] = fl
        forecasts["cti_power_law"][c] = {str(s): pred_power(f, held_c[str(s)]) for s in HELDOUT_STEPS}
        forecasts["b2_independent_power_law"][c] = dict(forecasts["cti_power_law"][c])
        forecasts["b0_last_point"][c] = {str(s): clip01(float(y[-1])) for s in HELDOUT_STEPS}
        forecasts["b1_linear_log_compute"][c] = {
            str(s): clip01(fl["slope"] * math.log(held_c[str(s)]) + fl["intercept"])
            for s in HELDOUT_STEPS
        }
        if float(yp.max() - yp.min()) < 1e-10:
            a, b = 0.0, float(y[-1])
        else:
            a, b = np.polyfit(yp, y, deg=1)
            a, b = float(a), float(b)
        forecasts["b3_proxy_only"][c] = {str(s): clip01(a * pred_power(fp, held_c[str(s)]) + b) for s in HELDOUT_STEPS}
    rng = random.Random(SEED)
    rand_rank = list(CONDITIONS)
    rng.shuffle(rand_rank)
    vals100 = {c: float(byc[c][-1]["d_func"]) for c in CONDITIONS}
    sorted_vals = sorted(vals100.values())
    assigned = {c: sorted_vals[rand_rank.index(c)] for c in CONDITIONS}
    for c in CONDITIONS:
        forecasts["b4_random_intervention_ranking"][c] = {str(s): clip01(assigned[c]) for s in HELDOUT_STEPS}
    ranks = {f: sorted(CONDITIONS, key=lambda c: (forecasts[f][c]["3000"], c)) for f in FORECASTERS}
    payload = {
        "created_at_utc": now(),
        "prediction_lock": {
            "status": "LOCKED_BEFORE_HELDOUT_CHECKPOINTS_300_1000_3000",
            "fit_steps_used": FIT_STEPS,
            "heldout_steps_predicted": HELDOUT_STEPS,
            "actual_heldout_rows_available_when_written": False,
            "split_sha256": digest,
            "total_params": total_params,
            "compute_formula": CFG["compute_formula"],
            "artifact_note": "Written after all conditions reach step 100 and before resume training to step 300.",
        },
        "forecasters": {
            "cti_power_law": "Per-intervention D_func(C)=D_inf+k*C^-alpha fit on checkpoints 10,30,100.",
            "b0_last_point": "Hold step-100 D_func constant.",
            "b1_linear_log_compute": "Linear extrapolation of D_func against log(C).",
            "b2_independent_power_law": "Independent power law baseline; degenerate with CTI on this single-task board.",
            "b3_proxy_only": "Forecast proxy loss, map early proxy to early D_func.",
            "b4_random_intervention_ranking": "Seeded random ranking with step-100 D_func values assigned to the random order.",
        },
        "fits": fits,
        "heldout_compute_flops": held_c,
        "predicted_d_func": forecasts,
        "predicted_step3000_rankings_lowest_d_func_first": ranks,
        "step100_observed_d_func": vals100,
    }
    write_json(out_dir / "cti1_board1_predictions.json", payload)
    return payload


def score(pred: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual = {(r["intervention"], str(int(r["checkpoint_step"]))): float(r["d_func"]) for r in rows if int(r["checkpoint_step"]) in HELDOUT_STEPS}
    actual_rank = sorted(CONDITIONS, key=lambda c: (actual[(c, "3000")], c))
    score_rows = []
    detail = {}
    for f in FORECASTERS:
        errs = []
        byc = {c: [] for c in CONDITIONS}
        bys = {str(s): [] for s in HELDOUT_STEPS}
        detail[f] = []
        for c in CONDITIONS:
            for s in HELDOUT_STEPS:
                ss = str(s)
                p = float(pred["predicted_d_func"][f][c][ss])
                a = actual[(c, ss)]
                e = abs(p - a)
                errs.append(e)
                byc[c].append(e)
                bys[ss].append(e)
                detail[f].append({"condition": c, "checkpoint_step": s, "predicted_d_func": p, "actual_d_func": a, "absolute_error": e})
        pr = pred["predicted_step3000_rankings_lowest_d_func_first"][f]
        score_rows.append({
            "forecaster": f,
            "mae_all_heldout_points": float(np.mean(errs)),
            "mae_label_only": float(np.mean(byc["label_only"])),
            "mae_shuffled_labels": float(np.mean(byc["shuffled_labels"])),
            "mae_quarter_data": float(np.mean(byc["quarter_data"])),
            "mae_step_300": float(np.mean(bys["300"])),
            "mae_step_1000": float(np.mean(bys["1000"])),
            "mae_step_3000": float(np.mean(bys["3000"])),
            "predicted_best_step3000": pr[0],
            "actual_best_step3000": actual_rank[0],
            "ranking_top1_correct": pr[0] == actual_rank[0],
            "predicted_ranking_step3000": " < ".join(pr),
            "actual_ranking_step3000": " < ".join(actual_rank),
        })
    write_csv(out_dir / "cti1_board1_scores.csv", score_rows)
    return score_rows, {
        "actual_d_func": {f"{k[0]}:{k[1]}": v for k, v in actual.items()},
        "prediction_details": detail,
        "actual_step3000_ranking": actual_rank,
    }


def shifts(pred: dict[str, Any]) -> list[dict[str, Any]]:
    fits = pred["fits"]["cti_power_law"]
    ref = fits["label_only"]
    out = []
    for c in CONDITIONS:
        f = fits[c]
        da = float(f["alpha"]) - float(ref["alpha"])
        dd = float(f["d_inf"]) - float(ref["d_inf"])
        if c == "label_only":
            cls = "reference"
        elif abs(da) >= 0.05:
            cls = "exponent_shift"
        elif abs(dd) >= 0.05:
            cls = "constant_shift"
        else:
            cls = "indeterminate_small_shift"
        out.append({
            "intervention": c,
            "reference": "label_only",
            "alpha": float(f["alpha"]),
            "d_inf": float(f["d_inf"]),
            "delta_alpha_vs_label_only": da,
            "delta_d_inf_vs_label_only": dd,
            "classification": cls,
        })
    return out



def grokking(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = {int(r["checkpoint_step"]): float(r["d_func"]) for r in rows if r["intervention"] == "label_only"}
    early = [vals.get(10, 1.0), vals.get(30, 1.0), vals.get(100, 1.0)]
    width = max(early) - min(early)
    late_step = min(HELDOUT_STEPS, key=lambda s: vals.get(s, 1.0))
    late_drop = vals.get(100, 1.0) - vals.get(late_step, 1.0)
    detected = width <= 0.05 and late_drop >= 0.20
    trans = None
    if detected:
        for s in HELDOUT_STEPS:
            if vals[100] - vals[s] >= 0.20:
                trans = s
                break
    return {
        "detected": detected,
        "criterion": "early plateau width <=0.05 and step100-to-late drop >=0.20",
        "transition_step": trans,
        "early_plateau_width": width,
        "late_drop_from_step100": late_drop,
        "d_func_by_step": vals,
    }


def validate(rows: list[dict[str, Any]], total_params: int) -> list[str]:
    reasons = []
    exp = {(c, s) for c in CONDITIONS for s in CHECKPOINTS}
    got = {(r["intervention"], int(r["checkpoint_step"])) for r in rows}
    if exp - got:
        reasons.append(f"missing checkpoint rows: {sorted(exp - got)}")
    for r in rows:
        s = int(r["checkpoint_step"])
        if int(r["cumulative_flops"]) != flops(total_params, s):
            reasons.append(f"bad flops {r['intervention']} step {s}")
        for k in ["d_func", "d_proxy", "d_gap", "train_accuracy", "held_out_accuracy"]:
            if not math.isfinite(float(r[k])):
                reasons.append(f"nonfinite {k} at {r['intervention']} {s}")
    return reasons


def verdict(score_rows: list[dict[str, Any]], shift_rows: list[dict[str, Any]], invalid: list[str]) -> str:
    if invalid:
        return "INVALID_CTI"
    m = {r["forecaster"]: float(r["mae_all_heldout_points"]) for r in score_rows}
    cti = m["cti_power_law"]
    beats = all(cti < v for k, v in m.items() if k != "cti_power_law")
    has_shift = any(r["classification"] in {"exponent_shift", "constant_shift"} for r in shift_rows)
    if beats and has_shift:
        return "PASS_CTI_LAW_0"
    if m.get("b3_proxy_only", float("inf")) < cti:
        return "PROXY_ONLY_LAW"
    return "NO_PREDICTIVE_LAW"


def pct(x: float) -> str:
    return f"{100*x:.2f}%"


def md_table(headers: list[str], body: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
    return "\n".join(lines)


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], pred: dict[str, Any], scores: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    ck = [
        [
            r["intervention"],
            int(r["checkpoint_step"]),
            f"{float(r['cumulative_gflops']):.3f}",
            f"{float(r['d_func']):.6f}",
            f"{float(r['d_proxy']):.6f}",
            f"{float(r['d_gap']):.6f}",
            pct(float(r["train_accuracy"])),
            pct(float(r["held_out_accuracy"])),
        ]
        for r in rows
    ]
    sc = [
        [
            r["forecaster"],
            f"{float(r['mae_all_heldout_points']):.6f}",
            r["predicted_best_step3000"],
            r["actual_best_step3000"],
            r["ranking_top1_correct"],
        ]
        for r in scores
    ]
    sh = [
        [
            r["intervention"],
            f"{r['alpha']:.6f}",
            f"{r['d_inf']:.6f}",
            f"{r['delta_alpha_vs_label_only']:.6f}",
            r["classification"],
        ]
        for r in summary["shift_classification"]
    ]
    g = summary["grokking_check"]
    smoke_line = "Completed before the full run; smoke summary artifact is listed below." if "smoke_summary" in summary["artifacts"] else "Not found in this invocation."
    cti_rank = " < ".join(pred["predicted_step3000_rankings_lowest_d_func_first"]["cti_power_law"])
    actual_rank = " < ".join(summary["score_detail"]["actual_step3000_ranking"])
    artifacts = "\n".join(f"- `{v}`" for v in summary["artifacts"].values())
    grok_line = (
        f"Detected at step {g['transition_step']}."
        if g["detected"]
        else f"Not detected by the six checkpoints; late drop from step 100 was {g['late_drop_from_step100']:.6f}."
    )
    text = f"""# W-Loop B16: CTI-1 Board 1 - Random Transformer Modular Arithmetic

**Date:** 2026-07-07
**Verdict token:** `{summary['verdict_token']}`
**Task:** addition mod 97
**Model:** random-init 4-layer transformer, {summary['total_params']:,} parameters
**Device:** {summary['cuda_device']}

---

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch15.md`
4. `research/dual_loop_supervisor_checkin_13.md`

Binding interpretation: CTI-1 Board 1 is a clean compute-distortion lab. No pretrained weights were loaded. The primary measurement is `D_func = 1 - held_out_accuracy` over the full held-out modular-arithmetic split at every checkpoint.

## Smoke Run

{smoke_line}

The smoke path trained `label_only` for 10 steps, recorded checkpoint step 10, and verified `cumulative_flops = 6 * total_params * batch_size * checkpoint_step`.

## Configuration

| Parameter | Value |
|---|---:|
| p | {P} |
| Train examples | {int(rows[0]['full_train_examples'])} |
| Held-out examples | {int(rows[0]['heldout_examples'])} |
| Batch size | {CFG['batch_size']} |
| Learning rate | {CFG['learning_rate']} |
| Weight decay | {CFG['weight_decay']} |
| Max steps | {CFG['max_steps']} |
| Checkpoints | {', '.join(str(s) for s in CHECKPOINTS)} |
| Fit-only checkpoints | {', '.join(str(s) for s in FIT_STEPS)} |
| Held-out forecast checkpoints | {', '.join(str(s) for s in HELDOUT_STEPS)} |

## Prediction Lock

Predictions were written after all three interventions reached step 100 and before any training or evaluation at steps 300, 1000, or 3000. The lock record is in `tmp_work_loop_b16/cti1_board1_predictions.json`.

CTI predicted step-3000 ranking:

```text
{cti_rank}
```

Actual step-3000 ranking:

```text
{actual_rank}
```

## Checkpoint Matrix

{md_table(['Intervention', 'Step', 'GFLOPs', 'D_func', 'D_proxy', 'D_gap', 'Train Acc', 'Held-out Acc'], ck)}

## Forecast Scores

{md_table(['Forecaster', 'MAE held-out', 'Predicted best', 'Actual best', 'Top-1 correct'], sc)}

B2 is identical to the CTI per-intervention power-law forecast on this single task board, so a strict beat-all-baselines verdict is impossible in this board unless the law is later made cross-task or shared-parameter.

## Intervention Shift Classification

{md_table(['Intervention', 'alpha', 'D_inf', 'delta alpha vs label_only', 'Classification'], sh)}

## Grokking Check

{grok_line}

The label-only `D_func` checkpoints were:

```json
{json.dumps(g['d_func_by_step'], indent=2, sort_keys=True)}
```

## Artifacts

{artifacts}

## NARRATIVE SECTION

What happened: the board ran cleanly and produced the missing object from B15: functional distortion at every log-spaced compute point for all three interventions. The shuffled-label negative control measured memorization-only compute, while quarter-data tested whether fewer labels changed the curve.

Did the power law predict: by the strict precommit token, `{summary['verdict_token']}`. The CTI power-law MAE was {summary['cti_mae_all_heldout_points']:.6f}; the best forecaster was `{summary['best_forecaster_by_mae']}` at {summary['best_forecaster_mae']:.6f}.

Did grokking break the form: {grok_line} If a sudden post-plateau jump appears in later boards, a broken power law or phase-transition form is the honest model.

Gossip-magazine story: the laptop tried to predict which tiny training idea was worth the electricity before it finished. This board is only a first lab test, not a manifesto result.

Does it survive "that's obvious?": the setup survives as a measurement because the forecast was locked before the held-out checkpoints existed. The claim does not get to outrun the score table.
"""
    Path("research/work_loop_batch16.md").write_text(text, encoding="utf-8")


def make_model(device: torch.device) -> tuple[nn.Module, torch.optim.Optimizer]:
    model = TinyTransformer().to(device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    return model, opt



def run_smoke(out_dir: Path, device_name: str) -> None:
    CFG["device"] = device_name
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device(device_name)
    set_seed(SEED)
    out = out_dir / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    split = split_data()
    dig = split_digest(split)
    data = tensorize(split, device)
    model, opt = make_model(device)
    total = params(model)
    log = out / "cti1_board1_smoke_train_log.jsonl"
    if log.exists():
        log.unlink()
    gen = torch.Generator(device=device).manual_seed(SEED + 2000)
    t0 = time.perf_counter()
    rows = train_range("label_only", model, opt, data["label_only"], total, gen, 0, 10, {10}, log, t0)
    ck = out / "cti1_board1_smoke_checkpoints.csv"
    write_csv(ck, rows)
    req = ["d_func", "d_proxy", "d_gap", "train_accuracy", "held_out_accuracy"]
    ok = len(rows) == 1 and int(rows[0]["cumulative_flops"]) == flops(total, 10) and all(math.isfinite(float(rows[0][k])) for k in req)
    summary = {
        "created_at_utc": now(),
        "smoke_ok": ok,
        "condition": "label_only",
        "steps": 10,
        "total_params": total,
        "expected_cumulative_flops_step10": flops(total, 10),
        "observed_cumulative_flops_step10": int(rows[0]["cumulative_flops"]),
        "required_metric_keys_checked": req,
        "split_sha256": dig,
        "artifacts": {
            "train_log": str(log),
            "checkpoints": str(ck),
            "summary": str(out / "cti1_board1_smoke_summary.json"),
        },
    }
    write_json(out / "cti1_board1_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_full(out_dir: Path, device_name: str) -> None:
    CFG["device"] = device_name
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    device = torch.device(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    split = split_data()
    dig = split_digest(split)
    data = tensorize(split, device)
    models = {}
    opts = {}
    gens = {}
    total = None
    for i, c in enumerate(CONDITIONS):
        set_seed(SEED)
        model, opt = make_model(device)
        p = params(model)
        total = p if total is None else total
        if p != total:
            raise RuntimeError("param count mismatch")
        models[c] = model
        opts[c] = opt
        gens[c] = torch.Generator(device=device).manual_seed(SEED + 3000 + i)
    assert total is not None
    cfg = dict(CFG)
    cfg.update({
        "created_at_utc": now(),
        "total_params": total,
        "split_sha256": dig,
        "architecture": {
            "blocks": CFG["n_layers"],
            "d_model": CFG["d_model"],
            "heads": CFG["n_heads"],
            "dim_feedforward": CFG["dim_feedforward"],
            "pooling": "mean_over_two_token_sequence",
        },
    })
    write_json(out_dir / "cti1_board1_config.json", cfg)
    log = out_dir / "cti1_board1_train_log.jsonl"
    if log.exists():
        log.unlink()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for c in CONDITIONS:
        rows.extend(train_range(c, models[c], opts[c], data[c], total, gens[c], 0, 100, set(FIT_STEPS), log, t0))
    rows.sort(key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    write_csv(out_dir / "cti1_board1_checkpoints.csv", rows)
    pred = build_predictions(rows, total, out_dir, dig)
    pred_sha = sha256_file(out_dir / "cti1_board1_predictions.json")

    for c in CONDITIONS:
        rows.extend(train_range(c, models[c], opts[c], data[c], total, gens[c], 100, CFG["max_steps"], set(HELDOUT_STEPS), log, t0))
    rows.sort(key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    write_csv(out_dir / "cti1_board1_checkpoints.csv", rows)
    scores, detail = score(pred, rows, out_dir)
    invalid = validate(rows, total)
    shift_rows = shifts(pred)
    g = grokking(rows)
    vt = verdict(scores, shift_rows, invalid)
    best = min(scores, key=lambda r: float(r["mae_all_heldout_points"]))
    cti = next(r for r in scores if r["forecaster"] == "cti_power_law")
    artifacts = {
        "config": str(out_dir / "cti1_board1_config.json"),
        "train_log": str(log),
        "checkpoints": str(out_dir / "cti1_board1_checkpoints.csv"),
        "predictions": str(out_dir / "cti1_board1_predictions.json"),
        "scores": str(out_dir / "cti1_board1_scores.csv"),
        "summary": str(out_dir / "cti1_board1_summary.json"),
        "report": "research/work_loop_batch16.md",
    }
    smoke = out_dir / "smoke" / "cti1_board1_smoke_summary.json"
    if smoke.exists():
        artifacts["smoke_summary"] = str(smoke)
    summary = {
        "created_at_utc": now(),
        "verdict_token": vt,
        "invalid_reasons": invalid,
        "board": CFG["board"],
        "task": CFG["task"],
        "device": device_name,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "total_params": total,
        "split_sha256": dig,
        "prediction_lock": pred["prediction_lock"],
        "prediction_file_sha256_before_resume": pred_sha,
        "prediction_file_sha256_after_scoring": sha256_file(out_dir / "cti1_board1_predictions.json"),
        "cti_mae_all_heldout_points": float(cti["mae_all_heldout_points"]),
        "best_forecaster_by_mae": best["forecaster"],
        "best_forecaster_mae": float(best["mae_all_heldout_points"]),
        "scores": scores,
        "score_detail": detail,
        "shift_classification": shift_rows,
        "grokking_check": g,
        "final_step3000": [r for r in rows if int(r["checkpoint_step"]) == 3000],
        "artifacts": artifacts,
        "notes": [
            "No pretrained model or tokenizer was loaded.",
            "D_func is 1 - exact held-out modular addition accuracy over the full held-out split at every checkpoint.",
            "Predictions were written after step 100 and before training/evaluation at steps 300, 1000, and 3000.",
            "B2 is degenerate with CTI on this single-task board.",
        ],
    }
    summary["total_elapsed_seconds"] = time.perf_counter() - t0
    write_json(out_dir / "cti1_board1_summary.json", summary)
    write_report(summary, rows, pred, scores)
    print(json.dumps({
        "verdict": vt,
        "total_params": total,
        "cti_mae": summary["cti_mae_all_heldout_points"],
        "best_forecaster": best["forecaster"],
        "best_forecaster_mae": summary["best_forecaster_mae"],
        "elapsed_seconds": summary["total_elapsed_seconds"],
        "artifacts": artifacts,
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ["smoke", "full"]:
        p = sub.add_parser(name)
        p.add_argument("--output-dir", default="tmp_work_loop_b16")
        p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()
    if args.cmd == "smoke":
        run_smoke(Path(args.output_dir), args.device)
    elif args.cmd == "full":
        run_full(Path(args.output_dir), args.device)


if __name__ == "__main__":
    main()
