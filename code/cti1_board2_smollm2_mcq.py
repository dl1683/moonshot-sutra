"""CTI-1 Board 2: SmolLM2-135M LoRA on MCQ monotone adaptation.

Reuses the B14 forced-choice MCQ scoring path, but evaluates D_func at every
log-spaced checkpoint and locks predictions after step 100.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from smollm2_mechanism_control import (  # noqa: E402
    BENCHMARKS,
    SMOL_TEACHER,
    STUDENT,
    PreparedChoiceSet,
    cache_one_teacher,
    flatten_examples,
    load_lm,
    load_tokenizer,
    make_split,
    model_manifest,
    prepare_choice_set,
    score_prepared_flat,
    score_prepared_indices,
    teacher_records_to_targets,
)

SEED = 42
B14_SPLIT_SEED = 20260707
CONDITIONS = ["label_only", "single_teacher", "shuffled_labels"]
CHECKPOINTS = [10, 30, 100, 300, 1000, 3000]
FIT_STEPS = [10, 30, 100]
HELDOUT_STEPS = [300, 1000, 3000]
FORECASTERS = [
    "cti_power_law",
    "b0_last_point",
    "b1_linear_log_compute",
    "b2_per_benchmark_power_law",
    "b3_proxy_only",
    "b4_random_intervention_ranking",
]

CFG: dict[str, Any] = {
    "board": "CTI-1 Board 2",
    "task": "mcq_forced_choice_hellaswag_piqa_arc_easy",
    "model_name": STUDENT,
    "teacher_name": SMOL_TEACHER,
    "seed": SEED,
    "split_seed": B14_SPLIT_SEED,
    "benchmark_split": "train",
    "benchmarks": list(BENCHMARKS),
    "train_per_benchmark": 96,
    "eval_per_benchmark": 48,
    "interventions": CONDITIONS,
    "checkpoint_steps": CHECKPOINTS,
    "fit_steps": FIT_STEPS,
    "heldout_steps": HELDOUT_STEPS,
    "optimizer": "AdamW",
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "batch_examples": 12,
    "eval_batch_choices": 32,
    "max_steps": 3000,
    "dtype": "bfloat16",
    "device": "cuda",
    "max_length": 768,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "kd_alpha_label_ce": 0.5,
    "teacher_temperature": 1.0,
    "kd_temperature": 1.0,
    "grad_clip": 1.0,
    "model_birth": "pretrained_smollm2_135m_lora_local_only",
    "local_files_only_required": True,
    "compute_formula": "cumulative_flops = 6 * total_parameters_with_lora * batch_examples * checkpoint_step",
    "trainable_compute_formula": "cumulative_trainable_flops = 6 * trainable_lora_parameters * batch_examples * checkpoint_step",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def force_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


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


def example_digest(examples: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(str(ex.get("id")).encode("utf-8"))
        h.update(b"\0")
        h.update(str(ex.get("source_index")).encode("utf-8"))
        h.update(b"\0")
        h.update(str(ex.get("label")).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def flops(total_params: int, step: int) -> int:
    return int(6 * total_params * CFG["batch_examples"] * step)


def trainable_flops(trainable_params: int, step: int) -> int:
    return int(6 * trainable_params * CFG["batch_examples"] * step)


def force_cuda(device_name: str) -> torch.device:
    if device_name != "cuda":
        raise RuntimeError("Board 2 prompt requires device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    return torch.device("cuda")


def make_student_model(device: torch.device) -> torch.nn.Module:
    base = load_lm(STUDENT, device, CFG["dtype"])
    lora_cfg = LoraConfig(
        r=int(CFG["lora_rank"]),
        lora_alpha=int(CFG["lora_alpha"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=float(CFG["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base, lora_cfg)
    model.config.use_cache = False
    return model


def param_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return total, trainable


def ece(conf: list[float], corr: list[int], bins: int = 10) -> float:
    if not conf:
        return 0.0
    c = np.asarray(conf, dtype=np.float64)
    r = np.asarray(corr, dtype=np.float64)
    out = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = ((c >= lo) if i == 0 else (c > lo)) & (c <= hi)
        if bool(mask.any()):
            out += float(mask.mean()) * abs(float(c[mask].mean()) - float(r[mask].mean()))
    return float(out)


@torch.no_grad()
def evaluate_prepared_labels(
    model: torch.nn.Module,
    prepared: PreparedChoiceSet,
    labels: list[int],
    pad_token_id: int,
    device: torch.device,
    teacher_targets: list[list[float]] | None = None,
    alpha: float = 1.0,
    kd_temperature: float = 1.0,
    batch_choices: int | None = None,
    progress: bool = False,
    name: str = "eval",
) -> dict[str, Any]:
    model.eval()
    started = time.time()
    flat_nlls, token_counts = score_prepared_flat(
        model,
        prepared,
        pad_token_id,
        device,
        int(batch_choices or CFG["eval_batch_choices"]),
        progress,
        name,
    )
    losses: list[float] = []
    ce_values: list[float] = []
    kl_values: list[float] = []
    correct: list[int] = []
    margins: list[float] = []
    confs: list[float] = []
    predictions: list[int] = []
    for i, label in enumerate(labels):
        start, end = prepared.spans[i]
        nlls = flat_nlls[start:end].float()
        logits = -nlls
        label_i = int(label)
        if label_i < 0 or label_i >= int(logits.numel()):
            raise ValueError(f"label {label_i} out of range for example {i} with {logits.numel()} choices")
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        ce = float((-log_probs[label_i]).item())
        pred = int(torch.argmax(logits).item())
        if teacher_targets is not None:
            target = torch.tensor(teacher_targets[i], dtype=torch.float32)
            log_s = F.log_softmax(logits / float(kd_temperature), dim=-1)
            kl = float(F.kl_div(log_s, target, reduction="sum").item() * (float(kd_temperature) ** 2))
            loss = float(alpha) * ce + (1.0 - float(alpha)) * kl
        else:
            kl = 0.0
            loss = ce
        wrong = [float(nlls[j].item()) for j in range(int(nlls.numel())) if j != label_i]
        gold = float(nlls[label_i].item())
        if wrong:
            margins.append(float(min(wrong) - gold))
        losses.append(loss)
        ce_values.append(ce)
        kl_values.append(kl)
        ok = int(pred == label_i)
        correct.append(ok)
        confs.append(float(probs[pred].item()))
        predictions.append(pred)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "ce": float(np.mean(ce_values)) if ce_values else 0.0,
        "kl": float(np.mean(kl_values)) if kl_values else 0.0,
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "n_examples": len(labels),
        "mean_margin_best_wrong_minus_gold_nll": float(np.mean(margins)) if margins else None,
        "median_margin_best_wrong_minus_gold_nll": float(np.median(margins)) if margins else None,
        "positive_margin_fraction": float(np.mean([m > 0.0 for m in margins])) if margins else None,
        "ece": ece(confs, correct),
        "elapsed_s": round(time.time() - started, 3),
        "mean_choice_tokens": float(torch.mean(token_counts.float()).item()) if int(token_counts.numel()) else 0.0,
        "predictions": predictions,
    }


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [(float(r[key]), int(r["n_examples"])) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    total = sum(n for _, n in vals)
    return float(sum(v * n for v, n in vals) / max(1, total))


def training_objective(
    nll_groups: list[torch.Tensor],
    labels: list[int],
    teacher_targets: list[list[float]] | None,
    alpha: float,
    kd_temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses, ce_values, kl_values, correct = [], [], [], []
    for local_i, nlls in enumerate(nll_groups):
        logits = -nlls.float()
        label = int(labels[local_i])
        ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long, device=device))
        ce_values.append(float(ce.detach().item()))
        correct.append(int(torch.argmax(logits.detach()).item() == label))
        if teacher_targets is None:
            losses.append(ce)
            kl_values.append(0.0)
            continue
        target = torch.tensor(teacher_targets[local_i], dtype=torch.float32, device=device)
        log_s = F.log_softmax(logits / kd_temperature, dim=-1)
        kl = F.kl_div(log_s, target, reduction="sum") * (kd_temperature**2)
        kl_values.append(float(kl.detach().item()))
        losses.append(float(alpha) * ce + (1.0 - float(alpha)) * kl)
    return torch.stack(losses).mean(), {
        "ce": float(np.mean(ce_values)),
        "kl": float(np.mean(kl_values)),
        "batch_accuracy": float(np.mean(correct)),
    }


def grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            g = p.grad.detach().float()
            total += float((g * g).sum().detach().cpu())
    return float(math.sqrt(total))


def build_shuffled_labels(train_examples: list[dict[str, Any]], seed: int) -> tuple[list[int], dict[str, Any]]:
    labels = [int(ex["label"]) for ex in train_examples]
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, ex in enumerate(train_examples):
        groups[(str(ex.get("benchmark")), len(ex["choices"]))].append(i)
    rng = np.random.default_rng(seed)
    out = list(labels)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        source = np.asarray([labels[i] for i in indices], dtype=np.int64)
        perm = rng.permutation(len(indices))
        shuffled = source[perm].tolist()
        if all(int(a) == int(b) for a, b in zip(source.tolist(), shuffled)):
            shuffled = source[np.roll(np.arange(len(indices)), 1)].tolist()
        for idx, value in zip(indices, shuffled):
            out[idx] = int(value)
    matches = sum(int(a == b) for a, b in zip(labels, out))
    return out, {
        "method": "per_benchmark_choice_count_label_permutation",
        "seed": seed,
        "n_train_examples": len(labels),
        "labels_matching_true_after_shuffle": matches,
        "fraction_matching_true_after_shuffle": matches / max(1, len(labels)),
        "groups": {f"{k[0]}:{k[1]}choices": len(v) for k, v in groups.items()},
    }


def examples_match_cache(examples: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> bool:
    if len(examples) != len(predictions):
        return False
    return all(str(ex.get("id")) == str(pred.get("id")) for ex, pred in zip(examples, predictions))


def load_or_generate_teacher_cache(
    train_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    out_dir: Path,
    device: torch.device,
    progress: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    b14_cache = Path("tmp_work_loop_b14") / "teacher_choice_cache.json"
    cache_status: dict[str, Any] = {
        "requested_teacher": SMOL_TEACHER,
        "b14_cache_path": str(b14_cache),
        "source": None,
        "validated_against_current_split": False,
    }
    if b14_cache.exists():
        payload = read_json(b14_cache)
        smol = payload.get("teachers", {}).get("smollm2_360m", {})
        train_preds = smol.get("train", {}).get("predictions", [])
        eval_preds = smol.get("eval", {}).get("predictions", [])
        if not examples_match_cache(train_examples, train_preds) or not examples_match_cache(eval_examples, eval_preds):
            raise RuntimeError("B14 teacher cache exists but does not match the reconstructed B14 split")
        cache_status.update(
            {
                "source": "reused_b14_teacher_choice_cache",
                "validated_against_current_split": True,
                "train_predictions": len(train_preds),
                "eval_predictions": len(eval_preds),
            }
        )
        return payload, cache_status

    args = SimpleNamespace(
        output_dir=str(out_dir),
        dtype=CFG["dtype"],
        max_length=int(CFG["max_length"]),
        eval_batch_choices=int(CFG["eval_batch_choices"]),
        progress=progress,
    )
    examples_by_split = {"train": train_examples, "eval": eval_examples}
    teacher_payload, manifest = cache_one_teacher(SMOL_TEACHER, examples_by_split, args, device)
    payload = {
        "teachers": {"smollm2_360m": teacher_payload},
        "manifests": {"smollm2_360m": manifest},
        "teacher_temperature": CFG["teacher_temperature"],
    }
    cache_path = out_dir / "teacher_choice_cache.json"
    write_json(cache_path, payload)
    cache_status.update(
        {
            "source": "generated_local_only_teacher_cache",
            "generated_cache_path": str(cache_path),
            "validated_against_current_split": True,
            "train_predictions": len(teacher_payload["train"]["predictions"]),
            "eval_predictions": len(teacher_payload["eval"]["predictions"]),
        }
    )
    return payload, cache_status


def prepare_board_data(out_dir: Path, device: torch.device, progress: bool) -> dict[str, Any]:
    train_by_bench, eval_by_bench, split_meta = make_split(
        int(CFG["train_per_benchmark"]),
        int(CFG["eval_per_benchmark"]),
        str(CFG["benchmark_split"]),
        B14_SPLIT_SEED,
        False,
    )
    train_examples = flatten_examples(train_by_bench)
    eval_examples = flatten_examples(eval_by_bench)
    teacher_cache, teacher_status = load_or_generate_teacher_cache(train_examples, eval_examples, out_dir, device, progress)
    tokenizer = load_tokenizer(STUDENT)
    train_prepared = prepare_choice_set(tokenizer, train_examples, int(CFG["max_length"]))
    eval_prepared_by_bench = {
        bench: prepare_choice_set(tokenizer, eval_by_bench[bench], int(CFG["max_length"])) for bench in BENCHMARKS
    }
    smol_train_records = teacher_cache["teachers"]["smollm2_360m"]["train"]["predictions"]
    single_targets = [
        list(t["probs"]) for t in teacher_records_to_targets(smol_train_records, float(CFG["teacher_temperature"]))
    ]
    true_train_labels = [int(ex["label"]) for ex in train_examples]
    shuffled_labels, shuffle_meta = build_shuffled_labels(train_examples, SEED + 7700)
    return {
        "tokenizer": tokenizer,
        "train_by_bench": train_by_bench,
        "eval_by_bench": eval_by_bench,
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "train_prepared": train_prepared,
        "eval_prepared_by_bench": eval_prepared_by_bench,
        "split_meta": split_meta,
        "teacher_cache_status": teacher_status,
        "single_teacher_targets": single_targets,
        "objective_labels": {
            "label_only": true_train_labels,
            "single_teacher": true_train_labels,
            "shuffled_labels": shuffled_labels,
        },
        "teacher_targets": {
            "label_only": None,
            "single_teacher": single_targets,
            "shuffled_labels": None,
        },
        "true_train_labels": true_train_labels,
        "shuffled_label_meta": shuffle_meta,
        "split_digest": {
            "train_examples_sha256": example_digest(train_examples),
            "eval_examples_sha256": example_digest(eval_examples),
        },
    }


def checkpoint_row(
    condition: str,
    step: int,
    model: torch.nn.Module,
    board_data: dict[str, Any],
    total_params: int,
    trainable_params: int,
    mb_loss: float,
    mb_stats: dict[str, float],
    elapsed: float,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    pad_id = int(board_data["tokenizer"].pad_token_id)
    train_prepared = board_data["train_prepared"]
    objective_labels = board_data["objective_labels"][condition]
    teacher_targets = board_data["teacher_targets"][condition]
    train_obj = evaluate_prepared_labels(
        model,
        train_prepared,
        objective_labels,
        pad_id,
        device,
        teacher_targets=teacher_targets,
        alpha=float(CFG["kd_alpha_label_ce"]),
        kd_temperature=float(CFG["kd_temperature"]),
        name=f"{condition}:train_objective:{step}",
    )
    train_true = evaluate_prepared_labels(
        model,
        train_prepared,
        board_data["true_train_labels"],
        pad_id,
        device,
        teacher_targets=None,
        name=f"{condition}:train_true:{step}",
    )
    bench_rows: dict[str, dict[str, Any]] = {}
    for bench, prepared in board_data["eval_prepared_by_bench"].items():
        labels = [int(ex["label"]) for ex in prepared.examples]
        bench_rows[bench] = evaluate_prepared_labels(
            model,
            prepared,
            labels,
            pad_id,
            device,
            teacher_targets=None,
            name=f"{condition}:eval:{bench}:{step}",
        )
    held_acc = weighted_mean(list(bench_rows.values()), "accuracy")
    assert held_acc is not None
    held_loss = weighted_mean(list(bench_rows.values()), "loss")
    held_margin = weighted_mean(list(bench_rows.values()), "mean_margin_best_wrong_minus_gold_nll")
    held_ece = weighted_mean(list(bench_rows.values()), "ece")
    row: dict[str, Any] = {
        "model_birth": CFG["model_birth"],
        "model_name": STUDENT,
        "task_family": "mcq_forced_choice",
        "task_id": "hellaswag_piqa_arc_easy_train_safe_b14_split",
        "intervention": condition,
        "seed": SEED,
        "split_seed": B14_SPLIT_SEED,
        "checkpoint_step": step,
        "cumulative_flops": flops(total_params, step),
        "cumulative_gflops": flops(total_params, step) / 1e9,
        "cumulative_trainable_flops": trainable_flops(trainable_params, step),
        "cumulative_trainable_gflops": trainable_flops(trainable_params, step) / 1e9,
        "total_params_with_lora": total_params,
        "trainable_params": trainable_params,
        "batch_size": CFG["batch_examples"],
        "train_examples_seen": int(CFG["batch_examples"]) * step,
        "train_examples_total": len(board_data["train_examples"]),
        "heldout_examples_total": len(board_data["eval_examples"]),
        "eval_split_id": "b14_train_safe_eval_seed20260707_no_source_overlap",
        "d_func": 1.0 - held_acc,
        "d_proxy": train_obj["loss"],
        "d_proxy_train_objective_loss": train_obj["loss"],
        "d_proxy_current_minibatch_loss": mb_loss,
        "d_gap": abs(float(train_obj["accuracy"]) - held_acc),
        "train_accuracy": train_obj["accuracy"],
        "train_loss": train_obj["loss"],
        "train_ce": train_obj["ce"],
        "train_kl": train_obj["kl"],
        "train_task_accuracy": train_true["accuracy"],
        "train_task_loss": train_true["loss"],
        "held_out_accuracy": held_acc,
        "held_out_loss": held_loss,
        "d_margin": held_margin,
        "d_cal": held_ece,
        "current_minibatch_accuracy": mb_stats.get("batch_accuracy"),
        "current_minibatch_ce": mb_stats.get("ce"),
        "current_minibatch_kl": mb_stats.get("kl"),
        "elapsed_seconds": elapsed,
        "created_at_utc": now(),
    }
    for bench in BENCHMARKS:
        br = bench_rows[bench]
        row[f"accuracy_{bench}"] = br["accuracy"]
        row[f"d_func_{bench}"] = 1.0 - float(br["accuracy"])
        row[f"loss_{bench}"] = br["loss"]
        row[f"margin_{bench}"] = br["mean_margin_best_wrong_minus_gold_nll"]
        row[f"ece_{bench}"] = br["ece"]
        row[f"n_{bench}"] = br["n_examples"]
    return row


def train_range(
    condition: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    rng: np.random.Generator,
    board_data: dict[str, Any],
    total_params: int,
    trainable_params: int,
    start: int,
    end: int,
    ckpts: set[int],
    log_path: Path,
    t0: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_prepared = board_data["train_prepared"]
    labels_all = board_data["objective_labels"][condition]
    targets_all = board_data["teacher_targets"][condition]
    trainable = [p for p in model.parameters() if p.requires_grad]
    mb_loss = float("nan")
    mb_stats: dict[str, float] = {"ce": float("nan"), "kl": float("nan"), "batch_accuracy": float("nan")}
    device = next(model.parameters()).device
    for step in range(start + 1, end + 1):
        model.train()
        idx = rng.choice(
            train_prepared.n_examples,
            size=min(int(CFG["batch_examples"]), train_prepared.n_examples),
            replace=False,
        )
        nll_groups = score_prepared_indices(
            model,
            train_prepared,
            idx,
            int(board_data["tokenizer"].pad_token_id),
            device,
        )
        batch_labels = [int(labels_all[int(i)]) for i in idx]
        batch_targets = None if targets_all is None else [targets_all[int(i)] for i in idx]
        loss, stats = training_objective(
            nll_groups,
            batch_labels,
            batch_targets,
            float(CFG["kd_alpha_label_ce"]),
            float(CFG["kd_temperature"]),
            device,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        unclipped_grad = grad_norm(trainable)
        clipped = torch.nn.utils.clip_grad_norm_(trainable, float(CFG["grad_clip"]))
        opt.step()
        mb_loss = float(loss.detach().cpu())
        mb_stats = stats
        append_jsonl(
            log_path,
            {
                "created_at_utc": now(),
                "condition": condition,
                "step": step,
                "loss": mb_loss,
                "ce": float(stats["ce"]),
                "kl": float(stats["kl"]),
                "batch_accuracy": float(stats["batch_accuracy"]),
                "grad_norm_unclipped": unclipped_grad,
                "grad_norm_clipped_return": float(clipped.detach().cpu()),
                "learning_rate": CFG["learning_rate"],
                "weight_decay": CFG["weight_decay"],
                "cumulative_flops": flops(total_params, step),
                "cumulative_trainable_flops": trainable_flops(trainable_params, step),
                "elapsed_seconds": time.perf_counter() - t0,
            },
        )
        if step in ckpts:
            rows.append(
                checkpoint_row(
                    condition,
                    step,
                    model,
                    board_data,
                    total_params,
                    trainable_params,
                    mb_loss,
                    mb_stats,
                    time.perf_counter() - t0,
                )
            )
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
    val = power_fn(np.array([x / fit["compute_normalizer_flops"]]), fit["d_inf"], fit["k"], fit["alpha"])[0]
    return clip01(float(val))


def fit_linear_log(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    lx = np.log(np.asarray(x, dtype=float))
    slope, intercept = np.polyfit(lx, y, deg=1)
    return {"slope": float(slope), "intercept": float(intercept)}


def build_predictions(rows: list[dict[str, Any]], total_params: int, out_dir: Path, split_digest: dict[str, str]) -> dict[str, Any]:
    early = [r for r in rows if int(r["checkpoint_step"]) in FIT_STEPS]
    if len(early) != len(CONDITIONS) * len(FIT_STEPS):
        raise RuntimeError(f"expected {len(CONDITIONS) * len(FIT_STEPS)} early rows, got {len(early)}")
    if any(int(r["checkpoint_step"]) > 100 for r in rows):
        raise RuntimeError("prediction lock attempted after held-out checkpoint rows were present")
    byc: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for c in CONDITIONS:
        rr = [r for r in early if r["intervention"] == c]
        byc[c] = sorted(rr, key=lambda r: int(r["checkpoint_step"]))
    held_c = {str(s): flops(total_params, s) for s in HELDOUT_STEPS}
    forecasts: dict[str, dict[str, dict[str, float]]] = {f: {} for f in FORECASTERS}
    fits: dict[str, Any] = {
        "cti_power_law": {},
        "proxy_power_law": {},
        "linear_log_compute": {},
        "per_benchmark_power_law": {},
    }
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
        forecasts["b0_last_point"][c] = {str(s): clip01(float(y[-1])) for s in HELDOUT_STEPS}
        forecasts["b1_linear_log_compute"][c] = {
            str(s): clip01(fl["slope"] * math.log(held_c[str(s)]) + fl["intercept"]) for s in HELDOUT_STEPS
        }
        per_bench_fits = {}
        per_bench_forecasts: dict[str, dict[str, float]] = {str(s): {} for s in HELDOUT_STEPS}
        for bench in BENCHMARKS:
            yb = np.array([float(r[f"d_func_{bench}"]) for r in rr])
            fb = fit_power(x, yb)
            per_bench_fits[bench] = fb
            for s in HELDOUT_STEPS:
                per_bench_forecasts[str(s)][bench] = pred_power(fb, held_c[str(s)])
        fits["per_benchmark_power_law"][c] = per_bench_fits
        forecasts["b2_per_benchmark_power_law"][c] = {
            str(s): clip01(float(np.mean(list(per_bench_forecasts[str(s)].values())))) for s in HELDOUT_STEPS
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
            "max_checkpoint_step_available_when_written": max(int(r["checkpoint_step"]) for r in rows),
            "actual_heldout_rows_available_when_written": False,
            "split_digest": split_digest,
            "total_params_with_lora": total_params,
            "compute_formula": CFG["compute_formula"],
            "artifact_note": "Written after all conditions reached step 100 and before any training/evaluation at step 300+.",
        },
        "forecasters": {
            "cti_power_law": "Aggregate D_func(C)=D_inf+k*C^-alpha fit on checkpoints 10,30,100.",
            "b0_last_point": "Hold step-100 aggregate D_func constant.",
            "b1_linear_log_compute": "Linear extrapolation of aggregate D_func against log(C).",
            "b2_per_benchmark_power_law": "Fit independent power laws for HellaSwag, PIQA, and ARC-Easy, then average.",
            "b3_proxy_only": "Forecast proxy loss, then map early proxy to early aggregate D_func.",
            "b4_random_intervention_ranking": "Seeded random intervention ranking with step-100 values assigned to the random order.",
        },
        "fits": fits,
        "heldout_compute_flops": held_c,
        "predicted_d_func": forecasts,
        "predicted_step3000_rankings_lowest_d_func_first": ranks,
        "step100_observed_d_func": vals100,
    }
    write_json(out_dir / "cti1_board2_predictions.json", payload)
    return payload


def score_predictions(pred: dict[str, Any], rows: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual = {
        (r["intervention"], str(int(r["checkpoint_step"]))): float(r["d_func"])
        for r in rows
        if int(r["checkpoint_step"]) in HELDOUT_STEPS
    }
    actual_rank = sorted(CONDITIONS, key=lambda c: (actual[(c, "3000")], c))
    score_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
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
        score_rows.append(
            {
                "forecaster": f,
                "mae_all_heldout_points": float(np.mean(errs)),
                "mae_label_only": float(np.mean(byc["label_only"])),
                "mae_single_teacher": float(np.mean(byc["single_teacher"])),
                "mae_shuffled_labels": float(np.mean(byc["shuffled_labels"])),
                "mae_step_300": float(np.mean(bys["300"])),
                "mae_step_1000": float(np.mean(bys["1000"])),
                "mae_step_3000": float(np.mean(bys["3000"])),
                "predicted_best_step3000": pr[0],
                "actual_best_step3000": actual_rank[0],
                "ranking_top1_correct": pr[0] == actual_rank[0],
                "predicted_ranking_step3000": " < ".join(pr),
                "actual_ranking_step3000": " < ".join(actual_rank),
            }
        )
    write_csv(out_dir / "cti1_board2_scores.csv", score_rows)
    return score_rows, {
        "actual_d_func": {f"{k[0]}:{k[1]}": v for k, v in actual.items()},
        "prediction_details": detail,
        "actual_step3000_ranking": actual_rank,
    }


def shift_classification(pred: dict[str, Any]) -> list[dict[str, Any]]:
    fits = pred["fits"]["cti_power_law"]
    ref = fits["label_only"]
    rows = []
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
        rows.append(
            {
                "intervention": c,
                "reference": "label_only",
                "alpha": float(f["alpha"]),
                "d_inf": float(f["d_inf"]),
                "delta_alpha_vs_label_only": da,
                "delta_d_inf_vs_label_only": dd,
                "classification": cls,
            }
        )
    return rows


def monotone_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in CONDITIONS:
        rr = sorted([r for r in rows if r["intervention"] == c], key=lambda r: int(r["checkpoint_step"]))
        vals = [float(r["d_func"]) for r in rr]
        steps = [int(r["checkpoint_step"]) for r in rr]
        improvements = [vals[i - 1] - vals[i] for i in range(1, len(vals))]
        violations = [steps[i] for i, imp in enumerate(improvements, start=1) if imp < -1e-12]
        out[c] = {
            "steps": steps,
            "d_func_by_step": {str(s): v for s, v in zip(steps, vals)},
            "monotone_nonincreasing_d_func": len(violations) == 0,
            "violation_steps": violations,
            "total_drop_step10_to_3000": vals[0] - vals[-1] if vals else None,
            "largest_single_step_improvement": max(improvements) if improvements else None,
            "largest_single_step_regression": min(improvements) if improvements else None,
        }
    return out


def validate(rows: list[dict[str, Any]], total_params: int, trainable_params: int, split_meta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    exp = {(c, s) for c in CONDITIONS for s in CHECKPOINTS}
    got = {(r["intervention"], int(r["checkpoint_step"])) for r in rows}
    if exp - got:
        reasons.append(f"missing checkpoint rows: {sorted(exp - got)}")
    for bench, meta in split_meta.items():
        if int(meta.get("source_overlap", 0)) != 0:
            reasons.append(f"train/eval source overlap for {bench}: {meta.get('source_overlap')}")
    for r in rows:
        s = int(r["checkpoint_step"])
        if int(r["cumulative_flops"]) != flops(total_params, s):
            reasons.append(f"bad cumulative_flops {r['intervention']} step {s}")
        if int(r["cumulative_trainable_flops"]) != trainable_flops(trainable_params, s):
            reasons.append(f"bad cumulative_trainable_flops {r['intervention']} step {s}")
        for key in ["d_func", "d_proxy", "d_gap", "train_accuracy", "held_out_accuracy"]:
            if not math.isfinite(float(r[key])):
                reasons.append(f"nonfinite {key} at {r['intervention']} step {s}")
    return reasons


def verdict(score_rows: list[dict[str, Any]], shift_rows: list[dict[str, Any]], invalid: list[str]) -> str:
    if invalid:
        return "INVALID_CTI"
    mae = {r["forecaster"]: float(r["mae_all_heldout_points"]) for r in score_rows}
    cti = mae["cti_power_law"]
    beats = all(cti < v for k, v in mae.items() if k != "cti_power_law")
    has_shift = any(r["classification"] in {"exponent_shift", "constant_shift"} for r in shift_rows)
    if beats and has_shift:
        return "PASS_CTI_LAW_0"
    if mae.get("b3_proxy_only", float("inf")) < cti:
        return "PROXY_ONLY_LAW"
    return "NO_PREDICTIVE_LAW"


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def md_table(headers: list[str], body: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in body]
    return "\n".join(lines)


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], pred: dict[str, Any], scores: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    ck = [
        [
            r["intervention"],
            int(r["checkpoint_step"]),
            f"{float(r['cumulative_gflops']):.3f}",
            f"{float(r['d_func']):.6f}",
            f"{float(r['d_proxy']):.6f}",
            f"{float(r['d_gap']):.6f}",
            pct(float(r["train_accuracy"])),
            pct(float(r["train_task_accuracy"])),
            pct(float(r["held_out_accuracy"])),
        ]
        for r in ordered
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
    cti_rank = " < ".join(pred["predicted_step3000_rankings_lowest_d_func_first"]["cti_power_law"])
    actual_rank = " < ".join(summary["score_detail"]["actual_step3000_ranking"])
    artifacts = "\n".join(f"- `{v}`" for v in summary["artifacts"].values())
    monotone_lines = []
    for c, m in summary["monotone_check"].items():
        monotone_lines.append(
            [
                c,
                m["monotone_nonincreasing_d_func"],
                f"{m['total_drop_step10_to_3000']:.6f}",
                ", ".join(str(s) for s in m["violation_steps"]) or "none",
            ]
        )
    text = f"""# W-Loop B17: CTI-1 Board 2 - SmolLM2-135M LoRA MCQ

**Date:** 2026-07-07
**Verdict token:** `{summary['verdict_token']}`
**Task:** HellaSwag, PIQA, ARC-Easy forced-choice MCQ
**Model:** SmolLM2-135M with rank-16 LoRA, {summary['trainable_params']:,} trainable / {summary['total_params_with_lora']:,} total parameters
**Device:** {summary['cuda_device']}

---

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch16.md`
4. `research/work_loop_batch15.md`
5. `research/question_loop_batch21.md`

Binding interpretation: Board 2 is the monotone adaptation check after Board 1's grokking confound. The primary measurement is `D_func = 1 - mean_accuracy` over HellaSwag, PIQA, and ARC-Easy at every log-spaced checkpoint. The train/eval split is reconstructed from B14 with split seed `{B14_SPLIT_SEED}`; the training seed for this board is `{SEED}`.

## Smoke Run

Completed before the full run. The smoke path loaded `HuggingFaceTB/SmolLM2-135M` with `local_files_only=True`, trained `label_only` for 10 steps on cuda, recorded checkpoint step 10, and ran MCQ evaluation at step 10.

Smoke artifact: `{summary['artifacts'].get('smoke_summary', 'missing')}`.

## Configuration

| Parameter | Value |
|---|---:|
| Train examples | {summary['train_examples_total']} |
| Held-out examples | {summary['heldout_examples_total']} |
| Batch size | {CFG['batch_examples']} |
| Learning rate | {CFG['learning_rate']} |
| Weight decay | {CFG['weight_decay']} |
| Max steps | {CFG['max_steps']} |
| Checkpoints | {', '.join(str(s) for s in CHECKPOINTS)} |
| Fit-only checkpoints | {', '.join(str(s) for s in FIT_STEPS)} |
| Held-out forecast checkpoints | {', '.join(str(s) for s in HELDOUT_STEPS)} |
| Primary compute formula | `{CFG['compute_formula']}` |

## Teacher Cache

Teacher cache source: `{summary['teacher_cache_status']['source']}`. Cache/split validation: `{summary['teacher_cache_status']['validated_against_current_split']}`.

## Prediction Lock

Predictions were written after all three interventions reached step 100 and before any training or evaluation at steps 300, 1000, or 3000. The lock record is in `tmp_work_loop_b17/cti1_board2_predictions.json`.

CTI predicted step-3000 ranking:

```text
{cti_rank}
```

Actual step-3000 ranking:

```text
{actual_rank}
```

## Checkpoint Matrix

{md_table(['Intervention', 'Step', 'GFLOPs', 'D_func', 'D_proxy', 'D_gap', 'Train Obj Acc', 'Train True Acc', 'Held-out Acc'], ck)}

## Forecast Scores

{md_table(['Forecaster', 'MAE held-out', 'Predicted best', 'Actual best', 'Top-1 correct'], sc)}

## Intervention Shift Classification

{md_table(['Intervention', 'alpha', 'D_inf', 'delta alpha vs label_only', 'Classification'], sh)}

## Monotone Check

{md_table(['Intervention', 'Monotone nonincreasing D_func', 'D_func drop 10->3000', 'Regression steps'], monotone_lines)}

## Artifacts

{artifacts}

## NARRATIVE SECTION

Does CTI predict on a monotone task: by the strict precommit token, `{summary['verdict_token']}`. The CTI aggregate power-law MAE was {summary['cti_mae_all_heldout_points']:.6f}; the best forecaster was `{summary['best_forecaster_by_mae']}` at {summary['best_forecaster_mae']:.6f}.

If yes, grokking was the confound. If no, CTI is dead or at least not alive on the natural monotone adaptation domain this board was designed to rescue.

Honest gossip-magazine story: the laptop got the clean monotone rematch after grokking spoiled the first board. It saw only the first three checkpoints, locked its prediction, then had to call the final intervention ranking before the late values were opened. The score table is the story; no thermodynamics language gets to outrun it.
"""
    Path("research/work_loop_batch17.md").write_text(text, encoding="utf-8")


def initialize_states(device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, torch.optim.Optimizer], dict[str, np.random.Generator], int, int]:
    models: dict[str, torch.nn.Module] = {}
    opts: dict[str, torch.optim.Optimizer] = {}
    rngs: dict[str, np.random.Generator] = {}
    total_params = None
    trainable_params = None
    for i, condition in enumerate(CONDITIONS):
        set_seed(SEED)
        model = make_student_model(device)
        total, trainable = param_counts(model)
        if total_params is None:
            total_params, trainable_params = total, trainable
        elif total != total_params or trainable != trainable_params:
            raise RuntimeError("parameter count mismatch across conditions")
        opts[condition] = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(CFG["learning_rate"]),
            weight_decay=float(CFG["weight_decay"]),
        )
        models[condition] = model
        rngs[condition] = np.random.default_rng(SEED + 3000 + i)
    assert total_params is not None and trainable_params is not None
    return models, opts, rngs, total_params, trainable_params


def run_smoke(out_dir: Path, device_name: str, progress: bool) -> None:
    force_offline()
    device = force_cuda(device_name)
    out = out_dir / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    board_data = prepare_board_data(out_dir, device, progress)
    model = make_student_model(device)
    total, trainable = param_counts(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(CFG["learning_rate"]), weight_decay=float(CFG["weight_decay"]))
    log = out / "cti1_board2_smoke_train_log.jsonl"
    if log.exists():
        log.unlink()
    rng = np.random.default_rng(SEED + 9100)
    t0 = time.perf_counter()
    rows = train_range("label_only", model, opt, rng, board_data, total, trainable, 0, 10, {10}, log, t0)
    ck = out / "cti1_board2_smoke_checkpoints.csv"
    write_csv(ck, rows)
    req = ["d_func", "d_proxy", "d_gap", "train_accuracy", "held_out_accuracy"]
    ok = (
        len(rows) == 1
        and int(rows[0]["checkpoint_step"]) == 10
        and int(rows[0]["cumulative_flops"]) == flops(total, 10)
        and all(math.isfinite(float(rows[0][k])) for k in req)
    )
    summary = {
        "created_at_utc": now(),
        "smoke_ok": ok,
        "condition": "label_only",
        "steps": 10,
        "student_model": STUDENT,
        "local_files_only": True,
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0),
        "dtype": CFG["dtype"],
        "total_params_with_lora": total,
        "trainable_params": trainable,
        "expected_cumulative_flops_step10": flops(total, 10),
        "observed_cumulative_flops_step10": int(rows[0]["cumulative_flops"]),
        "required_metric_keys_checked": req,
        "split_digest": board_data["split_digest"],
        "teacher_cache_status": board_data["teacher_cache_status"],
        "artifacts": {
            "train_log": str(log),
            "checkpoints": str(ck),
            "summary": str(out / "cti1_board2_smoke_summary.json"),
        },
    }
    write_json(out / "cti1_board2_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_full(out_dir: Path, device_name: str, progress: bool) -> None:
    force_offline()
    device = force_cuda(device_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    board_data = prepare_board_data(out_dir, device, progress)
    models, opts, rngs, total, trainable = initialize_states(device)
    cfg = dict(CFG)
    cfg.update(
        {
            "created_at_utc": now(),
            "total_params_with_lora": total,
            "trainable_params": trainable,
            "split_digest": board_data["split_digest"],
            "split_meta": board_data["split_meta"],
            "teacher_cache_status": board_data["teacher_cache_status"],
            "shuffled_label_meta": board_data["shuffled_label_meta"],
            "student_manifest": model_manifest(models["label_only"], STUDENT, board_data["tokenizer"]),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        }
    )
    write_json(out_dir / "cti1_board2_config.json", cfg)
    log = out_dir / "cti1_board2_train_log.jsonl"
    if log.exists():
        log.unlink()
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for condition in CONDITIONS:
        rows.extend(train_range(condition, models[condition], opts[condition], rngs[condition], board_data, total, trainable, 0, 100, set(FIT_STEPS), log, t0))
    rows.sort(key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    write_csv(out_dir / "cti1_board2_checkpoints.csv", rows)
    pred = build_predictions(rows, total, out_dir, board_data["split_digest"])
    pred_sha = sha256_file(out_dir / "cti1_board2_predictions.json")

    for condition in CONDITIONS:
        rows.extend(train_range(condition, models[condition], opts[condition], rngs[condition], board_data, total, trainable, 100, int(CFG["max_steps"]), set(HELDOUT_STEPS), log, t0))
    rows.sort(key=lambda r: (r["intervention"], int(r["checkpoint_step"])))
    write_csv(out_dir / "cti1_board2_checkpoints.csv", rows)
    scores, detail = score_predictions(pred, rows, out_dir)
    invalid = validate(rows, total, trainable, board_data["split_meta"])
    shifts = shift_classification(pred)
    vt = verdict(scores, shifts, invalid)
    best = min(scores, key=lambda r: float(r["mae_all_heldout_points"]))
    cti = next(r for r in scores if r["forecaster"] == "cti_power_law")
    artifacts = {
        "config": str(out_dir / "cti1_board2_config.json"),
        "train_log": str(log),
        "checkpoints": str(out_dir / "cti1_board2_checkpoints.csv"),
        "predictions": str(out_dir / "cti1_board2_predictions.json"),
        "scores": str(out_dir / "cti1_board2_scores.csv"),
        "summary": str(out_dir / "cti1_board2_summary.json"),
        "report": "research/work_loop_batch17.md",
    }
    smoke = out_dir / "smoke" / "cti1_board2_smoke_summary.json"
    if smoke.exists():
        artifacts["smoke_summary"] = str(smoke)
    summary = {
        "created_at_utc": now(),
        "verdict_token": vt,
        "invalid_reasons": invalid,
        "board": CFG["board"],
        "task": CFG["task"],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "dtype": CFG["dtype"],
        "student_model": STUDENT,
        "local_files_only": True,
        "total_params_with_lora": total,
        "trainable_params": trainable,
        "train_examples_total": len(board_data["train_examples"]),
        "heldout_examples_total": len(board_data["eval_examples"]),
        "split_digest": board_data["split_digest"],
        "split_meta": board_data["split_meta"],
        "teacher_cache_status": board_data["teacher_cache_status"],
        "shuffled_label_meta": board_data["shuffled_label_meta"],
        "prediction_lock": pred["prediction_lock"],
        "prediction_file_sha256_before_resume": pred_sha,
        "prediction_file_sha256_after_scoring": sha256_file(out_dir / "cti1_board2_predictions.json"),
        "cti_mae_all_heldout_points": float(cti["mae_all_heldout_points"]),
        "best_forecaster_by_mae": best["forecaster"],
        "best_forecaster_mae": float(best["mae_all_heldout_points"]),
        "scores": scores,
        "score_detail": detail,
        "shift_classification": shifts,
        "monotone_check": monotone_check(rows),
        "final_step3000": [r for r in rows if int(r["checkpoint_step"]) == 3000],
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "artifacts": artifacts,
        "notes": [
            "Every AutoTokenizer/AutoModelForCausalLM load in the reused B14 helpers uses local_files_only=True.",
            "HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE, and HF_DATASETS_OFFLINE were forced before data/model loading.",
            "D_func is 1 - mean held-out MCQ forced-choice accuracy over HellaSwag, PIQA, and ARC-Easy.",
            "Predictions were written after step 100 and before training/evaluation at steps 300, 1000, and 3000.",
            "The B14 teacher cache was reused and validated against the reconstructed B14 split when available.",
        ],
    }
    summary["total_elapsed_seconds"] = time.perf_counter() - t0
    write_json(out_dir / "cti1_board2_summary.json", summary)
    write_report(summary, rows, pred, scores)
    print(json.dumps({"verdict": vt, "cti_mae": summary["cti_mae_all_heldout_points"], "best_forecaster": best["forecaster"], "best_forecaster_mae": summary["best_forecaster_mae"], "elapsed_seconds": summary["total_elapsed_seconds"], "artifacts": artifacts}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ["smoke", "full"]:
        p = sub.add_parser(name)
        p.add_argument("--output-dir", default="tmp_work_loop_b17")
        p.add_argument("--device", default="cuda", choices=["cuda"])
        p.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.cmd == "smoke":
        run_smoke(Path(args.output_dir), args.device, args.progress)
    elif args.cmd == "full":
        run_full(Path(args.output_dir), args.device, args.progress)


if __name__ == "__main__":
    main()
