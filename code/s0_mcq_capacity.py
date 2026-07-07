"""S0/Wide7 multiple-choice capacity and teacher audit harness.

This is the W-loop B13 harness.  It deliberately reuses the benchmark record
semantics from coordinate_inheritance.py / margin_distillation.py while using a
real trained SutraS0-family checkpoint instead of the B12 MarginStudent.

Primary S0 capacity path:
- score choices with the checkpoint's native byte-NLL continuation scorer;
- freeze S0/Wide7;
- train a small zero-initialized residual MCQ head on S0 hidden summaries;
- compare held-out accuracy against the exact native byte-NLL baseline.

The residual head starts at zero, so untrained-head scores equal baseline S0
byte-NLL scores.  This keeps the before/after scoring surface comparable while
testing whether the pretrained byte representations carry MCQ-discriminative
information.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from coordinate_inheritance import (  # noqa: E402
    ScoredCompletion,
    bootstrap_accuracy_delta,
    bootstrap_scalar_delta,
    build_choice_prediction_record,
    choose_device,
    ensure_offline,
    evaluate_teacher_rankings,
    load_limited_benchmark,
    load_teacher,
    strip_predictions,
    summarize_prediction_records,
    write_json,
)
from s0_architecture import S0Config, SutraS0  # noqa: E402
from tier3_brainseed_chart_probe import load_tokenizer  # noqa: E402


BENCHMARKS = ("hellaswag", "piqa", "arc_easy")
PASS_S0 = "PASS_S0_CAPACITY"
MARGINAL_S0 = "MARGINAL_S0_CAPACITY"
FAIL_S0 = "FAIL_S0_CAPACITY"
PASS_FMD = "PASS_FMD_ON_S0"
MARGINAL_FMD = "MARGINAL_FMD_ON_S0"
FAIL_FMD = "FAIL_FMD_ON_S0"
UPGRADE_TEACHER = "UPGRADE_TEACHER"
MAINTAIN_QWEN = "MAINTAIN_QWEN"


@dataclass
class FeatureBundle:
    features: torch.Tensor
    native_nll: torch.Tensor
    spans: list[tuple[int, int]]
    labels: list[int]
    examples: list[dict]


def set_all_seeds(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def finite(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_s0_checkpoint(path: str, device: torch.device) -> tuple[SutraS0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("model_cfg") or S0Config()
    model = SutraS0(cfg)
    state = payload["model"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:10]}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    manifest = {
        "path": path,
        "step": int(payload.get("step", -1)),
        "eval_bpb": finite(payload.get("eval_bpb")),
        "model_cfg": asdict(cfg),
        "missing_keys": list(missing),
        "param_counts": model.count_parameters(),
    }
    return model, manifest


def prepare_s0_scoring_batch(
    contexts: list[str],
    choices: list[str],
    max_bytes: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    rows: list[np.ndarray] = []
    masks_flat: list[np.ndarray] = []
    max_len = patch_size * 2
    for context, choice in zip(contexts, choices):
        ctx_raw = context.encode("utf-8", errors="replace")
        choice_raw = choice.encode("utf-8", errors="replace") or b" "
        if len(choice_raw) >= max_bytes - patch_size:
            choice_raw = choice_raw[: max_bytes // 2]
        keep_ctx = max(patch_size, max_bytes - len(choice_raw))
        if len(ctx_raw) > keep_ctx:
            ctx_raw = ctx_raw[-keep_ctx:]
        if len(ctx_raw) < patch_size:
            ctx_raw = (b" " * (patch_size - len(ctx_raw))) + ctx_raw
        raw = (ctx_raw + choice_raw)[:max_bytes]
        usable_len = len(raw)
        padded_len = max(
            patch_size * 2,
            int(math.ceil(usable_len / patch_size) * patch_size),
        )
        max_len = max(max_len, padded_len)
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64).copy()
        arr[arr == 0xFF] = 32
        byte_mask = np.zeros((padded_len // patch_size - 1, patch_size), dtype=np.bool_)
        start = min(len(ctx_raw), usable_len)
        for pos in range(start, usable_len):
            patch_idx = pos // patch_size
            if patch_idx == 0:
                continue
            byte_pos = pos % patch_size
            byte_mask[patch_idx - 1, byte_pos] = True
        rows.append(arr)
        masks_flat.append(byte_mask)

    padded_rows: list[np.ndarray] = []
    masks: list[torch.Tensor] = []
    for arr, mask in zip(rows, masks_flat):
        pad = max_len - len(arr)
        if pad > 0:
            arr = np.pad(arr, (0, pad), constant_values=32)
        n_minus_1 = max_len // patch_size - 1
        full_mask = np.zeros((n_minus_1, patch_size), dtype=np.bool_)
        full_mask[: mask.shape[0], :] = mask
        padded_rows.append(arr)
        masks.append(torch.from_numpy(full_mask))
    byte_ids = torch.from_numpy(np.stack(padded_rows)).long().to(device)
    return byte_ids, masks


@torch.no_grad()
def score_and_extract_features(
    model: SutraS0,
    contexts: list[str],
    choices: list[str],
    max_bytes: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    byte_ids, masks = prepare_s0_scoring_batch(
        contexts,
        choices,
        max_bytes,
        int(model.cfg.patch_size),
        device,
    )
    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        out = model(byte_ids, return_aux=False)
    logits = out["logits"].float()
    hidden = out["hidden"].float()
    bsz, n_minus_1, psize, vocab = logits.shape
    targets = byte_ids.reshape(bsz, n_minus_1 + 1, psize)[:, 1:]
    per_byte = F.cross_entropy(
        logits.reshape(-1, vocab),
        targets.reshape(-1),
        reduction="none",
    ).reshape(bsz, n_minus_1, psize)

    features: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    pred_hidden = hidden[:, :-1, :]
    for i, mask_cpu in enumerate(masks):
        mask = mask_cpu.to(device=device)
        if int(mask.sum().item()) == 0:
            nll = per_byte[i].mean() * 0.0 + 1e6
            feat = pred_hidden[i, -1]
        else:
            nll = per_byte[i][mask].mean()
            patch_mask = mask.any(dim=1)
            feat = pred_hidden[i][patch_mask].mean(dim=0)
        nlls.append(nll.float())
        features.append(feat.float())
    return torch.stack(features), torch.stack(nlls)


def make_split(
    examples_per_benchmark: int,
    train_per_benchmark: int,
    eval_per_benchmark: int,
    split: str,
    seed: int,
    allow_downloads: bool,
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict]:
    train: dict[str, list[dict]] = {}
    evals: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    needed = train_per_benchmark + eval_per_benchmark
    count = max(examples_per_benchmark, needed)
    for bench in BENCHMARKS:
        examples = load_limited_benchmark(
            bench,
            count,
            split,
            seed + 1700 + len(bench),
            allow_downloads,
        )
        tr = examples[:train_per_benchmark]
        ev = examples[train_per_benchmark:train_per_benchmark + eval_per_benchmark]
        train[bench] = tr
        evals[bench] = ev
        train_ids = {ex["source_index"] for ex in tr}
        eval_ids = {ex["source_index"] for ex in ev}
        meta[bench] = {
            "loaded_examples": len(examples),
            "train_examples": len(tr),
            "eval_examples": len(ev),
            "source_overlap": len(train_ids & eval_ids),
        }
    return train, evals, meta


def flatten_examples(examples_by_bench: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for bench in BENCHMARKS:
        for ex in examples_by_bench[bench]:
            row = dict(ex)
            row["benchmark"] = bench
            rows.append(row)
    return rows


def extract_feature_bundle(
    model: SutraS0,
    examples: list[dict],
    max_bytes: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    feature_batch_choices: int,
    progress: bool,
    name: str,
) -> FeatureBundle:
    flat_contexts: list[str] = []
    flat_choices: list[str] = []
    spans: list[tuple[int, int]] = []
    labels: list[int] = []
    cursor = 0
    for ex in examples:
        n = len(ex["choices"])
        flat_contexts.extend([ex["context"] for _ in range(n)])
        flat_choices.extend(ex["choices"])
        spans.append((cursor, cursor + n))
        labels.append(int(ex["label"]))
        cursor += n

    features: list[torch.Tensor] = []
    nlls: list[torch.Tensor] = []
    started = time.time()
    for start in range(0, len(flat_choices), feature_batch_choices):
        end = min(len(flat_choices), start + feature_batch_choices)
        feat, nll = score_and_extract_features(
            model,
            flat_contexts[start:end],
            flat_choices[start:end],
            max_bytes,
            device,
            amp_dtype,
        )
        features.append(feat.cpu())
        nlls.append(nll.cpu())
        if progress:
            print(f"  [{name}] choices {end}/{len(flat_choices)}", flush=True)
    if progress:
        print(f"  [{name}] feature extraction {time.time() - started:.1f}s", flush=True)
    return FeatureBundle(
        features=torch.cat(features, dim=0),
        native_nll=torch.cat(nlls, dim=0),
        spans=spans,
        labels=labels,
        examples=examples,
    )


class ResidualChoiceHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, native_nll: torch.Tensor) -> torch.Tensor:
        residual = self.net(features).squeeze(-1)
        return -native_nll + residual


def grouped_ce_loss(logits: torch.Tensor, spans: list[tuple[int, int]], labels: list[int], indices: np.ndarray) -> torch.Tensor:
    losses = []
    for idx in indices:
        start, end = spans[int(idx)]
        y = torch.tensor([int(labels[int(idx)])], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits[start:end].unsqueeze(0), y))
    return torch.stack(losses).mean()


def grouped_fmd_loss(
    logits: torch.Tensor,
    spans: list[tuple[int, int]],
    labels: list[int],
    teacher_scores: list[list[float]],
    indices: np.ndarray,
    margin_scale: float,
    teacher_correct_only: bool,
) -> tuple[torch.Tensor, dict]:
    losses = []
    used = 0
    skipped = 0
    teacher_correct = 0
    for idx in indices:
        idx = int(idx)
        start, end = spans[idx]
        label = int(labels[idx])
        t = teacher_scores[idx]
        if len(t) != end - start or not all(math.isfinite(float(x)) for x in t):
            skipped += 1
            continue
        teacher_pred = int(np.argmin(np.asarray(t, dtype=np.float64)))
        teacher_ok = teacher_pred == label
        teacher_correct += int(teacher_ok)
        if teacher_correct_only and not teacher_ok:
            skipped += 1
            continue
        wrong = [j for j in range(len(t)) if j != label]
        if not wrong:
            skipped += 1
            continue
        hard_wrong = min(wrong, key=lambda j: (float(t[j]), j))
        teacher_margin = float(t[hard_wrong]) - float(t[label])
        target = max(0.05, min(2.0, abs(teacher_margin))) * margin_scale
        gold_logit = logits[start + label]
        wrong_logit = logits[start + hard_wrong]
        losses.append(F.softplus(-(gold_logit - wrong_logit - target)))
        used += 1
    if not losses:
        return logits.sum() * 0.0, {
            "used": used,
            "skipped": skipped,
            "teacher_correct_in_batch": teacher_correct,
        }
    return torch.stack(losses).mean(), {
        "used": used,
        "skipped": skipped,
        "teacher_correct_in_batch": teacher_correct,
    }


def predictions_from_bundle(
    bundle: FeatureBundle,
    logits: torch.Tensor | None,
    name: str,
) -> dict:
    predictions = []
    scores_flat = bundle.native_nll if logits is None else -logits.detach().cpu()
    for i, ex in enumerate(bundle.examples):
        start, end = bundle.spans[i]
        scored = [ScoredCompletion(float(scores_flat[j].item()), 1) for j in range(start, end)]
        predictions.append(build_choice_prediction_record(ex, scored))
    summary = summarize_prediction_records(predictions)
    summary["predictions"] = predictions
    summary["score"] = name
    return summary


def evaluate_head(head: ResidualChoiceHead, bundle: FeatureBundle, device: torch.device, batch_choices: int = 4096) -> dict:
    head.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, int(bundle.native_nll.numel()), batch_choices):
            end = min(int(bundle.native_nll.numel()), start + batch_choices)
            logits_parts.append(
                head(
                    bundle.features[start:end].to(device),
                    bundle.native_nll[start:end].to(device),
                ).cpu()
            )
    logits = torch.cat(logits_parts, dim=0)
    return predictions_from_bundle(bundle, logits, "s0_residual_head_pseudo_nll")


def train_label_head(
    train_bundle: FeatureBundle,
    eval_bundle_by_bench: dict[str, FeatureBundle],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualChoiceHead, dict, dict[str, dict], dict[str, dict]]:
    d_model = int(train_bundle.features.shape[1])
    head = ResidualChoiceHead(d_model, args.head_dim, args.head_dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 3100)
    n_examples = len(train_bundle.spans)
    history = []
    head.train()
    for step in range(1, args.train_steps + 1):
        idx = rng.choice(n_examples, size=min(args.batch_examples, n_examples), replace=False)
        logits = head(train_bundle.features.to(device), train_bundle.native_nll.to(device))
        loss = grouped_ce_loss(logits, train_bundle.spans, train_bundle.labels, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % max(1, args.log_every) == 0 or step == args.train_steps:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
            })
            if args.progress:
                print(f"  label head step {step}/{args.train_steps}: loss={loss.item():.4f}", flush=True)

    train_eval = {"all": evaluate_head(head, train_bundle, device)}
    evals = {bench: evaluate_head(head, bundle, device) for bench, bundle in eval_bundle_by_bench.items()}
    training = {
        "objective": "label_only_choice_ce",
        "steps": args.train_steps,
        "batch_examples": args.batch_examples,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "head_dim": args.head_dim,
        "trainable_parameters": int(sum(p.numel() for p in head.parameters())),
        "history": history,
    }
    return head, training, train_eval, evals


def verdict_s0_capacity(benchmarks: dict[str, dict]) -> dict:
    rows = {}
    pass_count = 0
    marginal_count = 0
    for bench, result in benchmarks.items():
        baseline = float(result["baseline_untrained"]["accuracy"])
        trained = float(result["label_ce_trained"]["accuracy"])
        delta = trained - baseline
        passed = delta >= 0.05
        marginal = 0.02 <= delta < 0.05
        pass_count += int(passed)
        marginal_count += int(marginal)
        rows[bench] = {
            "baseline_accuracy": baseline,
            "label_ce_accuracy": trained,
            "delta_accuracy": delta,
            "passes_plus_5pp": passed,
            "marginal_plus_2_to_5pp": marginal,
        }
    if pass_count >= 2:
        verdict = PASS_S0
        story = "trained_byte_representations_support_mcq_discrimination"
    elif pass_count + marginal_count >= 2:
        verdict = MARGINAL_S0
        story = "small_or_unstable_mcq_capacity_signal"
    else:
        verdict = FAIL_S0
        story = "trained_byte_model_did_not_show_heldout_label_capacity"
    return {
        "verdict": verdict,
        "causal_story": story,
        "pass_threshold_accuracy_delta": 0.05,
        "marginal_band": [0.02, 0.05],
        "required_pass_benchmarks": 2,
        "passed_benchmarks": pass_count,
        "marginal_benchmarks": marginal_count,
        "benchmarks": rows,
    }


def run_capacity(args: argparse.Namespace) -> dict:
    started = time.time()
    set_all_seeds(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    amp_dtype = getattr(torch, args.dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_by_bench, eval_by_bench, split_meta = make_split(
        args.examples_per_benchmark,
        args.train_per_benchmark,
        args.eval_per_benchmark,
        args.benchmark_split,
        args.seed,
        args.allow_downloads,
    )
    model, checkpoint_manifest = load_s0_checkpoint(args.checkpoint, device)

    train_examples = flatten_examples(train_by_bench)
    train_bundle = extract_feature_bundle(
        model,
        train_examples,
        args.max_bytes,
        device,
        amp_dtype,
        args.feature_batch_choices,
        args.progress,
        "train",
    )
    eval_bundles = {
        bench: extract_feature_bundle(
            model,
            eval_by_bench[bench],
            args.max_bytes,
            device,
            amp_dtype,
            args.feature_batch_choices,
            args.progress,
            f"eval:{bench}",
        )
        for bench in BENCHMARKS
    }

    baseline_train = {"all": predictions_from_bundle(train_bundle, None, "s0_native_byte_nll")}
    baseline_eval = {
        bench: predictions_from_bundle(bundle, None, "s0_native_byte_nll")
        for bench, bundle in eval_bundles.items()
    }
    head, training, trained_train, trained_eval = train_label_head(train_bundle, eval_bundles, args, device)

    benchmark_details: dict[str, dict] = {}
    benchmark_results: dict[str, dict] = {}
    for bench in BENCHMARKS:
        base = baseline_eval[bench]
        trained = trained_eval[bench]
        benchmark_details[bench] = {
            "metadata": {
                "split": args.benchmark_split,
                "train_safe": args.benchmark_split == "train",
                "train_examples": len(train_by_bench[bench]),
                "eval_examples": len(eval_by_bench[bench]),
                "score": "native S0 byte-NLL baseline vs zero-init residual MCQ head pseudo-NLL",
            },
            "baseline_untrained": base,
            "label_ce_trained": trained,
            "delta_label_ce_trained_minus_baseline_untrained": {
                "accuracy": bootstrap_accuracy_delta(
                    trained["predictions"],
                    base["predictions"],
                    args.bootstrap_samples,
                    args.seed + 4000 + len(bench),
                ),
                "margin_best_wrong_minus_gold_nll": bootstrap_scalar_delta(
                    trained["predictions"],
                    base["predictions"],
                    "margin_best_wrong_minus_gold_nll",
                    args.bootstrap_samples,
                    args.seed + 4100 + len(bench),
                ),
            },
        }
        benchmark_results[bench] = strip_predictions(benchmark_details[bench])

    verdict = verdict_s0_capacity(benchmark_details)
    payload = {
        "mode": "s0_wide7_scaffold_capacity",
        "run": {
            "seed": args.seed,
            "device": str(device),
            "checkpoint": args.checkpoint,
            "checkpoint_label": args.checkpoint_label,
            "elapsed_s": round(time.time() - started, 3),
            "benchmark_split": args.benchmark_split,
            "benchmarks": list(BENCHMARKS),
            "train_examples_total": len(train_examples),
            "eval_examples_total": sum(len(v) for v in eval_by_bench.values()),
            "max_bytes": args.max_bytes,
            "dtype": args.dtype,
        },
        "method": {
            "architecture_option": "Option A: frozen S0/Wide7 with trainable residual MCQ head",
            "baseline": "native checkpoint byte-NLL continuation score",
            "head": "zero-initialized residual score added to -native_nll; step-0 equals baseline",
            "primary_metric": "held-out benchmark MCQ accuracy and gold-vs-best-wrong margins",
        },
        "precommitted_verdict_tokens": {
            "pass": PASS_S0,
            "marginal": MARGINAL_S0,
            "fail": FAIL_S0,
        },
        "checkpoint": checkpoint_manifest,
        "split": split_meta,
        "training": training,
        "train_results": {
            "baseline_untrained": strip_predictions(baseline_train),
            "label_ce_trained": strip_predictions(trained_train),
        },
        "scaffold_capacity": verdict,
        "benchmarks": benchmark_results,
        "benchmark_details": benchmark_details if args.save_predictions else {},
        "limitations": [
            "This is a train-safe held-out capacity probe, not public validation.",
            "The primary capacity path freezes the byte model and trains a small residual head.",
            "A pass means the trained byte representations support MCQ discrimination; it is not yet a byte-LM fine-tune pass.",
        ],
    }
    write_json(out_dir / "s0_capacity.json", payload)
    torch.save(
        {
            "head_state_dict": head.cpu().state_dict(),
            "head_dim": args.head_dim,
            "checkpoint": args.checkpoint,
            "training": training,
            "verdict": verdict,
        },
        out_dir / "s0_capacity_head.pt",
    )
    return payload


def teacher_extra_metrics(predictions: list[dict], confident_threshold: float) -> dict:
    confident_wrong = []
    shortest = []
    longest = []
    pos_counts: dict[str, int] = {}
    for pred in predictions:
        margin = pred.get("margin_best_wrong_minus_gold_nll")
        confident_wrong.append(int((not pred["correct"]) and margin is not None and float(margin) <= -confident_threshold))
        choices = pred.get("choice_scores", [])
        best_wrong = int(pred.get("best_wrong_index", -1))
        if 0 <= best_wrong < len(choices):
            lengths = [int(c.get("n_tokens", 0)) for c in choices]
            bw_len = lengths[best_wrong]
            shortest.append(int(bw_len == min(lengths)))
            longest.append(int(bw_len == max(lengths)))
            pos_counts[str(best_wrong)] = pos_counts.get(str(best_wrong), 0) + 1
    return {
        "confident_wrong_threshold": confident_threshold,
        "confident_wrong_rate": float(np.mean(confident_wrong)) if confident_wrong else None,
        "hard_negative_shortest_fraction": float(np.mean(shortest)) if shortest else None,
        "hard_negative_longest_fraction": float(np.mean(longest)) if longest else None,
        "hard_negative_position_counts": pos_counts,
    }


def teacher_verdict(results: dict[str, dict]) -> dict:
    qwen_b12 = {
        "hellaswag": {"accuracy": 0.495, "mean_margin": -0.0263, "confident_wrong": 0.495},
        "piqa": {"accuracy": 0.675, "mean_margin": 0.1920, "confident_wrong": 0.285},
        "arc_easy": {"accuracy": 0.340, "mean_margin": -1.2170, "confident_wrong": 0.655},
    }
    rows = {}
    acc_pass = 0
    margin_pass = 0
    for bench in BENCHMARKS:
        smol = results[bench]
        qwen = qwen_b12[bench]
        acc_delta = float(smol["accuracy"] - qwen["accuracy"])
        margin_delta = None
        if smol.get("mean_margin_best_wrong_minus_gold_nll") is not None:
            margin_delta = float(smol["mean_margin_best_wrong_minus_gold_nll"] - qwen["mean_margin"])
        confident_delta = None
        if smol.get("confident_wrong_rate") is not None:
            confident_delta = float(smol["confident_wrong_rate"] - qwen["confident_wrong"])
        acc_ok = acc_delta >= 0.05
        margin_ok = margin_delta is not None and margin_delta > 0.0
        acc_pass += int(acc_ok)
        margin_pass += int(margin_ok)
        rows[bench] = {
            "qwen_b12_accuracy": qwen["accuracy"],
            "smollm2_accuracy": float(smol["accuracy"]),
            "accuracy_delta_smollm2_minus_qwen": acc_delta,
            "qwen_b12_mean_margin": qwen["mean_margin"],
            "smollm2_mean_margin": smol.get("mean_margin_best_wrong_minus_gold_nll"),
            "margin_delta_smollm2_minus_qwen": margin_delta,
            "qwen_b12_confident_wrong": qwen["confident_wrong"],
            "smollm2_confident_wrong": smol.get("confident_wrong_rate"),
            "confident_wrong_delta_smollm2_minus_qwen": confident_delta,
            "accuracy_pass_plus_5pp": acc_ok,
            "better_margin_quality": margin_ok,
        }
    verdict = UPGRADE_TEACHER if acc_pass >= 2 and margin_pass >= 2 else MAINTAIN_QWEN
    return {
        "verdict": verdict,
        "required_accuracy_pass_benchmarks": 2,
        "required_margin_quality_benchmarks": 2,
        "accuracy_pass_benchmarks": acc_pass,
        "margin_quality_pass_benchmarks": margin_pass,
        "benchmarks": rows,
    }


def run_teacher_audit(args: argparse.Namespace) -> dict:
    started = time.time()
    set_all_seeds(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)

    results: dict[str, dict] = {}
    details: dict[str, dict] = {}
    for bench in BENCHMARKS:
        examples = load_limited_benchmark(
            bench,
            args.teacher_examples,
            args.benchmark_split,
            args.seed + 5700 + len(bench),
            args.allow_downloads,
        )
        summary = evaluate_teacher_rankings(
            teacher,
            tokenizer,
            examples,
            device,
            args.progress,
            f"{bench}:teacher",
        )
        extras = teacher_extra_metrics(summary["predictions"], args.confident_wrong_threshold)
        summary.update(extras)
        details[bench] = {
            "metadata": {
                "split": args.benchmark_split,
                "train_safe": args.benchmark_split == "train",
                "examples": len(examples),
                "score": "teacher full continuation NLL/token",
            },
            "teacher": summary,
        }
        results[bench] = strip_predictions(summary)

    verdict = teacher_verdict(results)
    payload = {
        "mode": "smollm2_teacher_quality_audit",
        "run": {
            "seed": args.seed,
            "device": str(device),
            "teacher": args.teacher,
            "elapsed_s": round(time.time() - started, 3),
            "benchmark_split": args.benchmark_split,
            "examples_per_benchmark": args.teacher_examples,
        },
        "precommitted_verdict_tokens": {
            "upgrade": UPGRADE_TEACHER,
            "maintain": MAINTAIN_QWEN,
        },
        "teacher_quality": verdict,
        "benchmarks": results,
        "benchmark_details": details if args.save_predictions else {},
        "qwen_b12_reference": {
            "source": "research/work_loop_batch12.md Probe B",
            "hellaswag": {"accuracy": 0.495, "mean_margin": -0.0263, "confident_wrong": 0.495},
            "piqa": {"accuracy": 0.675, "mean_margin": 0.1920, "confident_wrong": 0.285},
            "arc_easy": {"accuracy": 0.340, "mean_margin": -1.2170, "confident_wrong": 0.655},
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "smollm2_teacher_audit.json", payload)
    return payload


def teacher_scores_for_examples(teacher, tokenizer, examples: list[dict], device: torch.device, progress: bool, name: str) -> list[list[float]]:
    summary = evaluate_teacher_rankings(teacher, tokenizer, examples, device, progress, name)
    return [[float(c["nll_per_token"]) for c in pred["choice_scores"]] for pred in summary["predictions"]]


def fmd_verdict(benchmarks: dict[str, dict]) -> dict:
    rows = {}
    pass_count = 0
    marginal_count = 0
    for bench, result in benchmarks.items():
        baseline = float(result["baseline_untrained"]["accuracy"])
        label = float(result["label_ce_trained"]["accuracy"])
        fmd = float(result["fmd_trained"]["accuracy"])
        delta_baseline = fmd - baseline
        delta_label = fmd - label
        passed = delta_baseline >= 0.03 and delta_label >= 0.03
        marginal = delta_baseline > 0.0 and delta_label < 0.03
        pass_count += int(passed)
        marginal_count += int(marginal)
        rows[bench] = {
            "baseline_accuracy": baseline,
            "label_ce_accuracy": label,
            "fmd_accuracy": fmd,
            "delta_fmd_minus_baseline": delta_baseline,
            "delta_fmd_minus_label_ce": delta_label,
            "passes_plus_3pp_over_both": passed,
            "beats_untrained_not_label": marginal,
        }
    if pass_count >= 2:
        verdict = PASS_FMD
        story = "teacher_margins_add_residual_value_on_capable_s0"
    elif marginal_count >= 1:
        verdict = MARGINAL_FMD
        story = "teacher_margins_beat_untrained_but_not_label_only"
    else:
        verdict = FAIL_FMD
        story = "teacher_margins_do_not_add_value_over_label_only"
    return {
        "verdict": verdict,
        "causal_story": story,
        "threshold_over_baseline_and_label_ce": 0.03,
        "required_pass_benchmarks": 2,
        "passed_benchmarks": pass_count,
        "marginal_benchmarks": marginal_count,
        "benchmarks": rows,
    }


def run_fmd(args: argparse.Namespace) -> dict:
    started = time.time()
    set_all_seeds(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    amp_dtype = getattr(torch, args.dtype)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_by_bench, eval_by_bench, split_meta = make_split(
        args.examples_per_benchmark,
        args.train_per_benchmark,
        args.eval_per_benchmark,
        args.benchmark_split,
        args.seed,
        args.allow_downloads,
    )
    model, checkpoint_manifest = load_s0_checkpoint(args.checkpoint, device)
    train_examples = flatten_examples(train_by_bench)
    train_bundle = extract_feature_bundle(
        model, train_examples, args.max_bytes, device, amp_dtype,
        args.feature_batch_choices, args.progress, "fmd:train",
    )
    eval_bundles = {
        bench: extract_feature_bundle(
            model, eval_by_bench[bench], args.max_bytes, device, amp_dtype,
            args.feature_batch_choices, args.progress, f"fmd:eval:{bench}",
        )
        for bench in BENCHMARKS
    }

    baseline_eval = {bench: predictions_from_bundle(bundle, None, "s0_native_byte_nll") for bench, bundle in eval_bundles.items()}
    label_head, label_training, _, label_eval = train_label_head(train_bundle, eval_bundles, args, device)
    del label_head

    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)
    teacher_scores = teacher_scores_for_examples(
        teacher, tokenizer, train_examples, device, args.progress, "fmd:teacher_train_scores",
    )
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    d_model = int(train_bundle.features.shape[1])
    head = ResidualChoiceHead(d_model, args.head_dim, args.head_dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 9100)
    n_examples = len(train_bundle.spans)
    history = []
    head.train()
    for step in range(1, args.train_steps + 1):
        idx = rng.choice(n_examples, size=min(args.batch_examples, n_examples), replace=False)
        logits = head(train_bundle.features.to(device), train_bundle.native_nll.to(device))
        loss, info = grouped_fmd_loss(
            logits,
            train_bundle.spans,
            train_bundle.labels,
            teacher_scores,
            idx,
            args.fmd_margin_scale,
            args.fmd_teacher_correct_only,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % max(1, args.log_every) == 0 or step == args.train_steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "grad_norm": float(grad_norm.detach().cpu().item()),
                **info,
            }
            history.append(row)
            if args.progress:
                print(f"  fmd head step {step}/{args.train_steps}: loss={loss.item():.4f} used={info['used']}", flush=True)

    fmd_eval = {bench: evaluate_head(head, bundle, device) for bench, bundle in eval_bundles.items()}
    benchmark_details = {}
    benchmark_results = {}
    for bench in BENCHMARKS:
        benchmark_details[bench] = {
            "metadata": {
                "split": args.benchmark_split,
                "train_safe": args.benchmark_split == "train",
                "train_examples": len(train_by_bench[bench]),
                "eval_examples": len(eval_by_bench[bench]),
            },
            "baseline_untrained": baseline_eval[bench],
            "label_ce_trained": label_eval[bench],
            "fmd_trained": fmd_eval[bench],
        }
        benchmark_results[bench] = strip_predictions(benchmark_details[bench])

    verdict = fmd_verdict(benchmark_details)
    payload = {
        "mode": "fmd_shadow_288_on_s0",
        "run": {
            "seed": args.seed,
            "device": str(device),
            "checkpoint": args.checkpoint,
            "teacher": args.teacher,
            "elapsed_s": round(time.time() - started, 3),
            "train_examples_total": len(train_examples),
            "eval_examples_total": sum(len(v) for v in eval_by_bench.values()),
            "max_bytes": args.max_bytes,
        },
        "precommitted_verdict_tokens": {
            "pass": PASS_FMD,
            "marginal": MARGINAL_FMD,
            "fail": FAIL_FMD,
        },
        "checkpoint": checkpoint_manifest,
        "split": split_meta,
        "label_training": label_training,
        "fmd_training": {
            "objective": "teacher_margin_gold_vs_hard_wrong_rank_loss",
            "teacher_correct_only": bool(args.fmd_teacher_correct_only),
            "margin_scale": args.fmd_margin_scale,
            "steps": args.train_steps,
            "batch_examples": args.batch_examples,
            "history": history,
        },
        "fmd_on_s0": verdict,
        "benchmarks": benchmark_results,
        "benchmark_details": benchmark_details if args.save_predictions else {},
    }
    write_json(out_dir / "fmd_on_s0.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="W-loop B13 S0/Wide7 capacity harness")
    sub = parser.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--output-dir", default="tmp_work_loop_b13")
        p.add_argument("--seed", type=int, default=20260707)
        p.add_argument("--device", default="auto")
        p.add_argument("--allow-downloads", action="store_true")
        p.add_argument("--benchmark-split", default="train")
        p.add_argument("--progress", action="store_true")
        p.add_argument("--save-predictions", action="store_true")

    def capacity_common(p: argparse.ArgumentParser) -> None:
        common(p)
        p.add_argument("--checkpoint", default="C:/sutra_fast/checkpoints/wide7_scout/s0_best.pt")
        p.add_argument("--checkpoint-label", default="wide7_scout_s0_best")
        p.add_argument("--examples-per-benchmark", type=int, default=144)
        p.add_argument("--train-per-benchmark", type=int, default=96)
        p.add_argument("--eval-per-benchmark", type=int, default=48)
        p.add_argument("--max-bytes", type=int, default=768)
        p.add_argument("--feature-batch-choices", type=int, default=24)
        p.add_argument("--dtype", default="bfloat16")
        p.add_argument("--head-dim", type=int, default=256)
        p.add_argument("--head-dropout", type=float, default=0.0)
        p.add_argument("--train-steps", type=int, default=80)
        p.add_argument("--batch-examples", type=int, default=24)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--weight-decay", type=float, default=0.01)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--log-every", type=int, default=10)
        p.add_argument("--bootstrap-samples", type=int, default=2000)

    p_capacity = sub.add_parser("capacity")
    capacity_common(p_capacity)
    p_capacity.set_defaults(func=run_capacity)

    p_teacher = sub.add_parser("teacher-audit")
    common(p_teacher)
    p_teacher.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-360M")
    p_teacher.add_argument("--teacher-examples", type=int, default=200)
    p_teacher.add_argument("--confident-wrong-threshold", type=float, default=0.05)
    p_teacher.set_defaults(func=run_teacher_audit)

    p_fmd = sub.add_parser("fmd")
    capacity_common(p_fmd)
    p_fmd.add_argument("--teacher", default="HuggingFaceTB/SmolLM2-360M")
    p_fmd.add_argument("--fmd-margin-scale", type=float, default=1.0)
    p_fmd.add_argument("--fmd-teacher-correct-only", action="store_true", default=True)
    p_fmd.set_defaults(func=run_fmd)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    payload = args.func(args)
    verdict = None
    if args.mode == "capacity":
        verdict = payload["scaffold_capacity"]["verdict"]
    elif args.mode == "teacher-audit":
        verdict = payload["teacher_quality"]["verdict"]
    elif args.mode == "fmd":
        verdict = payload["fmd_on_s0"]["verdict"]
    print(json.dumps({"mode": args.mode, "verdict": verdict, "output_dir": args.output_dir}, indent=2))


if __name__ == "__main__":
    main()
