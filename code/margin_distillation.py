"""Functional-margin distillation prototype.

This is the first post-coordinate-inheritance pivot experiment.  It trains a
small byte-autoregressive student on teacher per-continuation NLL differences,
then evaluates benchmark-facing multiple-choice margins from day one.

The prototype is intentionally narrow:
- Frozen semantic codec: bytes -> causal codec patch states.
- Trainable tiny student core/readout: codec patches -> next-patch byte NLL.
- Teacher target: Qwen continuation NLLs over unlabeled shard-derived candidate
  continuations.
- Evaluation: train-safe HellaSwag/PIQA/ARC-Easy forced-choice margins using
  the same record semantics as coordinate_inheritance.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from coordinate_inheritance import (  # noqa: E402
    DEFAULT_CODEC,
    DEFAULT_DATA_DIR,
    DEFAULT_TEACHER,
    ScoredCompletion,
    bootstrap_accuracy_delta,
    bootstrap_scalar_delta,
    build_choice_prediction_record,
    choose_device,
    ensure_offline,
    evaluate_teacher_rankings,
    load_limited_benchmark,
    load_teacher,
    score_teacher_completion,
    strip_predictions,
    summarize_prediction_records,
    write_json,
)
from s0_architecture import ByteDecoder, GlobalReasoner, RMSNorm, S0Config  # noqa: E402
from tier3_brainseed_chart_probe import ByteShardSampler, load_codec, load_tokenizer  # noqa: E402


DEFAULT_OUTPUT_DIR = "tmp_margin_distillation_b11/smoke"

PASS_TOKEN = "PASS_MARGIN_SMOKE"
FAIL_TOKEN = "FAIL_MARGIN_SMOKE"
MARGINAL_TOKEN = "MARGINAL_MARGIN"


@dataclass
class MarginExample:
    """One unlabeled context with teacher-scored candidate continuations."""

    context: str
    candidates: list[str]
    teacher_nlls: list[float]
    source: str

    def to_json(self) -> dict:
        return {
            "context_preview": self.context[:120],
            "candidates_preview": [c[:120] for c in self.candidates],
            "teacher_nlls": self.teacher_nlls,
            "source": self.source,
        }


@dataclass
class MarginStudentConfig:
    """Tiny smoke student. This is a prototype, not the final 121M S0."""

    codec_dim: int = 256
    d_model: int = 256
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4
    ffn_mult: float = 2.0
    patch_size: int = 4
    max_bytes: int = 768
    decoder_dim: int = 256
    decoder_layers: int = 1
    decoder_heads: int = 4
    dropout: float = 0.0

    @property
    def max_patches(self) -> int:
        return max(2, math.ceil(self.max_bytes / self.patch_size))


class MarginStudent(nn.Module):
    """Frozen-codec byte LM used as the functional-margin student."""

    def __init__(self, codec, cfg: MarginStudentConfig):
        super().__init__()
        self.codec = codec
        self.cfg = cfg
        for p in self.codec.parameters():
            p.requires_grad_(False)
        self.codec.eval()

        s0_cfg = S0Config(
            byte_dim=cfg.codec_dim,
            patch_size=cfg.patch_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            ffn_mult=cfg.ffn_mult,
            max_seq_len=cfg.max_patches,
            decoder_dim=cfg.decoder_dim,
            decoder_layers=cfg.decoder_layers,
            decoder_heads=cfg.decoder_heads,
            dropout=cfg.dropout,
        )
        self.input_norm = RMSNorm(cfg.codec_dim)
        self.input_proj = nn.Linear(cfg.codec_dim, cfg.d_model, bias=False)
        self.reasoner = GlobalReasoner(s0_cfg)
        self.decoder = ByteDecoder(s0_cfg)

    def train(self, mode: bool = True):
        super().train(mode)
        self.codec.eval()
        return self

    def forward(self, byte_ids: torch.Tensor) -> dict:
        if byte_ids.shape[1] % self.cfg.patch_size != 0:
            raise ValueError("byte_ids length must be divisible by patch_size")
        with torch.no_grad():
            patch_hidden = self.codec.encoder.get_patch_states(byte_ids)
        patch_states = self.input_proj(self.input_norm(patch_hidden.float()))
        hidden = self.reasoner(patch_states)

        bsz, n_patches, _ = hidden.shape
        if n_patches < 2:
            raise ValueError("need at least two patches to score next-patch bytes")
        target_bytes = byte_ids.reshape(bsz, n_patches, self.cfg.patch_size)[:, 1:]
        pred_hidden = hidden[:, :-1]
        prev_padded = F.pad(pred_hidden, (0, 0, 1, 0))[:, : pred_hidden.shape[1]]
        nearby = torch.stack([prev_padded, pred_hidden], dim=2)
        logits = self.decoder(pred_hidden, target_bytes, nearby)
        return {"logits": logits, "hidden": hidden}

    def score_prepared(self, byte_ids: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        """Return per-continuation byte NLLs. Lower is better."""

        out = self.forward(byte_ids)
        logits = out["logits"].float()
        bsz, n_minus_1, psize, vocab = logits.shape
        targets = byte_ids.reshape(bsz, n_minus_1 + 1, psize)[:, 1:]
        per_byte = F.cross_entropy(
            logits.reshape(-1, vocab),
            targets.reshape(-1),
            reduction="none",
        ).reshape(bsz, n_minus_1, psize)

        scores: list[torch.Tensor] = []
        for i, mask in enumerate(masks):
            mask = mask.to(device=byte_ids.device)
            if int(mask.sum().item()) == 0:
                scores.append(per_byte[i].mean() * 0.0 + 1e6)
            else:
                scores.append(per_byte[i][mask].mean())
        return torch.stack(scores)


def set_all_seeds(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def normalize_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\ufffd", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def candidate_source_slices(
    sampler: ByteShardSampler,
    rng: np.random.Generator,
    needed: int,
    seq_len: int,
    min_chars: int,
) -> list[str]:
    texts: list[str] = []
    attempts = 0
    while len(texts) < needed and attempts < needed * 20:
        attempts += 1
        rows = sampler.sample(1, rng)
        raw = bytes(rows[0].cpu().numpy().astype(np.uint8).tolist())
        text = normalize_text(raw)
        if len(text) >= min_chars:
            texts.append(text)
    if len(texts) < needed:
        raise RuntimeError(f"collected only {len(texts)} usable text slices for margin targets")
    return texts


def make_unlabeled_candidate_sets(
    sampler: ByteShardSampler,
    rng: np.random.Generator,
    count: int,
    n_candidates: int,
    context_chars: int,
    continuation_chars: int,
    seq_len: int,
) -> list[tuple[str, list[str], str]]:
    """Build unlabeled examples: one context, several natural continuation candidates."""

    min_chars = context_chars + continuation_chars + 16
    pool = candidate_source_slices(
        sampler,
        rng,
        needed=max(count * (n_candidates + 2), n_candidates + 8),
        seq_len=seq_len,
        min_chars=min_chars,
    )
    examples: list[tuple[str, list[str], str]] = []
    cursor = 0
    for idx in range(count):
        base = pool[cursor % len(pool)]
        cursor += 1
        max_start = max(0, len(base) - min_chars)
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
        context = base[start : start + context_chars].strip()
        true_cont = base[start + context_chars : start + context_chars + continuation_chars].strip()
        if not context or not true_cont:
            continue
        candidates = [true_cont]
        while len(candidates) < n_candidates:
            other = pool[cursor % len(pool)]
            cursor += 1
            other_max = max(0, len(other) - continuation_chars - 1)
            other_start = int(rng.integers(0, other_max + 1)) if other_max else 0
            cand = other[other_start : other_start + continuation_chars].strip()
            if cand and cand not in candidates:
                candidates.append(cand)
        order = list(range(len(candidates)))
        rng.shuffle(order)
        shuffled = [candidates[i] for i in order]
        examples.append((context, shuffled, f"shard_text:{idx}"))
    return examples


@torch.no_grad()
def attach_teacher_targets(
    raw_examples: list[tuple[str, list[str], str]],
    teacher,
    tokenizer,
    device: torch.device,
    min_teacher_spread: float,
    progress: bool,
) -> list[MarginExample]:
    examples: list[MarginExample] = []
    for i, (context, candidates, source) in enumerate(raw_examples):
        scored = [score_teacher_completion(teacher, tokenizer, context, candidate, device) for candidate in candidates]
        nlls = [s.nll_per_token for s in scored]
        finite = [x for x in nlls if math.isfinite(x)]
        if len(finite) == len(candidates) and (max(finite) - min(finite)) >= min_teacher_spread:
            examples.append(MarginExample(context, candidates, [float(x) for x in nlls], source))
        if progress and (i + 1) % 10 == 0:
            print(f"  teacher targets {i + 1}/{len(raw_examples)} kept={len(examples)}", flush=True)
    return examples


def prepare_scoring_batch(
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
        choice_raw = choice.encode("utf-8", errors="replace")
        if not choice_raw:
            choice_raw = b" "
        if len(choice_raw) >= max_bytes - patch_size:
            choice_raw = choice_raw[: max_bytes // 2]
        keep_ctx = max(patch_size, max_bytes - len(choice_raw))
        if len(ctx_raw) > keep_ctx:
            ctx_raw = ctx_raw[-keep_ctx:]
        if len(ctx_raw) < patch_size:
            ctx_raw = (b" " * (patch_size - len(ctx_raw))) + ctx_raw
        raw = ctx_raw + choice_raw
        raw = raw[:max_bytes]
        usable_len = len(raw)
        padded_len = max(patch_size * 2, int(math.ceil(usable_len / patch_size) * patch_size))
        max_len = max(max_len, padded_len)
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int64).copy()
        arr[arr == 0xFF] = 32
        mask = np.zeros(padded_len // patch_size - 1, dtype=np.bool_)
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


def score_choice_group(
    student: MarginStudent,
    context: str,
    choices: list[str],
    max_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    contexts = [context for _ in choices]
    byte_ids, masks = prepare_scoring_batch(contexts, choices, max_bytes, student.cfg.patch_size, device)
    return student.score_prepared(byte_ids, masks)


def score_margin_examples(
    student: MarginStudent,
    examples: list[MarginExample],
    max_bytes: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[slice]]:
    contexts: list[str] = []
    choices: list[str] = []
    spans: list[slice] = []
    cursor = 0
    for ex in examples:
        contexts.extend([ex.context for _ in ex.candidates])
        choices.extend(ex.candidates)
        spans.append(slice(cursor, cursor + len(ex.candidates)))
        cursor += len(ex.candidates)
    byte_ids, masks = prepare_scoring_batch(contexts, choices, max_bytes, student.cfg.patch_size, device)
    scores = student.score_prepared(byte_ids, masks)
    return scores, spans


def pairwise_margin_loss(
    student_scores: torch.Tensor,
    teacher_nlls: list[float],
    min_margin: float,
    margin_clip: float,
    rank_temperature: float,
    margin_temperature: float,
    lambda_margin_mse: float,
) -> tuple[torch.Tensor, dict]:
    teacher = torch.tensor(teacher_nlls, dtype=student_scores.dtype, device=student_scores.device)
    n = int(teacher.numel())
    if n < 2:
        z = student_scores.sum() * 0.0
        return z, {"pairs": 0, "rank_loss": 0.0, "margin_loss": 0.0}

    pair_i, pair_j = torch.triu_indices(n, n, offset=1, device=student_scores.device)
    target = teacher[pair_j] - teacher[pair_i]
    pred = student_scores[pair_j] - student_scores[pair_i]
    mask = target.abs() >= min_margin
    if int(mask.sum().item()) == 0:
        mask = torch.ones_like(target, dtype=torch.bool)
    target = target[mask].clamp(min=-margin_clip, max=margin_clip)
    pred = pred[mask]
    sign = target.sign()
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    weight = target.abs().clamp(min=min_margin, max=margin_clip).detach()
    rank_loss = (F.softplus(-sign * pred / rank_temperature) * weight).mean()
    margin_loss = F.smooth_l1_loss(pred / margin_temperature, target / margin_temperature)
    loss = rank_loss + lambda_margin_mse * margin_loss
    return loss, {
        "pairs": int(mask.sum().item()),
        "rank_loss": float(rank_loss.detach().item()),
        "margin_loss": float(margin_loss.detach().item()),
    }


def train_margin_student(
    student: MarginStudent,
    examples: list[MarginExample],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict:
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    student.train()
    history: list[dict] = []
    for step in range(1, args.train_steps + 1):
        idx = rng.choice(len(examples), size=min(args.train_batch_examples, len(examples)), replace=False)
        batch = [examples[int(i)] for i in idx]
        opt.zero_grad(set_to_none=True)
        scores, spans = score_margin_examples(student, batch, args.max_bytes, device)
        losses = []
        pair_count = 0
        rank_loss = 0.0
        margin_loss = 0.0
        for ex, span in zip(batch, spans):
            loss, stats = pairwise_margin_loss(
                scores[span],
                ex.teacher_nlls,
                args.min_teacher_margin,
                args.margin_clip,
                args.rank_temperature,
                args.margin_temperature,
                args.lambda_margin_mse,
            )
            losses.append(loss)
            pair_count += stats["pairs"]
            rank_loss += stats["rank_loss"]
            margin_loss += stats["margin_loss"]
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        entry = {
            "step": step,
            "loss": float(loss.detach().item()),
            "rank_loss_mean": rank_loss / max(1, len(losses)),
            "margin_loss_mean": margin_loss / max(1, len(losses)),
            "pairs": pair_count,
            "grad_norm": float(grad_norm.detach().item()),
        }
        history.append(entry)
        if args.progress:
            print(
                f"  margin step {step}/{args.train_steps}: "
                f"loss={entry['loss']:.4f} pairs={pair_count} grad={entry['grad_norm']:.3f}",
                flush=True,
            )
    student.eval()
    return {
        "steps": args.train_steps,
        "batch_examples": args.train_batch_examples,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "history": history,
    }


@torch.no_grad()
def evaluate_student_rankings(
    student: MarginStudent,
    examples: list[dict],
    teacher_predictions: list[dict] | None,
    max_bytes: int,
    device: torch.device,
    progress: bool,
    name: str,
) -> dict:
    student.eval()
    predictions = []
    started = time.time()
    for i, ex in enumerate(examples):
        scores = score_choice_group(student, ex["context"], ex["choices"], max_bytes, device)
        scored = [ScoredCompletion(float(s.detach().item()), 1) for s in scores]
        teacher_record = teacher_predictions[i] if teacher_predictions is not None else None
        predictions.append(build_choice_prediction_record(ex, scored, teacher_record))
        if progress and (i + 1) % 25 == 0:
            acc = float(np.mean([int(p["correct"]) for p in predictions]))
            print(f"  [{name}] {i + 1}/{len(examples)} acc={acc:.3f}", flush=True)
    summary = summarize_prediction_records(predictions)
    summary["elapsed_s"] = round(time.time() - started, 3)
    summary["predictions"] = predictions
    return summary


def margin_smoke_verdict(benchmarks: dict) -> dict:
    rows = {}
    pass_count = 0
    marginal_count = 0
    flat_or_negative = 0
    for bench, result in benchmarks.items():
        baseline = result["baseline_untrained"]["accuracy"]
        trained = result["margin_trained"]["accuracy"]
        delta = float(trained - baseline)
        pass_bench = delta >= 0.03
        marginal = 0.01 <= delta < 0.03
        pass_count += int(pass_bench)
        marginal_count += int(marginal)
        flat_or_negative += int(delta <= 0.0)
        rows[bench] = {
            "baseline_accuracy": baseline,
            "trained_accuracy": trained,
            "delta_accuracy": delta,
            "passes_plus_3pp": pass_bench,
            "marginal_plus_1_to_3pp": marginal,
        }
    if pass_count >= 2:
        verdict = PASS_TOKEN
        story = "functional_margin_signal_present"
    elif flat_or_negative == len(rows):
        verdict = FAIL_TOKEN
        story = "flat_or_negative_functional_signal"
    elif marginal_count > 0:
        verdict = MARGINAL_TOKEN
        story = "ambiguous_small_functional_signal"
    else:
        verdict = FAIL_TOKEN
        story = "insufficient_functional_margin_improvement"
    return {
        "verdict": verdict,
        "causal_story": story,
        "required_pass_benchmarks": 2,
        "threshold_accuracy_delta": 0.03,
        "marginal_band": [0.01, 0.03],
        "passed_benchmarks": pass_count,
        "benchmarks": rows,
    }


def build_student(codec, args: argparse.Namespace) -> MarginStudent:
    cfg = MarginStudentConfig(
        codec_dim=int(codec.cfg.codec_dim),
        d_model=args.student_dim,
        n_layers=args.student_layers,
        n_heads=args.student_heads,
        n_kv_heads=args.student_kv_heads,
        ffn_mult=args.student_ffn_mult,
        patch_size=int(codec.cfg.patch_size),
        max_bytes=args.max_bytes,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
    )
    return MarginStudent(codec, cfg)


def run_smoke(args: argparse.Namespace) -> dict:
    started = time.time()
    rng = set_all_seeds(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)
    sampler = ByteShardSampler(args.data_dir, args.train_seq_len)

    raw_count = max(args.train_examples * 3, args.train_examples + 20)
    raw_sets = make_unlabeled_candidate_sets(
        sampler,
        rng,
        raw_count,
        args.candidates,
        args.context_chars,
        args.continuation_chars,
        args.train_seq_len,
    )
    margin_examples = attach_teacher_targets(
        raw_sets,
        teacher,
        tokenizer,
        device,
        args.min_teacher_spread,
        args.progress,
    )[: args.train_examples]
    if len(margin_examples) < args.train_examples:
        raise RuntimeError(f"only {len(margin_examples)} teacher-margin examples survived filtering")

    student = build_student(codec, args).to(device)
    baseline_student = copy.deepcopy(student).to(device).eval()

    benchmark_results: dict[str, dict] = {}
    benchmark_details: dict[str, dict] = {}
    benchmark_examples: dict[str, list[dict]] = {}
    teacher_benchmark_predictions: dict[str, list[dict]] = {}
    for bench in args.benchmarks:
        examples = load_limited_benchmark(
            bench,
            args.benchmark_examples,
            args.benchmark_split,
            args.seed + 1700 + len(bench),
            args.allow_downloads,
        )
        benchmark_examples[bench] = examples
        teacher_result = evaluate_teacher_rankings(
            teacher,
            tokenizer,
            examples,
            device,
            args.progress,
            f"{bench}:qwen_teacher_full",
        )
        teacher_benchmark_predictions[bench] = teacher_result["predictions"]
        base_result = evaluate_student_rankings(
            baseline_student,
            examples,
            teacher_benchmark_predictions[bench],
            args.max_bytes,
            device,
            args.progress,
            f"{bench}:baseline_untrained",
        )
        benchmark_details[bench] = {
            "metadata": {
                "split": args.benchmark_split,
                "train_safe": args.benchmark_split == "train",
                "n_examples": len(examples),
                "score": "student byte NLL per continuation",
            },
            "qwen_teacher_full": teacher_result,
            "baseline_untrained": base_result,
        }

    training = train_margin_student(student, margin_examples, args, device, rng)

    for bench, examples in benchmark_examples.items():
        trained_result = evaluate_student_rankings(
            student,
            examples,
            teacher_benchmark_predictions[bench],
            args.max_bytes,
            device,
            args.progress,
            f"{bench}:margin_trained",
        )
        benchmark_details[bench]["margin_trained"] = trained_result
        base_preds = benchmark_details[bench]["baseline_untrained"]["predictions"]
        trained_preds = benchmark_details[bench]["margin_trained"]["predictions"]
        benchmark_details[bench]["delta_margin_trained_minus_baseline"] = {
            "accuracy": bootstrap_accuracy_delta(
                trained_preds,
                base_preds,
                args.bootstrap_samples,
                args.seed + 2000,
            ),
            "margin_best_wrong_minus_gold_nll": bootstrap_scalar_delta(
                trained_preds,
                base_preds,
                "margin_best_wrong_minus_gold_nll",
                args.bootstrap_samples,
                args.seed + 2001,
            ),
        }
        benchmark_results[bench] = strip_predictions(benchmark_details[bench])

    verdict = margin_smoke_verdict(benchmark_details)
    payload = {
        "mode": "functional_margin_distillation_smoke",
        "run": {
            "seed": args.seed,
            "device": str(device),
            "teacher": args.teacher,
            "codec_checkpoint": args.codec_checkpoint,
            "data_dir": args.data_dir,
            "elapsed_s": round(time.time() - started, 3),
            "train_examples": len(margin_examples),
            "train_steps": args.train_steps,
            "benchmark_examples": args.benchmark_examples,
            "benchmark_split": args.benchmark_split,
            "max_bytes": args.max_bytes,
        },
        "precommitted_verdict_tokens": {
            "pass": PASS_TOKEN,
            "fail": FAIL_TOKEN,
            "marginal": MARGINAL_TOKEN,
        },
        "loss_design": {
            "teacher_signal": "Qwen per-continuation NLL/token differences",
            "student_signal": "student byte NLL/byte differences",
            "pairwise_rank_loss": "weighted RankNet softplus on teacher-preferred pairs",
            "margin_loss": "SmoothL1 on pairwise student-vs-teacher NLL differences",
            "min_teacher_margin": args.min_teacher_margin,
            "margin_clip": args.margin_clip,
            "lambda_margin_mse": args.lambda_margin_mse,
        },
        "student_config": asdict(student.cfg),
        "codec": codec_manifest,
        "training": training,
        "train_targets": {
            "n": len(margin_examples),
            "examples": [ex.to_json() for ex in margin_examples[: min(10, len(margin_examples))]],
        },
        "functional_margin_smoke": verdict,
        "benchmarks": benchmark_results,
        "benchmark_details": benchmark_details if args.save_predictions else {},
        "limitations": [
            "The smoke student is a tiny prototype, not the final 121M S0 checkpoint.",
            "The codec is frozen; only the margin student core/readout is trained.",
            "Training targets come from unlabeled shard text, not labeled MCQ data.",
            "Evaluation uses train-safe benchmark splits to preserve the shadow-test role.",
            "Scores are byte-NLL continuation scores, so margins are benchmark-facing but not public validation evidence.",
        ],
    }
    write_json(out_dir / "functional_margin_distillation_smoke.json", payload)
    torch.save(
        {
            "student_state_dict": student.cpu().state_dict(),
            "student_config": asdict(student.cfg),
            "codec_checkpoint": args.codec_checkpoint,
            "teacher": args.teacher,
            "training": training,
        },
        out_dir / "margin_student.pt",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke"], default="smoke")
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")

    parser.add_argument("--train-examples", type=int, default=50)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--train-batch-examples", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--context-chars", type=int, default=320)
    parser.add_argument("--continuation-chars", type=int, default=96)
    parser.add_argument("--train-seq-len", type=int, default=2048)
    parser.add_argument("--min-teacher-spread", type=float, default=0.02)

    parser.add_argument("--student-dim", type=int, default=256)
    parser.add_argument("--student-layers", type=int, default=2)
    parser.add_argument("--student-heads", type=int, default=4)
    parser.add_argument("--student-kv-heads", type=int, default=4)
    parser.add_argument("--student-ffn-mult", type=float, default=2.0)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--decoder-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-bytes", type=int, default=768)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-teacher-margin", type=float, default=0.01)
    parser.add_argument("--margin-clip", type=float, default=2.0)
    parser.add_argument("--rank-temperature", type=float, default=0.25)
    parser.add_argument("--margin-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-margin-mse", type=float, default=0.25)

    parser.add_argument("--benchmarks", nargs="+", choices=["hellaswag", "piqa", "arc_easy"], default=["hellaswag", "piqa", "arc_easy"])
    parser.add_argument("--benchmark-examples", type=int, default=50)
    parser.add_argument("--benchmark-split", choices=["train", "validation"], default="train")
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(args)
    print(
        json.dumps(
            {
                "mode": payload["mode"],
                "functional_margin_smoke": payload["functional_margin_smoke"],
                "benchmarks": payload["benchmarks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
