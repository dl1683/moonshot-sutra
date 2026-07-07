"""W-loop B14 SmolLM2 mechanism-control harness.

Runs the terminal Eklavya control on SmolLM2-135M with forced-choice
continuation NLL scoring, LoRA label-only training, single-teacher KD, oracle
routing, non-oracle disagreement routing, and random routing control.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from coordinate_inheritance import (  # noqa: E402
    ScoredCompletion,
    bootstrap_accuracy_delta,
    bootstrap_scalar_delta,
    build_choice_prediction_record,
    choose_device,
    ensure_offline,
    load_limited_benchmark,
    strip_predictions,
    summarize_prediction_records,
    write_json,
)

BENCHMARKS = ("hellaswag", "piqa", "arc_easy")
STUDENT = "HuggingFaceTB/SmolLM2-135M"
SMOL_TEACHER = "HuggingFaceTB/SmolLM2-360M"
QWEN_TEACHER = "Qwen/Qwen3-0.6B"

PASS_LABEL_ONLY = "PASS_LABEL_ONLY"
MARGINAL_LABEL_ONLY = "MARGINAL_LABEL_ONLY"
FLAT_LABEL_ONLY = "FLAT_LABEL_ONLY"
PASS_SINGLE_TEACHER = "PASS_SINGLE_TEACHER"
MARGINAL_SINGLE_TEACHER = "MARGINAL_SINGLE_TEACHER"
FAIL_SINGLE_TEACHER = "FAIL_SINGLE_TEACHER"
PASS_EKLAVYA_MECHANISM = "PASS_EKLAVYA_MECHANISM"
MARGINAL_EKLAVYA = "MARGINAL_EKLAVYA"
FAIL_EKLAVYA_MECHANISM = "FAIL_EKLAVYA_MECHANISM"


@dataclass
class PreparedChoiceSet:
    examples: list[dict[str, Any]]
    flat_input_ids: list[list[int]]
    flat_context_lens: list[int]
    spans: list[tuple[int, int]]
    labels: list[int]

    @property
    def n_examples(self) -> int:
        return len(self.examples)


def set_all_seeds(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def dtype_from_name(name: str, device: torch.device):
    if name == "auto":
        return "auto"
    if device.type != "cuda":
        return torch.float32
    return getattr(torch, name)


def finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_lm(name: str, device: torch.device, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype_from_name(dtype_name, device),
    )
    model.to(device)
    model.config.use_cache = False
    return model


def model_manifest(model, name: str, tokenizer) -> dict:
    return {
        "name": name,
        "class": model.__class__.__name__,
        "params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "vocab_size": int(getattr(model.config, "vocab_size", -1)),
        "hidden_size": int(getattr(model.config, "hidden_size", -1)),
        "num_hidden_layers": int(getattr(model.config, "num_hidden_layers", -1)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def make_split(train_per_benchmark: int, eval_per_benchmark: int, split: str, seed: int, allow_downloads: bool):
    train, evals, meta = {}, {}, {}
    needed = train_per_benchmark + eval_per_benchmark
    for bench in BENCHMARKS:
        examples = load_limited_benchmark(bench, needed, split, seed + 1700 + len(bench), allow_downloads)
        tr = examples[:train_per_benchmark]
        ev = examples[train_per_benchmark:train_per_benchmark + eval_per_benchmark]
        for row in tr + ev:
            row["benchmark"] = bench
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
    rows = []
    for bench in BENCHMARKS:
        rows.extend(examples_by_bench[bench])
    return rows


def zero_shot_examples(count_per_benchmark: int, split: str, seed: int, allow_downloads: bool):
    out = {}
    for bench in BENCHMARKS:
        rows = load_limited_benchmark(bench, count_per_benchmark, split, seed + 5700 + len(bench), allow_downloads)
        for row in rows:
            row["benchmark"] = bench
        out[bench] = rows
    return out


def prepare_choice_set(tokenizer, examples: list[dict], max_length: int) -> PreparedChoiceSet:
    flat_input_ids, flat_context_lens, spans, labels = [], [], [], []
    cursor = 0
    for ex in examples:
        ctx_ids = tokenizer(ex["context"], add_special_tokens=False).input_ids
        for choice in ex["choices"]:
            full_ids = tokenizer(ex["context"] + choice, add_special_tokens=False).input_ids
            ctx_len = len(ctx_ids)
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                ctx_len = max(0, ctx_len - overflow)
            if len(full_ids) < 2:
                full_ids = full_ids + [tokenizer.eos_token_id]
            flat_input_ids.append([int(x) for x in full_ids])
            flat_context_lens.append(int(min(ctx_len, max(0, len(full_ids) - 1))))
        spans.append((cursor, cursor + len(ex["choices"])))
        labels.append(int(ex["label"]))
        cursor += len(ex["choices"])
    return PreparedChoiceSet(examples, flat_input_ids, flat_context_lens, spans, labels)


def build_padded_batch(rows: list[list[int]], context_lens: list[int], pad_token_id: int, device: torch.device):
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((len(rows), max_len), dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for i, (row, ctx_len) in enumerate(zip(rows, context_lens)):
        n = len(row)
        input_ids[i, :n] = torch.tensor(row, dtype=torch.long)
        attention[i, :n] = 1
        labels[i, :n] = input_ids[i, :n]
        start = min(int(ctx_len), max(0, n - 1))
        labels[i, :start] = -100
    return input_ids.to(device), attention.to(device), labels.to(device)


def choice_nlls_from_rows(model, rows: list[list[int]], context_lens: list[int], pad_token_id: int, device: torch.device):
    input_ids, attention, labels = build_padded_batch(rows, context_lens, pad_token_id, device)
    out = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    loss_flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    token_counts = mask.sum(dim=1).clamp_min(1)
    nll = (loss_flat * mask.float()).sum(dim=1) / token_counts.float()
    no_tokens = mask.sum(dim=1).eq(0)
    if bool(no_tokens.any().item()):
        nll = torch.where(no_tokens, torch.full_like(nll, 1e6), nll)
    return nll, token_counts


def score_prepared_flat(model, prepared: PreparedChoiceSet, pad_token_id: int, device: torch.device, batch_choices: int, progress: bool, name: str):
    nll_parts, count_parts = [], []
    for start in range(0, len(prepared.flat_input_ids), batch_choices):
        end = min(len(prepared.flat_input_ids), start + batch_choices)
        nll, counts = choice_nlls_from_rows(
            model,
            prepared.flat_input_ids[start:end],
            prepared.flat_context_lens[start:end],
            pad_token_id,
            device,
        )
        nll_parts.append(nll.detach().cpu())
        count_parts.append(counts.detach().cpu())
        if progress and (end == len(prepared.flat_input_ids) or end % max(batch_choices * 4, 64) == 0):
            print(f"  [{name}] choices {end}/{len(prepared.flat_input_ids)}", flush=True)
    return torch.cat(nll_parts), torch.cat(count_parts)


def score_prepared_indices(model, prepared: PreparedChoiceSet, indices: np.ndarray, pad_token_id: int, device: torch.device):
    rows, context_lens, local_spans = [], [], []
    cursor = 0
    for idx in indices:
        start, end = prepared.spans[int(idx)]
        rows.extend(prepared.flat_input_ids[start:end])
        context_lens.extend(prepared.flat_context_lens[start:end])
        local_spans.append((cursor, cursor + (end - start)))
        cursor += end - start
    flat_nll, _ = choice_nlls_from_rows(model, rows, context_lens, pad_token_id, device)
    return [flat_nll[start:end] for start, end in local_spans]


def predictions_from_nlls(prepared: PreparedChoiceSet, flat_nlls: torch.Tensor, token_counts: torch.Tensor, teacher_predictions=None):
    predictions = []
    for i, ex in enumerate(prepared.examples):
        start, end = prepared.spans[i]
        scored = [
            ScoredCompletion(float(flat_nlls[j].item()) * int(token_counts[j].item()), int(token_counts[j].item()))
            for j in range(start, end)
        ]
        teacher_record = teacher_predictions[i] if teacher_predictions is not None else None
        predictions.append(build_choice_prediction_record(ex, scored, teacher_record))
    return predictions


@torch.no_grad()
def evaluate_prepared(model, prepared: PreparedChoiceSet, pad_token_id: int, device: torch.device, batch_choices: int, progress: bool, name: str, teacher_predictions=None):
    model.eval()
    started = time.time()
    flat_nlls, token_counts = score_prepared_flat(model, prepared, pad_token_id, device, batch_choices, progress, name)
    predictions = predictions_from_nlls(prepared, flat_nlls, token_counts, teacher_predictions)
    summary = summarize_prediction_records(predictions)
    summary["elapsed_s"] = round(time.time() - started, 3)
    summary["score"] = "continuation_nll_per_token"
    summary["predictions"] = predictions
    return summary


def evaluate_by_benchmark(model, tokenizer, examples_by_bench: dict[str, list[dict]], max_length: int, device: torch.device, batch_choices: int, progress: bool, prefix: str):
    results = {}
    for bench in BENCHMARKS:
        prepared = prepare_choice_set(tokenizer, examples_by_bench[bench], max_length)
        results[bench] = evaluate_prepared(
            model,
            prepared,
            int(tokenizer.pad_token_id),
            device,
            batch_choices,
            progress,
            f"{prefix}:{bench}",
        )
    return results


def softmax_from_nlls(nlls: list[float], temperature: float) -> list[float]:
    arr = np.asarray(nlls, dtype=np.float64)
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return [1.0 / len(arr) for _ in arr]
    arr = np.where(finite_mask, arr, np.nanmax(arr[finite_mask]) + 100.0)
    logits = -arr / max(1e-6, float(temperature))
    logits -= float(np.max(logits))
    probs = np.exp(logits)
    probs /= max(float(probs.sum()), 1e-12)
    return [float(x) for x in probs]


def nlls_from_prediction(pred: dict) -> list[float]:
    return [float(c["nll_per_token"]) for c in pred["choice_scores"]]


def distribution_stats(probs: list[float]) -> dict:
    arr = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-arr)
    top = float(arr[order[0]]) if len(order) else 0.0
    second = float(arr[order[1]]) if len(order) > 1 else 0.0
    entropy = float(-(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum())
    return {"top_prob": top, "top2_margin": top - second, "entropy": entropy, "pred": int(order[0]) if len(order) else -1}


def teacher_records_to_targets(records: list[dict], temperature: float) -> list[dict]:
    targets = []
    for pred in records:
        nlls = nlls_from_prediction(pred)
        probs = softmax_from_nlls(nlls, temperature)
        targets.append({"nlls": nlls, "probs": probs, **distribution_stats(probs)})
    return targets


def blend_probs(a: list[float], b: list[float], wa: float) -> list[float]:
    wa = max(0.0, min(1.0, float(wa)))
    arr = wa * np.asarray(a, dtype=np.float64) + (1.0 - wa) * np.asarray(b, dtype=np.float64)
    arr /= max(float(arr.sum()), 1e-12)
    return [float(x) for x in arr]


def build_routed_targets(smol_records: list[dict], qwen_records: list[dict], labels: list[int], route: str, temperature: float, seed: int):
    smol = teacher_records_to_targets(smol_records, temperature)
    qwen = teacher_records_to_targets(qwen_records, temperature)
    rng = np.random.default_rng(seed)
    targets = []
    counts = {"smol": 0, "qwen": 0, "mixture": 0, "agree": 0, "disagree": 0, "oracle_smol_correct": 0, "oracle_qwen_correct": 0, "oracle_neither_top1_correct": 0}
    for s, q, label in zip(smol, qwen, labels):
        s_pred, q_pred = int(s["pred"]), int(q["pred"])
        disagree = s_pred != q_pred
        counts["disagree" if disagree else "agree"] += 1
        chosen, probs = "smol", s["probs"]
        if route == "single_smol":
            chosen, probs = "smol", s["probs"]
        elif route == "oracle":
            if disagree and s_pred == label and q_pred != label:
                chosen, probs = "smol", s["probs"]
                counts["oracle_smol_correct"] += 1
            elif disagree and q_pred == label and s_pred != label:
                chosen, probs = "qwen", q["probs"]
                counts["oracle_qwen_correct"] += 1
            elif disagree:
                counts["oracle_neither_top1_correct"] += 1
                chosen, probs = ("qwen", q["probs"]) if q["probs"][label] > s["probs"][label] else ("smol", s["probs"])
        elif route == "confidence":
            if disagree and q["top_prob"] > s["top_prob"]:
                chosen, probs = "qwen", q["probs"]
        elif route == "confidence_mixture":
            if disagree:
                s_conf, q_conf = float(s["top2_margin"]), float(q["top2_margin"])
                weight = math.exp(s_conf) / (math.exp(s_conf) + math.exp(q_conf))
                chosen, probs = "mixture", blend_probs(s["probs"], q["probs"], weight)
        elif route == "random":
            if disagree and bool(rng.integers(0, 2)):
                chosen, probs = "qwen", q["probs"]
        else:
            raise ValueError(route)
        counts[chosen] = counts.get(chosen, 0) + 1
        targets.append(probs)
    meta = {"route": route, "temperature": temperature, "counts": counts, "disagreement_rate": counts["disagree"] / max(1, len(labels))}
    return targets, meta


def summarize_teacher_pair(smol_records: list[dict], qwen_records: list[dict], labels: list[int], temperature: float) -> dict:
    smol = teacher_records_to_targets(smol_records, temperature)
    qwen = teacher_records_to_targets(qwen_records, temperature)
    rows = []
    for s, q, label in zip(smol, qwen, labels):
        s_ok, q_ok = int(s["pred"] == label), int(q["pred"] == label)
        disagree = int(s["pred"] != q["pred"])
        rows.append({"smol_correct": s_ok, "qwen_correct": q_ok, "disagree": disagree, "useful_disagreement": int(disagree and (s_ok or q_ok)), "oracle_correct": int(s_ok or q_ok)})
    smol_acc = float(np.mean([r["smol_correct"] for r in rows])) if rows else 0.0
    qwen_acc = float(np.mean([r["qwen_correct"] for r in rows])) if rows else 0.0
    oracle = float(np.mean([r["oracle_correct"] for r in rows])) if rows else 0.0
    return {
        "n_examples": len(rows),
        "smol_accuracy": smol_acc,
        "qwen_accuracy": qwen_acc,
        "top1_disagreement": float(np.mean([r["disagree"] for r in rows])) if rows else 0.0,
        "useful_disagreement": float(np.mean([r["useful_disagreement"] for r in rows])) if rows else 0.0,
        "oracle_ceiling_accuracy": oracle,
        "oracle_gap_over_best_teacher": oracle - max(smol_acc, qwen_acc),
        "temperature": temperature,
    }


def cache_one_teacher(teacher_name: str, examples_by_split: dict[str, list[dict]], args, device: torch.device):
    tokenizer = load_tokenizer(teacher_name)
    model = load_lm(teacher_name, device, args.dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    manifest = model_manifest(model, teacher_name, tokenizer)
    out = {}
    for split_name, examples in examples_by_split.items():
        prepared = prepare_choice_set(tokenizer, examples, args.max_length)
        out[split_name] = evaluate_prepared(
            model,
            prepared,
            int(tokenizer.pad_token_id),
            device,
            args.eval_batch_choices,
            args.progress,
            f"teacher:{teacher_name}:{split_name}",
        )
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out, manifest


def build_teacher_cache(train_examples: list[dict], eval_examples: list[dict], args, device: torch.device) -> dict:
    cache_path = Path(args.output_dir) / "teacher_choice_cache.json"
    if args.reuse_teacher_cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    examples_by_split = {"train": train_examples, "eval": eval_examples}
    teachers, manifests = {}, {}
    for name, key in [(SMOL_TEACHER, "smollm2_360m"), (QWEN_TEACHER, "qwen3_0_6b")]:
        teacher_payload, manifest = cache_one_teacher(name, examples_by_split, args, device)
        teachers[key] = teacher_payload
        manifests[key] = manifest
    labels = {"train": [int(ex["label"]) for ex in train_examples], "eval": [int(ex["label"]) for ex in eval_examples]}
    disagreement = {
        split_name: summarize_teacher_pair(
            teachers["smollm2_360m"][split_name]["predictions"],
            teachers["qwen3_0_6b"][split_name]["predictions"],
            labels[split_name],
            args.teacher_temperature,
        )
        for split_name in ("train", "eval")
    }
    payload = {"teachers": teachers, "manifests": manifests, "disagreement": disagreement, "teacher_temperature": args.teacher_temperature}
    write_json(cache_path, payload)
    return payload


def add_lora(model, args):
    cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.config.use_cache = False
    return model


def training_objective(nll_groups: list[torch.Tensor], labels: list[int], teacher_targets, alpha: float, kd_temperature: float, device: torch.device):
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
        kl = F.kl_div(log_s, target, reduction="sum") * (kd_temperature ** 2)
        kl_values.append(float(kl.detach().item()))
        losses.append(float(alpha) * ce + (1.0 - float(alpha)) * kl)
    return torch.stack(losses).mean(), {"ce": float(np.mean(ce_values)), "kl": float(np.mean(kl_values)), "batch_accuracy": float(np.mean(correct))}


def train_lora_condition(run_name: str, train_prepared: PreparedChoiceSet, eval_by_bench: dict[str, PreparedChoiceSet], student_tokenizer, teacher_targets, route_meta, args, device: torch.device, seed_offset: int):
    started = time.time()
    model = add_lora(load_lm(STUDENT, device, args.dtype), args)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + seed_offset)
    history = []
    for step in range(1, args.train_steps + 1):
        idx = rng.choice(train_prepared.n_examples, size=min(args.batch_examples, train_prepared.n_examples), replace=False)
        nll_groups = score_prepared_indices(model, train_prepared, idx, int(student_tokenizer.pad_token_id), device)
        batch_labels = [train_prepared.labels[int(i)] for i in idx]
        batch_targets = None if teacher_targets is None else [teacher_targets[int(i)] for i in idx]
        loss, stats = training_objective(nll_groups, batch_labels, batch_targets, args.alpha, args.kd_temperature, device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            row = {"step": step, "loss": float(loss.detach().item()), "grad_norm": float(grad_norm.detach().item()), **stats}
            history.append(row)
            if args.progress:
                print(f"  [{run_name}] step {step}/{args.train_steps}: loss={row['loss']:.4f} ce={row['ce']:.4f} kl={row['kl']:.4f} acc={row['batch_accuracy']:.3f}", flush=True)
    model.eval()
    train_summary = evaluate_prepared(model, train_prepared, int(student_tokenizer.pad_token_id), device, args.eval_batch_choices, args.progress, f"{run_name}:train_eval")
    eval_results = {}
    for bench, prep in eval_by_bench.items():
        eval_results[bench] = evaluate_prepared(model, prep, int(student_tokenizer.pad_token_id), device, args.eval_batch_choices, args.progress, f"{run_name}:eval:{bench}")
    adapter_dir = Path(args.output_dir) / run_name / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    payload = {
        "run_name": run_name,
        "elapsed_s": round(time.time() - started, 3),
        "objective": "label_only_choice_ce" if teacher_targets is None else "label_ce_plus_choice_distribution_kl",
        "alpha_label_ce": 1.0 if teacher_targets is None else args.alpha,
        "kd_temperature": None if teacher_targets is None else args.kd_temperature,
        "route_meta": route_meta,
        "training": {
            "steps": args.train_steps,
            "batch_examples": args.batch_examples,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "history": history,
            "trainable_parameters": trainable,
            "total_parameters_with_lora": total,
            "adapter_dir": str(adapter_dir),
        },
        "train_results": train_summary,
        "benchmarks": eval_results,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def benchmark_deltas(main: dict[str, dict], control: dict[str, dict], samples: int, seed: int):
    rows = {}
    for bench in BENCHMARKS:
        rows[bench] = {
            "accuracy": bootstrap_accuracy_delta(main[bench]["predictions"], control[bench]["predictions"], samples, seed + len(bench)),
            "margin_best_wrong_minus_gold_nll": bootstrap_scalar_delta(main[bench]["predictions"], control[bench]["predictions"], "margin_best_wrong_minus_gold_nll", samples, seed + 100 + len(bench)),
            "main_accuracy": float(main[bench]["accuracy"]),
            "control_accuracy": float(control[bench]["accuracy"]),
            "delta_accuracy": float(main[bench]["accuracy"] - control[bench]["accuracy"]),
            "main_mean_margin": main[bench].get("mean_margin_best_wrong_minus_gold_nll"),
            "control_mean_margin": control[bench].get("mean_margin_best_wrong_minus_gold_nll"),
        }
    return rows


def label_verdict(label_results: dict[str, dict], baseline_results: dict[str, dict]) -> dict:
    rows, pass_count, marginal_count = {}, 0, 0
    for bench in BENCHMARKS:
        delta = float(label_results[bench]["accuracy"] - baseline_results[bench]["accuracy"])
        passed = delta >= 0.03
        marginal = 0.01 <= delta < 0.03
        pass_count += int(passed)
        marginal_count += int(marginal)
        rows[bench] = {"zero_shot_accuracy": float(baseline_results[bench]["accuracy"]), "label_only_accuracy": float(label_results[bench]["accuracy"]), "delta_accuracy": delta, "passes_plus_3pp": passed, "marginal_plus_1_to_3pp": marginal}
    verdict = PASS_LABEL_ONLY if pass_count >= 2 else (MARGINAL_LABEL_ONLY if pass_count + marginal_count >= 1 else FLAT_LABEL_ONLY)
    return {"verdict": verdict, "required_pass_benchmarks": 2, "pass_threshold": 0.03, "marginal_band": [0.01, 0.03], "passed_benchmarks": pass_count, "marginal_benchmarks": marginal_count, "benchmarks": rows}


def single_teacher_verdict(single_results: dict[str, dict], label_results: dict[str, dict]) -> dict:
    rows, pass_count, marginal_count = {}, 0, 0
    for bench in BENCHMARKS:
        delta = float(single_results[bench]["accuracy"] - label_results[bench]["accuracy"])
        passed = delta >= 0.02
        marginal = 0.005 <= delta < 0.02
        pass_count += int(passed)
        marginal_count += int(marginal)
        rows[bench] = {"label_only_accuracy": float(label_results[bench]["accuracy"]), "single_teacher_accuracy": float(single_results[bench]["accuracy"]), "delta_accuracy": delta, "passes_plus_2pp": passed, "marginal_plus_0_5_to_2pp": marginal}
    verdict = PASS_SINGLE_TEACHER if pass_count >= 2 else (MARGINAL_SINGLE_TEACHER if pass_count + marginal_count >= 1 else FAIL_SINGLE_TEACHER)
    return {"verdict": verdict, "required_pass_benchmarks": 2, "pass_threshold": 0.02, "marginal_band": [0.005, 0.02], "passed_benchmarks": pass_count, "marginal_benchmarks": marginal_count, "benchmarks": rows}


def eklavya_verdict(routed_results: dict[str, dict], label_results: dict[str, dict], single_results: dict[str, dict]) -> dict:
    rows, pass_count, beats_single_count = {}, 0, 0
    for bench in BENCHMARKS:
        delta_label = float(routed_results[bench]["accuracy"] - label_results[bench]["accuracy"])
        delta_single = float(routed_results[bench]["accuracy"] - single_results[bench]["accuracy"])
        passed = delta_label >= 0.03 and delta_single >= 0.03
        beats_single = delta_single > 0.0
        pass_count += int(passed)
        beats_single_count += int(beats_single)
        rows[bench] = {"routed_accuracy": float(routed_results[bench]["accuracy"]), "label_only_accuracy": float(label_results[bench]["accuracy"]), "single_teacher_accuracy": float(single_results[bench]["accuracy"]), "delta_routed_minus_label": delta_label, "delta_routed_minus_single_teacher": delta_single, "passes_plus_3pp_over_both": passed, "beats_single_teacher": beats_single}
    routed_mean = float(np.mean([routed_results[b]["accuracy"] for b in BENCHMARKS]))
    label_mean = float(np.mean([label_results[b]["accuracy"] for b in BENCHMARKS]))
    single_mean = float(np.mean([single_results[b]["accuracy"] for b in BENCHMARKS]))
    verdict = PASS_EKLAVYA_MECHANISM if pass_count >= 2 else (MARGINAL_EKLAVYA if routed_mean > single_mean or beats_single_count >= 1 else FAIL_EKLAVYA_MECHANISM)
    return {"verdict": verdict, "required_pass_benchmarks": 2, "pass_threshold_over_label_and_single": 0.03, "passed_benchmarks": pass_count, "benchmarks_beating_single_teacher": beats_single_count, "aggregate": {"routed_mean_accuracy": routed_mean, "label_only_mean_accuracy": label_mean, "single_teacher_mean_accuracy": single_mean, "delta_routed_minus_label_mean_accuracy": routed_mean - label_mean, "delta_routed_minus_single_teacher_mean_accuracy": routed_mean - single_mean}, "benchmarks": rows}


def remove_predictions_unless(payload: dict, save_predictions: bool) -> dict:
    return payload if save_predictions else strip_predictions(payload)


def run_full(args) -> dict:
    started = time.time()
    ensure_offline(args.allow_downloads)
    if not args.allow_downloads:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    set_all_seeds(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_by_bench, eval_by_bench, split_meta = make_split(args.train_per_benchmark, args.eval_per_benchmark, args.benchmark_split, args.seed, args.allow_downloads)
    train_examples = flatten_examples(train_by_bench)
    eval_examples = flatten_examples(eval_by_bench)

    student_tokenizer = load_tokenizer(STUDENT)
    student_model = load_lm(STUDENT, device, args.dtype).eval()
    for p in student_model.parameters():
        p.requires_grad_(False)
    student_manifest = model_manifest(student_model, STUDENT, student_tokenizer)

    zero_examples = zero_shot_examples(args.zero_shot_examples, args.benchmark_split, args.seed, args.allow_downloads)
    zero_shot_200plus = evaluate_by_benchmark(student_model, student_tokenizer, zero_examples, args.max_length, device, args.eval_batch_choices, args.progress, "zero_shot_200plus")
    heldout_baseline = evaluate_by_benchmark(student_model, student_tokenizer, eval_by_bench, args.max_length, device, args.eval_batch_choices, args.progress, "zero_shot_heldout")
    train_prepared = prepare_choice_set(student_tokenizer, train_examples, args.max_length)
    train_baseline = evaluate_prepared(student_model, train_prepared, int(student_tokenizer.pad_token_id), device, args.eval_batch_choices, args.progress, "zero_shot_train")
    del student_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    eval_prepared_by_bench = {bench: prepare_choice_set(student_tokenizer, eval_by_bench[bench], args.max_length) for bench in BENCHMARKS}
    teacher_cache = build_teacher_cache(train_examples, eval_examples, args, device)
    smol_train = teacher_cache["teachers"]["smollm2_360m"]["train"]["predictions"]
    qwen_train = teacher_cache["teachers"]["qwen3_0_6b"]["train"]["predictions"]
    train_labels = [int(ex["label"]) for ex in train_examples]

    single_targets, single_route_meta = build_routed_targets(smol_train, qwen_train, train_labels, "single_smol", args.teacher_temperature, args.seed + 100)
    oracle_targets, oracle_route_meta = build_routed_targets(smol_train, qwen_train, train_labels, "oracle", args.teacher_temperature, args.seed + 101)
    non_oracle_targets, non_oracle_route_meta = build_routed_targets(smol_train, qwen_train, train_labels, args.non_oracle_route, args.teacher_temperature, args.seed + 102)
    random_targets, random_route_meta = build_routed_targets(smol_train, qwen_train, train_labels, "random", args.teacher_temperature, args.seed + 103)

    runs = {}
    runs["label_only"] = train_lora_condition("label_only", train_prepared, eval_prepared_by_bench, student_tokenizer, None, None, args, device, 2100)
    runs["single_teacher_smol360"] = train_lora_condition("single_teacher_smol360", train_prepared, eval_prepared_by_bench, student_tokenizer, single_targets, single_route_meta, args, device, 2200)
    runs["oracle_route_ceiling"] = train_lora_condition("oracle_route_ceiling", train_prepared, eval_prepared_by_bench, student_tokenizer, oracle_targets, oracle_route_meta, args, device, 2300)
    non_oracle_name = f"non_oracle_{args.non_oracle_route}"
    runs[non_oracle_name] = train_lora_condition(non_oracle_name, train_prepared, eval_prepared_by_bench, student_tokenizer, non_oracle_targets, non_oracle_route_meta, args, device, 2400)
    if args.run_random_control:
        runs["random_route_control"] = train_lora_condition("random_route_control", train_prepared, eval_prepared_by_bench, student_tokenizer, random_targets, random_route_meta, args, device, 2500)

    label_results = runs["label_only"]["benchmarks"]
    single_results = runs["single_teacher_smol360"]["benchmarks"]
    non_oracle_results = runs[non_oracle_name]["benchmarks"]
    verdicts = {
        "label_only": label_verdict(label_results, heldout_baseline),
        "single_teacher": single_teacher_verdict(single_results, label_results),
        "eklavya_non_oracle": eklavya_verdict(non_oracle_results, label_results, single_results),
        "oracle_ceiling_vs_single": eklavya_verdict(runs["oracle_route_ceiling"]["benchmarks"], label_results, single_results),
    }
    deltas = {
        "label_only_minus_zero_shot": benchmark_deltas(label_results, heldout_baseline, args.bootstrap_samples, args.seed + 3000),
        "single_teacher_minus_label_only": benchmark_deltas(single_results, label_results, args.bootstrap_samples, args.seed + 3100),
        f"{non_oracle_name}_minus_single_teacher": benchmark_deltas(non_oracle_results, single_results, args.bootstrap_samples, args.seed + 3200),
        f"{non_oracle_name}_minus_label_only": benchmark_deltas(non_oracle_results, label_results, args.bootstrap_samples, args.seed + 3300),
        "oracle_route_minus_single_teacher": benchmark_deltas(runs["oracle_route_ceiling"]["benchmarks"], single_results, args.bootstrap_samples, args.seed + 3400),
    }
    if "random_route_control" in runs:
        deltas["random_route_minus_single_teacher"] = benchmark_deltas(runs["random_route_control"]["benchmarks"], single_results, args.bootstrap_samples, args.seed + 3500)

    payload = {
        "mode": "smollm2_mechanism_control_full",
        "run": {"seed": args.seed, "device": str(device), "dtype": args.dtype, "elapsed_s": round(time.time() - started, 3), "benchmark_split": args.benchmark_split, "benchmarks": list(BENCHMARKS), "zero_shot_examples_per_benchmark": args.zero_shot_examples, "train_examples_total": len(train_examples), "eval_examples_total": len(eval_examples), "train_per_benchmark": args.train_per_benchmark, "eval_per_benchmark": args.eval_per_benchmark, "max_length": args.max_length},
        "precommitted_verdict_tokens": {"label_only": [PASS_LABEL_ONLY, MARGINAL_LABEL_ONLY, FLAT_LABEL_ONLY], "single_teacher": [PASS_SINGLE_TEACHER, MARGINAL_SINGLE_TEACHER, FAIL_SINGLE_TEACHER], "eklavya": [PASS_EKLAVYA_MECHANISM, MARGINAL_EKLAVYA, FAIL_EKLAVYA_MECHANISM]},
        "student": student_manifest,
        "split": split_meta,
        "teacher_cache": teacher_cache,
        "zero_shot_200plus": zero_shot_200plus,
        "zero_shot_train": train_baseline,
        "zero_shot_heldout": heldout_baseline,
        "runs": runs,
        "deltas": deltas,
        "verdicts": verdicts,
        "primary_verdict": verdicts["eklavya_non_oracle"]["verdict"],
        "non_oracle_route": non_oracle_name,
        "limitations": ["Train-safe held-out split is not public benchmark validation.", "Oracle routing is a label-leaking ceiling and is not a deployable Eklavya method.", "Non-oracle routing uses cached teacher choice distributions only; teachers are not loaded during student training.", "All student conditions use the same LoRA target modules, step budget, optimizer, alpha, and data split."],
    }
    write_json(out_dir / "smollm2_mechanism_control.json", remove_predictions_unless(payload, args.save_predictions))
    return payload


def run_smoke(args) -> dict:
    args.zero_shot_examples = min(args.zero_shot_examples, 4)
    args.train_per_benchmark = min(args.train_per_benchmark, 2)
    args.eval_per_benchmark = min(args.eval_per_benchmark, 2)
    args.train_steps = min(args.train_steps, 2)
    args.batch_examples = min(args.batch_examples, 3)
    args.eval_batch_choices = min(args.eval_batch_choices, 8)
    args.bootstrap_samples = min(args.bootstrap_samples, 20)
    args.output_dir = str(Path(args.output_dir) / "smoke")
    args.run_random_control = False
    args.reuse_teacher_cache = False
    return run_full(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--output-dir", default="tmp_work_loop_b14")
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--reuse-teacher-cache", action="store_true")
    parser.add_argument("--benchmark-split", default="train")
    parser.add_argument("--zero-shot-examples", type=int, default=240)
    parser.add_argument("--train-per-benchmark", type=int, default=96)
    parser.add_argument("--eval-per-benchmark", type=int, default=48)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--eval-batch-choices", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--batch-examples", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--non-oracle-route", choices=["confidence", "confidence_mixture"], default="confidence")
    parser.add_argument("--run-random-control", action="store_true", default=True)
    parser.add_argument("--skip-random-control", dest="run_random_control", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_smoke(args) if args.mode == "smoke" else run_full(args)
    print(json.dumps({"mode": payload["mode"], "primary_verdict": payload["primary_verdict"], "label_verdict": payload["verdicts"]["label_only"]["verdict"], "single_teacher_verdict": payload["verdicts"]["single_teacher"]["verdict"], "output": str(Path(args.output_dir) / "smollm2_mechanism_control.json")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

