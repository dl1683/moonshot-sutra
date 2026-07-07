"""HellaSwag margin diagnostics for Sutra checkpoints.

Scores S0, E1, and Option C checkpoints one at a time, using the same byte
scoring path as benchmark_harness.py. Reports whether BPB gains improve the
decision margin between the gold ending and the best wrong ending.

Usage:
    python margin_diagnostic.py \
        --s0-checkpoint C:/sutra_fast/checkpoints/s0/s0_best.pt \
        --e1-checkpoint C:/sutra_fast/checkpoints/e1/e1_best.pt \
        --option-c-checkpoint C:/sutra_fast/checkpoints/option_c/optc_final.pt \
        --output results/hellaswag_margins.jsonl \
        --summary-output results/hellaswag_margins_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from benchmark_harness import load_hellaswag, score_multiple_choice
from s0_architecture import SutraS0
from s0_training import TrainConfig  # noqa: F401 -- needed for checkpoint unpickling


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: str


def _choice_variant(choices: list[str], variant: str) -> list[str]:
    if variant == "as_is":
        return choices
    if variant == "leading_space":
        return [c if c.startswith(" ") else " " + c for c in choices]
    raise ValueError(f"unknown variant: {variant}")


def _load_done_keys(output_path: str) -> set[tuple[str, str, int]]:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("model", ""), row.get("variant", ""), row.get("example_idx", -1)))
    return done


def _load_model(path: str, device: torch.device) -> tuple[SutraS0, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SutraS0(ckpt["model_cfg"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def score_one_example(
    model: SutraS0,
    ex: dict,
    device: torch.device,
    variant: str,
) -> dict:
    choices = _choice_variant(ex["choices"], variant)
    _, scored = score_multiple_choice(
        model, ex["context"], choices, device, length_normalize=True)

    label = int(ex["label"])
    correct = scored[label]
    wrong_indices = [i for i in range(len(scored)) if i != label]
    best_wrong_idx = min(wrong_indices, key=lambda i: scored[i].bpb)
    best_wrong = scored[best_wrong_idx]
    pred_idx = min(range(len(scored)), key=lambda i: scored[i].bpb)

    margin = best_wrong.bpb - correct.bpb
    return {
        "label": label,
        "pred_norm": pred_idx,
        "is_correct_norm": pred_idx == label,
        "correct_bpb": correct.bpb,
        "best_wrong_bpb": best_wrong.bpb,
        "best_wrong_idx": best_wrong_idx,
        "margin": margin,
        "choice_bpbs": [s.bpb for s in scored],
        "choice_total_nll": [s.total_nll for s in scored],
        "choice_n_bytes": [s.n_bytes for s in scored],
    }


def summarize(rows: list[dict]) -> dict:
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_key.setdefault((row["model"], row["variant"]), []).append(row)

    summary = {}
    for (model_name, variant), group in sorted(by_key.items()):
        margins = np.array([r["margin"] for r in group], dtype=np.float64)
        correct_bpbs = np.array([r["correct_bpb"] for r in group], dtype=np.float64)
        wrong_bpbs = np.array([r["best_wrong_bpb"] for r in group], dtype=np.float64)
        wins = margins > 0
        key = f"{model_name}:{variant}"
        summary[key] = {
            "model": model_name,
            "variant": variant,
            "n_examples": int(len(group)),
            "accuracy_norm": float(wins.mean()) if len(group) else 0.0,
            "percent_correct_lt_best_wrong": float(wins.mean() * 100.0) if len(group) else 0.0,
            "mean_margin": float(margins.mean()) if len(group) else 0.0,
            "median_margin": float(np.median(margins)) if len(group) else 0.0,
            "mean_correct_bpb": float(correct_bpbs.mean()) if len(group) else 0.0,
            "mean_best_wrong_bpb": float(wrong_bpbs.mean()) if len(group) else 0.0,
            "p10_margin": float(np.percentile(margins, 10)) if len(group) else 0.0,
            "p90_margin": float(np.percentile(margins, 90)) if len(group) else 0.0,
        }
    return summary


def _read_rows(output_path: str) -> list[dict]:
    rows = []
    if not os.path.exists(output_path):
        return rows
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda_index)
    torch.set_num_threads(args.cpu_threads)

    examples = load_hellaswag(args.split)
    if args.max_examples > 0:
        examples = examples[:args.max_examples]

    variants = ["as_is", "leading_space"] if args.leading_space_ablation else ["as_is"]
    specs = [
        CheckpointSpec("s0", args.s0_checkpoint),
        CheckpointSpec("e1", args.e1_checkpoint),
        CheckpointSpec("option_c", args.option_c_checkpoint),
    ]

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    if args.summary_output:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_output)), exist_ok=True)

    done = _load_done_keys(args.output) if args.resume else set()
    t0 = time.time()

    with open(args.output, "a", encoding="utf-8") as out_f:
        for spec in specs:
            print(f"\nLoading {spec.name}: {spec.path}", flush=True)
            model, ckpt = _load_model(spec.path, device)
            step = ckpt.get("step", -1)
            eval_bpb = ckpt.get("eval_bpb", None)
            print(f"  step={step} eval_bpb={eval_bpb}", flush=True)

            for variant in variants:
                print(f"  scoring variant={variant} n={len(examples)}", flush=True)
                for idx, ex in enumerate(examples):
                    key = (spec.name, variant, idx)
                    if key in done:
                        continue
                    scored = score_one_example(model, ex, device, variant)
                    row = {
                        "model": spec.name,
                        "checkpoint": spec.path,
                        "checkpoint_step": step,
                        "checkpoint_eval_bpb": eval_bpb,
                        "variant": variant,
                        "split": args.split,
                        "example_idx": idx,
                        "context": ex["context"] if args.include_text else None,
                        "choices": _choice_variant(ex["choices"], variant) if args.include_text else None,
                        **scored,
                    }
                    out_f.write(json.dumps(row) + "\n")
                    if (idx + 1) % args.progress_every == 0:
                        elapsed = time.time() - t0
                        print(f"    {spec.name}/{variant}: {idx+1}/{len(examples)} "
                              f"({elapsed:.0f}s)", flush=True)
                out_f.flush()

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    rows = _read_rows(args.output)
    summary = summarize(rows)
    if args.summary_output:
        with open(args.summary_output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="HellaSwag BPB margin diagnostic for Sutra checkpoints")
    parser.add_argument("--s0-checkpoint", required=True)
    parser.add_argument("--e1-checkpoint", required=True)
    parser.add_argument("--option-c-checkpoint", required=True)
    parser.add_argument("--output", default="results/hellaswag_margins.jsonl")
    parser.add_argument("--summary-output",
                        default="results/hellaswag_margins_summary.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="0 means full split")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-index", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-leading-space-ablation",
                        dest="leading_space_ablation", action="store_false")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--include-text", action="store_true",
                        help="Include HellaSwag text in per-example JSONL")
    parser.set_defaults(leading_space_ablation=True, resume=True)
    args = parser.parse_args()

    summary = run(args)
    print("\nSUMMARY")
    for key, metrics in summary.items():
        print(
            f"  {key}: acc_norm={metrics['accuracy_norm']:.4f} "
            f"mean_margin={metrics['mean_margin']:.4f} "
            f"median_margin={metrics['median_margin']:.4f} "
            f"mean_correct_bpb={metrics['mean_correct_bpb']:.3f} "
            f"mean_best_wrong_bpb={metrics['mean_best_wrong_bpb']:.3f}")


if __name__ == "__main__":
    main()
