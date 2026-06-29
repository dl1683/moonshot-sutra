"""Standard benchmark evaluation for byte-level Sutra models.

Evaluates S0 (and E1/E2) checkpoints on standard NLP benchmarks, producing
results directly comparable to SmolLM2-135M, Pythia-160M, and other models.

Supported benchmarks:
  - HellaSwag (4-choice commonsense NLI)
  - PIQA (2-choice physical intuition)
  - ARC-Easy / ARC-Challenge (4-choice reasoning)
  - LAMBADA (last-word prediction accuracy + BPB)
  - WinoGrande (2-choice coreference)
  - WikiText-103 (corpus BPB for byte-level model comparison)

Usage:
    python benchmark_harness.py \
        --checkpoint C:/sutra_fast/checkpoints/s0_full/s0_best.pt \
        --benchmarks hellaswag piqa arc_easy arc_challenge lambada \
        --output results/benchmarks.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from s0_architecture import SutraS0


@dataclass
class ScoredChoice:
    text: str
    total_nll: float
    n_bytes: int
    bpb: float
    predicted_bytes: list[int] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    name: str
    n_examples: int
    accuracy: float
    accuracy_norm: float
    mean_bpb: float
    elapsed_s: float
    details: dict = field(default_factory=dict)


@torch.no_grad()
def score_completion(
    model: SutraS0,
    context_bytes: list[int],
    completion_bytes: list[int],
    device: torch.device,
) -> ScoredChoice:
    """Score a completion given context using the byte-level model.

    Concatenates [context | completion], pads to patch boundaries, runs the
    model, and extracts cross-entropy loss only on completion byte positions.
    """
    P = model.cfg.patch_size

    context_len = len(context_bytes)
    completion_len = len(completion_bytes)

    if completion_len == 0:
        return ScoredChoice("", float("inf"), 0, float("inf"))

    ctx_pad = (P - (context_len % P)) % P
    padded_context = [0] * ctx_pad + context_bytes
    full_seq = padded_context + completion_bytes
    end_pad = (P - (len(full_seq) % P)) % P
    full_seq = full_seq + [0] * end_pad

    if len(full_seq) < 2 * P:
        full_seq = [0] * (2 * P - len(full_seq)) + full_seq
        ctx_pad += 2 * P - len(full_seq)

    byte_ids = torch.tensor([full_seq], dtype=torch.long, device=device)
    B, T = byte_ids.shape
    N = T // P

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(byte_ids)
    logits = out["logits"]  # (1, N-1, P, 256)

    targets = byte_ids.reshape(1, N, P)[:, 1:]  # (1, N-1, P)

    completion_start_in_full = len(padded_context)
    completion_end_in_full = completion_start_in_full + completion_len

    logit_rows = []
    target_list = []
    for flat_pos in range(completion_start_in_full, completion_end_in_full):
        patch_idx = flat_pos // P
        byte_in_patch = flat_pos % P
        target_patch_idx = patch_idx - 1
        if target_patch_idx < 0 or target_patch_idx >= logits.shape[1]:
            continue
        logit_rows.append(logits[0, target_patch_idx, byte_in_patch])
        target_list.append(targets[0, target_patch_idx, byte_in_patch])

    n_scored = len(logit_rows)
    if n_scored == 0:
        return ScoredChoice("", float("inf"), 0, float("inf"))

    all_logits = torch.stack(logit_rows).float()
    all_targets = torch.stack(target_list)
    total_nll = F.cross_entropy(all_logits, all_targets, reduction="sum").item()
    predicted = all_logits.argmax(dim=-1).tolist()

    bpb = total_nll / (n_scored * math.log(2))
    completion_text = bytes(completion_bytes).decode("utf-8", errors="replace")
    return ScoredChoice(completion_text, total_nll, n_scored, bpb, predicted)


def score_multiple_choice(
    model: SutraS0,
    context: str,
    choices: list[str],
    device: torch.device,
    length_normalize: bool = True,
) -> tuple[int, list[ScoredChoice]]:
    """Score multiple choices and return the best one.

    Uses byte-count normalized BPB by default (matches lm-eval-harness
    acc_norm metric). Set length_normalize=False for raw log-likelihood.
    """
    ctx_bytes = list(context.encode("utf-8"))
    scored = []
    for choice in choices:
        comp_bytes = list(choice.encode("utf-8"))
        result = score_completion(model, ctx_bytes, comp_bytes, device)
        scored.append(result)

    if length_normalize:
        best_idx = min(range(len(scored)), key=lambda i: scored[i].bpb)
    else:
        best_idx = min(range(len(scored)), key=lambda i: scored[i].total_nll)

    return best_idx, scored


def _hellaswag_preprocess(text: str) -> str:
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag(split: str = "validation") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split=split)
    examples = []
    for row in ds:
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        query = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
        choices = [_hellaswag_preprocess(e) for e in row["endings"]]
        label = int(row["label"])
        examples.append({"context": query, "choices": choices, "label": label})
    return examples


def load_piqa(split: str = "validation") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("baber/piqa", split=split)
    examples = []
    for row in ds:
        context = f"Question: {row['goal']}\nAnswer:"
        choices = [row["sol1"], row["sol2"]]
        label = row["label"]
        examples.append({"context": context, "choices": choices, "label": label})
    return examples


def load_arc(difficulty: str = "easy", split: str = "test") -> list[dict]:
    subset = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split=split)
    examples = []
    for row in ds:
        question = row["question"]
        choices = row["choices"]["text"]
        label_key = row["answerKey"]
        label_map = {k: i for i, k in enumerate(row["choices"]["label"])}
        label = label_map.get(label_key, 0)
        context = f"Question: {question}\nAnswer:"
        examples.append({"context": context, "choices": choices, "label": label})
    return examples


def load_lambada(split: str = "test") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", split=split)
    examples = []
    for row in ds:
        text = row["text"]
        last_space = text.rfind(" ")
        if last_space < 0:
            continue
        context = text[:last_space]
        target_word = text[last_space:]
        examples.append({"context": context, "target": target_word, "full_text": text})
    return examples


def load_winogrande(split: str = "validation") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split=split)
    examples = []
    for row in ds:
        sentence = row["sentence"]
        option1 = row["option1"]
        option2 = row["option2"]
        answer = int(row["answer"]) - 1
        idx = sentence.index("_")
        before = sentence[:idx]
        after = sentence[idx + 1:].strip()
        contexts = [before + option1, before + option2]
        examples.append({
            "context": contexts,
            "completion": after,
            "label": answer,
        })
    return examples


def _add_noise(text: str, noise_rate: float = 0.1, seed: int = 42) -> str:
    import random
    rng = random.Random(seed)
    chars = list(text)
    ops = ["swap", "delete", "insert", "substitute"]
    n_ops = max(1, int(len(chars) * noise_rate))
    for _ in range(n_ops):
        if not chars:
            break
        op = rng.choice(ops)
        pos = rng.randint(0, max(0, len(chars) - 1))
        if op == "swap" and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif op == "delete" and len(chars) > 1:
            chars.pop(pos)
        elif op == "insert":
            chars.insert(pos, rng.choice("abcdefghijklmnopqrstuvwxyz "))
        elif op == "substitute":
            chars[pos] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)


def noisify_examples(
    examples: list[dict],
    noise_rate: float = 0.1,
    noise_context: bool = True,
    noise_choices: bool = True,
) -> list[dict]:
    noised = []
    for i, ex in enumerate(examples):
        new_ex = dict(ex)
        if noise_context and isinstance(ex.get("context"), str):
            new_ex["context"] = _add_noise(ex["context"], noise_rate, seed=i)
        if noise_choices and "choices" in ex:
            new_ex["choices"] = [
                _add_noise(c, noise_rate, seed=i * 100 + j)
                for j, c in enumerate(ex["choices"])
            ]
        noised.append(new_ex)
    return noised


def eval_multiple_choice(
    model: SutraS0,
    examples: list[dict],
    device: torch.device,
    benchmark_name: str,
) -> BenchmarkResult:
    correct = 0
    correct_norm = 0
    total = 0
    bpbs = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        _, scored = score_multiple_choice(
            model, ex["context"], ex["choices"], device, length_normalize=False
        )
        pred_raw = min(range(len(scored)), key=lambda j: scored[j].total_nll)
        pred_norm = min(range(len(scored)), key=lambda j: scored[j].bpb)
        label = ex["label"]
        if pred_raw == label:
            correct += 1
        if pred_norm == label:
            correct_norm += 1
        total += 1
        bpbs.append(scored[label].bpb)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = correct / total
            acc_n = correct_norm / total
            print(f"  [{benchmark_name}] {i+1}/{len(examples)}: "
                  f"acc={acc:.3f} acc_norm={acc_n:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    return BenchmarkResult(
        name=benchmark_name,
        n_examples=total,
        accuracy=correct / max(total, 1),
        accuracy_norm=correct_norm / max(total, 1),
        mean_bpb=float(np.mean(bpbs)) if bpbs else 0.0,
        elapsed_s=elapsed,
    )


def eval_winogrande(
    model: SutraS0,
    examples: list[dict],
    device: torch.device,
) -> BenchmarkResult:
    correct = 0
    correct_norm = 0
    total = 0
    bpbs = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        comp_bytes = list(ex["completion"].encode("utf-8"))
        scored = []
        for ctx_str in ex["context"]:
            ctx_bytes = list(ctx_str.encode("utf-8"))
            scored.append(score_completion(model, ctx_bytes, comp_bytes, device))

        pred_raw = min(range(len(scored)), key=lambda j: scored[j].total_nll)
        pred_norm = min(range(len(scored)), key=lambda j: scored[j].bpb)
        label = ex["label"]
        if pred_raw == label:
            correct += 1
        if pred_norm == label:
            correct_norm += 1
        total += 1
        bpbs.append(scored[label].bpb)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            acc = correct / total
            acc_n = correct_norm / total
            print(f"  [winogrande] {i+1}/{len(examples)}: "
                  f"acc={acc:.3f} acc_norm={acc_n:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    return BenchmarkResult(
        name="winogrande",
        n_examples=total,
        accuracy=correct / max(total, 1),
        accuracy_norm=correct_norm / max(total, 1),
        mean_bpb=float(np.mean(bpbs)) if bpbs else 0.0,
        elapsed_s=elapsed,
    )


def eval_lambada(
    model: SutraS0,
    examples: list[dict],
    device: torch.device,
) -> BenchmarkResult:
    correct = 0
    total = 0
    bpbs = []
    byte_correct = 0
    byte_total = 0
    t0 = time.time()

    for i, ex in enumerate(examples):
        ctx_bytes = list(ex["context"].encode("utf-8"))
        target_bytes = list(ex["target"].encode("utf-8"))

        scored = score_completion(model, ctx_bytes, target_bytes, device)
        bpbs.append(scored.bpb)

        if scored.predicted_bytes == target_bytes:
            correct += 1
        n_match = sum(
            1 for p, t in zip(scored.predicted_bytes, target_bytes) if p == t
        )
        byte_correct += n_match
        byte_total += len(target_bytes)
        total += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            acc = correct / total
            byte_acc = byte_correct / max(byte_total, 1)
            print(f"  [LAMBADA] {i+1}/{len(examples)}: "
                  f"acc={acc:.3f} byte_acc={byte_acc:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    return BenchmarkResult(
        name="lambada_openai",
        n_examples=total,
        accuracy=correct / max(total, 1),
        accuracy_norm=correct / max(total, 1),
        mean_bpb=float(np.mean(bpbs)) if bpbs else 0.0,
        elapsed_s=elapsed,
        details={"byte_accuracy": byte_correct / max(byte_total, 1)},
    )


GENERATION_PROMPTS = [
    "The quick brown fox",
    "In a groundbreaking study, researchers",
    "The capital of France is",
    "Once upon a time in a small village,",
    "The most important thing about machine learning is",
    "Water boils at a temperature of",
    "The theory of relativity states that",
    "In the year 2025, technology has",
]


@torch.no_grad()
def eval_generation(
    model: SutraS0,
    device: torch.device,
    prompts: list[str] | None = None,
    n_patches: int = 32,
    temperature: float = 0.8,
    top_k: int = 50,
) -> dict:
    """Generate text from prompts and compute quality metrics."""
    from s0_eval import generate_bytes, bytes_to_text

    if prompts is None:
        prompts = GENERATION_PROMPTS

    model.eval()
    P = model.cfg.patch_size
    samples = []
    all_generated_text = []
    t0 = time.time()

    for prompt_text in prompts:
        prompt_raw = prompt_text.encode("utf-8")
        pad_len = (P - (len(prompt_raw) % P)) % P
        prompt_raw = prompt_raw + b'\x00' * pad_len
        prompt_tensor = torch.tensor(
            list(prompt_raw), dtype=torch.long
        ).unsqueeze(0).to(device)

        gen_start = time.time()
        generated = generate_bytes(
            model, prompt_tensor, n_patches=n_patches,
            temperature=temperature, top_k=top_k,
        )
        gen_time = time.time() - gen_start

        gen_text = bytes_to_text(generated[0])
        gen_bytes = generated.shape[1]
        bytes_per_sec = gen_bytes / max(gen_time, 1e-6)

        all_generated_text.append(gen_text)
        samples.append({
            "prompt": prompt_text,
            "generated": gen_text[:500],
            "n_bytes": gen_bytes,
            "gen_time_s": round(gen_time, 3),
            "bytes_per_sec": round(bytes_per_sec, 1),
        })

    full_text = " ".join(all_generated_text)
    words = full_text.split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    distinct_1 = len(set(words)) / max(len(words), 1)
    distinct_2 = len(set(bigrams)) / max(len(bigrams), 1)

    byte_counts = {}
    for ch in full_text:
        byte_counts[ch] = byte_counts.get(ch, 0) + 1
    total_chars = max(len(full_text), 1)
    char_entropy = -sum(
        (c / total_chars) * math.log2(c / total_chars)
        for c in byte_counts.values()
    )

    ascii_ratio = sum(1 for c in full_text if 32 <= ord(c) < 127) / total_chars

    elapsed = time.time() - t0

    return {
        "samples": samples,
        "metrics": {
            "distinct_1": round(distinct_1, 4),
            "distinct_2": round(distinct_2, 4),
            "char_entropy_bits": round(char_entropy, 3),
            "ascii_ratio": round(ascii_ratio, 4),
            "n_prompts": len(prompts),
            "total_generated_bytes": sum(s["n_bytes"] for s in samples),
            "elapsed_s": round(elapsed, 1),
        },
    }


def eval_wikitext_bpb(
    model: SutraS0,
    device: torch.device,
    max_chars: int = 500_000,
) -> dict:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n\n".join(row["text"] for row in ds if row["text"].strip())
    if max_chars > 0:
        text = text[:max_chars]
    text_bytes = list(text.encode("utf-8"))
    P = model.cfg.patch_size
    seq_len = model.cfg.max_seq_len * P
    stride = seq_len // 2

    total_nll = 0.0
    total_bytes = 0
    n_windows = 0
    t0 = time.time()

    for start in range(0, len(text_bytes) - P, stride):
        window = text_bytes[start:start + seq_len]
        actual_len = len(window)
        pad_len = (P - (actual_len % P)) % P
        window = window + [0] * pad_len

        byte_ids = torch.tensor([window], dtype=torch.long, device=device)
        N = byte_ids.shape[1] // P

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(byte_ids)
        logits = out["logits"].float()  # (1, N-1, P, 256)
        targets = byte_ids.reshape(1, N, P)[:, 1:]  # (1, N-1, P)

        if start == 0:
            score_start_patch = 0
        else:
            context_patches = stride // P
            score_start_patch = max(0, context_patches - 1)
        actual_patches = (actual_len + pad_len) // P
        score_end_patch = actual_patches - 1

        if score_start_patch >= score_end_patch:
            score_start_patch = 0

        score_logits = logits[:, score_start_patch:score_end_patch]
        score_targets = targets[:, score_start_patch:score_end_patch]

        loss = F.cross_entropy(
            score_logits.reshape(-1, score_logits.shape[-1]),
            score_targets.reshape(-1),
            reduction="sum",
        ).item()

        n_target_bytes = score_targets.numel()
        total_nll += loss
        total_bytes += n_target_bytes
        n_windows += 1

        if n_windows % 20 == 0:
            elapsed = time.time() - t0
            running_bpb = total_nll / (total_bytes * math.log(2))
            print(f"  [wikitext] {n_windows} windows, "
                  f"{total_bytes:,} bytes, bpb={running_bpb:.3f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    bpb = total_nll / (total_bytes * math.log(2))
    return {
        "bpb": round(bpb, 4),
        "total_bytes": total_bytes,
        "n_windows": n_windows,
        "elapsed_s": round(elapsed, 1),
    }


BENCHMARK_LOADERS = {
    "hellaswag": lambda: load_hellaswag(),
    "piqa": lambda: load_piqa(),
    "arc_easy": lambda: load_arc("easy"),
    "arc_challenge": lambda: load_arc("challenge"),
    "lambada": lambda: load_lambada(),
    "winogrande": lambda: load_winogrande(),
    "hellaswag_noised": lambda: noisify_examples(load_hellaswag(), noise_rate=0.1),
    "piqa_noised": lambda: noisify_examples(load_piqa(), noise_rate=0.1),
}


def run_benchmarks(
    checkpoint_path: str,
    benchmark_names: list[str],
    output_path: str | None = None,
    max_examples: int = 0,
) -> dict:
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = SutraS0(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint from step {step} (eval_bpb={ckpt.get('eval_bpb', '?')})")

    counts = model.count_parameters()
    print(f"Parameters: {counts['total']:,} ({counts['total']/1e6:.1f}M)")

    results = {}
    for name in benchmark_names:
        if name == "wikitext":
            print(f"\n{'='*60}")
            print(f"Running: wikitext (corpus BPB)")
            print(f"{'='*60}")
            wt_results = eval_wikitext_bpb(model, device)
            results["wikitext"] = wt_results
            print(f"\n  WikiText-103 BPB: {wt_results['bpb']:.4f}")
            print(f"  ({wt_results['total_bytes']:,} bytes, "
                  f"{wt_results['n_windows']} windows, "
                  f"{wt_results['elapsed_s']:.0f}s)")
            continue

        if name == "generation":
            print(f"\n{'='*60}")
            print(f"Running: generation")
            print(f"{'='*60}")
            gen_results = eval_generation(model, device)
            results["generation"] = gen_results
            print(f"\n  Generation quality:")
            m = gen_results["metrics"]
            print(f"    distinct-1: {m['distinct_1']:.4f}")
            print(f"    distinct-2: {m['distinct_2']:.4f}")
            print(f"    char_entropy: {m['char_entropy_bits']:.3f} bits")
            print(f"    ascii_ratio: {m['ascii_ratio']:.4f}")
            print(f"\n  Samples:")
            for s in gen_results["samples"]:
                print(f"    [{s['prompt']}] -> {s['generated'][:80]}...")
            continue

        if name not in BENCHMARK_LOADERS:
            print(f"Unknown benchmark: {name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        examples = BENCHMARK_LOADERS[name]()
        if max_examples > 0:
            examples = examples[:max_examples]
        print(f"  Loaded {len(examples)} examples")

        if name == "lambada":
            result = eval_lambada(model, examples, device)
        elif name == "winogrande":
            result = eval_winogrande(model, examples, device)
        else:
            result = eval_multiple_choice(model, examples, device, name)

        results[name] = {
            "accuracy": result.accuracy,
            "accuracy_norm": result.accuracy_norm,
            "mean_bpb": result.mean_bpb,
            "n_examples": result.n_examples,
            "elapsed_s": result.elapsed_s,
        }

        print(f"\n  {name}: acc={result.accuracy:.4f} "
              f"acc_norm={result.accuracy_norm:.4f} "
              f"bpb={result.mean_bpb:.3f} "
              f"({result.elapsed_s:.0f}s)")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Benchmark':<18} {'Acc':>8} {'Acc_Norm':>10} {'BPB':>8}")
    print("-" * 48)
    for name, r in results.items():
        if name in ("generation", "wikitext"):
            continue
        print(f"{name:<18} {r['accuracy']:>8.4f} {r['accuracy_norm']:>10.4f} {r['mean_bpb']:>8.3f}")
    if "wikitext" in results:
        print(f"{'wikitext':<18} {'---':>8} {'---':>10} {results['wikitext']['bpb']:>8.3f}")

    full_results = {
        "checkpoint": checkpoint_path,
        "step": step,
        "model_params": counts["total"],
        "device": str(device),
        "benchmarks": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="Sutra S0 Benchmark Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--benchmarks", nargs="+",
                        default=["hellaswag", "piqa", "arc_easy", "arc_challenge",
                                 "lambada", "winogrande", "wikitext", "generation"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit examples per benchmark (0 = all)")
    args = parser.parse_args()
    run_benchmarks(args.checkpoint, args.benchmarks, args.output, args.max_examples)


if __name__ == "__main__":
    main()
