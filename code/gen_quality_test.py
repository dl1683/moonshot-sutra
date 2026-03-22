"""Generation Quality Test for Sutra v0.6.0a.

Tests actual text generation quality across diverse prompts.
This is the "does it produce coherent text?" test — the most important
reality check beyond BPT numbers.

Uses greedy decoding (temperature=0) for reproducibility and
top-k sampling (k=40, temp=0.9) for diversity assessment.

Run: python code/gen_quality_test.py [--checkpoint PATH] [--device cpu|cuda]
"""

import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from launch_v060a import create_v060a

DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
SEQ_LEN = 96
VOCAB_SIZE = 50257

# Diverse test prompts covering different text types
TEST_PROMPTS = [
    # Simple continuation
    "The cat sat on the",
    "Once upon a time, there was a",
    # Knowledge / factual
    "The capital of France is",
    "Water freezes at a temperature of",
    "The largest planet in our solar system is",
    # Reasoning
    "If it rains, the ground gets wet. It rained yesterday, so",
    "Two plus two equals four. Three plus three equals",
    # Code-like
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "for i in range(10):\n    print(",
    # Longer context
    "The quick brown fox jumps over the lazy dog. The dog was sleeping peacefully in the sun when suddenly",
    # Narrative
    "She opened the door and found",
    "In the year 2050, humanity had finally",
    # Technical
    "Machine learning models are trained by",
    "The transformer architecture uses attention to",
    # Conversational
    "Question: What is the meaning of life?\nAnswer:",
]


def generate(model, tokenizer, prompt_text, max_new=80, temperature=0.0, top_k=40,
             device="cpu", rep_penalty=1.0, rep_window=64, no_repeat_ngram=0):
    """Generate text from a prompt.

    Args:
        temperature: 0.0 = greedy, >0 = sampling
        top_k: only used when temperature > 0
        rep_penalty: multiplicative penalty for tokens seen in last rep_window tokens.
            1.0 = disabled. 1.2-1.5 typical. Divides positive logits, multiplies negative.
        rep_window: how many recent tokens to check for repetition penalty
        no_repeat_ngram: block exact n-gram repetition (0 = disabled, 3+ typical)
    """
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new):
            # Use last SEQ_LEN tokens as context
            ctx = tokens[:, -SEQ_LEN:] if tokens.size(1) > SEQ_LEN else tokens
            logits, _ = model(ctx, y=None, collect_history=False)
            next_logits = logits[:, -1, :].float()

            # Repetition penalty: penalize tokens seen in recent window
            if rep_penalty != 1.0:
                recent = tokens[0, -rep_window:].tolist()
                for tid in set(recent):
                    if next_logits[0, tid] > 0:
                        next_logits[0, tid] /= rep_penalty
                    else:
                        next_logits[0, tid] *= rep_penalty

            # N-gram blocking: prevent exact n-gram repetition
            if no_repeat_ngram > 0 and tokens.size(1) >= no_repeat_ngram:
                gen_so_far = tokens[0].tolist()
                ngram_prefix = tuple(gen_so_far[-(no_repeat_ngram - 1):])
                for i in range(len(gen_so_far) - no_repeat_ngram + 1):
                    if tuple(gen_so_far[i:i + no_repeat_ngram - 1]) == ngram_prefix:
                        blocked_id = gen_so_far[i + no_repeat_ngram - 1]
                        next_logits[0, blocked_id] = float("-inf")

            if temperature <= 0:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # Top-k sampling
                next_logits = next_logits / temperature
                topk_vals, topk_idx = next_logits.topk(top_k)
                filtered = torch.full_like(next_logits, float("-inf"))
                filtered.scatter_(-1, topk_idx, topk_vals)
                probs = F.softmax(filtered, dim=-1)
                next_token = torch.multinomial(probs, 1)

            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop on EOS or newline-heavy output
            if next_token.item() == tokenizer.eos_token_id:
                break

    gen_ids = tokens[0, len(prompt_ids):].cpu().tolist()
    gen_text = tokenizer.decode(gen_ids)
    return gen_text


def assess_quality(gen_text):
    """Simple heuristic quality assessment."""
    scores = {}

    # Length (did it generate anything?)
    scores["length"] = len(gen_text.split())

    # Repetition (count repeated n-grams)
    words = gen_text.lower().split()
    if len(words) > 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / max(len(trigrams), 1)
        scores["trigram_diversity"] = round(unique_ratio, 3)
    else:
        scores["trigram_diversity"] = 0.0

    # ASCII safety (no garbage characters)
    ascii_chars = sum(1 for c in gen_text if ord(c) < 128)
    scores["ascii_ratio"] = round(ascii_chars / max(len(gen_text), 1), 3)

    # Word diversity
    if words:
        scores["word_diversity"] = round(len(set(words)) / len(words), 3)
    else:
        scores["word_diversity"] = 0.0

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new", type=int, default=80)
    args = parser.parse_args()

    device = torch.device(args.device)

    # Find checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"

    print(f"Generation Quality Test — Sutra v0.6.0a")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Device: {device}")
    print(f"  Max new tokens: {args.max_new}")

    # Load model
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    step = ckpt.get("step", "?")
    best_bpt = ckpt.get("best_bpt", "?")
    print(f"  Step: {step}, Best BPT: {best_bpt}")

    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    del ckpt

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    results = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*70}")
        print(f"PROMPT {i+1}/{len(TEST_PROMPTS)}: {prompt[:60]}...")
        print(f"{'='*70}")

        # Greedy generation
        greedy_text = generate(model, tokenizer, prompt, max_new=args.max_new,
                              temperature=0.0, device=device)
        greedy_quality = assess_quality(greedy_text)

        print(f"\n  [GREEDY] {greedy_text[:200]}")
        print(f"  Quality: {greedy_quality}")

        # Sampled generation
        sampled_text = generate(model, tokenizer, prompt, max_new=args.max_new,
                               temperature=0.9, top_k=40, device=device)
        sampled_quality = assess_quality(sampled_text)

        print(f"\n  [SAMPLED] {sampled_text[:200]}")
        print(f"  Quality: {sampled_quality}")

        # Anti-repetition generation (rep penalty + n-gram blocking)
        antirep_text = generate(model, tokenizer, prompt, max_new=args.max_new,
                                temperature=0.9, top_k=40, device=device,
                                rep_penalty=1.3, rep_window=64, no_repeat_ngram=3)
        antirep_quality = assess_quality(antirep_text)

        print(f"\n  [ANTI-REP] {antirep_text[:200]}")
        print(f"  Quality: {antirep_quality}")

        results.append({
            "prompt": prompt,
            "greedy": {"text": greedy_text[:500], "quality": greedy_quality},
            "sampled": {"text": sampled_text[:500], "quality": sampled_quality},
            "antirep": {"text": antirep_text[:500], "quality": antirep_quality},
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (step {step})")
    print(f"{'='*70}")

    for mode in ["greedy", "sampled", "antirep"]:
        avg_diversity = sum(r[mode]["quality"]["trigram_diversity"] for r in results) / len(results)
        avg_ascii = sum(r[mode]["quality"]["ascii_ratio"] for r in results) / len(results)
        avg_word_div = sum(r[mode]["quality"]["word_diversity"] for r in results) / len(results)
        avg_length = sum(r[mode]["quality"]["length"] for r in results) / len(results)

        label = {"greedy": "Greedy", "sampled": "Sampled (t=0.9, k=40)",
                 "antirep": "Anti-rep (t=0.9, k=40, rep=1.3, ngram=3)"}[mode]
        print(f"  {label}:")
        print(f"    Trigram diversity: {avg_diversity:.3f} (1.0 = no repetition)")
        print(f"    ASCII ratio: {avg_ascii:.3f} (1.0 = clean text)")
        print(f"    Word diversity: {avg_word_div:.3f}")
        print(f"    Avg length: {avg_length:.0f} words")

    # Save
    output = {
        "test": "generation_quality",
        "checkpoint": ckpt_path.name,
        "step": step,
        "best_bpt": best_bpt,
        "prompts": len(TEST_PROMPTS),
        "max_new_tokens": args.max_new,
        "summary": {mode: {
            "trigram_diversity": round(sum(r[mode]["quality"]["trigram_diversity"] for r in results) / len(results), 3),
            "ascii_ratio": round(sum(r[mode]["quality"]["ascii_ratio"] for r in results) / len(results), 3),
            "word_diversity": round(sum(r[mode]["quality"]["word_diversity"] for r in results) / len(results), 3),
            "avg_length": round(sum(r[mode]["quality"]["length"] for r in results) / len(results), 1),
        } for mode in ["greedy", "sampled", "antirep"]},
        "results": results,
    }

    out_path = REPO / "results" / f"gen_quality_step{step}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
