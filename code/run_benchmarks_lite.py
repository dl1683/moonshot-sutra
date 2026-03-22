"""Lightweight benchmark scorer for Sutra — no lm-eval framework.

Runs: ARC-Easy, ARC-Challenge, HellaSwag, PIQA, SciQ, WinoGrande, LAMBADA.
Scores each by log-likelihood of choices. Saves JSON immediately.

Usage: python code/run_benchmarks_lite.py [--checkpoint PATH] [--device cuda|cpu]
"""

import sys, json, gc, torch, torch.nn.functional as F, math, time, argparse
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from transformers import AutoTokenizer
from datasets import load_dataset

# Save results incrementally so we don't lose work on crash
RESULTS_PATH = REPO / "results" / "sutra_lm_eval_results.json"

def save_partial(results):
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(checkpoint, device, version="auto"):
    """Load Sutra model, auto-detecting version from checkpoint config."""
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")

    # Auto-detect version from checkpoint
    if version == "auto":
        config = ckpt.get("config", {})
        if config.get("max_steps", 8) == 12 or "v060" in str(checkpoint):
            version = "v060a"
        else:
            version = "v054"

    if version == "v060a":
        from launch_v060a import create_v060a
        model = create_v060a(dim=768, ff_dim=1536, max_steps=12, window=4, k_retrieval=8)
    else:
        from launch_v054 import create_v054
        model = create_v054(dim=768, ff_dim=1536, max_steps=8, window=4, k_retrieval=8)

    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) and "step" in ckpt else "?"
    best_bpt = ckpt.get("best_bpt", "?") if isinstance(ckpt, dict) and "best_bpt" in ckpt else "?"
    del ckpt, state
    gc.collect()
    model.to(device).eval()
    print(f"Loaded {version} model (step {step}, BPT {best_bpt})")
    return model


def loglikelihood(model, tok, device, context, continuation):
    """Compute log-likelihood of continuation given context."""
    ctx_ids = tok.encode(context)
    cont_ids = tok.encode(continuation)
    all_ids = (ctx_ids + cont_ids)[-512:]
    ctx_len = len(all_ids) - len(cont_ids)
    input_ids = torch.tensor([all_ids], device=device)

    with torch.no_grad():
        logits, _ = model(input_ids)
    lp = F.log_softmax(logits[0].float(), dim=-1)

    ll = 0.0
    for k in range(max(ctx_len, 1), len(all_ids)):
        if k > 0 and k - 1 < lp.size(0):
            ll += lp[k - 1, all_ids[k]].item()
    del input_ids, logits, lp
    return ll


def score_arc(model, tok, device, config):
    ds = load_dataset("allenai/ai2_arc", config, split="test", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        q = item["question"]
        texts = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer = item["answerKey"]

        best_ll, best_label = float("-inf"), None
        for text, label in zip(texts, labels):
            ll = loglikelihood(model, tok, device, q + " ", text)
            if ll > best_ll:
                best_ll, best_label = ll, label
        if best_label == answer:
            correct += 1
        total += 1
        if (i + 1) % 200 == 0:
            print(f"  {config}: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"{config}: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def score_hellaswag(model, tok, device):
    ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])

        best_ll, best_idx = float("-inf"), 0
        for j, ending in enumerate(endings):
            ll = loglikelihood(model, tok, device, ctx + " ", ending)
            if ll > best_ll:
                best_ll, best_idx = ll, j
        if best_idx == label:
            correct += 1
        total += 1
        if (i + 1) % 500 == 0:
            print(f"  hellaswag: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"hellaswag: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def score_piqa(model, tok, device):
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        goal = item["goal"]
        sols = [item["sol1"], item["sol2"]]
        label = item["label"]

        best_ll, best_idx = float("-inf"), 0
        for j, sol in enumerate(sols):
            ll = loglikelihood(model, tok, device, goal + " ", sol)
            if ll > best_ll:
                best_ll, best_idx = ll, j
        if best_idx == label:
            correct += 1
        total += 1
        if (i + 1) % 200 == 0:
            print(f"  piqa: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"piqa: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def score_sciq(model, tok, device):
    ds = load_dataset("allenai/sciq", split="validation", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        q = item["question"]
        choices = [item["correct_answer"], item["distractor1"], item["distractor2"], item["distractor3"]]
        label = 0  # correct_answer is always first in our list

        best_ll, best_idx = float("-inf"), 0
        for j, choice in enumerate(choices):
            ll = loglikelihood(model, tok, device, q + " ", choice)
            if ll > best_ll:
                best_ll, best_idx = ll, j
        if best_idx == label:
            correct += 1
        total += 1
        if (i + 1) % 200 == 0:
            print(f"  sciq: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"sciq: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def score_winogrande(model, tok, device):
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        sentence = item["sentence"]
        opt1, opt2 = item["option1"], item["option2"]
        label = int(item["answer"]) - 1  # 1-indexed -> 0-indexed

        s1 = sentence.replace("_", opt1)
        s2 = sentence.replace("_", opt2)

        ll1 = loglikelihood(model, tok, device, "", s1)
        ll2 = loglikelihood(model, tok, device, "", s2)

        pred = 0 if ll1 > ll2 else 1
        if pred == label:
            correct += 1
        total += 1
        if (i + 1) % 200 == 0:
            print(f"  winogrande: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"winogrande: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


def score_lambada(model, tok, device):
    ds = load_dataset("lambada", split="test", trust_remote_code=True)
    correct = total = 0
    for i, item in enumerate(ds):
        text = item["text"]
        words = text.rsplit(" ", 1)
        if len(words) < 2:
            continue
        context, last_word = words

        # Encode separately, concat, THEN truncate (fixes ctx_len mismatch)
        ctx_ids = tok.encode(context)
        cont_ids = tok.encode(" " + last_word)  # Space matters for BPE
        all_ids = (ctx_ids + cont_ids)[-512:]
        ctx_len = len(all_ids) - len(cont_ids)  # Recalculate from truncated

        input_ids = torch.tensor([all_ids], device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)

        # Position ctx_len-1 predicts token at ctx_len (first continuation token)
        if ctx_len > 0 and ctx_len - 1 < logits.size(1):
            pred_id = logits[0, ctx_len - 1].argmax().item()
            if pred_id == cont_ids[0]:
                correct += 1
        total += 1
        del input_ids, logits
        if (i + 1) % 500 == 0:
            print(f"  lambada: {i+1}/{len(ds)} acc={correct/total:.3f}", flush=True)

    acc = correct / total
    print(f"lambada: {correct}/{total} = {acc:.4f}", flush=True)
    return {"accuracy": round(acc, 4), "correct": correct, "total": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(REPO / "results/v060a_best.pt"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = load_model(args.checkpoint, device)
    print(f"Model loaded on {device}", flush=True)

    # Load partial results if resuming from crash
    results = {}
    if RESULTS_PATH.exists():
        try:
            results = json.load(open(RESULTS_PATH))
            print(f"Resuming: {list(results.keys())} already done", flush=True)
        except Exception:
            pass

    t0 = time.time()
    print("\n=== BENCHMARKS ===", flush=True)

    # Run each benchmark, save after each, free memory between
    benchmarks = [
        ("sciq", lambda: score_sciq(model, tok, device)),
        ("piqa", lambda: score_piqa(model, tok, device)),
        ("winogrande", lambda: score_winogrande(model, tok, device)),
        ("arc_easy", lambda: score_arc(model, tok, device, "ARC-Easy")),
        ("arc_challenge", lambda: score_arc(model, tok, device, "ARC-Challenge")),
        ("hellaswag", lambda: score_hellaswag(model, tok, device)),
        ("lambada", lambda: score_lambada(model, tok, device)),
    ]

    for name, fn in benchmarks:
        if name in results:
            print(f"  {name}: SKIPPED (already done: {results[name]['accuracy']*100:.1f}%)", flush=True)
            continue
        results[name] = fn()
        save_partial(results)  # Save after EVERY benchmark
        free_mem()             # Free RAM/VRAM between benchmarks

    print(f"\n{'='*60}", flush=True)
    print(f"SUTRA BENCHMARKS ({time.time()-t0:.0f}s)", flush=True)
    print(f"{'='*60}", flush=True)
    for name, r in results.items():
        print(f"  {name:20s} {r['accuracy']*100:5.1f}%  ({r['correct']}/{r['total']})", flush=True)

    save_partial(results)
    print(f"\nSaved: {RESULTS_PATH}", flush=True)
