"""Probe D: Scratchpad Load-Bearing Audit (CPU)

Tests whether scratchpad is truly load-bearing by evaluating in 4 modes:
  1. FULL — normal forward pass (baseline)
  2. NO_READ — scratchpad.read returns zeros (write still active)
  3. NO_WRITE — scratchpad.write returns initial memory (read still active)
  4. REMOVED — both read and write return zeros/initial

Measures: full-vocab BPT per mode, plus a synthetic long-range recall test.

Requested by Tesla+Leibniz Round 1, Probe D.
"""

import sys, json, time, math
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn.functional as F
from datasets import load_dataset

# Force CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_model(ckpt_path):
    """Load v0.6.0a from checkpoint."""
    from launch_v060a import SutraV060a
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SutraV060a(
        vocab_size=50257, dim=768, ff_dim=1536,
        max_steps=12, window=4, k_retrieval=4, n_scratch_slots=8,
    )
    # Load state dict, handling potential key mismatches
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def patch_scratchpad(model, mode):
    """Monkey-patch scratchpad for ablation modes.

    mode: 'full', 'no_read', 'no_write', 'removed'
    Returns: unpatch function
    """
    scratch = model.scratchpad
    orig_read = scratch.read
    orig_write = scratch.write

    if mode == 'full':
        return lambda: None  # no-op

    if mode in ('no_read', 'removed'):
        def zero_read(h, mem):
            return torch.zeros_like(h)
        scratch.read = zero_read

    if mode in ('no_write', 'removed'):
        def identity_write(h, mem, pi_write):
            return mem  # return unchanged memory
        scratch.write = identity_write

    def unpatch():
        scratch.read = orig_read
        scratch.write = orig_write

    return unpatch


def eval_bpt(model, tokenizer, n_batches=20, batch_size=2, seq_len=512):
    """Compute full-vocab BPT on held-out data."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([t for t in ds["text"] if len(t) > 50])
    tokens = tokenizer.encode(text)

    total_loss = 0.0
    total_tokens = 0

    for i in range(n_batches):
        start = i * batch_size * seq_len
        batch_tokens = []
        for b in range(batch_size):
            s = start + b * seq_len
            if s + seq_len + 1 > len(tokens):
                break
            batch_tokens.append(tokens[s:s + seq_len + 1])

        if len(batch_tokens) < batch_size:
            break

        x = torch.tensor([t[:-1] for t in batch_tokens])
        y = torch.tensor([t[1:] for t in batch_tokens])

        with torch.no_grad():
            logits, _ = model(x, collect_history=False)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')

        total_loss += loss.item()
        total_tokens += y.numel()

    bpt = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpt, total_tokens


def recall_test(model, tokenizer, n_tests=50, context_len=256, target_distance=128):
    """Synthetic long-range recall test.

    Creates sequences where a key fact appears early, and the model must
    predict a token that depends on it. Measures if the model's predictions
    are better with scratchpad (which should help recall).

    Method: Take a natural text, identify a repeated word, check if the model
    assigns higher probability to the repeated word at its second occurrence.
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([t for t in ds["text"] if len(t) > 200])
    all_tokens = tokenizer.encode(text)

    # Find positions where a token repeats with distance >= target_distance
    correct = 0
    total = 0
    avg_rank = 0.0

    for start_offset in range(0, len(all_tokens) - context_len * 2, context_len):
        if total >= n_tests:
            break

        window = all_tokens[start_offset:start_offset + context_len]

        # Find a token that appears twice with sufficient distance
        seen = {}
        for pos, tok in enumerate(window):
            if tok in seen and pos - seen[tok] >= target_distance // 2:
                # Found a repeat — test if model predicts it
                context = window[:pos]
                target = tok

                x = torch.tensor([context])
                with torch.no_grad():
                    logits, _ = model(x, collect_history=False)
                    # Get prediction at the last position
                    probs = F.softmax(logits[0, -1], dim=-1)
                    rank = (probs > probs[target]).sum().item()

                    if rank < 100:  # top-100
                        correct += 1
                    avg_rank += rank
                    total += 1
                break
            seen[tok] = pos

    if total == 0:
        return {"accuracy_top100": 0.0, "avg_rank": float('inf'), "total": 0}

    return {
        "accuracy_top100": correct / total,
        "avg_rank": avg_rank / total,
        "total": total,
    }


def main():
    import tiktoken

    ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    model = load_model(str(ckpt_path))
    tokenizer = tiktoken.get_encoding("gpt2")

    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Scratchpad params: {sum(p.numel() for p in model.scratchpad.parameters()):,}")

    modes = ['full', 'no_read', 'no_write', 'removed']
    results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"Mode: {mode}")
        print(f"{'='*50}")

        unpatch = patch_scratchpad(model, mode)

        t0 = time.time()
        bpt, n_tokens = eval_bpt(model, tokenizer, n_batches=15, batch_size=2, seq_len=512)
        bpt_time = time.time() - t0
        print(f"  BPT: {bpt:.4f} ({n_tokens} tokens, {bpt_time:.1f}s)")

        t0 = time.time()
        recall = recall_test(model, tokenizer, n_tests=50)
        recall_time = time.time() - t0
        print(f"  Recall top-100: {recall['accuracy_top100']:.3f}, avg rank: {recall['avg_rank']:.1f} ({recall_time:.1f}s)")

        results[mode] = {
            "bpt": bpt,
            "n_tokens": n_tokens,
            "recall_top100": recall["accuracy_top100"],
            "recall_avg_rank": recall["avg_rank"],
            "recall_total": recall["total"],
        }

        unpatch()

    # Compute deltas
    baseline = results['full']
    for mode in modes:
        if mode != 'full':
            r = results[mode]
            r['bpt_delta'] = r['bpt'] - baseline['bpt']
            r['bpt_delta_pct'] = (r['bpt'] - baseline['bpt']) / baseline['bpt'] * 100
            r['recall_delta'] = r['recall_top100'] - baseline['recall_top100']

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<12} {'BPT':>8} {'Delta':>8} {'Delta%':>8} {'Recall':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for mode in modes:
        r = results[mode]
        delta = r.get('bpt_delta', 0)
        delta_pct = r.get('bpt_delta_pct', 0)
        print(f"{mode:<12} {r['bpt']:>8.4f} {delta:>+8.4f} {delta_pct:>+7.2f}% {r['recall_top100']:>8.3f}")

    # Verdict
    removed_delta = results['removed'].get('bpt_delta_pct', 0)
    if abs(removed_delta) > 3.0:
        verdict = "LOAD-BEARING"
    elif abs(removed_delta) > 1.0:
        verdict = "MODERATE"
    else:
        verdict = "NEGLIGIBLE"

    results['verdict'] = verdict
    results['probe'] = 'D_scratchpad_audit'
    results['checkpoint_step'] = 'rolling_latest'

    out_path = REPO / "results" / "probe_d_scratchpad_audit.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nVerdict: {verdict}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
