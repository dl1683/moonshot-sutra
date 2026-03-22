"""Recurrence Gate Experiment — THE critical test.

Question: Does D>1 actually beat D=1 at matched compute for Sutra at 68M?

Three tests:
  1. Pass-truncation sweep on latest checkpoint (where does quality come from?)
  2. Per-pass marginal BPT (how much does each pass contribute?)
  3. Matched-compute training comparison at dim=128:
     D=1 (wider FF), D=4, D=8, D=12 — same FLOPs per token

If D>1 doesn't win test 3, recurrence at 68M is dead.

Usage: CUDA_VISIBLE_DEVICES="" python code/probe_recurrence_gate.py [--test 1|2|3|all]
"""

import argparse, json, math, os, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

DEVICE = "cpu"
RESULTS_PATH = REPO / "results" / "probe_recurrence_gate.json"


def load_eval_data(tokenizer, n_batches=15, batch_size=2, seq_len=256):
    """Load WikiText-103 validation data."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if len(t) > 100])
    tokens = tokenizer.encode(text)
    batches = []
    for i in range(n_batches):
        start = i * batch_size * (seq_len + 1)
        if start + batch_size * (seq_len + 1) > len(tokens):
            break
        batch_tokens = []
        for b in range(batch_size):
            s = start + b * (seq_len + 1)
            batch_tokens.append(tokens[s:s + seq_len + 1])
        t = torch.tensor(batch_tokens)
        batches.append((t[:, :-1], t[:, 1:]))
    return batches


# ═══════════════════════════════════════════════════════════
# TEST 1: Pass truncation on latest checkpoint
# ═══════════════════════════════════════════════════════════

def test1_pass_truncation():
    """Evaluate BPT at truncated pass counts on the current checkpoint."""
    from transformers import AutoTokenizer
    from launch_v060a import create_v060a

    print("\n" + "=" * 60)
    print("TEST 1: Pass Truncation Sweep (latest checkpoint)")
    print("=" * 60)

    # Load checkpoint
    ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", "?")
    print(f"Checkpoint step: {step}")

    model = create_v060a()
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batches = load_eval_data(tokenizer, n_batches=15, batch_size=2, seq_len=256)
    print(f"Eval: {len(batches)} batches × 2 × 256")

    results = []
    pass_counts = [1, 2, 3, 4, 6, 8, 10, 11, 12]

    for D in pass_counts:
        # Temporarily override max_steps
        orig_max = model.max_steps
        model.max_steps = D

        total_ce = 0.0
        total_tokens = 0
        with torch.no_grad():
            for x, y in batches:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x, collect_history=False)
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     y.reshape(-1), reduction="sum").item()
                total_ce += ce
                total_tokens += y.numel()

        model.max_steps = orig_max
        bpt = total_ce / total_tokens / math.log(2)
        results.append({"passes": D, "bpt": round(bpt, 4)})
        print(f"  D={D:2d}: BPT = {bpt:.4f}")

    # Compute marginal gains
    print("\n  Per-pass marginal BPT improvement:")
    for i in range(1, len(results)):
        delta = results[i-1]["bpt"] - results[i]["bpt"]
        pct = delta / results[i-1]["bpt"] * 100 if results[i-1]["bpt"] > 0 else 0
        print(f"    D={results[i-1]['passes']}->{results[i]['passes']}: "
              f"delta={delta:+.4f} BPT ({pct:+.2f}%)")

    return {"test": "pass_truncation", "step": step, "results": results}


# ═══════════════════════════════════════════════════════════
# TEST 2: Per-pass full-vocab CE (honest, not sampled)
# ═══════════════════════════════════════════════════════════

def test2_perpass_ce():
    """Full-vocab CE at each pass — honest version (not sampled from final logits)."""
    from transformers import AutoTokenizer
    from launch_v060a import create_v060a

    print("\n" + "=" * 60)
    print("TEST 2: Per-Pass Full-Vocab CE (honest)")
    print("=" * 60)

    ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", "?")

    model = create_v060a()
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Fewer batches since this is expensive (full logits per pass)
    batches = load_eval_data(tokenizer, n_batches=8, batch_size=2, seq_len=256)

    per_pass_ce = [0.0] * 12
    total_tokens = 0

    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Run with history collection
            model.training = False
            logits, aux = model(x, y=y, collect_history=True)
            mu_hist = aux["mu_hist"]  # (B, T, 12, D)

            B, T = y.shape
            total_tokens += B * T

            # Full-vocab CE for each pass
            for p in range(12):
                mu_p = model.ln(mu_hist[:, :, p, :])  # (B, T, D)
                logits_p = F.linear(mu_p, model.emb.weight) / math.sqrt(model.dim)
                ce_p = F.cross_entropy(logits_p.reshape(-1, logits_p.size(-1)),
                                       y.reshape(-1), reduction="sum").item()
                per_pass_ce[p] += ce_p

    per_pass_bpt = [ce / total_tokens / math.log(2) for ce in per_pass_ce]

    print(f"\n  Step: {step}")
    print(f"  Per-pass full-vocab BPT:")
    for p in range(12):
        delta = per_pass_bpt[p-1] - per_pass_bpt[p] if p > 0 else 0
        bar = "#" * max(0, int(delta * 20))
        print(f"    Pass {p:2d}: BPT = {per_pass_bpt[p]:.4f}  "
              f"delta = {delta:+.4f}  {bar}")

    return {"test": "perpass_ce", "step": step, "per_pass_bpt": per_pass_bpt}


# ═══════════════════════════════════════════════════════════
# TEST 3: Matched-compute training comparison (dim=128)
# ═══════════════════════════════════════════════════════════

class MiniSutra(nn.Module):
    """Minimal Sutra-like model for gate testing. Strips out scratchpad,
    pheromone, frozen cache — just the core recurrent computation.

    At D=1, ff_dim is scaled up to match total FLOPs of D=12 baseline.
    """

    def __init__(self, vocab_size=50257, dim=128, ff_dim=256,
                 max_steps=12):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps

        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(512, dim)
        self.init_proj = nn.Linear(dim, dim)

        # The recurrent core: 2 FF layers + LN
        self.ln1 = nn.LayerNorm(dim)
        self.ff_up = nn.Linear(dim, ff_dim)
        self.ff_down = nn.Linear(ff_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mix = nn.Linear(dim * 2, dim)  # combine mu + residual

        self.ln_out = nn.LayerNorm(dim)

    def forward(self, x, y=None):
        B, T = x.shape
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=x.device))
        mu = self.init_proj(h)

        for p in range(self.max_steps):
            # One recurrent pass
            r = self.ln1(mu)
            r = self.ff_down(F.silu(self.ff_up(r)))
            r = self.ln2(r)
            mu = self.mix(torch.cat([mu, r], dim=-1))

        out = self.ln_out(mu)
        logits = F.linear(out, self.emb.weight) / math.sqrt(self.dim)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def test3_matched_compute():
    """Train MiniSutra at D=1,4,8,12 with matched FLOPs. D=1 gets wider FF."""
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("TEST 3: Matched-Compute Training (dim=128, 1000 steps)")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    batches = load_eval_data(tokenizer, n_batches=80, batch_size=4, seq_len=128)
    eval_batches = batches[-10:]
    train_batches = batches[:-10]

    if len(train_batches) < 20:
        print("  Not enough data, reducing batch count...")
        batches = load_eval_data(tokenizer, n_batches=120, batch_size=4, seq_len=128)
        eval_batches = batches[-10:]
        train_batches = batches[:-10]

    dim = 128
    # D=12 baseline: ff_dim=256. FLOPs per pass ≈ 2*dim*ff_dim + dim*dim*2 = 2*128*256 + 2*128*128 = 98304
    # Total FLOPs for D=12 ≈ 12 * 98304 = 1,179,648
    # For D=1 to match: ff_dim_1 such that 1 * (2*dim*ff + 2*dim*dim) ≈ 1,179,648
    # 2*128*ff + 2*128*128 ≈ 1,179,648
    # 256*ff ≈ 1,179,648 - 32768 = 1,146,880
    # ff ≈ 4480
    # For D=4: 4 * (2*128*ff + 32768) ~= 1,179,648 -> ff ~= 832
    # For D=8: 8 * (2*128*ff + 32768) ~= 1,179,648 -> ff ~= 448

    configs = [
        {"name": "D=1",  "max_steps": 1,  "ff_dim": 4480},
        {"name": "D=4",  "max_steps": 4,  "ff_dim": 832},
        {"name": "D=8",  "max_steps": 8,  "ff_dim": 448},
        {"name": "D=12", "max_steps": 12, "ff_dim": 256},
    ]

    # LR sweep: pick best LR for each config (prevents divergence like D=8 NaN)
    lr_options = [1e-3, 5e-4, 3e-4, 1e-4]
    n_train_steps = 1000
    results = []

    for cfg in configs:
        print(f"\n  --- {cfg['name']} (ff_dim={cfg['ff_dim']}) ---")
        best_bpt = float("inf")
        best_lr = None

        for lr in lr_options:
            torch.manual_seed(42)
            model = MiniSutra(dim=dim, ff_dim=cfg["ff_dim"],
                              max_steps=cfg["max_steps"])
            n_params = model.count_params()
            opt = torch.optim.AdamW(model.parameters(), lr=lr,
                                    weight_decay=0.01, betas=(0.9, 0.95))

            model.train()
            step = 0
            diverged = False
            t0 = time.time()

            for epoch in range(max(1, n_train_steps // len(train_batches) + 1)):
                for x, y in train_batches:
                    if step >= n_train_steps:
                        break
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                           y.reshape(-1))
                    if torch.isnan(loss) or torch.isinf(loss):
                        diverged = True
                        break
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    step += 1
                if diverged or step >= n_train_steps:
                    break

            wall_time = time.time() - t0

            if diverged:
                print(f"    LR={lr:.0e}: DIVERGED at step {step}")
                continue

            # Eval
            model.eval()
            total_ce = 0.0
            total_tokens = 0
            with torch.no_grad():
                for x, y in eval_batches:
                    logits = model(x)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                         y.reshape(-1), reduction="sum").item()
                    total_ce += ce
                    total_tokens += y.numel()
            bpt = total_ce / total_tokens / math.log(2)
            print(f"    LR={lr:.0e}: BPT={bpt:.4f} ({wall_time:.1f}s, {n_params:,} params)")

            if bpt < best_bpt:
                best_bpt = bpt
                best_lr = lr

        if best_lr is not None:
            results.append({
                "config": cfg["name"],
                "max_steps": cfg["max_steps"],
                "ff_dim": cfg["ff_dim"],
                "best_lr": best_lr,
                "best_bpt": round(best_bpt, 4),
                "params": n_params,
            })
            print(f"  ★ Best: LR={best_lr:.0e}, BPT={best_bpt:.4f}")
        else:
            results.append({
                "config": cfg["name"],
                "max_steps": cfg["max_steps"],
                "ff_dim": cfg["ff_dim"],
                "best_lr": None,
                "best_bpt": float("inf"),
                "params": n_params,
                "note": "ALL LRs DIVERGED",
            })
            print(f"  ✗ ALL LRs DIVERGED")

    # Summary
    print("\n" + "=" * 60)
    print("MATCHED-COMPUTE GATE TEST SUMMARY")
    print("=" * 60)
    print(f"  {'Config':<8} {'FF_dim':<8} {'Params':<10} {'Best BPT':<10} {'Best LR':<8}")
    for r in results:
        lr_str = f"{r['best_lr']:.0e}" if r.get("best_lr") else "N/A"
        bpt_str = f"{r['best_bpt']:.4f}" if r['best_bpt'] < float("inf") else "FAILED"
        print(f"  {r['config']:<8} {r['ff_dim']:<8} {r.get('params', '?'):<10} {bpt_str:<10} {lr_str:<8}")

    if len(results) >= 2:
        d12 = next((r for r in results if r["max_steps"] == 12), None)
        d1 = next((r for r in results if r["max_steps"] == 1), None)
        if d12 and d1 and d12["best_bpt"] < float("inf") and d1["best_bpt"] < float("inf"):
            if d12["best_bpt"] < d1["best_bpt"]:
                gap = d1["best_bpt"] - d12["best_bpt"]
                print(f"\n  ✓ RECURRENCE WINS: D=12 beats D=1 by {gap:.4f} BPT at matched compute")
            else:
                gap = d12["best_bpt"] - d1["best_bpt"]
                print(f"\n  ✗ RECURRENCE LOSES: D=1 beats D=12 by {gap:.4f} BPT at matched compute")

    return {"test": "matched_compute", "dim": dim, "steps": n_train_steps, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="all", help="1, 2, 3, or all")
    args = parser.parse_args()

    all_results = {}

    if args.test in ("1", "all"):
        all_results["test1"] = test1_pass_truncation()

    if args.test in ("2", "all"):
        all_results["test2"] = test2_perpass_ce()

    if args.test in ("3", "all"):
        all_results["test3"] = test3_matched_compute()

    # Save
    json.dump(all_results, open(RESULTS_PATH, "w"), indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
