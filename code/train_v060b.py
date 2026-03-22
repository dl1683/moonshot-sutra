"""Sutra v0.6.0b Training: Random-depth (rd12) warm-start from v0.6.0a.

Purpose: Fix early-pass collapse by training with random depth.
Each batch samples D ~ P(D=d) = d^alpha / Z for d in {1..12}.
Alpha ramps from 1.0 to 2.0 over first 1K steps.

Loss = L_final + 0.20 * L_probe (NO L_step — random depth makes it meaningless)

Warm-start: loads v0.6.0a best weights, RESETS optimizer (WSD restart).
Fresh cosine LR schedule, 500-step warmup, 3K total steps.

Success criteria (from Codex R4):
  - late_pct drops from 91.5% to <70%
  - Passes 1-6 contribute materially to BPT
  - Final BPT matches or beats source checkpoint
  - Greedy trigram diversity improves over 0.265

Usage: python code/train_v060b.py
"""

import json, math, os, random, time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 16         # Effective batch = 64
LR = 3.5e-4
WARMUP_STEPS = 500
MAX_TRAIN_STEPS = 3000  # Short warm-start: 3K steps
EVAL_EVERY = 500
SAVE_EVERY = 1000
ROLLING_SAVE = 100
LOG_EVERY = 50
VOCAB_SIZE = 50257
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Random-depth alpha schedule
ALPHA_START = 1.0
ALPHA_END = 2.0
ALPHA_RAMP_STEPS = 1000

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v060a import create_v060a, build_negative_set, sampled_pass_ce
from data_loader import ShardedDataset


def sample_depth(max_d, alpha):
    """Sample depth D ~ P(D=d) = d^alpha / Z for d in {1..max_d}."""
    weights = torch.arange(1, max_d + 1, dtype=torch.float) ** alpha
    return torch.multinomial(weights, 1).item() + 1


def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compute_v060b_losses(logits, aux, y):
    """Two-part loss: final CE + probe calibration. NO L_step."""
    B, T, V = logits.shape
    n_steps = aux["avg_steps"]

    # 1. L_final: full-vocab CE on final pass (at sampled depth)
    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    # 2. L_probe: residual-gain prediction (over visited passes only)
    L_probe = torch.tensor(0.0, device=logits.device)
    if "probe_pred" in aux and "sampled_ce_hist" in aux:
        probe_pred = aux["probe_pred"]  # (B, T, n_steps)
        sampled_ce = aux["sampled_ce_hist"]  # (B, T, n_steps)
        with torch.no_grad():
            targets = torch.zeros_like(sampled_ce)
            for p in range(n_steps - 1):
                future_min = sampled_ce[:, :, p+1:].min(dim=2).values
                targets[:, :, p] = (sampled_ce[:, :, p] - future_min).clamp(min=0, max=4)
        L_probe = F.smooth_l1_loss(probe_pred, targets.detach())

    L_total = L_final + PROBE_LOSS_COEF * L_probe

    return L_total, {
        "L_final": L_final.item(),
        "L_probe": L_probe.item(),
        "L_total": L_total.item(),
        "depth": n_steps,
    }


def evaluate(model, dataset, n_batches=20):
    """Evaluate at full depth (12 passes) for fair comparison."""
    model.eval()
    total_loss = 0
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(min(BATCH_SIZE, 4), SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
                    # Always eval at full depth for fair comparison with v0.6.0a
                    logits, _ = model(x, n_steps=MAX_STEPS)
                    Tc = min(logits.size(1), y.size(1))
                    loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                total_loss += loss.item()
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"bpt": (total_loss / n_batches) / math.log(2)}


def evaluate_per_depth(model, dataset, n_batches=10):
    """Evaluate at each depth 1..12 to measure per-pass contribution."""
    model.eval()
    results = {}
    try:
        with torch.no_grad():
            for d in range(1, MAX_STEPS + 1):
                total = 0
                for _ in range(n_batches):
                    x, y = dataset.sample_batch(2, SEQ_LEN, device=DEVICE, split="test")
                    with autocast_ctx():
                        logits, _ = model(x, n_steps=d)
                        Tc = min(logits.size(1), y.size(1))
                        loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                    total += loss.item()
                results[d] = round((total / n_batches) / math.log(2), 4)
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def main():
    print(f"SUTRA v0.6.0b: RANDOM-DEPTH (rd12) WARM-START")
    print(f"  Random depth d~d^alpha/Z, alpha ramp {ALPHA_START}->{ALPHA_END} over {ALPHA_RAMP_STEPS} steps")
    print(f"  Loss = L_final + {PROBE_LOSS_COEF} * L_probe (NO L_step)")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}, {MAX_TRAIN_STEPS} steps")
    print(f"{'='*60}")

    dataset = ShardedDataset()

    ckpt_dir = REPO / "results" / "checkpoints_v060b"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v060b_log.txt"
    metrics_file = REPO / "results" / "v060b_metrics.json"

    # Create model
    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                         window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # Warm-start: load v0.6.0a best weights
    source_ckpt = REPO / "results" / "v060a_best.pt"
    if not source_ckpt.exists():
        print(f"ERROR: Source checkpoint not found: {source_ckpt}")
        print("Train v0.6.0a to at least 14K steps first.")
        return

    state_dict = torch.load(source_ckpt, weights_only=True, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded v0.6.0a weights from {source_ckpt}")
    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Check for resume
    rolling = ckpt_dir / "rolling_latest.pt"
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    depth_history = []

    if rolling.exists():
        try:
            ckpt = torch.load(rolling, weights_only=False, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            start_step = ckpt["step"]
            best_bpt = ckpt.get("best_bpt", float("inf"))
            metrics_history = ckpt.get("metrics", [])
            depth_history = ckpt.get("depth_history", [])
            print(f"RESUMED v0.6.0b from step {start_step}")
        except Exception as e:
            print(f"Failed to resume: {e}. Starting fresh from v0.6.0a.")

    # FRESH optimizer (WSD restart — do NOT load optimizer state)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))

    model.train()
    step = start_step
    running_losses = {"L_final": 0, "L_probe": 0, "L_total": 0}
    running_depth = 0
    loss_count = 0
    start = time.time()

    # Depth distribution tracking
    depth_counts = [0] * (MAX_STEPS + 1)

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        # Cosine LR with warmup
        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS)
                                 / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS))))
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Sample random depth
        alpha = ALPHA_START + min(step / ALPHA_RAMP_STEPS, 1.0) * (ALPHA_END - ALPHA_START)
        D = sample_depth(MAX_STEPS, alpha)
        depth_counts[D] += 1

        with autocast_ctx():
            logits, aux = model(x, y=y, n_steps=D)
            Tc = min(logits.size(1), y.size(1))
            loss, loss_parts = compute_v060b_losses(logits[:, :Tc], aux, y[:, :Tc])
            loss = loss / GRAD_ACCUM

        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss at step {step} (D={D}), skipping", flush=True)
            opt.zero_grad()
            loss_count = 0
            running_losses = {k: 0 for k in running_losses}
            running_depth = 0
            continue

        loss.backward()
        for k, v in loss_parts.items():
            if isinstance(v, (int, float)):
                if k in running_losses:
                    running_losses[k] += v
        running_depth += D
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                print(f"WARNING: NaN/Inf grad at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                continue

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avgs = {k: v / loss_count for k, v in running_losses.items()}
                avg_d = running_depth / loss_count
                elapsed = time.time() - start
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                msg = (f"Step {step:>5d}: L={avgs['L_total']:.4f} "
                       f"(fin={avgs['L_final']:.3f} prb={avgs['L_probe']:.3f}) "
                       f"D={avg_d:.1f} alpha={alpha:.2f} lr={lr:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                loss_count = 0

            if step % ROLLING_SAVE == 0:
                atomic_torch_save({
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history, "depth_history": depth_history,
                    "depth_counts": depth_counts,
                }, ckpt_dir / "rolling_latest.pt")

            if step % EVAL_EVERY == 0:
                try:
                    # Full-depth eval
                    metrics = evaluate(model, dataset)
                    bpt = metrics["bpt"]
                    is_best = bpt < best_bpt
                    if is_best:
                        best_bpt = bpt
                        atomic_torch_save(model.state_dict(), REPO / "results" / "v060b_best.pt")

                    entry = {
                        "step": step, "test_bpt": round(bpt, 4),
                        "best_bpt": round(best_bpt, 4), "is_best": is_best,
                        "lr": lr, "alpha": round(alpha, 3),
                        "timestamp": datetime.now().isoformat(),
                    }
                    metrics_history.append(entry)

                    # Per-depth eval (key diagnostic for rd12 success)
                    per_depth = evaluate_per_depth(model, dataset, n_batches=5)
                    entry["per_depth_bpt"] = per_depth
                    depth_history.append({"step": step, "per_depth": per_depth})

                    json.dump(metrics_history, open(metrics_file, "w"), indent=2)

                    marker = " *BEST*" if is_best else ""
                    eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{marker}"
                    print(eval_msg, flush=True)

                    # Print per-depth summary
                    late_bpt = per_depth.get(12, 99)
                    early_bpt = per_depth.get(3, 99)
                    mid_bpt = per_depth.get(6, 99)
                    d1_bpt = per_depth.get(1, 99)
                    depth_msg = (f"    D=1:{d1_bpt:.2f} D=3:{early_bpt:.2f} "
                                 f"D=6:{mid_bpt:.2f} D=12:{late_bpt:.2f}")
                    print(depth_msg, flush=True)

                    # Compute late_pct (success metric)
                    if d1_bpt > 0 and late_bpt > 0:
                        total_improv = d1_bpt - late_bpt
                        late_improv = mid_bpt - late_bpt
                        late_pct = (late_improv / total_improv * 100) if total_improv > 0 else 100
                        print(f"    late_pct: {late_pct:.1f}% (target: <70%)", flush=True)

                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n" + depth_msg + "\n")

                except Exception as e:
                    print(f"EVAL FAILED: {e}", flush=True)

            if step % SAVE_EVERY == 0:
                atomic_torch_save({
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history, "depth_history": depth_history,
                    "depth_counts": depth_counts,
                }, ckpt_dir / f"step_{step}.pt")

    # Final summary
    print(f"\n{'='*60}")
    print(f"v0.6.0b COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
    print(f"\nDepth distribution:")
    total_samples = sum(depth_counts)
    for d in range(1, MAX_STEPS + 1):
        pct = depth_counts[d] / max(total_samples, 1) * 100
        print(f"  D={d:2d}: {depth_counts[d]:5d} ({pct:5.1f}%)")

    # Final per-depth eval
    print(f"\nFinal per-depth BPT:")
    per_depth = evaluate_per_depth(model, dataset, n_batches=10)
    for d in sorted(per_depth.keys()):
        print(f"  D={d:2d}: {per_depth[d]:.4f}")


if __name__ == "__main__":
    main()
