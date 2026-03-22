"""Sutra v0.6.1 Training: Warm-start probe of adaLN pass conditioning.

This is a PROBE, not a committed direction. Tests whether pass-conditioned
normalization reduces attractor collapse and improves BPT.

Changes from v0.6.0a trainer:
  - Warm-starts from v0.6.0a best checkpoint
  - Lower LR (1e-4 vs 3.5e-4) — warm-start needs less
  - Shorter warmup (500 vs 1500)
  - Default 5K steps (not 100K) — just enough to validate
  - Same 3-part loss (L_final + L_step + L_probe)

Gate: BPT improves AND pass-to-pass cosine collapse decreases.
Kill: No improvement after 3K steps, or collapse worsens.

Usage: python code/train_v061.py [--from_scratch]
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

# Architecture (same as v0.6.0a)
DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 16
VOCAB_SIZE = 50257
STEP_LOSS_COEF = 0.25
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Warm-start training recipe (lower LR, shorter)
LR = 1.0e-4          # Lower than v0.6.0a (3.5e-4)
WARMUP_STEPS = 500    # Shorter warmup
MAX_TRAIN_STEPS = 5000  # Just enough to validate
EVAL_EVERY = 500      # More frequent eval for short run
SAVE_EVERY = 1000
ROLLING_SAVE = 100
LOG_EVERY = 50        # More frequent logging

STEP_WEIGHTS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.00]

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v061 import create_v061, warm_start_from_v060a
from data_loader import ShardedDataset


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


def compute_losses(logits, aux, y):
    """Same 3-part loss as v0.6.0a."""
    B, T, V = logits.shape
    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    sampled_ce = aux["sampled_ce_hist"]
    weights = torch.tensor(STEP_WEIGHTS, device=sampled_ce.device, dtype=sampled_ce.dtype)
    L_step = (sampled_ce * weights.unsqueeze(0).unsqueeze(0)).mean()
    probe_pred = aux["probe_pred"]
    with torch.no_grad():
        targets = torch.zeros_like(sampled_ce)
        for p in range(11):
            future_min = sampled_ce[:, :, p+1:].min(dim=2).values
            targets[:, :, p] = (sampled_ce[:, :, p] - future_min).clamp(min=0, max=4)
    L_probe = F.smooth_l1_loss(probe_pred, targets.detach())
    L_total = L_final + STEP_LOSS_COEF * L_step + PROBE_LOSS_COEF * L_probe
    return L_total, {
        "L_final": L_final.item(),
        "L_step": L_step.item(),
        "L_probe": L_probe.item(),
        "L_total": L_total.item(),
    }


def evaluate(model, dataset, n_batches=20):
    """Evaluate + measure collapse metrics."""
    model.eval()
    total_loss = 0
    pass_bpt_accum = [0.0] * MAX_STEPS
    pass_cos_accum = [0.0] * (MAX_STEPS - 1)
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(min(BATCH_SIZE, 4), SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
                    logits, aux = model(x, y=y, collect_history=True)
                    Tc = min(logits.size(1), y.size(1))
                    loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                total_loss += loss.item()

                # Per-pass BPT from full-vocab CE
                if aux.get("mu_hist") is not None:
                    mu_hist = aux["mu_hist"]
                    for p in range(MAX_STEPS):
                        mu_p = model.ln(mu_hist[:, :Tc, p])
                        p_logits = F.linear(mu_p, model.emb.weight) / math.sqrt(model.dim)
                        p_ce = F.cross_entropy(p_logits.reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                        pass_bpt_accum[p] += p_ce.item() / math.log(2)

                    # Cosine similarity between consecutive passes
                    for p in range(1, MAX_STEPS):
                        cos = F.cosine_similarity(
                            mu_hist[:, :Tc, p].reshape(-1, DIM),
                            mu_hist[:, :Tc, p-1].reshape(-1, DIM), dim=-1
                        ).mean()
                        pass_cos_accum[p-1] += cos.item()
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_bpt = (total_loss / n_batches) / math.log(2)
    per_pass_bpt = [b / n_batches for b in pass_bpt_accum]
    per_pass_cos = [c / n_batches for c in pass_cos_accum]

    total_improv = per_pass_bpt[0] - per_pass_bpt[-1] if per_pass_bpt[0] > per_pass_bpt[-1] else 0
    late_improv = per_pass_bpt[7] - per_pass_bpt[-1] if per_pass_bpt[7] > per_pass_bpt[-1] else 0
    avg_cos = sum(per_pass_cos) / len(per_pass_cos) if per_pass_cos else 0

    return {
        "bpt": avg_bpt,
        "per_pass_bpt": [round(b, 4) for b in per_pass_bpt],
        "per_pass_cosine": [round(c, 4) for c in per_pass_cos],
        "total_improvement": round(total_improv, 4),
        "late_improvement": round(late_improv, 4),
        "late_pct": round(late_improv / total_improv * 100, 1) if total_improv > 0 else 0,
        "avg_cosine": round(avg_cos, 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train from scratch (no warm-start, for comparison)")
    parser.add_argument("--max_steps", type=int, default=MAX_TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    lr = args.lr
    max_steps = args.max_steps

    print(f"SUTRA v0.6.1: PASS-CONDITIONED NORMALIZATION (adaLN)")
    print(f"  {'FROM SCRATCH' if args.from_scratch else 'WARM-START from v0.6.0a'}")
    print(f"  This is a PROBE — validate pass conditioning hypothesis")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"LR={lr}, warmup={WARMUP_STEPS}, max_steps={max_steps}")
    print(f"{'='*60}")

    dataset = ShardedDataset()

    ckpt_dir = REPO / "results" / "checkpoints_v061"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v061_log.txt"
    metrics_file = REPO / "results" / "v061_metrics.json"

    model = create_v061(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                        window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)
    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Warm-start or resume
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    source_optimizer = None

    rolling = ckpt_dir / "rolling_latest.pt"
    permanents = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))

    # Try to resume v0.6.1 training first
    ckpt = None
    for cand in [rolling] + list(reversed(permanents)):
        if cand.exists():
            try:
                c = torch.load(cand, weights_only=False, map_location=DEVICE)
                if ckpt is None or c.get("step", 0) > ckpt.get("step", 0):
                    ckpt = c
                    ckpt["_path"] = cand.name
            except Exception as e:
                print(f"Skip corrupt checkpoint {cand.name}: {e}")

    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        best_bpt = ckpt.get("best_bpt", float("inf"))
        metrics_history = ckpt.get("metrics", [])
        source_optimizer = ckpt.get("optimizer", None)
        print(f"RESUMED v0.6.1 from step {start_step} ({ckpt['_path']})")
    elif not args.from_scratch:
        # Warm-start from v0.6.0a
        v060a_best = REPO / "results" / "v060a_best.pt"
        v060a_rolling = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
        v060a_step10k = REPO / "results" / "checkpoints_v060a" / "step_10000.pt"

        # Try best, then rolling, then latest permanent
        for src in [v060a_best, v060a_rolling, v060a_step10k]:
            if src.exists():
                try:
                    ws_info = warm_start_from_v060a(model, str(src))
                    print(f"Warm-started from: {src.name}")
                    # Don't load v0.6.0a optimizer — different param set
                    break
                except Exception as e:
                    print(f"Warm-start failed from {src.name}: {e}")
        else:
            print("WARNING: No v0.6.0a checkpoint found. Training from scratch.")
    else:
        print("Training from SCRATCH (--from_scratch flag)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    if source_optimizer is not None:
        try:
            opt.load_state_dict(source_optimizer)
        except Exception:
            print("  Could not load optimizer state (param mismatch expected)")

    # Initial eval before training
    print("\nInitial eval (before training)...")
    init_metrics = evaluate(model, dataset)
    print(f"  Initial BPT: {init_metrics['bpt']:.4f}")
    print(f"  Late improvement: {init_metrics['late_improvement']:.4f} ({init_metrics['late_pct']:.1f}%)")
    print(f"  Avg cosine: {init_metrics['avg_cosine']:.4f}")
    metrics_history.append({
        "step": start_step, "test_bpt": round(init_metrics["bpt"], 4),
        "best_bpt": round(init_metrics["bpt"], 4),
        "collapse": init_metrics, "is_initial": True,
        "timestamp": datetime.now().isoformat()
    })
    best_bpt = min(best_bpt, init_metrics["bpt"])

    model.train()
    step = start_step
    running_losses = {"L_final": 0, "L_step": 0, "L_probe": 0, "L_total": 0}
    loss_count = 0
    start = time.time()

    while step < max_steps:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        cur_lr = lr * min(1.0, (step - start_step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - start_step - WARMUP_STEPS)
                                 / max(1, max_steps - start_step - WARMUP_STEPS))))
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        with autocast_ctx():
            logits, aux = model(x, y=y)
            Tc = min(logits.size(1), y.size(1))
            loss, loss_parts = compute_losses(logits[:, :Tc], aux, y[:, :Tc])
            loss = loss / GRAD_ACCUM

        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite loss at step {step}, skipping", flush=True)
            opt.zero_grad()
            loss_count = 0
            running_losses = {k: 0 for k in running_losses}
            continue

        loss.backward()
        for k, v in loss_parts.items():
            running_losses[k] += v
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                print(f"WARNING: NaN/Inf grad at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_losses = {k: 0 for k in running_losses}
                continue

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avgs = {k: v / loss_count for k, v in running_losses.items()}
                elapsed = time.time() - start
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                msg = (f"Step {step:>6d}: L={avgs['L_total']:.4f} "
                       f"(fin={avgs['L_final']:.3f} stp={avgs['L_step']:.3f} prb={avgs['L_probe']:.3f}) "
                       f"lr={cur_lr:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                loss_count = 0

            if step % ROLLING_SAVE == 0:
                atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "step": step, "best_bpt": best_bpt}, ckpt_dir / "rolling_latest.pt")

            if step % EVAL_EVERY == 0:
                try:
                    metrics = evaluate(model, dataset)
                    bpt = metrics["bpt"]
                    is_best = bpt < best_bpt
                    if is_best:
                        best_bpt = bpt
                        atomic_torch_save(model.state_dict(), REPO / "results" / "v061_best.pt")
                    entry = {
                        "step": step, "test_bpt": round(bpt, 4),
                        "best_bpt": round(best_bpt, 4), "is_best": is_best,
                        "lr": cur_lr, "collapse": metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    metrics_history.append(entry)
                    json.dump(metrics_history, open(metrics_file, "w"), indent=2)
                    marker = " *BEST*" if is_best else ""
                    eval_msg = (f"  EVAL Step {step}: BPT={bpt:.4f}{marker} "
                                f"late_improv={metrics['late_improvement']:.4f} "
                                f"({metrics['late_pct']:.1f}%) "
                                f"avg_cos={metrics['avg_cosine']:.4f}")
                    print(eval_msg, flush=True)
                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n")
                except Exception as e:
                    print(f"EVAL FAILED: {e}", flush=True)

            if step % SAVE_EVERY == 0:
                atomic_torch_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                                   "step": step, "best_bpt": best_bpt, "metrics": metrics_history},
                                  ckpt_dir / f"step_{step}.pt")

    # Final summary
    print(f"\n{'='*60}")
    print(f"v0.6.1 PROBE COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
    if len(metrics_history) >= 2:
        init = metrics_history[0]
        final = metrics_history[-1]
        bpt_delta = final["test_bpt"] - init["test_bpt"]
        cos_delta = final["collapse"]["avg_cosine"] - init["collapse"]["avg_cosine"]
        late_delta = final["collapse"]["late_improvement"] - init["collapse"]["late_improvement"]
        print(f"  BPT: {init['test_bpt']:.4f} -> {final['test_bpt']:.4f} ({bpt_delta:+.4f})")
        print(f"  Avg cosine: {init['collapse']['avg_cosine']:.4f} -> {final['collapse']['avg_cosine']:.4f} ({cos_delta:+.4f})")
        print(f"  Late improvement: {init['collapse']['late_improvement']:.4f} -> {final['collapse']['late_improvement']:.4f} ({late_delta:+.4f})")

        # Verdict
        bpt_improved = bpt_delta < -0.05
        collapse_reduced = cos_delta < -0.005
        if bpt_improved and collapse_reduced:
            print("  VERDICT: KEEP — both BPT and collapse improved")
        elif bpt_improved:
            print("  VERDICT: KEEP (BPT) — BPT improved, collapse inconclusive")
        elif collapse_reduced:
            print("  VERDICT: INCONCLUSIVE — collapse reduced but BPT not improved")
        else:
            print("  VERDICT: KILL — no improvement in BPT or collapse")


if __name__ == "__main__":
    main()
