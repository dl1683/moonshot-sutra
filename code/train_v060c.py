"""Sutra v0.6.0c Training: Knowledge-preserving random-depth canary (P0 from R11).

Purpose: Disentangle optimizer-reset damage from random-depth benefit.
v0.6.0b RESET the optimizer (WSD restart) when branching from v0.6.0a, destroying
SciQ -12.3% and LAMBADA -9.7%. This canary preserves optimizer moments.

Parent: v0.6.0a step 20000 (model + optimizer state, NOT weights only).
Changes from v0.6.0b:
  1. Preserves optimizer moments (no WSD restart).
  2. Deep-biased curriculum instead of flat alpha ramp:
     - Steps 0-100:  D in {10,11,12} (protect late-pass knowledge)
     - Steps 101-300: D in {8..12} with deep bias
     - Steps 301-750: D in {4..12} with mild deep bias
  3. Continuation LR (~3.2e-4 from parent) with gentle cosine to 1e-4.
  4. Evals at 0, 250, 500, 750 (with per-depth BPT + per-pass entropy).
  5. Checkpoints at every eval point for post-hoc lm_eval (SciQ/LAMBADA).

Success criteria (from R11):
  - SciQ drop <3pts from 48.1% (post-hoc lm_eval on checkpoints)
  - LAMBADA drop <2pts from 11.2%
  - D=8 within 0.02 BPT of D=12
  - Per-pass entropy no longer shows v0.6.0a cliff

Usage: python code/train_v060c.py
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

# Architecture (same as v0.6.0a/b)
DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 16          # Effective batch = 64
VOCAB_SIZE = 50257
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Training schedule: 15K minimum for fair evaluation (was 750 canary)
MAX_TRAIN_STEPS = 15000
EVAL_EVERY = 1000       # Eval + checkpoint every 1K steps
SAVE_EVERY = 1000       # Named checkpoints
ROLLING_SAVE = 100      # Rolling checkpoint every 100 steps for resume
LOG_EVERY = 50

# LR: continue from parent's ~3.2e-4, gentle cosine decay to 1e-4
# (no restart — optimizer moments preserved)
CONTINUATION_LR = 3.2e-4    # Approximate parent LR at step 20K
MIN_LR = 1e-4               # Floor
WARMUP_STEPS = 0             # NO warmup — already trained

# Deep-biased curriculum phases
PHASE_1_END = 100     # Steps 0-100: D in {10,11,12}
PHASE_2_END = 300     # Steps 101-300: D in {8..12} deep bias
                       # Steps 301-750: D in {4..12} mild bias

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v060a import create_v060a
from data_loader import ShardedDataset


def sample_depth_p0(step):
    """Deep-biased curriculum from R11 P0 spec.

    Phase 1 (0-100):   D in {10,11,12} uniform — protect late-pass knowledge
    Phase 2 (101-300): D in {8..12}  with deep bias (P(d) ~ d^2)
    Phase 3 (301-750): D in {4..12}  with mild deep bias (P(d) ~ d^1)
    """
    if step <= PHASE_1_END:
        return random.choice([10, 11, 12])
    elif step <= PHASE_2_END:
        weights = torch.arange(8, MAX_STEPS + 1, dtype=torch.float) ** 2
        idx = torch.multinomial(weights, 1).item()
        return idx + 8
    else:
        weights = torch.arange(4, MAX_STEPS + 1, dtype=torch.float) ** 1
        idx = torch.multinomial(weights, 1).item()
        return idx + 4


def cosine_continuation_lr(step):
    """Gentle cosine decay from continuation LR to min LR over total steps."""
    progress = step / max(MAX_TRAIN_STEPS, 1)
    return MIN_LR + 0.5 * (CONTINUATION_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


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
    """Two-part loss: final CE + probe calibration. Same as v0.6.0b."""
    B, T, V = logits.shape
    n_steps = aux["avg_steps"]

    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    L_probe = torch.tensor(0.0, device=logits.device)
    if "probe_pred" in aux and "sampled_ce_hist" in aux:
        probe_pred = aux["probe_pred"]
        sampled_ce = aux["sampled_ce_hist"]
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
    """Evaluate at full depth (12 passes)."""
    model.eval()
    total_loss = 0
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(min(BATCH_SIZE, 4), SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
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
    """Evaluate at each depth 1..12 with cached batches."""
    model.eval()
    results = {}
    try:
        with torch.no_grad():
            cached = []
            for _ in range(n_batches):
                x, y = dataset.sample_batch(2, SEQ_LEN, device=DEVICE, split="test")
                cached.append((x, y))
            for d in range(1, MAX_STEPS + 1):
                total = 0
                for x, y in cached:
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


def evaluate_per_pass_entropy(model, dataset, n_batches=5):
    """Per-pass logit entropy for halting diagnostics."""
    model.eval()
    results = {}
    try:
        with torch.no_grad():
            cached = []
            for _ in range(n_batches):
                x, y = dataset.sample_batch(2, SEQ_LEN, device=DEVICE, split="test")
                cached.append((x, y))
            accum = {}
            for x, y in cached:
                with autocast_ctx():
                    logits, aux = model(x, y=y, n_steps=MAX_STEPS, collect_history=True)
                if "mu_hist" in aux and aux["mu_hist"] is not None:
                    mu_hist = aux["mu_hist"]
                    P = mu_hist.shape[2]
                    for p in range(P):
                        h = mu_hist[:, :, p, :]
                        h = model.ln(h)
                        pass_logits = F.linear(h, model.emb.weight) / math.sqrt(model.dim)
                        probs = F.softmax(pass_logits.float(), dim=-1)
                        ent = -(probs * (probs + 1e-10).log()).sum(dim=-1)
                        accum.setdefault(p + 1, []).append(ent.mean().item())
            for p, vals in accum.items():
                results[p] = round(sum(vals) / len(vals), 4)
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_eval(model, dataset, step, ckpt_dir, metrics_history, best_bpt):
    """Full evaluation suite at a checkpoint step."""
    print(f"\n{'='*50}")
    print(f"EVAL at step {step}")
    print(f"{'='*50}")

    # 1. Full-depth BPT
    metrics = evaluate(model, dataset)
    bpt = metrics["bpt"]
    is_best = bpt < best_bpt
    if is_best:
        best_bpt = bpt

    print(f"  BPT (D=12): {bpt:.4f} {'*BEST*' if is_best else ''}")

    # 2. Per-depth BPT
    per_depth = evaluate_per_depth(model, dataset, n_batches=10)
    d8_bpt = per_depth.get(8, 99)
    d12_bpt = per_depth.get(12, 99)
    d8_gap = d8_bpt - d12_bpt
    print(f"  Per-depth: D=1:{per_depth.get(1,99):.2f} D=4:{per_depth.get(4,99):.2f} "
          f"D=8:{d8_bpt:.2f} D=12:{d12_bpt:.2f}")
    print(f"  D=8 vs D=12 gap: {d8_gap:.4f} (target: <0.02)")

    # 3. Per-pass entropy
    per_pass_ent = evaluate_per_pass_entropy(model, dataset, n_batches=5)
    if per_pass_ent:
        ent_vals = [f"P{p}={per_pass_ent[p]:.2f}" for p in sorted(per_pass_ent.keys())
                    if p in [1, 4, 8, 10, 11, 12]]
        print(f"  Entropy: {' '.join(ent_vals)}")

    # 4. late_pct
    d1_bpt = per_depth.get(1, 99)
    if d1_bpt > 0 and d12_bpt > 0 and d1_bpt > d12_bpt:
        total_improv = d1_bpt - d12_bpt
        mid_bpt = per_depth.get(6, 99)
        late_improv = mid_bpt - d12_bpt
        late_pct = late_improv / total_improv * 100
        print(f"  late_pct: {late_pct:.1f}% (v0.6.0a was 91.5%)")

    # Save checkpoint (lm_eval compatible: has "model" key)
    atomic_torch_save({
        "model": model.state_dict(),
        "step": step,
        "bpt": bpt,
        "best_bpt": best_bpt if is_best else best_bpt,
        "per_depth_bpt": per_depth,
        "per_pass_entropy": per_pass_ent,
    }, ckpt_dir / f"step_{step}.pt")
    print(f"  Checkpoint saved: step_{step}.pt")

    entry = {
        "step": step,
        "test_bpt": round(bpt, 4),
        "best_bpt": round(best_bpt, 4),
        "is_best": is_best,
        "per_depth_bpt": {str(k): v for k, v in per_depth.items()},
        "per_pass_entropy": {str(k): v for k, v in per_pass_ent.items()},
        "d8_d12_gap": round(d8_gap, 4),
        "timestamp": datetime.now().isoformat(),
    }
    metrics_history.append(entry)
    return best_bpt, metrics_history


def save_rolling(model, opt, dataset, step, best_bpt, metrics_history, depth_counts, ckpt_dir):
    """Save rolling checkpoint with full state for resume."""
    atomic_torch_save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_bpt": best_bpt,
        "metrics": metrics_history,
        "depth_counts": depth_counts,
        "dataset": dataset.state_dict() if hasattr(dataset, 'state_dict') else None,
        "torch_rng": torch.random.get_rng_state(),
        "random_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, ckpt_dir / "rolling_latest.pt")


def main():
    print(f"SUTRA v0.6.0c: KNOWLEDGE-PRESERVING RANDOM-DEPTH (15K)")
    print(f"  Parent: v0.6.0a step 20K (model + optimizer, NO RESET)")
    print(f"  Curriculum: Phase1 D=10-12, Phase2 D=8-12, Phase3 D=4-12")
    print(f"  LR: {CONTINUATION_LR:.1e} -> {MIN_LR:.1e} cosine (no warmup)")
    print(f"  Steps: {MAX_TRAIN_STEPS}, eval every {EVAL_EVERY}")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    dataset = ShardedDataset()

    ckpt_dir = REPO / "results" / "checkpoints_v060c"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v060c_log.txt"
    metrics_file = REPO / "results" / "v060c_metrics.json"

    # Create model
    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                         window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # --- Check for v060c resume FIRST ---
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    depth_counts = [0] * (MAX_STEPS + 1)
    resumed = False

    resume_candidates = []
    rolling = ckpt_dir / "rolling_latest.pt"
    if rolling.exists():
        resume_candidates.append(rolling)
    for p in sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]), reverse=True):
        resume_candidates.append(p)

    resumed_ckpt = None
    for cand in resume_candidates:
        try:
            c = torch.load(cand, weights_only=False, map_location=DEVICE)
            if "optimizer" in c and (resumed_ckpt is None or c.get("step", 0) > resumed_ckpt.get("step", 0)):
                resumed_ckpt = c
                resumed_ckpt["_path"] = cand.name
        except Exception:
            continue

    if resumed_ckpt is not None:
        model.load_state_dict(resumed_ckpt["model"])
        start_step = resumed_ckpt["step"]
        best_bpt = resumed_ckpt.get("best_bpt", float("inf"))
        metrics_history = resumed_ckpt.get("metrics", [])
        depth_counts = resumed_ckpt.get("depth_counts", [0] * (MAX_STEPS + 1))
        if "torch_rng" in resumed_ckpt:
            rng = resumed_ckpt["torch_rng"]
            torch.random.set_rng_state(rng.cpu() if isinstance(rng, torch.Tensor) else rng)
        if "random_rng" in resumed_ckpt:
            random.setstate(resumed_ckpt["random_rng"])
        if "cuda_rng" in resumed_ckpt and resumed_ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            cuda_rng = resumed_ckpt["cuda_rng"]
            torch.cuda.set_rng_state(cuda_rng.cpu() if isinstance(cuda_rng, torch.Tensor) else cuda_rng)
        if "dataset" in resumed_ckpt and resumed_ckpt["dataset"] is not None and hasattr(dataset, 'load_state_dict'):
            dataset.load_state_dict(resumed_ckpt["dataset"])
        print(f"RESUMED v0.6.0c from step {start_step} ({resumed_ckpt['_path']})")
        resumed = True

    if not resumed:
        # --- Load from v0.6.0a step 20K: model + optimizer ---
        source_ckpt_path = REPO / "results" / "checkpoints_v060a" / "step_20000.pt"
        if not source_ckpt_path.exists():
            print(f"ERROR: Parent checkpoint not found: {source_ckpt_path}")
            return
        source_ckpt = torch.load(source_ckpt_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(source_ckpt["model"])
        print(f"Loaded model from v0.6.0a step {source_ckpt['step']} (BPT={source_ckpt['best_bpt']:.4f})")

    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=CONTINUATION_LR, weight_decay=0.01, betas=(0.9, 0.95))

    if resumed and "optimizer" in resumed_ckpt:
        opt.load_state_dict(resumed_ckpt["optimizer"])
        print(f"  Restored optimizer state from {resumed_ckpt['_path']}")
        del resumed_ckpt
    elif not resumed:
        opt.load_state_dict(source_ckpt["optimizer"])
        for pg in opt.param_groups:
            pg["lr"] = CONTINUATION_LR
        print(f"Loaded optimizer moments from parent (preserved, NOT reset)")
        del source_ckpt

    model.train()

    # --- Step 0 eval if fresh start ---
    if start_step == 0:
        best_bpt, metrics_history = run_eval(model, dataset, 0, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    # --- Training loop ---
    step = start_step
    running_losses = {"L_final": 0, "L_probe": 0, "L_total": 0}
    running_depth = 0
    loss_count = 0
    start = time.time()

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        # Cosine continuation LR
        lr = cosine_continuation_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Deep-biased curriculum
        D = sample_depth_p0(step)
        depth_counts[D] += 1

        with autocast_ctx():
            logits, aux = model(x, y=y, n_steps=D)
            Tc = min(logits.size(1), y.size(1))
            loss, loss_parts = compute_losses(logits[:, :Tc], aux, y[:, :Tc])
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
            if isinstance(v, (int, float)) and k in running_losses:
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
                       f"D={avg_d:.1f} lr={lr:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                loss_count = 0

            # Rolling checkpoint for resume
            if step % ROLLING_SAVE == 0:
                save_rolling(model, opt, dataset, step, best_bpt, metrics_history, depth_counts, ckpt_dir)

            # Periodic eval
            if step % EVAL_EVERY == 0:
                best_bpt, metrics_history = run_eval(
                    model, dataset, step, ckpt_dir, metrics_history, best_bpt)
                with open(metrics_file, "w") as mf:
                    json.dump(metrics_history, mf, indent=2)

    # --- Final eval if not already done ---
    if step > 0 and step % EVAL_EVERY != 0:
        best_bpt, metrics_history = run_eval(
            model, dataset, step, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    # --- Final summary ---
    print(f"\n{'='*60}")
    print(f"v0.6.0c COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
    print(f"\nDepth distribution:")
    total_samples = sum(depth_counts)
    for d in range(1, MAX_STEPS + 1):
        if depth_counts[d] > 0:
            pct = depth_counts[d] / max(total_samples, 1) * 100
            print(f"  D={d:2d}: {depth_counts[d]:5d} ({pct:5.1f}%)")

    print(f"\nRun full eval: python code/lm_eval_wrapper.py --checkpoint results/checkpoints_v060c/step_{step}.pt --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq,lambada_openai")


if __name__ == "__main__":
    main()
