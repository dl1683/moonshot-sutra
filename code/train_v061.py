"""Sutra v0.6.1 Training: Controller-Local Canary.

Warm-start from v0.6.0b-rd12 step 3000. Tests the core R8 hypothesis:
  Does a tokenwise factorized controller with bilinear content x pass
  interaction produce content-dependent routing?

Key changes from v0.6.0b trainer:
  - v0.6.1 model (ControlledTransition + PassAdapters)
  - Dual LR: existing weights 1.5e-4, new controller/adapters 4.5e-4
  - Loss: L_final + 0.10*L_step(attached) + 0.20*L_probe + 0.02*L_collapse
  - Alpha cap at 0.7 (softer depth bias than v0.6.0b's 1.0)
  - L_collapse: penalty on late-pass verify mode dominance
  - Random depth training (same schedule as v0.6.0b)

Success criteria (500 steps, from R8 spec):
  - BPT <= parent + 0.05
  - MI(mode, token_class) >= 0.10
  - late_verify_share(passes 8-11) <= 0.75
  - MI(mode, pass_index) <= 0.40
  - D7-D9 frontier within 0.03 BPT of parent

Usage: python code/train_v061.py
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

# Architecture
DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 16         # Effective batch = 64
VOCAB_SIZE = 50257
GRAD_CLIP_NORM = 0.5

# Loss coefficients (from R8 spec)
STEP_LOSS_COEF = 0.10   # L_step (ATTACHED, trains recurrent core)
PROBE_LOSS_COEF = 0.20  # L_probe (detached, shadow only)
COLLAPSE_COEF = 0.02    # L_collapse (penalizes verify dominance)

# Dual LR (from R8 spec)
LR_EXISTING = 1.5e-4    # Existing weights (inherited from parent)
LR_NEW = 4.5e-4         # New controller + pass adapters (3x higher)
WARMUP_STEPS = 100
MAX_TRAIN_STEPS = 15000  # 15K minimum for fair evaluation (was 3K)
EVAL_EVERY = 500
SAVE_EVERY = 500
ROLLING_SAVE = 100
LOG_EVERY = 50

# Random-depth alpha schedule (from R8: cap at 0.7, not 1.0)
ALPHA_START = 0.0
ALPHA_END = 0.7          # Softer late-bias than v0.6.0b's 1.0
ALPHA_RAMP_STEPS = 300   # Fast ramp (per R8 spec)

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v061 import create_v061, load_v061_from_v060b, N_MODES
from launch_v060a import build_negative_set, sampled_pass_ce
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


def compute_v061_losses(logits, aux, y, n_steps):
    """Four-part loss: L_final + L_step(attached) + L_probe + L_collapse.

    Key difference from v0.6.0b: L_step is ATTACHED (trains recurrent core).
    """
    B, T, V = logits.shape

    # 1. L_final: full-vocab CE on final pass
    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    # 2. L_step (ATTACHED): per-pass CE at full vocab
    L_step = torch.tensor(0.0, device=logits.device)
    if "mu_hist" in aux and aux["mu_hist"] is not None:
        mu_hist = aux["mu_hist"]  # (B, T, P, D) — ATTACHED
        ln = None  # We'll need model reference — compute inline
        # Compute per-pass CE using the sampled_ce_hist (which uses attached mu_hist)
        if "sampled_ce_hist" in aux:
            # sampled_ce_hist is computed from attached mu_hist, so gradients flow
            s_ce = aux["sampled_ce_hist"]  # (B, T, P)
            L_step = s_ce.mean()

    # 3. L_probe: residual-gain prediction (detached, shadow only)
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

    # 4. L_collapse: penalize verify mode dominance in late passes
    L_collapse = torch.tensor(0.0, device=logits.device)
    if "alpha_hist" in aux:
        alpha_hist = aux["alpha_hist"]  # (B, T, P, 4)
        P = alpha_hist.shape[2]
        if P >= 8:
            # alpha[:, :, p, 3] = verify mode probability at pass p
            # Penalize mean(alpha_verify) > 0.80 for passes 7-11
            late_start = 7
            late_end = min(P, 12)
            for p in range(late_start, late_end):
                verify_mean = alpha_hist[:, :, p, 3].mean()
                penalty = F.relu(verify_mean - 0.80).pow(2)
                L_collapse = L_collapse + penalty
            L_collapse = L_collapse / max(late_end - late_start, 1)

    L_total = L_final + STEP_LOSS_COEF * L_step + PROBE_LOSS_COEF * L_probe + COLLAPSE_COEF * L_collapse

    return L_total, {
        "L_final": L_final.item(),
        "L_step": L_step.item(),
        "L_probe": L_probe.item(),
        "L_collapse": L_collapse.item(),
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


def evaluate_mode_stats(model, dataset, n_batches=5):
    """Compute mode distribution statistics for diagnostics."""
    model.eval()
    all_alpha = []
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(2, SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
                    _, aux = model(x, n_steps=MAX_STEPS, collect_history=True)
                if "alpha_hist" in aux:
                    all_alpha.append(aux["alpha_hist"].float().cpu())
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_alpha:
        return {}

    alpha = torch.cat(all_alpha, dim=0)  # (N, T, P, 4)
    P = alpha.shape[2]

    # Per-pass mode distribution
    per_pass_modes = {}
    for p in range(P):
        means = alpha[:, :, p, :].mean(dim=(0, 1))
        per_pass_modes[p + 1] = [round(m.item(), 4) for m in means]

    # Late verify share (passes 8-11 = indices 7-10)
    late_verify = alpha[:, :, 7:11, 3].mean().item() if P >= 11 else 0.0

    # Mode entropy (higher = more diverse mode usage)
    mode_entropy = -(alpha.clamp(min=1e-8) * alpha.clamp(min=1e-8).log()).sum(dim=-1).mean().item()

    return {
        "per_pass_modes": per_pass_modes,
        "late_verify_share": round(late_verify, 4),
        "mode_entropy": round(mode_entropy, 4),
    }


def evaluate_per_pass_entropy(model, dataset, n_batches=5):
    """Logit entropy at each pass on fixed eval batches."""
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
                    _, aux = model(x, y=y, n_steps=MAX_STEPS, collect_history=True)
                if "mu_hist" in aux and aux["mu_hist"] is not None:
                    mu_hist = aux["mu_hist"]
                    for p in range(mu_hist.shape[2]):
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


def main():
    print(f"SUTRA v0.6.1: CONTROLLER-LOCAL CANARY")
    print(f"  Tokenwise factorized controller + 12 rank-8 pass adapters")
    print(f"  Loss = L_final + {STEP_LOSS_COEF} L_step + {PROBE_LOSS_COEF} L_probe + {COLLAPSE_COEF} L_collapse")
    print(f"  Dual LR: existing={LR_EXISTING}, new={LR_NEW}")
    print(f"  Random depth, alpha ramp 0->{ALPHA_END} over {ALPHA_RAMP_STEPS} steps")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}, {MAX_TRAIN_STEPS} steps")
    print(f"{'='*60}")

    dataset = ShardedDataset()

    ckpt_dir = REPO / "results" / "checkpoints_v061"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "v061_log.txt"
    metrics_file = REPO / "results" / "v061_metrics.json"

    # Create model
    model = create_v061(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                        window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # Check for v061 resume FIRST
    rolling = ckpt_dir / "rolling_latest.pt"
    permanent = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    depth_history = []
    resumed_ckpt = None

    resume_candidates = []
    if rolling.exists():
        resume_candidates.append(rolling)
    for p in reversed(permanent):
        resume_candidates.append(p)

    for cand in resume_candidates:
        try:
            c = torch.load(cand, weights_only=False, map_location=DEVICE)
            if resumed_ckpt is None or c.get("step", 0) > resumed_ckpt.get("step", 0):
                resumed_ckpt = c
                resumed_ckpt["_path"] = cand.name
        except Exception as e:
            print(f"Skip corrupt checkpoint {cand.name}: {e}")

    if resumed_ckpt is not None:
        model.load_state_dict(resumed_ckpt["model"])
        start_step = resumed_ckpt["step"]
        best_bpt = resumed_ckpt.get("best_bpt", float("inf"))
        metrics_history = resumed_ckpt.get("metrics", [])
        depth_history = resumed_ckpt.get("depth_history", [])
        if "torch_rng" in resumed_ckpt:
            rng = resumed_ckpt["torch_rng"]
            torch.set_rng_state(torch.ByteTensor(rng.cpu().numpy()))
        if "random_rng" in resumed_ckpt:
            random.setstate(resumed_ckpt["random_rng"])
        if "cuda_rng" in resumed_ckpt and torch.cuda.is_available():
            cuda_rng = resumed_ckpt["cuda_rng"]
            torch.cuda.set_rng_state(torch.ByteTensor(cuda_rng.cpu().numpy()))
        if "dataset" in resumed_ckpt:
            dataset.load_state_dict(resumed_ckpt["dataset"])
        print(f"RESUMED v0.6.1 from step {start_step} ({resumed_ckpt['_path']})")
    else:
        # Warm-start from v0.6.0b step 3000
        parent_ckpt = REPO / "results" / "checkpoints_v060b" / "step_3000.pt"
        if not parent_ckpt.exists():
            # Try rolling latest
            parent_ckpt = REPO / "results" / "checkpoints_v060b" / "rolling_latest.pt"
        if not parent_ckpt.exists():
            print("ERROR: No v0.6.0b checkpoint found for warm-start")
            return
        info = load_v061_from_v060b(model, parent_ckpt, device=DEVICE)
        print(f"  Parent step: {info['parent_step']}, BPT: {info.get('parent_bpt', 'N/A')}")

    total_params = model.count_params()
    print(f"Params: {total_params:,} ({total_params/1e6:.1f}M)")

    # Dual LR optimizer: separate param groups
    existing_params = []
    new_params = []
    for name, param in model.named_parameters():
        if ("transition.controller" in name or "pass_adapters" in name):
            new_params.append(param)
        else:
            existing_params.append(param)

    print(f"  Existing params: {sum(p.numel() for p in existing_params):,}")
    print(f"  New params: {sum(p.numel() for p in new_params):,}")

    opt = torch.optim.AdamW([
        {"params": existing_params, "lr": LR_EXISTING},
        {"params": new_params, "lr": LR_NEW},
    ], weight_decay=0.01, betas=(0.9, 0.95))

    if resumed_ckpt is not None and "optimizer" in resumed_ckpt:
        opt.load_state_dict(resumed_ckpt["optimizer"])
        print(f"  Restored optimizer state")

    model.train()
    step = start_step
    running_losses = {"L_final": 0, "L_step": 0, "L_probe": 0, "L_collapse": 0, "L_total": 0}
    running_depth = 0
    loss_count = 0
    start = time.time()
    depth_counts = [0] * (MAX_STEPS + 1)

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        # Cosine LR with warmup (applied to both param groups with their base LR)
        progress = max(0, step - WARMUP_STEPS) / max(1, MAX_TRAIN_STEPS - WARMUP_STEPS)
        warmup_mult = min(1.0, (step + 1) / WARMUP_STEPS)
        decay_mult = 0.5 * (1 + math.cos(math.pi * progress))
        lr_mult = warmup_mult * decay_mult
        opt.param_groups[0]["lr"] = LR_EXISTING * lr_mult
        opt.param_groups[1]["lr"] = LR_NEW * lr_mult

        # Random depth (same schedule as v0.6.0b, but alpha caps at 0.7)
        alpha = ALPHA_START + min(step / ALPHA_RAMP_STEPS, 1.0) * (ALPHA_END - ALPHA_START)
        D = sample_depth(MAX_STEPS, alpha)
        depth_counts[D] += 1

        with autocast_ctx():
            logits, aux = model(x, y=y, n_steps=D)
            Tc = min(logits.size(1), y.size(1))
            loss, loss_parts = compute_v061_losses(logits[:, :Tc], aux, y[:, :Tc], D)
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
                lr_show = opt.param_groups[0]["lr"]
                msg = (f"Step {step:>5d}: L={avgs['L_total']:.4f} "
                       f"(fin={avgs['L_final']:.3f} stp={avgs['L_step']:.3f} "
                       f"prb={avgs['L_probe']:.3f} col={avgs['L_collapse']:.4f}) "
                       f"D={avg_d:.1f} a={alpha:.2f} lr={lr_show:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                loss_count = 0

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
                        "lr_existing": opt.param_groups[0]["lr"],
                        "lr_new": opt.param_groups[1]["lr"],
                        "alpha": round(alpha, 3),
                        "timestamp": datetime.now().isoformat(),
                    }
                    metrics_history.append(entry)

                    # Per-depth
                    per_depth = evaluate_per_depth(model, dataset, n_batches=5)
                    entry["per_depth_bpt"] = per_depth
                    depth_history.append({"step": step, "per_depth": per_depth})

                    # Per-pass entropy
                    per_pass_entropy = evaluate_per_pass_entropy(model, dataset, n_batches=5)
                    entry["per_pass_entropy"] = per_pass_entropy

                    # Mode stats (v0.6.1 specific)
                    mode_stats = evaluate_mode_stats(model, dataset, n_batches=3)
                    entry["mode_stats"] = mode_stats

                    with open(metrics_file, "w") as mf:
                        json.dump(metrics_history, mf, indent=2)

                    marker = " *BEST*" if is_best else ""
                    eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{marker}"
                    print(eval_msg, flush=True)

                    # Per-depth summary
                    d1 = per_depth.get(1, 99)
                    d8 = per_depth.get(8, 99)
                    d9 = per_depth.get(9, 99)
                    d12 = per_depth.get(12, 99)
                    print(f"    D=1:{d1:.2f} D=8:{d8:.2f} D=9:{d9:.2f} D=12:{d12:.2f}", flush=True)

                    # Mode stats summary
                    if mode_stats:
                        lv = mode_stats.get("late_verify_share", -1)
                        me = mode_stats.get("mode_entropy", -1)
                        print(f"    late_verify={lv:.3f} mode_ent={me:.3f}", flush=True)

                    # Adapter gate magnitudes
                    gates = [abs(model.pass_adapters[i].gate.item()) for i in range(MAX_STEPS)]
                    max_gate = max(gates)
                    print(f"    adapter_gates: max={max_gate:.4f} mean={sum(gates)/len(gates):.4f}", flush=True)

                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n")

                except Exception as e:
                    print(f"EVAL FAILED: {e}", flush=True)
                    import traceback; traceback.print_exc()

            if step % SAVE_EVERY == 0:
                rng_state = {
                    "torch_rng": torch.get_rng_state(),
                    "random_rng": random.getstate(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda_rng"] = torch.cuda.get_rng_state()
                atomic_torch_save({
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history, "depth_history": depth_history,
                    "depth_counts": depth_counts,
                    "dataset": dataset.state_dict(),
                    **rng_state,
                }, ckpt_dir / f"step_{step}.pt")

            if step % ROLLING_SAVE == 0:
                rng_state = {
                    "torch_rng": torch.get_rng_state(),
                    "random_rng": random.getstate(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda_rng"] = torch.cuda.get_rng_state()
                atomic_torch_save({
                    "model": model.state_dict(), "optimizer": opt.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history, "depth_history": depth_history,
                    "depth_counts": depth_counts,
                    "dataset": dataset.state_dict(),
                    **rng_state,
                }, ckpt_dir / "rolling_latest.pt")

    # Final summary
    print(f"\n{'='*60}")
    print(f"v0.6.1 COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
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

    # Final mode stats
    print(f"\nFinal mode stats:")
    mode_stats = evaluate_mode_stats(model, dataset, n_batches=5)
    if mode_stats:
        print(f"  late_verify_share: {mode_stats.get('late_verify_share', -1):.4f}")
        print(f"  mode_entropy: {mode_stats.get('mode_entropy', -1):.4f}")


if __name__ == "__main__":
    main()
