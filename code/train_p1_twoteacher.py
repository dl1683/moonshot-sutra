"""Sutra P1: O4-First Two-Teacher 15K Continuation (R12 top priority).

PURPOSE: First multi-source learning experiment. Absorb external knowledge from
pretrained models into Sutra's recurrent architecture.

Parent: v0.6.0a step 20K (model + optimizer, preserved)
Teachers (frozen):
  1. GPT-2 small (124M) — AR logit teacher (same tokenizer = direct KL, no remapping)
  2. all-MiniLM-L6-v2 — Encoder geometry teacher (CKA alignment)

Loss = L_CE + α_logit * L_KD + α_repr * L_CKA + 0.20 * L_probe

Key design:
  - Top-30% highest-entropy tokens selected for distillation (R12 spec)
  - Steps 0-250: deep-biased rd burst (protect knowledge)
  - Steps 251+: mostly D=8 with occasional D=12 refresh
  - 15K steps, full 7-task eval + generation at 15K
  - Teachers run with no_grad, frozen, ~0.4GB extra VRAM

Success criteria (from R12):
  - Beat v0.6.0a on SciQ (>48.1%) or LAMBADA (>11.2%) or both
  - No regression on PIQA, ARC-E
  - Better generation quality
  - Prove O4 (data efficiency via multi-source learning) is viable

Usage: python code/train_p1_twoteacher.py
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
GRAD_ACCUM = 16          # Effective batch = 64
VOCAB_SIZE = 50257
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Distillation coefficients
ALPHA_LOGIT = 0.5        # KL divergence from AR teacher
ALPHA_REPR = 0.3         # CKA from encoder teacher
KD_TEMP = 2.0            # Temperature for KL distillation
ENTROPY_TOP_FRAC = 0.30  # Top-30% entropy tokens for distillation
CKA_EVERY_N = 4          # Compute CKA every N micro-batches (accumulate between)

# Training schedule
MAX_TRAIN_STEPS = 15000
EVAL_EVERY = 1000
SAVE_EVERY = 1000
ROLLING_SAVE = 100
LOG_EVERY = 50

# LR: continuation from parent
CONTINUATION_LR = 3.2e-4
MIN_LR = 1e-4
WARMUP_STEPS = 0

# Depth schedule: rd burst then D=8 default
RD_BURST_END = 250      # Deep-biased rd burst for 250 steps
D_DEFAULT = 8           # Default depth after burst
D_REFRESH_PROB = 0.05   # 5% chance of D=12 refresh after burst

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v060a import create_v060a
from data_loader import ShardedDataset


# ---- Teacher loading ----

def load_ar_teacher():
    """Load GPT-2 small (124M) as frozen AR logit teacher.
    Same GPT-2 tokenizer and vocab (50257) as the student = direct KL, no remapping.
    """
    from transformers import AutoModelForCausalLM
    print("Loading AR teacher: GPT-2 small (124M)...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in teacher.parameters())
    vram_mb = n_params * 2 / 1e6  # bf16
    print(f"  GPT-2 loaded: {n_params/1e6:.0f}M params, ~{vram_mb:.0f}MB VRAM")
    return teacher


def load_encoder_teacher():
    """Load all-MiniLM-L6-v2 as frozen encoder geometry teacher.
    Different tokenizer — we align at sentence/batch level via CKA.
    """
    from transformers import AutoModel, AutoTokenizer
    print("Loading encoder teacher: all-MiniLM-L6-v2...", flush=True)
    enc_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    enc_model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    enc_model.eval()
    for p in enc_model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in enc_model.parameters())
    vram_mb = n_params * 2 / 1e6
    print(f"  MiniLM loaded: {n_params/1e6:.0f}M params, ~{vram_mb:.0f}MB VRAM")
    return enc_model, enc_tokenizer


# ---- Distillation losses ----

def compute_kd_loss(student_logits, teacher_logits, mask):
    """KL divergence distillation on masked (top-entropy) tokens.

    Args:
        student_logits: (B, T, V=50257) student output logits
        teacher_logits: (B, T, V=50257) teacher output logits (same tokenizer)
        mask: (B, T) bool mask — True for tokens to distill
    Returns:
        scalar KL loss (averaged over masked tokens)
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    s_flat = student_logits[mask]   # (N, V)
    t_flat = teacher_logits[mask]   # (N, V)

    # Temperature-scaled KL divergence
    s_probs = F.log_softmax(s_flat / KD_TEMP, dim=-1)
    t_probs = F.softmax(t_flat / KD_TEMP, dim=-1)
    kl = F.kl_div(s_probs, t_probs, reduction="batchmean") * (KD_TEMP ** 2)
    return kl


def linear_cka(X, Y):
    """Linear CKA (Centered Kernel Alignment) between two sets of representations.

    Args:
        X: (N, D1) — student representations
        Y: (N, D2) — teacher representations
    Returns:
        scalar CKA similarity in [0, 1]
    """
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    XtX = X @ X.T  # (N, N)
    YtY = Y @ Y.T  # (N, N)

    # HSIC
    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-8)
    return hsic_xy / denom


def get_encoder_repr(enc_model, enc_tokenizer, texts):
    """Get mean-pooled encoder representations for a list of texts."""
    enc_inputs = enc_tokenizer(
        texts, padding=True, truncation=True, max_length=256,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        enc_out = enc_model(**enc_inputs)
        attn_mask = enc_inputs["attention_mask"].unsqueeze(-1).float()
        t_repr = (enc_out.last_hidden_state * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
    return t_repr.float()  # (N, D_encoder=384)


def compute_cka_loss(student_reprs, teacher_reprs):
    """CKA alignment loss between accumulated student and teacher representations.

    Args:
        student_reprs: (N, D_student) mean-pooled student hidden states
        teacher_reprs: (N, D_teacher) mean-pooled teacher hidden states
    Returns:
        scalar (1 - CKA) loss
    """
    cka_sim = linear_cka(student_reprs, teacher_reprs)
    return 1.0 - cka_sim


def select_high_entropy_mask(logits, top_frac=ENTROPY_TOP_FRAC):
    """Select top-fraction highest-entropy tokens for distillation.

    Args:
        logits: (B, T, V) student logits
        top_frac: fraction of tokens to select (0.3 = top 30%)
    Returns:
        mask: (B, T) bool tensor
    """
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (B, T)
        # Top-k per sample
        k = max(1, int(entropy.size(1) * top_frac))
        _, top_idx = entropy.topk(k, dim=1)
        mask = torch.zeros_like(entropy, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)
    return mask


# ---- Depth sampling ----

def sample_depth_p1(step):
    """P1 depth schedule: rd burst then D=8 default with occasional D=12."""
    if step <= RD_BURST_END:
        # Deep-biased burst (same as P0)
        if step <= 100:
            return random.choice([10, 11, 12])
        else:
            weights = torch.arange(8, MAX_STEPS + 1, dtype=torch.float) ** 2
            idx = torch.multinomial(weights, 1).item()
            return idx + 8
    else:
        # Mostly D=8 with occasional D=12 refresh
        if random.random() < D_REFRESH_PROB:
            return 12
        return D_DEFAULT


# ---- Standard functions (from v0.6.0c) ----

def cosine_continuation_lr(step):
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


def compute_base_losses(logits, aux, y):
    """Base CE + probe loss (same as v0.6.0c)."""
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

    return L_final, L_probe


def evaluate(model, dataset, n_batches=20):
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


def run_eval(model, dataset, step, ckpt_dir, metrics_history, best_bpt):
    print(f"\n{'='*50}")
    print(f"EVAL at step {step}")
    print(f"{'='*50}")

    metrics = evaluate(model, dataset)
    bpt = metrics["bpt"]
    is_best = bpt < best_bpt
    if is_best:
        best_bpt = bpt

    print(f"  BPT (D={MAX_STEPS}): {bpt:.4f} {'*BEST*' if is_best else ''}")

    per_depth = evaluate_per_depth(model, dataset, n_batches=10)
    d8_bpt = per_depth.get(8, 99)
    d12_bpt = per_depth.get(12, 99)
    print(f"  Per-depth: D=1:{per_depth.get(1,99):.2f} D=4:{per_depth.get(4,99):.2f} "
          f"D=8:{d8_bpt:.2f} D=12:{d12_bpt:.2f}")

    # late_pct
    d1_bpt = per_depth.get(1, 99)
    if d1_bpt > d12_bpt > 0:
        total_improv = d1_bpt - d12_bpt
        mid_bpt = per_depth.get(6, 99)
        late_improv = mid_bpt - d12_bpt
        late_pct = late_improv / total_improv * 100
        print(f"  late_pct: {late_pct:.1f}%")

    # Save checkpoint
    atomic_torch_save({
        "model": model.state_dict(),
        "step": step,
        "bpt": bpt,
        "best_bpt": best_bpt,
        "per_depth_bpt": per_depth,
    }, ckpt_dir / f"step_{step}.pt")
    print(f"  Checkpoint saved: step_{step}.pt")

    entry = {
        "step": step,
        "test_bpt": round(bpt, 4),
        "best_bpt": round(best_bpt, 4),
        "is_best": is_best,
        "per_depth_bpt": {str(k): v for k, v in per_depth.items()},
        "timestamp": datetime.now().isoformat(),
    }
    metrics_history.append(entry)
    return best_bpt, metrics_history


def save_rolling(model, opt, dataset, step, best_bpt, metrics_history, depth_counts, ckpt_dir):
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


# ---- GPT-2 tokenizer for text decoding ----

_gpt2_tokenizer = None

def get_gpt2_tokenizer():
    global _gpt2_tokenizer
    if _gpt2_tokenizer is None:
        from transformers import AutoTokenizer
        _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _gpt2_tokenizer


def decode_tokens_to_texts(token_ids):
    """Decode GPT-2 token IDs to text strings for encoder teacher."""
    tok = get_gpt2_tokenizer()
    texts = []
    for seq in token_ids:
        text = tok.decode(seq.tolist(), skip_special_tokens=True)
        texts.append(text[:1024])  # truncate for encoder
    return texts


# ---- Main ----

def main():
    print(f"SUTRA P1: O4-FIRST TWO-TEACHER CONTINUATION (R12)")
    print(f"  Parent: v0.6.0a step 20K (model + optimizer, preserved)")
    print(f"  AR Teacher: GPT-2 small (KL, α={ALPHA_LOGIT})")
    print(f"  Enc Teacher: all-MiniLM-L6-v2 (CKA, α={ALPHA_REPR})")
    print(f"  Entropy selection: top {ENTROPY_TOP_FRAC*100:.0f}%")
    print(f"  Depth: rd burst 0-{RD_BURST_END}, then D={D_DEFAULT}")
    print(f"  LR: {CONTINUATION_LR:.1e} -> {MIN_LR:.1e} cosine")
    print(f"  Steps: {MAX_TRAIN_STEPS}, eval every {EVAL_EVERY}")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    # Load dataset
    dataset = ShardedDataset()

    ckpt_dir = REPO / "results" / "checkpoints_p1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / "p1_log.txt"
    metrics_file = REPO / "results" / "p1_metrics.json"

    # Create student model
    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                         window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # --- Resume or load from parent ---
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    depth_counts = [0] * (MAX_STEPS + 1)
    resumed = False

    # Check for P1 resume
    resume_candidates = []
    rolling = ckpt_dir / "rolling_latest.pt"
    if rolling.exists():
        resume_candidates.append(rolling)
    for p in sorted(ckpt_dir.glob("step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]), reverse=True):
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
        print(f"RESUMED P1 from step {start_step} ({resumed_ckpt['_path']})")
        resumed = True

    if not resumed:
        # Load from v0.6.0a step 20K
        source_path = REPO / "results" / "checkpoints_v060a" / "step_20000.pt"
        if not source_path.exists():
            print(f"ERROR: Parent checkpoint not found: {source_path}")
            return
        source_ckpt = torch.load(source_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(source_ckpt["model"])
        print(f"Loaded model from v0.6.0a step {source_ckpt['step']} (BPT={source_ckpt['best_bpt']:.4f})")

    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Optimizer (preserve moments from parent)
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

    # ---- Load teachers ----
    ar_teacher = load_ar_teacher()
    enc_model, enc_tokenizer = load_encoder_teacher()

    print(f"\n{'='*60}")
    print("All models loaded. Starting training...")
    print(f"{'='*60}\n")

    # Step 0 eval
    if start_step == 0:
        best_bpt, metrics_history = run_eval(model, dataset, 0, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    # ---- Training loop ----
    step = start_step
    running_losses = {"L_CE": 0, "L_probe": 0, "L_KD": 0, "L_CKA": 0, "L_total": 0}
    running_depth = 0
    loss_count = 0
    start_time = time.time()

    # CKA accumulation buffers (Fix #3: larger effective batch for stable Gram matrices)
    cka_student_buf = []   # list of (B, D) detached tensors
    cka_text_buf = []      # list of text strings

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        # LR schedule
        lr = cosine_continuation_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Depth schedule
        D = sample_depth_p1(step)
        depth_counts[D] += 1

        with autocast_ctx():
            # Student forward
            logits, aux = model(x, y=y, n_steps=D)
            Tc = min(logits.size(1), y.size(1))
            s_logits = logits[:, :Tc]

            # Base losses (CE + probe)
            L_CE, L_probe = compute_base_losses(s_logits, aux, y[:, :Tc])

            # ---- Distillation losses ----
            # 1. AR teacher logits (GPT-2 small, same tokenizer = direct alignment)
            with torch.no_grad():
                t_out = ar_teacher(x, use_cache=False)
                t_logits = t_out.logits[:, :Tc]  # (B, T, V=50257) same space

            # Select high-entropy tokens for distillation
            entropy_mask = select_high_entropy_mask(s_logits)

            # KL divergence on selected tokens
            L_KD = compute_kd_loss(s_logits, t_logits, entropy_mask)

            # 2. CKA with encoder teacher — accumulate across CKA_EVERY_N micro-batches
            #    Fix #3: larger Gram matrix (N=CKA_EVERY_N*B) + unbiased weighting
            L_CKA = torch.tensor(0.0, device=DEVICE)
            if "mu_hist" in aux and aux["mu_hist"] is not None:
                student_hidden = aux["mu_hist"][:, :Tc, -1, :]  # (B, T, D)
                s_repr_live = student_hidden.float().mean(dim=1)  # (B, D) with gradients
                texts = decode_tokens_to_texts(x)

                # Accumulate detached representations for stable Gram matrix
                cka_student_buf.append(s_repr_live.detach())
                cka_text_buf.extend(texts)

                # Compute CKA at end of accumulation window
                if len(cka_student_buf) >= CKA_EVERY_N:
                    # Concat: first (N-1) detached + last live (has gradients)
                    all_s = torch.cat(cka_student_buf[:-1] + [s_repr_live], dim=0)
                    all_texts = cka_text_buf[:]

                    # Teacher representations for all accumulated texts at once
                    t_repr = get_encoder_repr(enc_model, enc_tokenizer, all_texts)
                    L_CKA = compute_cka_loss(all_s, t_repr)

                    # Clear buffers
                    cka_student_buf.clear()
                    cka_text_buf.clear()

            # Combined loss (CKA compensated for 1-in-N frequency)
            cka_weight = ALPHA_REPR * CKA_EVERY_N if L_CKA.item() > 0 else 0.0
            L_total = L_CE + ALPHA_LOGIT * L_KD + cka_weight * L_CKA + PROBE_LOSS_COEF * L_probe
            L_total = L_total / GRAD_ACCUM

        if not torch.isfinite(L_total):
            print(f"WARNING: Non-finite loss at step {step} (D={D}), skipping", flush=True)
            opt.zero_grad()
            loss_count = 0
            running_losses = {k: 0 for k in running_losses}
            running_depth = 0
            cka_student_buf.clear()
            cka_text_buf.clear()
            continue

        L_total.backward()
        running_losses["L_CE"] += L_CE.item()
        running_losses["L_probe"] += L_probe.item()
        running_losses["L_KD"] += L_KD.item()
        running_losses["L_CKA"] += L_CKA.item()
        running_losses["L_total"] += (L_total * GRAD_ACCUM).item()
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
                cka_student_buf.clear()
                cka_text_buf.clear()
                continue

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avgs = {k: v / loss_count for k, v in running_losses.items()}
                avg_d = running_depth / loss_count
                elapsed = time.time() - start_time
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                msg = (f"Step {step:>5d}: L={avgs['L_total']:.4f} "
                       f"(CE={avgs['L_CE']:.3f} KD={avgs['L_KD']:.3f} "
                       f"CKA={avgs['L_CKA']:.3f} prb={avgs['L_probe']:.3f}) "
                       f"D={avg_d:.1f} lr={lr:.2e} {tps:.0f}tok/s")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                loss_count = 0

            if step % ROLLING_SAVE == 0:
                save_rolling(model, opt, dataset, step, best_bpt, metrics_history, depth_counts, ckpt_dir)

            if step % EVAL_EVERY == 0:
                best_bpt, metrics_history = run_eval(
                    model, dataset, step, ckpt_dir, metrics_history, best_bpt)
                with open(metrics_file, "w") as mf:
                    json.dump(metrics_history, mf, indent=2)

    # Final eval
    if step > 0 and step % EVAL_EVERY != 0:
        best_bpt, metrics_history = run_eval(
            model, dataset, step, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    print(f"\n{'='*60}")
    print(f"P1 COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
    print(f"\nDepth distribution:")
    total_samples = sum(depth_counts)
    for d in range(1, MAX_STEPS + 1):
        if depth_counts[d] > 0:
            pct = depth_counts[d] / max(total_samples, 1) * 100
            print(f"  D={d:2d}: {depth_counts[d]:5d} ({pct:5.1f}%)")
    print(f"\nRun full eval: python code/lm_eval_wrapper.py --checkpoint results/checkpoints_p1/step_{step}.pt --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq,lambada_openai")


if __name__ == "__main__":
    main()
