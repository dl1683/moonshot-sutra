"""Dense transformer with optional early-exit heads (EDSR).

Standard decoder-only transformer: RoPE + RMSNorm + SwiGLU + causal attention.
Uses same data pipeline and eval infrastructure as the recurrent Sutra model.

Two modes:
  1. F4 baseline: 11L/d=512/8h/ff=1536 (~51M params, no exits)
  2. EDSR-98M: 12L/d=768/12h/ff=2048 (~97M params, exits at 4/8/12)

Usage:
  # F4 baseline (original)
  python code/dense_baseline.py --max-steps 5000 --run-name dense_f4

  # EDSR-98M with early exits (R21 architecture)
  python code/dense_baseline.py --edsr --max-steps 20000 --run-name edsr_98m

  # Det-eval on checkpoint
  python code/dense_baseline.py --det-eval results/checkpoints_edsr_98m/step_5000.pt
"""

import sys, os, math, json, time, gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from data_loader import ShardedDataset

# ---- Config ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

VOCAB_SIZE = 16000
DIM = 512
N_LAYERS = 11
N_HEADS = 8
FF_DIM = 1536
MAX_SEQ_LEN = 512

# Training
BATCH_SIZE = 16
GRAD_ACCUM = 2
SEQ_LEN = 512
MAX_TRAIN_STEPS = 5000
EVAL_EVERY = 1000
ROLLING_SAVE = 500
LOG_EVERY = 50
LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 200
GRAD_CLIP = 1.0


# ---- RoPE ----
def precompute_rope(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to x: (B, T, H, D)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2).to(x.dtype)  # (1, T, 1, D//2)
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---- Model ----
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # (B, T, H, Dh)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)  # (B, H, T, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GainHead(nn.Module):
    """Predicts residual gain from continuing to deeper layers.
    Inputs: hidden state + logit entropy + top-1/top-2 margin = dim+2 -> 256 -> 1"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 2, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 1, bias=False),
        )

    def forward(self, h, logits):
        """h: (B,T,D), logits: (B,T,V) -> gain: (B,T,1)"""
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(-1, keepdim=True)  # (B,T,1)
            topk = logits.topk(2, dim=-1).values  # (B,T,2)
            margin = (topk[:, :, 0:1] - topk[:, :, 1:2])  # (B,T,1)
        feat = torch.cat([h, entropy.to(h.dtype), margin.to(h.dtype)], dim=-1)
        return self.net(feat)  # (B,T,1)


class DenseTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, max_seq_len=MAX_SEQ_LEN,
                 exit_layers=None):
        """
        Args:
            exit_layers: list of 0-indexed layer numbers for early exits.
                e.g. [3,7,11] for exits after layers 4,8,12 (1-indexed).
                If None, no early exits (standard dense model).
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.exit_layers = sorted(exit_layers) if exit_layers else []
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)

        # Early-exit norms and gain heads (one per exit, NOT for final layer)
        self.exit_norms = nn.ModuleDict()
        self.gain_heads = nn.ModuleDict()
        for i in self.exit_layers:
            if i < n_layers - 1:  # final layer uses self.norm
                self.exit_norms[str(i)] = RMSNorm(dim)
                self.gain_heads[str(i)] = GainHead(dim)

        # RoPE cache
        cos, sin = precompute_rope(dim // n_heads, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_exits=False):
        """
        Args:
            return_exits: if True, return dict with exit logits and gain predictions.
        Returns:
            If return_exits=False: final logits (B, T, V)
            If return_exits=True: dict {
                'logits': final logits,
                'exit_logits': {layer_idx: logits},
                'exit_gains': {layer_idx: gain predictions},
            }
        """
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.dim)
        cos, sin = self.rope_cos, self.rope_sin

        exit_logits = {}
        exit_gains = {}

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin)

            # Early exit at this layer?
            if return_exits and i in self.exit_layers and i < self.n_layers - 1:
                h_normed = self.exit_norms[str(i)](h)
                e_logits = F.linear(h_normed, self.emb.weight)
                exit_logits[i] = e_logits
                exit_gains[i] = self.gain_heads[str(i)](h_normed, e_logits)

        h = self.norm(h)
        logits = F.linear(h, self.emb.weight)  # tied embeddings

        if return_exits:
            return {
                'logits': logits,
                'exit_logits': exit_logits,
                'exit_gains': exit_gains,
            }
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---- LR schedule ----
def get_lr_cosine(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def get_lr_wsd(step, warmup, max_steps, lr, min_lr):
    """Warmup-Stable-Decay: flat LR after warmup, then linear decay in last 20%."""
    if step < warmup:
        return lr * step / max(warmup, 1)
    decay_start = int(max_steps * 0.8)
    if step < decay_start:
        return lr
    progress = (step - decay_start) / max(max_steps - decay_start, 1)
    return lr - progress * (lr - min_lr)


# ---- Exit loss weights (R21 spec) ----
EXIT_WEIGHTS = {3: 0.20, 7: 0.35}  # layer 4 (idx=3) and layer 8 (idx=7)
GAIN_WEIGHT = 0.05


def compute_edsr_loss(out, tgt, exit_weights=EXIT_WEIGHTS, gain_weight=GAIN_WEIGHT):
    """Multi-exit loss: L = 1.00*CE_final + sum(w_i * CE_exit_i) + gain_weight * L_gain.

    Returns: (total_loss, metrics_dict) where metrics_dict has ce_final and per-exit CEs.
    """
    V = out['logits'].size(-1)
    Tc = min(out['logits'].size(1), tgt.size(1))
    tgt_flat = tgt[:, :Tc].reshape(-1)

    # Final exit loss (always weight 1.0)
    ce_final = F.cross_entropy(out['logits'][:, :Tc].reshape(-1, V), tgt_flat)
    total = ce_final
    metrics = {"ce_final": ce_final.item()}

    # Early exit losses
    for layer_idx, weight in exit_weights.items():
        if layer_idx in out['exit_logits']:
            e_logits = out['exit_logits'][layer_idx][:, :Tc].reshape(-1, V)
            ce_exit = F.cross_entropy(e_logits, tgt_flat)
            total = total + weight * ce_exit
            metrics[f"ce_exit_{layer_idx}"] = ce_exit.item()

    # Gain loss: gain head should predict actual CE improvement from continuing
    if gain_weight > 0 and out['exit_gains']:
        gain_loss = 0
        n_gain = 0
        for layer_idx, gain_pred in out['exit_gains'].items():
            if layer_idx in out['exit_logits']:
                with torch.no_grad():
                    # Target: how much does CE improve from exit -> final?
                    e_logits = out['exit_logits'][layer_idx][:, :Tc]
                    ce_exit_pt = F.cross_entropy(
                        e_logits.reshape(-1, V), tgt_flat, reduction='none'
                    ).reshape(e_logits.shape[0], -1).mean(-1, keepdim=True)  # (B,1)
                    ce_final_pt = F.cross_entropy(
                        out['logits'][:, :Tc].reshape(-1, V), tgt_flat, reduction='none'
                    ).reshape(e_logits.shape[0], -1).mean(-1, keepdim=True)  # (B,1)
                    target_gain = (ce_exit_pt - ce_final_pt).clamp(min=0)  # positive = improvement
                # Mean gain across sequence
                mean_gain_pred = gain_pred[:, :Tc].mean(1)  # (B, 1)
                gain_loss = gain_loss + F.mse_loss(mean_gain_pred, target_gain)
                n_gain += 1
                metrics[f"gain_{layer_idx}"] = mean_gain_pred.mean().item()
        if n_gain > 0:
            total = total + gain_weight * gain_loss / n_gain

    return total, metrics


def train(run_name="dense_f4", edsr=False):
    # Select config
    if edsr:
        dim, n_layers, n_heads, ff_dim = 768, 12, 12, 2048
        exit_layers = [3, 7, 11]  # after layers 4, 8, 12 (1-indexed)
        lr, min_lr = 3e-4, 1e-5
        warmup = 500
        batch_size, grad_accum = 8, 4  # 32 effective, fits in 24GB
        lr_schedule = "wsd"
        mode_name = "EDSR-98M"
    else:
        dim, n_layers, n_heads, ff_dim = DIM, N_LAYERS, N_HEADS, FF_DIM
        exit_layers = None
        lr, min_lr = LR, MIN_LR
        warmup = WARMUP_STEPS
        batch_size, grad_accum = BATCH_SIZE, GRAD_ACCUM
        lr_schedule = "cosine"
        mode_name = "Dense Baseline (F4)"

    print(f"{mode_name}")
    print(f"  Arch: {n_layers}L / d={dim} / {n_heads}h / ff={ff_dim} / vocab={VOCAB_SIZE}")
    if exit_layers:
        print(f"  Exits at layers: {[i+1 for i in exit_layers]}")
    print(f"  LR: {lr:.1e} -> {min_lr:.1e} ({lr_schedule}), warmup={warmup}")
    print(f"  Steps: {MAX_TRAIN_STEPS}, eval every {EVAL_EVERY}")
    print(f"  Batch: {batch_size}x{grad_accum}={batch_size*grad_accum}, seq={SEQ_LEN}")
    print(f"  Device: {DEVICE}, dtype: bf16")

    model = DenseTransformer(
        vocab_size=VOCAB_SIZE, dim=dim, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim, exit_layers=exit_layers
    ).to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    print("=" * 60)

    # Data — use 16K retokenized shards
    shard_dir = REPO / "data" / "shards_16k"
    if not shard_dir.exists() or len(list(shard_dir.glob("*.pt"))) < 64:
        print(f"ERROR: Need >=64 shards in {shard_dir}. Currently: {len(list(shard_dir.glob('*.pt')))}")
        sys.exit(1)
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # Resume — try rolling first, fall back to newest step_*.pt
    start_step = 0
    best_bpt = float("inf")
    rolling = ckpt_dir / "rolling_latest.pt"
    resume_path = None
    if rolling.exists():
        try:
            ckpt = torch.load(rolling, weights_only=False, map_location=DEVICE)
            resume_path = rolling
        except Exception as e:
            print(f"WARNING: rolling checkpoint corrupted ({e}), trying step_*.pt fallback")
    if resume_path is None:
        step_ckpts = sorted(ckpt_dir.glob("step_*.pt"),
                            key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
                            reverse=True)
        for sc in step_ckpts:
            try:
                ckpt = torch.load(sc, weights_only=False, map_location=DEVICE)
                resume_path = sc
                print(f"Recovered from {sc.name}")
                break
            except Exception:
                continue
    if resume_path is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        best_bpt = ckpt.get("best_bpt", float("inf"))
        print(f"Resumed from step {start_step}, best BPT={best_bpt:.4f}")

    step = start_step
    running_loss = 0
    n_tokens = 0
    t0 = time.time()
    use_exits = edsr and exit_layers

    model.train()
    while step < MAX_TRAIN_STEPS:
        opt.zero_grad()
        bad_step = False
        for micro in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                if use_exits:
                    out = model(inp, return_exits=True)
                    loss, loss_metrics = compute_edsr_loss(out, tgt)
                    ce_val = loss_metrics["ce_final"]
                    loss = loss / grad_accum
                else:
                    logits = model(inp)
                    Tc = min(logits.size(1), tgt.size(1))
                    loss = F.cross_entropy(logits[:, :Tc].reshape(-1, logits.size(-1)),
                                           tgt[:, :Tc].reshape(-1))
                    ce_val = loss.item()
                    loss_metrics = {"ce_final": ce_val}
                    loss = loss / grad_accum

            if not torch.isfinite(loss):
                print(f"WARNING: NaN/Inf loss at step {step}, skipping")
                opt.zero_grad()
                bad_step = True
                break
            scaler.scale(loss).backward()
            running_loss += ce_val
            n_tokens += inp.numel()

        if bad_step:
            continue

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR schedule
        if lr_schedule == "wsd":
            cur_lr = get_lr_wsd(step, warmup, MAX_TRAIN_STEPS, lr, min_lr)
        else:
            cur_lr = get_lr_cosine(step, warmup, MAX_TRAIN_STEPS, lr, min_lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

        scaler.step(opt)
        scaler.update()
        step += 1

        # Logging
        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            tok_s = n_tokens / max(dt, 1e-6)
            ce = running_loss / max(step - start_step, 1)
            with open(log_file, "a") as f:
                msg = f"Step {step:5d}: CE={ce:.4f} lr={cur_lr:.2e} {tok_s:.0f}tok/s"
                if use_exits:
                    extras = " ".join(f"{k}={v:.3f}" for k, v in loss_metrics.items()
                                      if k != "ce_final")
                    if extras:
                        msg += f" [{extras}]"
                print(msg, flush=True)
                f.write(msg + "\n")

        # Checkpoint (save early at step 100 for safety, then every ROLLING_SAVE)
        if step % ROLLING_SAVE == 0 or (step == 100 and start_step < 100):
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_bpt": best_bpt,
                "config": {
                    "vocab_size": VOCAB_SIZE, "dim": dim, "n_layers": n_layers,
                    "n_heads": n_heads, "ff_dim": ff_dim,
                    "exit_layers": exit_layers,
                }
            }
            # Atomic save: write to temp file, then rename
            tmp = rolling.with_suffix('.tmp')
            torch.save(ckpt, tmp)
            os.replace(str(tmp), str(rolling))
            torch.save(ckpt, ckpt_dir / f"step_{step}.pt")

        # Eval
        if step % EVAL_EVERY == 0:
            model.eval()
            cache_path = REPO / "results" / "eval_cache_16k.pt"
            if cache_path.exists():
                det_result = deterministic_eval_dense(model, cache_path)
                if det_result["bpt"] < best_bpt:
                    best_bpt = det_result["bpt"]
                print(f"Det-eval step {step}: {det_result}")
                with open(log_file, "a") as f:
                    f.write(f"Det-eval step {step}: {json.dumps(det_result)}\n")
            model.train()

        if step % 500 == 0:
            torch.cuda.empty_cache()

    print(f"Training complete at step {step}")
    # Final save
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_bpt": best_bpt,
        "config": {
            "vocab_size": VOCAB_SIZE, "dim": dim, "n_layers": n_layers,
            "n_heads": n_heads, "ff_dim": ff_dim,
            "exit_layers": exit_layers,
        }
    }
    tmp = rolling.with_suffix('.tmp')
    torch.save(ckpt, tmp)
    os.replace(str(tmp), str(rolling))
    torch.save(ckpt, ckpt_dir / f"step_{step}.pt")


def deterministic_eval_dense(model, cache_path, depths=None, device=None):
    """Evaluate dense model with fixed cache. Reports full-depth and per-exit metrics."""
    if device is None:
        device = DEVICE
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    windows = cache["windows"]  # list of (seq_len,) tensors
    total_loss = 0
    exit_losses = {}  # layer_idx -> total loss
    n_tokens = 0
    has_exits = hasattr(model, 'exit_layers') and len(model.exit_layers) > 0
    model.eval()
    use_amp = device != "cpu" and torch.cuda.is_available()
    with torch.no_grad():
        for w in windows:
            x = w.unsqueeze(0).to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            if use_amp:
                with torch.amp.autocast("cuda", dtype=DTYPE):
                    if has_exits:
                        out = model(inp, return_exits=True)
                        logits = out['logits']
                    else:
                        logits = model(inp)
            else:
                if has_exits:
                    out = model(inp, return_exits=True)
                    logits = out['logits']
                else:
                    logits = model(inp.float() if inp.is_floating_point() else inp)
            V = logits.size(-1)
            tgt_flat = tgt.reshape(-1)
            loss = F.cross_entropy(logits.float().reshape(-1, V),
                                   tgt_flat, reduction="sum")
            total_loss += loss.item()
            n_tokens += tgt.numel()
            # Per-exit losses
            if has_exits and use_amp:
                for layer_idx, e_logits in out.get('exit_logits', {}).items():
                    ce = F.cross_entropy(e_logits.float().reshape(-1, V),
                                         tgt_flat, reduction="sum").item()
                    exit_losses[layer_idx] = exit_losses.get(layer_idx, 0) + ce
    bpt = (total_loss / n_tokens) / math.log(2)
    result = {"bpt": round(bpt, 4), "n_windows": len(windows), "n_tokens": n_tokens}
    for layer_idx, el in sorted(exit_losses.items()):
        result[f"bpt_exit_{layer_idx}"] = round((el / n_tokens) / math.log(2), 4)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dense Transformer (F4 baseline / EDSR-98M)")
    parser.add_argument("--run-name", default=None,
                        help="Run name (default: dense_f4 or edsr_98m)")
    parser.add_argument("--edsr", action="store_true",
                        help="Enable EDSR-98M mode (12L/d768/12h/ff2048, exits at 4/8/12)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None,
                        help="Override FF dim (1312 for 42M match, 1536 default)")
    parser.add_argument("--det-eval", type=str, default=None,
                        help="Evaluate checkpoint(s)")
    args = parser.parse_args()

    if args.ff_dim is not None:
        FF_DIM = args.ff_dim
    if args.max_steps is not None:
        MAX_TRAIN_STEPS = args.max_steps

    if args.det_eval:
        # Load model from checkpoint and evaluate
        ckpt = torch.load(args.det_eval, weights_only=False, map_location=DEVICE)
        cfg = ckpt.get("config", {})
        model = DenseTransformer(
            vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
            dim=cfg.get("dim", DIM),
            n_layers=cfg.get("n_layers", N_LAYERS),
            n_heads=cfg.get("n_heads", N_HEADS),
            ff_dim=cfg.get("ff_dim", FF_DIM),
            exit_layers=cfg.get("exit_layers", None),
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        cache_path = REPO / "results" / "eval_cache_16k.pt"
        if not cache_path.exists():
            print(f"ERROR: {cache_path} not found. Build 16K eval cache first.")
            sys.exit(1)
        result = deterministic_eval_dense(model, cache_path)
        print(f"Step {ckpt.get('step', '?')}: BPT={result['bpt']}")
        sys.exit(0)

    run_name = args.run_name or ("edsr_98m" if args.edsr else "dense_f4")
    train(run_name=run_name, edsr=args.edsr)
