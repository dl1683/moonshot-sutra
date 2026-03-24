"""Dense transformer baseline for matched control experiments (F4).

Standard decoder-only transformer: RoPE + RMSNorm + SwiGLU + causal attention.
Uses same data pipeline and eval infrastructure as the recurrent Sutra model.

Default config: 11 layers, d=512, 8 heads, SwiGLU ff=1536, tied 16K embeddings.
~45.7M params (vs recurrent Sutra's 42M with same 16K vocab).

Usage:
  # Train from scratch (teacher-free, 5K steps)
  python code/dense_baseline.py --max-steps 5000 --run-name dense_f4

  # Det-eval on checkpoint
  python code/dense_baseline.py --det-eval results/checkpoints_dense_f4/step_5000.pt

  # Override architecture
  python code/dense_baseline.py --ff-dim 1312 --max-steps 5000  # match 42M params
"""

import sys, math, json, time, gc
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
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2)
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


class DenseTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        # Tied output head
        self.out_proj = None  # use emb.weight

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

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.dim)
        cos, sin = self.rope_cos, self.rope_sin
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        logits = F.linear(h, self.emb.weight)  # tied embeddings
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---- Training ----
def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(MAX_TRAIN_STEPS - WARMUP_STEPS, 1)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def train(run_name="dense_f4"):
    print(f"DENSE TRANSFORMER BASELINE — F4 Control Experiment")
    print(f"  Arch: {N_LAYERS}L / d={DIM} / {N_HEADS}h / ff={FF_DIM} / vocab={VOCAB_SIZE}")
    print(f"  LR: {LR:.1e} → {MIN_LR:.1e} cosine, warmup={WARMUP_STEPS}")
    print(f"  Steps: {MAX_TRAIN_STEPS}, eval every {EVAL_EVERY}")
    print(f"  Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"  Device: {DEVICE}, dtype: bf16")

    model = DenseTransformer().to(DEVICE)
    print(f"  Params: {model.count_params():,}")
    print("=" * 60)

    # Data — use 16K retokenized shards
    shard_dir = REPO / "data" / "shards_16k"
    if not shard_dir.exists() or len(list(shard_dir.glob("*.pt"))) < 64:
        print(f"ERROR: Need ≥64 shards in {shard_dir}. Currently: {len(list(shard_dir.glob('*.pt')))}")
        sys.exit(1)
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95),
                            weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # Resume
    start_step = 0
    best_bpt = float("inf")
    rolling = ckpt_dir / "rolling_latest.pt"
    if rolling.exists():
        ckpt = torch.load(rolling, weights_only=False, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_bpt = ckpt.get("best_bpt", float("inf"))
        print(f"Resumed from step {start_step}, best BPT={best_bpt:.4f}")

    step = start_step
    running_loss = 0
    n_tokens = 0
    t0 = time.time()

    model.train()
    while step < MAX_TRAIN_STEPS:
        opt.zero_grad()
        for micro in range(GRAD_ACCUM):
            inp, tgt = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(inp)
                Tc = min(logits.size(1), tgt.size(1))
                loss = F.cross_entropy(logits[:, :Tc].reshape(-1, logits.size(-1)),
                                       tgt[:, :Tc].reshape(-1))
                loss = loss / GRAD_ACCUM

            if not torch.isfinite(loss):
                print(f"WARNING: NaN/Inf loss at step {step}, skipping")
                opt.zero_grad()
                break
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM
            n_tokens += inp.numel()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR schedule
        lr = get_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        scaler.step(opt)
        scaler.update()
        step += 1

        # Logging
        if step % LOG_EVERY == 0:
            dt = time.time() - t0
            tok_s = n_tokens / max(dt, 1e-6)
            bpt = (running_loss / max(n_tokens, 1)) / math.log(2) * (inp.numel() / max(n_tokens / (step - start_step), 1))
            ce = running_loss / max(step - start_step, 1)
            with open(log_file, "a") as f:
                msg = f"Step {step:5d}: CE={ce:.4f} lr={lr:.2e} {tok_s:.0f}tok/s"
                print(msg, flush=True)
                f.write(msg + "\n")

        # Checkpoint
        if step % ROLLING_SAVE == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "step": step,
                "best_bpt": best_bpt,
                "config": {
                    "vocab_size": VOCAB_SIZE, "dim": DIM, "n_layers": N_LAYERS,
                    "n_heads": N_HEADS, "ff_dim": FF_DIM,
                }
            }
            torch.save(ckpt, rolling)
            torch.save(ckpt, ckpt_dir / f"step_{step}.pt")

        # Eval
        if step % EVAL_EVERY == 0:
            model.eval()
            cache_path = REPO / "results" / "eval_cache_16k.pt"
            if cache_path.exists():
                det_result = deterministic_eval_dense(model, cache_path)
                print(f"Det-eval step {step}: {det_result}")
            model.train()

        if step % 500 == 0:
            torch.cuda.empty_cache()

    print(f"Training complete at step {step}")
    # Final save
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_bpt": best_bpt,
        "config": {
            "vocab_size": VOCAB_SIZE, "dim": DIM, "n_layers": N_LAYERS,
            "n_heads": N_HEADS, "ff_dim": FF_DIM,
        }
    }
    torch.save(ckpt, rolling)
    torch.save(ckpt, ckpt_dir / f"step_{step}.pt")


def deterministic_eval_dense(model, cache_path, depths=None, device=None):
    """Evaluate dense model with fixed cache. No depth variation (always full forward)."""
    if device is None:
        device = DEVICE
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    windows = cache["windows"]  # list of (seq_len,) tensors
    total_loss = 0
    n_tokens = 0
    model.eval()
    use_amp = device != "cpu" and torch.cuda.is_available()
    with torch.no_grad():
        for w in windows:
            x = w.unsqueeze(0).to(device)
            inp, tgt = x[:, :-1], x[:, 1:]
            if use_amp:
                with torch.amp.autocast("cuda", dtype=DTYPE):
                    logits = model(inp)
            else:
                logits = model(inp.float() if inp.is_floating_point() else inp)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), reduction="sum")
            total_loss += loss.item()
            n_tokens += tgt.numel()
    bpt = (total_loss / n_tokens) / math.log(2)
    return {"bpt": round(bpt, 4), "n_windows": len(windows), "n_tokens": n_tokens}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dense Transformer Baseline (F4)")
    parser.add_argument("--run-name", default="dense_f4")
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
        ).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        cache_path = REPO / "results" / "eval_cache_16k.pt"
        if not cache_path.exists():
            print(f"ERROR: {cache_path} not found. Build 16K eval cache first.")
            sys.exit(1)
        result = deterministic_eval_dense(model, cache_path)
        print(f"Step {ckpt.get('step', '?')}: BPT={result['bpt']}")
        sys.exit(0)

    train(run_name=args.run_name)
