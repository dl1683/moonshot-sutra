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
MAX_TRAIN_STEPS = 10000
EVAL_EVERY = 1000
ROLLING_SAVE = 500
LOG_EVERY = 50
LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 200
GRAD_CLIP = 1.0


class NorMuon(torch.optim.Optimizer):
    """NorMuon optimizer (Li et al., arxiv:2510.05491).

    Extends Muon with per-neuron second-order normalization to fix
    the late-training instability we observed (max_act=59.6 spike).

    Algorithm:
        1. First-order momentum: M = β₁M + (1-β₁)G
        2. Newton-Schulz orthogonalization (5 iterations)
        3. Per-neuron second moment: v = β₂v + (1-β₂)mean_cols(O⊙O)
        4. Row-wise normalize: Ô = O / sqrt(v + eps)
        5. Scaled update: W -= η*λ*W + η̂*Ô where η̂ = 0.2*η*sqrt(mn)/‖Ô‖_F
    """
    def __init__(self, params, lr=0.01, beta1=0.95, beta2=0.95,
                 weight_decay=0.1, ns_iters=5, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2,
                        weight_decay=weight_decay, ns_iters=ns_iters, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz5(M, iters=5):
        """5-iteration Newton-Schulz orthogonalization."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = M / (M.norm() + 1e-7)
        for _ in range(iters):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            b1 = group['beta1']
            b2 = group['beta2']
            wd = group['weight_decay']
            ns_iters = group['ns_iters']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(g)
                    state['v'] = torch.zeros(g.shape[0], device=g.device)  # per-row

                state['step'] += 1
                m = state['momentum']
                v = state['v']

                # 1. First-order momentum
                m.mul_(b1).add_(g, alpha=1.0 - b1)

                # 2. Newton-Schulz orthogonalization
                if m.ndim == 2:
                    O = self._newton_schulz5(m, ns_iters)
                else:
                    O = m  # fallback for non-2D (shouldn't happen)

                # 3. Per-neuron second moment
                row_sq = (O * O).mean(dim=-1)  # mean over columns
                v.mul_(b2).add_(row_sq, alpha=1.0 - b2)

                # 4. Row-wise normalization
                O_hat = O / (v.unsqueeze(-1).sqrt() + eps)

                # 5. Scaled update
                mn_sqrt = math.sqrt(p.shape[0] * p.shape[1]) if p.ndim == 2 else 1.0
                fro = O_hat.norm() + 1e-7
                eta_hat = 0.2 * lr * mn_sqrt / fro

                # Weight decay
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                # Apply update
                p.data.add_(O_hat.to(p.dtype), alpha=-eta_hat)

        return loss


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


class SingleScaleRMSNorm(nn.Module):
    """Single-Scale RMSNorm: one learned scalar per block, not per channel.

    Norm(x) = g * x / sqrt(mean(x^2) + eps)
    where g is one scalar, not a dim-sized vector.

    Benefits over standard RMSNorm:
    - Fewer outlier-amplifying per-channel scales
    - Better quantization behavior (no per-channel gain differences)
    - Still controls residual stream magnitude
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))  # single scalar
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


class DyT(nn.Module):
    """Dynamic Tanh normalization (arXiv:2503.10622).

    Drop-in replacement for RMSNorm/LayerNorm. Element-wise, no reduction ops.
    Better quantization behavior: tanh bounds activations, preventing outliers.

    DyT(x) = gamma * tanh(alpha * x) + beta
    """
    def __init__(self, dim, init_alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), init_alpha))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.gamma * torch.tanh(self.alpha * x.float()).to(x.dtype) + self.beta


def make_norm(dim, norm_type="rmsnorm"):
    """Factory for normalization layers."""
    if norm_type == "dyt":
        return DyT(dim)
    elif norm_type == "ss_rmsnorm":
        return SingleScaleRMSNorm(dim)
    return RMSNorm(dim)


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
    def __init__(self, dim, n_heads, ff_dim, norm_type="rmsnorm"):
        super().__init__()
        self.attn_norm = make_norm(dim, norm_type)
        self.attn = CausalAttention(dim, n_heads)
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GatedConvBlock(nn.Module):
    """Hyena-inspired gated causal convolution block.

    Drop-in replacement for TransformerBlock (same I/O shape: B,T,D -> B,T,D).
    Uses depthwise causal conv1d with sigmoid gating instead of attention.
    Much cheaper than attention at long sequences, O(T*K) vs O(T^2).

    Architecture:
        u = Norm(x)
        a, b = split(W_in(u))   # a = value path, b = gate path
        f = CausalConv1D(a)     # depthwise causal convolution
        m = W_out(f * sigmoid(b))  # gated output
        r = x + m               # residual
        h = r + SwiGLU(Norm(r)) # FFN + residual
    """
    def __init__(self, dim, ff_dim, kernel_size=64, norm_type="rmsnorm"):
        super().__init__()
        self.mix_norm = make_norm(dim, norm_type)
        # Project to 2*dim: value path + gate path
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        # Depthwise causal conv on value path
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size - 1,  # left-pad for causal
            groups=dim, bias=False
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x, cos=None, sin=None):
        """Forward pass. cos/sin accepted but ignored (API compat with TransformerBlock)."""
        B, T, D = x.shape
        u = self.mix_norm(x)
        ab = self.in_proj(u)  # (B, T, 2*D)
        a, b = ab.chunk(2, dim=-1)  # each (B, T, D)

        # Causal depthwise conv: (B, D, T) -> conv -> truncate to causal
        a_conv = self.conv(a.transpose(1, 2))[:, :, :T]  # truncate padding
        a_conv = a_conv.transpose(1, 2)  # back to (B, T, D)

        # Gated output
        m = self.out_proj(a_conv * torch.sigmoid(b))
        r = x + m

        # FFN
        r = r + self.ffn(self.ffn_norm(r))
        return r


class ParallelHybridBlock(nn.Module):
    """Intra-layer parallel hybrid: attention + gated conv run in parallel.

    Inspired by Hymba (NVIDIA, ICLR 2025) and Falcon-H1 (TII).
    Both paths process the full hidden dim. Outputs averaged (Hymba style).
    Shared FFN on top.

    Architecture:
        u = Norm(x)
        a = Attention(u)         # full-dim attention path
        c = GatedConv(u)         # full-dim gated conv path
        x = x + 0.5*(a + c)     # Hymba-style mean combination
        x = x + FFN(Norm(x))    # shared FFN
    """
    def __init__(self, dim, n_heads, ff_dim, conv_kernel_size=64, norm_type="rmsnorm"):
        super().__init__()
        self.mix_norm = make_norm(dim, norm_type)
        # Attention path
        self.attn = CausalAttention(dim, n_heads)
        # Conv path (gated causal depthwise conv, same as GatedConvBlock mixing)
        self.conv_in = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=dim, bias=False
        )
        self.conv_out = nn.Linear(dim, dim, bias=False)
        # Shared FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path
        attn_out = self.attn(u, cos, sin)

        # Conv path (gated)
        ab = self.conv_in(u)
        a, b = ab.chunk(2, dim=-1)
        a_conv = self.conv(a.transpose(1, 2))[:, :, :T].transpose(1, 2)
        conv_out = self.conv_out(a_conv * torch.sigmoid(b))

        # Mean combination (Hymba style)
        x = x + 0.5 * (attn_out + conv_out)

        # Shared FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GQAttention(nn.Module):
    """Grouped-Query Attention with asymmetric input/output dims.

    Supports projecting from a larger hidden dim to a smaller attention dim,
    with GQA (fewer KV heads than Q heads) for KV-cache efficiency.
    """
    def __init__(self, in_dim, d_attn, n_q_heads, n_kv_heads, head_dim=64):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.d_attn = d_attn
        self.q_proj = nn.Linear(in_dim, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_q_heads * head_dim, d_attn, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)  # (B, Hq, T, Dh)
        k = k.transpose(1, 2)  # (B, Hkv, T, Dh)
        v = v.transpose(1, 2)
        # Expand KV heads for GQA
        if self.n_kv_heads < self.n_q_heads:
            rep = self.n_q_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, self.n_q_heads * self.head_dim)
        return self.out_proj(out)


class ConcatProjectHybridBlock(nn.Module):
    """HEMD-R4-S intra-layer parallel block with concat-then-project fusion.

    Architecture (Codex Round 4 spec):
        u = SSNorm(h_l)
        a = GQAttn(u; d_att, GQA)                                    # attention path
        c = W_o^c( DWConv_k(W_v^c u, k) ⊙ sigmoid(W_g^c u) )      # gated conv path
        m = W_mix [ s_a * a ; s_c * c ]                              # concat + project
        r = h_l + m
        v = SSNorm(r)
        h_{l+1} = r + SwiGLU(v)
    """
    def __init__(self, dim, ff_dim, d_attn=256, d_conv=384,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=8, n_layers=18, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.d_attn = d_attn
        self.d_conv = d_conv
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path
        self.attn = GQAttention(dim, d_attn, n_q_heads, n_kv_heads, head_dim)

        # Conv path (gated depthwise causal conv)
        self.conv_v = nn.Linear(dim, d_conv, bias=False)
        self.conv_g = nn.Linear(dim, d_conv, bias=False)
        self.dwconv = nn.Conv1d(
            d_conv, d_conv, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_conv, bias=False
        )
        self.conv_out = nn.Linear(d_conv, d_conv, bias=False)

        # Branch scales (muP-inspired init)
        self.s_a = nn.Parameter(torch.ones(1))
        self.s_c = nn.Parameter(torch.full((1,), math.sqrt(d_attn / d_conv)))

        # Fusion: concat [d_attn + d_conv] -> dim
        self.w_mix = nn.Linear(d_attn + d_conv, dim, bias=False)
        # Residual-output scaling: std = 0.02 / sqrt(2*L)
        nn.init.normal_(self.w_mix.weight, std=0.02 / math.sqrt(2 * n_layers))

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        # Scale FFN output proj too
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path -> (B, T, d_attn)
        a = self.attn(u, cos, sin)

        # Conv path -> (B, T, d_conv)
        v = self.conv_v(u)  # (B, T, d_conv)
        g = self.conv_g(u)  # (B, T, d_conv)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c = self.conv_out(v_conv * torch.sigmoid(g))

        # Concat + project with branch scales
        fused = torch.cat([self.s_a * a, self.s_c * c], dim=-1)  # (B, T, d_attn+d_conv)
        m = self.w_mix(fused)  # (B, T, dim)

        r = x + m

        # FFN
        r = r + self.ffn(self.ffn_norm(r))
        return r


class NormalizedAdditiveHybridBlock(nn.Module):
    """HEMD-R5-G intra-layer parallel block with normalized additive fusion.

    Codex Round 5 design. Synthesizes Hymba per-branch normalization with
    mean fusion stability and GQA inference efficiency.

    Architecture:
        u  = SSNorm(h)
        a0 = GQAttn(u)                          # raw dim d_attn
        c0 = DWConv1D_k(Wv u) * sigmoid(Wg u)   # raw dim d_conv
        a  = beta_a * Norm_a(Wao a0)             # project d_attn -> dim, per-branch norm
        c  = beta_c * Norm_c(Wco c0)             # project d_conv -> dim, per-channel beta
        m  = Wo(0.5 * (a + c))                   # mean fusion + output projection
        r  = h + m
        h' = r + SwiGLU(SSNorm(r))
    """
    def __init__(self, dim, ff_dim, d_attn=256, d_conv=256,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=4, n_layers=24, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.d_attn = d_attn
        self.d_conv = d_conv
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path: GQA with asymmetric dims
        self.attn = GQAttention(dim, d_attn, n_q_heads, n_kv_heads, head_dim)

        # Conv path: gated depthwise causal conv
        self.conv_v = nn.Linear(dim, d_conv, bias=False)
        self.conv_g = nn.Linear(dim, d_conv, bias=False)
        self.dwconv = nn.Conv1d(
            d_conv, d_conv, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_conv, bias=False
        )
        self.conv_out = nn.Linear(d_conv, d_conv, bias=False)

        # Branch projections: d_attn/d_conv -> dim
        self.attn_proj = nn.Linear(d_attn, dim, bias=False)
        self.conv_proj = nn.Linear(d_conv, dim, bias=False)

        # Per-branch normalization (RMSNorm, independent)
        self.norm_a = RMSNorm(dim)
        self.norm_c = RMSNorm(dim)

        # Per-channel learnable beta (Hymba style), init to 1
        self.beta_a = nn.Parameter(torch.ones(dim))
        self.beta_c = nn.Parameter(torch.ones(dim))

        # Output projection after mean fusion
        self.w_out = nn.Linear(dim, dim, bias=False)
        # Residual-output scaling: std = 0.02 / sqrt(2*L)
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2 * n_layers))

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path -> (B, T, d_attn)
        a0 = self.attn(u, cos, sin)

        # Conv path -> (B, T, d_conv)
        v = self.conv_v(u)
        g = self.conv_g(u)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c0 = self.conv_out(v_conv * torch.sigmoid(g))

        # Project to full dim + per-branch norm + per-channel beta
        a = self.beta_a * self.norm_a(self.attn_proj(a0))
        c = self.beta_c * self.norm_c(self.conv_proj(c0))

        # Normalized additive fusion + output projection
        m = self.w_out(0.5 * (a + c))

        r = x + m

        # FFN
        r = r + self.ffn(self.ffn_norm(r))
        return r


class R6FixedHybridBlock(nn.Module):
    """R6-F: Stabilized hybrid — no learned β, SS-RMSNorm branches, mean fusion.

    Codex Round 6 design. Removes per-channel β vectors (the suspected
    instability source) and replaces full RMSNorm branches with scalar
    SS-RMSNorm. Keeps projected asymmetric branches and 0.5*(a+c) fusion.

    Architecture:
        u  = g1 * h / rms(h)
        a0 = GQAttn(u)                          # d_attn=256
        c0 = DWConv1D_k(Wv u) * sigmoid(Wg u)   # d_conv=256
        a  = sa * AttnProj(a0) / rms(AttnProj(a0))   # scalar SS-RMSNorm
        c  = sc * ConvProj(c0) / rms(ConvProj(c0))   # scalar SS-RMSNorm
        m  = Wout(0.5 * (a + c))
        r  = h + m
        h' = r + SwiGLU(g2 * r / rms(r))
    """
    def __init__(self, dim, ff_dim, d_attn=256, d_conv=256,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=4, n_layers=24, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.d_attn = d_attn
        self.d_conv = d_conv
        self.layer_idx = layer_idx
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path: GQA with asymmetric dims
        self.attn = GQAttention(dim, d_attn, n_q_heads, n_kv_heads, head_dim)

        # Conv path: gated depthwise causal conv
        self.conv_v = nn.Linear(dim, d_conv, bias=False)
        self.conv_g = nn.Linear(dim, d_conv, bias=False)
        self.dwconv = nn.Conv1d(
            d_conv, d_conv, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_conv, bias=False
        )
        self.conv_out = nn.Linear(d_conv, d_conv, bias=False)

        # Branch projections: d_attn/d_conv -> dim
        self.attn_proj = nn.Linear(d_attn, dim, bias=False)
        self.conv_proj = nn.Linear(d_conv, dim, bias=False)

        # Per-branch SS-RMSNorm (scalar only — no per-channel gain)
        self.norm_a = SingleScaleRMSNorm(dim)
        self.norm_c = SingleScaleRMSNorm(dim)

        # Output projection after mean fusion
        self.w_out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2 * n_layers))

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

        # Telemetry storage (populated during forward, read at eval)
        self._telemetry = {}

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path
        a0 = self.attn(u, cos, sin)

        # Conv path
        v = self.conv_v(u)
        g = self.conv_g(u)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c0 = self.conv_out(v_conv * torch.sigmoid(g))

        # Project to full dim + scalar branch norm (no beta)
        a_proj = self.attn_proj(a0)
        c_proj = self.conv_proj(c0)
        a = self.norm_a(a_proj)
        c = self.norm_c(c_proj)

        # Mean fusion + output projection
        fused = 0.5 * (a + c)
        m = self.w_out(fused)

        r = x + m
        r = r + self.ffn(self.ffn_norm(r))

        # Telemetry (detached, no grad impact)
        if not self.training:
            with torch.no_grad():
                self._telemetry = {
                    "layer": self.layer_idx,
                    "branch_a_rms": a_proj.float().pow(2).mean().sqrt().item(),
                    "branch_c_rms": c_proj.float().pow(2).mean().sqrt().item(),
                    "branch_a_normed_rms": a.float().pow(2).mean().sqrt().item(),
                    "branch_c_normed_rms": c.float().pow(2).mean().sqrt().item(),
                    "fused_rms": fused.float().pow(2).mean().sqrt().item(),
                    "norm_a_scale": self.norm_a.weight.item(),
                    "norm_c_scale": self.norm_c.weight.item(),
                }

        return r


class R6SoftmaxHybridBlock(nn.Module):
    """R6-S: Stabilized hybrid — softmax mixing instead of β.

    Codex Round 6 backup design. Replaces per-channel β with per-channel
    softmax mixing so α_a + α_c = 1 (convex combination). Uses SS-RMSNorm
    for branches.

    Architecture:
        u  = g1 * h / rms(h)
        a0 = GQAttn(u)
        c0 = DWConv1D_k(Wv u) * sigmoid(Wg u)
        a  = sa * AttnProj(a0) / rms(AttnProj(a0))   # scalar SS-RMSNorm
        c  = sc * ConvProj(c0) / rms(ConvProj(c0))   # scalar SS-RMSNorm
        alpha = softmax([logit_a, logit_c], dim=-1)    # per-channel, sums to 1
        m  = Wout(alpha_a * a + alpha_c * c)
        r  = h + m
        h' = r + SwiGLU(g2 * r / rms(r))
    """
    def __init__(self, dim, ff_dim, d_attn=256, d_conv=256,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=4, n_layers=24, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.d_attn = d_attn
        self.d_conv = d_conv
        self.layer_idx = layer_idx
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path
        self.attn = GQAttention(dim, d_attn, n_q_heads, n_kv_heads, head_dim)

        # Conv path
        self.conv_v = nn.Linear(dim, d_conv, bias=False)
        self.conv_g = nn.Linear(dim, d_conv, bias=False)
        self.dwconv = nn.Conv1d(
            d_conv, d_conv, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_conv, bias=False
        )
        self.conv_out = nn.Linear(d_conv, d_conv, bias=False)

        # Branch projections
        self.attn_proj = nn.Linear(d_attn, dim, bias=False)
        self.conv_proj = nn.Linear(d_conv, dim, bias=False)

        # Per-branch SS-RMSNorm
        self.norm_a = SingleScaleRMSNorm(dim)
        self.norm_c = SingleScaleRMSNorm(dim)

        # Softmax mixing logits (per-channel, init to 0 = equal mixing)
        self.logit_a = nn.Parameter(torch.zeros(dim))
        self.logit_c = nn.Parameter(torch.zeros(dim))

        # Output projection
        self.w_out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2 * n_layers))

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

        # Telemetry
        self._telemetry = {}

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path
        a0 = self.attn(u, cos, sin)

        # Conv path
        v = self.conv_v(u)
        g = self.conv_g(u)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c0 = self.conv_out(v_conv * torch.sigmoid(g))

        # Project + scalar branch norm
        a_proj = self.attn_proj(a0)
        c_proj = self.conv_proj(c0)
        a = self.norm_a(a_proj)
        c = self.norm_c(c_proj)

        # Softmax mixing: alpha_a + alpha_c = 1 per channel
        logits = torch.stack([self.logit_a, self.logit_c], dim=0)  # (2, dim)
        alpha = torch.softmax(logits, dim=0)  # (2, dim)
        fused = alpha[0] * a + alpha[1] * c

        m = self.w_out(fused)

        r = x + m
        r = r + self.ffn(self.ffn_norm(r))

        # Telemetry
        if not self.training:
            with torch.no_grad():
                self._telemetry = {
                    "layer": self.layer_idx,
                    "branch_a_rms": a_proj.float().pow(2).mean().sqrt().item(),
                    "branch_c_rms": c_proj.float().pow(2).mean().sqrt().item(),
                    "branch_a_normed_rms": a.float().pow(2).mean().sqrt().item(),
                    "branch_c_normed_rms": c.float().pow(2).mean().sqrt().item(),
                    "fused_rms": fused.float().pow(2).mean().sqrt().item(),
                    "alpha_a_mean": alpha[0].mean().item(),
                    "alpha_a_min": alpha[0].min().item(),
                    "alpha_a_max": alpha[0].max().item(),
                    "norm_a_scale": self.norm_a.weight.item(),
                    "norm_c_scale": self.norm_c.weight.item(),
                }

        return r


class R7PostFusionHybridBlock(nn.Module):
    """R7-P: Post-fusion normalization hybrid — controls fused magnitude before W_out.

    Adds SS-RMSNorm AFTER mean fusion but BEFORE output projection, addressing
    the root cause identified by R6 telemetry: projected branches diverge in
    direction causing fused magnitude to shrink/spike unpredictably.

    Architecture:
        u  = g1 * h / rms(h)
        a0 = GQAttn(u)
        c0 = DWConv1D_k(Wv u) * sigmoid(Wg u)
        a  = AttnProj(a0)                      # NO branch norm (unnecessary with post-fusion)
        c  = ConvProj(c0)                       # NO branch norm
        f  = sf * (0.5*(a+c)) / rms(0.5*(a+c))  # POST-FUSION SS-RMSNorm
        m  = Wout(f)
        r  = h + m
        h' = r + SwiGLU(g2 * r / rms(r))
    """
    def __init__(self, dim, ff_dim, d_attn=256, d_conv=256,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=4, n_layers=24, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.d_attn = d_attn
        self.d_conv = d_conv
        self.layer_idx = layer_idx
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path
        self.attn = GQAttention(dim, d_attn, n_q_heads, n_kv_heads, head_dim)

        # Conv path
        self.conv_v = nn.Linear(dim, d_conv, bias=False)
        self.conv_g = nn.Linear(dim, d_conv, bias=False)
        self.dwconv = nn.Conv1d(
            d_conv, d_conv, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=d_conv, bias=False
        )
        self.conv_out = nn.Linear(d_conv, d_conv, bias=False)

        # Branch projections (no branch norms — post-fusion norm handles scale)
        self.attn_proj = nn.Linear(d_attn, dim, bias=False)
        self.conv_proj = nn.Linear(d_conv, dim, bias=False)

        # Post-fusion normalization (SS-RMSNorm on the fused output)
        self.fuse_norm = SingleScaleRMSNorm(dim)

        # Output projection
        self.w_out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w_out.weight, std=0.02 / math.sqrt(2 * n_layers))

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

        # Telemetry
        self._telemetry = {}

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path
        a0 = self.attn(u, cos, sin)

        # Conv path
        v = self.conv_v(u)
        g = self.conv_g(u)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c0 = self.conv_out(v_conv * torch.sigmoid(g))

        # Project to full dim (no branch norm)
        a = self.attn_proj(a0)
        c = self.conv_proj(c0)

        # Mean fusion + post-fusion norm
        raw_fused = 0.5 * (a + c)
        fused = self.fuse_norm(raw_fused)

        m = self.w_out(fused)

        r = x + m
        r = r + self.ffn(self.ffn_norm(r))

        # Telemetry
        if not self.training:
            with torch.no_grad():
                a_flat = a.float().reshape(-1, D)
                c_flat = c.float().reshape(-1, D)
                cos_sim = F.cosine_similarity(a_flat, c_flat, dim=-1).mean().item()
                self._telemetry = {
                    "layer": self.layer_idx,
                    "branch_a_rms": a.float().pow(2).mean().sqrt().item(),
                    "branch_c_rms": c.float().pow(2).mean().sqrt().item(),
                    "raw_fused_rms": raw_fused.float().pow(2).mean().sqrt().item(),
                    "fused_normed_rms": fused.float().pow(2).mean().sqrt().item(),
                    "fuse_norm_scale": self.fuse_norm.weight.item(),
                    "branch_cosine_sim": cos_sim,
                }

        return r


class PGQAHybridBlock(nn.Module):
    """P-GQA: Full-dim parallel hybrid with GQA — no projections.

    The only stable hybrid at 42M was the P-block (full-dim, no projection).
    This is the 100M version using GQA to keep param count viable.

    Architecture:
        u  = g1 * h / rms(h)
        a  = GQAttn(u)                    # full-dim output (512), GQA (4Q/2KV, head_dim=64)
        c  = Wco(DWConv1D_k(Wcv u) * sigmoid(Wcg u))  # full-dim (512)
        r  = h + 0.5 * (a + c)            # direct mean fusion, no projection
        h' = r + SwiGLU(g2 * r / rms(r))
    """
    def __init__(self, dim, ff_dim, d_attn=None, d_conv=None,
                 n_q_heads=4, n_kv_heads=2, head_dim=64,
                 conv_kernel_size=4, n_layers=24, layer_idx=0,
                 norm_type="ss_rmsnorm"):
        super().__init__()
        self.layer_idx = layer_idx
        self.mix_norm = make_norm(dim, norm_type)

        # Attention path — GQA with full-dim output (d_attn=dim)
        self.attn = GQAttention(dim, dim, n_q_heads, n_kv_heads, head_dim)

        # Conv path — full-dim gated depthwise conv
        self.conv_v = nn.Linear(dim, dim, bias=False)
        self.conv_g = nn.Linear(dim, dim, bias=False)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
            groups=dim, bias=False
        )
        self.conv_out = nn.Linear(dim, dim, bias=False)

        # No branch projections, no branch norms, no W_out — direct addition

        # FFN
        self.ffn_norm = make_norm(dim, norm_type)
        self.ffn = SwiGLU(dim, ff_dim)
        nn.init.normal_(self.ffn.down.weight, std=0.02 / math.sqrt(2 * n_layers))

        # Telemetry
        self._telemetry = {}

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        u = self.mix_norm(x)

        # Attention path (full-dim output)
        a = self.attn(u, cos, sin)

        # Conv path (full-dim gated depthwise conv)
        v = self.conv_v(u)
        g = self.conv_g(u)
        v_conv = self.dwconv(v.transpose(1, 2))[:, :, :T].transpose(1, 2)
        c = self.conv_out(v_conv * torch.sigmoid(g))

        # Direct mean fusion — no projection
        r = x + 0.5 * (a + c)

        r = r + self.ffn(self.ffn_norm(r))

        # Telemetry
        if not self.training:
            with torch.no_grad():
                a_flat = a.float().reshape(-1, D)
                c_flat = c.float().reshape(-1, D)
                # Per-position cosine similarity between branches
                cos_sim = F.cosine_similarity(a_flat, c_flat, dim=-1).mean().item()
                fused = 0.5 * (a + c)
                self._telemetry = {
                    "layer": self.layer_idx,
                    "branch_a_rms": a.float().pow(2).mean().sqrt().item(),
                    "branch_c_rms": c.float().pow(2).mean().sqrt().item(),
                    "fused_rms": fused.float().pow(2).mean().sqrt().item(),
                    "branch_cosine_sim": cos_sim,
                }

        return r


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


class MTPModule(nn.Module):
    """Multi-Token Prediction head (D=1). Training-only, discarded at inference.

    Takes final-layer hidden state h_t and next-token embedding E[x_{t+1}],
    predicts x_{t+2}. Uses one transformer block with shared embedding head.
    """

    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.concat_proj = nn.Linear(2 * dim, dim, bias=False)
        self.block = TransformerBlock(dim, n_heads, ff_dim)
        self.norm = RMSNorm(dim)
        # Zero-init the projection so MTP doesn't affect backbone at start
        nn.init.zeros_(self.concat_proj.weight)

    def forward(self, h_final, next_tok_emb, cos, sin):
        """
        Args:
            h_final: (B, T, D) — stopgrad'd final hidden state
            next_tok_emb: (B, T, D) — embedding of x_{t+1}
            cos, sin: RoPE buffers
        Returns:
            mtp_hidden: (B, T, D) — hidden state for predicting x_{t+2}
        """
        u = self.concat_proj(torch.cat([h_final, next_tok_emb], dim=-1))
        u = self.block(u, cos, sin)
        return self.norm(u)


class NgramMemoryFusion(nn.Module):
    """Engram-style hash-indexed n-gram memory with learned gating.

    CPU-resident sparse embedding tables (bigram + trigram) are looked up
    by hashing the recent token context. Looked-up vectors are projected
    to model dim, gated by the current hidden state, and added to residual.

    Key property: tables are on CPU (SparseAdam), so they do NOT compete
    with the backbone for GPU gradient capacity. Only the small projection
    + gate params are on GPU.
    """

    def __init__(self, dim, num_buckets=1_000_000, mem_dim=48):
        super().__init__()
        self.dim = dim
        self.mem_dim = mem_dim
        self.num_buckets = num_buckets

        # CPU-resident sparse embedding tables (bigram + trigram)
        self.bigram_table = nn.Embedding(num_buckets, mem_dim, sparse=True)
        self.trigram_table = nn.Embedding(num_buckets, mem_dim, sparse=True)
        # Zero-init so model starts at exact parity
        nn.init.zeros_(self.bigram_table.weight)
        nn.init.zeros_(self.trigram_table.weight)

        # GPU-resident projection: 2*mem_dim -> dim (combine bigram + trigram)
        self.proj = nn.Linear(2 * mem_dim, dim, bias=False)
        nn.init.zeros_(self.proj.weight)

        # GPU-resident gate: sigmoid(W_g * h) controls how much memory enters
        self.gate = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # start near-zero gate

    def _hash_bigrams(self, tokens):
        """Hash bigram context. tokens: (B, T) long tensor -> (B, T-1) bucket indices."""
        t = tokens.long()
        a = t[:, :-1]
        b = t[:, 1:]
        # Murmur3-inspired hash in PyTorch (matches numpy probe)
        C1 = 0xcc9e2d51 & 0xFFFFFFFF
        C2 = 0x1b873593 & 0xFFFFFFFF
        SEED = 0x9747b28c & 0xFFFFFFFF
        C3 = 0xe6546b64 & 0xFFFFFFFF

        h = torch.bitwise_xor(torch.tensor(SEED, device=t.device, dtype=torch.long),
                               a * C1)
        h = torch.bitwise_or(h << 15, h.remainder(2**32) >> 17) & 0xFFFFFFFF
        h = (h * C2) & 0xFFFFFFFF
        h = torch.bitwise_or(h << 13, h.remainder(2**32) >> 19) & 0xFFFFFFFF
        h = (h * 5 + C3) & 0xFFFFFFFF

        h = torch.bitwise_xor(h, b * C1)
        h = torch.bitwise_or(h << 15, h.remainder(2**32) >> 17) & 0xFFFFFFFF
        h = (h * C2) & 0xFFFFFFFF
        h = torch.bitwise_or(h << 13, h.remainder(2**32) >> 19) & 0xFFFFFFFF
        h = (h * 5 + C3) & 0xFFFFFFFF

        h = torch.bitwise_xor(h, h >> 16)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h = torch.bitwise_xor(h, h >> 13)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h = torch.bitwise_xor(h, h >> 16)

        return h.remainder(self.num_buckets)

    def _hash_trigrams(self, tokens):
        """Hash trigram context. tokens: (B, T) long tensor -> (B, T-2) bucket indices."""
        t = tokens.long()
        a, b, c = t[:, :-2], t[:, 1:-1], t[:, 2:]
        C1 = 0xcc9e2d51 & 0xFFFFFFFF
        C2 = 0x1b873593 & 0xFFFFFFFF
        SEED = 0x9747b28c & 0xFFFFFFFF
        C3 = 0xe6546b64 & 0xFFFFFFFF

        h = torch.bitwise_xor(torch.tensor(SEED, device=t.device, dtype=torch.long),
                               a * C1)
        h = torch.bitwise_or(h << 15, h.remainder(2**32) >> 17) & 0xFFFFFFFF
        h = (h * C2) & 0xFFFFFFFF
        h = torch.bitwise_or(h << 13, h.remainder(2**32) >> 19) & 0xFFFFFFFF
        h = (h * 5 + C3) & 0xFFFFFFFF

        h = torch.bitwise_xor(h, b * C1)
        h = torch.bitwise_or(h << 15, h.remainder(2**32) >> 17) & 0xFFFFFFFF
        h = (h * C2) & 0xFFFFFFFF
        h = torch.bitwise_or(h << 13, h.remainder(2**32) >> 19) & 0xFFFFFFFF
        h = (h * 5 + C3) & 0xFFFFFFFF

        h = torch.bitwise_xor(h, c * C1)
        h = torch.bitwise_or(h << 15, h.remainder(2**32) >> 17) & 0xFFFFFFFF
        h = (h * C2) & 0xFFFFFFFF
        h = torch.bitwise_or(h << 13, h.remainder(2**32) >> 19) & 0xFFFFFFFF
        h = (h * 5 + C3) & 0xFFFFFFFF

        h = torch.bitwise_xor(h, h >> 16)
        h = (h * 0x85ebca6b) & 0xFFFFFFFF
        h = torch.bitwise_xor(h, h >> 13)
        h = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h = torch.bitwise_xor(h, h >> 16)

        return h.remainder(self.num_buckets)

    def forward(self, tokens, h):
        """Look up n-gram memory and gate with hidden state.

        Args:
            tokens: (B, T) input token IDs (on GPU)
            h: (B, T, D) current hidden state (on GPU)
        Returns:
            (B, T, D) memory contribution to add to residual
        """
        B, T, D = h.shape

        # Hash n-gram context on GPU
        bi_idx = self._hash_bigrams(tokens)  # (B, T-1)
        tri_idx = self._hash_trigrams(tokens)  # (B, T-2)

        # Look up from tables (all on same device)
        bi_mem = self.bigram_table(bi_idx)  # (B, T-1, mem_dim)
        tri_mem = self.trigram_table(tri_idx)  # (B, T-2, mem_dim)

        # Pad to T length (first 1-2 tokens have no bigram/trigram context)
        bi_padded = F.pad(bi_mem, (0, 0, 1, 0))  # (B, T, mem_dim) — pad time dim
        tri_padded = F.pad(tri_mem, (0, 0, 2, 0))  # (B, T, mem_dim)

        # Concatenate and project
        mem_cat = torch.cat([bi_padded, tri_padded], dim=-1)  # (B, T, 2*mem_dim)
        mem_proj = self.proj(mem_cat)  # (B, T, D)

        # Gate with hidden state
        gate = torch.sigmoid(self.gate(h))  # (B, T, D)

        return gate * mem_proj


class DenseTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, ff_dim=FF_DIM, max_seq_len=MAX_SEQ_LEN,
                 exit_layers=None, use_mtp=False, memory_layers=None,
                 mem_buckets=1_000_000, mem_dim=48, norm_type="rmsnorm",
                 block_schedule=None, conv_kernel_size=64,
                 d_attn=None, d_conv=None, n_q_heads=None, n_kv_heads=None,
                 head_dim=64):
        """
        Args:
            exit_layers: list of 0-indexed layer numbers for early exits.
            use_mtp: if True, add D=1 MTP module (training-only).
            memory_layers: list of layers AFTER which to inject n-gram memory.
            norm_type: "rmsnorm", "ss_rmsnorm", or "dyt".
            block_schedule: list of block types per layer.
                "A" = attention (TransformerBlock), "H" = gated conv (GatedConvBlock),
                "P" = parallel hybrid (mean fusion), "C" = concat-project hybrid (R4-S),
                "N" = normalized additive hybrid (R5-G, Hymba-style branch norm),
                "F" = R6-F fixed hybrid (no β, SS-RMSNorm branches),
                "S" = R6-S softmax hybrid (softmax mixing, SS-RMSNorm branches),
                "G" = R7-P post-fusion hybrid (no branch norms, post-fusion SS-RMSNorm),
                "Q" = P-GQA full-dim hybrid (GQA + gated conv, no projections, direct mean fusion).
                If None, all layers are attention blocks.
            conv_kernel_size: kernel size for conv blocks (default 64).
            d_attn: attention dim for "C" blocks (default: dim).
            d_conv: conv dim for "C" blocks (default: dim).
            n_q_heads: query heads for GQA in "C" blocks (default: n_heads).
            n_kv_heads: KV heads for GQA in "C" blocks (default: n_q_heads).
            head_dim: head dimension for "C" blocks (default: 64).
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.norm_type = norm_type
        self.exit_layers = sorted(exit_layers) if exit_layers else []
        self.use_mtp = use_mtp
        self.memory_layers = sorted(memory_layers) if memory_layers else []
        self.emb = nn.Embedding(vocab_size, dim)

        # Build layer stack from block_schedule
        if block_schedule is None:
            block_schedule = ["A"] * n_layers  # default: all attention
        assert len(block_schedule) == n_layers, \
            f"block_schedule length {len(block_schedule)} != n_layers {n_layers}"
        self.block_schedule = block_schedule

        # Defaults for "C" block params
        _d_attn = d_attn or dim
        _d_conv = d_conv or dim
        _n_q = n_q_heads or n_heads
        _n_kv = n_kv_heads or _n_q

        layers = []
        for li, btype in enumerate(block_schedule):
            if btype == "H":
                layers.append(GatedConvBlock(dim, ff_dim, kernel_size=conv_kernel_size,
                                             norm_type=norm_type))
            elif btype == "P":
                layers.append(ParallelHybridBlock(dim, n_heads, ff_dim,
                                                   conv_kernel_size=conv_kernel_size,
                                                   norm_type=norm_type))
            elif btype == "C":
                layers.append(ConcatProjectHybridBlock(
                    dim, ff_dim, d_attn=_d_attn, d_conv=_d_conv,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            elif btype == "N":
                layers.append(NormalizedAdditiveHybridBlock(
                    dim, ff_dim, d_attn=_d_attn, d_conv=_d_conv,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            elif btype == "F":
                layers.append(R6FixedHybridBlock(
                    dim, ff_dim, d_attn=_d_attn, d_conv=_d_conv,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            elif btype == "S":
                layers.append(R6SoftmaxHybridBlock(
                    dim, ff_dim, d_attn=_d_attn, d_conv=_d_conv,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            elif btype == "G":
                layers.append(R7PostFusionHybridBlock(
                    dim, ff_dim, d_attn=_d_attn, d_conv=_d_conv,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            elif btype == "Q":
                layers.append(PGQAHybridBlock(
                    dim, ff_dim,
                    n_q_heads=_n_q, n_kv_heads=_n_kv, head_dim=head_dim,
                    conv_kernel_size=conv_kernel_size, n_layers=n_layers,
                    layer_idx=li, norm_type=norm_type))
            else:
                layers.append(TransformerBlock(dim, n_heads, ff_dim, norm_type=norm_type))
        self.layers = nn.ModuleList(layers)
        self.norm = make_norm(dim, norm_type)

        # Early-exit norms and gain heads (one per exit, NOT for final layer)
        self.exit_norms = nn.ModuleDict()
        self.gain_heads = nn.ModuleDict()
        for i in self.exit_layers:
            if i < n_layers - 1:  # final layer uses self.norm
                self.exit_norms[str(i)] = make_norm(dim, norm_type)
                self.gain_heads[str(i)] = GainHead(dim)

        # MTP module (training-only, discarded at inference)
        self.mtp = MTPModule(dim, n_heads, ff_dim) if use_mtp else None

        # N-gram memory fusion (one shared module, applied at multiple layers)
        if self.memory_layers:
            self.memory = NgramMemoryFusion(dim, num_buckets=mem_buckets, mem_dim=mem_dim)
        else:
            self.memory = None

        # RoPE cache — use explicit head_dim if "C" blocks present, else dim//n_heads
        rope_dim = head_dim if "C" in block_schedule else dim // n_heads
        cos, sin = precompute_rope(rope_dim, max_seq_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self._init_weights()

    def _init_weights(self, resumed_ckpt=None):
        # Collect modules to skip (MTP and Memory have their own zero-init)
        skip_modules = set()
        for m in self.modules():
            if isinstance(m, (MTPModule, NgramMemoryFusion)):
                for child in m.modules():
                    skip_modules.add(id(child))
        for m in self.modules():
            if id(m) in skip_modules:
                continue
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_exits=False, return_hidden=False):
        """
        Args:
            return_exits: if True, return dict with exit logits and gain predictions.
            return_hidden: if True, include pre-norm final hidden state in output dict.
                Also includes per-exit normed hidden states ('exit_hidden') and
                input embeddings ('h_input') for halting controller.
        Returns:
            If return_exits=False and return_hidden=False: final logits (B, T, V)
            If return_exits=True or return_hidden=True: dict {
                'logits': final logits,
                'exit_logits': {layer_idx: logits},
                'exit_gains': {layer_idx: gain predictions},
                'hidden': pre-norm hidden state (only if return_hidden),
                'exit_hidden': {layer_idx: normed hidden} (only if return_hidden),
                'h_input': input embeddings (only if return_hidden),
            }
        """
        B, T = x.shape
        h = self.emb(x) * math.sqrt(self.dim)
        h_input = h  # Save for halting controller
        cos, sin = self.rope_cos, self.rope_sin

        exit_logits = {}
        exit_gains = {}
        exit_hidden = {}

        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin)

            # Inject n-gram memory after this layer?
            if self.memory is not None and i in self.memory_layers:
                h = h + self.memory(x, h)

            # Early exit at this layer?
            if return_exits and i in self.exit_layers and i < self.n_layers - 1:
                h_normed = self.exit_norms[str(i)](h)
                e_logits = F.linear(h_normed, self.emb.weight)
                exit_logits[i] = e_logits
                exit_gains[i] = self.gain_heads[str(i)](h_normed, e_logits)
                if return_hidden:
                    exit_hidden[i] = h_normed

        h_pre_norm = h  # Save pre-norm hidden for MTP
        h = self.norm(h)
        logits = F.linear(h, self.emb.weight)  # tied embeddings

        if return_exits or return_hidden:
            out = {
                'logits': logits,
                'exit_logits': exit_logits,
                'exit_gains': exit_gains,
            }
            if return_hidden:
                out['hidden'] = h_pre_norm
                # Final exit hidden = normed final
                exit_hidden[self.exit_layers[-1] if self.exit_layers else self.n_layers - 1] = h
                out['exit_hidden'] = exit_hidden
                out['h_input'] = h_input
            return out
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


# ---- Exit loss weights ----
# Auto-derived from actual exit layers. Deeper exits get higher weight.
# For exits at [7, 15, 23] in a 24L model: {7: 0.15, 15: 0.30} (final always 1.0)
EXIT_WEIGHTS = {3: 0.20, 7: 0.35}  # legacy fallback for 12-layer models
GAIN_WEIGHT = 0.05


def derive_exit_weights(exit_layers, n_layers):
    """Derive exit weights from positions. Deeper exits get more weight. Final layer excluded."""
    weights = {}
    non_final = [l for l in exit_layers if l < n_layers - 1]
    for l in non_final:
        depth_frac = (l + 1) / n_layers  # 0..1
        weights[l] = round(0.10 + 0.25 * depth_frac, 2)  # 0.10..0.35 range
    return weights


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


# ---- RMFD: Routed Multi-Family Distillation ----
# Cross-tokenizer, cross-architecture KD system designed in T+L Round 8.
# Components: teacher loading, byte-span bridge, KD losses, routing.

class TeacherAdapter:
    """Wraps a HuggingFace model for use as a KD teacher.

    Handles: loading, tokenization, forward pass, hidden state extraction.
    All teacher inference is no-grad, no-train.
    """

    def __init__(self, model_name, device="cuda", dtype=torch.float16, quantize=None):
        from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
        self.name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.is_encoder = "roberta" in model_name.lower() or "bert" in model_name.lower()
        self.is_embedding = "embedding" in model_name.lower() or "bge" in model_name.lower()

        print(f"  Loading teacher: {model_name}...", flush=True)
        if self.is_encoder or self.is_embedding:
            self.model = AutoModel.from_pretrained(model_name, dtype=dtype)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, output_hidden_states=True
            )
        self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        self.hidden_dim = self.model.config.hidden_size
        self.vocab_size = getattr(self.model.config, "vocab_size", 0)
        n_params = sum(p.numel() for p in self.model.parameters())
        vram_mb = n_params * (2 if dtype == torch.float16 else 4) / 1e6
        print(f"    {n_params/1e6:.1f}M params, ~{vram_mb:.0f}MB VRAM, hidden={self.hidden_dim}")

    @torch.no_grad()
    def forward(self, texts, max_length=512):
        """Run teacher forward pass on raw text strings.

        Returns dict with:
            'logits': (B, T_teacher, V_teacher) or None for encoders
            'hidden': (B, T_teacher, D_teacher) final layer hidden states
            'token_ids': (B, T_teacher) teacher token IDs
            'byte_offsets': list of (start, end) byte offsets per token per batch item
        """
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        offsets = enc.get("offset_mapping", None)  # (B, T, 2)
        if offsets is not None and hasattr(offsets, 'to'):
            offsets = offsets.to(self.device)

        if self.is_encoder or self.is_embedding:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            return {
                "logits": None,
                "hidden": hidden,
                "token_ids": input_ids,
                "attention_mask": attention_mask,
                "byte_offsets": offsets,
            }
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
            hidden = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states else None
            return {
                "logits": logits,
                "hidden": hidden,
                "token_ids": input_ids,
                "attention_mask": attention_mask,
                "byte_offsets": offsets,
            }

    def vram_mb(self):
        return sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6


def byte_span_pool(hidden, byte_offsets, attention_mask, n_spans=16):
    """Pool hidden states into M=n_spans equal byte spans.

    This is the cross-tokenizer bridge from R8 design.
    For each raw window, split its raw bytes into n_spans equal spans.
    For each span, average the hidden states of tokens whose byte range intersects it.

    Args:
        hidden: (B, T, D) hidden states from teacher
        byte_offsets: (B, T, 2) start/end byte offsets per token (from tokenizer)
        attention_mask: (B, T) — 1 for real tokens, 0 for padding
        n_spans: number of equal byte spans

    Returns:
        span_matrix: (B, n_spans, D) averaged hidden states per byte span
    """
    B, T, D = hidden.shape
    device = hidden.device

    if byte_offsets is None:
        # Fallback: uniform position-based pooling
        span_matrix = torch.zeros(B, n_spans, D, device=device, dtype=hidden.dtype)
        for b in range(B):
            real_len = int(attention_mask[b].sum().item())
            if real_len == 0:
                continue
            chunk_size = max(1, real_len // n_spans)
            for s in range(n_spans):
                start = s * chunk_size
                end = min(start + chunk_size, real_len)
                if start < end:
                    span_matrix[b, s] = hidden[b, start:end].mean(0)
        return span_matrix

    span_matrix = torch.zeros(B, n_spans, D, device=device, dtype=hidden.dtype)
    for b in range(B):
        offsets_b = byte_offsets[b]  # (T, 2)
        mask_b = attention_mask[b]  # (T,)
        # Find total byte range
        real_offsets = offsets_b[mask_b.bool()]
        if len(real_offsets) == 0:
            continue
        max_byte = real_offsets[:, 1].max().item()
        if max_byte == 0:
            continue
        span_size = max(1, max_byte / n_spans)

        for s in range(n_spans):
            span_start = s * span_size
            span_end = (s + 1) * span_size
            # Find tokens that intersect this byte span
            tok_starts = offsets_b[:, 0].float()
            tok_ends = offsets_b[:, 1].float()
            intersects = (tok_ends > span_start) & (tok_starts < span_end) & mask_b.bool()
            if intersects.any():
                span_matrix[b, s] = hidden[b, intersects].mean(0)

    return span_matrix


def compute_cka(X, Y):
    """Linear CKA between two matrices. X: (B, D1), Y: (B, D2).

    Returns scalar CKA similarity in [0, 1].
    """
    # Center
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    # Gram matrices
    K_X = X @ X.T  # (B, B)
    K_Y = Y @ Y.T  # (B, B)
    # HSIC
    hsic_xy = (K_X * K_Y).sum()
    hsic_xx = (K_X * K_X).sum()
    hsic_yy = (K_Y * K_Y).sum()
    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-8)
    return hsic_xy / denom


def compute_state_kd_loss(student_hidden, teacher_hidden, student_offsets, teacher_offsets,
                          student_mask, teacher_mask, projector, n_spans=16):
    """State-level KD via CKA on byte-span-pooled hidden states.

    Args:
        student_hidden: (B, T_s, D_s)
        teacher_hidden: (B, T_t, D_t)
        student/teacher_offsets: (B, T, 2) byte offsets
        student/teacher_mask: (B, T) attention masks
        projector: nn.Linear(D_s, D_t) projecting student to teacher dim
        n_spans: number of byte spans

    Returns:
        loss: 1 - CKA(projected_student_spans, teacher_spans)
    """
    # Pool into byte spans
    S_s = byte_span_pool(student_hidden, student_offsets, student_mask, n_spans)  # (B, M, D_s)
    S_t = byte_span_pool(teacher_hidden, teacher_offsets, teacher_mask, n_spans)  # (B, M, D_t)

    # Project student to teacher dim
    B, M, D_s = S_s.shape
    S_s_proj = projector(S_s.reshape(B * M, D_s)).reshape(B, M, -1)  # (B, M, D_t)

    # CKA on flattened span representations
    S_s_flat = S_s_proj.reshape(B, -1)  # (B, M*D_t)
    S_t_flat = S_t.reshape(B, -1)  # (B, M*D_t)

    return 1.0 - compute_cka(S_s_flat, S_t_flat)


def compute_semantic_kd_loss(student_hidden, teacher_hidden, student_mask, teacher_mask,
                             projector):
    """Semantic KD via relational loss on pooled representations.

    The student and teacher should produce similar PAIRWISE similarity structures
    within a batch, even if the actual vectors differ.

    Args:
        student_hidden: (B, T_s, D_s)
        teacher_hidden: (B, T_t, D_t)
        student/teacher_mask: (B, T) attention masks
        projector: nn.Linear(D_s, D_sem) projecting student to semantic dim

    Returns:
        loss: Frobenius norm of Gram matrix difference, normalized by B^2
    """
    # Mean-pool over sequence (masked)
    s_mask = student_mask.unsqueeze(-1).float()  # (B, T, 1)
    t_mask = teacher_mask.unsqueeze(-1).float()

    s_pooled = (student_hidden * s_mask).sum(1) / s_mask.sum(1).clamp(min=1)  # (B, D_s)
    t_pooled = (teacher_hidden * t_mask).sum(1) / t_mask.sum(1).clamp(min=1)  # (B, D_t)

    # Project and normalize
    s_proj = F.normalize(projector(s_pooled), dim=-1)  # (B, D_sem)
    t_norm = F.normalize(t_pooled, dim=-1)  # (B, D_t)

    # Gram matrices
    K_s = s_proj @ s_proj.T  # (B, B)
    K_t = t_norm @ t_norm.T  # (B, B)

    B = K_s.size(0)
    return (K_s - K_t).pow(2).sum() / (B * B)


def compute_vocab_overlap(student_tokenizer, teacher_tokenizer, device="cuda"):
    """Pre-compute vocabulary overlap between student and teacher tokenizers.

    Uses DSKDv2 ETA approach: finds tokens with identical string representations
    in both vocabularies. Returns aligned index tensors for shared tokens.

    Args:
        student_tokenizer: tokenizers.Tokenizer (our 16K BPE)
        teacher_tokenizer: HuggingFace AutoTokenizer
        device: target device for index tensors

    Returns:
        shared_s_ids: (N_shared,) LongTensor — student vocab indices
        shared_t_ids: (N_shared,) LongTensor — teacher vocab indices
        n_shared: int — number of shared tokens
    """
    stu_vocab = student_tokenizer.get_vocab()  # {str: int}
    tea_vocab = teacher_tokenizer.get_vocab()   # {str: int}

    shared_s = []
    shared_t = []
    for tok_str, s_id in stu_vocab.items():
        if tok_str in tea_vocab:
            shared_s.append(s_id)
            shared_t.append(tea_vocab[tok_str])

    shared_s_ids = torch.tensor(shared_s, dtype=torch.long, device=device)
    shared_t_ids = torch.tensor(shared_t, dtype=torch.long, device=device)
    return shared_s_ids, shared_t_ids, len(shared_s)


def compute_cross_tok_logit_kd(
    student_logits,     # (B, T_s, V_s) — from student forward, in computation graph
    teacher_logits,     # (B, T_t, V_t) — from teacher forward, detached
    student_offsets,    # (B, T_s, 2) — student byte/char offsets
    teacher_offsets,    # (B, T_t, 2) — teacher byte/char offsets
    teacher_mask,       # (B, T_t) — teacher attention mask
    shared_s_ids,       # (N_shared,) — student vocab indices for shared tokens
    shared_t_ids,       # (N_shared,) — teacher vocab indices for shared tokens
    temperature=2.0,
    top_k=64,
    confidence_gating=False,
):
    """Cross-tokenizer logit-level KD via vocabulary overlap (DSKDv2 ETA).

    Aligns student→teacher by causal char-end (teacher end <= student end),
    then computes KL divergence on shared vocabulary with top-K teacher filtering.

    Fixes applied per Codex correctness review:
    - Causal alignment: teacher end must be <= student end (no future leaking)
    - Per-token normalization: KL divided by B*T, not just B
    - Truncation masking: positions where teacher runs out get zero loss
    - Edge-case guards for None offsets and N_shared=0

    Memory: ~2x (B, T_s, top_k) plus temporary (B, T_s, N_shared).
    """
    B, T_s, V_s = student_logits.shape
    T_t = teacher_logits.shape[1]
    N_shared = shared_s_ids.shape[0]
    device = student_logits.device

    # Validate temperature
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    # Edge case: no shared vocabulary
    if N_shared == 0:
        return torch.tensor(0.0, device=device)

    # Edge case: missing offsets
    if teacher_offsets is None or student_offsets is None:
        return torch.tensor(0.0, device=device)

    # Step 1: Causal alignment — find greatest teacher char-end <= student char-end
    # This ensures the teacher prediction only uses context up to or before the
    # student's context boundary (no future leaking).
    stu_ends = student_offsets[:, :, 1].float()   # (B, T_s)
    tea_ends = teacher_offsets[:, :, 1].float()    # (B, T_t)

    # Mask invalid teacher positions (padding)
    tea_ends_masked = tea_ends.masked_fill(~teacher_mask.bool(), -1.0)

    # For each student position, find the teacher position with the greatest
    # char-end that is <= the student's char-end.
    # signed_diff[b,i,j] = stu_ends[b,i] - tea_ends[b,j]
    # We want the j that minimizes signed_diff where signed_diff >= 0
    signed_diff = stu_ends.unsqueeze(2) - tea_ends_masked.unsqueeze(1)  # (B, T_s, T_t)
    # Make future teacher positions (negative diff) very large so argmin ignores them
    signed_diff = signed_diff.masked_fill(signed_diff < 0, 1e9)
    # Also mask padding teacher positions
    signed_diff = signed_diff.masked_fill(
        ~teacher_mask.bool().unsqueeze(1).expand(-1, T_s, -1), 1e9)
    aligned_idx = signed_diff.argmin(dim=2)  # (B, T_s)

    # Positions where no valid teacher token exists (all diffs are 1e9)
    valid_mask = signed_diff.min(dim=2).values < 1e8  # (B, T_s)
    del signed_diff

    # Pre-compute full-distribution confidence for gating (before shared-vocab extraction)
    if confidence_gating:
        t_full_max = teacher_logits.max(dim=-1).values  # (B, T_t)
        t_full_lse = torch.logsumexp(teacher_logits.float(), dim=-1)  # (B, T_t)
        aligned_full_max = t_full_max.gather(1, aligned_idx)  # (B, T_s)
        aligned_full_lse = t_full_lse.gather(1, aligned_idx)  # (B, T_s)
        full_p_max = torch.exp(aligned_full_max.float() - aligned_full_lse)  # (B, T_s)
        del t_full_max, t_full_lse, aligned_full_max, aligned_full_lse

    # Step 2: Extract shared-vocab teacher logits at aligned positions
    t_shared_all = teacher_logits[:, :, shared_t_ids]  # (B, T_t, N_shared)
    gather_idx = aligned_idx.unsqueeze(-1).expand(-1, -1, N_shared)
    aligned_t = t_shared_all.gather(1, gather_idx)  # (B, T_s, N_shared)
    del t_shared_all, gather_idx

    # Step 3: Top-K filtering on aligned teacher logits
    K = min(top_k, N_shared)
    topk_vals, topk_idx = aligned_t.topk(K, dim=-1)  # (B, T_s, K)
    del aligned_t

    # Gather student logits at same shared-vocab positions
    s_shared = student_logits[:, :, shared_s_ids]  # (B, T_s, N_shared)
    s_at_topk = s_shared.gather(-1, topk_idx)       # (B, T_s, K)
    del s_shared

    # Step 4: Per-token KL with temperature, masked for valid positions
    t_probs = F.softmax(topk_vals / temperature, dim=-1)
    s_log_probs = F.log_softmax(s_at_topk / temperature, dim=-1)
    # Per-token KL: sum over vocab dim, then average over valid positions
    per_token_kl = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)  # (B, T_s)
    # Confidence gating: scale per-token KD by teacher confidence over full distribution
    if confidence_gating:
        conf_w = torch.where(full_p_max > 0.5, 1.5,
                             torch.where(full_p_max < 0.1, 0.3, 1.0))
        per_token_kl = per_token_kl * conf_w
        del full_p_max
    per_token_kl = per_token_kl * valid_mask.float()  # zero out invalid positions
    n_valid = valid_mask.float().sum().clamp(min=1.0)
    kl = per_token_kl.sum() / n_valid  # per-token average
    return temperature * temperature * kl


def compute_token_kd_loss(student_logits, teacher_logits, temperature=2.0, top_k=32):
    """Token-level KD via KL divergence.

    For same-tokenizer or pre-aligned logits. For cross-tokenizer, use
    transported logits (computed externally).

    Args:
        student_logits: (B, T, V_s)
        teacher_logits: (B, T, V_t) — same V if same tokenizer, or transported
        temperature: softmax temperature
        top_k: only use top-K teacher logits for efficiency

    Returns:
        loss: tau^2 * KL(teacher || student), averaged over B*T
    """
    T_min = min(student_logits.size(1), teacher_logits.size(1))
    s_logits = student_logits[:, :T_min]
    t_logits = teacher_logits[:, :T_min]

    # Top-K filtering on teacher for efficiency
    if top_k and top_k < t_logits.size(-1):
        topk_vals, topk_idx = t_logits.topk(top_k, dim=-1)
        # Gather student logits at teacher's top-K positions
        s_at_topk = s_logits.gather(-1, topk_idx)
        t_soft = F.log_softmax(topk_vals / temperature, dim=-1)
        s_soft = F.log_softmax(s_at_topk / temperature, dim=-1)
    else:
        V_min = min(s_logits.size(-1), t_logits.size(-1))
        t_soft = F.log_softmax(t_logits[:, :, :V_min] / temperature, dim=-1)
        s_soft = F.log_softmax(s_logits[:, :, :V_min] / temperature, dim=-1)

    # KL(teacher || student)
    kl = F.kl_div(s_soft, t_soft.exp(), reduction="batchmean")
    return temperature * temperature * kl


def measure_activation_kurtosis(model, dataset, device, n_batches=4, seq_len=512):
    """Measure activation kurtosis across model layers.

    High kurtosis = outlier-heavy activations = bad for quantization.
    DyT should have much lower kurtosis than RMSNorm.
    """
    model.eval()
    kurtosis_per_layer = {}
    max_acts = {}

    with torch.no_grad():
        all_acts = {}
        for _ in range(n_batches):
            x, _ = dataset.sample_batch(4, seq_len, device=device, split='test')
            inp = x[:, :-1]
            h = model.emb(inp) * math.sqrt(model.dim)
            cos, sin = model.rope_cos, model.rope_sin
            for i, layer in enumerate(model.layers):
                h = layer(h, cos, sin)
                # Sample activations
                flat = h.float().reshape(-1)
                if i not in all_acts:
                    all_acts[i] = []
                # Take a subsample to avoid OOM
                idx = torch.randperm(flat.numel(), device=flat.device)[:10000]
                all_acts[i].append(flat[idx].cpu())

        for i, acts_list in all_acts.items():
            acts = torch.cat(acts_list)
            mean = acts.mean()
            std = acts.std() + 1e-8
            kurt = ((acts - mean) / std).pow(4).mean().item() - 3.0  # excess kurtosis
            kurtosis_per_layer[f"layer_{i}"] = round(kurt, 2)
            max_acts[f"layer_{i}"] = round(acts.abs().max().item(), 2)

    # Collect per-layer telemetry from R6-F/R6-S blocks (if present)
    branch_telemetry = []
    for layer in model.layers:
        if hasattr(layer, '_telemetry') and layer._telemetry:
            branch_telemetry.append(layer._telemetry.copy())
            layer._telemetry = {}

    model.train()
    result = {"kurtosis": kurtosis_per_layer, "max_activation": max_acts}
    if branch_telemetry:
        result["branch_telemetry"] = branch_telemetry
    return result


def compute_top_loss(logits, target_tokens, K=4):
    """Token Order Prediction (TOP) auxiliary loss (arXiv:2508.19228).

    Instead of predicting exact future tokens, predict their ORDER by proximity.
    Uses ListNet-style KL divergence as a learning-to-rank loss.

    Args:
        logits: (B, T, V) model output logits (from final layer)
        target_tokens: (B, T) target token IDs (shifted by 1 from input)
        K: horizon - how many future tokens to consider
    Returns:
        scalar TOP loss
    """
    B, T, V = logits.shape
    if T <= K:
        return torch.tensor(0.0, device=logits.device)

    # Build ranking target: for each position t, score each vocab token v
    # by how often and how soon it appears in the next K positions
    # s_t(v) = sum_{j=1..K} 1[y_{t+j} = v] * (K - j + 1)
    device = logits.device

    # Only compute for positions where we have K future tokens
    valid_T = T - K
    if valid_T <= 0:
        return torch.tensor(0.0, device=device)

    # Gather future tokens: (B, valid_T, K)
    future_idx = torch.arange(K, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, K)
    pos_idx = torch.arange(valid_T, device=device).unsqueeze(0).unsqueeze(-1)  # (1, valid_T, 1)
    gather_idx = (pos_idx + future_idx + 1).expand(B, -1, -1)  # +1 because targets are already shifted
    # Clamp to avoid out-of-bounds
    gather_idx = gather_idx.clamp(max=T - 1)
    future_tokens = target_tokens.unsqueeze(1).expand(-1, valid_T, -1).gather(2, gather_idx)  # (B, valid_T, K)

    # Compute proximity weights: K, K-1, ..., 1
    weights = torch.arange(K, 0, -1, device=device, dtype=logits.dtype)  # (K,)

    # Build sparse target scores using scatter_add
    target_scores = torch.zeros(B, valid_T, V, device=device, dtype=logits.dtype)
    weights_expanded = weights.unsqueeze(0).unsqueeze(0).expand(B, valid_T, -1)
    target_scores.scatter_add_(2, future_tokens.long(), weights_expanded)

    # Convert to probability distributions
    p_target = F.softmax(target_scores, dim=-1)  # (B, valid_T, V)
    q_student = F.log_softmax(logits[:, :valid_T].float(), dim=-1)  # (B, valid_T, V)

    # KL divergence: sum over vocab, mean over positions and batch
    loss = F.kl_div(q_student, p_target, reduction='batchmean')
    return loss


def compute_mtp_loss(model, hidden, input_tokens, target_tokens):
    """Compute D=1 MTP loss: predict x_{t+2} from stopgrad(h_t) + E[x_{t+1}].

    Args:
        model: DenseTransformer with MTP module
        hidden: (B, T, D) pre-norm hidden state from final layer
        input_tokens: (B, T) input token IDs
        target_tokens: (B, T) target token IDs (shifted by 1 from input)
    Returns:
        mtp_loss: scalar
    """
    if model.mtp is None:
        return torch.tensor(0.0, device=hidden.device)

    B, T, D = hidden.shape
    if T < 2:
        return torch.tensor(0.0, device=hidden.device)

    # h_t predicts x_{t+2}, so we use positions 0..T-2
    h_detached = hidden[:, :-1].detach()  # (B, T-1, D) — stopgrad

    # Next-token embeddings: E[x_{t+1}] = E[target_tokens[t]] for t in 0..T-2
    next_tok_emb = model.emb(target_tokens[:, :-1])  # (B, T-1, D)

    # MTP forward
    cos, sin = model.rope_cos, model.rope_sin
    mtp_hidden = model.mtp(h_detached, next_tok_emb, cos, sin)  # (B, T-1, D)

    # Predict x_{t+2} = target_tokens[t+1] for t in 0..T-2
    mtp_logits = F.linear(mtp_hidden, model.emb.weight)  # (B, T-1, V)

    # Target: x_{t+2} — which is target_tokens shifted by 1
    mtp_target = target_tokens[:, 1:]  # (B, T-1)

    # Align lengths (mtp_logits and mtp_target should match)
    min_len = min(mtp_logits.size(1), mtp_target.size(1))
    mtp_loss = F.cross_entropy(
        mtp_logits[:, :min_len].reshape(-1, mtp_logits.size(-1)),
        mtp_target[:, :min_len].reshape(-1)
    )
    return mtp_loss


def train(run_name="dense_f4", edsr=False, from_ckpt=None, weight_file=None, init_from=None):
    # Select config
    if from_ckpt:
        # Load config from checkpoint — used for 200M and custom models
        _ckpt = torch.load(from_ckpt, weights_only=False, map_location="cpu")
        cfg = _ckpt.get("config", {})
        dim = cfg.get("dim", 768)
        n_layers = cfg.get("n_layers", 12)
        n_heads = cfg.get("n_heads", 12)
        ff_dim = cfg.get("ff_dim", 2048)
        exit_layers = cfg.get("exit_layers", [3, 7, 11])
        lr, min_lr = 3e-4, 1e-5
        warmup = 500
        # Adjust batch size for larger models
        if dim >= 1024:
            batch_size, grad_accum = 4, 8  # 32 effective, less VRAM per micro
        else:
            batch_size, grad_accum = 8, 4
        lr_schedule = "wsd"
        mode_name = f"Custom-{sum(p.numel() for p in DenseTransformer(dim=dim, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim, exit_layers=exit_layers).parameters())/1e6:.0f}M"
        del _ckpt  # Free memory before training
    elif edsr:
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
    dataset = ShardedDataset(shard_dir=str(shard_dir), weight_file=weight_file)
    if weight_file:
        print(f"  Weighted sampling: {weight_file}")

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
        # Support both checkpoint schemas: probe checkpoints use model_state_dict/optimizer_state_dict
        model_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
        opt_key = "optimizer_state_dict" if "optimizer_state_dict" in ckpt else "optimizer"
        model.load_state_dict(ckpt[model_key])
        opt.load_state_dict(ckpt[opt_key])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        best_bpt = ckpt.get("best_bpt", float("inf"))
        print(f"Resumed from step {start_step}, best BPT={best_bpt:.4f}")
    elif init_from is not None:
        # Initialize from another run's checkpoint (for A/B tests)
        init_ckpt = torch.load(init_from, weights_only=False, map_location=DEVICE)
        init_model_key = "model_state_dict" if "model_state_dict" in init_ckpt else "model"
        init_opt_key = "optimizer_state_dict" if "optimizer_state_dict" in init_ckpt else "optimizer"
        model.load_state_dict(init_ckpt[init_model_key])
        opt.load_state_dict(init_ckpt[init_opt_key])
        if "scaler" in init_ckpt:
            scaler.load_state_dict(init_ckpt["scaler"])
        start_step = init_ckpt.get("step", 0)
        best_bpt = init_ckpt.get("best_bpt", float("inf"))
        print(f"Initialized from {Path(init_from).name} at step {start_step}")
        del init_ckpt

    step = start_step
    running_loss = 0
    n_tokens = 0
    t0 = time.time()
    use_exits = edsr and exit_layers
    # Derive exit weights from actual configured exits (fixes hard-coded {3,7} for 24L models)
    actual_exit_weights = derive_exit_weights(exit_layers, n_layers) if exit_layers else EXIT_WEIGHTS
    if use_exits:
        print(f"  Exit weights: {actual_exit_weights}")

    model.train()
    while step < MAX_TRAIN_STEPS:
        opt.zero_grad()
        bad_step = False
        for micro in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                if use_exits:
                    out = model(inp, return_exits=True)
                    loss, loss_metrics = compute_edsr_loss(out, tgt, exit_weights=actual_exit_weights)
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
            tokens_seen = step * grad_accum * batch_size * SEQ_LEN
            with open(log_file, "a") as f:
                msg = f"Step {step:5d}: CE={ce:.4f} lr={cur_lr:.2e} {tok_s:.0f}tok/s tok={tokens_seen/1e6:.1f}M"
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


def calibrate_exits(ckpt_path, run_name="exit_calibration"):
    """Post-hoc exit threshold calibration: sweep entropy/margin thresholds.

    Codex R2 Probe 3.4: Test whether token-level adaptive exit can work
    WITHOUT a trained routing network. Uses entropy and margin thresholds
    on existing exit logits to decide per-token exit depth.

    Compares: always-full-depth vs always-exit-8 vs thresholded policies.
    """
    print(f"=== POST-HOC EXIT CALIBRATION ===")
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 12),
        n_heads=cfg.get("n_heads", 12),
        ff_dim=cfg.get("ff_dim", 2048),
        exit_layers=cfg.get("exit_layers", [3, 7, 11]),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded step {step}, exits at {cfg.get('exit_layers', [3,7,11])}")

    # Load eval cache
    cache_path = REPO / "results" / "eval_cache_16k.pt"
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    windows = cache["windows"]

    # Collect per-token metrics at each exit
    print(f"Collecting per-token exit metrics on {len(windows)} windows...")
    all_data = []  # list of (entropy_4, margin_4, entropy_8, margin_8, ce_4, ce_8, ce_12)
    exit_layers = cfg.get("exit_layers", [3, 7, 11])

    with torch.no_grad():
        for w in windows:
            x = w.unsqueeze(0).to(DEVICE)
            inp, tgt = x[:, :-1], x[:, 1:]
            with torch.amp.autocast("cuda", dtype=DTYPE):
                out = model(inp, return_exits=True)

            V = out['logits'].size(-1)
            tgt_flat = tgt.reshape(-1)
            T = tgt.size(1)

            # Per-token CE at each exit
            ce_final = F.cross_entropy(
                out['logits'][:, :T].float().reshape(-1, V), tgt_flat,
                reduction='none')  # (T,)
            ce_exits = {}
            entropy_exits = {}
            margin_exits = {}

            for li in exit_layers[:-1]:  # non-final exits
                if li in out['exit_logits']:
                    e_logits = out['exit_logits'][li][:, :T].float()
                    ce_exits[li] = F.cross_entropy(
                        e_logits.reshape(-1, V), tgt_flat,
                        reduction='none')  # (T,)
                    # Entropy
                    probs = F.softmax(e_logits.squeeze(0), dim=-1)
                    ent = -(probs * (probs + 1e-10).log()).sum(-1)  # (T,)
                    entropy_exits[li] = ent
                    # Margin
                    topk = e_logits.squeeze(0).topk(2, dim=-1).values
                    margin_exits[li] = topk[:, 0] - topk[:, 1]  # (T,)

            # Store per-token data
            for t_idx in range(T):
                row = {"ce_final": ce_final[t_idx].item()}
                for li in exit_layers[:-1]:
                    if li in ce_exits:
                        row[f"ce_{li}"] = ce_exits[li][t_idx].item()
                        row[f"ent_{li}"] = entropy_exits[li][t_idx].item()
                        row[f"margin_{li}"] = margin_exits[li][t_idx].item()
                all_data.append(row)

    print(f"Collected {len(all_data)} tokens")

    # Now sweep thresholds
    import numpy as np
    data = {k: np.array([d[k] for d in all_data]) for k in all_data[0].keys()}

    ce_final_mean = data["ce_final"].mean()
    bpt_final = ce_final_mean / math.log(2)
    print(f"\nBaseline: always-full-depth BPT = {bpt_final:.4f}")

    # Always exit at layer 8
    if "ce_7" in data:
        bpt_exit8 = data["ce_7"].mean() / math.log(2)
        print(f"Always exit-8: BPT = {bpt_exit8:.4f} (at 67% compute)")
    elif "ce_3" in data:
        bpt_exit4 = data["ce_3"].mean() / math.log(2)
        print(f"Always exit-4: BPT = {bpt_exit4:.4f} (at 33% compute)")

    results = {
        "checkpoint": str(ckpt_path),
        "step": step,
        "n_tokens": len(all_data),
        "bpt_full": round(bpt_final, 4),
        "policies": [],
    }

    # Threshold sweep on exit-8 (layer index 7) using entropy
    if "ent_7" in data and "ce_7" in data:
        print(f"\n--- Entropy threshold sweep (exit-8) ---")
        ent_vals = data["ent_7"]
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        for pct in percentiles:
            threshold = np.percentile(ent_vals, pct)
            # Exit early if entropy < threshold (confident tokens exit early)
            exit_early = ent_vals < threshold
            frac_early = exit_early.mean()
            # Mix: early tokens use exit-8 CE, rest use final CE
            mixed_ce = np.where(exit_early, data["ce_7"], data["ce_final"])
            mixed_bpt = mixed_ce.mean() / math.log(2)
            avg_depth = frac_early * 8/12 + (1 - frac_early) * 1.0
            compute_saving = (1 - avg_depth) * 100
            policy = {
                "type": "entropy_exit8",
                "threshold": round(float(threshold), 4),
                "percentile": pct,
                "frac_early": round(float(frac_early), 4),
                "bpt": round(float(mixed_bpt), 4),
                "bpt_delta": round(float(mixed_bpt - bpt_final), 4),
                "avg_depth_frac": round(float(avg_depth), 4),
                "compute_saving_pct": round(float(compute_saving), 1),
            }
            results["policies"].append(policy)
            print(f"  p{pct:2d}: thresh={threshold:.2f} exit={frac_early*100:.0f}% "
                  f"BPT={mixed_bpt:.4f} (delta={mixed_bpt-bpt_final:+.4f}) "
                  f"save={compute_saving:.0f}%")

    # Threshold sweep on exit-4 (layer index 3) using entropy
    if "ent_3" in data and "ce_3" in data:
        print(f"\n--- Entropy threshold sweep (exit-4) ---")
        ent_vals = data["ent_3"]
        percentiles = [5, 10, 15, 20, 30]
        for pct in percentiles:
            threshold = np.percentile(ent_vals, pct)
            exit_early = ent_vals < threshold
            frac_early = exit_early.mean()
            mixed_ce = np.where(exit_early, data["ce_3"], data["ce_final"])
            mixed_bpt = mixed_ce.mean() / math.log(2)
            avg_depth = frac_early * 4/12 + (1 - frac_early) * 1.0
            compute_saving = (1 - avg_depth) * 100
            policy = {
                "type": "entropy_exit4",
                "threshold": round(float(threshold), 4),
                "percentile": pct,
                "frac_early": round(float(frac_early), 4),
                "bpt": round(float(mixed_bpt), 4),
                "bpt_delta": round(float(mixed_bpt - bpt_final), 4),
                "avg_depth_frac": round(float(avg_depth), 4),
                "compute_saving_pct": round(float(compute_saving), 1),
            }
            results["policies"].append(policy)
            print(f"  p{pct:2d}: thresh={threshold:.2f} exit={frac_early*100:.0f}% "
                  f"BPT={mixed_bpt:.4f} (delta={mixed_bpt-bpt_final:+.4f}) "
                  f"save={compute_saving:.0f}%")

    # Two-stage cascade: exit-4 if very confident, else exit-8 if confident, else full
    if all(k in data for k in ["ent_3", "ent_7", "ce_3", "ce_7"]):
        print(f"\n--- Two-stage cascade (exit-4 -> exit-8 -> full) ---")
        for p4 in [5, 10]:
            t4 = np.percentile(data["ent_3"], p4)
            for p8 in [30, 50, 70]:
                t8 = np.percentile(data["ent_7"], p8)
                exit_at_4 = data["ent_3"] < t4
                exit_at_8 = (~exit_at_4) & (data["ent_7"] < t8)
                exit_at_12 = ~(exit_at_4 | exit_at_8)
                mixed_ce = (
                    exit_at_4 * data["ce_3"] +
                    exit_at_8 * data["ce_7"] +
                    exit_at_12 * data["ce_final"]
                )
                mixed_bpt = mixed_ce.mean() / math.log(2)
                f4 = exit_at_4.mean()
                f8 = exit_at_8.mean()
                f12 = exit_at_12.mean()
                avg_depth = f4 * 4/12 + f8 * 8/12 + f12 * 1.0
                compute_saving = (1 - avg_depth) * 100
                policy = {
                    "type": "cascade_4_8_12",
                    "threshold_4": round(float(t4), 4),
                    "threshold_8": round(float(t8), 4),
                    "percentile_4": p4,
                    "percentile_8": p8,
                    "frac_exit4": round(float(f4), 4),
                    "frac_exit8": round(float(f8), 4),
                    "frac_exit12": round(float(f12), 4),
                    "bpt": round(float(mixed_bpt), 4),
                    "bpt_delta": round(float(mixed_bpt - bpt_final), 4),
                    "avg_depth_frac": round(float(avg_depth), 4),
                    "compute_saving_pct": round(float(compute_saving), 1),
                }
                results["policies"].append(policy)
                print(f"  p4={p4} p8={p8}: "
                      f"exit4={f4*100:.0f}% exit8={f8*100:.0f}% full={f12*100:.0f}% "
                      f"BPT={mixed_bpt:.4f} (delta={mixed_bpt-bpt_final:+.4f}) "
                      f"save={compute_saving:.0f}%")

    # Save results
    out_path = REPO / "results" / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCalibration complete. Results: {out_path}")


def early_exit_inference(ckpt_path, run_name="early_exit_bench"):
    """Real early-exit inference engine with wall-clock benchmarking.

    Measures actual latency/throughput for:
    1. Full-depth (all 12 layers)
    2. Always exit-8 (layers 1-8 only)
    3. Calibrated exit-8 (entropy threshold, exit early when confident)

    Uses the eval cache for consistent comparison. Reports BPT + real latency.
    """
    print("=== EARLY-EXIT INFERENCE ENGINE ===")
    print(f"Checkpoint: {ckpt_path}")

    # Load model
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    exit_layers = cfg.get("exit_layers", [3, 7, 11])
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 12),
        n_heads=cfg.get("n_heads", 12),
        ff_dim=cfg.get("ff_dim", 2048),
        exit_layers=exit_layers,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.half()  # FP16 for inference speed
    step = ckpt.get("step", "?")
    print(f"Loaded step {step}, exits at {exit_layers}")

    # Load eval cache
    cache_path = REPO / "results" / "eval_cache_16k.pt"
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    windows = cache["windows"]
    print(f"Eval windows: {len(windows)}")

    # Get calibrated thresholds from exit_calibration.json if available
    cal_path = REPO / "results" / "exit_calibration.json"
    ent_threshold_p30 = 2.6607  # default from step_3000 calibration
    if cal_path.exists():
        import json as _json
        cal = _json.load(open(cal_path))
        for p in cal.get("policies", []):
            if p.get("name") == "exit8_ent_p30":
                ent_threshold_p30 = p.get("threshold", ent_threshold_p30)
                break
    print(f"Exit-8 entropy threshold (p30): {ent_threshold_p30:.4f}")

    # Helper: forward through layers with optional early stopping
    @torch.no_grad()
    def forward_adaptive(model, inp, policy="full"):
        """Forward pass with exit policy.

        policy: "full" | "always_exit8" | "calibrated_p30"
        Returns: logits (B, T, V), dict with per-token exit info
        """
        B, T = inp.shape
        h = model.emb(inp) * math.sqrt(model.dim)
        cos, sin = model.rope_cos, model.rope_sin

        exit8_idx = exit_layers[-2] if len(exit_layers) >= 2 else exit_layers[0]
        final_idx = model.n_layers - 1

        for i, layer in enumerate(model.layers):
            h = layer(h, cos, sin)

            # Check for early exit at exit-8
            if i == exit8_idx and policy != "full":
                h_normed = model.exit_norms[str(i)](h)
                e_logits = F.linear(h_normed, model.emb.weight.half())

                if policy == "always_exit8":
                    return e_logits, {"exit_layer": i, "fraction_early": 1.0}

                if policy == "calibrated_p30":
                    # Per-token entropy check
                    probs = F.softmax(e_logits.float(), dim=-1)
                    ent = -(probs * (probs + 1e-10).log()).sum(-1)  # (B, T)
                    early_mask = ent < ent_threshold_p30  # (B, T) bool

                    # If ALL tokens can exit early, return exit logits
                    frac = early_mask.float().mean().item()
                    if frac > 0.95:  # batch-level: if >95% can exit, all exit
                        return e_logits, {"exit_layer": i, "fraction_early": frac}

                    # Otherwise, continue to full depth
                    # In a real per-token engine, we'd split the batch.
                    # For this benchmark: continue to full depth if any token needs it.

        h = model.norm(h)
        logits = F.linear(h, model.emb.weight.half())
        return logits, {"exit_layer": final_idx, "fraction_early": 0.0}

    # Benchmark function
    def bench_policy(policy_name, n_warmup=3, n_runs=5):
        """Run eval with a given policy, return metrics and timing."""
        # Warmup
        for _ in range(n_warmup):
            w = windows[0].unsqueeze(0).to(DEVICE)
            inp = w[:, :-1]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                forward_adaptive(model, inp, policy=policy_name)
            torch.cuda.synchronize()

        total_loss = 0.0
        n_tokens = 0
        early_fracs = []

        torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(n_runs):
            for w in windows:
                x = w.unsqueeze(0).to(DEVICE)
                inp, tgt = x[:, :-1], x[:, 1:]
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits, info = forward_adaptive(model, inp, policy=policy_name)
                V = logits.size(-1)
                T = min(logits.size(1), tgt.size(1))
                loss = F.cross_entropy(
                    logits[:, :T].float().reshape(-1, V),
                    tgt[:, :T].reshape(-1), reduction="sum")
                total_loss += loss.item()
                n_tokens += tgt[:, :T].numel()
                early_fracs.append(info["fraction_early"])

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        bpt = (total_loss / n_tokens) / math.log(2)
        tok_per_sec = n_tokens / elapsed
        avg_early = sum(early_fracs) / len(early_fracs) if early_fracs else 0

        return {
            "policy": policy_name,
            "bpt": round(bpt, 4),
            "latency_s": round(elapsed / n_runs, 4),
            "throughput_tok_s": round(tok_per_sec, 0),
            "avg_early_exit_frac": round(avg_early, 4),
            "n_tokens_total": n_tokens,
            "n_runs": n_runs,
        }

    # Run all policies
    results = {}
    for policy in ["full", "always_exit8", "calibrated_p30"]:
        print(f"\nBenchmarking: {policy}...")
        r = bench_policy(policy)
        results[policy] = r
        print(f"  BPT={r['bpt']:.4f}  latency={r['latency_s']:.3f}s/pass  "
              f"throughput={r['throughput_tok_s']:.0f}tok/s  "
              f"early_frac={r['avg_early_exit_frac']:.2%}")

    # Summary
    full = results["full"]
    print("\n" + "=" * 60)
    print("EARLY-EXIT INFERENCE RESULTS")
    print("=" * 60)
    for name, r in results.items():
        delta_bpt = r["bpt"] - full["bpt"]
        speedup = full["latency_s"] / max(r["latency_s"], 1e-6)
        print(f"  {name:20s}: BPT={r['bpt']:.4f} ({delta_bpt:+.4f})  "
              f"speedup={speedup:.2f}x  {r['throughput_tok_s']:.0f}tok/s")

    # Save
    out_path = REPO / "results" / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


def expand_98m_to_200m(parent_ckpt, output_path=None):
    """Net2Wider + layer insertion: expand 98M checkpoint to ~197M model.

    Codex T+L R3 spec:
    - 98M: 12 layers, d=768, 12 heads, ff=2048, exits at [3,7,11]
    - 200M: 14 layers, d=1024, 16 heads, ff=2816, exits at [4,9,13]

    Expansion method:
    1. Width: Net2Wider duplication (768→1024). First 768 channels copy,
       remaining 256 duplicate random parent channels. Outgoing weights
       divided by replication count.
    2. Depth: Insert 2 new layers with zero-init residual gates.
    3. Heads: 12→16 by duplicating 4 parent heads, dividing output.

    Step-0 parity: not exact (width expansion changes output slightly due
    to channel duplication), but close enough for warm-start.
    """
    import random

    print("=== 98M -> 200M WARM-START EXPANSION ===")
    print(f"Parent: {parent_ckpt}")

    # Load parent
    ckpt = torch.load(parent_ckpt, weights_only=False, map_location="cpu")
    parent_cfg = ckpt.get("config", {})
    parent_sd = ckpt["model"]

    # Parent config
    p_dim = parent_cfg.get("dim", 768)
    p_layers = parent_cfg.get("n_layers", 12)
    p_heads = parent_cfg.get("n_heads", 12)
    p_ff = parent_cfg.get("ff_dim", 2048)
    p_exits = parent_cfg.get("exit_layers", [3, 7, 11])
    p_head_dim = p_dim // p_heads

    # Child config
    c_dim = 1024
    c_layers = 14
    c_heads = 16
    c_ff = 2816
    c_head_dim = c_dim // c_heads  # 64, same as parent
    c_exits = [4, 9, 13]  # exits after layers 5, 10, 14 (1-indexed)

    assert p_head_dim == c_head_dim == 64, "Head dim must match"

    print(f"Parent: {p_layers}L d={p_dim} {p_heads}h ff={p_ff} exits={p_exits}")
    print(f"Child:  {c_layers}L d={c_dim} {c_heads}h ff={c_ff} exits={c_exits}")

    # Build channel duplication map: 768 → 1024
    random.seed(42)
    dup_map = list(range(p_dim))  # first 768 map to themselves
    extra = c_dim - p_dim  # 256 extra channels
    dup_sources = [random.randint(0, p_dim - 1) for _ in range(extra)]
    dup_map.extend(dup_sources)

    # Count how many times each parent channel is used
    from collections import Counter
    channel_counts = Counter(dup_map)

    # Similarly for FF dim: 2048 → 2816
    ff_dup_map = list(range(p_ff))
    ff_extra = c_ff - p_ff
    ff_dup_sources = [random.randint(0, p_ff - 1) for _ in range(ff_extra)]
    ff_dup_map.extend(ff_dup_sources)
    ff_channel_counts = Counter(ff_dup_map)

    # Head duplication map: 12 → 16
    head_dup_map = list(range(p_heads))
    head_extra = c_heads - p_heads
    head_dup_sources = [random.randint(0, p_heads - 1) for _ in range(head_extra)]
    head_dup_map.extend(head_dup_sources)
    head_counts = Counter(head_dup_map)

    def widen_incoming(w, dim_map, axis=1):
        """Duplicate channels on the given axis using dim_map."""
        return torch.index_select(w, axis, torch.tensor(dim_map))

    def widen_outgoing(w, dim_map, counts, axis=0):
        """Duplicate channels and divide by replication count."""
        expanded = torch.index_select(w, axis, torch.tensor(dim_map))
        # Build scale factors
        scale = torch.tensor([1.0 / counts[dim_map[i]] for i in range(len(dim_map))])
        shape = [1] * expanded.dim()
        shape[axis] = len(dim_map)
        return expanded * scale.reshape(shape)

    # Create child model
    child = DenseTransformer(
        vocab_size=parent_cfg.get("vocab_size", VOCAB_SIZE),
        dim=c_dim, n_layers=c_layers, n_heads=c_heads, ff_dim=c_ff,
        exit_layers=c_exits,
    )

    child_sd = child.state_dict()

    # Layer mapping: which child layer gets which parent layer
    # Parent: 12 layers [0..11], Child: 14 layers [0..13]
    # Insert 2 new layers. Strategy: insert after parent layers 5 and 10
    # Parent 0-5 → Child 0-5, new child 6, Parent 6-10 → Child 7-11, new child 12, Parent 11 → Child 13
    # This gives even spacing and preserves exit-relative positions.
    parent_to_child = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
        # child 6 is NEW (zero-init)
        6: 7, 7: 8, 8: 9, 9: 10, 10: 11,
        # child 12 is NEW (zero-init)
        11: 13,
    }
    new_layers = {6, 12}  # child layers that are new (zero-init)

    print(f"Layer mapping: parent->child = {parent_to_child}")
    print(f"New layers (zero-init): {new_layers}")

    # 1. Embedding: widen from 768 → 1024
    emb_w = parent_sd["emb.weight"]  # (V, 768)
    child_sd["emb.weight"] = widen_incoming(emb_w, dup_map, axis=1)
    print(f"Embedding: {emb_w.shape} -> {child_sd['emb.weight'].shape}")

    # 2. Copy transformer layers
    for p_idx, c_idx in parent_to_child.items():
        prefix_p = f"layers.{p_idx}"
        prefix_c = f"layers.{c_idx}"

        # Attention norms (RMSNorm): weight is (dim,) → widen
        norm_w = parent_sd[f"{prefix_p}.attn_norm.weight"]
        child_sd[f"{prefix_c}.attn_norm.weight"] = norm_w[dup_map]

        # FFN norms
        ffn_norm_w = parent_sd[f"{prefix_p}.ffn_norm.weight"]
        child_sd[f"{prefix_c}.ffn_norm.weight"] = ffn_norm_w[dup_map]

        # QKV projection: (3*dim, dim) → widen both dims
        # QKV is stored as (3*768, 768). For widening:
        # - Input dim (axis=1): duplicate channels
        # - Output dim (axis=0): expand heads from 12→16
        qkv_w = parent_sd[f"{prefix_p}.attn.qkv.weight"]  # (3*768, 768)
        # Split into Q, K, V each (768, 768)
        q_w, k_w, v_w = qkv_w.chunk(3, dim=0)

        # Each of Q, K, V has shape (n_heads * head_dim, dim)
        # Reshape to (n_heads, head_dim, dim) to duplicate heads
        def expand_qkv_block(w):
            """Expand (p_heads*64, p_dim) → (c_heads*64, c_dim)"""
            w_heads = w.reshape(p_heads, p_head_dim, p_dim)  # (12, 64, 768)
            # Duplicate heads: 12→16
            expanded_heads = w_heads[head_dup_map]  # (16, 64, 768)
            # Widen input dim: 768→1024
            expanded_heads = widen_incoming(expanded_heads, dup_map, axis=2)  # (16, 64, 1024)
            return expanded_heads.reshape(c_heads * c_head_dim, c_dim)

        new_q = expand_qkv_block(q_w)
        new_k = expand_qkv_block(k_w)
        new_v = expand_qkv_block(v_w)
        child_sd[f"{prefix_c}.attn.qkv.weight"] = torch.cat([new_q, new_k, new_v], dim=0)

        # Output projection: (dim, n_heads*head_dim) → widen + head duplication
        out_w = parent_sd[f"{prefix_p}.attn.out.weight"]  # (768, 768)
        # Input side: (dim, n_heads*64) → duplicate heads on axis=1
        out_heads = out_w.reshape(p_dim, p_heads, p_head_dim)  # (768, 12, 64)
        out_expanded = out_heads[:, head_dup_map, :]  # (768, 16, 64)
        # Divide by head count to preserve output magnitude
        for i in range(c_heads):
            parent_head = head_dup_map[i]
            out_expanded[:, i, :] /= head_counts[parent_head]
        out_expanded = out_expanded.reshape(p_dim, c_heads * c_head_dim)
        # Widen output dim: 768→1024
        child_sd[f"{prefix_c}.attn.out.weight"] = widen_outgoing(
            widen_incoming(out_expanded.T, dup_map, axis=1).T,
            dup_map, channel_counts, axis=0)

        # SwiGLU: gate (ff_dim, dim), up (ff_dim, dim), down (dim, ff_dim)
        gate_w = parent_sd[f"{prefix_p}.ffn.gate.weight"]  # (2048, 768)
        up_w = parent_sd[f"{prefix_p}.ffn.up.weight"]      # (2048, 768)
        down_w = parent_sd[f"{prefix_p}.ffn.down.weight"]  # (768, 2048)

        # gate/up: widen input (768→1024), widen output (2048→2816)
        child_sd[f"{prefix_c}.ffn.gate.weight"] = widen_incoming(
            widen_outgoing(gate_w, ff_dup_map, ff_channel_counts, axis=0),
            dup_map, axis=1) * (p_ff / c_ff)  # scale to preserve magnitude
        # Actually, for gate and up, we just duplicate — no need for count division
        # since these are INCOMING to the FF. Let me redo:
        gate_widened = widen_incoming(gate_w, dup_map, axis=1)  # (2048, 1024)
        gate_widened = widen_incoming(gate_widened.T, ff_dup_map, axis=1).T  # (2816, 1024)
        child_sd[f"{prefix_c}.ffn.gate.weight"] = gate_widened

        up_widened = widen_incoming(up_w, dup_map, axis=1)  # (2048, 1024)
        up_widened = widen_incoming(up_widened.T, ff_dup_map, axis=1).T  # (2816, 1024)
        child_sd[f"{prefix_c}.ffn.up.weight"] = up_widened

        # down: widen input (2048→2816 on axis=1), widen output (768→1024 on axis=0)
        down_widened = widen_incoming(down_w, ff_dup_map, axis=1)  # (768, 2816)
        # Divide by ff dup count on input axis
        ff_scale = torch.tensor([1.0 / ff_channel_counts[ff_dup_map[i]]
                                 for i in range(c_ff)])
        down_widened = down_widened * ff_scale.unsqueeze(0)
        down_widened = widen_outgoing(down_widened, dup_map, channel_counts, axis=0)
        child_sd[f"{prefix_c}.ffn.down.weight"] = down_widened

    # 3. New layers (child 6 and 12): zero-init residual contribution
    # The model already init'd these randomly. We want zero-init for the
    # output projections so the new layers start as identity (residual skip).
    for new_idx in new_layers:
        prefix = f"layers.{new_idx}"
        # Zero out attention output and FFN down projection
        child_sd[f"{prefix}.attn.out.weight"].zero_()
        child_sd[f"{prefix}.ffn.down.weight"].zero_()
        print(f"  Layer {new_idx}: zero-init output projections (residual skip)")

    # 4. Final norm: widen
    child_sd["norm.weight"] = parent_sd["norm.weight"][dup_map]

    # 5. Exit norms and gain heads for new exit positions
    # Parent exits: [3,7,11] → Child exits: [4,9,13]
    # Mapping: parent exit 3 (after parent layer 4) → child exit 4 (after child layer 5)
    # parent exit 7 (after parent layer 8) → child exit 9 (after child layer 10)
    # parent exit 11 is final (no exit norm needed)
    exit_map = {3: 4, 7: 9}  # parent exit → child exit (non-final exits only)
    for p_exit, c_exit in exit_map.items():
        p_key = f"exit_norms.{p_exit}.weight"
        c_key = f"exit_norms.{c_exit}.weight"
        if p_key in parent_sd:
            child_sd[c_key] = parent_sd[p_key][dup_map]

        # Gain heads: Linear(dim+2, 256) and Linear(256, 1)
        p_prefix = f"gain_heads.{p_exit}"
        c_prefix = f"gain_heads.{c_exit}"
        for suffix in [".fc1.weight", ".fc1.bias", ".fc2.weight", ".fc2.bias"]:
            p_k = p_prefix + suffix
            c_k = c_prefix + suffix
            if p_k in parent_sd:
                w = parent_sd[p_k]
                if suffix == ".fc1.weight":
                    # (256, dim+2) — widen input dim+2 → c_dim+2
                    # First dim channels are hidden state, last 2 are entropy+margin
                    new_w = torch.zeros(w.shape[0], c_dim + 2)
                    new_w[:, :p_dim] = w[:, :p_dim][:, dup_map]
                    new_w[:, c_dim:] = w[:, p_dim:]  # copy entropy/margin weights
                    child_sd[c_k] = new_w
                else:
                    child_sd[c_k] = w.clone()

    # Load into child model
    child.load_state_dict(child_sd)
    total_params = child.count_params()
    print(f"\nChild model: {total_params:,} parameters")

    # Save
    if output_path is None:
        output_path = REPO / "results" / "checkpoints_200m" / "expanded_from_98m.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_ckpt = {
        "model": child.state_dict(),
        "step": 0,  # Reset step for 200M line
        "best_bpt": float("inf"),
        "parent_step": ckpt.get("step", 0),
        "parent_ckpt": str(parent_ckpt),
        "config": {
            "vocab_size": parent_cfg.get("vocab_size", VOCAB_SIZE),
            "dim": c_dim, "n_layers": c_layers,
            "n_heads": c_heads, "ff_dim": c_ff,
            "exit_layers": c_exits,
        },
    }
    torch.save(save_ckpt, output_path)
    print(f"Saved expanded checkpoint: {output_path}")
    print(f"Config: {save_ckpt['config']}")
    return str(output_path)


def probe_mtp(parent_ckpt, steps=1000, run_name="probe_mtp"):
    """MTP probe: add D=1 MTP to EDSR parent, train, compare to control.

    T+L Round 1, Probe 3: tests whether MTP improves data efficiency.
    """
    MTP_WEIGHT = 0.30
    print(f"=== MTP PROBE (D=1) ===")
    print(f"Parent: {parent_ckpt}")
    print(f"Steps: {steps}, MTP weight: {MTP_WEIGHT}")

    # Load parent
    ckpt = torch.load(parent_ckpt, weights_only=False, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    dim = cfg.get("dim", 768)
    n_layers = cfg.get("n_layers", 12)
    n_heads = cfg.get("n_heads", 12)
    ff_dim = cfg.get("ff_dim", 2048)
    exit_layers = cfg.get("exit_layers", [3, 7, 11])

    # Create model WITH MTP
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=dim, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim,
        exit_layers=exit_layers, use_mtp=True,
    ).to(DEVICE)

    # Load parent weights (strict=False since MTP is new)
    model.load_state_dict(ckpt["model"], strict=False)
    parent_step = ckpt.get("step", 0)
    print(f"Loaded parent step {parent_step}, added MTP module")
    print(f"  Backbone params: {sum(p.numel() for n,p in model.named_parameters() if 'mtp' not in n):,}")
    print(f"  MTP params: {sum(p.numel() for n,p in model.named_parameters() if 'mtp' in n):,}")
    print(f"  Total: {model.count_params():,}")

    # Data
    shard_dir = REPO / "data" / "shards_16k"
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"

    # Optimizer: separate LR for backbone vs MTP
    backbone_params = [p for n, p in model.named_parameters() if 'mtp' not in n]
    mtp_params = [p for n, p in model.named_parameters() if 'mtp' in n]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": 1.5e-4},
        {"params": mtp_params, "lr": 4e-4},
    ], betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    batch_size, grad_accum = 8, 4
    warmup = 100
    use_exits = bool(exit_layers)

    model.train()
    running_ar_loss = 0
    running_mtp_loss = 0
    n_tokens = 0
    t0 = time.time()

    for step in range(1, steps + 1):
        opt.zero_grad()
        for micro in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                if use_exits:
                    out = model(inp, return_exits=True, return_hidden=True)
                    ar_loss, metrics = compute_edsr_loss(out, tgt)
                else:
                    out = model(inp, return_hidden=True)
                    if isinstance(out, dict):
                        logits = out['logits']
                        Tc = min(logits.size(1), tgt.size(1))
                        ar_loss = F.cross_entropy(logits[:, :Tc].reshape(-1, logits.size(-1)),
                                                   tgt[:, :Tc].reshape(-1))
                        metrics = {"ce_final": ar_loss.item()}
                    else:
                        Tc = min(out.size(1), tgt.size(1))
                        ar_loss = F.cross_entropy(out[:, :Tc].reshape(-1, out.size(-1)),
                                                   tgt[:, :Tc].reshape(-1))
                        metrics = {"ce_final": ar_loss.item()}

                # MTP loss
                hidden = out['hidden'] if isinstance(out, dict) else None
                if hidden is not None:
                    mtp_loss = compute_mtp_loss(model, hidden, inp, tgt)
                else:
                    mtp_loss = torch.tensor(0.0, device=DEVICE)

                total_loss = (ar_loss + MTP_WEIGHT * mtp_loss) / grad_accum

            if not torch.isfinite(total_loss):
                print(f"WARNING: NaN loss at step {step}")
                opt.zero_grad()
                break
            scaler.scale(total_loss).backward()
            running_ar_loss += metrics["ce_final"]
            running_mtp_loss += mtp_loss.item()
            n_tokens += inp.numel()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        # LR warmup
        if step <= warmup:
            frac = step / warmup
            for i, pg in enumerate(opt.param_groups):
                base_lr = 1.5e-4 if i == 0 else 4e-4
                pg["lr"] = base_lr * frac

        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            dt = time.time() - t0
            ar_avg = running_ar_loss / step
            mtp_avg = running_mtp_loss / step
            tok_s = n_tokens / max(dt, 1e-6)
            msg = f"Step {step:5d}: AR={ar_avg:.4f} MTP={mtp_avg:.4f} {tok_s:.0f}tok/s"
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        if step % 500 == 0 or step == steps:
            save = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "parent_step": parent_step,
                "config": {
                    "vocab_size": VOCAB_SIZE, "dim": dim, "n_layers": n_layers,
                    "n_heads": n_heads, "ff_dim": ff_dim,
                    "exit_layers": exit_layers, "use_mtp": True,
                }
            }
            torch.save(save, ckpt_dir / f"step_{step}.pt")

        if step % 500 == 0 or step == steps:
            model.eval()
            cache_path = REPO / "results" / "eval_cache_16k.pt"
            if cache_path.exists():
                det_result = deterministic_eval_dense(model, cache_path)
                msg = f"Det-eval step {step}: {det_result}"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            model.train()

        if step % 500 == 0:
            torch.cuda.empty_cache()

    # Save final results
    results = {
        "probe": "mtp_d1",
        "parent": str(parent_ckpt),
        "steps": steps,
        "mtp_weight": MTP_WEIGHT,
        "final_ar_loss": running_ar_loss / steps,
        "final_mtp_loss": running_mtp_loss / steps,
    }
    with open(REPO / "results" / f"{run_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMTP probe complete. Results: results/{run_name}_metrics.json")


class HaltingController(nn.Module):
    """Differentiable per-token halting for 3-exit mixture.

    T+L Round 1 design: at each non-final exit, compute continue probability
    rho from features (entropy, logit margin, state-change magnitude, hidden state).
    Mixture weights beta follow a cascade: if you stop at exit k, you didn't
    continue past it.

    For exits at layers [4, 8, 12] (0-indexed [3, 7, 11]):
      rho_4 = sigma(w_4 . q_4 + v_4 . h_4)
      rho_8 = sigma(w_8 . q_8 + v_8 . h_8)
      beta_4  = 1 - rho_4
      beta_8  = rho_4 * (1 - rho_8)
      beta_12 = rho_4 * rho_8

    q_l = [entropy(logits_l), margin(logits_l), ||h_l - h_{l-1}||_2]
    """

    def __init__(self, dim, exit_layers):
        """
        Args:
            dim: model dimension
            exit_layers: list of 0-indexed exit layer indices (e.g., [3, 7, 11])
        """
        super().__init__()
        self.exit_layers = sorted(exit_layers)
        # For each non-final exit: small linear from features -> scalar
        # Features: entropy(1) + margin(1) + state_change(1) + hidden(dim) = dim+3
        self.halt_projs = nn.ModuleDict()
        for i in self.exit_layers[:-1]:  # all except last exit
            self.halt_projs[str(i)] = nn.Linear(dim + 3, 1, bias=True)
            # Initialize bias to +2.0 so rho starts near sigmoid(2)=0.88
            # meaning "continue" is the default — model starts at full depth
            nn.init.zeros_(self.halt_projs[str(i)].weight)
            nn.init.constant_(self.halt_projs[str(i)].bias, 2.0)

    def compute_features(self, h_curr, h_prev, logits):
        """Compute halting features q_l for a given exit.

        Args:
            h_curr: (B, T, D) hidden state at this exit
            h_prev: (B, T, D) hidden state at previous exit (or input embeds)
            logits: (B, T, V) logits at this exit
        Returns:
            features: (B, T, D+3)
        """
        with torch.no_grad():
            probs = F.softmax(logits.float(), dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(-1, keepdim=True)  # (B,T,1)
            topk = logits.topk(2, dim=-1).values
            margin = (topk[:, :, 0:1] - topk[:, :, 1:2])  # (B,T,1)
            state_change = (h_curr - h_prev).norm(dim=-1, keepdim=True)  # (B,T,1)
        return torch.cat([
            entropy.to(h_curr.dtype),
            margin.to(h_curr.dtype),
            state_change.to(h_curr.dtype),
            h_curr
        ], dim=-1)  # (B, T, D+3)

    def forward(self, exit_hidden, exit_logits, h_input):
        """Compute mixture weights for all exits.

        Args:
            exit_hidden: dict {layer_idx: (B,T,D)} normed hidden at each exit
            exit_logits: dict {layer_idx: (B,T,V)} logits at each exit
            h_input: (B,T,D) input embeddings (for state-change at first exit)
        Returns:
            betas: dict {layer_idx: (B,T,1)} mixture weights summing to 1
            rhos: dict {layer_idx: (B,T,1)} continue probabilities (for logging)
        """
        rhos = {}
        betas = {}

        # Get ordered exits
        exits = sorted(exit_hidden.keys())

        # Compute continue probabilities for non-final exits
        for idx, layer_i in enumerate(exits[:-1]):
            h_curr = exit_hidden[layer_i]
            h_prev = h_input if idx == 0 else exit_hidden[exits[idx - 1]]
            logits = exit_logits[layer_i]
            feat = self.compute_features(h_curr, h_prev, logits)
            rho = torch.sigmoid(self.halt_projs[str(layer_i)](feat))  # (B,T,1)
            rhos[layer_i] = rho

        # Compute mixture weights using cascade
        # beta_0 = 1 - rho_0
        # beta_1 = rho_0 * (1 - rho_1)
        # beta_2 = rho_0 * rho_1  (last exit gets all remaining probability)
        cum_continue = torch.ones_like(list(rhos.values())[0])  # (B,T,1)
        for idx, layer_i in enumerate(exits):
            if layer_i in rhos:
                betas[layer_i] = cum_continue * (1 - rhos[layer_i])
                cum_continue = cum_continue * rhos[layer_i]
            else:
                # Last exit gets remaining probability
                betas[layer_i] = cum_continue

        return betas, rhos


def compute_halting_loss(out, tgt, betas, lambda_cost=0.0):
    """Compute loss with differentiable halting mixture.

    Instead of fixed-weight multi-exit loss, the prediction is a weighted
    mixture of exit softmax distributions, and there's a compute penalty
    that encourages early stopping.

    Args:
        out: model output dict with 'logits', 'exit_logits'
        tgt: target tokens (B, T)
        betas: dict {layer_idx: (B,T,1)} mixture weights from HaltingController
        lambda_cost: compute penalty coefficient (ramped from 0 to 0.08)
    Returns:
        (total_loss, metrics_dict)
    """
    V = out['logits'].size(-1)
    B, T = tgt.shape

    # Collect all exit softmax distributions
    all_exits = sorted(betas.keys())
    exit_probs = []
    exit_weights = []

    for layer_i in all_exits:
        if layer_i in out['exit_logits']:
            logits_i = out['exit_logits'][layer_i]
        else:
            # Final exit
            logits_i = out['logits']
        Tc = min(logits_i.size(1), T)
        probs_i = F.softmax(logits_i[:, :Tc].float(), dim=-1)  # (B, Tc, V)
        exit_probs.append(probs_i)
        exit_weights.append(betas[layer_i][:, :Tc])

    # Mixture distribution: p_t = sum_l beta_l * softmax(logits_l)
    Tc = min(p.size(1) for p in exit_probs)
    mixture = torch.zeros(B, Tc, V, device=tgt.device, dtype=torch.float32)
    for probs_i, weight_i in zip(exit_probs, exit_weights):
        mixture += weight_i[:, :Tc] * probs_i[:, :Tc]

    # Cross-entropy of mixture
    mixture = mixture.clamp(min=1e-10)
    tgt_flat = tgt[:, :Tc].reshape(-1)
    log_probs = mixture.log().reshape(-1, V)
    ce_loss = F.nll_loss(log_probs, tgt_flat)

    # Compute penalty: weighted sum of betas * relative cost
    # cost_coeffs: fraction of compute at each exit
    n_exits = len(all_exits)
    cost_coeffs = [(i + 1) / n_exits for i in range(n_exits)]
    compute_cost = torch.zeros(B, Tc, 1, device=tgt.device, dtype=torch.float32)
    for idx, (layer_i, coeff) in enumerate(zip(all_exits, cost_coeffs)):
        compute_cost += betas[layer_i][:, :Tc] * coeff
    cost_penalty = lambda_cost * compute_cost.mean()

    total_loss = ce_loss + cost_penalty

    # Metrics
    metrics = {
        "ce_mixture": ce_loss.item(),
        "cost_penalty": cost_penalty.item(),
        "lambda_cost": lambda_cost,
    }
    for idx, layer_i in enumerate(all_exits):
        beta_mean = betas[layer_i][:, :Tc].mean().item()
        metrics[f"beta_{layer_i}"] = beta_mean

    return total_loss, metrics


def probe_halting(parent_ckpt, steps=1000, run_name="probe_halting"):
    """Halting controller probe: differentiable 3-exit mixture with compute penalty.

    T+L Round 1, Probe 6: tests whether learned halting improves efficiency.
    Replaces fixed exit weights with a learned mixture, adds compute penalty.
    """
    LAMBDA_COST_MAX = 0.08
    LAMBDA_RAMP_STEPS = 500  # ramp from 0 to LAMBDA_COST_MAX over first 500 steps
    print(f"=== HALTING CONTROLLER PROBE ===")
    print(f"Parent: {parent_ckpt}")
    print(f"Steps: {steps}, lambda_cost ramp: 0 -> {LAMBDA_COST_MAX} over {LAMBDA_RAMP_STEPS} steps")

    # Load parent
    ckpt = torch.load(parent_ckpt, weights_only=False, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    dim = cfg.get("dim", 768)
    n_layers = cfg.get("n_layers", 12)
    n_heads = cfg.get("n_heads", 12)
    ff_dim = cfg.get("ff_dim", 2048)
    exit_layers = cfg.get("exit_layers", [3, 7, 11])

    # Create model (no MTP for this probe — isolate halting effect)
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=dim, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim,
        exit_layers=exit_layers,
    ).to(DEVICE)

    # Load parent weights
    model.load_state_dict(ckpt["model"], strict=False)
    parent_step = ckpt.get("step", 0)

    # Create halting controller
    halt_ctrl = HaltingController(dim, exit_layers).to(DEVICE)

    backbone_params_n = sum(p.numel() for p in model.parameters())
    halt_params_n = sum(p.numel() for p in halt_ctrl.parameters())
    print(f"Loaded parent step {parent_step}")
    print(f"  Backbone params: {backbone_params_n:,}")
    print(f"  Halting params: {halt_params_n:,}")
    print(f"  Total: {backbone_params_n + halt_params_n:,}")

    # Data
    shard_dir = REPO / "data" / "shards_16k"
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"

    # Optimizer: backbone at lower LR, halting at higher LR
    opt = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1.5e-4},
        {"params": halt_ctrl.parameters(), "lr": 4e-4},
    ], betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    batch_size, grad_accum = 8, 4
    warmup = 100

    model.train()
    halt_ctrl.train()
    running_metrics = {}
    n_tokens = 0
    t0 = time.time()

    for step in range(1, steps + 1):
        # Ramp lambda_cost
        lambda_cost = LAMBDA_COST_MAX * min(step / LAMBDA_RAMP_STEPS, 1.0)

        opt.zero_grad()
        for micro in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                # Forward with exits + per-exit hidden states
                out = model(inp, return_exits=True, return_hidden=True)

                # exit_hidden and exit_logits now include all exits
                exit_hidden_dict = out['exit_hidden']
                exit_logits_dict = dict(out['exit_logits'])
                exit_logits_dict[exit_layers[-1]] = out['logits']
                h_input = out['h_input']

                # Compute mixture weights
                betas, rhos = halt_ctrl(exit_hidden_dict, exit_logits_dict, h_input)

                # Halting loss
                loss, metrics = compute_halting_loss(out, tgt, betas, lambda_cost)
                loss = loss / grad_accum

            if not torch.isfinite(loss):
                print(f"WARNING: NaN loss at step {step}")
                opt.zero_grad()
                break
            scaler.scale(loss).backward()

            for k, v in metrics.items():
                running_metrics[k] = running_metrics.get(k, 0) + v
            n_tokens += inp.numel()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(halt_ctrl.parameters()), GRAD_CLIP)

        # LR warmup
        if step <= warmup:
            frac = step / warmup
            for i, pg in enumerate(opt.param_groups):
                base_lr = 1.5e-4 if i == 0 else 4e-4
                pg["lr"] = base_lr * frac

        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            dt = time.time() - t0
            tok_s = n_tokens / max(dt, 1e-6)
            denom = step * grad_accum  # accumulated over microbatches
            ce = running_metrics.get("ce_mixture", 0) / denom
            cost = running_metrics.get("cost_penalty", 0) / denom
            beta_strs = []
            for i in exit_layers:
                b = running_metrics.get(f"beta_{i}", 0) / denom
                beta_strs.append(f"b{i}={b:.3f}")
            msg = f"Step {step:5d}: CE={ce:.4f} cost={cost:.4f} {' '.join(beta_strs)} {tok_s:.0f}tok/s lam={lambda_cost:.4f}"
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        if step % 500 == 0 or step == steps:
            model.eval()
            halt_ctrl.eval()
            cache_path = REPO / "results" / "eval_cache_16k.pt"
            if cache_path.exists():
                det_result = deterministic_eval_dense(model, cache_path)
                msg = f"Det-eval step {step}: {det_result}"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            model.train()
            halt_ctrl.train()

    # Save results
    results = {
        "probe": "halting_controller",
        "parent": str(parent_ckpt),
        "steps": steps,
        "lambda_cost_max": LAMBDA_COST_MAX,
    }
    for k, v in running_metrics.items():
        results[k] = v / steps
    with open(REPO / "results" / f"{run_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nHalting probe complete. Results: results/{run_name}_metrics.json")


def probe_memory(parent_ckpt, steps=1000, run_name="probe_memory",
                  mem_buckets=1_000_000, mem_dim=48):
    """Memory fusion probe: add Engram-style n-gram memory to EDSR parent.

    T+L Round 1, Probe 2: tests whether CPU-resident n-gram memory with
    learned gating improves BPT without competing for GPU gradient capacity.

    Memory tables are on CPU (SparseAdam). Only projection + gate on GPU.
    Zero-initialized so model starts at exact parity with parent.
    """
    MEMORY_LAYERS = [1, 4, 7]  # After layers 2, 5, 8 (1-indexed)
    print(f"=== MEMORY FUSION PROBE ===")
    print(f"Parent: {parent_ckpt}")
    print(f"Steps: {steps}, memory at layers {[l+1 for l in MEMORY_LAYERS]}")
    print(f"Tables: {mem_buckets:,} buckets x {mem_dim}d, bigram + trigram")

    # Load parent
    ckpt = torch.load(parent_ckpt, weights_only=False, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    dim = cfg.get("dim", 768)
    n_layers = cfg.get("n_layers", 12)
    n_heads = cfg.get("n_heads", 12)
    ff_dim = cfg.get("ff_dim", 2048)
    exit_layers = cfg.get("exit_layers", [3, 7, 11])

    # Create model WITH memory
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=dim, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim,
        exit_layers=exit_layers, memory_layers=MEMORY_LAYERS,
        mem_buckets=mem_buckets, mem_dim=mem_dim,
    ).to(DEVICE)

    # Load parent weights (strict=False since memory is new)
    model.load_state_dict(ckpt["model"], strict=False)
    parent_step = ckpt.get("step", 0)

    # Count params
    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'memory' not in n)
    mem_table_params = sum(p.numel() for n, p in model.named_parameters()
                           if 'table' in n)
    mem_gate_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'memory' in n and 'table' not in n)
    print(f"Loaded parent step {parent_step}")
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Memory table params: {mem_table_params:,} ({mem_table_params * 4 / 1e6:.0f}MB)")
    print(f"  Memory gate+proj params: {mem_gate_params:,}")
    print(f"  Estimated VRAM for tables: {mem_table_params * 2 / 1e6:.0f}MB (bf16)")

    # Data
    shard_dir = REPO / "data" / "shards_16k"
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"

    # Optimizer: separate groups for backbone, memory gate/proj, memory tables
    backbone_p = [p for n, p in model.named_parameters()
                  if 'memory' not in n and p.requires_grad]
    mem_dense_p = [p for n, p in model.named_parameters()
                   if 'memory' in n and 'table' not in n and p.requires_grad]
    mem_sparse_p = [p for n, p in model.named_parameters()
                    if 'table' in n and p.requires_grad]

    # Dense optimizer for backbone + memory gate/proj
    opt_dense = torch.optim.AdamW([
        {"params": backbone_p, "lr": 1.5e-4},
        {"params": mem_dense_p, "lr": 4e-4},
    ], betas=(0.9, 0.95), weight_decay=0.1)

    # SparseAdam for sparse embedding tables (handles sparse gradients)
    opt_sparse = torch.optim.SparseAdam(mem_sparse_p, lr=4e-4)

    scaler = torch.amp.GradScaler("cuda")

    batch_size, grad_accum = 8, 4
    warmup = 100
    use_exits = bool(exit_layers)

    model.train()
    running_loss = 0
    n_tokens = 0
    t0 = time.time()

    for step in range(1, steps + 1):
        opt_dense.zero_grad()
        opt_sparse.zero_grad()

        for micro in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, SEQ_LEN, device=DEVICE)
            with torch.amp.autocast("cuda", dtype=DTYPE):
                if use_exits:
                    out = model(inp, return_exits=True)
                    loss, metrics = compute_edsr_loss(out, tgt)
                    ce_val = metrics["ce_final"]
                    loss = loss / grad_accum
                else:
                    logits = model(inp)
                    Tc = min(logits.size(1), tgt.size(1))
                    loss = F.cross_entropy(
                        logits[:, :Tc].reshape(-1, logits.size(-1)),
                        tgt[:, :Tc].reshape(-1))
                    ce_val = loss.item()
                    loss = loss / grad_accum

            if not torch.isfinite(loss):
                print(f"WARNING: NaN loss at step {step}")
                opt_dense.zero_grad()
                opt_sparse.zero_grad()
                break
            scaler.scale(loss).backward()
            running_loss += ce_val
            n_tokens += inp.numel()

        scaler.unscale_(opt_dense)
        # Manually unscale sparse grads (SparseAdam doesn't work with GradScaler)
        inv_scale = 1.0 / scaler.get_scale()
        for p in mem_sparse_p:
            if p.grad is not None:
                if p.grad.is_sparse:
                    p.grad = p.grad.coalesce()
                    p.grad._values().mul_(inv_scale)
                else:
                    p.grad.data.mul_(inv_scale)

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None and not p.grad.is_sparse],
            GRAD_CLIP)

        # LR warmup
        if step <= warmup:
            frac = step / warmup
            for i, pg in enumerate(opt_dense.param_groups):
                base_lr = 1.5e-4 if i == 0 else 4e-4
                pg["lr"] = base_lr * frac
            for pg in opt_sparse.param_groups:
                pg["lr"] = 4e-4 * frac

        scaler.step(opt_dense)
        opt_sparse.step()
        scaler.update()

        if step % 50 == 0:
            dt = time.time() - t0
            tok_s = n_tokens / max(dt, 1e-6)
            ce_avg = running_loss / (step * grad_accum)
            # Check memory gate activity
            with torch.no_grad():
                gate_mean = torch.sigmoid(model.memory.gate.bias).mean().item()
            msg = (f"Step {step:5d}: CE={ce_avg:.4f} gate={gate_mean:.4f} "
                   f"{tok_s:.0f}tok/s")
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        if step % 500 == 0 or step == steps:
            model.eval()
            cache_path = REPO / "results" / "eval_cache_16k.pt"
            if cache_path.exists():
                det_result = deterministic_eval_dense(model, cache_path)
                msg = f"Det-eval step {step}: {det_result}"
                print(msg)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
            model.train()

        if step % 500 == 0:
            torch.cuda.empty_cache()

    # Save results
    results = {
        "probe": "memory_fusion",
        "parent": str(parent_ckpt),
        "steps": steps,
        "memory_layers": MEMORY_LAYERS,
        "mem_buckets": mem_buckets,
        "mem_dim": mem_dim,
        "final_ce": running_loss / (steps * grad_accum),
        "backbone_params": backbone_params,
        "mem_table_params": mem_table_params,
        "mem_gate_params": mem_gate_params,
    }
    with open(REPO / "results" / f"{run_name}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMemory probe complete. Results: results/{run_name}_metrics.json")


def probe_committee_map(student_ckpt, n_windows=50, seq_len=512, n_spans=16):
    """Audit-only committee map: run student + all 4 teachers on validation windows.

    Collects per-span metrics for T+L R2 Ekalavya design:
    - compatibility: cosine(student_span_repr, teacher_span_repr) via byte_span_pool
    - confidence: mean max(softmax(teacher_logits)) per span
    - student_difficulty: mean entropy(student_softmax) per span
    - gap: KL(teacher || student) on shared vocab per span
    - novelty: mean JS divergence between teacher pairs per span
    - target route shares based on R1 routing formula

    Processes teachers one at a time for VRAM efficiency.
    No training — pure diagnostic.
    """
    import numpy as np

    TEACHERS = [
        "Qwen/Qwen3-0.6B-Base",
        "LiquidAI/LFM2.5-1.2B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "google/embeddinggemma-300m",
    ]
    TEACHER_SHORT = ["q06", "lfm", "q17", "emb"]

    print(f"\n{'='*60}")
    print(f"COMMITTEE MAP AUDIT")
    print(f"  Student: {student_ckpt}")
    print(f"  Teachers: {TEACHERS}")
    print(f"  Windows: {n_windows}, seq_len={seq_len}, spans={n_spans}")
    print(f"{'='*60}\n")

    # ---- Load student ----
    ckpt = torch.load(student_ckpt, weights_only=False, map_location="cpu")
    stu_cfg = ckpt.get("config", {})
    model = DenseTransformer(
        vocab_size=stu_cfg.get("vocab_size", VOCAB_SIZE),
        dim=stu_cfg.get("dim", 768),
        n_layers=stu_cfg.get("n_layers", 24),
        n_heads=stu_cfg.get("n_heads", 12),
        ff_dim=stu_cfg.get("ff_dim", 2048),
        exit_layers=stu_cfg.get("exit_layers", [7, 15, 23]),
        norm_type=stu_cfg.get("norm_type", "rmsnorm"),
        block_schedule=stu_cfg.get("block_schedule", None),
        n_q_heads=stu_cfg.get("n_q_heads", None),
        n_kv_heads=stu_cfg.get("n_kv_heads", None),
        head_dim=stu_cfg.get("head_dim", 64),
    ).to(DEVICE)
    init_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[init_key])
    model.eval()
    student_dim = stu_cfg.get("dim", 768)
    step = ckpt.get("step", "?")
    del ckpt
    torch.cuda.empty_cache()
    print(f"  Student loaded (step {step}, dim={student_dim})")

    # ---- Load student tokenizer ----
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    student_tokenizer = Tokenizer.from_file(str(tok_path))

    # ---- Prepare validation data: get raw text windows ----
    shard_dir = REPO / "data" / "shards_16k"
    dataset = ShardedDataset(shard_dir=str(shard_dir))

    # Sample validation windows as token IDs, then decode to text for teachers
    print(f"  Sampling {n_windows} validation windows...")
    windows_tokens = []
    windows_text = []
    for _ in range(n_windows):
        x, _ = dataset.sample_batch(1, seq_len, device="cpu", split="test")
        tokens = x[0].tolist()
        windows_tokens.append(tokens)
        text = student_tokenizer.decode(tokens)
        windows_text.append(text)

    # ---- Student inference: collect logits + hidden states ----
    print("  Running student inference...")
    student_logits_all = []   # list of (T, V_s) on CPU
    student_hidden_all = []   # list of (T, D_s) on CPU
    student_entropy_all = []  # list of (T,) on CPU

    with torch.no_grad():
        for i, tokens in enumerate(windows_tokens):
            inp = torch.tensor([tokens], device=DEVICE)
            out = model(inp, return_hidden=True)
            logits = out["logits"][0].float().cpu()  # (T, V)
            hidden = out["hidden"][0].float().cpu()   # (T, D)
            # Compute per-token entropy
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # (T,)
            student_logits_all.append(logits)
            student_hidden_all.append(hidden)
            student_entropy_all.append(entropy)
            if (i + 1) % 10 == 0:
                print(f"    Student: {i+1}/{n_windows} windows")

    # ---- Process each teacher one at a time ----
    # Per-teacher results: teacher_spans[t][w] = (n_spans, D_t) hidden repr per span
    # teacher_logit_metrics[t] = dict of per-span metrics
    teacher_results = {}

    for ti, (t_name, t_short) in enumerate(zip(TEACHERS, TEACHER_SHORT)):
        print(f"\n  Loading teacher {ti+1}/4: {t_name}...")
        teacher = TeacherAdapter(t_name, device=DEVICE, dtype=torch.float16)
        is_generative = not teacher.is_embedding and not teacher.is_encoder

        # Compute shared vocab (for generative teachers)
        shared_s_ids, shared_t_ids, n_shared = None, None, 0
        if is_generative:
            shared_s_ids, shared_t_ids, n_shared = compute_vocab_overlap(
                student_tokenizer, teacher.tokenizer, device="cpu"
            )
            print(f"    Shared vocab: {n_shared} tokens")

        # Per-window metrics
        span_compat = []     # cosine compatibility per span
        span_conf = []       # teacher confidence per span
        span_gap = []        # KL gap per span
        span_t_hidden = []   # teacher span hidden states for novelty computation

        with torch.no_grad():
            for wi in range(n_windows):
                text = windows_text[wi]
                s_hidden = student_hidden_all[wi]  # (T, D_s) on CPU

                # Teacher forward
                t_out = teacher.forward([text], max_length=seq_len)
                t_hidden = t_out["hidden"]  # (1, T_t, D_t) on GPU
                t_logits = t_out.get("logits", None)  # (1, T_t, V_t) on GPU or None
                t_mask = t_out["attention_mask"]  # (1, T_t)
                t_offsets = t_out.get("byte_offsets", None)

                # ---- Representation compatibility via byte_span_pool ----
                # Student: create fake batch dim and offsets
                s_h = s_hidden.unsqueeze(0).to(DEVICE)  # (1, T_s, D_s)
                s_mask = torch.ones(1, s_hidden.shape[0], device=DEVICE)

                # Pool both into spans (using position-based fallback since student has no byte offsets)
                s_spans = byte_span_pool(s_h, None, s_mask, n_spans)  # (1, M, D_s)
                t_spans = byte_span_pool(t_hidden, t_offsets, t_mask, n_spans)  # (1, M, D_t)

                # Store teacher spans for novelty computation
                span_t_hidden.append(t_spans[0].cpu())  # (M, D_t)

                # Compatibility: cosine per span (project to same dim via mean-normalized comparison)
                # Use CKA-style: compute per-span cosine after L2 norm (dimension-agnostic)
                s_norm = F.normalize(s_spans[0].cpu().float(), dim=-1)  # (M, D_s)
                t_norm = F.normalize(t_spans[0].cpu().float(), dim=-1)  # (M, D_t)
                # Since dims differ, use Gram matrix similarity per span
                # For each span m: compat = how similar the representation "fingerprint" is
                # Use min-dim projection: truncate to smaller dim
                min_d = min(s_norm.shape[-1], t_norm.shape[-1])
                compat_per_span = F.cosine_similarity(
                    s_norm[:, :min_d], t_norm[:, :min_d], dim=-1
                )  # (M,)
                span_compat.append(compat_per_span.numpy())

                # ---- Logit-level metrics (generative teachers only) ----
                if is_generative and t_logits is not None and n_shared > 0:
                    t_log = t_logits[0].float().cpu()  # (T_t, V_t)
                    s_log = student_logits_all[wi]      # (T_s, V_s)
                    s_ent = student_entropy_all[wi]      # (T_s,)

                    # Align: use simple position-based (proportional mapping)
                    T_s = s_log.shape[0]
                    T_t = t_log.shape[0]

                    # Teacher confidence: max prob per position
                    t_probs_full = F.softmax(t_log, dim=-1)  # (T_t, V_t)
                    t_max_prob = t_probs_full.max(dim=-1).values  # (T_t,)

                    # KL on shared vocab: student vs teacher
                    t_shared = t_log[:, shared_t_ids.long()]  # (T_t, N_shared)
                    s_shared = s_log[:, shared_s_ids.long()]  # (T_s, N_shared)

                    # Map teacher positions to student positions (proportional)
                    T_min = min(T_s, T_t)
                    t_probs_sh = F.softmax(t_shared[:T_min] / 2.0, dim=-1)
                    s_lprobs_sh = F.log_softmax(s_shared[:T_min] / 2.0, dim=-1)
                    per_tok_kl = F.kl_div(s_lprobs_sh, t_probs_sh, reduction="none").sum(-1)  # (T_min,)

                    # Aggregate per span
                    tokens_per_span = max(1, T_min // n_spans)
                    conf_per_span = []
                    gap_per_span = []
                    for m in range(n_spans):
                        start = m * tokens_per_span
                        end = min((m + 1) * tokens_per_span, T_min)
                        if start >= end:
                            conf_per_span.append(0.0)
                            gap_per_span.append(0.0)
                            continue
                        # Confidence: mean max prob on hardest 25% student-entropy tokens
                        span_s_ent = s_ent[start:end]
                        n_hard = max(1, len(span_s_ent) // 4)
                        hard_idx = span_s_ent.topk(n_hard).indices
                        # Map hard_idx to teacher positions
                        t_start = int(start * T_t / T_min)
                        t_hard_idx = (hard_idx.float() * T_t / T_min).long().clamp(0, T_t - 1)
                        conf_per_span.append(t_max_prob[t_hard_idx].mean().item())
                        # Gap: mean KL on this span
                        gap_per_span.append(per_tok_kl[start:end].mean().item())

                    span_conf.append(np.array(conf_per_span))
                    span_gap.append(np.array(gap_per_span))
                else:
                    span_conf.append(np.zeros(n_spans))
                    span_gap.append(np.zeros(n_spans))

                if (wi + 1) % 10 == 0:
                    print(f"    {t_short}: {wi+1}/{n_windows} windows")

        # Cleanup teacher
        del teacher.model
        del teacher
        torch.cuda.empty_cache()

        teacher_results[t_short] = {
            "name": t_name,
            "is_generative": is_generative,
            "n_shared_vocab": n_shared,
            "compat": np.stack(span_compat),      # (n_windows, M)
            "conf": np.stack(span_conf),           # (n_windows, M)
            "gap": np.stack(span_gap),             # (n_windows, M)
            "span_hidden": span_t_hidden,          # list of (M, D_t) tensors
        }
        print(f"    {t_short} done: mean_compat={np.stack(span_compat).mean():.4f}, "
              f"mean_conf={np.stack(span_conf).mean():.4f}, mean_gap={np.stack(span_gap).mean():.4f}")

    # ---- Cross-teacher novelty ----
    print("\n  Computing cross-teacher novelty...")
    generative_teachers = [t for t in TEACHER_SHORT if teacher_results[t]["is_generative"]]
    for t_short in TEACHER_SHORT:
        others = [o for o in TEACHER_SHORT if o != t_short]
        # Novelty: mean cosine distance to other teachers' span representations
        novelty_all = []
        for wi in range(n_windows):
            t_h = teacher_results[t_short]["span_hidden"][wi]  # (M, D_t)
            dists = []
            for o_short in others:
                o_h = teacher_results[o_short]["span_hidden"][wi]  # (M, D_o)
                min_d = min(t_h.shape[-1], o_h.shape[-1])
                cos = F.cosine_similarity(
                    t_h[:, :min_d].float(), o_h[:, :min_d].float(), dim=-1
                )
                dists.append(1.0 - cos.numpy())  # distance = 1 - cosine
            novelty_all.append(np.stack(dists).mean(axis=0))  # (M,)
        teacher_results[t_short]["novelty"] = np.stack(novelty_all)  # (n_windows, M)

    # ---- Compute routing scores (R1 formula) ----
    print("  Computing routing scores...")
    pi = {"q06": 0.5, "lfm": 0.2, "q17": 0.15, "emb": 0.15}  # prior shares
    all_scores = {t: np.zeros((n_windows, n_spans)) for t in TEACHER_SHORT}

    for t_short in TEACHER_SHORT:
        r = teacher_results[t_short]
        for wi in range(n_windows):
            for m in range(n_spans):
                compat = max(0.01, r["compat"][wi, m])
                conf = max(0.01, r["conf"][wi, m])
                gap = max(0.01, r["gap"][wi, m])
                novel = max(0.01, r["novelty"][wi, m])
                score = pi[t_short] * compat * conf * (gap ** 0.75) * ((0.25 + novel) ** 0.5)
                all_scores[t_short][wi, m] = score

    # Normalize scores to get route shares
    route_shares = {t: 0.0 for t in TEACHER_SHORT}
    route_winner_counts = {t: 0 for t in TEACHER_SHORT}
    total_spans = n_windows * n_spans
    disagreement_spans = 0

    for wi in range(n_windows):
        for m in range(n_spans):
            scores = {t: all_scores[t][wi, m] for t in TEACHER_SHORT}
            winner = max(scores, key=scores.get)
            route_winner_counts[winner] += 1
            # Check disagreement (margin < 20% of top score)
            sorted_scores = sorted(scores.values(), reverse=True)
            if len(sorted_scores) > 1 and sorted_scores[0] > 0:
                margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
                if margin < 0.2:
                    disagreement_spans += 1

    for t in TEACHER_SHORT:
        route_shares[t] = route_winner_counts[t] / total_spans

    # ---- Student difficulty profile ----
    difficulty_per_span = np.zeros((n_windows, n_spans))
    for wi in range(n_windows):
        ent = student_entropy_all[wi].numpy()
        T = len(ent)
        tps = max(1, T // n_spans)
        for m in range(n_spans):
            start = m * tps
            end = min((m + 1) * tps, T)
            if start < end:
                difficulty_per_span[wi, m] = ent[start:end].mean()

    # ---- Compile results ----
    results = {
        "probe": "committee_map_audit",
        "student_checkpoint": str(student_ckpt),
        "student_step": step,
        "n_windows": n_windows,
        "n_spans": n_spans,
        "seq_len": seq_len,
        "route_shares": route_shares,
        "route_winner_counts": route_winner_counts,
        "disagreement_fraction": disagreement_spans / total_spans,
        "student_mean_entropy": float(np.mean([e.mean().item() for e in student_entropy_all])),
        "student_difficulty_by_span": difficulty_per_span.mean(axis=0).tolist(),
        "teachers": {},
    }
    for t_short in TEACHER_SHORT:
        r = teacher_results[t_short]
        results["teachers"][t_short] = {
            "name": r["name"],
            "is_generative": r["is_generative"],
            "n_shared_vocab": r["n_shared_vocab"],
            "mean_compatibility": float(r["compat"].mean()),
            "mean_confidence": float(r["conf"].mean()),
            "mean_gap": float(r["gap"].mean()),
            "mean_novelty": float(r["novelty"].mean()),
            "compat_by_span": r["compat"].mean(axis=0).tolist(),
            "conf_by_span": r["conf"].mean(axis=0).tolist(),
            "gap_by_span": r["gap"].mean(axis=0).tolist(),
            "novelty_by_span": r["novelty"].mean(axis=0).tolist(),
            "route_share": route_shares[t_short],
        }

    # Clean up span_hidden before saving (not JSON serializable)
    out_path = REPO / "results" / "committee_map_audit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"COMMITTEE MAP AUDIT COMPLETE")
    print(f"  Route shares: {route_shares}")
    print(f"  Disagreement: {results['disagreement_fraction']:.1%} of spans")
    print(f"  Student mean entropy: {results['student_mean_entropy']:.3f}")
    for t in TEACHER_SHORT:
        tr = results["teachers"][t]
        print(f"  {t}: compat={tr['mean_compatibility']:.3f} conf={tr['mean_confidence']:.3f} "
              f"gap={tr['mean_gap']:.3f} novelty={tr['mean_novelty']:.3f}")
    print(f"  Results: {out_path}")
    print(f"{'='*60}")

    del model
    torch.cuda.empty_cache()
    return results


def run_ab_probe(config_path):
    """Run a clean A/B probe from a JSON config file.

    Config format:
    {
        "name": "dyt_vs_rmsnorm",
        "variants": [
            {"tag": "rmsnorm", "norm_type": "rmsnorm"},
            {"tag": "dyt", "norm_type": "dyt"}
        ],
        "dim": 512, "n_layers": 10, "n_heads": 8, "ff_dim": 1536,
        "steps": 5000, "eval_every": 500, "batch_size": 16, "grad_accum": 2,
        "lr": 3e-4, "min_lr": 1e-5, "warmup": 200,
        "use_top": false, "top_weight": 0.05, "top_K": 4
    }
    """
    with open(config_path) as f:
        cfg = json.load(f)

    probe_name = cfg["name"]
    dim = cfg.get("dim", 512)
    n_layers = cfg.get("n_layers", 10)
    n_heads = cfg.get("n_heads", 8)
    ff_dim = cfg.get("ff_dim", 1536)
    steps = cfg.get("steps", 5000)
    eval_every = cfg.get("eval_every", 500)
    batch_size = cfg.get("batch_size", 16)
    grad_accum = cfg.get("grad_accum", 2)
    lr = cfg.get("lr", 3e-4)
    min_lr = cfg.get("min_lr", 1e-5)
    warmup = cfg.get("warmup", 200)
    use_top = cfg.get("use_top", False)
    top_weight = cfg.get("top_weight", 0.05)
    top_K = cfg.get("top_K", 4)
    seq_len = cfg.get("seq_len", 512)
    exit_layers = cfg.get("exit_layers", None)

    shard_dir = REPO / "data" / "shards_16k"
    default_weight_file = cfg.get("weight_file", None)

    # Build or find eval cache (unweighted — eval is always on raw test set)
    cache_path = REPO / "results" / "eval_cache_16k.pt"
    if not cache_path.exists():
        print("Building eval cache...")
        _cache_ds = ShardedDataset(shard_dir=str(shard_dir))
        test_tokens = _cache_ds.get_test_tokens()
        windows = []
        wlen = seq_len + 1
        for start in range(0, len(test_tokens) - wlen, wlen):
            windows.append(test_tokens[start:start + wlen])
        torch.save({"windows": windows}, cache_path)
        print(f"  Saved {len(windows)} windows to {cache_path}")

    all_results = {}
    for variant in cfg["variants"]:
        tag = variant["tag"]
        # Per-variant weight file (for O4 data probes), falls back to config-level default
        v_weight_file = variant.get("weight_file", default_weight_file)
        dataset = ShardedDataset(shard_dir=str(shard_dir), weight_file=v_weight_file)
        norm_type = variant.get("norm_type", cfg.get("norm_type", "rmsnorm"))
        # Per-variant overrides for architecture params
        v_n_layers = variant.get("n_layers", n_layers)
        v_dim = variant.get("dim", dim)
        v_n_heads = variant.get("n_heads", n_heads)
        v_ff_dim = variant.get("ff_dim", ff_dim)
        v_use_top = variant.get("use_top", use_top)
        v_top_weight = variant.get("top_weight", top_weight)
        v_exit_layers = variant.get("exit_layers", exit_layers)
        v_block_schedule = variant.get("block_schedule", cfg.get("block_schedule", None))
        v_conv_kernel = variant.get("conv_kernel_size", cfg.get("conv_kernel_size", 64))

        # Auto-infer n_layers from block_schedule if provided
        if v_block_schedule:
            v_n_layers = len(v_block_schedule)

        run_name = f"probe_{probe_name}_{tag}"
        print(f"\n{'='*60}")
        print(f"PROBE: {run_name}")
        print(f"  norm={norm_type}, top={'ON' if v_use_top else 'OFF'}")
        if v_exit_layers:
            print(f"  exits={v_exit_layers}")
        if v_block_schedule:
            n_attn = sum(1 for b in v_block_schedule if b == "A")
            n_conv = sum(1 for b in v_block_schedule if b == "H")
            n_par = sum(1 for b in v_block_schedule if b == "P")
            n_cat = sum(1 for b in v_block_schedule if b == "C")
            n_nad = sum(1 for b in v_block_schedule if b == "N")
            n_r6f = sum(1 for b in v_block_schedule if b == "F")
            n_r6s = sum(1 for b in v_block_schedule if b == "S")
            n_r7p = sum(1 for b in v_block_schedule if b == "G")
            n_pgqa = sum(1 for b in v_block_schedule if b == "Q")
            parts = []
            if n_attn: parts.append(f"{n_attn}A")
            if n_conv: parts.append(f"{n_conv}H")
            if n_par: parts.append(f"{n_par}P")
            if n_cat: parts.append(f"{n_cat}C")
            if n_nad: parts.append(f"{n_nad}N")
            if n_r6f: parts.append(f"{n_r6f}F")
            if n_r6s: parts.append(f"{n_r6s}S")
            if n_r7p: parts.append(f"{n_r7p}G")
            if n_pgqa: parts.append(f"{n_pgqa}Q")
            print(f"  blocks={'+'.join(parts)} (kernel={v_conv_kernel})")
        print(f"  {v_n_layers}L/d={v_dim}/{v_n_heads}h/ff={v_ff_dim}")
        print(f"  steps={steps}, BS={batch_size}x{grad_accum}, LR={lr}")
        print(f"{'='*60}")

        # Extra params for "C" (concat-project hybrid) blocks
        v_d_attn = variant.get("d_attn", cfg.get("d_attn", None))
        v_d_conv = variant.get("d_conv", cfg.get("d_conv", None))
        v_n_q_heads = variant.get("n_q_heads", cfg.get("n_q_heads", None))
        v_n_kv_heads = variant.get("n_kv_heads", cfg.get("n_kv_heads", None))
        v_head_dim = variant.get("head_dim", cfg.get("head_dim", 64))

        model = DenseTransformer(
            vocab_size=VOCAB_SIZE, dim=v_dim, n_layers=v_n_layers,
            n_heads=v_n_heads, ff_dim=v_ff_dim, exit_layers=v_exit_layers,
            norm_type=norm_type, block_schedule=v_block_schedule,
            conv_kernel_size=v_conv_kernel,
            d_attn=v_d_attn, d_conv=v_d_conv,
            n_q_heads=v_n_q_heads, n_kv_heads=v_n_kv_heads,
            head_dim=v_head_dim,
        ).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {n_params:,}")

        v_optimizer = variant.get("optimizer", cfg.get("optimizer", "adamw"))
        muon_lr = variant.get("muon_lr", cfg.get("muon_lr", 0.02))
        normuon_lr = variant.get("normuon_lr", cfg.get("normuon_lr", 0.01))
        use_muon = v_optimizer == "muon"
        use_normuon = v_optimizer == "normuon"

        if use_muon or use_normuon:
            # Split: 2D hidden weights -> Muon/NorMuon, everything else -> AdamW
            special_params = []
            adam_params = []
            for name, p in model.named_parameters():
                if p.ndim == 2 and "emb" not in name:
                    special_params.append(p)
                else:
                    adam_params.append(p)
            if use_normuon:
                opt = NorMuon(special_params, lr=normuon_lr, beta1=0.95, beta2=0.95,
                              weight_decay=0.1)
                opt_label = f"NorMuon (lr={normuon_lr})"
            else:
                from torch.optim import Muon as TorchMuon
                opt = TorchMuon(special_params, lr=muon_lr, momentum=0.95,
                                weight_decay=0.1)
                opt_label = f"Muon (lr={muon_lr})"
            opt_adam = torch.optim.AdamW(adam_params, lr=lr, betas=(0.9, 0.95),
                                        weight_decay=0.1)
            print(f"  Optimizer: {opt_label} + AdamW (lr={lr})")
            print(f"    Special params: {sum(p.numel() for p in special_params):,}")
            print(f"    AdamW params: {sum(p.numel() for p in adam_params):,}")
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                    weight_decay=0.1)
            opt_adam = None
            print(f"  Optimizer: AdamW (lr={lr})")
        scaler = torch.amp.GradScaler("cuda")

        log_file = REPO / "results" / f"{run_name}_log.txt"
        metrics_log = []
        running_loss = 0.0
        running_top = 0.0
        model.train()

        for step in range(1, steps + 1):
            current_lr = get_lr_wsd(step, warmup, steps, lr, min_lr)
            for pg in opt.param_groups:
                if use_muon:
                    pg["lr"] = muon_lr * (current_lr / lr)
                elif use_normuon:
                    pg["lr"] = normuon_lr * (current_lr / lr)
                else:
                    pg["lr"] = current_lr
            if opt_adam is not None:
                for pg in opt_adam.param_groups:
                    pg["lr"] = current_lr

            # Micro-batch accumulation
            opt.zero_grad()
            if opt_adam is not None:
                opt_adam.zero_grad()
            for _ in range(grad_accum):
                x, y = dataset.sample_batch(batch_size, seq_len, device=DEVICE)
                with torch.amp.autocast("cuda", dtype=DTYPE):
                    if v_exit_layers:
                        out = model(x, return_exits=True)
                        logits = out['logits']
                        loss, _ = compute_edsr_loss(out, y)
                    else:
                        logits = model(x)
                        V = logits.size(-1)
                        Tc = min(logits.size(1), y.size(1))
                        loss = F.cross_entropy(
                            logits[:, :Tc].reshape(-1, V), y[:, :Tc].reshape(-1))

                    # TOP auxiliary loss
                    top_loss = torch.tensor(0.0, device=DEVICE)
                    if v_use_top and step > warmup:
                        top_loss = compute_top_loss(logits, y, K=top_K)
                        loss = loss + v_top_weight * top_loss

                    loss_scaled = loss / grad_accum
                scaler.scale(loss_scaled).backward()
                running_loss += loss.item() / grad_accum
                running_top += top_loss.item() / grad_accum

            scaler.unscale_(opt)
            if opt_adam is not None:
                scaler.unscale_(opt_adam)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            if opt_adam is not None:
                scaler.step(opt_adam)
            scaler.update()

            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                avg_top = running_top / LOG_EVERY
                bpt = avg_loss / math.log(2)
                msg = f"  Step {step:5d}: CE={avg_loss:.4f} BPT={bpt:.4f}"
                if v_use_top:
                    msg += f" TOP={avg_top:.4f}"
                msg += f" LR={current_lr:.2e}"
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_loss = 0.0
                running_top = 0.0

            if step % eval_every == 0 or step == steps:
                model.eval()
                # Deterministic eval
                det = deterministic_eval_dense(model, cache_path)
                # Kurtosis measurement
                kurt = measure_activation_kurtosis(model, dataset, DEVICE)
                entry = {
                    "step": step, "tag": tag, "norm_type": norm_type,
                    "bpt": det["bpt"],
                    "avg_kurtosis": round(sum(kurt["kurtosis"].values()) / len(kurt["kurtosis"]), 2),
                    "max_kurtosis": round(max(kurt["kurtosis"].values()), 2),
                    "max_activation": round(max(kurt["max_activation"].values()), 2),
                }
                if exit_layers:
                    for k, v in det.items():
                        if k.startswith("bpt_exit"):
                            entry[k] = v
                # R6 branch telemetry
                if "branch_telemetry" in kurt:
                    entry["branch_telemetry"] = kurt["branch_telemetry"]
                metrics_log.append(entry)
                msg = (f"  EVAL step {step}: BPT={det['bpt']:.4f} "
                       f"kurtosis_avg={entry['avg_kurtosis']:.1f} "
                       f"kurtosis_max={entry['max_kurtosis']:.1f} "
                       f"max_act={entry['max_activation']:.1f}")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                model.train()
                torch.cuda.empty_cache()

        # Final generation quality check
        model.eval()
        from tokenizers import Tokenizer
        tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
        if tok_path.exists():
            tokenizer = Tokenizer.from_file(str(tok_path))
            prompts = ["The meaning of life is", "In a distant future",
                       "def fibonacci(n):\n"]
            generations = []
            for prompt_text in prompts:
                input_ids = tokenizer.encode(prompt_text).ids
                inp = torch.tensor([input_ids], device=DEVICE)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                    for _ in range(64):
                        logits = model(inp)
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        inp = torch.cat([inp, next_token], dim=1)
                        if inp.size(1) > seq_len:
                            break
                gen_text = tokenizer.decode(inp[0].tolist())
                generations.append({"prompt": prompt_text, "generation": gen_text})
        else:
            generations = []
        model.train()

        all_results[tag] = {
            "params": n_params,
            "norm_type": norm_type,
            "use_top": v_use_top,
            "metrics": metrics_log,
            "generations": generations,
        }

        # Save final checkpoint for continuation (8-10K runs)
        ckpt_save_dir = REPO / "results" / f"checkpoints_probe_{probe_name}"
        ckpt_save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_save_dir / f"{tag}_step{steps}.pt"
        torch.save({
            "step": steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "config": {
                "dim": v_dim, "n_layers": v_n_layers, "n_heads": v_n_heads,
                "ff_dim": v_ff_dim, "norm_type": norm_type,
                "block_schedule": v_block_schedule,
                "exit_layers": v_exit_layers,
                "d_attn": v_d_attn, "d_conv": v_d_conv,
                "n_q_heads": v_n_q_heads, "n_kv_heads": v_n_kv_heads,
                "head_dim": v_head_dim, "conv_kernel_size": v_conv_kernel,
            },
            "metrics": metrics_log,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

        # Free GPU memory before next variant
        del model, opt, scaler
        if opt_adam is not None:
            del opt_adam
        torch.cuda.empty_cache()
        gc.collect()

    # Save combined results
    output = {
        "probe": probe_name,
        "config": cfg,
        "results": all_results,
    }
    out_path = REPO / "results" / f"probe_{probe_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*60}")
    print(f"PROBE COMPLETE: {probe_name}")
    print(f"Results: {out_path}")

    # Summary comparison
    for tag, res in all_results.items():
        final = res["metrics"][-1] if res["metrics"] else {}
        print(f"  {tag}: BPT={final.get('bpt', '?')}, "
              f"kurtosis_avg={final.get('avg_kurtosis', '?')}, "
              f"max_act={final.get('max_activation', '?')}")
    print(f"{'='*60}")


def train_kd(config_path, variant_filter=None):
    """Knowledge Distillation training with RMFD system.

    Loads student from warm-start checkpoint, loads teacher models,
    and trains with representation-level KD losses (state CKA + semantic relational).
    Supports multi-arm probes with different teacher configurations.

    Config JSON format:
    {
        "name": "kd_probe_01",
        "init_from": "path/to/student_checkpoint.pt",
        "total_steps": 3000,
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 3e-4,
        "min_lr": 1e-5,
        "warmup": 200,
        "eval_every": 500,
        "seq_len": 512,
        "n_spans": 16,
        "variants": [
            {
                "tag": "control",
                "teachers": [],
                "alpha_state": 0.0,
                "alpha_semantic": 0.0
            },
            {
                "tag": "single_anchor",
                "teachers": [
                    {"name": "Qwen/Qwen3-1.7B-Base", "surfaces": ["state", "semantic"]}
                ],
                "alpha_state": 0.5,
                "alpha_semantic": 0.3
            }
        ]
    }
    """
    with open(config_path) as f:
        cfg = json.load(f)

    run_name = cfg["name"]
    init_checkpoint = cfg["init_from"]
    total_steps = cfg.get("total_steps", 3000)
    batch_size = cfg.get("batch_size", 8)
    grad_accum = cfg.get("grad_accum", 4)
    lr = cfg.get("lr", 3e-4)
    min_lr = cfg.get("min_lr", 1e-5)
    warmup = cfg.get("warmup", 200)
    eval_every = cfg.get("eval_every", 500)
    seq_len = cfg.get("seq_len", 512)
    n_spans = cfg.get("n_spans", 16)
    alpha_state_default = cfg.get("alpha_state", 0.5)
    alpha_semantic_default = cfg.get("alpha_semantic", 0.3)
    alpha_logit_default = cfg.get("alpha_logit", 0.0)
    logit_temperature = cfg.get("logit_temperature", 2.0)
    logit_top_k = cfg.get("logit_top_k", 64)
    save_at_steps = set(cfg.get("save_at", []))  # save named checkpoints at these steps

    print(f"\n{'='*60}")
    print(f"RMFD KD TRAINING: {run_name}")
    print(f"  Student init: {init_checkpoint}")
    print(f"  Steps: {total_steps}, BS={batch_size}x{grad_accum}={batch_size*grad_accum}")
    print(f"  LR: {lr} -> {min_lr}, warmup={warmup}")
    print(f"  Byte spans: {n_spans}")
    print(f"{'='*60}")

    # Load student tokenizer for decoding tokens -> raw text (for cross-tokenizer KD)
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    student_tokenizer = Tokenizer.from_file(str(tok_path))

    # Dataset
    shard_dir = REPO / "data" / "shards_16k"
    weight_file = cfg.get("weight_file", None)
    dataset = ShardedDataset(shard_dir=str(shard_dir), weight_file=weight_file)

    # Eval cache
    cache_path = REPO / "results" / "eval_cache_16k.pt"
    if not cache_path.exists():
        print("Building eval cache...")
        test_tokens = dataset.get_test_tokens()
        windows = []
        wlen = seq_len + 1
        for start in range(0, len(test_tokens) - wlen, wlen):
            windows.append(test_tokens[start:start + wlen])
        torch.save({"windows": windows}, cache_path)
        print(f"  Saved {len(windows)} windows to {cache_path}")

    variants = cfg.get("variants", [cfg])
    if variant_filter:
        variants = [v for v in variants if v.get("tag", "") == variant_filter]
        if not variants:
            print(f"ERROR: No variant with tag '{variant_filter}' found in config.")
            print(f"  Available tags: {[v.get('tag', '?') for v in cfg.get('variants', [cfg])]}")
            return
        print(f"  Variant filter: running only '{variant_filter}'")
    all_results = {}
    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for variant in variants:
        tag = variant.get("tag", run_name)
        v_teachers_cfg = variant.get("teachers", [])
        v_alpha_state = variant.get("alpha_state", alpha_state_default)
        v_alpha_semantic = variant.get("alpha_semantic", alpha_semantic_default)
        v_alpha_logit = variant.get("alpha_logit", alpha_logit_default)
        v_alpha_schedule = variant.get("alpha_schedule", None)
        v_confidence_gating = variant.get("confidence_gating", False)
        v_tau_schedule = variant.get("tau_schedule", None)

        print(f"\n{'='*60}")
        print(f"KD VARIANT: {tag}")
        print(f"  Teachers: {[t['name'] for t in v_teachers_cfg] if v_teachers_cfg else 'NONE (control)'}")
        print(f"  alpha_state={v_alpha_state}, alpha_semantic={v_alpha_semantic}, alpha_logit={v_alpha_logit}")
        if v_alpha_schedule:
            print(f"  Alpha schedule: warmup={v_alpha_schedule.get('warmup_steps', 2000)}, "
                  f"start_frac={v_alpha_schedule.get('start_frac', 0.2)}, "
                  f"decay_start={v_alpha_schedule.get('decay_start_step', int(total_steps * 0.67))}, "
                  f"zero_start={v_alpha_schedule.get('zero_start_step', total_steps)}")

        # ---- Load student from checkpoint (CPU first to avoid VRAM spike) ----
        init_ckpt = torch.load(init_checkpoint, weights_only=False, map_location="cpu")
        stu_cfg = init_ckpt.get("config", {})
        dim = stu_cfg.get("dim", 512)
        n_layers = stu_cfg.get("n_layers", 24)
        n_heads = stu_cfg.get("n_heads", 8)
        ff_dim = stu_cfg.get("ff_dim", 1536)
        exit_layers = stu_cfg.get("exit_layers", [7, 15, 23])

        model = DenseTransformer(
            vocab_size=VOCAB_SIZE, dim=dim, n_layers=n_layers,
            n_heads=n_heads, ff_dim=ff_dim, exit_layers=exit_layers,
            norm_type=stu_cfg.get("norm_type", "rmsnorm"),
            block_schedule=stu_cfg.get("block_schedule", None),
            conv_kernel_size=stu_cfg.get("conv_kernel_size", 64),
            d_attn=stu_cfg.get("d_attn", None),
            d_conv=stu_cfg.get("d_conv", None),
            n_q_heads=stu_cfg.get("n_q_heads", None),
            n_kv_heads=stu_cfg.get("n_kv_heads", None),
            head_dim=stu_cfg.get("head_dim", 64),
        ).to(DEVICE)

        init_key = "model_state_dict" if "model_state_dict" in init_ckpt else "model"
        model.load_state_dict(init_ckpt[init_key])
        start_step = init_ckpt.get("step", 0)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Student: {n_params:,} params, warm-start from step {start_step}")
        del init_ckpt

        # ---- Load teachers ----
        teachers = []
        projectors = nn.ModuleDict()  # trainable projectors for state/semantic KD
        for t_cfg in v_teachers_cfg:
            teacher = TeacherAdapter(t_cfg["name"], device=DEVICE, dtype=torch.float16)
            teacher.surfaces = t_cfg.get("surfaces", ["state"])
            teachers.append(teacher)

            # Create learned projectors for each KD surface
            if "state" in teacher.surfaces:
                key = t_cfg["name"].replace("/", "_").replace("-", "_").replace(".", "_") + "__state"
                projectors[key] = nn.Linear(dim, teacher.hidden_dim)
            if "semantic" in teacher.surfaces:
                key = t_cfg["name"].replace("/", "_").replace("-", "_").replace(".", "_") + "__semantic"
                projectors[key] = nn.Linear(dim, teacher.hidden_dim)
        projectors = projectors.to(DEVICE)

        # ---- Pre-compute vocabulary overlap for logit KD ----
        teacher_vocab_overlaps = {}
        if v_alpha_logit > 0:
            for teacher in teachers:
                if teacher.is_encoder or teacher.is_embedding:
                    continue  # encoders don't produce causal logits
                if "logit" in teacher.surfaces:
                    s_ids, t_ids, n_shared = compute_vocab_overlap(
                        student_tokenizer, teacher.tokenizer, device=DEVICE)
                    teacher_vocab_overlaps[teacher.name] = (s_ids, t_ids, n_shared)
                    print(f"    Vocab overlap {teacher.name}: {n_shared}/{VOCAB_SIZE} "
                          f"({100*n_shared/VOCAB_SIZE:.1f}%)")

        # ---- Optimizer: student + projectors ----
        param_groups = [
            {"params": list(model.parameters()), "lr": lr},
            {"params": list(projectors.parameters()), "lr": lr * 3, "is_projector": True},
        ]
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
        scaler = torch.amp.GradScaler("cuda")

        # ---- Training state ----
        log_file = REPO / "results" / f"kd_{tag}_log.txt"
        metrics_log = []
        running_ce = 0.0
        running_kd = 0.0
        actual_exit_weights = derive_exit_weights(exit_layers, n_layers) if exit_layers else EXIT_WEIGHTS
        has_kd = len(teachers) > 0
        needs_hidden = any("state" in t.surfaces or "semantic" in t.surfaces
                           for t in teachers)
        use_hidden = needs_hidden  # only compute hidden states if rep KD is active

        print(f"  Exit weights: {actual_exit_weights}")
        if projectors:
            proj_params = sum(p.numel() for p in projectors.parameters())
            print(f"  Projector params: {proj_params:,}")

        # ---- Resume from rolling checkpoint if available ----
        resume_step = 0
        rolling_path = ckpt_dir / f"{tag}_rolling.pt"
        if rolling_path.exists():
            try:
                rckpt = torch.load(rolling_path, weights_only=False, map_location=DEVICE)
                # Verify tag matches to prevent cross-variant contamination
                saved_tag = rckpt.get("kd_config", {}).get("tag", tag)
                if saved_tag != tag:
                    raise ValueError(f"Rolling checkpoint tag '{saved_tag}' != current '{tag}'")
                # Load all state — if any fails, except block re-inits from scratch
                model.load_state_dict(rckpt["model_state_dict"])
                opt.load_state_dict(rckpt["optimizer_state_dict"])
                if "projectors" in rckpt and projectors:
                    projectors.load_state_dict(rckpt["projectors"])
                if "scaler" in rckpt:
                    scaler.load_state_dict(rckpt["scaler"])
                if "metrics" in rckpt:
                    metrics_log = rckpt["metrics"]
                resume_step = rckpt.get("step", 0) - start_step
                print(f"  RESUMED from rolling checkpoint at step {resume_step}")
            except Exception as e:
                print(f"  WARNING: rolling checkpoint failed ({e}), trying named checkpoints...")
                resume_step = 0
                # Fallback: scan for newest named checkpoint (tag_step*.pt)
                named_ckpts = sorted(
                    ckpt_dir.glob(f"{tag}_step*.pt"),
                    key=lambda p: int(p.stem.split("step")[-1]),
                    reverse=True)
                fallback_loaded = False
                for nckpt_path in named_ckpts:
                    try:
                        nckpt = torch.load(nckpt_path, weights_only=False, map_location=DEVICE)
                        model.load_state_dict(nckpt["model_state_dict"])
                        opt.load_state_dict(nckpt["optimizer_state_dict"])
                        if "projectors" in nckpt and projectors:
                            projectors.load_state_dict(nckpt["projectors"])
                        if "scaler" in nckpt:
                            scaler.load_state_dict(nckpt["scaler"])
                        if "metrics" in nckpt:
                            metrics_log = nckpt["metrics"]
                        resume_step = nckpt.get("step", 0) - start_step
                        print(f"  RESUMED from named checkpoint {nckpt_path.name} at step {resume_step}")
                        del nckpt
                        fallback_loaded = True
                        break
                    except Exception as e2:
                        print(f"    Named checkpoint {nckpt_path.name} failed: {e2}")
                        continue
                if not fallback_loaded:
                    print(f"  No valid checkpoint found, re-initializing from init_from")
                    resume_step = 0
                    fresh_ckpt = torch.load(init_checkpoint, weights_only=False, map_location=DEVICE)
                    fresh_key = "model_state_dict" if "model_state_dict" in fresh_ckpt else "model"
                    model.load_state_dict(fresh_ckpt[fresh_key])
                    del fresh_ckpt
                opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
                scaler = torch.amp.GradScaler("cuda")
                if projectors:
                    for p in projectors.parameters():
                        if p.dim() > 1:
                            torch.nn.init.xavier_uniform_(p)
                        else:
                            torch.nn.init.zeros_(p)

        print(f"  Starting training from step {resume_step + 1}...\n")

        model.train()
        for step in range(resume_step + 1, total_steps + 1):
            current_lr = get_lr_wsd(step, warmup, total_steps, lr, min_lr)
            for pg in opt.param_groups:
                if pg.get("is_projector", False):
                    pg["lr"] = current_lr * 3
                else:
                    pg["lr"] = current_lr

            # ---- Alpha schedule (inverted-U or custom) ----
            if v_alpha_schedule:
                ws = v_alpha_schedule.get("warmup_steps", 2000)
                sf = v_alpha_schedule.get("start_frac", 0.2)
                ds = v_alpha_schedule.get("decay_start_step", int(total_steps * 0.67))
                ef = v_alpha_schedule.get("decay_end_frac", 0.0)
                zs = v_alpha_schedule.get("zero_start_step", total_steps)
                tf = v_alpha_schedule.get("tail_frac", 0.0)
                if step <= ws:
                    alpha_frac = sf + (1.0 - sf) * (step / ws)
                elif step <= ds:
                    alpha_frac = 1.0
                elif step <= zs:
                    alpha_frac = 1.0 - (1.0 - ef) * ((step - ds) / max(zs - ds, 1))
                else:
                    alpha_frac = tf
                sa_state = v_alpha_state * alpha_frac
                sa_semantic = v_alpha_semantic * alpha_frac
                sa_logit = v_alpha_logit * alpha_frac
            else:
                sa_state = v_alpha_state
                sa_semantic = v_alpha_semantic
                sa_logit = v_alpha_logit

            # ---- Tau schedule (rising temperature for logit KD) ----
            if v_tau_schedule:
                tau_start = v_tau_schedule.get("start", 1.5)
                tau_end = v_tau_schedule.get("end", 3.0)
                tau_ramp = v_tau_schedule.get("ramp_steps", int(total_steps * 0.67))
                step_tau = tau_start + (tau_end - tau_start) * min(step / tau_ramp, 1.0)
            else:
                step_tau = logit_temperature

            opt.zero_grad()
            step_ce = 0.0
            step_kd = 0.0
            bad_step = False

            for _ in range(grad_accum):
                inp, tgt = dataset.sample_batch(batch_size, seq_len, device=DEVICE)

                with torch.amp.autocast("cuda", dtype=DTYPE):
                    # Student forward (with hidden states if KD active)
                    out = model(inp, return_exits=True, return_hidden=use_hidden)

                    # Base CE + exit supervision loss
                    base_loss, loss_metrics = compute_edsr_loss(
                        out, tgt, exit_weights=actual_exit_weights)
                    ce_val = loss_metrics["ce_final"]

                    # KD losses (representation + logit level)
                    kd_loss = torch.tensor(0.0, device=DEVICE)

                    if has_kd and (sa_state + sa_semantic + sa_logit) > 0:
                        # Decode student tokens -> raw text for teacher tokenizers
                        texts = [student_tokenizer.decode(inp[b].tolist())
                                 for b in range(inp.size(0))]

                        # Compute student byte offsets via re-encoding
                        stu_offsets = torch.zeros(
                            inp.shape[0], inp.shape[1], 2,
                            device=DEVICE, dtype=torch.long)
                        for b_idx, text in enumerate(texts):
                            enc_s = student_tokenizer.encode(text)
                            n_tok = min(len(enc_s.offsets), inp.shape[1])
                            for ti in range(n_tok):
                                stu_offsets[b_idx, ti, 0] = enc_s.offsets[ti][0]
                                stu_offsets[b_idx, ti, 1] = enc_s.offsets[ti][1]

                        # Student hidden state (only needed for state/semantic KD)
                        student_hidden = out.get("hidden", None)  # (B, T_s, D_s) or None
                        student_mask = torch.ones(inp.shape[0], inp.shape[1],
                                                  device=DEVICE, dtype=torch.long)

                        # Count teachers per surface for fair loss normalization
                        n_state_teachers = sum(1 for t in teachers
                                               if "state" in t.surfaces)
                        n_sem_teachers = sum(1 for t in teachers
                                             if "semantic" in t.surfaces)
                        n_logit_teachers = sum(1 for t in teachers
                                               if "logit" in t.surfaces)

                        for teacher in teachers:
                            with torch.no_grad():
                                t_out = teacher.forward(texts, max_length=seq_len)

                            safe_name = (teacher.name.replace("/", "_")
                                         .replace("-", "_")
                                         .replace(".", "_"))

                            # State-level KD: CKA on byte-span-pooled hidden states
                            if "state" in teacher.surfaces:
                                assert t_out["hidden"] is not None, (
                                    f"Teacher {teacher.name} requested state surface "
                                    f"but returned hidden=None")
                                proj = projectors[safe_name + "__state"]
                                s_kd = compute_state_kd_loss(
                                    student_hidden, t_out["hidden"],
                                    student_offsets=stu_offsets,
                                    teacher_offsets=t_out["byte_offsets"],
                                    student_mask=student_mask,
                                    teacher_mask=t_out["attention_mask"],
                                    projector=proj, n_spans=n_spans,
                                )
                                kd_loss = kd_loss + (sa_state / max(n_state_teachers, 1)) * s_kd

                            # Semantic KD: relational Gram matrix loss
                            if "semantic" in teacher.surfaces:
                                assert t_out["hidden"] is not None, (
                                    f"Teacher {teacher.name} requested semantic surface "
                                    f"but returned hidden=None")
                                proj = projectors[safe_name + "__semantic"]
                                sem_kd = compute_semantic_kd_loss(
                                    student_hidden, t_out["hidden"],
                                    student_mask=student_mask,
                                    teacher_mask=t_out["attention_mask"],
                                    projector=proj,
                                )
                                kd_loss = kd_loss + (sa_semantic / max(n_sem_teachers, 1)) * sem_kd

                            # Logit-level KD: cross-tokenizer via shared vocabulary
                            if "logit" in teacher.surfaces and teacher.name in teacher_vocab_overlaps:
                                assert t_out["logits"] is not None, (
                                    f"Teacher {teacher.name} requested logit surface "
                                    f"but returned logits=None (encoder model?)")
                                s_ids, t_ids, _ = teacher_vocab_overlaps[teacher.name]
                                logit_kd = compute_cross_tok_logit_kd(
                                    student_logits=out["logits"],
                                    teacher_logits=t_out["logits"],
                                    student_offsets=stu_offsets,
                                    teacher_offsets=t_out.get("byte_offsets", None),
                                    teacher_mask=t_out["attention_mask"],
                                    shared_s_ids=s_ids,
                                    shared_t_ids=t_ids,
                                    temperature=step_tau,
                                    top_k=logit_top_k,
                                    confidence_gating=v_confidence_gating,
                                )
                                kd_loss = kd_loss + (sa_logit / max(n_logit_teachers, 1)) * logit_kd

                    total_loss = (base_loss + kd_loss) / grad_accum

                if not torch.isfinite(total_loss):
                    print(f"  WARNING: NaN/Inf at step {step}, skipping")
                    opt.zero_grad()
                    bad_step = True
                    break

                scaler.scale(total_loss).backward()
                step_ce += ce_val / grad_accum
                step_kd += kd_loss.item() / grad_accum

            if bad_step:
                continue

            scaler.unscale_(opt)
            all_params = list(model.parameters()) + list(projectors.parameters())
            torch.nn.utils.clip_grad_norm_(
                [p for p in all_params if p.grad is not None], GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

            running_ce += step_ce
            running_kd += step_kd

            # ---- Logging ----
            if step % LOG_EVERY == 0:
                avg_ce = running_ce / LOG_EVERY
                avg_kd = running_kd / LOG_EVERY
                bpt = avg_ce / math.log(2)
                msg = (f"  Step {step:5d}: CE={avg_ce:.4f} BPT={bpt:.4f} "
                       f"KD={avg_kd:.4f} LR={current_lr:.2e}")
                if v_alpha_schedule:
                    msg += f" aF={alpha_frac:.2f}"
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_ce = 0.0
                running_kd = 0.0

            # ---- Eval ----
            if step % eval_every == 0 or step == total_steps:
                model.eval()
                det = deterministic_eval_dense(model, cache_path)
                kurt = measure_activation_kurtosis(model, dataset, DEVICE)
                entry = {
                    "step": step, "tag": tag,
                    "bpt": det["bpt"],
                    "avg_kurtosis": round(
                        sum(kurt["kurtosis"].values()) / len(kurt["kurtosis"]), 2),
                    "max_kurtosis": round(max(kurt["kurtosis"].values()), 2),
                    "max_activation": round(max(kurt["max_activation"].values()), 2),
                    "teachers": [t.name for t in teachers],
                }
                for k, v in det.items():
                    if k.startswith("bpt_exit"):
                        entry[k] = v
                metrics_log.append(entry)
                msg = (f"  EVAL step {step}: BPT={det['bpt']:.4f} "
                       f"kurtosis_max={entry['max_kurtosis']:.1f} "
                       f"max_act={entry['max_activation']:.1f}")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                model.train()
                torch.cuda.empty_cache()

            # ---- Rolling checkpoint (includes full resume state) ----
            if step % ROLLING_SAVE == 0:
                ckpt = {
                    "step": start_step + step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "projectors": projectors.state_dict(),
                    "config": stu_cfg,
                    "kd_config": variant,
                    "metrics": metrics_log,
                }
                rolling = ckpt_dir / f"{tag}_rolling.pt"
                tmp = rolling.with_suffix(".tmp")
                torch.save(ckpt, tmp)
                os.replace(str(tmp), str(rolling))

            # ---- Named checkpoint at specific steps (full state for resume) ----
            if step in save_at_steps:
                ckpt = {
                    "step": start_step + step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "projectors": projectors.state_dict(),
                    "config": stu_cfg,
                    "kd_config": variant,
                    "metrics": metrics_log,
                }
                save_path = ckpt_dir / f"{tag}_step{step}.pt"
                torch.save(ckpt, save_path)
                print(f"  Saved named checkpoint: {save_path}")

        # ---- Final generation check ----
        model.eval()
        generations = []
        tok_path_gen = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
        if tok_path_gen.exists():
            prompts = ["The meaning of life is", "In a distant future",
                       "def fibonacci(n):\n"]
            for prompt_text in prompts:
                input_ids = student_tokenizer.encode(prompt_text).ids
                gen_inp = torch.tensor([input_ids], device=DEVICE)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                    for _ in range(64):
                        logits = model(gen_inp)
                        if isinstance(logits, dict):
                            logits = logits["logits"]
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen_inp = torch.cat([gen_inp, next_token], dim=1)
                        if gen_inp.size(1) > seq_len:
                            break
                gen_text = student_tokenizer.decode(gen_inp[0].tolist())
                generations.append({"prompt": prompt_text, "generation": gen_text})
        model.train()

        # ---- Save variant results ----
        all_results[tag] = {
            "params": n_params,
            "teachers": [t.name for t in teachers],
            "alpha_state": v_alpha_state,
            "alpha_semantic": v_alpha_semantic,
            "metrics": metrics_log,
            "generations": generations,
        }

        # Final checkpoint (full resume state)
        torch.save({
            "step": start_step + total_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "projectors": projectors.state_dict(),
            "config": stu_cfg,
            "kd_config": variant,
            "metrics": metrics_log,
        }, ckpt_dir / f"{tag}_step{total_steps}.pt")
        print(f"  Saved checkpoint: {ckpt_dir / f'{tag}_step{total_steps}.pt'}")

        # ---- Cleanup GPU ----
        del model, opt, scaler
        for t in teachers:
            del t.model
        del teachers, projectors
        torch.cuda.empty_cache()
        gc.collect()

    # ---- Save combined results ----
    output = {"probe": run_name, "config": cfg, "results": all_results}
    out_path = REPO / "results" / f"kd_{run_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"KD PROBE COMPLETE: {run_name}")
    print(f"Results: {out_path}")
    for tag, res in all_results.items():
        final = res["metrics"][-1] if res["metrics"] else {}
        print(f"  {tag}: BPT={final.get('bpt', '?')}, teachers={res['teachers']}")
    print(f"{'='*60}")


def train_kd_phased(config_path):
    """Phase-aware RMFD training for production runs.

    Unlike train_kd (which runs multiple independent variants), this function
    runs a SINGLE continuous training loop with phase transitions.
    Optimizer state is preserved across phases. Teachers are hot-swapped at
    phase boundaries.

    Config JSON format:
    {
        "name": "rmfd_production",
        "init_from": "path/to/checkpoint.pt",
        "resume_from": null,  // or path to mid-training checkpoint
        "total_steps": 30000,
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "warmup": 500,
        "eval_every": 1000,
        "seq_len": 512,
        "n_spans": 16,
        "phases": [
            {
                "name": "stabilization",
                "end_step": 500,
                "teachers": [],
                "alpha_state": 0.0,
                "alpha_semantic": 0.0
            },
            {
                "name": "single_anchor",
                "end_step": 5000,
                "teachers": [
                    {"name": "Qwen/Qwen3-1.7B-Base", "surfaces": ["state", "semantic"]}
                ],
                "alpha_state": 0.5,
                "alpha_semantic": 0.3
            }
        ]
    }
    """
    with open(config_path) as f:
        cfg = json.load(f)

    run_name = cfg["name"]
    init_checkpoint = cfg["init_from"]
    total_steps = cfg.get("total_steps", 30000)
    batch_size = cfg.get("batch_size", 8)
    grad_accum = cfg.get("grad_accum", 4)
    lr = cfg.get("lr", 1e-4)
    min_lr = cfg.get("min_lr", 1e-5)
    warmup = cfg.get("warmup", 500)
    eval_every = cfg.get("eval_every", 1000)
    seq_len = cfg.get("seq_len", 512)
    n_spans = cfg.get("n_spans", 16)
    phases = cfg["phases"]

    print(f"\n{'='*60}")
    print(f"RMFD PHASED TRAINING: {run_name}")
    print(f"  Student init: {init_checkpoint}")
    print(f"  Steps: {total_steps}, BS={batch_size}x{grad_accum}={batch_size*grad_accum}")
    print(f"  LR: {lr} -> {min_lr}, warmup={warmup}")
    print(f"  Phases: {len(phases)}")
    for p in phases:
        print(f"    {p['name']}: end_step={p['end_step']}, "
              f"teachers={[t['name'] for t in p.get('teachers', [])] or 'NONE'}")
    print(f"{'='*60}")

    # Load student tokenizer
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    student_tokenizer = Tokenizer.from_file(str(tok_path))

    # Dataset
    shard_dir = REPO / "data" / "shards_16k"
    weight_file = cfg.get("weight_file", None)
    dataset = ShardedDataset(shard_dir=str(shard_dir), weight_file=weight_file)

    # Eval cache
    cache_path = REPO / "results" / "eval_cache_16k.pt"
    if not cache_path.exists():
        print("Building eval cache...")
        test_tokens = dataset.get_test_tokens()
        windows = []
        wlen = seq_len + 1
        for start in range(0, len(test_tokens) - wlen, wlen):
            windows.append(test_tokens[start:start + wlen])
        torch.save({"windows": windows}, cache_path)

    # ---- Load student model (CPU first to avoid VRAM spike) ----
    resume_path = cfg.get("resume_from", None)
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, weights_only=False, map_location="cpu")
        resume_step = ckpt.get("step", 0)
        print(f"  Resuming from step {resume_step}: {resume_path}")
    else:
        ckpt = torch.load(init_checkpoint, weights_only=False, map_location="cpu")
        resume_step = 0

    stu_cfg = ckpt.get("config", {})
    dim = stu_cfg.get("dim", 512)
    n_layers = stu_cfg.get("n_layers", 24)
    n_heads = stu_cfg.get("n_heads", 8)
    ff_dim = stu_cfg.get("ff_dim", 1536)
    exit_layers = stu_cfg.get("exit_layers", [7, 15, 23])

    model = DenseTransformer(
        vocab_size=VOCAB_SIZE, dim=dim, n_layers=n_layers,
        n_heads=n_heads, ff_dim=ff_dim, exit_layers=exit_layers,
        norm_type=stu_cfg.get("norm_type", "rmsnorm"),
        block_schedule=stu_cfg.get("block_schedule", None),
        conv_kernel_size=stu_cfg.get("conv_kernel_size", 64),
        d_attn=stu_cfg.get("d_attn", None),
        d_conv=stu_cfg.get("d_conv", None),
        n_q_heads=stu_cfg.get("n_q_heads", None),
        n_kv_heads=stu_cfg.get("n_kv_heads", None),
        head_dim=stu_cfg.get("head_dim", 64),
    ).to(DEVICE)

    init_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[init_key])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Student: {n_params:,} params")

    # ---- Collect all teacher names across phases for projector creation ----
    all_teacher_names = set()
    all_teacher_surfaces = {}  # name -> set of surfaces
    for phase in phases:
        for t_cfg in phase.get("teachers", []):
            all_teacher_names.add(t_cfg["name"])
            if t_cfg["name"] not in all_teacher_surfaces:
                all_teacher_surfaces[t_cfg["name"]] = set()
            all_teacher_surfaces[t_cfg["name"]].update(t_cfg.get("surfaces", ["state"]))

    # Pre-create projectors for ALL teachers (so optimizer is stable across phases)
    projectors = nn.ModuleDict()
    teacher_dims = {}  # cache hidden dims
    for t_name in sorted(all_teacher_names):
        # Temporarily load to get hidden dim, then unload
        temp_teacher = TeacherAdapter(t_name, device=DEVICE, dtype=torch.float16)
        teacher_dims[t_name] = temp_teacher.hidden_dim
        safe = t_name.replace("/", "_").replace("-", "_").replace(".", "_")
        if "state" in all_teacher_surfaces[t_name]:
            projectors[safe + "__state"] = nn.Linear(dim, temp_teacher.hidden_dim)
        if "semantic" in all_teacher_surfaces[t_name]:
            projectors[safe + "__semantic"] = nn.Linear(dim, temp_teacher.hidden_dim)
        del temp_teacher.model
        del temp_teacher
        torch.cuda.empty_cache()
    projectors = projectors.to(DEVICE)

    # Load projectors from resume checkpoint if available
    if resume_path and "projectors" in ckpt:
        projectors.load_state_dict(ckpt["projectors"])
        print(f"  Loaded projector state from resume checkpoint")

    # ---- Optimizer ----
    param_groups = [
        {"params": list(model.parameters()), "lr": lr},
        {"params": list(projectors.parameters()), "lr": lr * 3, "is_projector": True},
    ]
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    # Load optimizer state from resume
    if resume_path and "optimizer_state_dict" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"  Loaded optimizer state from resume checkpoint")

    del ckpt
    torch.cuda.empty_cache()

    # ---- Phase management ----
    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"kd_{run_name}_log.txt"
    metrics_log = []
    running_ce = 0.0
    running_kd = 0.0
    actual_exit_weights = derive_exit_weights(exit_layers, n_layers) if exit_layers else EXIT_WEIGHTS

    # Current phase state
    current_teachers = []
    current_phase_idx = -1
    current_alpha_state = 0.0
    current_alpha_semantic = 0.0

    def get_phase(step):
        """Get phase index for current step."""
        for i, phase in enumerate(phases):
            if step < phase["end_step"]:
                return i
        return len(phases) - 1  # last phase

    def load_teachers_for_phase(phase):
        """Load teachers specified by phase config."""
        teachers = []
        for t_cfg in phase.get("teachers", []):
            teacher = TeacherAdapter(t_cfg["name"], device=DEVICE, dtype=torch.float16)
            teacher.surfaces = t_cfg.get("surfaces", ["state"])
            teachers.append(teacher)
        return teachers

    def unload_teachers(teachers):
        """Free teacher VRAM."""
        for t in teachers:
            del t.model
        del teachers[:]
        torch.cuda.empty_cache()
        gc.collect()

    print(f"  Exit weights: {actual_exit_weights}")
    print(f"  Starting from step {resume_step + 1}...\n")

    model.train()
    for step in range(resume_step + 1, total_steps + 1):
        # ---- Phase transition check ----
        phase_idx = get_phase(step)
        if phase_idx != current_phase_idx:
            phase = phases[phase_idx]
            print(f"\n  >>> PHASE TRANSITION: {phase['name']} (step {step})")

            # Unload old teachers
            if current_teachers:
                unload_teachers(current_teachers)

            # Load new teachers
            current_teachers = load_teachers_for_phase(phase)
            current_alpha_state = phase.get("alpha_state", 0.0)
            current_alpha_semantic = phase.get("alpha_semantic", 0.0)
            current_phase_idx = phase_idx

            print(f"      Teachers: {[t.name for t in current_teachers] or 'NONE'}")
            print(f"      alpha_state={current_alpha_state}, "
                  f"alpha_semantic={current_alpha_semantic}")

        # ---- LR schedule ----
        current_lr = get_lr_wsd(step, warmup, total_steps, lr, min_lr)
        for pg in opt.param_groups:
            if pg.get("is_projector", False):
                pg["lr"] = current_lr * 3
            else:
                pg["lr"] = current_lr

        opt.zero_grad()
        step_ce = 0.0
        step_kd = 0.0
        bad_step = False
        has_kd = len(current_teachers) > 0
        needs_hidden = any("state" in t.surfaces or "semantic" in t.surfaces
                           for t in current_teachers)
        use_hidden = needs_hidden  # only compute hidden states if rep KD active

        for _ in range(grad_accum):
            inp, tgt = dataset.sample_batch(batch_size, seq_len, device=DEVICE)

            with torch.amp.autocast("cuda", dtype=DTYPE):
                out = model(inp, return_exits=True, return_hidden=use_hidden)
                base_loss, loss_metrics = compute_edsr_loss(
                    out, tgt, exit_weights=actual_exit_weights)
                ce_val = loss_metrics["ce_final"]

                kd_loss = torch.tensor(0.0, device=DEVICE)

                if has_kd:
                    texts = [student_tokenizer.decode(inp[b].tolist())
                             for b in range(inp.size(0))]

                    # Compute student byte offsets
                    stu_offsets = torch.zeros(
                        inp.shape[0], inp.shape[1], 2,
                        device=DEVICE, dtype=torch.long)
                    for b_idx, text in enumerate(texts):
                        enc_s = student_tokenizer.encode(text)
                        n_tok = min(len(enc_s.offsets), inp.shape[1])
                        for ti in range(n_tok):
                            stu_offsets[b_idx, ti, 0] = enc_s.offsets[ti][0]
                            stu_offsets[b_idx, ti, 1] = enc_s.offsets[ti][1]

                    student_hidden = out["hidden"]
                    student_mask = torch.ones(inp.shape[0], inp.shape[1],
                                              device=DEVICE, dtype=torch.long)

                    # Count teachers per surface for fair loss normalization
                    n_state_teachers = sum(1 for t in current_teachers
                                           if "state" in t.surfaces)
                    n_sem_teachers = sum(1 for t in current_teachers
                                         if "semantic" in t.surfaces)

                    for teacher in current_teachers:
                        with torch.no_grad():
                            t_out = teacher.forward(texts, max_length=seq_len)

                        safe_name = (teacher.name.replace("/", "_")
                                     .replace("-", "_").replace(".", "_"))

                        if "state" in teacher.surfaces:
                            assert t_out["hidden"] is not None, (
                                f"Teacher {teacher.name} requested state surface "
                                f"but returned hidden=None")
                            proj = projectors[safe_name + "__state"]
                            s_kd = compute_state_kd_loss(
                                student_hidden, t_out["hidden"],
                                student_offsets=stu_offsets,
                                teacher_offsets=t_out["byte_offsets"],
                                student_mask=student_mask,
                                teacher_mask=t_out["attention_mask"],
                                projector=proj, n_spans=n_spans,
                            )
                            kd_loss = kd_loss + (current_alpha_state / max(n_state_teachers, 1)) * s_kd

                        if "semantic" in teacher.surfaces:
                            assert t_out["hidden"] is not None, (
                                f"Teacher {teacher.name} requested semantic surface "
                                f"but returned hidden=None")
                            proj = projectors[safe_name + "__semantic"]
                            sem_kd = compute_semantic_kd_loss(
                                student_hidden, t_out["hidden"],
                                student_mask=student_mask,
                                teacher_mask=t_out["attention_mask"],
                                projector=proj,
                            )
                            kd_loss = kd_loss + (current_alpha_semantic / max(n_sem_teachers, 1)) * sem_kd

                total_loss = (base_loss + kd_loss) / grad_accum

            if not torch.isfinite(total_loss):
                print(f"  WARNING: NaN/Inf at step {step}, skipping")
                opt.zero_grad()
                bad_step = True
                break

            scaler.scale(total_loss).backward()
            step_ce += ce_val / grad_accum
            step_kd += kd_loss.item() / grad_accum

        if bad_step:
            continue

        scaler.unscale_(opt)
        all_params = list(model.parameters()) + list(projectors.parameters())
        torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.grad is not None], GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        running_ce += step_ce
        running_kd += step_kd

        # ---- Logging ----
        if step % LOG_EVERY == 0:
            avg_ce = running_ce / LOG_EVERY
            avg_kd = running_kd / LOG_EVERY
            bpt = avg_ce / math.log(2)
            phase_name = phases[current_phase_idx]["name"]
            msg = (f"  Step {step:5d} [{phase_name}]: CE={avg_ce:.4f} BPT={bpt:.4f} "
                   f"KD={avg_kd:.4f} LR={current_lr:.2e}")
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            running_ce = 0.0
            running_kd = 0.0

        # ---- Eval ----
        if step % eval_every == 0 or step == total_steps:
            model.eval()
            det = deterministic_eval_dense(model, cache_path)
            kurt = measure_activation_kurtosis(model, dataset, DEVICE)
            entry = {
                "step": step,
                "phase": phases[current_phase_idx]["name"],
                "bpt": det["bpt"],
                "avg_kurtosis": round(
                    sum(kurt["kurtosis"].values()) / len(kurt["kurtosis"]), 2),
                "max_kurtosis": round(max(kurt["kurtosis"].values()), 2),
                "max_activation": round(max(kurt["max_activation"].values()), 2),
                "teachers": [t.name for t in current_teachers],
            }
            for k, v in det.items():
                if k.startswith("bpt_exit"):
                    entry[k] = v
            metrics_log.append(entry)
            msg = (f"  EVAL step {step} [{phases[current_phase_idx]['name']}]: "
                   f"BPT={det['bpt']:.4f} "
                   f"kurtosis_max={entry['max_kurtosis']:.1f} "
                   f"max_act={entry['max_activation']:.1f}")
            print(msg, flush=True)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            model.train()
            torch.cuda.empty_cache()

        # ---- Rolling checkpoint (every 1000 steps) ----
        if step % max(ROLLING_SAVE, 1000) == 0:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "projectors": projectors.state_dict(),
                "config": stu_cfg,
                "kd_config": cfg,
                "phase": phases[current_phase_idx]["name"],
                "metrics": metrics_log,
            }
            rolling = ckpt_dir / f"rolling_step{step}.pt"
            tmp = rolling.with_suffix(".tmp")
            torch.save(ckpt, tmp)
            os.replace(str(tmp), str(rolling))
            # Keep only 2 most recent rolling checkpoints
            rolling_files = sorted(ckpt_dir.glob("rolling_step*.pt"))
            for old in rolling_files[:-2]:
                old.unlink()

    # ---- Final checkpoint ----
    torch.save({
        "step": total_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "projectors": projectors.state_dict(),
        "config": stu_cfg,
        "kd_config": cfg,
        "metrics": metrics_log,
    }, ckpt_dir / f"final_step{total_steps}.pt")

    # ---- Final generation check ----
    model.eval()
    generations = []
    tok_path_gen = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    if tok_path_gen.exists():
        prompts = ["The meaning of life is", "In a distant future",
                   "def fibonacci(n):\n"]
        for prompt_text in prompts:
            input_ids = student_tokenizer.encode(prompt_text).ids
            gen_inp = torch.tensor([input_ids], device=DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=DTYPE):
                for _ in range(64):
                    logits = model(gen_inp)
                    if isinstance(logits, dict):
                        logits = logits["logits"]
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    gen_inp = torch.cat([gen_inp, next_token], dim=1)
                    if gen_inp.size(1) > seq_len:
                        break
            gen_text = student_tokenizer.decode(gen_inp[0].tolist())
            generations.append({"prompt": prompt_text, "generation": gen_text})

    # ---- Save results ----
    output = {
        "name": run_name,
        "config": cfg,
        "metrics": metrics_log,
        "generations": generations,
    }
    out_path = REPO / "results" / f"kd_{run_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PHASED TRAINING COMPLETE: {run_name}")
    print(f"Results: {out_path}")
    if metrics_log:
        final = metrics_log[-1]
        print(f"  Final: BPT={final.get('bpt', '?')}, "
              f"step={final.get('step', '?')}, "
              f"phase={final.get('phase', '?')}")
    print(f"{'='*60}")

    # Cleanup
    if current_teachers:
        unload_teachers(current_teachers)
    del model, opt, scaler, projectors
    torch.cuda.empty_cache()
    gc.collect()


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
    parser.add_argument("--mtp-probe", type=str, default=None,
                        help="Run MTP probe from this parent checkpoint")
    parser.add_argument("--mtp-steps", type=int, default=1000,
                        help="Number of steps for MTP probe")
    parser.add_argument("--halt-probe", type=str, default=None,
                        help="Run halting controller probe from this parent checkpoint")
    parser.add_argument("--halt-steps", type=int, default=1000,
                        help="Number of steps for halting probe")
    parser.add_argument("--mem-probe", type=str, default=None,
                        help="Run memory fusion probe from this parent checkpoint")
    parser.add_argument("--mem-steps", type=int, default=1000,
                        help="Number of steps for memory probe")
    parser.add_argument("--mem-buckets", type=int, default=1_000_000,
                        help="Number of hash buckets for memory tables")
    parser.add_argument("--calibrate-exits", type=str, default=None,
                        help="Run post-hoc exit threshold calibration on checkpoint")
    parser.add_argument("--exit-bench", type=str, default=None,
                        help="Run early-exit inference benchmark on checkpoint")
    parser.add_argument("--expand-200m", type=str, default=None,
                        help="Expand 98M checkpoint to 200M via Net2Wider")
    parser.add_argument("--from-ckpt", type=str, default=None,
                        help="Train from checkpoint (loads config from ckpt)")
    parser.add_argument("--weight-file", type=str, default=None,
                        help="Shard importance weights JSON for weighted sampling")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Initialize model+optimizer from another run's checkpoint (for A/B tests)")
    parser.add_argument("--ab-probe", type=str, default=None,
                        help="Run A/B probe from JSON config file")
    parser.add_argument("--kd-train", type=str, default=None,
                        help="Run RMFD KD training from JSON config file")
    parser.add_argument("--kd-variant", type=str, default=None,
                        help="Run only this variant tag from the KD config (skip others)")
    parser.add_argument("--kd-phased", type=str, default=None,
                        help="Run phase-aware RMFD production training from JSON config")
    parser.add_argument("--committee-map", type=str, default=None,
                        help="Run committee map audit on this student checkpoint (Ekalavya Probe 1)")
    parser.add_argument("--committee-windows", type=int, default=50,
                        help="Number of validation windows for committee map audit")
    args = parser.parse_args()

    if args.ff_dim is not None:
        FF_DIM = args.ff_dim
    if args.max_steps is not None:
        MAX_TRAIN_STEPS = args.max_steps

    if args.committee_map:
        probe_committee_map(args.committee_map, n_windows=args.committee_windows)
        sys.exit(0)

    if args.kd_phased:
        train_kd_phased(args.kd_phased)
        sys.exit(0)

    if args.kd_train:
        train_kd(args.kd_train, variant_filter=args.kd_variant)
        sys.exit(0)

    if args.ab_probe:
        run_ab_probe(args.ab_probe)
        sys.exit(0)

    if args.mtp_probe:
        probe_mtp(args.mtp_probe, steps=args.mtp_steps,
                  run_name=args.run_name or "probe_mtp")
        sys.exit(0)

    if args.halt_probe:
        probe_halting(args.halt_probe, steps=args.halt_steps,
                      run_name=args.run_name or "probe_halting")
        sys.exit(0)

    if args.mem_probe:
        probe_memory(args.mem_probe, steps=args.mem_steps,
                     run_name=args.run_name or "probe_memory",
                     mem_buckets=args.mem_buckets)
        sys.exit(0)

    if args.calibrate_exits:
        calibrate_exits(args.calibrate_exits,
                        run_name=args.run_name or "exit_calibration")
        sys.exit(0)

    if args.exit_bench:
        early_exit_inference(args.exit_bench,
                             run_name=args.run_name or "early_exit_bench")
        sys.exit(0)

    if args.expand_200m:
        expand_98m_to_200m(args.expand_200m)
        sys.exit(0)

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
    train(run_name=run_name, edsr=args.edsr, from_ckpt=args.from_ckpt,
          weight_file=args.weight_file, init_from=args.init_from)
