"""Option G: Teacher-Anchored Semantic Codec.

A CausalByteTransformer that maps byte sequences to semantically meaningful
patch representations, pre-trained to retrieve teacher token embeddings.

Codex R63 sign-off:
- 256-byte sliding window causal attention (NOT full 4096)
- 4 layers, 256 dim
- InfoNCE retrieval loss (not cosine alone)
- Shuffled-target controls must fail
- Phase 1 gate: top-1 retrieval >>chance, shuffled ≈ chance
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class CodecConfig:
    byte_vocab: int = 260
    codec_dim: int = 256
    codec_layers: int = 4
    codec_heads: int = 4
    codec_ffn_mult: int = 4
    window_size: int = 256
    patch_size: int = 4
    teacher_dim: int = 1024
    temperature: float = 0.07


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class WindowedCausalAttention(nn.Module):
    """Causal attention with a sliding window.

    Each position attends to at most `window_size` previous positions.
    This is O(T × W) instead of O(T²) where W = window_size.
    """
    def __init__(self, d_model: int, n_heads: int, window_size: int = 256):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        HD = self.head_dim

        q = self.wq(x).reshape(B, T, H, HD).transpose(1, 2)
        k = self.wk(x).reshape(B, T, H, HD).transpose(1, 2)
        v = self.wv(x).reshape(B, T, H, HD).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention with causal mask
        # For windowed attention, we construct a custom mask
        # But for simplicity and GPU efficiency, use full causal attention
        # with the sequence chunked to window_size if T > window_size
        # Actually, PyTorch 2.0+ SDPA with is_causal=True + sliding window
        # isn't natively supported. Use manual mask.
        if T <= self.window_size:
            # Short sequence: standard causal attention
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Sliding window: custom mask
            # Each position i attends to max(0, i-window_size+1) through i
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool)
            mask = torch.triu(mask, diagonal=1)  # causal: upper triangle blocked
            # Also block positions more than window_size behind
            window_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                                     diagonal=-self.window_size)
            mask = mask | ~window_mask
            attn_mask = mask.float().masked_fill(mask, float('-inf'))
            # Manual attention with mask
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.wo(out)


class CodecTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, window_size: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = WindowedCausalAttention(d_model, n_heads, window_size)
        self.norm2 = RMSNorm(d_model)
        ffn_dim = d_model * ffn_mult
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        normed = self.norm2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(normed)) * self.up_proj(normed))
        return x


class CausalByteTransformer(nn.Module):
    """Small causal byte transformer with windowed attention.

    Produces a representation at every byte position, using left-context
    within a sliding window. For producing patch representations, sample
    every P-th position.
    """
    def __init__(self, cfg: CodecConfig):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.byte_vocab, cfg.codec_dim)
        self.layers = nn.ModuleList([
            CodecTransformerLayer(cfg.codec_dim, cfg.codec_heads,
                                  cfg.codec_ffn_mult, cfg.window_size)
            for _ in range(cfg.codec_layers)
        ])
        self.norm = RMSNorm(cfg.codec_dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: (B, T) byte token IDs
        Returns:
            hidden: (B, T, codec_dim) — per-byte contextual representations
        """
        x = self.byte_emb(byte_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def get_patch_states(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Get patch representations by sampling every P-th byte position."""
        hidden = self.forward(byte_ids)  # (B, T, codec_dim)
        P = self.cfg.patch_size
        # Sample at the END of each patch (position P-1, 2P-1, 3P-1, ...)
        patch_states = hidden[:, P-1::P, :]  # (B, T//P, codec_dim)
        return patch_states


class PatchProjection(nn.Module):
    """Project from codec space (codec_dim) to global reasoner space (d_model)."""
    def __init__(self, codec_dim: int, d_model: int):
        super().__init__()
        hidden = d_model * 2
        self.gate_proj = nn.Linear(codec_dim, hidden, bias=False)
        self.up_proj = nn.Linear(codec_dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.norm(self.down_proj(gate * up))


class AlignmentHead(nn.Module):
    """Project codec hidden states to teacher embedding space for alignment."""
    def __init__(self, codec_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(codec_dim, teacher_dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize."""
        h = self.proj(hidden)
        return F.normalize(h, dim=-1)


class SemanticCodec(nn.Module):
    """Full semantic codec: CausalByteTransformer + PatchProjection + AlignmentHead.

    Phase 1 (pre-training): use alignment_head for InfoNCE against teacher embeddings
    Phase 2 (core training): use patch_projection to feed global reasoner, freeze codec
    """
    def __init__(self, cfg: CodecConfig, d_model: int = 1152):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.encoder = CausalByteTransformer(cfg)
        self.patch_projection = PatchProjection(cfg.codec_dim, d_model)
        self.alignment_head = AlignmentHead(cfg.codec_dim, cfg.teacher_dim)

    def forward_phase1(self, byte_ids: torch.Tensor, anchor_positions: torch.Tensor):
        """Phase 1: produce alignment embeddings at anchor positions.

        Args:
            byte_ids: (B, T) byte token IDs
            anchor_positions: (B, N_anchors) byte positions where tokens end
        Returns:
            projected: (B, N_anchors, teacher_dim) — L2-normalized
        """
        hidden = self.encoder(byte_ids)  # (B, T, codec_dim)
        B, T, D = hidden.shape

        # Gather hidden states at anchor positions
        # anchor_positions: (B, N) — indices into T dimension
        N = anchor_positions.shape[1]
        idx = anchor_positions.unsqueeze(-1).expand(B, N, D)
        anchored = torch.gather(hidden, 1, idx)  # (B, N, codec_dim)

        # Project to teacher space
        return self.alignment_head(anchored)  # (B, N, teacher_dim)

    def forward_phase2(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Phase 2: produce patch states for global reasoner.

        Args:
            byte_ids: (B, T)
        Returns:
            patch_states: (B, T//P, d_model)
        """
        patch_hidden = self.encoder.get_patch_states(byte_ids)  # (B, T//P, codec_dim)
        return self.patch_projection(patch_hidden)  # (B, T//P, d_model)

    def count_params(self) -> dict[str, int]:
        return {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "patch_projection": sum(p.numel() for p in self.patch_projection.parameters()),
            "alignment_head": sum(p.numel() for p in self.alignment_head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


def infonce_loss(
    projected: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, float]:
    """InfoNCE loss for retrieval-based alignment.

    Args:
        projected: (B, N, D) — L2-normalized student projections at anchors
        teacher_embeddings: (B, N, D) — L2-normalized teacher embeddings at anchors
        temperature: softmax temperature
    Returns:
        loss: scalar
        top1_accuracy: float (for monitoring)
    """
    B, N, D = projected.shape

    # Reshape to (B*N, D) for all-pairs similarity
    proj_flat = projected.reshape(B * N, D)  # queries
    teach_flat = teacher_embeddings.reshape(B * N, D)  # keys

    # Similarity matrix: (B*N, B*N)
    sim = torch.matmul(proj_flat, teach_flat.t()) / temperature

    # Labels: diagonal (each query should match its own key)
    labels = torch.arange(B * N, device=sim.device)

    # Cross-entropy loss
    loss = F.cross_entropy(sim, labels)

    # Top-1 accuracy
    with torch.no_grad():
        preds = sim.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

    return loss, acc


def build_codec(d_model: int = 1152, window: int = 256) -> SemanticCodec:
    """Build a semantic codec with default settings."""
    cfg = CodecConfig(window_size=window)
    return SemanticCodec(cfg, d_model=d_model)


if __name__ == "__main__":
    # Quick test
    cfg = CodecConfig()
    codec = SemanticCodec(cfg, d_model=1152)
    params = codec.count_params()
    print("Semantic Codec Parameter Counts:")
    for k, v in params.items():
        print(f"  {k}: {v:,} ({v/1e6:.2f}M)")

    # Test forward passes
    B, T = 2, 256
    byte_ids = torch.randint(0, 256, (B, T))

    # Phase 1
    anchors = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    proj = codec.forward_phase1(byte_ids, anchors)
    print(f"\nPhase 1 output: {proj.shape}")  # (2, 4, 1024)
    print(f"  L2 norm check: {proj.norm(dim=-1).mean():.4f}")  # should be ~1.0

    # Phase 2
    patches = codec.forward_phase2(byte_ids)
    print(f"Phase 2 output: {patches.shape}")  # (2, 64, 1152)

    # InfoNCE test
    teacher_emb = F.normalize(torch.randn(B, 4, 1024), dim=-1)
    loss, acc = infonce_loss(proj, teacher_emb)
    print(f"\nInfoNCE loss: {loss.item():.4f}")
    print(f"Top-1 accuracy: {acc:.4f} (chance = {1/(B*4):.4f})")
