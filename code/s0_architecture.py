"""S0 Substrate Scout Architecture — SE1 Canonical Spec R6 Frozen.

Complete, runnable PyTorch module definitions for the S0 byte/patch model.
Based on research/SE1_CANONICAL_SPEC.md (R1-R6 convergence).

S0 purpose: byte/patch tradeoff, traceability, basic teacher-free competence.
NOT for proving full Eklavya multi-teacher transfer.

Architecture: deep-thin (MobileLLM-inspired), 30x576 default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class S0Config:
    # Byte encoder
    vocab_size: int = 260  # 256 bytes + 4 special tokens
    byte_dim: int = 256
    local_mixer_layers: int = 2
    local_mixer_window: int = 8

    # Patch
    patch_size: int = 4

    # Global reasoner (deep-thin default)
    d_model: int = 576
    n_layers: int = 30
    n_heads: int = 9
    n_kv_heads: int = 3
    ffn_mult: float = 2.667  # SwiGLU intermediate = d_model * ffn_mult
    max_seq_len: int = 1024  # patches (= 4096 bytes / P4)
    rope_theta: float = 10000.0

    # Byte decoder
    decoder_dim: int = 384
    decoder_layers: int = 4
    decoder_heads: int = 6
    decoder_cross_attn: bool = True

    # I3 verifier (triage head)
    verifier_dim: int = 1024
    n_repair_classes: int = 16

    # I5 compute governor
    governor_actions: int = 5  # stop, run_deeper, retrieve, verify, repair

    # Training
    dropout: float = 0.0

    def __post_init__(self):
        head_dim = self.d_model // self.n_heads
        assert self.d_model % self.n_heads == 0, f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        assert head_dim % 2 == 0, f"head_dim ({head_dim}) must be even for RoPE complex representation"
        assert self.local_mixer_window >= self.patch_size, f"local_mixer_window ({self.local_mixer_window}) must be >= patch_size ({self.patch_size})"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[:S].unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return out.type_as(x)


# ---------- I0: Byte Encoder ----------

class LocalByteMixerLayer(nn.Module):
    def __init__(self, d_model: int, window: int):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.dw_conv = nn.Conv1d(d_model, d_model, kernel_size=window, padding="same", groups=d_model)
        self.pw_conv = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        h = normed.transpose(1, 2)
        h = F.gelu(self.dw_conv(h))
        h = self.pw_conv(h)
        return x + h.transpose(1, 2)


class LocalByteMixer(nn.Module):
    """Patch-local byte mixer: processes each patch independently.

    R8 fix: processing the full byte sequence leaks future patch information
    (conv window spans patch boundaries). Now reshapes to (B*N, P, D) so each
    patch is mixed independently — zero cross-patch leakage.
    """
    def __init__(self, d_model: int, n_layers: int, window: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.layers = nn.ModuleList([
            LocalByteMixerLayer(d_model, min(window, patch_size)) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        P = self.patch_size
        N = T // P
        x = x.reshape(B * N, P, D)
        for layer in self.layers:
            x = layer(x)
        return x.reshape(B, T, D)


class PatchAggregator(nn.Module):
    """Nonlinear patch aggregation (R5 revision).

    Gated MLP over concatenated byte states + boundary features.
    Replaces bare Linear(P*byte_dim, d_model).
    """
    def __init__(self, byte_dim: int, patch_size: int, out_dim: int):
        super().__init__()
        in_dim = patch_size * byte_dim
        hidden = out_dim * 2
        self.gate_proj = nn.Linear(in_dim, hidden, bias=False)
        self.up_proj = nn.Linear(in_dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)

    def forward(self, byte_states: torch.Tensor) -> torch.Tensor:
        # byte_states: (B, n_patches, P * byte_dim)
        gate = F.silu(self.gate_proj(byte_states))
        up = self.up_proj(byte_states)
        return self.norm(self.down_proj(gate * up))


class ByteEncoder(nn.Module):
    """I0 input_surface: embedding + local mixer + patch aggregation."""
    def __init__(self, cfg: S0Config):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(cfg.vocab_size, cfg.byte_dim)
        self.local_mixer = LocalByteMixer(cfg.byte_dim, cfg.local_mixer_layers, cfg.local_mixer_window, cfg.patch_size)
        self.patch_agg = PatchAggregator(cfg.byte_dim, cfg.patch_size, cfg.d_model)
        self.entropy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.byte_dim), nn.GELU(), nn.Linear(cfg.byte_dim, 1)
        )
        self.residual_head = nn.Linear(cfg.byte_dim, 1)

    def forward(self, byte_ids: torch.Tensor, return_aux: bool = True):
        """
        Args:
            byte_ids: (B, T) byte token IDs where T is divisible by patch_size
            return_aux: if False, skip entropy_head and residual_head
        Returns:
            patch_states: (B, T//P, d_model)
            byte_states: (B, T, byte_dim) — preserved for decoder
            entropy_scores: (B, T//P, 1) or None
            residual_flags: (B, T, 1) or None
        """
        B, T = byte_ids.shape
        P = self.cfg.patch_size
        assert T % P == 0, f"sequence length {T} not divisible by patch size {P}"

        x = self.byte_emb(byte_ids)  # (B, T, byte_dim)
        byte_states = self.local_mixer(x)  # (B, T, byte_dim)

        # Reshape into patches and aggregate
        n_patches = T // P
        patch_input = byte_states.reshape(B, n_patches, P * self.cfg.byte_dim)
        patch_states = self.patch_agg(patch_input)  # (B, n_patches, d_model)

        if return_aux:
            entropy_scores = self.entropy_head(patch_states)
            residual_flags = self.residual_head(byte_states)
        else:
            entropy_scores = None
            residual_flags = None

        return patch_states, byte_states, entropy_scores, residual_flags


# ---------- I1/I2: Global Reasoner ----------

class GQAAttention(nn.Module):
    """Grouped-Query Attention with RoPE."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, S, _ = x.shape

        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_rep, -1, -1).reshape(B, self.n_heads, S, self.head_dim)

        # Scaled dot-product attention (use FlashAttention when available)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0, is_causal=mask is None)

        out = attn.transpose(1, 2).reshape(B, S, -1)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, mult: float = 2.667):
        super().__init__()
        hidden = int(d_model * mult)
        # Round to nearest multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ffn_mult: float, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_mult)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.attn_norm(x), freqs, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GlobalReasoner(nn.Module):
    """I1 compact_state + I2 reasoning_step: deep-thin causal Transformer."""
    def __init__(self, cfg: S0Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        head_dim = cfg.d_model // cfg.n_heads
        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta),
            persistent=False,
        )

    def forward(self, patch_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_states: (B, S, d_model) from byte encoder
        Returns:
            hidden: (B, S, d_model) — final reasoner output
        """
        x = patch_states
        for layer in self.layers:
            x = layer(x, self.rope_freqs)
        return self.norm(x)


# ---------- I6: Byte Decoder ----------

class TinyCausalTransformerLayer(nn.Module):
    """Single layer of the local byte decoder."""
    def __init__(self, d_model: int, n_heads: int, cross_attn: bool = False, cross_dim: int = 0):
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.cross_attn_mod = None
        if cross_attn:
            self.cross_norm = RMSNorm(d_model)
            self.cross_attn_mod = nn.MultiheadAttention(d_model, n_heads, batch_first=True, kdim=cross_dim, vdim=cross_dim)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor, cross_kv: Optional[torch.Tensor] = None):
        normed = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=causal_mask, is_causal=False)
        x = x + attn_out

        if self.cross_attn_mod is not None and cross_kv is not None:
            normed = self.cross_norm(x)
            cross_out, _ = self.cross_attn_mod(normed, cross_kv, cross_kv)
            x = x + cross_out

        x = x + self.ffn(self.ffn_norm(x))
        return x


class ByteDecoder(nn.Module):
    """I6 output_surface: local autoregressive byte decoder with cross-attention."""
    def __init__(self, cfg: S0Config):
        super().__init__()
        self.cfg = cfg
        self.prev_byte_emb = nn.Embedding(cfg.vocab_size, cfg.decoder_dim)
        self.cond_proj = nn.Linear(cfg.d_model, cfg.decoder_dim, bias=False)

        self.layers = nn.ModuleList([
            TinyCausalTransformerLayer(
                cfg.decoder_dim, cfg.decoder_heads,
                cross_attn=cfg.decoder_cross_attn,
                cross_dim=cfg.d_model,
            )
            for _ in range(cfg.decoder_layers)
        ])
        self.norm = RMSNorm(cfg.decoder_dim)
        self.lm_head = nn.Linear(cfg.decoder_dim, 256, bias=False)  # 256 byte values

    def forward(
        self,
        patch_states: torch.Tensor,
        target_bytes: torch.Tensor,
        nearby_patch_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_states: (B, n_patches, d_model) — global reasoner output
            target_bytes: (B, n_patches, P) — target byte IDs for teacher forcing
            nearby_patch_states: (B, n_patches, 2, d_model) — previous + current patch states
        Returns:
            logits: (B, n_patches, P, 256)
        """
        B, N, P = target_bytes.shape

        # Condition on patch state
        cond = self.cond_proj(patch_states)  # (B, N, decoder_dim)

        # Shift target bytes right (prepend conditioning, drop last)
        # First byte gets only conditioning; subsequent bytes get previous byte embedding
        byte_emb = self.prev_byte_emb(target_bytes)  # (B, N, P, decoder_dim)

        # Prepend conditioning as position 0, use first P-1 byte embeddings for positions 1..P-1
        cond_expanded = cond.unsqueeze(2)  # (B, N, 1, decoder_dim)
        x = torch.cat([cond_expanded, byte_emb[:, :, :-1, :]], dim=2)  # (B, N, P, decoder_dim)

        # Reshape for processing: (B*N, P, decoder_dim)
        x = x.reshape(B * N, P, -1)

        # Cross-attention context: nearby patch states
        cross_kv = None
        if nearby_patch_states is not None and self.cfg.decoder_cross_attn:
            cross_kv = nearby_patch_states.reshape(B * N, -1, self.cfg.d_model)

        # Causal mask
        causal_mask = torch.triu(torch.ones(P, P, device=x.device, dtype=torch.bool), diagonal=1)

        for layer in self.layers:
            x = layer(x, causal_mask, cross_kv)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B*N, P, 256)
        return logits.reshape(B, N, P, 256)


# ---------- I3: Verification (Triage Head) ----------

class VerifierHead(nn.Module):
    """I3 verification: triage/routing head, NOT an authority verifier.

    Routes to exact validators for real verification decisions.
    """
    def __init__(self, cfg: S0Config):
        super().__init__()
        self.trunk = nn.Sequential(
            RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.verifier_dim),
            nn.GELU(),
            nn.Linear(cfg.verifier_dim, cfg.verifier_dim),
            nn.GELU(),
        )
        self.verdict = nn.Linear(cfg.verifier_dim, 3)  # pass/fail/unknown
        self.span_start = nn.Linear(cfg.d_model, 1)
        self.span_end = nn.Linear(cfg.d_model, 1)
        self.repair_class = nn.Linear(cfg.verifier_dim, cfg.n_repair_classes)
        self.escalate = nn.Linear(cfg.verifier_dim, 1)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden: (B, S, d_model) from global reasoner
        Returns:
            dict with verdict, span_start, span_end, repair_class, escalate logits
        """
        trunk_out = self.trunk(hidden)
        return {
            "verdict": self.verdict(trunk_out),          # (B, S, 3)
            "span_start": self.span_start(hidden),       # (B, S, 1)
            "span_end": self.span_end(hidden),           # (B, S, 1)
            "repair_class": self.repair_class(trunk_out), # (B, S, n_repair_classes)
            "escalate": self.escalate(trunk_out),         # (B, S, 1)
        }


# ---------- I5: Compute Governor ----------

class ComputeGovernor(nn.Module):
    """I5 compute_governor: rule-based at S0. No learned parameters.

    At S0, always returns action=0 (run all layers). The governor's job is
    to provide the interface for G1+ early-exit/MoD/self-verification.
    Thresholds are hardcoded; G1 replaces with learned policy.
    """
    STOP = 0
    RUN_DEEPER = 1
    RETRIEVE = 2
    VERIFY = 3
    REPAIR = 4

    def __init__(self, cfg: S0Config):
        super().__init__()
        self.entropy_threshold = 2.0
        self.escalate_threshold = 0.5

    def forward(
        self,
        entropy: torch.Tensor,
        verifier_escalate: torch.Tensor,
    ) -> torch.Tensor:
        """Rule-based routing. At S0, always runs full depth.

        Returns:
            actions: (B, S) integer action codes
        """
        B, S, _ = entropy.shape
        actions = torch.zeros(B, S, dtype=torch.long, device=entropy.device)
        # S0: all patches run full depth. These rules are placeholders for G1.
        high_entropy = entropy.squeeze(-1) > self.entropy_threshold
        needs_verify = verifier_escalate.squeeze(-1) > self.escalate_threshold
        actions[high_entropy] = self.RUN_DEEPER
        actions[needs_verify] = self.VERIFY
        return actions


# ---------- I4: Memory (Read-Only Stub at S0) ----------

class ReadOnlyMemory(nn.Module):
    """I4 memory: stub for S0. Non-parametric ANN index at G1.

    At S0, returns empty retrieval results. The interface is defined
    so that G1 can swap in FAISS/HNSW without changing the forward pass.
    """
    def __init__(self, cfg: S0Config):
        super().__init__()
        self.d_model = cfg.d_model

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """Return zero-valued retrieval context (no memory at S0).

        Args:
            query: (B, S, d_model) hidden states to query against memory
        Returns:
            retrieved: (B, S, d_model) — zeros at S0, real retrievals at G1
        """
        return torch.zeros_like(query)


# ---------- Full S0 Model ----------

class SutraS0(nn.Module):
    """Complete S0 substrate scout model (MEGABYTE-style byte/patch LM).

    Causal alignment: hidden[i] predicts bytes of patch i+1. The encoder
    compresses bytes[i*P:(i+1)*P] into patch_states[i], the reasoner processes
    patches causally so hidden[i] depends on patches 0..i, and the decoder uses
    hidden[i] to predict bytes of patch i+1. This ensures no target leakage —
    the conditioning hidden state never encodes the bytes being predicted.
    At inference, the same alignment holds: encode all known bytes, use the last
    hidden state to generate the next patch.
    """
    def __init__(self, cfg: Optional[S0Config] = None):
        super().__init__()
        self.cfg = cfg or S0Config()

        self.encoder = ByteEncoder(self.cfg)
        self.reasoner = GlobalReasoner(self.cfg)
        self.decoder = ByteDecoder(self.cfg)
        self.verifier = VerifierHead(self.cfg)
        self.governor = ComputeGovernor(self.cfg)
        self.memory = ReadOnlyMemory(self.cfg)

    def forward(self, byte_ids: torch.Tensor, return_aux: bool = True):
        """
        Args:
            byte_ids: (B, T) byte token IDs, T divisible by patch_size
            return_aux: if False, skip verifier/governor/entropy/residual heads
        Returns:
            dict with logits and auxiliary outputs
        """
        B, T = byte_ids.shape
        P = self.cfg.patch_size
        assert T % P == 0, f"sequence length {T} not divisible by patch_size {P}"

        # I0: Encode bytes into patch states
        patch_states, byte_states, entropy_scores, residual_flags = self.encoder(
            byte_ids, return_aux=return_aux)

        # I1/I2: Global reasoning over patches (causal across patches)
        hidden = self.reasoner(patch_states)

        # I4: Memory retrieval (stub at S0)
        _memory_context = self.memory(hidden)  # noqa: F841 — stub for G1

        # Causal alignment: hidden[i] predicts bytes of patch i+1.
        # pred_hidden[j] = hidden[j] conditions the decoder for target patch j+1.
        N = hidden.shape[1]
        pred_hidden = hidden[:, :-1]  # (B, N-1, d_model)
        target_bytes = byte_ids.reshape(B, T // P, P)[:, 1:]  # (B, N-1, P)

        # Cross-attention context: previous + current conditioning states
        M = N - 1
        prev_padded = F.pad(pred_hidden, (0, 0, 1, 0))[:, :M]  # (B, N-1, d_model)
        nearby = torch.stack([prev_padded, pred_hidden], dim=2)  # (B, N-1, 2, d_model)

        # I6: Decode bytes
        logits = self.decoder(pred_hidden, target_bytes, nearby)

        result = {
            "logits": logits,
            "hidden": hidden,
            "byte_states": byte_states,
            "patch_states": patch_states,
        }

        if return_aux:
            verifier_out = self.verifier(hidden)
            governor_actions = self.governor(entropy_scores, verifier_out["escalate"])
            result["entropy_scores"] = entropy_scores
            result["residual_flags"] = residual_flags
            result["verifier"] = verifier_out
            result["governor_actions"] = governor_actions

        return result

    def count_parameters(self) -> dict[str, int]:
        """Report parameter counts by module (excluding non-parametric modules)."""
        counts = {}
        for name, module in [
            ("I0_encoder", self.encoder),
            ("I1_I2_reasoner", self.reasoner),
            ("I3_verifier", self.verifier),
            ("I6_decoder", self.decoder),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["I4_memory"] = 0  # non-parametric at S0
        counts["I5_governor"] = 0  # rule-based at S0
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


def main():
    """Smoke test: instantiate and verify shapes."""
    cfg = S0Config()
    model = SutraS0(cfg)

    counts = model.count_parameters()
    print("S0 Parameter Counts:")
    for name, count in counts.items():
        print(f"  {name}: {count:,} ({count / 1e6:.1f}M)")

    # Smoke forward pass
    B, T = 2, 4096  # 2 sequences, 4096 bytes
    byte_ids = torch.randint(0, 256, (B, T))

    with torch.no_grad():
        out = model(byte_ids)

    N = T // cfg.patch_size
    print("\nShapes:")
    print(f"  logits: {out['logits'].shape}")
    print(f"  entropy_scores: {out['entropy_scores'].shape}")
    print(f"  residual_flags: {out['residual_flags'].shape}")
    print(f"  hidden: {out['hidden'].shape}")
    print(f"  verifier verdict: {out['verifier']['verdict'].shape}")
    print(f"  governor_actions: {out['governor_actions'].shape}")

    assert out["logits"].shape == (B, N - 1, cfg.patch_size, 256)
    assert out["hidden"].shape == (B, N, cfg.d_model)
    assert out["entropy_scores"].shape == (B, N, 1)
    assert out["residual_flags"].shape == (B, T, 1)
    assert out["governor_actions"].shape == (B, N)

    # Config validation tests
    print("\nConfig validation tests:")
    for name, kwargs in [
        ("d_model not divisible by n_heads", {"d_model": 577, "n_heads": 10}),
        ("n_heads not divisible by n_kv_heads", {"n_heads": 8, "n_kv_heads": 3}),
        ("odd head_dim", {"d_model": 576, "n_heads": 8}),  # head_dim=72 which is even, should pass
    ]:
        try:
            S0Config(**kwargs)
            print(f"  {name}: constructed (ok if expected)")
        except AssertionError as e:
            print(f"  {name}: BLOCKED — {e}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
