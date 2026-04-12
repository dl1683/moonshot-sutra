"""Sutra-Dyad: Compression-native byte-level language model.

Megabyte-style architecture: Patch Embed → Global Transformer → Local Decoder.
Stage 0: flat-byte baseline with fixed-size patches (no learned patching).
Stage 1 (future): learned dyadic patching with {1,2,4,8,16,32}-byte spans.

Architecture (Stage 0, ~155M params):
  - Input: raw UTF-8 bytes (vocab=256)
  - Patch embedding: concatenate P byte embeddings → project to d_global=1024
  - Global transformer: 12 causal layers, d=1024, 16 heads, SwiGLU, RoPE
  - Local decoder: 1 causal layer, d=256, 4 heads (lightweight — global does heavy lifting)
  - Output: 256-way next-byte prediction

Causality: global output is shifted by 1 patch position so predictions for
patch j use only global context from patches 0..j-1. The local decoder's
causal attention ensures byte t uses only bytes 0..t-1.

Usage:
  python code/sutra_dyad.py                       # Train Stage 0 baseline
  python code/sutra_dyad.py --max-steps 3000      # Short probe
  python code/sutra_dyad.py --det-eval <ckpt>     # Evaluate checkpoint
"""

import sys, os, math, json, time, gc, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))
from data_loader import ByteShardedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# ---- Architecture Constants (Stage 0) ----
N_BYTES = 256           # byte vocabulary (0-255)
D_LOCAL = 256           # local decoder dimension (small — Megabyte-style)
N_LOCAL_HEADS = 4
N_LOCAL_DEC_LAYERS = 1  # single decoder layer (global model does heavy lifting)
FF_LOCAL = 682          # SwiGLU: ~2.67x d_local

D_GLOBAL = 1024         # global transformer dimension
N_GLOBAL_HEADS = 16
N_GLOBAL_LAYERS = 12
FF_GLOBAL = 2730        # SwiGLU: ~2.67x d_global

PATCH_SIZE = 6          # fixed patch size
SEQ_BYTES = 1536        # 1536 bytes = 256 patches of 6

# Training defaults
BATCH_SIZE = 64
GRAD_ACCUM = 2          # effective batch = 128 sequences
MAX_TRAIN_STEPS = 10000
EVAL_EVERY = 500
LOG_EVERY = 50
ROLLING_SAVE = 500
LR = 3e-4
MIN_LR = 1e-5
WARMUP_STEPS = 300
GRAD_CLIP = 1.0
WD = 0.1

# ---- Components ----

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, self.weight.shape, self.weight.to(x.dtype), self.eps)


class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.gate = nn.Linear(dim, ff_dim, bias=False)
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---- RoPE ----

def precompute_rope(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, T, H, D)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos = cos[:x.shape[1], :].unsqueeze(0).unsqueeze(2).to(x.dtype)
    sin = sin[:x.shape[1], :].unsqueeze(0).unsqueeze(2).to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---- Transformer Block (used for both global and local) ----

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x


# ---- Main Model ----

class SutraDyad(nn.Module):
    """Megabyte-style byte-level language model.

    Patch Embed → Global Transformer (causal over patches) → Local Decoder (causal over bytes)

    Causality contract:
    - patch_emb[j] = Linear(byte_embed[j*P : (j+1)*P].flatten())
    - global_out[j] = CausalTransformer(patch_emb[0..j])  (sees patches 0..j)
    - global_shifted[j] = global_out[j-1]  (shifted: bytes in patch j only see patches 0..j-1)
    - local_decoder is causal over all T bytes
    - byte t sees: all bytes 0..t-1 + global context from patches 0..floor(t/P)-1
    """

    def __init__(self, n_bytes=N_BYTES, d_local=D_LOCAL, d_global=D_GLOBAL,
                 n_local_heads=N_LOCAL_HEADS, n_global_heads=N_GLOBAL_HEADS,
                 n_local_dec_layers=N_LOCAL_DEC_LAYERS, n_global_layers=N_GLOBAL_LAYERS,
                 ff_local=FF_LOCAL, ff_global=FF_GLOBAL,
                 patch_size=PATCH_SIZE, max_seq_bytes=SEQ_BYTES):
        super().__init__()
        self.n_bytes = n_bytes
        self.d_local = d_local
        self.d_global = d_global
        self.patch_size = patch_size
        self.max_seq_bytes = max_seq_bytes
        self.max_patches = max_seq_bytes // patch_size

        # Byte embedding
        self.byte_embed = nn.Embedding(n_bytes, d_local)

        # Patch embedding: flatten P byte embeddings → project to d_global
        self.patch_proj = nn.Linear(patch_size * d_local, d_global, bias=False)

        # Global transformer (causal over patches)
        self.global_layers = nn.ModuleList([
            TransformerBlock(d_global, n_global_heads, ff_global)
            for _ in range(n_global_layers)
        ])
        self.global_norm = RMSNorm(d_global)

        # RoPE for global (over patches)
        max_patches = max_seq_bytes // patch_size + 16
        g_head_dim = d_global // n_global_heads
        g_cos, g_sin = precompute_rope(g_head_dim, max_patches)
        self.register_buffer('g_rope_cos', g_cos, persistent=False)
        self.register_buffer('g_rope_sin', g_sin, persistent=False)

        # Global-to-local projection
        self.global_to_local = nn.Linear(d_global, d_local, bias=False)

        # Local decoder (causal over full byte sequence)
        self.local_decoder = nn.ModuleList([
            TransformerBlock(d_local, n_local_heads, ff_local)
            for _ in range(n_local_dec_layers)
        ])
        self.local_norm = RMSNorm(d_local)

        # RoPE for local (over bytes)
        max_bytes = max_seq_bytes + 16
        l_head_dim = d_local // n_local_heads
        l_cos, l_sin = precompute_rope(l_head_dim, max_bytes)
        self.register_buffer('l_rope_cos', l_cos, persistent=False)
        self.register_buffer('l_rope_sin', l_sin, persistent=False)

        # Output head (tied with byte_embed)
        self.output_head = nn.Linear(d_local, n_bytes, bias=False)
        self.output_head.weight = self.byte_embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, byte_ids, targets=None):
        """Forward pass.

        Args:
            byte_ids: (B, T) long tensor of byte values [0, 255]
            targets: (B, T) long tensor of next-byte targets
        Returns:
            loss scalar if targets provided, else logits (B, T, 256)
        """
        B, T = byte_ids.shape
        P = self.patch_size

        # Pad T to multiple of P
        pad = (P - T % P) % P
        if pad > 0:
            byte_ids = F.pad(byte_ids, (0, pad), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad), value=-100)  # -100 = ignore in CE
        T_padded = byte_ids.shape[1]
        N = T_padded // P

        # 1. Byte embeddings
        byte_emb = self.byte_embed(byte_ids)  # (B, T_padded, d_local)

        # 2. Form patch embeddings: concatenate P byte embeddings, project
        byte_emb_patched = byte_emb.reshape(B, N, P * self.d_local)
        patch_emb = self.patch_proj(byte_emb_patched)  # (B, N, d_global)

        # 3. Global causal transformer over patches
        g_cos = self.g_rope_cos.to(patch_emb.device)
        g_sin = self.g_rope_sin.to(patch_emb.device)
        x = patch_emb
        for layer in self.global_layers:
            x = layer(x, g_cos, g_sin)
        global_out = self.global_norm(x)  # (B, N, d_global)

        # 4. Shift global output by 1 patch (causality: patch j uses global 0..j-1)
        global_shifted = torch.cat([
            torch.zeros(B, 1, self.d_global, device=byte_ids.device, dtype=global_out.dtype),
            global_out[:, :-1, :]
        ], dim=1)  # (B, N, d_global)

        # 5. Project to d_local THEN expand to byte level (avoids (B,T,d_global) temporary)
        global_local = self.global_to_local(global_shifted)  # (B, N, d_local)
        global_local = global_local.unsqueeze(2).expand(B, N, P, self.d_local)
        global_local = global_local.reshape(B, T_padded, self.d_local)

        # 6. Combine byte embeddings + global context for local decoder
        local_input = byte_emb + global_local

        # 7. Local causal decoder over full byte sequence
        l_cos = self.l_rope_cos.to(local_input.device)
        l_sin = self.l_rope_sin.to(local_input.device)
        x = local_input
        for layer in self.local_decoder:
            x = layer(x, l_cos, l_sin)
        x = self.local_norm(x)

        # 8. Output logits
        logits = self.output_head(x)  # (B, T_padded, n_bytes)

        if targets is not None:
            # Trim padding for loss
            loss = F.cross_entropy(
                logits.reshape(-1, self.n_bytes),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return loss

        return logits[:, :T, :]  # remove padding

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        unique = total - self.output_head.weight.numel()  # tied weight
        return {"total": total, "unique_untied": unique}

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=256, temperature=0.8, top_k=50):
        """Autoregressive byte generation."""
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_bytes):
            if ids.shape[1] > self.max_seq_bytes:
                ids = ids[:, -self.max_seq_bytes:]
            logits = self.forward(ids)  # (1, T, 256)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_byte], dim=1)

        return ids[0].tolist()


# ---- Stage 1 Components ----

# Stage 1 constants
S1_D_LOCAL = 640
S1_N_LOCAL_HEADS = 10
S1_FF_LOCAL = 1920       # SwiGLU ~3x
S1_N_ENC_LAYERS = 2      # within-patch bidirectional encoder
S1_N_DEC_LAYERS = 4      # byte decoder layers
S1_BYPASS_DIM = 160       # byte-residual bypass dimension
S1_WINDOW = 96            # reserved for future sliding window (currently full causal)



class LocalDecoderBlockS1(nn.Module):
    """Stage 1 local decoder block: causal self-attn + bypass + FFN.

    Cross-attention was removed after Phase B ablation proved it harmful
    (BPB improved by 0.009 without it). All cross-level communication
    flows through the bypass path and global_to_local projection.
    Saves 8.5M params (4.3%) and ~2GB activation VRAM.
    """
    def __init__(self, dim, n_heads, ff_dim, kv_dim, bypass_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)
        # Byte-residual bypass: U*silu(V*r)
        self.bypass_v = nn.Linear(bypass_dim, dim, bias=False)
        self.bypass_u = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, bypass_r):
        # Self-attention (causal)
        x = x + self.attn(self.norm1(x), cos, sin)
        # Bypass injection (load-bearing: +0.679 BPB contribution)
        x = x + self.bypass_u(F.silu(self.bypass_v(bypass_r)))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class WithinPatchEncoder(nn.Module):
    """2-layer bidirectional transformer over bytes within each patch.
    Processes each patch independently to produce compressed patch summaries.
    Includes learnable positional embeddings for byte ordering within patch."""
    def __init__(self, dim, n_heads, ff_dim, max_patch_size=32, n_layers=2):
        super().__init__()
        self.pos_embed = nn.Embedding(max_patch_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': RMSNorm(dim),
                'attn': nn.MultiheadAttention(dim, n_heads, batch_first=True, bias=False),
                'norm2': RMSNorm(dim),
                'ffn': SwiGLU(dim, ff_dim),
            }))

    def forward(self, x):
        """x: (B*N, P, dim) — bytes within each patch, flattened across batch and patches."""
        P = x.shape[1]
        pos_ids = torch.arange(P, device=x.device)
        x = x + self.pos_embed(pos_ids).unsqueeze(0)
        for layer in self.layers:
            residual = x
            x_norm = layer['norm1'](x)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm)
            x = residual + attn_out
            x = x + layer['ffn'](layer['norm2'](x))
        return x


class AttentionalPooling(nn.Module):
    """Pool within-patch encoder outputs to a single patch summary using learned query."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=1, batch_first=True, bias=False)

    def forward(self, x):
        """x: (B*N, P, dim). Returns: (B*N, dim)."""
        BN = x.shape[0]
        q = self.query.expand(BN, -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class SutraDyadS1(nn.Module):
    """Sutra-Dyad Stage 1: expanded local path with cross-attention and byte-residual bypass.

    Changes from Stage 0:
    - Within-patch encoder (2 bidir layers, d=640) replaces flat concat→project
    - Local decoder expanded to 4 layers, d=640, 10 heads, SwiGLU 1920
    - Cross-attention: decoder attends to all previous global patch states
    - Byte-residual bypass: r_t in R^160 injected at every decoder layer
    - Attentional pooling for patch summaries → project to d_global

    Global trunk (151M) is warm-started from Stage 0 and can be frozen.
    """

    def __init__(self, d_local=S1_D_LOCAL, d_global=D_GLOBAL,
                 n_local_heads=S1_N_LOCAL_HEADS, n_global_heads=N_GLOBAL_HEADS,
                 n_enc_layers=S1_N_ENC_LAYERS, n_dec_layers=S1_N_DEC_LAYERS,
                 n_global_layers=N_GLOBAL_LAYERS,
                 ff_local=S1_FF_LOCAL, ff_global=FF_GLOBAL,
                 bypass_dim=S1_BYPASS_DIM,
                 patch_size=PATCH_SIZE, max_seq_bytes=SEQ_BYTES,
                 n_bytes=N_BYTES):
        super().__init__()
        self.n_bytes = n_bytes
        self.d_local = d_local
        self.d_global = d_global
        self.patch_size = patch_size
        self.max_seq_bytes = max_seq_bytes
        self.max_patches = max_seq_bytes // patch_size
        self.bypass_dim = bypass_dim

        # Byte embedding (expanded to d_local=640)
        self.byte_embed = nn.Embedding(n_bytes, d_local)

        # Within-patch encoder (bidirectional, processes each patch of P bytes)
        self.patch_encoder = WithinPatchEncoder(d_local, n_local_heads, ff_local,
                                                max_patch_size=patch_size + 2,
                                                n_layers=n_enc_layers)
        self.patch_pool = AttentionalPooling(d_local)

        # Patch-to-global projection (pooled d_local → d_global)
        self.patch_proj = nn.Linear(d_local, d_global, bias=False)

        # Global transformer (warm-started from Stage 0)
        self.global_layers = nn.ModuleList([
            TransformerBlock(d_global, n_global_heads, ff_global)
            for _ in range(n_global_layers)
        ])
        self.global_norm = RMSNorm(d_global)

        # RoPE for global
        max_patches = max_seq_bytes // patch_size + 16
        g_head_dim = d_global // n_global_heads
        g_cos, g_sin = precompute_rope(g_head_dim, max_patches)
        self.register_buffer('g_rope_cos', g_cos, persistent=False)
        self.register_buffer('g_rope_sin', g_sin, persistent=False)

        # Global-to-local projection for shifted context
        self.global_to_local = nn.Linear(d_global, d_local, bias=False)

        # Byte-residual bypass: project byte_embed to bypass_dim
        self.bypass_proj = nn.Linear(d_local, bypass_dim, bias=False)

        # Local decoder (4 layers with cross-attention + bypass)
        self.local_decoder = nn.ModuleList([
            LocalDecoderBlockS1(d_local, n_local_heads, ff_local, d_global, bypass_dim)
            for _ in range(n_dec_layers)
        ])
        self.local_norm = RMSNorm(d_local)

        # RoPE for local
        max_bytes = max_seq_bytes + 16
        l_head_dim = d_local // n_local_heads
        l_cos, l_sin = precompute_rope(l_head_dim, max_bytes)
        self.register_buffer('l_rope_cos', l_cos, persistent=False)
        self.register_buffer('l_rope_sin', l_sin, persistent=False)

        # Output head (tied with byte_embed)
        self.output_head = nn.Linear(d_local, n_bytes, bias=False)
        self.output_head.weight = self.byte_embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Zero-init bypass output for near-identity start
        for block in self.local_decoder:
            nn.init.zeros_(block.bypass_u.weight)

    def load_stage0(self, ckpt_path):
        """Load global weights from a Stage 0 checkpoint, expanding local path.

        Also stores Stage 0's patch_proj for anchor loss during warm-start phase.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        s0 = ckpt['model']

        # Transfer global layers exactly
        for i in range(len(self.global_layers)):
            prefix = f'global_layers.{i}.'
            for k, v in s0.items():
                if k.startswith(prefix):
                    self.state_dict()[k].copy_(v)
        # Global norm
        self.global_norm.weight.data.copy_(s0['global_norm.weight'])

        # Expand byte embeddings: Stage 0 d_local=256 -> Stage 1 d_local=640
        s0_embed = s0['byte_embed.weight']  # (256, d_local_old)
        d_old = s0_embed.shape[1]
        self.byte_embed.weight.data[:, :d_old].copy_(s0_embed)
        nn.init.normal_(self.byte_embed.weight.data[:, d_old:], std=0.005)

        # Store Stage 0 weights for anchor loss (frozen — never updated)
        self._s0_patch_proj_weight = s0['patch_proj.weight'].clone()  # (d_global, P*d_old)
        self._s0_byte_embed = s0_embed.clone()  # (256, d_old) — frozen Stage 0 embeddings
        self._s0_d_old = d_old

        print(f"Loaded Stage 0 weights: global layers + byte_embed (expanded {d_old}->{self.d_local})")
        return ckpt.get('step', 0)

    def compute_anchor_loss(self, byte_ids, patch_emb_new):
        """Compute anchor loss: ||patch_emb_new - patch_emb_stage0||^2.

        Encourages new patch encoding path to produce similar inputs to global trunk
        as the Stage 0 concat-and-project path. Only used during Phase A warm-start.

        Uses FROZEN Stage 0 byte embeddings (stored at load time) so the target
        never drifts as the live byte_embed trains.

        Args:
            byte_ids: (B, T) raw byte IDs (used to look up frozen S0 embeddings)
            patch_emb_new: (B, N, d_global) from new encoder+pool+proj path
        Returns:
            scalar MSE loss
        """
        if not hasattr(self, '_s0_patch_proj_weight'):
            return torch.tensor(0.0, device=patch_emb_new.device)

        B, T = byte_ids.shape
        P = self.patch_size
        N = T // P
        d_old = self._s0_d_old

        # Use frozen Stage 0 embeddings (never updated during training)
        s0_embed = self._s0_byte_embed.to(byte_ids.device, dtype=patch_emb_new.dtype)
        byte_old = F.embedding(byte_ids, s0_embed)  # (B, T, d_old)
        byte_flat = byte_old.reshape(B, N, P * d_old)  # (B, N, P*d_old)
        w = self._s0_patch_proj_weight.to(byte_ids.device, dtype=patch_emb_new.dtype)
        patch_emb_s0 = F.linear(byte_flat, w)  # (B, N, d_global)

        return F.mse_loss(patch_emb_new, patch_emb_s0.detach())

    def freeze_global(self):
        """Freeze all global transformer parameters."""
        for p in self.global_layers.parameters():
            p.requires_grad = False
        self.global_norm.weight.requires_grad = False
        print("Global layers frozen")

    def unfreeze_global(self, layer_range=None):
        """Unfreeze global layers. layer_range=(start, end) for partial unfreeze."""
        if layer_range is None:
            for p in self.global_layers.parameters():
                p.requires_grad = True
            self.global_norm.weight.requires_grad = True
            print("All global layers unfrozen")
        else:
            start, end = layer_range
            for i in range(start, end):
                for p in self.global_layers[i].parameters():
                    p.requires_grad = True
            print(f"Global layers {start}-{end-1} unfrozen")

    def forward(self, byte_ids, targets=None, return_anchor=False):
        B, T = byte_ids.shape
        P = self.patch_size

        # Pad T to multiple of P
        pad = (P - T % P) % P
        if pad > 0:
            byte_ids = F.pad(byte_ids, (0, pad), value=0)
            if targets is not None:
                targets = F.pad(targets, (0, pad), value=-100)
        T_padded = byte_ids.shape[1]
        N = T_padded // P

        # 1. Byte embeddings
        byte_emb = self.byte_embed(byte_ids)  # (B, T_padded, d_local)

        # 2. Byte-residual bypass signal
        bypass_r = self.bypass_proj(byte_emb.detach())  # (B, T_padded, bypass_dim), detached

        # 3. Within-patch encoder: reshape to (B*N, P, d_local), encode, pool
        byte_patches = byte_emb.reshape(B * N, P, self.d_local)
        enc_out = self.patch_encoder(byte_patches)  # (B*N, P, d_local)
        patch_summary = self.patch_pool(enc_out)  # (B*N, d_local)
        patch_summary = patch_summary.reshape(B, N, self.d_local)

        # 4. Project patch summaries to global dimension
        patch_emb = self.patch_proj(patch_summary)  # (B, N, d_global)

        # 5. Global causal transformer over patches
        g_cos = self.g_rope_cos.to(patch_emb.device)
        g_sin = self.g_rope_sin.to(patch_emb.device)
        x = patch_emb
        for layer in self.global_layers:
            x = layer(x, g_cos, g_sin)
        global_out = self.global_norm(x)  # (B, N, d_global)

        # 6. Shift global output by 1 patch (causality)
        global_shifted = torch.cat([
            torch.zeros(B, 1, self.d_global, device=byte_ids.device, dtype=global_out.dtype),
            global_out[:, :-1, :]
        ], dim=1)  # (B, N, d_global)

        # 7. Project shifted global to d_local and expand to bytes
        global_local = self.global_to_local(global_shifted)  # (B, N, d_local)
        global_local_bytes = global_local.unsqueeze(2).expand(B, N, P, self.d_local)
        global_local_bytes = global_local_bytes.reshape(B, T_padded, self.d_local)

        # 8. Local decoder input: byte embeddings + projected global context
        local_input = byte_emb + global_local_bytes

        # 9. Local causal decoder with bypass
        l_cos = self.l_rope_cos.to(local_input.device)
        l_sin = self.l_rope_sin.to(local_input.device)
        x = local_input
        for layer in self.local_decoder:
            x = layer(x, l_cos, l_sin, bypass_r)
        x = self.local_norm(x)

        # 11. Output logits
        logits = self.output_head(x)  # (B, T_padded, n_bytes)

        if targets is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.n_bytes),
                targets.reshape(-1),
                ignore_index=-100,
            )
            if return_anchor:
                anchor_loss = self.compute_anchor_loss(byte_ids, patch_emb)
                return ce_loss, anchor_loss
            return ce_loss

        return logits[:, :T, :]

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        unique = total - self.output_head.weight.numel()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "unique_untied": unique, "trainable": trainable, "frozen": frozen}

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=256, temperature=0.8, top_k=50):
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor(prompt_bytes, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_new_bytes):
            if ids.shape[1] > self.max_seq_bytes:
                ids = ids[:, -self.max_seq_bytes:]
            logits = self.forward(ids)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_byte], dim=1)
        return ids[0].tolist()


# ---- Training ----

def get_lr(step, warmup, max_steps, max_lr, min_lr):
    """WSD schedule: warmup → stable → cosine decay (70/30 split)."""
    if step < warmup:
        return max_lr * step / warmup
    decay_start = int(max_steps * 0.7)
    if step < decay_start:
        return max_lr
    progress = (step - decay_start) / (max_steps - decay_start)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def eval_loss(model, dataset, n_batches=20, seq_len=SEQ_BYTES):
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = dataset.sample_batch(8, seq_len, device=device, split='test')
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y)
            total_loss += loss.item() * x.shape[0]
            total_count += x.shape[0]
    model.train()
    return total_loss / total_count


def train(cfg=None):
    """Train Sutra-Dyad Stage 0 flat-byte baseline."""
    if cfg is None:
        cfg = {}

    max_steps = cfg.get("max_steps", MAX_TRAIN_STEPS)
    batch_size = cfg.get("batch_size", BATCH_SIZE)
    grad_accum = cfg.get("grad_accum", GRAD_ACCUM)
    seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)
    lr = cfg.get("lr", LR)
    min_lr = cfg.get("min_lr", MIN_LR)
    warmup = cfg.get("warmup_steps", WARMUP_STEPS)
    run_name = cfg.get("run_name", "dyad_stage0")
    d_global = cfg.get("d_global", D_GLOBAL)
    d_local = cfg.get("d_local", D_LOCAL)
    n_global_layers = cfg.get("n_global_layers", N_GLOBAL_LAYERS)
    n_local_dec_layers = cfg.get("n_local_dec_layers", N_LOCAL_DEC_LAYERS)
    patch_size = cfg.get("patch_size", PATCH_SIZE)

    print(f"\n{'='*60}")
    print(f"SUTRA-DYAD STAGE 0 — Flat-Byte Baseline")
    print(f"{'='*60}")

    # Dataset
    dataset = ByteShardedDataset()
    print(f"Dataset: {dataset.total_bytes / 1e9:.1f}B est bytes, {len(dataset.index)} shards")

    # Model
    model = SutraDyad(
        d_global=d_global, d_local=d_local,
        n_global_layers=n_global_layers,
        n_local_dec_layers=n_local_dec_layers,
        patch_size=patch_size, max_seq_bytes=seq_bytes,
    )
    params = model.count_params()
    print(f"Model params: {params['total']/1e6:.1f}M total, {params['unique_untied']/1e6:.1f}M unique")
    model = model.to(DEVICE)

    # Optimizer: AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                   weight_decay=WD, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    # Checkpoint dir
    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_path = REPO / "results" / f"{run_name}.log"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Config: batch={batch_size}, grad_accum={grad_accum}, seq={seq_bytes}, "
        f"patch_size={patch_size}, lr={lr}, warmup={warmup}, max_steps={max_steps}")
    log(f"Patches per seq: {seq_bytes // patch_size}")
    log(f"Effective batch: {batch_size * grad_accum} sequences = "
        f"{batch_size * grad_accum * seq_bytes / 1e3:.0f}K bytes/step")

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    best_eval = float('inf')
    step = 0
    t0 = time.time()
    running_loss = 0.0
    nan_count = 0

    log(f"\nStarting training at {time.strftime('%H:%M:%S')}")

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(grad_accum):
            x, y = dataset.sample_batch(batch_size, seq_bytes, device=DEVICE, split='train')
            with torch.amp.autocast('cuda', dtype=DTYPE):
                loss = model(x, y)
                loss_scaled = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                log(f"  [NaN/Inf at step {step}, micro {micro}] — skipping")
                if nan_count > 10:
                    log("FATAL: >10 NaN losses, aborting")
                    return
                continue

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()

        current_lr = get_lr(step, warmup, max_steps, lr, min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = current_lr

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        avg_loss = accum_loss / grad_accum
        running_loss += avg_loss
        step += 1

        if step % LOG_EVERY == 0:
            rl = running_loss / LOG_EVERY
            elapsed = time.time() - t0
            throughput = step * batch_size * grad_accum * seq_bytes / elapsed
            bpb = rl / math.log(2)
            vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            log(f"  step {step:>5d} | loss {rl:.4f} | BPB {bpb:.3f} | "
                f"lr {current_lr:.2e} | grad {grad_norm:.2f} | "
                f"{throughput/1e6:.1f}MB/s | VRAM {vram:.1f}G")
            running_loss = 0.0

        if step % EVAL_EVERY == 0:
            el = eval_loss(model, dataset, n_batches=30, seq_len=seq_bytes)
            bpb_eval = el / math.log(2)
            log(f"  *** EVAL step {step}: loss={el:.4f}, BPB={bpb_eval:.3f}")
            if el < best_eval:
                best_eval = el
                torch.save({
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "eval_loss": el,
                    "config": cfg,
                }, ckpt_dir / "best.pt")
                log(f"      New best! Saved to {ckpt_dir / 'best.pt'}")

            # Generation sample
            prompt = b"The meaning of intelligence is"
            try:
                gen_ids = model.generate(list(prompt), max_new_bytes=128, temperature=0.8)
                gen_text = bytes(gen_ids).decode('utf-8', errors='replace')
                safe = gen_text.encode('ascii', errors='replace').decode('ascii')
                log(f"      GEN: {safe[:200]}")
            except Exception as e:
                log(f"      GEN failed: {e}")

        if step % ROLLING_SAVE == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "eval_loss": best_eval,
                "config": cfg,
            }, ckpt_dir / f"step_{step}.pt")

    final_loss = eval_loss(model, dataset, n_batches=50, seq_len=seq_bytes)
    final_bpb = final_loss / math.log(2)
    log(f"\n{'='*60}")
    log(f"TRAINING COMPLETE — {max_steps} steps")
    log(f"Final eval loss: {final_loss:.4f}, BPB: {final_bpb:.3f}")
    log(f"Best eval loss: {best_eval:.4f}, BPB: {best_eval/math.log(2):.3f}")
    log(f"{'='*60}")

    log_f.close()

    results = {
        "run_name": run_name,
        "final_eval_loss": final_loss,
        "final_bpb": final_bpb,
        "best_eval_loss": best_eval,
        "best_bpb": best_eval / math.log(2),
        "steps": max_steps,
        "config": cfg,
        "params": params,
    }
    with open(REPO / "results" / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)


def det_eval(ckpt_path):
    """Deterministic evaluation of a checkpoint (supports both Stage 0 and Stage 1)."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    stage = ckpt.get("stage", 0)

    if stage == 1:
        model = SutraDyadS1(max_seq_bytes=cfg.get("seq_bytes", SEQ_BYTES))
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model = SutraDyad(
            d_global=cfg.get("d_global", D_GLOBAL),
            d_local=cfg.get("d_local", D_LOCAL),
            n_global_layers=cfg.get("n_global_layers", N_GLOBAL_LAYERS),
            n_local_dec_layers=cfg.get("n_local_dec_layers", N_LOCAL_DEC_LAYERS),
            patch_size=cfg.get("patch_size", PATCH_SIZE),
            max_seq_bytes=cfg.get("seq_bytes", SEQ_BYTES),
        )
        model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE)
    params = model.count_params()
    print(f"Model: {params['total']/1e6:.1f}M params (Stage {stage})")

    dataset = ByteShardedDataset()
    el = eval_loss(model, dataset, n_batches=100)
    bpb = el / math.log(2)
    print(f"Eval loss: {el:.4f}, BPB: {bpb:.3f}")
    print(f"Step: {ckpt.get('step', '?')}")

    prompts = [
        b"The meaning of intelligence is",
        b"In the beginning, there was",
        b"def fibonacci(n):\n",
    ]
    for prompt in prompts:
        gen_ids = model.generate(list(prompt), max_new_bytes=200, temperature=0.7)
        gen_text = bytes(gen_ids).decode('utf-8', errors='replace')
        safe = gen_text.encode('ascii', errors='replace').decode('ascii')
        print(f"\n--- Prompt: {prompt.decode()} ---")
        print(safe[:300])


def ablation_eval(ckpt_path):
    """Run ablation study: zero out each Stage 1 component and measure BPB impact.

    Tests: cross-attention, bypass, global context, and encoder (vs raw embed).
    Reports BPB for each ablation to quantify component contribution.
    """
    import functools

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt.get("config", {})
    stage = ckpt.get("stage", 0)
    assert stage == 1, "Ablation only for Stage 1 checkpoints"

    model = SutraDyadS1(max_seq_bytes=cfg.get("seq_bytes", SEQ_BYTES))
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(DEVICE)
    params = model.count_params()
    print(f"Model: {params['total']/1e6:.1f}M params (Stage {stage}), step {ckpt.get('step', '?')}")

    dataset = ByteShardedDataset()
    n_eval = 50  # batches for reliable estimate

    results = {}

    # 1. Baseline (no ablation)
    print("\n[1/3] Baseline (no ablation)...")
    el = eval_loss(model, dataset, n_batches=n_eval)
    results["baseline"] = el / math.log(2)
    print(f"  BPB = {results['baseline']:.4f}")

    # 2. Ablate bypass (zero bypass_r input)
    print("\n[2/3] Ablating byte-residual bypass (zero bypass)...")
    orig_forwards = []
    for i, block in enumerate(model.local_decoder):
        orig_forwards.append(block.forward)
        def make_no_bypass_forward(blk, orig_fwd):
            def no_bypass_forward(x, cos, sin, bypass_r):
                zero_bypass = torch.zeros_like(bypass_r)
                return orig_fwd(x, cos, sin, zero_bypass)
            return no_bypass_forward
        block.forward = make_no_bypass_forward(block, block.forward)
    el = eval_loss(model, dataset, n_batches=n_eval)
    results["no_bypass"] = el / math.log(2)
    print(f"  BPB = {results['no_bypass']:.4f} (delta = +{results['no_bypass'] - results['baseline']:.4f})")
    for i, block in enumerate(model.local_decoder):
        block.forward = orig_forwards[i]

    # 3. Ablate global context (zero the shifted global in local input)
    print("\n[3/3] Ablating global context (zero global->local projection)...")
    orig_g2l = model.global_to_local.weight.data.clone()
    model.global_to_local.weight.data.zero_()
    el = eval_loss(model, dataset, n_batches=n_eval)
    results["no_global_ctx"] = el / math.log(2)
    print(f"  BPB = {results['no_global_ctx']:.4f} (delta = +{results['no_global_ctx'] - results['baseline']:.4f})")
    model.global_to_local.weight.data.copy_(orig_g2l)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'BPB':>8} {'Delta':>8} {'% Impact':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*10}")
    base = results["baseline"]
    for name, bpb in results.items():
        delta = bpb - base
        pct = (delta / base * 100) if base > 0 else 0
        print(f"{name:<25} {bpb:>8.4f} {delta:>+8.4f} {pct:>+9.1f}%")
    print(f"{'='*60}")

    out_path = REPO / "results" / "ablation_s1_phase_b.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


def profile_teachers(n_sequences=20, max_bytes=1024):
    """Ekalavya pre-requisite probe: profile teacher models for byte-level KD.

    For each teacher:
    - Token compression ratio (tokens per 1K bytes)
    - Per-byte next-byte prediction accuracy
    - Prediction entropy (signal quality)
    - Byte offsets per token (alignment density)

    Runs entirely on CPU. GPU reserved for training.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import numpy as np

    TEACHERS = {
        "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
        "pythia-1.4b": "EleutherAI/pythia-1.4b",
        "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    }

    # Load shared eval sequences (raw bytes from our training data)
    dataset = ByteShardedDataset()
    sequences = []
    for _ in range(n_sequences):
        x, y = dataset.sample_batch(1, max_bytes, device="cpu", split="test")
        sequences.append(x[0].tolist())

    all_teacher_byte_preds = {}  # name -> list of (seq_idx, byte_idx, predicted_byte_prob)
    teacher_profiles = {}

    for name, model_id in TEACHERS.items():
        print(f"\n{'='*50}")
        print(f"Profiling: {name} ({model_id})")
        print(f"{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, trust_remote_code=True,
        )
        model.eval()

        entropies = []
        token_counts = []
        byte_accuracies = []
        byte_probs_all = []  # per-byte teacher confidence in correct next byte

        for si, seq_bytes in enumerate(sequences):
            # Convert byte list to text (teacher expects text, not raw bytes)
            raw = bytes(seq_bytes)
            text = raw.decode("utf-8", errors="replace")
            if len(text) < 10:
                continue

            # Tokenize
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"]
            n_tok = input_ids.shape[1]
            token_counts.append(n_tok)

            # Get byte boundaries for each token
            token_byte_offsets = []
            byte_pos = 0
            for tid in input_ids[0].tolist():
                tok_str = tokenizer.decode([tid])
                tok_bytes = tok_str.encode("utf-8")
                token_byte_offsets.append((byte_pos, byte_pos + len(tok_bytes)))
                byte_pos += len(tok_bytes)

            # Forward pass (CPU)
            with torch.no_grad():
                out = model(input_ids=input_ids)

            logits = out.logits[0]  # (T, V)
            probs = F.softmax(logits, dim=-1)

            # Per-token entropy
            ent = -(probs * torch.log(probs + 1e-10)).sum(-1)
            entropies.append(ent.mean().item())

            # Project teacher predictions to byte level:
            # For each token position t, teacher predicts token t+1.
            # Token t+1 spans bytes [a, b). The byte-level signal is the
            # probability the teacher assigns to the correct next token.
            for t in range(n_tok - 1):
                target_id = input_ids[0, t + 1].item()
                p_correct = probs[t, target_id].item()
                byte_probs_all.append(p_correct)
                # Also check if top-1 prediction is correct
                byte_accuracies.append(int(probs[t].argmax().item() == target_id))

            if (si + 1) % 5 == 0:
                print(f"  Processed {si + 1}/{n_sequences} sequences")

        # Summary
        profile = {
            "model_id": model_id,
            "hidden_dim": model.config.hidden_size,
            "n_sequences": len(token_counts),
            "avg_tokens_per_1k_bytes": float(np.mean(token_counts)) / max_bytes * 1000
                if token_counts else 0,
            "mean_entropy": float(np.mean(entropies)) if entropies else 0,
            "std_entropy": float(np.std(entropies)) if entropies else 0,
            "top1_accuracy": float(np.mean(byte_accuracies)) if byte_accuracies else 0,
            "mean_correct_prob": float(np.mean(byte_probs_all)) if byte_probs_all else 0,
            "median_correct_prob": float(np.median(byte_probs_all)) if byte_probs_all else 0,
        }
        teacher_profiles[name] = profile
        all_teacher_byte_preds[name] = byte_probs_all

        print(f"  Hidden dim: {profile['hidden_dim']}")
        print(f"  Tokens/1K bytes: {profile['avg_tokens_per_1k_bytes']:.1f}")
        print(f"  Mean entropy: {profile['mean_entropy']:.3f}")
        print(f"  Top-1 accuracy: {profile['top1_accuracy']:.3f}")
        print(f"  Mean P(correct): {profile['mean_correct_prob']:.3f}")

        del model, tokenizer
        gc.collect()

    # Cross-teacher agreement: for overlapping byte predictions, how often do
    # teachers agree on top-1? (Only for teachers with same sequence count)
    names = list(all_teacher_byte_preds.keys())
    agreement = {}
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            p1 = all_teacher_byte_preds[n1]
            p2 = all_teacher_byte_preds[n2]
            min_len = min(len(p1), len(p2))
            if min_len > 0:
                # Correlation of confidence scores
                corr = float(np.corrcoef(p1[:min_len], p2[:min_len])[0, 1])
                agreement[f"{n1}_vs_{n2}"] = {
                    "confidence_correlation": corr,
                    "n_compared": min_len,
                }

    result = {
        "profiles": teacher_profiles,
        "cross_teacher_agreement": agreement,
        "config": {"n_sequences": n_sequences, "max_bytes": max_bytes},
    }

    out_path = REPO / "results" / "teacher_profiles.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("TEACHER PROFILE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Teacher':<18} {'HidDim':>7} {'Tok/KB':>7} {'Entropy':>8} {'Top1':>6} {'P(cor)':>7}")
    print(f"{'-'*18} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*7}")
    for name, p in teacher_profiles.items():
        print(f"{name:<18} {p['hidden_dim']:>7} {p['avg_tokens_per_1k_bytes']:>7.1f} "
              f"{p['mean_entropy']:>8.3f} {p['top1_accuracy']:>6.3f} {p['mean_correct_prob']:>7.3f}")
    if agreement:
        print(f"\nCross-teacher confidence correlation:")
        for pair, data in agreement.items():
            print(f"  {pair}: r={data['confidence_correlation']:.3f} (n={data['n_compared']})")
    print(f"{'='*70}")

    return result


def train_s1(stage0_ckpt, cfg=None):
    """Train Sutra-Dyad Stage 1: expanded local path with cross-attention."""
    if cfg is None:
        cfg = {}

    max_steps = cfg.get("max_steps", 1000)
    batch_size = cfg.get("batch_size", 48)
    grad_accum = cfg.get("grad_accum", 3)
    seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)
    lr_local = cfg.get("lr_local", 3e-4)
    lr_global = cfg.get("lr_global", 6e-5)
    min_lr_local = cfg.get("min_lr_local", 3e-5)
    min_lr_global = cfg.get("min_lr_global", 6e-6)
    warmup = cfg.get("warmup_steps", 100)
    run_name = cfg.get("run_name", "dyad_s1_probe")
    freeze_global = cfg.get("freeze_global", True)
    anchor_weight = cfg.get("anchor_weight", 0.1)  # Weight for anchor loss during warm-start
    anchor_steps = cfg.get("anchor_steps", max_steps)  # Steps to use anchor loss (all by default)

    print(f"\n{'='*60}")
    print(f"SUTRA-DYAD STAGE 1 -- Expanded Local Path")
    print(f"{'='*60}")

    # Dataset
    dataset = ByteShardedDataset()
    print(f"Dataset: {dataset.total_bytes / 1e9:.1f}B est bytes, {len(dataset.index)} shards")

    # Model
    model = SutraDyadS1(max_seq_bytes=seq_bytes)
    s0_step = model.load_stage0(stage0_ckpt)
    print(f"Loaded Stage 0 checkpoint from step {s0_step}")

    if freeze_global:
        model.freeze_global()

    unfreeze_layers = cfg.get("unfreeze_layers")
    if unfreeze_layers:
        model.unfreeze_global(layer_range=tuple(unfreeze_layers))

    params = model.count_params()
    print(f"Model params: {params['total']/1e6:.1f}M total, "
          f"{params['trainable']/1e6:.1f}M trainable, {params['frozen']/1e6:.1f}M frozen")
    model = model.to(DEVICE)

    # Optimizer with per-group LR
    global_params = []
    local_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'global_layers' in name or 'global_norm' in name:
            global_params.append(p)
        else:
            local_params.append(p)

    param_groups = []
    if local_params:
        param_groups.append({"params": local_params, "lr": lr_local, "name": "local"})
    if global_params:
        param_groups.append({"params": global_params, "lr": lr_global, "name": "global"})

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=WD, fused=True)
    scaler = torch.amp.GradScaler('cuda')

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / "results" / f"{run_name}.log"
    log_f = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"Config: batch={batch_size}, grad_accum={grad_accum}, seq={seq_bytes}, "
        f"lr_local={lr_local}, lr_global={lr_global}, warmup={warmup}, max_steps={max_steps}")
    log(f"Stage 0 checkpoint: {stage0_ckpt} (step {s0_step})")
    log(f"Global frozen: {freeze_global}")
    log(f"Effective batch: {batch_size * grad_accum} sequences = "
        f"{batch_size * grad_accum * seq_bytes / 1e3:.0f}K bytes/step")

    # Resume from S1 checkpoint if provided
    start_step = 0
    resume_ckpt = cfg.get("resume_ckpt")
    if resume_ckpt:
        rk = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(rk["model"], strict=False)
        # Only restore optimizer if param groups match (may differ if unfreeze config changed)
        saved_n_groups = len(rk["optimizer"]["param_groups"])
        curr_n_groups = len(optimizer.param_groups)
        if saved_n_groups == curr_n_groups:
            try:
                optimizer.load_state_dict(rk["optimizer"])
                if "scaler" in rk:
                    scaler.load_state_dict(rk["scaler"])
                log(f"Restored optimizer state from checkpoint")
            except Exception as e:
                log(f"Optimizer state mismatch, starting fresh optimizer: {e}")
        else:
            log(f"Optimizer param groups changed ({saved_n_groups}->{curr_n_groups}), fresh optimizer")
        start_step = rk["step"]
        best_eval = rk.get("eval_loss", float('inf'))
        log(f"Resumed from {resume_ckpt} at step {start_step}, best_eval={best_eval:.4f}")

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    if not resume_ckpt:
        best_eval = float('inf')
    step = start_step
    t0 = time.time()
    running_loss = 0.0
    nan_count = 0

    log(f"\nStarting Stage 1 training at {time.strftime('%H:%M:%S')} (step {step})")

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        use_anchor = anchor_weight > 0 and step < anchor_steps

        for micro in range(grad_accum):
            x, y = dataset.sample_batch(batch_size, seq_bytes, device=DEVICE, split='train')
            with torch.amp.autocast('cuda', dtype=DTYPE):
                if use_anchor:
                    ce_loss, anc_loss = model(x, y, return_anchor=True)
                    loss = ce_loss + anchor_weight * anc_loss
                else:
                    loss = model(x, y)
                loss_scaled = loss / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                log(f"  [NaN/Inf at step {step}, micro {micro}] -- skipping")
                if nan_count > 10:
                    log("FATAL: >10 NaN losses, aborting")
                    return
                continue

            scaler.scale(loss_scaled).backward()
            accum_loss += loss.item()

        # Per-group LR schedule
        for pg in optimizer.param_groups:
            if pg.get("name") == "global":
                pg['lr'] = get_lr(step, warmup, max_steps, lr_global, min_lr_global)
            else:
                pg['lr'] = get_lr(step, warmup, max_steps, lr_local, min_lr_local)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        avg_loss = accum_loss / grad_accum
        running_loss += avg_loss
        step += 1

        if step % LOG_EVERY == 0:
            rl = running_loss / LOG_EVERY
            elapsed = time.time() - t0
            throughput = step * batch_size * grad_accum * seq_bytes / elapsed
            bpb = rl / math.log(2)
            vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            local_lr = optimizer.param_groups[0]['lr']
            log(f"  step {step:>5d} | loss {rl:.4f} | BPB {bpb:.3f} | "
                f"lr {local_lr:.2e} | grad {grad_norm:.2f} | "
                f"{throughput/1e6:.1f}MB/s | VRAM {vram:.1f}G")
            running_loss = 0.0

        if step % EVAL_EVERY == 0:
            el = eval_loss(model, dataset, n_batches=30, seq_len=seq_bytes)
            bpb_eval = el / math.log(2)
            log(f"  *** EVAL step {step}: loss={el:.4f}, BPB={bpb_eval:.3f}")
            if el < best_eval:
                best_eval = el
                torch.save({
                    "step": step,
                    "stage": 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "eval_loss": el,
                    "config": cfg,
                }, ckpt_dir / "best.pt")
                log(f"      New best! Saved to {ckpt_dir / 'best.pt'}")

            prompt = b"The meaning of intelligence is"
            try:
                gen_ids = model.generate(list(prompt), max_new_bytes=128, temperature=0.8)
                gen_text = bytes(gen_ids).decode('utf-8', errors='replace')
                safe = gen_text.encode('ascii', errors='replace').decode('ascii')
                log(f"      GEN: {safe[:200]}")
            except Exception as e:
                log(f"      GEN failed: {e}")

        if step % ROLLING_SAVE == 0:
            torch.save({
                "step": step, "stage": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "eval_loss": best_eval, "config": cfg,
            }, ckpt_dir / f"step_{step}.pt")

    final_loss = eval_loss(model, dataset, n_batches=50, seq_len=seq_bytes)
    final_bpb = final_loss / math.log(2)
    log(f"\n{'='*60}")
    log(f"STAGE 1 TRAINING COMPLETE -- {max_steps} steps")
    log(f"Final eval loss: {final_loss:.4f}, BPB: {final_bpb:.3f}")
    log(f"Best eval loss: {best_eval:.4f}, BPB: {best_eval/math.log(2):.3f}")
    log(f"{'='*60}")

    log_f.close()

    results = {
        "run_name": run_name,
        "stage": 1,
        "final_eval_loss": final_loss,
        "final_bpb": final_bpb,
        "best_eval_loss": best_eval,
        "best_bpb": best_eval / math.log(2),
        "steps": max_steps,
        "config": cfg,
        "params": params,
        "stage0_ckpt": str(stage0_ckpt),
    }
    with open(REPO / "results" / f"{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sutra-Dyad")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max-steps", type=int, default=MAX_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-bytes", type=int, default=SEQ_BYTES)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--det-eval", type=str, default=None)
    parser.add_argument("--ablate", type=str, default=None, help="Run ablation eval on S1 checkpoint")
    parser.add_argument("--profile-teachers", action="store_true",
                        help="Profile teacher models for Ekalavya KD (CPU-only)")
    parser.add_argument("--config", type=str, default=None, help="JSON config file")
    parser.add_argument("--stage0-ckpt", type=str, default=None, help="Stage 0 checkpoint for Stage 1")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from S1 checkpoint")
    parser.add_argument("--freeze-global", action="store_true", default=True,
                        help="Freeze global trunk (default: True for warm-start)")
    parser.add_argument("--no-freeze-global", dest="freeze_global", action="store_false",
                        help="Unfreeze global trunk")
    parser.add_argument("--unfreeze-layers", type=int, nargs=2, default=None,
                        metavar=("START", "END"),
                        help="Partially unfreeze global layers [START, END) after initial freeze")
    args = parser.parse_args()

    if args.det_eval:
        det_eval(args.det_eval)
    elif args.ablate:
        ablation_eval(args.ablate)
    elif args.profile_teachers:
        profile_teachers()
    elif args.stage == 1:
        if args.stage0_ckpt is None:
            print("ERROR: --stage0-ckpt required for Stage 1 (needed for anchor loss even when resuming)")
            sys.exit(1)
        cfg = {}
        if args.config:
            with open(args.config) as f:
                cfg = json.load(f)
        cfg.setdefault("max_steps", args.max_steps)
        cfg.setdefault("batch_size", args.batch_size or 48)
        cfg.setdefault("seq_bytes", args.seq_bytes)
        cfg.setdefault("run_name", args.run_name or "dyad_s1_probe")
        cfg.setdefault("freeze_global", args.freeze_global)
        if args.lr:
            cfg.setdefault("lr_local", args.lr)
        if args.resume:
            cfg["resume_ckpt"] = args.resume
        if args.unfreeze_layers:
            cfg["unfreeze_layers"] = args.unfreeze_layers
        train_s1(args.stage0_ckpt, cfg)
    else:
        cfg = {}
        if args.config:
            with open(args.config) as f:
                cfg = json.load(f)
        cfg.setdefault("max_steps", args.max_steps)
        cfg.setdefault("batch_size", args.batch_size or BATCH_SIZE)
        cfg.setdefault("seq_bytes", args.seq_bytes)
        cfg.setdefault("lr", args.lr or LR)
        cfg.setdefault("run_name", args.run_name or "dyad_stage0")
        train(cfg)
