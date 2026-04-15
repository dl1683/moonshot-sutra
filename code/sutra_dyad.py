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
LOG_EVERY = 10
ROLLING_SAVE = 250
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

    def forward(self, byte_ids, targets=None, return_anchor=False, return_internals=False):
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
            if return_internals:
                return ce_loss, {
                    "logits": logits[:, :T, :],
                    "global_out": global_out,        # (B, N, d_global) pre-shift
                    "global_shifted": global_shifted, # (B, N, d_global) post-shift
                    "global_local": global_local,    # (B, N, d_local) post-projection bottleneck
                }
            return ce_loss

        if return_internals:
            return logits[:, :T, :], {
                "logits": logits[:, :T, :],
                "global_out": global_out,
                "global_shifted": global_shifted,
                "global_local": global_local,
            }
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


# ---- Ekalavya: Multi-Teacher Knowledge Distillation ----

def _build_first_byte_map(tokenizer, vocab_size, device='cpu'):
    """Map each token ID to its first UTF-8 byte. Returns tensor of shape (vocab_size,)."""
    fb = torch.zeros(vocab_size, dtype=torch.long, device=device)
    for tok_id in range(min(tokenizer.vocab_size, vocab_size)):
        try:
            decoded = tokenizer.decode([tok_id])
            if decoded:
                fb[tok_id] = decoded.encode('utf-8', errors='replace')[0]
        except Exception:
            pass
    return fb


def _teacher_byte_probs(logits, first_byte_map):
    """Convert teacher token logits to byte-level probabilities via first-byte marginal.

    DEPRECATED: This destroys 84% of teacher signal. Use covering decomposition instead.
    Kept for backward compatibility with legacy configs.

    Args:
        logits: (T_tok, V) raw teacher logits
        first_byte_map: (V,) mapping token_id -> first byte value

    Returns:
        byte_probs: (T_tok, 256) normalized byte probability distribution
    """
    probs = F.softmax(logits.float(), dim=-1)
    bp = torch.zeros(probs.shape[0], 256, device=probs.device, dtype=torch.float32)
    for b in range(256):
        mask = (first_byte_map == b)
        if mask.any():
            bp[:, b] = probs[:, mask].sum(dim=-1)
    bp = bp / (bp.sum(dim=-1, keepdim=True) + 1e-10)
    return bp


# ---- Covering Decomposition (Phan et al. ICLR 2025, lossless byte alignment) ----

def _build_covering_tables(tokenizer, vocab_size, device='cpu'):
    """Build prefix-to-token index for covering decomposition.

    For each byte prefix (tuple of bytes), stores which token IDs have that prefix
    in their UTF-8 byte representation. This enables computing P(b_{k+1} | b_1...b_k)
    from teacher token probabilities with zero information loss.

    Optimized: pre-builds first-byte sum matrix (256, V) for O(1) depth-0 computation,
    and pre-moves all index tensors to target device to avoid inner-loop transfers.

    Returns dict with:
        token_byte_seqs: list of byte tuples, indexed by token_id
        prefix_to_indices: dict mapping byte-prefix tuple -> tensor of token indices (on device)
        prefix_children: dict mapping prefix -> dict of {next_byte: extended_prefix}
        first_byte_matrix: (256, V) float32 tensor for depth-0 matmul (on device)
        max_token_bytes: maximum byte length of any token
        n_prefixes: total number of unique prefixes
        device: target device
    """
    token_byte_seqs = []
    prefix_to_idx_lists = {}  # prefix bytes tuple -> [tok_id, ...]

    for tok_id in range(min(tokenizer.vocab_size, vocab_size)):
        try:
            decoded = tokenizer.decode([tok_id])
            if decoded:
                b = tuple(decoded.encode('utf-8', errors='replace'))
            else:
                b = ()
        except Exception:
            b = ()
        token_byte_seqs.append(b)

        # Register this token under all its byte prefixes
        for depth in range(len(b) + 1):
            prefix = b[:depth]
            if prefix not in prefix_to_idx_lists:
                prefix_to_idx_lists[prefix] = []
            prefix_to_idx_lists[prefix].append(tok_id)

    # Pad for any token IDs beyond tokenizer.vocab_size
    while len(token_byte_seqs) < vocab_size:
        token_byte_seqs.append(())

    max_token_bytes = max((len(b) for b in token_byte_seqs if b), default=1)

    # Convert lists to LongTensors on target device
    prefix_to_indices = {}
    for prefix, indices in prefix_to_idx_lists.items():
        prefix_to_indices[prefix] = torch.tensor(indices, dtype=torch.long, device=device)

    # Build children map: for each prefix, which next bytes lead to valid extended prefixes
    prefix_children = {}
    for prefix in prefix_to_indices:
        if len(prefix) > 0:
            parent = prefix[:-1]
            last_byte = prefix[-1]
            if parent not in prefix_children:
                prefix_children[parent] = {}
            prefix_children[parent][last_byte] = prefix

    # Pre-build first-byte sum matrix (256, V) for depth-0: P(first_byte=b) = M[b,:] @ P(tokens)
    first_byte_matrix = torch.zeros(256, vocab_size, device=device, dtype=torch.float32)
    for tok_id, bseq in enumerate(token_byte_seqs):
        if bseq:
            first_byte_matrix[bseq[0], tok_id] = 1.0

    # Pre-build per-prefix child scatter matrices for common shallow prefixes (depth <= 2)
    # This avoids the inner loop over children for the most frequent cases
    prefix_child_indices = {}  # prefix -> (child_bytes tensor, list of index tensors)
    for prefix, children in prefix_children.items():
        if len(prefix) <= 3:  # cache depth 0, 1, 2, 3 prefixes
            child_bytes = sorted(children.keys())
            child_idx_list = [prefix_to_indices[children[b]] for b in child_bytes]
            prefix_child_indices[prefix] = (
                torch.tensor(child_bytes, dtype=torch.long, device=device),
                child_idx_list,
            )

    # Build numpy versions for fast CPU covering computation (avoids GPU kernel overhead)
    import numpy as np
    first_byte_matrix_np = first_byte_matrix.cpu().numpy()
    prefix_to_indices_np = {}
    for prefix, idx_tensor in prefix_to_indices.items():
        prefix_to_indices_np[prefix] = idx_tensor.cpu().numpy()

    return {
        "token_byte_seqs": token_byte_seqs,
        "prefix_to_indices": prefix_to_indices,
        "prefix_to_indices_np": prefix_to_indices_np,
        "prefix_children": prefix_children,
        "prefix_child_indices": prefix_child_indices,
        "first_byte_matrix": first_byte_matrix,
        "first_byte_matrix_np": first_byte_matrix_np,
        "max_token_bytes": max_token_bytes,
        "n_prefixes": len(prefix_to_indices),
        "device": device,
    }


def _covering_byte_conditionals_np(token_probs_np, observed_bytes, covering_tables_np):
    """Compute covering byte conditionals for one teacher position. NUMPY-ONLY (CPU).

    Args:
        token_probs_np: (V,) numpy array, teacher probability distribution
        observed_bytes: tuple of bytes for the actual next token
        covering_tables_np: dict with numpy index arrays from _build_covering_tables

    Returns:
        conditionals: list of (256,) numpy arrays
    """
    import numpy as np
    prefix_to_indices_np = covering_tables_np["prefix_to_indices_np"]
    prefix_children = covering_tables_np["prefix_children"]
    first_byte_matrix_np = covering_tables_np["first_byte_matrix_np"]
    conditionals = []

    for k in range(len(observed_bytes)):
        prefix = observed_bytes[:k]

        if k == 0:
            cond = first_byte_matrix_np @ token_probs_np  # (256,)
            cond_sum = cond.sum()
            if cond_sum > 1e-10:
                cond = cond / cond_sum
            conditionals.append(cond)
            continue

        if prefix in prefix_to_indices_np:
            normalizer = token_probs_np[prefix_to_indices_np[prefix]].sum()
        else:
            normalizer = 0.0

        if normalizer < 1e-10:
            cond = np.zeros(256, dtype=np.float32)
            cond[observed_bytes[k]] = 1.0
            conditionals.append(cond)
            continue

        cond = np.zeros(256, dtype=np.float32)
        children = prefix_children.get(prefix, {})
        for next_byte, ext_prefix in children.items():
            if ext_prefix in prefix_to_indices_np:
                cond[next_byte] = token_probs_np[prefix_to_indices_np[ext_prefix]].sum() / normalizer

        cond_sum = cond.sum()
        if cond_sum > 1e-10:
            cond = cond / cond_sum

        conditionals.append(cond)

    return conditionals


def _covering_one_sequence(b, probs_np_b, token_ids_b, raw_bytes_b,
                           token_byte_seqs, first_byte_matrix_np,
                           covering_tables, max_depth):
    """CPU-only covering computation for one sequence. Thread-safe (numpy only).

    Extracted from _get_teacher_targets_covering_batched for parallel execution.
    Returns (byte_probs_np, byte_mask_np, byte_offsets, coverage_ratio).
    """
    import numpy as np
    byte_offsets = []
    pos = 0
    for tid in token_ids_b:
        byte_offsets.append(pos)
        pos += len(token_byte_seqs[tid]) if tid < len(token_byte_seqs) else 1
    byte_offsets.append(pos)

    n_bytes = len(raw_bytes_b)
    byte_probs_np = np.zeros((n_bytes, 256), dtype=np.float32)
    byte_mask_np = np.zeros(n_bytes, dtype=np.bool_)
    supervised_count = 0

    # Vectorized depth-0
    all_depth0 = probs_np_b @ first_byte_matrix_np.T
    depth0_sums = all_depth0.sum(axis=1, keepdims=True)
    depth0_sums = np.maximum(depth0_sums, 1e-10)
    all_depth0 = all_depth0 / depth0_sums

    for i in range(len(token_ids_b) - 1):
        next_tid = token_ids_b[i + 1]
        if next_tid >= len(token_byte_seqs):
            continue
        next_bytes = token_byte_seqs[next_tid]
        if not next_bytes:
            continue

        token_start = byte_offsets[i + 1]
        byte_pos = token_start - 1
        if 0 <= byte_pos < n_bytes:
            byte_probs_np[byte_pos] = all_depth0[i]
            byte_mask_np[byte_pos] = True
            supervised_count += 1

        effective_depth = len(next_bytes) if max_depth is None else min(len(next_bytes), max_depth + 1)
        if effective_depth > 1:
            tok_probs_np = probs_np_b[i]
            conditionals = _covering_byte_conditionals_np(
                tok_probs_np, next_bytes[:effective_depth], covering_tables)
            for j in range(1, len(conditionals)):
                bp = token_start + j - 1
                if 0 <= bp < n_bytes:
                    byte_probs_np[bp] = conditionals[j]
                    byte_mask_np[bp] = True
                    supervised_count += 1

    coverage_ratio = supervised_count / max(n_bytes, 1)
    return byte_probs_np, byte_mask_np, byte_offsets, coverage_ratio


def _get_teacher_targets_covering_batched(teacher, tokenizer, covering_tables,
                                         batch_raw_bytes, device, temperature=2.0,
                                         extract_hidden=True, max_depth=None):
    """Batched version: single teacher forward for all sequences in micro-batch.

    5-10x faster than per-sequence teacher forward. Tokenizes all sequences,
    pads, runs one GPU forward pass, then splits for per-sequence covering.
    Per-sequence covering runs in parallel threads (numpy releases the GIL).

    Args:
        batch_raw_bytes: list of B raw byte lists
        Others: same as _get_teacher_targets_covering

    Returns:
        list of B result dicts (same format as _get_teacher_targets_covering)
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    token_byte_seqs = covering_tables["token_byte_seqs"]
    first_byte_matrix_np = covering_tables["first_byte_matrix_np"]
    B = len(batch_raw_bytes)

    # Batch tokenize (set pad_token if missing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [bytes(rb).decode('utf-8', errors='replace') for rb in batch_raw_bytes]
    encoded = tokenizer(texts, padding=True, return_tensors='pt', truncation=False)
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    # Single teacher forward pass
    with torch.inference_mode():
        outputs = teacher(input_ids, attention_mask=attention_mask,
                         output_hidden_states=extract_hidden, use_cache=False)

    # Pre-compute GPU→CPU transfers sequentially (PCIe transfers are serial anyway)
    seq_data = []
    for b in range(B):
        seq_len = attention_mask[b].sum().item()
        logits_b = outputs.logits[b, :seq_len]
        probs = F.softmax((logits_b / temperature).float(), dim=-1)
        probs_np = probs.cpu().numpy()
        token_ids = input_ids[b, :seq_len].tolist()
        seq_data.append((probs_np, token_ids))

    # Parallel CPU covering computation (numpy releases GIL for matmul/indexing)
    def _process_seq(b):
        probs_np, token_ids = seq_data[b]
        return _covering_one_sequence(
            b, probs_np, token_ids, batch_raw_bytes[b],
            token_byte_seqs, first_byte_matrix_np, covering_tables, max_depth)

    with ThreadPoolExecutor(max_workers=min(B, 12)) as executor:
        covering_results = list(executor.map(_process_seq, range(B)))

    # Assemble results with GPU transfers
    results = []
    for b in range(B):
        byte_probs_np, byte_mask_np, byte_offsets, coverage_ratio = covering_results[b]
        byte_probs = torch.from_numpy(byte_probs_np).to(device)
        byte_mask = torch.from_numpy(byte_mask_np).to(device)

        result = {
            "byte_probs": byte_probs,
            "byte_mask": byte_mask,
            "token_offsets": byte_offsets,
            "coverage_ratio": coverage_ratio,
        }

        if extract_hidden:
            seq_len = attention_mask[b].sum().item()
            hidden = outputs.hidden_states[-1][b, :seq_len]
            result["hidden"] = hidden.float()
            result["n_tokens"] = seq_len

        results.append(result)

    return results


def _get_teacher_targets_covering(teacher, tokenizer, covering_tables, raw_bytes,
                                  device, temperature=2.0, extract_hidden=True,
                                  max_depth=None):
    """Run teacher forward pass and extract LOSSLESS byte-level targets via covering.

    Phan et al. (ICLR 2025) covering decomposition: produces autoregressive byte
    conditionals at EVERY byte position, preserving all teacher information.

    FAST PATH: Teacher forward on GPU, covering math on CPU (numpy), single transfer back.
    Eliminates per-position GPU kernel launch overhead that made v1 unusably slow.

    Args:
        max_depth: If set, limit covering depth (0=first-byte-marginal at all positions,
                   1=depth-0+1, None=full covering). Trades information for speed.

    Returns dict with:
        byte_probs: (T_bytes, 256) soft byte targets at ALL supervised positions
        byte_mask: (T_bytes,) bool, True at every position with teacher coverage
        hidden: (N_tok, d_teacher) last-layer hidden states (if extract_hidden)
        token_offsets: list of byte offsets for each token boundary
        coverage_ratio: fraction of byte positions with teacher supervision
    """
    import numpy as np
    token_byte_seqs = covering_tables["token_byte_seqs"]

    text = bytes(raw_bytes).decode('utf-8', errors='replace')
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.inference_mode():
        outputs = teacher(tokens, output_hidden_states=extract_hidden, use_cache=False)
        logits = outputs.logits[0]  # (T_tok, V)

    # Temperature-scaled probabilities — compute on GPU, then move to CPU numpy
    probs = F.softmax((logits / temperature).float(), dim=-1)  # (T_tok, V)
    probs_np = probs.cpu().numpy()  # single GPU→CPU transfer

    # Compute byte offsets at token boundaries
    token_ids = tokens[0].tolist()
    byte_offsets = []
    pos = 0
    for tid in token_ids:
        byte_offsets.append(pos)
        pos += len(token_byte_seqs[tid]) if tid < len(token_byte_seqs) else 1
    byte_offsets.append(pos)

    # Allocate output on CPU (numpy) — single transfer to GPU at the end
    n_bytes = len(raw_bytes)
    byte_probs_np = np.zeros((n_bytes, 256), dtype=np.float32)
    byte_mask_np = np.zeros(n_bytes, dtype=np.bool_)
    supervised_count = 0

    # Vectorized depth-0: all positions at once via matmul
    first_byte_matrix_np = covering_tables["first_byte_matrix_np"]
    all_depth0 = probs_np @ first_byte_matrix_np.T  # (T_tok, 256) — single matmul!
    depth0_sums = all_depth0.sum(axis=1, keepdims=True)
    depth0_sums = np.maximum(depth0_sums, 1e-10)
    all_depth0 = all_depth0 / depth0_sums

    # Process each teacher position
    for i in range(len(token_ids) - 1):
        next_tid = token_ids[i + 1]
        if next_tid >= len(token_byte_seqs):
            continue
        next_bytes = token_byte_seqs[next_tid]
        if not next_bytes:
            continue

        token_start = byte_offsets[i + 1]

        # Depth 0: use pre-computed vectorized result
        byte_pos = token_start - 1  # student alignment: predict k+1 at position k
        if 0 <= byte_pos < n_bytes:
            byte_probs_np[byte_pos] = all_depth0[i]
            byte_mask_np[byte_pos] = True
            supervised_count += 1

        # Depth 1+: compute per-position (numpy, fast CPU)
        effective_depth = len(next_bytes) if max_depth is None else min(len(next_bytes), max_depth + 1)
        if effective_depth > 1:
            tok_probs_np = probs_np[i]
            conditionals = _covering_byte_conditionals_np(
                tok_probs_np, next_bytes[:effective_depth], covering_tables)
            for j in range(1, len(conditionals)):
                bp = token_start + j - 1
                if 0 <= bp < n_bytes:
                    byte_probs_np[bp] = conditionals[j]
                    byte_mask_np[bp] = True
                    supervised_count += 1

    coverage_ratio = supervised_count / max(n_bytes, 1)

    # Single CPU���GPU transfer
    byte_probs = torch.from_numpy(byte_probs_np).to(device)
    byte_mask = torch.from_numpy(byte_mask_np).to(device)

    result = {
        "byte_probs": byte_probs,
        "byte_mask": byte_mask,
        "token_offsets": byte_offsets,
        "coverage_ratio": coverage_ratio,
    }

    if extract_hidden:
        hidden = outputs.hidden_states[-1][0]  # (T_tok, d_teacher)
        result["hidden"] = hidden.float()
        result["n_tokens"] = tokens.shape[1]

    return result


def _teacher_patch_targets(targets_t, raw_len, patch_size, teacher_dim, device):
    """Project teacher token-aligned hidden states onto causal patch boundaries."""
    patch_count = (raw_len + patch_size - 1) // patch_size
    patch_hidden = torch.zeros(patch_count, teacher_dim, device=device, dtype=torch.float32)
    patch_mask = torch.zeros(patch_count, dtype=torch.bool, device=device)

    hidden = targets_t.get("hidden")
    offsets = targets_t.get("token_offsets")
    if hidden is None or offsets is None:
        return patch_hidden, patch_mask

    for j in range(patch_count):
        boundary = j * patch_size
        best_ti = -1
        for ti, offset in enumerate(offsets):
            if offset <= boundary:
                best_ti = ti
            else:
                break
        if 0 <= best_ti < hidden.shape[0]:
            patch_hidden[j] = hidden[best_ti]
            patch_mask[j] = True

    return patch_hidden, patch_mask


def _aggregate_teacher_byte_targets(sample_targets, teacher_priors,
                                    aggregation="arithmetic_mean",
                                    aggregation_temp=1.0,
                                    routing_cfg=None, teacher_cfgs=None):
    """Aggregate per-teacher byte targets into one student-facing byte target."""
    probs = torch.stack([t["byte_probs"].float() for t in sample_targets])   # (N, T, 256)
    masks = torch.stack([t["byte_mask"] for t in sample_targets])            # (N, T)
    priors = teacher_priors.to(probs.device, dtype=torch.float32)
    priors = priors / priors.sum().clamp_min(1e-10)
    available = masks.any(dim=-1)  # (N,)
    temp = max(float(aggregation_temp), 1e-6)

    if aggregation == "arithmetic_mean":
        weights = priors.unsqueeze(-1) * masks.float()
        weight_denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-10)
        weights = weights / weight_denom
        sample_weights = priors * available.float()
        sample_weights = sample_weights / sample_weights.sum().clamp_min(1e-10)
    elif aggregation == "entropy_weighted":
        safe_probs = probs.clamp_min(1e-10)
        entropies = -(safe_probs * safe_probs.log()).sum(dim=-1)  # (N, T)
        score = torch.log(priors.clamp_min(1e-10)).unsqueeze(-1) - entropies / temp
        score = score.masked_fill(~masks, -1e9)
        weights = torch.softmax(score, dim=0)
        weights = torch.where(masks, weights, torch.zeros_like(weights))
        weight_denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-10)
        weights = weights / weight_denom

        sample_entropy = []
        for i in range(entropies.shape[0]):
            if available[i]:
                sample_entropy.append(entropies[i][masks[i]].mean())
            else:
                sample_entropy.append(torch.tensor(1e9, device=probs.device))
        sample_entropy = torch.stack(sample_entropy)
        sample_score = torch.log(priors.clamp_min(1e-10)) - sample_entropy / temp
        sample_score = sample_score.masked_fill(~available, -1e9)
        sample_weights = torch.softmax(sample_score, dim=0)
        sample_weights = torch.where(available, sample_weights, torch.zeros_like(sample_weights))
        sample_weights = sample_weights / sample_weights.sum().clamp_min(1e-10)
    elif aggregation == "anchor_confidence_routing":
        # Codex-prescribed routing: anchor dominates, aux contributes only where
        # JS divergence > threshold AND aux is more confident.
        # Requires exactly 2 teachers with roles "anchor" and "aux" in teacher_cfgs.
        assert probs.shape[0] == 2, "anchor_confidence_routing requires exactly 2 teachers"
        rc = routing_cfg or {}
        js_thresh = rc.get("js_threshold", 0.02)
        conf_margin = rc.get("confidence_margin", 0.02)
        conf_scale = rc.get("confidence_scale", 0.08)
        aux_cap = rc.get("aux_weight_cap", 0.35)

        # Identify anchor vs aux by role in teacher_cfgs
        anchor_idx, aux_idx = 0, 1
        if teacher_cfgs and len(teacher_cfgs) >= 2:
            for i, tc in enumerate(teacher_cfgs):
                if tc.get("role") == "anchor":
                    anchor_idx = i
                elif tc.get("role") == "aux":
                    aux_idx = i

        p_anchor = probs[anchor_idx]  # (T, 256)
        p_aux = probs[aux_idx]        # (T, 256)
        m_anchor = masks[anchor_idx]  # (T,)
        m_aux = masks[aux_idx]        # (T,)

        # JS divergence per position: JSD(anchor || aux)
        m = 0.5 * (p_anchor + p_aux)
        m_safe = m.clamp_min(1e-10)
        kl_am = (p_anchor * (p_anchor.clamp_min(1e-10) / m_safe).log()).sum(-1)  # (T,)
        kl_bm = (p_aux * (p_aux.clamp_min(1e-10) / m_safe).log()).sum(-1)
        jsd = 0.5 * (kl_am + kl_bm)  # (T,)

        # Confidence = max prob at each position
        conf_anchor = p_anchor.max(dim=-1).values  # (T,)
        conf_aux = p_aux.max(dim=-1).values

        # Routing weight for aux: sigmoid((conf_aux - conf_anchor - margin) / scale)
        # Gated by JS divergence > threshold (teachers must disagree)
        raw_route = torch.sigmoid((conf_aux - conf_anchor - conf_margin) / max(conf_scale, 1e-6))
        # Zero out where teachers agree (JS below threshold) or either teacher missing
        both_available = m_anchor & m_aux
        disagree = (jsd > js_thresh) & both_available
        r = torch.where(disagree, raw_route.clamp(max=aux_cap), torch.zeros_like(raw_route))

        # Build per-position weights: (2, T)
        weights = torch.zeros_like(masks.float())
        weights[anchor_idx] = (1.0 - r) * m_anchor.float()
        weights[aux_idx] = r * m_aux.float()
        # Where only one teacher available, use that teacher alone
        only_anchor = m_anchor & ~m_aux
        only_aux = m_aux & ~m_anchor
        weights[anchor_idx] = torch.where(only_anchor, torch.ones_like(r), weights[anchor_idx])
        weights[aux_idx] = torch.where(only_aux, torch.ones_like(r), weights[aux_idx])

        weight_denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-10)
        weights = weights / weight_denom

        # Sample-level weights: anchor-dominated, respecting index order
        sample_weights = torch.zeros(2, device=probs.device)
        sample_weights[anchor_idx] = 1.0 - aux_cap * 0.5
        sample_weights[aux_idx] = aux_cap * 0.5
        if not available[anchor_idx]:
            sample_weights[anchor_idx] = 0.0
            sample_weights[aux_idx] = 1.0
        elif not available[aux_idx]:
            sample_weights[anchor_idx] = 1.0
            sample_weights[aux_idx] = 0.0
        sample_weights = sample_weights / sample_weights.sum().clamp_min(1e-10)
    else:
        raise ValueError(f"Unknown teacher aggregation: {aggregation}")

    combined_probs = (probs * weights.unsqueeze(-1)).sum(dim=0)  # (T, 256)
    combined_mask = masks.any(dim=0)
    combined_probs = torch.where(
        combined_mask.unsqueeze(-1),
        combined_probs,
        torch.zeros_like(combined_probs),
    )
    return combined_probs, combined_mask, sample_weights


def _get_kd_weights(step, max_steps, kd_alpha, kd_beta,
                    kd_ramp_steps, kd_hold_steps, kd_decay_start_step,
                    kd_peak_alpha_mult, kd_peak_beta_mult,
                    kd_final_alpha_mult, kd_final_beta_mult,
                    kd_alpha_decay_points=None, kd_beta_decay_points=None):
    """Piecewise KD schedule: ramp -> hold -> optional raise -> decay."""
    ramp = min(1.0, step / max(kd_ramp_steps, 1))
    hold_end = kd_ramp_steps + max(kd_hold_steps, 0)
    decay_start = max(hold_end, min(max_steps - 1, kd_decay_start_step))
    peak_alpha = kd_alpha * kd_peak_alpha_mult
    peak_beta = kd_beta * kd_peak_beta_mult
    raise_enabled = (
        hold_end < decay_start
        and (abs(kd_peak_alpha_mult - 1.0) > 1e-8 or abs(kd_peak_beta_mult - 1.0) > 1e-8)
    )

    if step < kd_ramp_steps:
        return kd_alpha * ramp, kd_beta * ramp, ramp

    if step < hold_end:
        return kd_alpha, kd_beta, 1.0

    if raise_enabled and step < decay_start:
        progress = (step - hold_end) / max(decay_start - hold_end, 1)
        alpha_eff = kd_alpha + (peak_alpha - kd_alpha) * progress
        beta_eff = kd_beta + (peak_beta - kd_beta) * progress
        return alpha_eff, beta_eff, 1.0

    def interp_decay_points(base_weight, points):
        if not points:
            return None
        pts = sorted((int(s), float(m)) for s, m in points)
        if step <= pts[0][0]:
            return base_weight * pts[0][1]
        for (s0, m0), (s1, m1) in zip(pts, pts[1:]):
            if step <= s1:
                progress = (step - s0) / max(s1 - s0, 1)
                mult = m0 + (m1 - m0) * progress
                return base_weight * mult
        return base_weight * pts[-1][1]

    alpha_start = peak_alpha if raise_enabled else kd_alpha
    beta_start = peak_beta if raise_enabled else kd_beta
    alpha_from_points = interp_decay_points(kd_alpha, kd_alpha_decay_points)
    beta_from_points = interp_decay_points(kd_beta, kd_beta_decay_points)
    progress = (step - decay_start) / max(max_steps - decay_start, 1)
    progress = min(max(progress, 0.0), 1.0)
    alpha_end = kd_alpha * kd_final_alpha_mult
    beta_end = kd_beta * kd_final_beta_mult
    alpha_eff = alpha_from_points if alpha_from_points is not None else alpha_start + (alpha_end - alpha_start) * progress
    beta_eff = beta_from_points if beta_from_points is not None else beta_start + (beta_end - beta_start) * progress
    return alpha_eff, beta_eff, 1.0


def _get_teacher_targets(teacher, tokenizer, first_byte_map, raw_bytes, device,
                         temperature=2.0, extract_hidden=True):
    """Run teacher forward pass and extract byte-level targets.

    LEGACY: Uses first-byte marginal (84% signal loss). For new runs, use
    _get_teacher_targets_covering() instead.

    Returns dict with:
        byte_probs: (T_bytes, 256) soft byte targets at token boundaries
        byte_mask: (T_bytes,) bool, True at positions where teacher makes a prediction
        hidden: (N_tok, d_teacher) last-layer hidden states (if extract_hidden)
        token_offsets: list of byte offsets for each token boundary
    """
    text = bytes(raw_bytes).decode('utf-8', errors='replace')
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)

    with torch.inference_mode():
        outputs = teacher(tokens, output_hidden_states=extract_hidden, use_cache=False)
        logits = outputs.logits[0]  # (T_tok, V)

    # Byte-level probabilities with temperature
    scaled_logits = logits / temperature
    byte_probs_tok = _teacher_byte_probs(scaled_logits, first_byte_map)  # (T_tok, 256)

    # Get byte offsets at token boundaries
    token_texts = [tokenizer.decode([t]) for t in tokens[0].tolist()]
    byte_offsets = []
    pos = 0
    for txt in token_texts:
        byte_offsets.append(pos)
        pos += len(txt.encode('utf-8', errors='replace'))

    # Scatter teacher predictions to byte positions.
    # Teacher predicts next-token at position i → byte at offset byte_offsets[i+1].
    # Student_logits[k] predicts byte k+1, so teacher target for byte k aligns to
    # student_logits[k-1]. We store at k-1 so the KL loss compares matching positions.
    n_bytes = len(raw_bytes)
    byte_probs = torch.zeros(n_bytes, 256, device=device, dtype=torch.float32)
    byte_mask = torch.zeros(n_bytes, dtype=torch.bool, device=device)

    for i in range(len(byte_offsets) - 1):
        target_off = byte_offsets[i + 1]
        student_pos = target_off - 1  # align teacher "byte k" → student_logits[k-1]
        if 0 <= student_pos < n_bytes:
            byte_probs[student_pos] = byte_probs_tok[i]
            byte_mask[student_pos] = True

    result = {
        "byte_probs": byte_probs,
        "byte_mask": byte_mask,
        "token_offsets": byte_offsets,
    }

    if extract_hidden:
        # Last-layer hidden states at each token position
        hidden = outputs.hidden_states[-1][0]  # (T_tok, d_teacher)
        result["hidden"] = hidden.float()
        result["n_tokens"] = tokens.shape[1]

    return result


def precompute_teacher_cache(cfg, output_path):
    """Pre-compute teacher byte probs for all training windows (offline caching).

    Eliminates teacher forward passes during training → 8-10x per-step speedup.
    Saves per-teacher sparse byte probs (top-K) + masks + metadata.
    Routing, TAID, and gating all compose with cached probs at train time.

    Usage:
        python code/sutra_dyad.py --cache-teachers results/config_xxx.json
        # Then train with: "use_teacher_cache": true, "teacher_cache_path": "results/teacher_cache_xxx.pt"
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import numpy as np

    max_steps = cfg.get("max_steps", 250)
    batch_size = cfg.get("batch_size", 12)
    grad_accum = cfg.get("grad_accum", 6)
    seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)
    kd_temperature = cfg.get("kd_temperature", 2.0)
    use_covering = cfg.get("use_covering", False)
    covering_max_depth = cfg.get("covering_max_depth", None)
    teacher_cfgs = cfg.get("teachers", [{"id": "HuggingFaceTB/SmolLM2-1.7B", "weight": 1.0}])
    cache_topk = cfg.get("cache_topk", 16)  # Top-K sparse storage per position
    cache_seed = cfg.get("cache_seed", 42)

    n_microbatches = max_steps * grad_accum
    print(f"\n{'='*60}")
    print(f"TEACHER CACHE PRE-COMPUTATION")
    print(f"{'='*60}")
    print(f"Steps: {max_steps}, micro-batches: {n_microbatches}, batch: {batch_size}, seq: {seq_bytes}")
    print(f"Top-K: {cache_topk}, seed: {cache_seed}")

    # Dataset
    dataset = ByteShardedDataset()
    print(f"Dataset: {dataset.total_bytes / 1e9:.1f}B est bytes")

    # Pre-sample all windows with fixed seed for reproducibility
    print(f"\nPre-sampling {n_microbatches} micro-batches ({n_microbatches * batch_size} windows)...")
    rng_state = random.getstate()
    random.seed(cache_seed)
    windows_x, windows_y = [], []
    for i in range(n_microbatches):
        x, y = dataset.sample_batch(batch_size, seq_bytes, device='cpu', split='train')
        windows_x.append(x.to(torch.uint8))
        windows_y.append(y.to(torch.uint8))
        if (i + 1) % 500 == 0:
            print(f"  Sampled {i+1}/{n_microbatches} micro-batches")
    random.setstate(rng_state)
    print(f"  Done. Total windows: {len(windows_x) * batch_size}")

    # Load teachers + build covering tables
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    teacher_cache_data = {}
    for t_cfg in teacher_cfgs:
        teacher_id = t_cfg["id"]
        teacher_role = t_cfg.get("role", "equal")
        print(f"\nLoading teacher: {teacher_id} (4-bit, role={teacher_role})")
        tok = AutoTokenizer.from_pretrained(teacher_id)
        model = AutoModelForCausalLM.from_pretrained(
            teacher_id, quantization_config=bnb_cfg, device_map=DEVICE)
        model.eval()
        vocab = model.config.vocab_size

        covering_tables, fb_map = None, None
        if use_covering:
            print(f"  Building covering tables for {vocab} tokens...")
            covering_tables = _build_covering_tables(tok, vocab, device=DEVICE)
            print(f"  Covering: {covering_tables['n_prefixes']} prefixes")
        else:
            fb_map = _build_first_byte_map(tok, vocab, DEVICE)

        # Process all micro-batches
        print(f"  Computing byte probs for {n_microbatches} micro-batches...")
        all_topk_vals = []   # list of (B, T, K) float16
        all_topk_idx = []    # list of (B, T, K) uint8
        all_masks = []       # list of (B, T) bool
        t0 = time.time()

        for mb_idx in range(n_microbatches):
            x = windows_x[mb_idx]
            batch_raw = [x[b].tolist() for b in range(x.shape[0])]

            with torch.no_grad():
                if use_covering:
                    targets = _get_teacher_targets_covering_batched(
                        model, tok, covering_tables, batch_raw, DEVICE,
                        temperature=kd_temperature, extract_hidden=False,
                        max_depth=covering_max_depth,
                    )
                else:
                    targets = []
                    for raw in batch_raw:
                        targets.append(_get_teacher_targets(
                            model, tok, fb_map, raw, DEVICE,
                            temperature=kd_temperature, extract_hidden=False,
                        ))

            # Stack and sparsify
            bp = torch.stack([t["byte_probs"] for t in targets])   # (B, T, 256)
            bm = torch.stack([t["byte_mask"] for t in targets])    # (B, T)

            # Top-K sparse storage
            topk_v, topk_i = bp.topk(cache_topk, dim=-1)  # (B, T, K)
            all_topk_vals.append(topk_v.half().cpu())
            all_topk_idx.append(topk_i.to(torch.uint8).cpu())
            all_masks.append(bm.cpu())

            if (mb_idx + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (mb_idx + 1) / elapsed
                eta = (n_microbatches - mb_idx - 1) / max(rate, 1e-6)
                print(f"    [{mb_idx+1}/{n_microbatches}] {rate:.1f} mb/s, "
                      f"ETA {eta/60:.1f}min")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed/60:.1f}min ({n_microbatches/elapsed:.1f} mb/s)")

        teacher_cache_data[teacher_id] = {
            "topk_vals": all_topk_vals,
            "topk_idx": all_topk_idx,
            "masks": all_masks,
            "role": teacher_role,
            "weight": float(t_cfg.get("weight", 1.0)),
        }

        # Free teacher model VRAM before loading next
        del model, tok, covering_tables, fb_map
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Package and save
    cache = {
        "version": 1,
        "config": cfg,
        "n_microbatches": n_microbatches,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "seq_bytes": seq_bytes,
        "temperature": kd_temperature,
        "cache_topk": cache_topk,
        "cache_seed": cache_seed,
        "windows_x": windows_x,
        "windows_y": windows_y,
        "teachers": teacher_cache_data,
    }

    # Compute cache size
    total_bytes = 0
    for tid, td in teacher_cache_data.items():
        for v in td["topk_vals"]:
            total_bytes += v.nelement() * 2
        for i in td["topk_idx"]:
            total_bytes += i.nelement()
        for m in td["masks"]:
            total_bytes += m.nelement()
    for x in windows_x:
        total_bytes += x.nelement()
    for y in windows_y:
        total_bytes += y.nelement()

    output_path = Path(output_path)
    print(f"\nSaving cache to {output_path} ({total_bytes/1e6:.0f}MB estimated)...")
    torch.save(cache, output_path)
    actual_size = output_path.stat().st_size
    print(f"Saved: {actual_size/1e6:.0f}MB on disk")
    print(f"Cache ready. Use with: \"use_teacher_cache\": true, \"teacher_cache_path\": \"{output_path}\"")
    return cache


def _reconstruct_byte_probs_from_cache(topk_vals, topk_idx, n_bytes=256):
    """Reconstruct full (B, T, 256) byte probs from sparse top-K cache.

    Distributes residual mass uniformly across non-top-K positions.
    """
    B, T, K = topk_vals.shape
    probs = torch.zeros(B, T, n_bytes, device=topk_vals.device, dtype=torch.float32)
    # Scatter top-K values
    idx_long = topk_idx.long()
    probs.scatter_(2, idx_long, topk_vals.float())
    # Distribute residual uniformly
    residual = (1.0 - probs.sum(dim=-1, keepdim=True)).clamp_min(0)
    # Count non-topk positions
    n_rest = n_bytes - K
    if n_rest > 0:
        uniform_fill = residual / n_rest
        # Create mask of top-K positions
        topk_mask = torch.zeros_like(probs, dtype=torch.bool)
        topk_mask.scatter_(2, idx_long, True)
        probs = probs + uniform_fill * (~topk_mask).float()
    return probs


def verify_teacher_cache(cache_path, n_batches=3):
    """Verify cached teacher targets match live teacher output (parity check).

    Loads cache, runs live teacher forward on a few cached windows,
    compares byte probs. Reports max and mean error per teacher.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    cfg = cache["config"]
    kd_temperature = cache["temperature"]
    cache_topk = cache["cache_topk"]
    use_covering = cfg.get("use_covering", False)
    covering_max_depth = cfg.get("covering_max_depth", None)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
    )

    print(f"\n{'='*60}")
    print(f"CACHE PARITY CHECK: {cache_path}")
    print(f"{'='*60}")
    print(f"Checking {n_batches} micro-batches per teacher, top-{cache_topk} sparse")

    all_pass = True
    for tid, td in cache["teachers"].items():
        print(f"\nTeacher: {tid}")
        tok = AutoTokenizer.from_pretrained(tid)
        model = AutoModelForCausalLM.from_pretrained(
            tid, quantization_config=bnb_cfg, device_map=DEVICE)
        model.eval()
        vocab = model.config.vocab_size

        covering_tables, fb_map = None, None
        if use_covering:
            covering_tables = _build_covering_tables(tok, vocab, device=DEVICE)
        else:
            fb_map = _build_first_byte_map(tok, vocab, DEVICE)

        errors = []
        for mb_idx in range(n_batches):
            x = cache["windows_x"][mb_idx]
            batch_raw = [x[b].tolist() for b in range(x.shape[0])]

            # Live teacher forward
            with torch.no_grad():
                if use_covering:
                    live_targets = _get_teacher_targets_covering_batched(
                        model, tok, covering_tables, batch_raw, DEVICE,
                        temperature=kd_temperature, extract_hidden=False,
                        max_depth=covering_max_depth,
                    )
                else:
                    live_targets = [_get_teacher_targets(
                        model, tok, fb_map, raw, DEVICE,
                        temperature=kd_temperature, extract_hidden=False,
                    ) for raw in batch_raw]

            live_bp = torch.stack([t["byte_probs"] for t in live_targets])
            live_mask = torch.stack([t["byte_mask"] for t in live_targets])

            # Cached reconstruction
            topk_v = td["topk_vals"][mb_idx].to(DEVICE)
            topk_i = td["topk_idx"][mb_idx].to(DEVICE)
            cached_bp = _reconstruct_byte_probs_from_cache(topk_v, topk_i)
            cached_mask = td["masks"][mb_idx].to(DEVICE)

            # Compare on masked positions (where teacher has valid probs)
            both_mask = live_mask & cached_mask
            if both_mask.any():
                diff = (live_bp[both_mask] - cached_bp[both_mask].to(live_bp.device)).abs()
                max_err = diff.max().item()
                mean_err = diff.mean().item()
                # Top-K error: compare only top-K positions (exact match expected)
                topk_live_v, topk_live_i = live_bp[both_mask].topk(cache_topk, dim=-1)
                topk_cache_v = cached_bp[both_mask].to(live_bp.device).gather(1, topk_live_i)
                topk_diff = (topk_live_v - topk_cache_v).abs()
                topk_max = topk_diff.max().item()
                errors.append((max_err, mean_err, topk_max))
                print(f"  mb {mb_idx}: max_err={max_err:.6f}, mean_err={mean_err:.6f}, "
                      f"topk_max_err={topk_max:.6f}, mask_match={both_mask.sum()}/{live_mask.sum()}")

        if errors:
            avg_max = sum(e[0] for e in errors) / len(errors)
            avg_topk = sum(e[2] for e in errors) / len(errors)
            status = "PASS" if avg_topk < 0.005 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  Result: {status} (avg max_err={avg_max:.6f}, avg topk_err={avg_topk:.6f})")

        del model, tok, covering_tables, fb_map
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


def train_ekalavya(s1_ckpt, cfg=None):
    """Ekalavya Phase C: multi-teacher knowledge distillation for Sutra-Dyad."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if cfg is None:
        cfg = {}

    # Training config
    max_steps = cfg.get("max_steps", 5000)
    batch_size = cfg.get("batch_size", 24)
    grad_accum = cfg.get("grad_accum", 6)
    seq_bytes = cfg.get("seq_bytes", SEQ_BYTES)
    lr_local = cfg.get("lr_local", 1.5e-4)
    min_lr_local = cfg.get("min_lr_local", 1.5e-5)
    warmup = cfg.get("warmup_steps", 200)
    run_name = cfg.get("run_name", "ekalavya_phase_c")
    log_every = cfg.get("log_every", LOG_EVERY)
    eval_every = cfg.get("eval_every", EVAL_EVERY)
    rolling_save = cfg.get("rolling_save", ROLLING_SAVE)
    grad_clip = cfg.get("grad_clip", GRAD_CLIP)

    # Teacher cache config (offline pre-computed targets)
    use_teacher_cache = cfg.get("use_teacher_cache", False)
    teacher_cache_path = cfg.get("teacher_cache_path", None)

    # KD config
    kd_alpha = cfg.get("kd_alpha", 0.5)      # Byte logit KD weight
    kd_beta = cfg.get("kd_beta", 0.15)       # Repr alignment weight
    kd_temperature = cfg.get("kd_temperature", 2.0)
    kd_ramp_steps = cfg.get("kd_ramp_steps", 500)  # Ramp KD weights from 0 over this many steps
    kd_hold_steps = cfg.get("kd_hold_steps", 500)
    kd_decay_start_step = cfg.get("kd_decay_start_step", int(max_steps * 0.7))
    kd_peak_alpha_mult = cfg.get("kd_peak_alpha_mult", 1.5)
    kd_peak_beta_mult = cfg.get("kd_peak_beta_mult", 1.67)
    kd_final_alpha_mult = cfg.get("kd_final_alpha_mult", 0.33)
    kd_final_beta_mult = cfg.get("kd_final_beta_mult", 0.60)
    kd_alpha_decay_points = cfg.get("kd_alpha_decay_points", None)
    kd_beta_decay_points = cfg.get("kd_beta_decay_points", None)
    use_covering = cfg.get("use_covering", False)  # Covering decomposition (lossless KD)
    covering_max_depth = cfg.get("covering_max_depth", None)  # None=full, 0=first-byte-only, 1=depth-0+1
    repr_anchor_only = cfg.get("repr_anchor_only", False)

    # TAID: Temporally Adaptive Interpolated Distillation (byte-space adaptation)
    # Target: p_taid ∝ p_student_det^(1-β) * p_route^β — progressive student→teacher
    use_taid = cfg.get("use_taid", False)
    taid_beta_start = cfg.get("taid_beta_start", 0.0)
    taid_beta_end = cfg.get("taid_beta_end", 0.8)
    taid_beta_ramp = cfg.get("taid_beta_ramp_steps", 600)

    # Uncertainty gating v2: gate = t_conf * (1-s_match)^exp, renorm to mean=1, clamp
    # s_match = student prob at teacher's top byte (not max student prob)
    use_uncertainty_gating = cfg.get("use_uncertainty_gating", False)
    ug_renormalize = cfg.get("ug_renormalize", True)
    ug_clamp = cfg.get("ug_clamp", 1.5)  # v2: lowered from 4.0 per Codex correctness review
    ug_exp_start = cfg.get("ug_exp_start", 1.0)
    ug_exp_end = cfg.get("ug_exp_end", 2.0)
    ug_exp_ramp = cfg.get("ug_exp_ramp_steps", 600)

    # Teacher config
    anchor_id = cfg.get("anchor_teacher", "HuggingFaceTB/SmolLM2-1.7B")
    aux_id = cfg.get("aux_teacher", None)  # None = anchor only
    aux_frequency = cfg.get("aux_frequency", 4)  # Run auxiliary every N steps
    teacher_cfgs = cfg.get("teachers", None)
    teacher_aggregation = cfg.get("teacher_aggregation", "arithmetic_mean")
    teacher_aggregation_temp = cfg.get("teacher_aggregation_temp", 1.0)
    # Anchor-confidence routing params
    teacher_routing_cfg = {
        "js_threshold": cfg.get("teacher_js_threshold", 0.02),
        "confidence_margin": cfg.get("teacher_confidence_margin", 0.02),
        "confidence_scale": cfg.get("teacher_confidence_scale", 0.08),
        "aux_weight_cap": cfg.get("teacher_aux_weight_cap", 0.35),
    }

    # Freeze config
    freeze_global = cfg.get("freeze_global", False)  # Default: full unfreeze for Ekalavya
    unfreeze_layers = cfg.get("unfreeze_layers", None)

    print(f"\n{'='*60}")
    print(f"EKALAVYA PHASE C -- Multi-Teacher Knowledge Distillation")
    print(f"{'='*60}")

    # Dataset
    dataset = ByteShardedDataset()
    print(f"Dataset: {dataset.total_bytes / 1e9:.1f}B est bytes, {len(dataset.index)} shards")

    # Student model
    print(f"\nLoading student from: {s1_ckpt}")
    ckpt = torch.load(s1_ckpt, map_location='cpu', weights_only=False)
    model = SutraDyadS1(max_seq_bytes=seq_bytes)
    model.load_state_dict(ckpt["model"], strict=False)
    s1_step = ckpt.get("step", "?")

    # Progressive unfreeze schedule (Codex design):
    # Phase 0 (step 0-400): only local + bridge + global 8-11
    # Phase 1 (step 400-1200): + global 4-7
    # Phase 2 (step 1200+): + global 0-3 (all unfrozen)
    unfreeze_phase1_step = cfg.get("unfreeze_phase1_step", 400)
    unfreeze_phase2_step = cfg.get("unfreeze_phase2_step", 1200)

    # Start with global 0-7 frozen, 8-11 + global_norm unfrozen
    model.freeze_global()
    model.unfreeze_global(layer_range=(8, 12))  # top 4 global layers
    model.global_norm.weight.requires_grad = True  # norm goes with top layers

    params = model.count_params()
    print(f"Student: {params['total']/1e6:.1f}M total, "
          f"{params['trainable']/1e6:.1f}M trainable, {params['frozen']/1e6:.1f}M frozen")
    model = model.to(DEVICE)

    # Load teacher cache OR live teacher models
    teacher_cache = None
    if use_teacher_cache and teacher_cache_path:
        print(f"\nLoading teacher cache from {teacher_cache_path}...")
        teacher_cache = torch.load(teacher_cache_path, map_location='cpu', weights_only=False)
        assert teacher_cache["version"] == 1, f"Unsupported cache version: {teacher_cache['version']}"
        assert teacher_cache["n_microbatches"] >= max_steps * grad_accum, \
            f"Cache has {teacher_cache['n_microbatches']} micro-batches, need {max_steps * grad_accum}"
        print(f"  Cache: {teacher_cache['n_microbatches']} micro-batches, "
              f"top-{teacher_cache['cache_topk']} sparse, "
              f"teachers: {list(teacher_cache['teachers'].keys())}")
        # Build teacher_cfgs from cache metadata
        if teacher_cfgs is None:
            teacher_cfgs = []
            for tid, td in teacher_cache["teachers"].items():
                teacher_cfgs.append({"id": tid, "weight": td["weight"], "role": td["role"]})
        # No live teachers needed — create lightweight teacher entries for logging/routing only
        teachers = []
        for idx, t_cfg in enumerate(teacher_cfgs):
            teachers.append({
                "id": t_cfg["id"],
                "weight": float(t_cfg.get("weight", 1.0)),
                "tokenizer": None,
                "model": None,
                "hidden_dim": 0,
                "vocab": 0,
                "covering": None,
                "first_byte_map": None,
                "repr_proj": nn.Linear(1, model.d_local, bias=False).to(DEVICE),  # placeholder
            })
        # Repr loss disabled in cache mode (no hidden states)
        if kd_beta > 0:
            print(f"  NOTE: repr loss (beta={kd_beta}) disabled in cache mode (no hidden states)")
            kd_beta = 0.0
    else:
        # Live teacher loading (original path)
        # 4-bit quantization config for teachers (Codex VRAM budget: ~2GB per teacher)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        if teacher_cfgs is None:
            teacher_cfgs = [{"id": anchor_id, "weight": 1.0}]
            if aux_id:
                teacher_cfgs.append({"id": aux_id, "weight": 1.0})

        teachers = []
        for idx, teacher_cfg in enumerate(teacher_cfgs):
            teacher_id = teacher_cfg["id"]
            teacher_weight = float(teacher_cfg.get("weight", 1.0))
            teacher_role = "anchor" if idx == 0 else f"teacher_{idx}"
            print(f"\nLoading {teacher_role}: {teacher_id} (4-bit, weight={teacher_weight:.3f})")
            teacher_tok = AutoTokenizer.from_pretrained(teacher_id)
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_id, quantization_config=bnb_cfg, device_map=DEVICE)
            teacher_model.eval()
            teacher_dim = teacher_model.config.hidden_size
            teacher_vocab = teacher_model.config.vocab_size
            teacher_covering, teacher_fb = None, None
            if use_covering:
                print(f"  Building covering tables for {teacher_vocab} tokens...")
                teacher_covering = _build_covering_tables(teacher_tok, teacher_vocab, device=DEVICE)
                print(f"  Covering: {teacher_covering['n_prefixes']} prefixes, "
                      f"max {teacher_covering['max_token_bytes']} bytes/token")
            else:
                teacher_fb = _build_first_byte_map(teacher_tok, teacher_vocab, DEVICE)
            repr_proj = nn.Linear(teacher_dim, model.d_local, bias=False).to(DEVICE)
            nn.init.normal_(repr_proj.weight, std=0.02)
            print(f"  Hidden dim: {teacher_dim}, vocab: {teacher_vocab}")
            teachers.append({
                "id": teacher_id,
                "weight": teacher_weight,
                "tokenizer": teacher_tok,
                "model": teacher_model,
                "hidden_dim": teacher_dim,
                "vocab": teacher_vocab,
                "covering": teacher_covering,
                "first_byte_map": teacher_fb,
                "repr_proj": repr_proj,
        })

    # Optimizer: layerwise LR groups per Codex spec
    # bridge: patch_encoder, patch_pool, patch_proj, global_to_local
    # local: byte_embed, bypass_proj, local_decoder, local_norm, output_head
    # global_top: global_layers.8-11, global_norm
    # global_mid: global_layers.4-7
    # global_bot: global_layers.0-3
    # proj: repr projection heads
    lr_bridge = cfg.get("lr_bridge", 2e-4)
    lr_global_top = cfg.get("lr_global_top", 7.5e-5)
    lr_global_mid = cfg.get("lr_global_mid", 5e-5)
    lr_global_bot = cfg.get("lr_global_bot", 3e-5)

    # Include ALL params upfront so progressive unfreeze doesn't miss them.
    # Frozen params get zero gradients -> optimizer step is a no-op for them.
    groups = {"bridge": [], "local": [], "global_top": [], "global_mid": [], "global_bot": []}
    for name, p in model.named_parameters():
        if any(k in name for k in ['patch_encoder', 'patch_pool', 'patch_proj', 'global_to_local']):
            groups["bridge"].append(p)
        elif any(k in name for k in ['global_layers.8', 'global_layers.9', 'global_layers.10', 'global_layers.11', 'global_norm']):
            groups["global_top"].append(p)
        elif any(k in name for k in ['global_layers.4', 'global_layers.5', 'global_layers.6', 'global_layers.7']):
            groups["global_mid"].append(p)
        elif any(k in name for k in ['global_layers.0', 'global_layers.1', 'global_layers.2', 'global_layers.3']):
            groups["global_bot"].append(p)
        else:
            groups["local"].append(p)

    proj_params = []
    for teacher in teachers:
        proj_params += list(teacher["repr_proj"].parameters())

    param_groups = []
    if groups["bridge"]:
        param_groups.append({"params": groups["bridge"], "lr": lr_bridge, "name": "bridge"})
    if groups["local"]:
        param_groups.append({"params": groups["local"], "lr": lr_local, "name": "local"})
    if groups["global_top"]:
        param_groups.append({"params": groups["global_top"], "lr": lr_global_top, "name": "global_top"})
    if groups["global_mid"]:
        param_groups.append({"params": groups["global_mid"], "lr": lr_global_mid, "name": "global_mid"})
    if groups["global_bot"]:
        param_groups.append({"params": groups["global_bot"], "lr": lr_global_bot, "name": "global_bot"})
    param_groups.append({"params": proj_params, "lr": lr_bridge, "name": "proj"})

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

    proj_param_count = sum(p.numel() for p in proj_params)
    log(f"Config: batch={batch_size}, grad_accum={grad_accum}, seq={seq_bytes}, "
        f"lr_local={lr_local}, lr_bridge={lr_bridge}, lr_gtop={lr_global_top}, "
        f"lr_gmid={lr_global_mid}, lr_gbot={lr_global_bot}, warmup={warmup}, max_steps={max_steps}")
    log(f"KD: alpha={kd_alpha} (byte logits), beta={kd_beta} (repr), T={kd_temperature}, "
        f"ramp={kd_ramp_steps}, covering={'ON' if use_covering else 'OFF'}"
        f"{f', max_depth={covering_max_depth}' if use_covering else ''}")
    log(f"KD schedule: hold={kd_hold_steps}, decay_start={kd_decay_start_step}, "
        f"peak=({kd_peak_alpha_mult:.2f},{kd_peak_beta_mult:.2f}), "
        f"final=({kd_final_alpha_mult:.2f},{kd_final_beta_mult:.2f})")
    if kd_alpha_decay_points:
        log(f"KD alpha decay points: {kd_alpha_decay_points}")
    if kd_beta_decay_points:
        log(f"KD beta decay points: {kd_beta_decay_points}")
    if repr_anchor_only:
        log("Representation KD: anchor-only")
    if use_taid:
        log(f"TAID: beta {taid_beta_start}->{taid_beta_end} over {taid_beta_ramp} steps")
    if use_uncertainty_gating:
        log(f"Uncertainty gating: exp {ug_exp_start}->{ug_exp_end} over {ug_exp_ramp} steps, "
            f"clamp={ug_clamp}, renorm={ug_renormalize}")
    log(f"Teachers ({len(teachers)}), aggregation={teacher_aggregation}, agg_temp={teacher_aggregation_temp}:")
    for i, teacher in enumerate(teachers):
        role = teacher_cfgs[i].get("role", "equal") if teacher_cfgs and i < len(teacher_cfgs) else "equal"
        log(f"  - {teacher['id']} (weight={teacher['weight']:.3f}, d={teacher['hidden_dim']}, vocab={teacher['vocab']}, role={role})")
    repr_teacher_indices = list(range(len(teachers)))
    if repr_anchor_only:
        repr_anchor_idx = 0
        if teacher_cfgs:
            for i, tc in enumerate(teacher_cfgs):
                if tc.get("role") == "anchor":
                    repr_anchor_idx = i
                    break
        repr_teacher_indices = [repr_anchor_idx]
    if teacher_aggregation == "anchor_confidence_routing":
        log(f"Routing: js_thresh={teacher_routing_cfg['js_threshold']}, margin={teacher_routing_cfg['confidence_margin']}, "
            f"scale={teacher_routing_cfg['confidence_scale']}, aux_cap={teacher_routing_cfg['aux_weight_cap']}")
    log(f"Projection params: {proj_param_count/1e3:.1f}K")
    log(f"Student checkpoint: {s1_ckpt} (step {s1_step})")

    # Resume support
    start_step = 0
    resume_ckpt = cfg.get("resume_ckpt")
    if resume_ckpt:
        rk = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(rk["model"], strict=False)
        if "repr_proj_teachers" in rk:
            saved_repr = rk["repr_proj_teachers"]
            for teacher in teachers:
                if teacher["id"] in saved_repr:
                    teacher["repr_proj"].load_state_dict(saved_repr[teacher["id"]])
        else:
            if "repr_proj_anchor" in rk and teachers:
                teachers[0]["repr_proj"].load_state_dict(rk["repr_proj_anchor"])
            if "repr_proj_aux" in rk and len(teachers) > 1:
                teachers[1]["repr_proj"].load_state_dict(rk["repr_proj_aux"])
        if "optimizer" in rk:
            optimizer.load_state_dict(rk["optimizer"])
        if "scaler" in rk:
            scaler.load_state_dict(rk["scaler"])
        start_step = rk["step"]
        # Restore unfreeze state based on resumed step
        if start_step >= unfreeze_phase1_step:
            model.unfreeze_global(layer_range=(4, 8))
        if start_step >= unfreeze_phase2_step:
            model.unfreeze_global(layer_range=(0, 4))
        log(f"Resumed from {resume_ckpt} at step {start_step}")

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for teacher in teachers:
        teacher["repr_proj"].train()
    best_eval = float('inf')
    step = start_step
    t0 = time.time()
    running_ce, running_kd, running_repr, running_total = 0.0, 0.0, 0.0, 0.0
    nan_count = 0

    log(f"\nStarting Ekalavya training at {time.strftime('%H:%M:%S')} (step {step})")

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_ce, accum_kd, accum_repr, accum_total = 0.0, 0.0, 0.0, 0.0

        alpha_eff, beta_eff, ramp = _get_kd_weights(
            step=step,
            max_steps=max_steps,
            kd_alpha=kd_alpha,
            kd_beta=kd_beta,
            kd_ramp_steps=kd_ramp_steps,
            kd_hold_steps=kd_hold_steps,
            kd_decay_start_step=kd_decay_start_step,
            kd_peak_alpha_mult=kd_peak_alpha_mult,
            kd_peak_beta_mult=kd_peak_beta_mult,
            kd_final_alpha_mult=kd_final_alpha_mult,
            kd_final_beta_mult=kd_final_beta_mult,
            kd_alpha_decay_points=kd_alpha_decay_points,
            kd_beta_decay_points=kd_beta_decay_points,
        )

        # Progressive unfreeze
        if step == unfreeze_phase1_step:
            model.unfreeze_global(layer_range=(4, 8))
            log(f"  >>> UNFREEZE global layers 4-7 at step {step}")
        if step == unfreeze_phase2_step:
            model.unfreeze_global(layer_range=(0, 4))
            log(f"  >>> UNFREEZE global layers 0-3 at step {step} (all params trainable)")

        for micro in range(grad_accum):
            # --- Data: cached windows or live sampling ---
            if teacher_cache is not None:
                mb_idx = (step - start_step) * grad_accum + micro
                x = teacher_cache["windows_x"][mb_idx].to(dtype=torch.long, device=DEVICE)
                y = teacher_cache["windows_y"][mb_idx].to(dtype=torch.long, device=DEVICE)
            else:
                x, y = dataset.sample_batch(batch_size, seq_bytes, device=DEVICE, split='train')

            # --- Teacher targets: cached reconstruction or live forward ---
            with torch.no_grad():
                B_actual = x.shape[0]
                all_byte_probs = []
                all_byte_masks = []
                teacher_patch_hidden_by_teacher = []
                teacher_patch_mask_by_teacher = []
                teacher_priors = torch.tensor(
                    [teacher["weight"] for teacher in teachers],
                    device=DEVICE,
                    dtype=torch.float32,
                )
                teacher_results = []

                if teacher_cache is not None:
                    # Reconstruct per-teacher byte probs from sparse cache
                    mb_idx = (step - start_step) * grad_accum + micro
                    for teacher in teachers:
                        tid = teacher["id"]
                        td = teacher_cache["teachers"][tid]
                        topk_v = td["topk_vals"][mb_idx].to(DEVICE)  # (B, T, K) float16
                        topk_i = td["topk_idx"][mb_idx].to(DEVICE)   # (B, T, K) uint8
                        mask = td["masks"][mb_idx].to(DEVICE)         # (B, T) bool
                        byte_probs = _reconstruct_byte_probs_from_cache(topk_v, topk_i)
                        # Build per-sample target dicts (same format as live path)
                        targets_batch = []
                        for b in range(B_actual):
                            targets_batch.append({
                                "byte_probs": byte_probs[b],
                                "byte_mask": mask[b],
                            })
                        teacher_results.append(targets_batch)
                else:
                    batch_raw = [x[b].tolist() for b in range(B_actual)]
                    for teacher in teachers:
                        if use_covering:
                            targets_batch = _get_teacher_targets_covering_batched(
                                teacher["model"], teacher["tokenizer"], teacher["covering"], batch_raw, DEVICE,
                                temperature=kd_temperature, extract_hidden=(beta_eff > 0),
                                max_depth=covering_max_depth,
                            )
                        else:
                            targets_batch = []
                            for raw in batch_raw:
                                targets_batch.append(_get_teacher_targets(
                                    teacher["model"], teacher["tokenizer"], teacher["first_byte_map"], raw, DEVICE,
                                    temperature=kd_temperature, extract_hidden=(beta_eff > 0),
                                ))
                        teacher_results.append(targets_batch)

                teacher_patch_hidden = None
                teacher_patch_mask = None
                for b in range(B_actual):
                    sample_targets = [targets_batch[b] for targets_batch in teacher_results]
                    combined_probs, combined_mask, sample_teacher_weights = _aggregate_teacher_byte_targets(
                        sample_targets=sample_targets,
                        teacher_priors=teacher_priors,
                        aggregation=teacher_aggregation,
                        aggregation_temp=teacher_aggregation_temp,
                        routing_cfg=teacher_routing_cfg,
                        teacher_cfgs=teacher_cfgs,
                    )
                    all_byte_probs.append(combined_probs)
                    all_byte_masks.append(combined_mask)

                    if beta_eff > 0:
                        patch_hidden_list = []
                        patch_mask_list = []
                        raw_len = len(batch_raw[b])
                        for teacher_idx, teacher in enumerate(teachers):
                            patch_hidden_t, patch_mask_t = _teacher_patch_targets(
                                targets_t=sample_targets[teacher_idx],
                                raw_len=raw_len,
                                patch_size=model.patch_size,
                                teacher_dim=teacher["hidden_dim"],
                                device=DEVICE,
                            )
                            patch_hidden_list.append(patch_hidden_t)
                            patch_mask_list.append(patch_mask_t)
                        teacher_patch_hidden_by_teacher.append(patch_hidden_list)
                        teacher_patch_mask_by_teacher.append((patch_mask_list, sample_teacher_weights))

                teacher_byte_probs = torch.stack(all_byte_probs)  # (B, T, 256)
                teacher_byte_mask = torch.stack(all_byte_masks)   # (B, T)

            # --- Student forward ---
            with torch.amp.autocast('cuda', dtype=DTYPE):
                ce_loss, internals = model(x, y, return_internals=True)
                student_logits = internals["logits"]  # (B, T, 256)

                # KD Loss 1: Byte-level KL divergence (float32 for numerical stability)
                # Supports TAID (progressive target) and uncertainty gating (per-position weighting)
                kd_loss = torch.tensor(0.0, device=DEVICE)
                ug_stats = {}
                if alpha_eff > 0:
                    # Compute in float32 outside autocast for log-space stability
                    student_log_probs = F.log_softmax(student_logits.float() / kd_temperature, dim=-1)
                    mask = teacher_byte_mask[:, :student_logits.shape[1]]
                    if mask.any():
                        t_probs = teacher_byte_probs[:, :student_logits.shape[1], :]

                        # Pre-compute student probs (shared by TAID + uncertainty gating)
                        s_probs_shared = None
                        if use_taid or use_uncertainty_gating:
                            with torch.no_grad():
                                s_probs_shared = F.softmax(student_logits.float() / kd_temperature, dim=-1)
                                s_probs_shared = s_probs_shared[:, :t_probs.shape[1], :]

                        # TAID: progressive intermediate target p_taid ∝ p_s_det^(1-β) * p_t^β
                        if use_taid:
                            beta_taid = taid_beta_start + (taid_beta_end - taid_beta_start) * min(1.0, step / max(taid_beta_ramp, 1))
                            # Geometric interpolation in probability space
                            taid_target = s_probs_shared.pow(1.0 - beta_taid) * t_probs.pow(beta_taid)
                            taid_target = taid_target / taid_target.sum(dim=-1, keepdim=True).clamp_min(1e-10)
                            target_log_probs = torch.log(taid_target + 1e-10)
                        else:
                            target_log_probs = torch.log(t_probs + 1e-10)

                        kl = F.kl_div(student_log_probs, target_log_probs, reduction='none', log_target=True)
                        kl = kl.sum(dim=-1)  # (B, T)

                        # Uncertainty gating v2: concentrate KD on high-value positions
                        # s_match = student prob at teacher's top byte (not max student prob)
                        # Gating formula: gate = t_conf * (1 - s_match)^exp
                        # Renormalized to mean=1, clamped at ug_clamp
                        if use_uncertainty_gating:
                            t_conf = t_probs.max(dim=-1).values  # (B, T) teacher confidence
                            with torch.no_grad():
                                teacher_top = t_probs.argmax(dim=-1, keepdim=True)  # (B, T, 1)
                                s_match = s_probs_shared.gather(-1, teacher_top).squeeze(-1)  # (B, T)
                            ug_exp = ug_exp_start + (ug_exp_end - ug_exp_start) * min(1.0, step / max(ug_exp_ramp, 1))
                            gate = t_conf * (1.0 - s_match).pow(ug_exp)
                            with torch.no_grad():
                                raw_gate_mean = (gate * mask.float()).sum() / mask.float().sum().clamp_min(1e-10)
                            if ug_renormalize:
                                gate = gate / raw_gate_mean.clamp_min(1e-10)
                            gate = gate.clamp(max=ug_clamp)
                            kl = kl * gate
                            # Stats for logging
                            with torch.no_grad():
                                ug_stats = {
                                    "ug_raw_mean": raw_gate_mean.item(),
                                    "ug_mean": gate[mask].mean().item() if mask.any() else 0,
                                    "ug_active": (gate[mask] > 0.5).float().mean().item() if mask.any() else 0,
                                    "ug_max": gate[mask].max().item() if mask.any() else 0,
                                    "ug_sat": (gate[mask] >= ug_clamp - 0.01).float().mean().item() if mask.any() else 0,
                                }

                        kl = (kl * mask.float()).sum() / mask.float().sum()
                        kd_loss = kd_temperature ** 2 * kl

                # KD Loss 2: Representation alignment (cosine on global_local bottleneck)
                repr_loss = torch.tensor(0.0, device=DEVICE)
                if beta_eff > 0 and teacher_patch_hidden_by_teacher:
                    global_local = internals["global_local"]  # (B, N, d_local=640)
                    teacher_proj_batches = []
                    teacher_proj_masks = []
                    for b in range(B_actual):
                        patch_hidden_list = teacher_patch_hidden_by_teacher[b]
                        patch_mask_list, sample_teacher_weights = teacher_patch_mask_by_teacher[b]
                        projected = []
                        projected_masks = []
                        for teacher_idx in repr_teacher_indices:
                            teacher = teachers[teacher_idx]
                            teacher_weight = 1.0 if repr_anchor_only else sample_teacher_weights[teacher_idx]
                            projected.append(
                                teacher["repr_proj"](patch_hidden_list[teacher_idx].to(DTYPE)).float()
                                * teacher_weight
                            )
                            projected_masks.append(patch_mask_list[teacher_idx])
                        teacher_proj_batches.append(torch.stack(projected).sum(dim=0))
                        teacher_proj_masks.append(torch.stack(projected_masks).any(dim=0))
                    proj_teacher = torch.stack(teacher_proj_batches)      # (B, N, d_local)
                    teacher_patch_mask = torch.stack(teacher_proj_masks)  # (B, N)

                    gl_norm = F.normalize(global_local.float(), dim=-1)
                    pt_norm = F.normalize(proj_teacher, dim=-1)
                    cos_sim = (gl_norm * pt_norm).sum(dim=-1)  # (B, N)
                    pmask = teacher_patch_mask[:, :cos_sim.shape[1]]
                    if pmask.any():
                        repr_loss = ((1.0 - cos_sim) * pmask.float()).sum() / pmask.float().sum()

                total_loss = ce_loss + alpha_eff * kd_loss + beta_eff * repr_loss
                loss_scaled = total_loss / grad_accum

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                nan_count += 1
                log(f"  [NaN/Inf at step {step}, micro {micro}] -- skipping")
                if nan_count > 10:
                    log("FATAL: >10 NaN losses, aborting")
                    log_f.close()
                    return
                continue

            scaler.scale(loss_scaled).backward()
            accum_ce += ce_loss.item()
            accum_kd += kd_loss.item()
            accum_repr += repr_loss.item()
            accum_total += total_loss.item()

        # Per-group LR schedule (layerwise)
        lr_map = {
            "bridge": (lr_bridge, lr_bridge * 0.1),
            "local": (lr_local, min_lr_local),
            "global_top": (lr_global_top, lr_global_top * 0.1),
            "global_mid": (lr_global_mid, lr_global_mid * 0.1),
            "global_bot": (lr_global_bot, lr_global_bot * 0.1),
            "proj": (lr_bridge, lr_bridge * 0.1),
        }
        for pg in optimizer.param_groups:
            name = pg.get("name", "local")
            max_lr, min_lr_pg = lr_map.get(name, (lr_local, min_lr_local))
            pg['lr'] = get_lr(step, warmup, max_steps, max_lr, min_lr_pg)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + proj_params,
            grad_clip)
        scaler.step(optimizer)
        scaler.update()

        avg_ce = accum_ce / grad_accum
        avg_kd = accum_kd / grad_accum
        avg_repr = accum_repr / grad_accum
        avg_total = accum_total / grad_accum
        running_ce += avg_ce
        running_kd += avg_kd
        running_repr += avg_repr
        running_total += avg_total
        step += 1

        if step % log_every == 0:
            n = log_every
            elapsed = time.time() - t0
            throughput = step * batch_size * grad_accum * seq_bytes / elapsed
            bpb = running_ce / n / math.log(2)
            vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
            local_lr = optimizer.param_groups[0]['lr']
            ug_suffix = ""
            if use_uncertainty_gating and ug_stats:
                ug_suffix = (f" | ug={ug_stats.get('ug_mean',0):.2f}/{ug_stats.get('ug_active',0):.0%}"
                             f" r{ug_stats.get('ug_raw_mean',0):.2f}/m{ug_stats.get('ug_max',0):.1f}/s{ug_stats.get('ug_sat',0):.0%}")
            taid_suffix = ""
            if use_taid:
                bt = taid_beta_start + (taid_beta_end - taid_beta_start) * min(1.0, step / max(taid_beta_ramp, 1))
                taid_suffix = f" | taid_b={bt:.2f}"
            log(f"  step {step:>5d} | CE {running_ce/n:.4f} | KD {running_kd/n:.4f} | "
                f"Repr {running_repr/n:.4f} | Total {running_total/n:.4f} | BPB {bpb:.3f} | "
                f"lr {local_lr:.2e} | grad {grad_norm:.2f} | "
                f"{throughput/1e6:.1f}MB/s | VRAM {vram:.1f}G | ramp {ramp:.2f}"
                f"{taid_suffix}{ug_suffix}")
            running_ce, running_kd, running_repr, running_total = 0.0, 0.0, 0.0, 0.0

        if step % eval_every == 0:
            el = eval_loss(model, dataset, n_batches=50, seq_len=seq_bytes)
            bpb_eval = el / math.log(2)
            log(f"  *** EVAL step {step}: loss={el:.4f}, BPB={bpb_eval:.3f}")
            if el < best_eval:
                best_eval = el
                save_dict = {
                    "step": step, "stage": 1,
                    "model": model.state_dict(),
                    "repr_proj_teachers": {teacher["id"]: teacher["repr_proj"].state_dict() for teacher in teachers},
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "eval_loss": el, "config": cfg,
                }
                torch.save(save_dict, ckpt_dir / "best.pt")
                log(f"      New best! Saved to {ckpt_dir / 'best.pt'}")

            prompt = b"The meaning of intelligence is"
            try:
                gen_ids = model.generate(list(prompt), max_new_bytes=128, temperature=0.8)
                gen_text = bytes(gen_ids).decode('utf-8', errors='replace')
                safe = gen_text.encode('ascii', errors='replace').decode('ascii')
                log(f"      GEN: {safe[:200]}")
            except Exception as e:
                log(f"      GEN failed: {e}")

        if step % rolling_save == 0:
            save_dict = {
                "step": step, "stage": 1,
                "model": model.state_dict(),
                "repr_proj_teachers": {teacher["id"]: teacher["repr_proj"].state_dict() for teacher in teachers},
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "eval_loss": best_eval, "config": cfg,
            }
            torch.save(save_dict, ckpt_dir / f"step_{step}.pt")
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    final_loss = eval_loss(model, dataset, n_batches=50, seq_len=seq_bytes)
    final_bpb = final_loss / math.log(2)
    log(f"\n{'='*60}")
    log(f"EKALAVYA TRAINING COMPLETE -- {max_steps} steps")
    log(f"Final eval loss: {final_loss:.4f}, BPB: {final_bpb:.3f}")
    log(f"Best eval loss: {best_eval:.4f}, BPB: {best_eval/math.log(2):.3f}")
    log(f"{'='*60}")

    log_f.close()

    results = {
        "run_name": run_name, "stage": 1, "phase": "ekalavya",
        "final_eval_loss": final_loss, "final_bpb": final_bpb,
        "best_eval_loss": best_eval, "best_bpb": best_eval / math.log(2),
        "steps": max_steps, "config": cfg,
        "params": params, "proj_params": proj_param_count,
        "teachers": [{"id": teacher["id"], "weight": teacher["weight"]} for teacher in teachers],
        "teacher_aggregation": teacher_aggregation,
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
    parser.add_argument("--ekalavya", type=str, default=None, metavar="S1_CKPT",
                        help="Run Ekalavya KD from Stage 1 checkpoint")
    parser.add_argument("--cache-teachers", type=str, default=None, metavar="CONFIG",
                        help="Pre-compute teacher cache from JSON config (offline caching)")
    parser.add_argument("--verify-cache", type=str, default=None, metavar="CACHE_PATH",
                        help="Verify teacher cache parity (compare cached vs live targets)")
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

    if args.cache_teachers:
        with open(args.cache_teachers) as f:
            cache_cfg = json.load(f)
        cache_name = cache_cfg.get("run_name", "teacher_cache")
        output_path = REPO / "results" / f"teacher_cache_{cache_name}.pt"
        precompute_teacher_cache(cache_cfg, output_path)
    elif args.verify_cache:
        verify_teacher_cache(args.verify_cache)
    elif args.det_eval:
        det_eval(args.det_eval)
    elif args.ablate:
        ablation_eval(args.ablate)
    elif args.profile_teachers:
        profile_teachers()
    elif args.ekalavya:
        cfg = {}
        if args.config:
            with open(args.config) as f:
                cfg = json.load(f)
        cfg.setdefault("max_steps", args.max_steps)
        cfg.setdefault("batch_size", args.batch_size or 24)
        cfg.setdefault("seq_bytes", args.seq_bytes)
        cfg.setdefault("run_name", args.run_name or "ekalavya_phase_c")
        if args.lr:
            cfg.setdefault("lr_local", args.lr)
        if args.resume:
            cfg["resume_ckpt"] = args.resume
        train_ekalavya(args.ekalavya, cfg)
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
