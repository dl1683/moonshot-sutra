"""Scratchpad Memory Module for Sutra v0.5.4.

8-16 shared memory slots for discourse-level state (topic, entities, context).
Tokens read from scratchpad each recurrent step; Stage 5 writes back.
Chrome-validated: +10.2% BPT improvement at dim=128.

Usage: integrate into SutraV05 forward pass:
    scratch = Scratchpad(dim=768, n_slots=8)
    # In recurrent loop:
    mem_ctx = scratch.read(mu, mem)     # tokens read discourse context
    mu = mu + 0.1 * mem_ctx             # inject as residual
    mem = scratch.write(mu, mem, pi[:,:,4:5])  # prefix-causal update for next step
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Scratchpad(nn.Module):
    """Shared causal scratchpad memory for discourse-level state.

    n_slots learned memory vectors that tokens read via attention
    and Stage 5 writes via gated EMA update.
    """

    def __init__(self, dim, n_slots=8, ema_decay=0.9):
        super().__init__()
        self.n_slots = n_slots
        self.ema_decay = ema_decay

        # Learnable initial memory
        self.mem_init = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)

        # Read: token queries memory via attention
        self.read_proj = nn.Linear(dim, dim)

        # Write: slot selection + gated EMA from prefix-causal summaries
        self.write_slot = nn.Linear(dim, n_slots)
        self.write_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.write_val = nn.Linear(dim, dim)

    def init_memory(self, batch_size, device):
        """Initialize memory for a new sequence."""
        return self.mem_init.expand(batch_size, -1, -1).to(device)

    def read(self, h, mem):
        """Tokens read from memory via attention.

        h: (B, T, D) token states
        mem: (B, S, D) shared slots or (B, T, S, D) per-position causal slots
        Returns: (B, T, D) memory context for each token
        """
        q = self.read_proj(h)
        scale = math.sqrt(h.size(-1))

        if mem.dim() == 3:
            attn = F.softmax(
                torch.bmm(q, mem.transpose(1, 2)) / scale,
                dim=-1,
            )  # (B, T, S)
            return torch.bmm(attn, mem)  # (B, T, D)

        if mem.dim() != 4 or mem.size(1) != h.size(1):
            raise ValueError(f"Expected mem to have shape (B,S,D) or (B,T,S,D), got {tuple(mem.shape)}")

        attn = F.softmax(
            torch.einsum("btd,btsd->bts", q, mem) / scale,
            dim=-1,
        )  # (B, T, S)
        return torch.einsum("bts,btsd->btd", attn, mem)  # (B, T, D)

    def write(self, h, mem, pi_write):
        """Stage 5 writes to memory via prefix-causal gated EMA.

        h: (B, T, D) token states after write stage
        mem: (B, S, D) shared slots or (B, T, S, D) per-position causal slots
        pi_write: (B, T, 1) write-stage probability (gates which tokens contribute)
        Returns: (B, T, S, D) updated memory, where position i only sees
                 summaries from positions < i on the next recurrent step
        """
        B, T, D = h.shape

        if mem.dim() == 3:
            base_mem = mem.unsqueeze(1).expand(B, T, self.n_slots, D)
        elif mem.dim() == 4 and mem.size(1) == T:
            base_mem = mem
        else:
            raise ValueError(f"Expected mem to have shape (B,S,D) or (B,T,S,D), got {tuple(mem.shape)}")

        slot_weights = F.softmax(self.write_slot(h), dim=-1) * pi_write  # (B, T, S)
        weighted_vals = slot_weights.unsqueeze(-1) * h.unsqueeze(2)      # (B, T, S, D)

        prefix_vals = weighted_vals.cumsum(dim=1) - weighted_vals
        prefix_weights = slot_weights.cumsum(dim=1) - slot_weights
        prefix_summary = prefix_vals / prefix_weights.unsqueeze(-1).clamp(min=1e-6)

        gate = self.write_gate(torch.cat([base_mem, prefix_summary], dim=-1))
        new_val = self.write_val(prefix_summary)
        updated = self.ema_decay * base_mem + (1 - self.ema_decay) * gate * new_val

        has_prefix = prefix_weights.unsqueeze(-1) > 0
        return torch.where(has_prefix, updated, base_mem)


if __name__ == "__main__":
    # Quick test
    B, T, D, S = 2, 32, 768, 8
    scratch = Scratchpad(D, n_slots=S)
    print(f"Scratchpad params: {sum(p.numel() for p in scratch.parameters()):,}")

    mem = scratch.init_memory(B, "cpu")
    h = torch.randn(B, T, D)
    pi_w = torch.ones(B, T, 1) * 0.3

    ctx = scratch.read(h, mem)
    print(f"Read: {ctx.shape}")

    mem_new = scratch.write(h, mem, pi_w)
    print(f"Write: {mem_new.shape}")

    # Causality: changing a future token must not change earlier-position memory
    h2 = h.clone()
    h2[:, -1] += 1.0
    mem_a = scratch.write(h, mem, pi_w)
    mem_b = scratch.write(h2, mem, pi_w)
    diff = (mem_a[:, :-1] - mem_b[:, :-1]).abs().max().item()
    print(f"Causality: diff={diff:.6f}")
    print(f"Ready for v0.5.4 integration")
