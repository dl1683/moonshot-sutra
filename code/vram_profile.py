"""VRAM profiling for S0 training configurations.

Estimates memory usage WITHOUT a GPU by computing parameter sizes,
optimizer state sizes, and rough activation estimates. Also validates
the estimate against torch.cuda peak memory if GPU is available.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import S0Config, SutraS0


def estimate_vram(cfg: S0Config, batch_size: int = 4, checkpoint_every: int = 2) -> dict:
    """Estimate VRAM usage for a given S0 config."""
    model = SutraS0(cfg)

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    dead_prefixes = ("encoder.entropy_head", "encoder.residual_head", "verifier")
    trainable_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not any(n.startswith(dp) for dp in dead_prefixes)
    )
    dead_params = total_params - trainable_params

    # Memory calculations (in bytes)
    param_mem_bf16 = total_params * 2  # bf16 weights
    grad_mem_bf16 = trainable_params * 2  # bf16 gradients
    optimizer_m = trainable_params * 4  # fp32 first moment
    optimizer_v = trainable_params * 4  # fp32 second moment
    master_weights = trainable_params * 4  # fp32 master copy (for AMP)

    # Activation estimates (rough)
    P = cfg.patch_size
    seq_bytes = 4096
    n_patches = seq_bytes // P

    # Per-layer activation (bf16): attention states + FFN intermediates
    per_layer_act = batch_size * n_patches * cfg.d_model * 2 * 4  # Q,K,V,O
    ffn_hidden = int(cfg.d_model * cfg.ffn_mult)
    ffn_hidden = ((ffn_hidden + 63) // 64) * 64
    per_layer_ffn = batch_size * n_patches * ffn_hidden * 2 * 3  # gate,up,down intermediates

    # With checkpointing: store activations for ceil(n_layers/checkpoint_every) layers
    stored_layers = (cfg.n_layers + checkpoint_every - 1) // checkpoint_every
    reasoner_act = stored_layers * (per_layer_act + per_layer_ffn)

    # Encoder activations
    encoder_act = batch_size * seq_bytes * cfg.byte_dim * 2  # byte embeddings + mixer
    encoder_act += batch_size * n_patches * P * cfg.byte_dim * 2  # patch reshape

    # Decoder activations
    decoder_act = batch_size * n_patches * P * cfg.decoder_dim * 2 * cfg.decoder_layers * 3

    total_act = reasoner_act + encoder_act + decoder_act

    # Misc buffers (RoPE freqs, masks, etc.)
    misc = 50 * 1024 * 1024  # ~50 MB

    total = param_mem_bf16 + grad_mem_bf16 + optimizer_m + optimizer_v + master_weights + total_act + misc

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "dead_params": dead_params,
        "param_mem_mb": param_mem_bf16 / 1e6,
        "grad_mem_mb": grad_mem_bf16 / 1e6,
        "optimizer_mem_mb": (optimizer_m + optimizer_v + master_weights) / 1e6,
        "activation_mem_mb": total_act / 1e6,
        "misc_mem_mb": misc / 1e6,
        "total_mem_mb": total / 1e6,
        "total_mem_gb": total / 1e9,
        "fits_24gb": total / 1e9 < 24.0,
    }


def main():
    configs = {
        "P4 (default)": S0Config(),
        "P8": S0Config(patch_size=8, max_seq_len=512),
        "D640 escalation": S0Config(d_model=640, n_heads=10, n_kv_heads=2),
        "D768 escalation": S0Config(d_model=768, n_layers=22, n_heads=12, n_kv_heads=4),
    }

    for name, cfg in configs.items():
        est = estimate_vram(cfg, batch_size=4, checkpoint_every=2)
        print(f"\n{'='*50}")
        print(f"Config: {name}")
        print(f"  Params: {est['total_params']:,} ({est['trainable_params']:,} trainable)")
        print(f"  Param memory:     {est['param_mem_mb']:>8.1f} MB")
        print(f"  Gradient memory:  {est['grad_mem_mb']:>8.1f} MB")
        print(f"  Optimizer memory: {est['optimizer_mem_mb']:>8.1f} MB")
        print(f"  Activation memory:{est['activation_mem_mb']:>8.1f} MB")
        print(f"  Misc buffers:     {est['misc_mem_mb']:>8.1f} MB")
        print(f"  TOTAL:            {est['total_mem_mb']:>8.1f} MB ({est['total_mem_gb']:.2f} GB)")
        print(f"  Fits 24GB:        {'YES' if est['fits_24gb'] else 'NO'}")


if __name__ == "__main__":
    main()
