"""VRAM profiling for S0/E1/E2 training configurations.

Estimates memory usage WITHOUT a GPU by computing parameter sizes,
optimizer state sizes, and rough activation estimates.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import S0Config, SutraS0


def estimate_vram(cfg: S0Config, batch_size: int = 4, checkpoint_every: int = 2) -> dict:
    """Estimate VRAM usage for S0-only training."""
    model = SutraS0(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    dead_prefixes = ("encoder.entropy_head", "encoder.residual_head", "verifier")
    trainable_params = sum(
        p.numel() for n, p in model.named_parameters()
        if not any(n.startswith(dp) for dp in dead_prefixes)
    )
    dead_params = total_params - trainable_params

    param_mem_bf16 = total_params * 2
    grad_mem_bf16 = trainable_params * 2
    optimizer_m = trainable_params * 4
    optimizer_v = trainable_params * 4
    master_weights = trainable_params * 4

    P = cfg.patch_size
    seq_bytes = 4096
    n_patches = seq_bytes // P

    per_layer_act = batch_size * n_patches * cfg.d_model * 2 * 4
    ffn_hidden = int(cfg.d_model * cfg.ffn_mult)
    ffn_hidden = ((ffn_hidden + 63) // 64) * 64
    per_layer_ffn = batch_size * n_patches * ffn_hidden * 2 * 3

    stored_layers = (cfg.n_layers + checkpoint_every - 1) // checkpoint_every
    reasoner_act = stored_layers * (per_layer_act + per_layer_ffn)

    encoder_act = batch_size * seq_bytes * cfg.byte_dim * 2
    encoder_act += batch_size * n_patches * P * cfg.byte_dim * 2

    decoder_act = batch_size * n_patches * P * cfg.decoder_dim * 2 * cfg.decoder_layers * 3

    total_act = reasoner_act + encoder_act + decoder_act

    misc = 50 * 1024 * 1024

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


def estimate_e2_overhead(student_dim: int = 576) -> dict:
    """Estimate additional VRAM for E2 multi-teacher projection ports + router."""
    from eklavya_e2_cache import TEACHER_REGISTRY

    port_params = 0
    port_details = {}

    for spec in TEACHER_REGISTRY:
        teacher_params = 0

        if spec.has_align:
            ln_params = student_dim * 2  # LayerNorm weight + bias
            proj_params = student_dim * spec.hidden_dim  # Linear, no bias
            teacher_params += ln_params + proj_params

        if spec.has_semantic:
            ln_params = student_dim * 2
            proj_params = student_dim * spec.hidden_dim
            teacher_params += ln_params + proj_params

        port_params += teacher_params
        if teacher_params > 0:
            port_details[spec.name] = {
                "params": teacher_params,
                "mem_mb": teacher_params * 2 / 1e6,
                "align": spec.has_align,
                "semantic": spec.has_semantic,
                "hidden_dim": spec.hidden_dim,
            }

    port_mem_bf16 = port_params * 2
    port_grad_bf16 = port_params * 2
    port_optimizer = port_params * 4 * 3  # m, v, master weights

    router_mem = 1 * 1024 * 1024  # ~1 MB for router buffers

    emb_mem = 0
    for spec in TEACHER_REGISTRY:
        if spec.has_align or spec.has_semantic:
            emb_mem += spec.hidden_dim * 32000 * 2

    total = (port_mem_bf16 + port_grad_bf16 + port_optimizer
             + router_mem + emb_mem)

    return {
        "port_params": port_params,
        "port_details": port_details,
        "port_mem_mb": port_mem_bf16 / 1e6,
        "port_grad_mb": port_grad_bf16 / 1e6,
        "port_optimizer_mb": port_optimizer / 1e6,
        "router_mem_mb": router_mem / 1e6,
        "emb_tables_mb": emb_mem / 1e6,
        "total_overhead_mb": total / 1e6,
        "total_overhead_gb": total / 1e9,
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

    print(f"\n{'='*50}")
    print("E2 Multi-Teacher Overhead (on top of S0 base)")
    print(f"{'='*50}")

    e2 = estimate_e2_overhead()
    print(f"  Total port params: {e2['port_params']:,}")
    for name, detail in e2["port_details"].items():
        flags = []
        if detail["align"]:
            flags.append("align")
        if detail["semantic"]:
            flags.append("semantic")
        print(f"    {name}: {detail['params']:,} params ({detail['mem_mb']:.2f} MB) "
              f"[{', '.join(flags)} -> dim {detail['hidden_dim']}]")
    print(f"  Port weights:     {e2['port_mem_mb']:>8.2f} MB")
    print(f"  Port gradients:   {e2['port_grad_mb']:>8.2f} MB")
    print(f"  Port optimizer:   {e2['port_optimizer_mb']:>8.2f} MB")
    print(f"  Router buffers:   {e2['router_mem_mb']:>8.2f} MB")
    print(f"  TOTAL E2 overhead:{e2['total_overhead_mb']:>8.2f} MB ({e2['total_overhead_gb']:.3f} GB)")

    s0_est = estimate_vram(S0Config(), batch_size=4, checkpoint_every=2)
    combined_gb = s0_est["total_mem_gb"] + e2["total_overhead_gb"]
    print(f"\n  S0 base + E2 overhead: {s0_est['total_mem_gb']:.2f} + {e2['total_overhead_gb']:.3f} = {combined_gb:.2f} GB")
    print(f"  Fits 24GB:             {'YES' if combined_gb < 24.0 else 'NO'}")
    print(f"  Headroom:              {24.0 - combined_gb:.2f} GB")


if __name__ == "__main__":
    main()
