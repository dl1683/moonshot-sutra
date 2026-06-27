"""S0 Configuration Presets.

Defines P4 (default) and P8 variants for patch-size comparison,
plus the escalation ladder configs (D=640, D=768).

Per SE1_CANONICAL_SPEC.md:
- Scout P4 (high-resolution control) and P8 (strong decoder test) as first two
- Escalation: 30x576 → 30x640 → 22x768
"""

from s0_architecture import S0Config


def s0_p4() -> S0Config:
    """Default S0: P4, 30x576 deep-thin. ~121.7M params."""
    return S0Config()


def s0_p8() -> S0Config:
    """P8 variant: coarser patches, stronger decoder test. ~121.7M params.

    Same global reasoner dimensions. Patch aggregator maps 8*256=2048 → 576
    (more aggressive compression). Decoder context doubles to P=8.
    Max seq len halves (512 patches for 4096 bytes).
    """
    return S0Config(
        patch_size=8,
        max_seq_len=512,  # 4096 bytes / P8 = 512 patches
    )


def s0_d640() -> S0Config:
    """Escalation step 1: D=640, 30 layers. ~145M params."""
    return S0Config(
        d_model=640,
        n_heads=10,
        n_kv_heads=2,
        decoder_dim=384,
    )


def s0_d768() -> S0Config:
    """Escalation step 2: D=768, 22 layers. ~150M params.

    MobileLLM tradeoff reversed — wider but shallower.
    """
    return S0Config(
        d_model=768,
        n_layers=22,
        n_heads=12,
        n_kv_heads=4,
        decoder_dim=384,
    )


ALL_CONFIGS = {
    "p4": s0_p4,
    "p8": s0_p8,
    "d640": s0_d640,
    "d768": s0_d768,
}


def main():
    """Print parameter counts for all configs."""
    from s0_architecture import SutraS0

    for name, cfg_fn in ALL_CONFIGS.items():
        cfg = cfg_fn()
        model = SutraS0(cfg)
        counts = model.count_parameters()
        print(f"{name:>6s}: {counts['total']:>12,} ({counts['total']/1e6:.1f}M) "
              f"| P={cfg.patch_size} D={cfg.d_model} L={cfg.n_layers} "
              f"H={cfg.n_heads}/{cfg.n_kv_heads}")


if __name__ == "__main__":
    main()
