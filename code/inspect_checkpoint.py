"""Checkpoint inspector — verify checkpoint contents and metadata.

Loads a checkpoint on CPU and reports its contents, step, phase,
model config, optimizer state, and parameter statistics. No GPU needed.

Usage:
    python inspect_checkpoint.py C:/sutra_fast/checkpoints/e2/e2_best.pt
    python inspect_checkpoint.py --dir C:/sutra_fast/checkpoints/e2/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def inspect_checkpoint(path: str):
    p = Path(path)
    if not p.exists():
        print(f"  NOT FOUND: {path}")
        return

    size_mb = p.stat().st_size / (1024 * 1024)
    print(f"\n  Checkpoint: {path}")
    print(f"  Size: {size_mb:.1f} MB")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return

    print(f"  Keys: {sorted(ckpt.keys())}")

    if "step" in ckpt:
        print(f"  Step: {ckpt['step']}")
    if "phase" in ckpt:
        print(f"  Phase: {ckpt['phase']}")
    if "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            for k, v in sorted(cfg.items()):
                print(f"    config.{k}: {v}")
        else:
            print(f"  Config: {cfg}")
    if "model_cfg" in ckpt:
        cfg = ckpt["model_cfg"]
        if hasattr(cfg, "__dict__"):
            for k, v in sorted(vars(cfg).items()):
                if not k.startswith("_"):
                    print(f"    model_cfg.{k}: {v}")
        else:
            print(f"  Model config: {cfg}")

    if "model" in ckpt:
        state = ckpt["model"]
        n_params = sum(p.numel() for p in state.values())
        n_tensors = len(state)
        print(f"\n  Model state: {n_tensors} tensors, "
              f"{n_params:,} parameters ({n_params * 2 / 1e6:.1f} MB bf16)")

        for name, tensor in sorted(state.items()):
            if tensor.numel() == 0:
                print(f"    EMPTY: {name} {tensor.shape}")
            elif tensor.isnan().any():
                print(f"    NAN: {name} {tensor.shape}")
            elif tensor.isinf().any():
                print(f"    INF: {name} {tensor.shape}")

    if "optimizer" in ckpt:
        opt = ckpt["optimizer"]
        n_groups = len(opt.get("param_groups", []))
        print(f"\n  Optimizer: {n_groups} param groups")
        for i, pg in enumerate(opt.get("param_groups", [])):
            lr = pg.get("lr", "?")
            wd = pg.get("weight_decay", "?")
            n = len(pg.get("params", []))
            print(f"    group {i}: lr={lr}, wd={wd}, n_params={n}")

    for rng_key in ("rng_state", "cuda_rng_state", "py_rng_state", "np_rng_state"):
        if rng_key in ckpt:
            print(f"  RNG state: {rng_key} present")

    if "ports" in ckpt:
        ports_state = ckpt["ports"]
        n_port_params = sum(p.numel() for p in ports_state.values())
        print(f"\n  Projection ports: {len(ports_state)} tensors, "
              f"{n_port_params:,} params")

    if "best_eval_bpb" in ckpt:
        print(f"  Best eval BPB: {ckpt['best_eval_bpb']:.4f}")

    if "scaler" in ckpt:
        print(f"  GradScaler state present")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect checkpoint files without GPU")
    parser.add_argument("path", nargs="?", help="Path to .pt checkpoint")
    parser.add_argument("--dir", default="",
                        help="Directory to scan for .pt files")
    args = parser.parse_args()

    if args.dir:
        d = Path(args.dir)
        if not d.exists():
            print(f"Directory not found: {args.dir}", file=sys.stderr)
            sys.exit(1)
        pts = sorted(d.glob("*.pt"))
        if not pts:
            print(f"No .pt files in {args.dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(pts)} checkpoints in {args.dir}")
        for pt in pts:
            inspect_checkpoint(str(pt))
    elif args.path:
        inspect_checkpoint(args.path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
