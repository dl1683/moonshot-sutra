"""Pre-training preflight checks.

S0 mode: validates GPU, data, model, checkpoints, tests.
E2 mode: validates E1 checkpoint, E2 cache integrity, OneDrive guard, tests.

Usage:
    python preflight.py [--config p4|p8|d640|d768] [--data-dir DIR]
    python preflight.py --cpu-only
    python preflight.py --mode e2 [--e1-checkpoint PATH] [--e2-cache-dir DIR]
"""

import os
import sys
from pathlib import Path


def check_gpu():
    """Check GPU availability and actual free VRAM."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        name = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        total = total_bytes / 1e9
        free = free_bytes / 1e9
        return True, f"{name}, {total:.1f} GB total, {free:.1f} GB free"
    except Exception as e:
        return False, str(e)


def check_data(data_dir: str, patch_size: int = 4, seq_len: int = 4096):
    """Check data shards exist and are readable."""
    shards = sorted(Path(data_dir).glob("*.bin"))
    if not shards:
        return False, f"No .bin shards in {data_dir}"

    total_bytes = sum(s.stat().st_size for s in shards)
    total_gb = total_bytes / 1e9

    n_eval = min(2, max(1, len(shards) // 10))
    train_shards = len(shards) - n_eval

    # Check seq_len divisibility
    sample_size = shards[0].stat().st_size
    n_seqs = sample_size // seq_len
    if n_seqs == 0:
        return False, f"Shard too small ({sample_size} bytes) for seq_len={seq_len}"

    # Check patch alignment
    if seq_len % patch_size != 0:
        return False, f"seq_len ({seq_len}) not divisible by patch_size ({patch_size})"

    # Try reading first shard
    try:
        with open(shards[0], "rb") as f:
            sample = f.read(1024)
        if len(sample) < 1024:
            return False, f"First shard too small: {len(sample)} bytes"
    except Exception as e:
        return False, f"Cannot read first shard: {e}"

    return True, f"{len(shards)} shards ({total_gb:.1f} GB), {train_shards} train / {n_eval} eval, ~{n_seqs} seqs/shard"


def check_model(config_name: str):
    """Check model builds and report parameter count."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from s0_configs import ALL_CONFIGS
        from s0_architecture import SutraS0

        import torch
        cfg = ALL_CONFIGS[config_name]()
        model = SutraS0(cfg)
        counts = model.count_parameters()
        total_m = counts['total'] / 1e6

        batch = torch.randint(0, 256, (1, cfg.patch_size * 4))
        out = model(batch)
        assert "logits" in out

        return True, f"{total_m:.1f}M params, forward pass OK"
    except Exception as e:
        return False, f"Model build failed: {e}"


def check_checkpoints(ckpt_dir: str):
    """Check checkpoint directory is writable."""
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        test_file = os.path.join(ckpt_dir, ".preflight_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True, f"{ckpt_dir} writable"
    except Exception as e:
        return False, f"Cannot write to {ckpt_dir}: {e}"


def check_opsec():
    """Scan tracked files for banned model names."""
    import subprocess
    code_dir = os.path.dirname(__file__)
    result = subprocess.run(
        [sys.executable, os.path.join(code_dir, "check_opsec.py")],
        capture_output=True, text=True, timeout=30,
        cwd=os.path.dirname(code_dir),
    )
    if result.returncode == 0:
        return True, "No banned model names in tracked files"
    last_line = result.stdout.strip().split("\n")[-1] if result.stdout else "check failed"
    return False, last_line


def check_e1_checkpoint(ckpt_path: str):
    """Check E1 checkpoint exists and loads."""
    if not os.path.exists(ckpt_path):
        return False, f"E1 checkpoint not found: {ckpt_path}"
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        step = ckpt.get("step", "?")
        has_align = "align_proj" in ckpt
        return True, f"Step {step}, align_proj={'yes' if has_align else 'no'}"
    except Exception as e:
        return False, f"Cannot load: {e}"


def check_e2_cache(cache_dir: str, data_dir: str):
    """Validate E2 cache integrity."""
    if not os.path.isdir(cache_dir):
        return False, f"Cache dir not found: {cache_dir}"
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from eklavya_e2_cache import E2CacheView
        with E2CacheView(cache_dir) as view:
            errors = view.validate(data_dir=data_dir)
            n_pos = view.n_positions
            n_teachers = len(view._teacher_names)
        if errors:
            return False, f"{len(errors)} error(s): {errors[0]}"
        return True, f"{n_pos} positions, {n_teachers} teachers"
    except Exception as e:
        return False, f"Cache load failed: {e}"


def check_onedrive_path(ckpt_dir: str):
    """Warn if checkpoint dir is under OneDrive."""
    norm = os.path.normpath(ckpt_dir).lower()
    if "onedrive" in norm:
        return False, f"Checkpoint dir {ckpt_dir} is under OneDrive — use C:/sutra_fast/"
    return True, f"{ckpt_dir} OK (not under OneDrive)"


def check_tests():
    """Run all test suites (S0 + E1 + E2)."""
    import subprocess
    code_dir = os.path.dirname(__file__)
    test_files = ["test_overfit.py", "test_eklavya.py", "test_eklavya_e2.py",
                   "test_burnin_verdict.py", "test_export_log_csv.py",
                   "test_utilities.py"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + test_files + ["-q"],
        capture_output=True, text=True, timeout=300, cwd=code_dir,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    if result.returncode == 0:
        return True, "All tests passed"
    last_lines = result.stdout.strip().split("\n")[-3:] + result.stderr.strip().split("\n")[-3:]
    return False, f"Tests failed: {' | '.join(l for l in last_lines if l)}"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["s0", "e2"], default="s0",
                        help="Preflight mode: s0 (default) or e2")
    parser.add_argument("--config", choices=["p4", "p8", "d640", "d768"], default="p4")
    parser.add_argument("--data-dir", default="data/shards_bytes_full")
    parser.add_argument("--burnin", action="store_true")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip GPU check, validate everything else on CPU")
    parser.add_argument("--e1-checkpoint",
                        default="C:/sutra_fast/checkpoints/e1/e1_best.pt")
    parser.add_argument("--e2-cache-dir",
                        default="C:/sutra_fast/eklavya_e2_cache")
    parser.add_argument("--e2-output-dir",
                        default="C:/sutra_fast/checkpoints/e2")
    args = parser.parse_args()

    if args.mode == "e2":
        return _preflight_e2(args)
    return _preflight_s0(args)


def _preflight_e2(args):
    checks = [
        ("Opsec", lambda: check_opsec()),
    ]
    if not args.cpu_only:
        checks.append(("GPU", lambda: check_gpu()))
    checks.extend([
        ("E1 Checkpoint", lambda: check_e1_checkpoint(args.e1_checkpoint)),
        ("E2 Cache", lambda: check_e2_cache(args.e2_cache_dir, args.data_dir)),
        ("OneDrive Guard", lambda: check_onedrive_path(args.e2_output_dir)),
        ("Output Dir", lambda: check_checkpoints(args.e2_output_dir)),
        ("Tests", lambda: check_tests()),
    ])

    mode_label = "CPU-only" if args.cpu_only else "GPU"
    print("=" * 60)
    print(f"E2 Pre-Training Preflight — {mode_label}")
    print("=" * 60)

    all_ok = True
    for name, check_fn in checks:
        ok, msg = check_fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        if args.cpu_only:
            print("ALL CPU CHECKS PASSED. Re-run without --cpu-only when GPU is free.")
        else:
            print("ALL CHECKS PASSED. Launch E2 training.")
    else:
        print("SOME CHECKS FAILED. Fix issues before E2 training.")

    return 0 if all_ok else 1


def _preflight_s0(args):
    ckpt_dir = "checkpoints/s0_burnin" if args.burnin else "checkpoints/s0"

    checks = [
        ("Opsec", lambda: check_opsec()),
    ]
    if not args.cpu_only:
        checks.append(("GPU", lambda: check_gpu()))
    checks.extend([
        ("Data", lambda: check_data(args.data_dir)),
        ("Model", lambda: check_model(args.config)),
        ("Checkpoints", lambda: check_checkpoints(ckpt_dir)),
        ("Tests", lambda: check_tests()),
    ])

    mode_label = "CPU-only" if args.cpu_only else f"config={args.config}"
    print("=" * 60)
    print(f"S0 Pre-Training Preflight — {mode_label}, burnin={args.burnin}")
    print("=" * 60)

    all_ok = True
    gpu_ok = args.cpu_only
    for name, check_fn in checks:
        ok, msg = check_fn()
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {msg}")
        if not ok:
            all_ok = False
        if name == "GPU" and ok:
            gpu_ok = True

    print()
    if all_ok:
        if args.cpu_only:
            print("ALL CPU CHECKS PASSED. Re-run without --cpu-only when GPU is free.")
        else:
            mode = "--burnin" if args.burnin else ""
            print("ALL CHECKS PASSED. Launch with:")
            print(f"  python code/s0_training.py --config {args.config} {mode}")
    elif not gpu_ok:
        print("GPU not available. Training cannot start yet.")
        print("Other checks ran on CPU — re-run preflight when GPU is free.")
    else:
        print("SOME CHECKS FAILED. Fix issues before training.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
