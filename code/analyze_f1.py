"""F1: Select best parent checkpoint (5K vs 6K) for F2 experiments.

Compares deterministic eval results from both checkpoints.
Decision: pick the one with LOWER D10. If tie, use step 6000 (more training).

Usage:
  python code/analyze_f1.py
"""

import json, sys, torch
from pathlib import Path

REPO = Path(__file__).parent.parent
CKPT_DIR = REPO / "results" / "checkpoints_p1"


def load_det_eval_from_checkpoint(step):
    """Load per-depth BPT from a step checkpoint."""
    p = CKPT_DIR / f"step_{step}.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    return {
        "step": ckpt["step"],
        "bpt": ckpt.get("bpt", 99),
        "per_depth_bpt": ckpt.get("per_depth_bpt", {}),
    }


def main():
    print("=" * 60)
    print("F1: SELECT BEST PARENT CHECKPOINT")
    print("=" * 60)

    s5 = load_det_eval_from_checkpoint(5000)
    s6 = load_det_eval_from_checkpoint(6000)

    if not s5:
        print("ERROR: step_5000.pt not found")
        return
    if not s6:
        print("ERROR: step_6000.pt not found. Training hasn't reached step 6000 yet.")
        return

    # Also check deterministic eval log if available
    det_5k = None
    det_6k = None
    det_log_5k = REPO / "results" / "det_eval_step5000.log"
    det_log_6k = REPO / "results" / "det_eval_step6000.log"

    print(f"\n{'Metric':<15} {'Step 5000':>12} {'Step 6000':>12} {'Delta':>10} {'Winner':>10}")
    print("-" * 60)

    for d in [1, 4, 8, 10, 12]:
        v5 = s5["per_depth_bpt"].get(d, s5["per_depth_bpt"].get(str(d), 99))
        v6 = s6["per_depth_bpt"].get(d, s6["per_depth_bpt"].get(str(d), 99))
        delta = v6 - v5  # negative means 6K is better
        winner = "6K" if delta < -0.01 else ("5K" if delta > 0.01 else "TIE")
        print(f"D{d:<14} {v5:>12.4f} {v6:>12.4f} {delta:>+10.4f} {winner:>10}")

    d10_5k = s5["per_depth_bpt"].get(10, s5["per_depth_bpt"].get("10", 99))
    d10_6k = s6["per_depth_bpt"].get(10, s6["per_depth_bpt"].get("10", 99))
    d12_5k = s5["per_depth_bpt"].get(12, s5["per_depth_bpt"].get("12", 99))
    d12_6k = s6["per_depth_bpt"].get(12, s6["per_depth_bpt"].get("12", 99))

    bpt_5k = s5["bpt"]
    bpt_6k = s6["bpt"]

    print(f"\n{'BPT (D=12)':<15} {bpt_5k:>12.4f} {bpt_6k:>12.4f} {bpt_6k - bpt_5k:>+10.4f}")

    print(f"\n{'=' * 60}")
    print("DECISION")
    print(f"{'=' * 60}")

    # Primary criterion: lower D10
    if d10_6k < d10_5k - 0.01:
        winner = 6000
        reason = f"Step 6K D10 ({d10_6k:.4f}) beats Step 5K D10 ({d10_5k:.4f})"
    elif d10_5k < d10_6k - 0.01:
        winner = 5000
        reason = f"Step 5K D10 ({d10_5k:.4f}) beats Step 6K D10 ({d10_6k:.4f})"
    else:
        # Tie-break: use step 6000 (more training)
        winner = 6000
        reason = f"D10 tie ({d10_5k:.4f} vs {d10_6k:.4f}), defaulting to step 6K (more training)"

    print(f"\n*** WINNER: Step {winner} ***")
    print(f"Reason: {reason}")
    print(f"\nNext: Copy rolling_latest.pt (step {winner}) to F2 checkpoint dirs")
    print(f"  cp results/checkpoints_p1/rolling_latest.pt results/checkpoints_f2_kd_on/rolling_latest.pt")
    print(f"  cp results/checkpoints_p1/rolling_latest.pt results/checkpoints_f2_kd_off/rolling_latest.pt")

    return winner


if __name__ == "__main__":
    main()
