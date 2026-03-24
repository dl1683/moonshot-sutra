"""Analyze F2 teacher-free A/B test results.

Compares KD-ON vs KD-OFF arms and makes the R20 decision:
  - KD-off within 0.05 BPT at D10+D12 → O4 drops, KD leaves critical path
  - KD-on wins >=0.10 at D10 or D12 + majority watcher tasks → KD survives

Usage:
  python code/analyze_f2.py
"""

import json, sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def load_det_eval(run_name):
    """Load deterministic eval from a run's latest checkpoint."""
    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    # Check for saved det-eval results
    for name in ["det_eval.json", "deterministic_eval_comparison.json"]:
        p = ckpt_dir / name
        if p.exists():
            return json.load(open(p))
    return None


def load_metrics(run_name):
    """Load training metrics."""
    p = REPO / "results" / f"{run_name}_metrics.json"
    if p.exists():
        return json.load(open(p))
    return None


def main():
    print("=" * 60)
    print("F2: TEACHER-FREE A/B TEST ANALYSIS")
    print("=" * 60)

    # Load det-eval results from comparison JSON
    comp_path = REPO / "results" / "deterministic_eval_comparison.json"
    if comp_path.exists():
        results = json.load(open(comp_path))
        print(f"\nLoaded {len(results)} comparison results from {comp_path.name}")
        for r in results:
            print(f"\n  {r['checkpoint']}:")
            print(f"    Step: {r['step']}")
            for d, bpt in sorted(r['per_depth_bpt'].items(), key=lambda x: int(x[0])):
                print(f"    D{d}: {bpt}")
    else:
        print(f"\nNo comparison file found at {comp_path}")
        print("Run: python code/train_p1_twoteacher.py --det-eval "
              "results/checkpoints_f2_kd_on/rolling_latest.pt,"
              "results/checkpoints_f2_kd_off/rolling_latest.pt")
        return

    # Extract KD-ON and KD-OFF results
    kd_on = next((r for r in results if "kd_on" in r["checkpoint"]), None)
    kd_off = next((r for r in results if "kd_off" in r["checkpoint"]), None)

    if not kd_on or not kd_off:
        print("\nERROR: Could not identify KD-ON and KD-OFF results.")
        print("Checkpoint names must contain 'kd_on' and 'kd_off'.")
        return

    print(f"\n{'=' * 60}")
    print("COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Depth':<10} {'KD-ON':>10} {'KD-OFF':>10} {'Delta':>10} {'Winner':>10}")
    print("-" * 50)

    d10_on = float(kd_on["per_depth_bpt"].get("10", 99))
    d10_off = float(kd_off["per_depth_bpt"].get("10", 99))
    d12_on = float(kd_on["per_depth_bpt"].get("12", 99))
    d12_off = float(kd_off["per_depth_bpt"].get("12", 99))

    for d in ["1", "4", "8", "10", "12"]:
        on_bpt = float(kd_on["per_depth_bpt"].get(d, 99))
        off_bpt = float(kd_off["per_depth_bpt"].get(d, 99))
        delta = off_bpt - on_bpt  # positive means KD-ON is better
        winner = "KD-ON" if delta > 0.01 else ("KD-OFF" if delta < -0.01 else "TIE")
        print(f"D{d:<9} {on_bpt:>10.4f} {off_bpt:>10.4f} {delta:>+10.4f} {winner:>10}")

    print(f"\n{'=' * 60}")
    print("R20 DECISION CRITERIA")
    print(f"{'=' * 60}")

    d10_delta = d10_off - d10_on
    d12_delta = d12_off - d12_on

    print(f"\nD10: KD-ON={d10_on:.4f}, KD-OFF={d10_off:.4f}, delta={d10_delta:+.4f}")
    print(f"D12: KD-ON={d12_on:.4f}, KD-OFF={d12_off:.4f}, delta={d12_delta:+.4f}")

    # Decision
    within_005 = abs(d10_delta) < 0.05 and abs(d12_delta) < 0.05
    kd_wins_010 = d10_delta > 0.10 or d12_delta > 0.10

    if within_005:
        print(f"\n*** DECISION: O4 DROPS ***")
        print(f"KD-OFF is within 0.05 BPT at D10 ({d10_delta:+.4f}) and D12 ({d12_delta:+.4f})")
        print(f"KD provides no causal improvement. Remove from critical path.")
    elif kd_wins_010:
        print(f"\n*** DECISION: KD SURVIVES (pending watcher task confirmation) ***")
        print(f"KD-ON wins by >= 0.10 on {'D10' if d10_delta > 0.10 else 'D12'}")
        print(f"Still need majority (4/7) watcher task wins to confirm.")
        print(f"Run lm-eval on both checkpoints for final decision.")
    else:
        print(f"\n*** DECISION: INCONCLUSIVE ***")
        print(f"Delta is between 0.05 and 0.10. Need more steps or watcher tasks.")
        print(f"Consider extending F2 to 2000 steps per arm.")


if __name__ == "__main__":
    main()
