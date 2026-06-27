"""Burn-in verdict — apply R11 criteria to training log.

Reads the JSONL log from a 500-step burn-in and produces a GO/NO-GO verdict
based on R11 concrete failure/success criteria.

Usage: python burnin_verdict.py [--log logs/s0_burnin.jsonl]
"""

import json
import math
import sys
from pathlib import Path


# R11 criteria (warmup=100, 500 steps)
HARD_FAIL_BPB_500 = 7.0      # held-out BPB > 7.0 at step 500 = architecture bug
RANDOM_BPB = 8.0              # ~log2(256) = 8.0 for uniform random
HEALTHY_BPB_500 = (3.5, 5.0)  # expected range at step 500
ACCEPTABLE_BPB_500 = 5.5      # upper end of acceptable
POS0_MIN_BPB = 6.8            # position 0 must beat this
POS0_MIN_ACC = 0.03           # position 0 must exceed 3% accuracy


def load_log(path: str) -> tuple[list[dict], list[dict]]:
    """Split log entries into train steps and eval steps."""
    train, eval_ = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "HARD_FAIL" in entry:
                print(f"  HARD FAIL logged at step {entry['step']}: {entry['HARD_FAIL']}")
                sys.exit(1)
            if "eval_bpb" in entry:
                eval_.append(entry)
            elif "bpb" in entry:
                train.append(entry)
    return train, eval_


def check_hard_fails(train: list[dict], eval_: list[dict]) -> list[str]:
    fails = []

    if not train:
        fails.append("No training steps logged")
        return fails

    # NaN/Inf in logged values
    for entry in train:
        if not math.isfinite(entry.get("loss", 0)):
            fails.append(f"NaN/Inf loss at step {entry['step']}")
        if not math.isfinite(entry.get("grad_norm", 0)):
            fails.append(f"NaN/Inf grad_norm at step {entry['step']}")

    # BPB still > 7.0 at final step
    if eval_:
        final_eval = eval_[-1]
        if final_eval["eval_bpb"] > HARD_FAIL_BPB_500:
            fails.append(f"Eval BPB {final_eval['eval_bpb']:.3f} > {HARD_FAIL_BPB_500} at step {final_eval['step']}")

    # Train drops but eval stays near random
    if len(eval_) >= 2:
        last_eval = eval_[-1]
        if last_eval["eval_bpb"] > 7.5 and train[-1]["bpb"] < 5.0:
            fails.append(f"Train BPB {train[-1]['bpb']:.3f} dropped but eval BPB {last_eval['eval_bpb']:.3f} stuck near random — possible data leak or eval bug")

    # Position 0 checks (R11: must beat random)
    if eval_ and "eval_pos_acc" in eval_[-1]:
        pos_acc = eval_[-1]["eval_pos_acc"]
        if len(pos_acc) >= 4:
            p0_acc = pos_acc[0]
            p123_acc = sum(pos_acc[1:]) / len(pos_acc[1:])
            if p0_acc < 0.01 and p123_acc > 0.05:
                fails.append(f"Position 0 acc {p0_acc:.4f} near random while positions 1-3 avg {p123_acc:.4f} — global context not learning")
            if p0_acc < POS0_MIN_ACC:
                fails.append(f"Position 0 acc {p0_acc:.4f} < {POS0_MIN_ACC} minimum — global context insufficient")

    # Byte accuracy < 1% after 500 steps
    if eval_ and "eval_byte_acc" in eval_[-1]:
        if eval_[-1]["eval_byte_acc"] < 0.01:
            fails.append(f"Byte accuracy {eval_[-1]['eval_byte_acc']:.4f} < 1% — model not learning")

    # Grad norm repeatedly > 100
    high_gnorm = sum(1 for e in train if e.get("grad_norm", 0) > 100)
    if high_gnorm > len(train) * 0.3:
        fails.append(f"Grad norm > 100 in {high_gnorm}/{len(train)} steps ({high_gnorm/len(train)*100:.0f}%)")

    return fails


def check_soft_concerns(train: list[dict], eval_: list[dict]) -> list[str]:
    concerns = []

    if eval_:
        last_bpb = eval_[-1]["eval_bpb"]
        if HEALTHY_BPB_500[1] < last_bpb <= ACCEPTABLE_BPB_500 + 1.0:
            concerns.append(f"Eval BPB {last_bpb:.3f} above healthy range {HEALTHY_BPB_500} but still falling")

    # Train/eval gap
    if eval_ and train:
        gap = eval_[-1]["eval_bpb"] - train[-1]["bpb"]
        if 0.3 < gap < 0.7:
            concerns.append(f"Train/eval BPB gap {gap:.3f} — moderate, watch for overfitting")
        elif gap > 0.7:
            concerns.append(f"Train/eval BPB gap {gap:.3f} — high, likely overfitting")

    # Grad clipping frequency
    clipped = sum(1 for e in train if e.get("grad_norm", 0) > 0.9)
    if clipped > len(train) * 0.5:
        concerns.append(f"Grad clipping triggered {clipped}/{len(train)} steps — check LR")

    # Eval jitter
    if len(eval_) >= 3:
        bpbs = [e["eval_bpb"] for e in eval_]
        diffs = [abs(bpbs[i] - bpbs[i-1]) for i in range(1, len(bpbs))]
        max_jitter = max(diffs)
        if max_jitter > 0.2:
            concerns.append(f"Eval BPB jitter up to {max_jitter:.3f}")

    return concerns


def check_trajectory(train: list[dict], eval_: list[dict]) -> list[str]:
    """Check that the loss trajectory has the right shape: steep early, smooth decline."""
    notes = []

    if len(eval_) >= 3:
        bpbs = [e["eval_bpb"] for e in eval_]
        # First half should drop more than second half
        mid = len(bpbs) // 2
        first_drop = bpbs[0] - bpbs[mid]
        second_drop = bpbs[mid] - bpbs[-1]
        if first_drop > 0 and second_drop > 0:
            notes.append(f"Trajectory shape: steep→smooth (first half -{first_drop:.2f}, second half -{second_drop:.2f})")
        elif first_drop <= 0:
            notes.append("WARNING: BPB not decreasing in first half")
        elif second_drop < 0:
            notes.append("WARNING: BPB increasing in second half (eval_bpb going up)")

    # Monotonicity check on 3-point moving average
    if len(eval_) >= 4:
        bpbs = [e["eval_bpb"] for e in eval_]
        smoothed = [(bpbs[max(0,i-1)] + bpbs[i] + bpbs[min(len(bpbs)-1,i+1)]) / 3 for i in range(len(bpbs))]
        reversals = sum(1 for i in range(2, len(smoothed)) if smoothed[i] > smoothed[i-2] + 0.1)
        if reversals > 0:
            notes.append(f"Smoothed trajectory has {reversals} reversal(s) > 0.1 BPB")

    return notes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/s0_burnin.jsonl")
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"Log file not found: {args.log}")
        sys.exit(1)

    train, eval_ = load_log(args.log)
    print(f"Loaded {len(train)} train entries, {len(eval_)} eval entries\n")

    if train:
        print(f"Step range: {train[0]['step']} → {train[-1]['step']}")
        print(f"Train BPB: {train[0]['bpb']:.3f} → {train[-1]['bpb']:.3f}")
    if eval_:
        print(f"Eval BPB:  {eval_[0]['eval_bpb']:.3f} → {eval_[-1]['eval_bpb']:.3f}")
        if "eval_pos_acc" in eval_[-1]:
            pos = eval_[-1]["eval_pos_acc"]
            print(f"Final per-position accuracy: {' '.join(f'p{i}={a:.4f}' for i, a in enumerate(pos))}")
        if "eval_byte_acc" in eval_[-1]:
            print(f"Final byte accuracy: {eval_[-1]['eval_byte_acc']:.4f}")

    print("\n=== HARD FAILS ===")
    hard = check_hard_fails(train, eval_)
    if hard:
        for f in hard:
            print(f"  FAIL: {f}")
    else:
        print("  None — all hard criteria pass")

    print("\n=== SOFT CONCERNS ===")
    soft = check_soft_concerns(train, eval_)
    if soft:
        for s in soft:
            print(f"  CONCERN: {s}")
    else:
        print("  None")

    print("\n=== TRAJECTORY ===")
    traj = check_trajectory(train, eval_)
    for t in traj:
        print(f"  {t}")

    print("\n" + "=" * 50)
    if hard:
        print("VERDICT: NO-GO — hard failures detected. Fix before proceeding.")
        sys.exit(1)
    elif soft:
        print("VERDICT: CONDITIONAL GO — tune the flagged concerns, but architecture is sound.")
    else:
        print("VERDICT: GO — burn-in passed all criteria. Proceed to full training.")


if __name__ == "__main__":
    main()
