"""Training monitor — live dashboard for S0 and E2 training logs.

Reads the JSONL log file and displays a summary of training progress,
loss trajectory, throughput, and anomaly detection.

Usage:
    python monitor.py                              # default: logs/s0_burnin.jsonl
    python monitor.py --log logs/s0_train.jsonl
    python monitor.py --log logs/e2_train.jsonl    # E2 mode (auto-detected)
    python monitor.py --watch                      # auto-refresh every 10s
"""

import json
import math
import sys
import time


def load_entries(log_path: str) -> tuple[list[dict], list[dict]]:
    train, eval_ = [], []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if "HARD_FAIL" in entry:
                    train.append(entry)
                elif "eval_bpb" in entry or "eval_loss" in entry:
                    eval_.append(entry)
                elif "bpb" in entry or "ce_loss" in entry:
                    train.append(entry)
    except FileNotFoundError:
        pass
    return train, eval_


def detect_mode(train: list[dict]) -> str:
    """Detect log format: 's0' or 'e2'."""
    for entry in train[:5]:
        if "phase" in entry or "teacher_losses_bits" in entry or "teacher_losses" in entry or "ce_loss" in entry:
            return "e2"
    return "s0"


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def display_s0(train: list[dict], eval_: list[dict], log_path: str):
    latest = train[-1]
    first = train[0]

    print(f"\n{'=' * 60}")
    print(f"  S0 Training Monitor — {log_path}")
    print(f"{'=' * 60}")

    step = latest["step"]
    print(f"\n  Step: {step}")
    print(f"  Train BPB: {first['bpb']:.3f} -> {latest['bpb']:.3f}  "
          f"(delta: {latest['bpb'] - first['bpb']:+.3f})")

    if latest.get("tok_per_sec"):
        print(f"  Throughput: {latest['tok_per_sec']:,.0f} bytes/s")
    if latest.get("lr"):
        print(f"  Learning rate: {latest['lr']:.2e}")
    if latest.get("grad_norm"):
        print(f"  Grad norm: {latest['grad_norm']:.4f}")

    if eval_:
        latest_eval = eval_[-1]
        first_eval = eval_[0]
        print(f"\n  Eval BPB: {first_eval['eval_bpb']:.3f} -> "
              f"{latest_eval['eval_bpb']:.3f}  "
              f"(delta: {latest_eval['eval_bpb'] - first_eval['eval_bpb']:+.3f})")
        if "eval_byte_acc" in latest_eval:
            print(f"  Byte accuracy: {latest_eval['eval_byte_acc']:.4f}")
        if "eval_pos_acc" in latest_eval:
            pos = latest_eval["eval_pos_acc"]
            print(f"  Per-position: "
                  f"{' '.join(f'p{i}={a:.4f}' for i, a in enumerate(pos))}")
        gap = latest_eval["eval_bpb"] - latest["bpb"]
        print(f"  Train/eval gap: {gap:+.3f} BPB")

    print("\n  Recent loss trajectory:")
    recent = train[-min(10, len(train)):]
    for entry in recent:
        bar_width = int(max(0, min(40, (entry["bpb"] / 8.0) * 40)))
        bar = "#" * bar_width
        gnorm = (f"gnorm={entry.get('grad_norm', 0):.2f}"
                 if "grad_norm" in entry else "")
        print(f"    step {entry['step']:>5d}: bpb {entry['bpb']:.3f} "
              f"|{bar:<40s}| {gnorm}")

    if len(eval_) > 1:
        print("\n  Eval trajectory:")
        for entry in eval_:
            bar_width = int(max(0, min(40, (entry["eval_bpb"] / 8.0) * 40)))
            bar = "#" * bar_width
            acc_str = (f"acc={entry.get('eval_byte_acc', 0):.4f}"
                       if "eval_byte_acc" in entry else "")
            print(f"    step {entry['step']:>5d}: bpb {entry['eval_bpb']:.3f} "
                  f"|{bar:<40s}| {acc_str}")


def display_e2(train: list[dict], eval_: list[dict], log_path: str):
    latest = train[-1]
    first = train[0]

    print(f"\n{'=' * 60}")
    print(f"  E2 Multi-Teacher KD Monitor — {log_path}")
    print(f"{'=' * 60}")

    step = latest["step"]
    phase = latest.get("phase", "?")
    ce = latest.get("ce_loss", 0)
    ce_bpb = ce / math.log(2) if ce > 0 else 0
    first_ce = first.get("ce_loss", 0)
    first_bpb = first_ce / math.log(2) if first_ce > 0 else 0

    print(f"\n  Step: {step}")
    print(f"  Phase: {phase}")
    print(f"  CE BPB: {first_bpb:.3f} -> {ce_bpb:.3f}  "
          f"(delta: {ce_bpb - first_bpb:+.3f})")

    if latest.get("lr"):
        print(f"  Learning rate: {latest['lr']:.2e}")
    if latest.get("elapsed"):
        print(f"  Elapsed: {format_time(latest['elapsed'])}")

    teacher_losses = latest.get("teacher_losses_bits", latest.get("teacher_losses", {}))
    if teacher_losses:
        print("\n  Teacher losses (bits):")
        for tname, tloss in sorted(teacher_losses.items()):
            print(f"    {tname}: {tloss:.4f}")

    phases_seen = set()
    phase_transitions = []
    for entry in train:
        p = entry.get("phase", "?")
        if p not in phases_seen:
            phases_seen.add(p)
            phase_transitions.append((entry["step"], p))

    if len(phase_transitions) > 1:
        print("\n  Phase transitions:")
        for s, p in phase_transitions:
            print(f"    step {s:>6d}: {p}")

    if eval_:
        latest_eval = eval_[-1]
        first_eval = eval_[0]
        print(f"\n  Eval BPB: {first_eval.get('eval_bpb', 0):.3f} -> "
              f"{latest_eval.get('eval_bpb', 0):.3f}  "
              f"(delta: {latest_eval.get('eval_bpb', 0) - first_eval.get('eval_bpb', 0):+.3f})")
        gap = latest_eval.get("eval_bpb", 0) - ce_bpb
        print(f"  Train/eval gap: {gap:+.3f} BPB")

    print("\n  Recent loss trajectory:")
    recent = train[-min(10, len(train)):]
    for entry in recent:
        bpb_val = entry.get("bpb", 0)
        if not bpb_val:
            ce_val = entry.get("ce_loss", 0)
            bpb_val = ce_val / math.log(2) if ce_val > 0 else 0
        bar_width = int(max(0, min(40, (bpb_val / 8.0) * 40)))
        bar = "#" * bar_width
        p = entry.get("phase", "?")
        tl = entry.get("teacher_losses_bits", entry.get("teacher_losses", {}))
        tl_str = " ".join(f"{k}={v:.3f}" for k, v in tl.items()) if tl else ""
        print(f"    step {entry['step']:>5d}: bpb {bpb_val:.3f} [{p}] "
              f"|{bar:<40s}| {tl_str}")

    if len(eval_) > 1:
        print("\n  Eval trajectory:")
        for entry in eval_:
            bpb_val = entry.get("eval_bpb", 0)
            bar_width = int(max(0, min(40, (bpb_val / 8.0) * 40)))
            bar = "#" * bar_width
            p = entry.get("phase", "?")
            print(f"    step {entry['step']:>5d}: bpb {bpb_val:.3f} [{p}] "
                  f"|{bar:<40s}|")


def display(log_path: str):
    train, eval_ = load_entries(log_path)

    if not train:
        print(f"No entries in {log_path}")
        return

    fails = [e for e in train if "HARD_FAIL" in e]
    if fails:
        print(f"\n  *** HARD FAIL at step {fails[-1]['step']}: "
              f"{fails[-1]['HARD_FAIL']} ***\n")

    train = [e for e in train if "HARD_FAIL" not in e]
    if not train:
        return

    mode = detect_mode(train)
    if mode == "e2":
        display_e2(train, eval_, log_path)
    else:
        display_s0(train, eval_, log_path)

    anomalies = []
    if len(train) >= 3:
        if mode == "e2":
            losses = [e.get("ce_loss", 0) / math.log(2) for e in train]
        else:
            losses = [e.get("bpb", 0) for e in train]
        for i in range(2, len(losses)):
            if losses[i] > losses[i-1] + 0.5:
                anomalies.append(
                    f"Loss spike at step {train[i]['step']}: "
                    f"{losses[i-1]:.3f} -> {losses[i]:.3f}")

    gnorms = [e.get("grad_norm", 0) for e in train]
    high_gnorm = sum(1 for g in gnorms if g > 10)
    if high_gnorm > len(gnorms) * 0.2:
        anomalies.append(
            f"High grad norm (>10) in {high_gnorm}/{len(gnorms)} steps "
            f"({high_gnorm/len(gnorms)*100:.0f}%)")

    if anomalies:
        print("\n  Anomalies:")
        for a in anomalies:
            print(f"    ! {a}")

    if len(train) >= 2 and "elapsed" in train[-1]:
        elapsed = train[-1]["elapsed"]
        steps_done = train[-1]["step"] - train[0]["step"] + 1
        if steps_done > 0:
            sec_per_step = elapsed / steps_done
            print(f"\n  Avg: {sec_per_step:.2f}s/step "
                  f"({1/sec_per_step:.2f} steps/s)")

    print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/s0_burnin.jsonl")
    parser.add_argument("--watch", action="store_true",
                        help="Auto-refresh every 10s")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                sys.stdout.write("\033[2J\033[H")
                display(args.log)
                time.sleep(10)
        except KeyboardInterrupt:
            pass
    else:
        display(args.log)


if __name__ == "__main__":
    main()
