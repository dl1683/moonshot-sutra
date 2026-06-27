"""Export JSONL training logs to CSV for plotting and analysis.

Usage:
    python export_log_csv.py --log logs/e2_a2.jsonl --output a2_trajectory.csv
    python export_log_csv.py --log logs/e2_a2.jsonl --eval-only --output a2_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def export_train_csv(log_path: str, output_path: str):
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "HARD_FAIL" in entry:
                continue
            if "eval_bpb" in entry or "eval_loss" in entry:
                continue

            step = entry.get("step", 0)
            phase = entry.get("phase", "s0")

            if "bpb" in entry:
                ce_bpb = entry["bpb"]
            elif "ce_loss" in entry:
                ce_bpb = entry["ce_loss"] / math.log(2)
            else:
                continue

            row = {
                "step": step,
                "phase": phase,
                "ce_bpb": round(ce_bpb, 6),
                "lr": entry.get("lr", ""),
                "grad_norm": round(entry["grad_norm"], 6) if "grad_norm" in entry else "",
                "elapsed_s": round(entry["elapsed"], 2) if "elapsed" in entry else "",
            }

            tl = entry.get("teacher_losses_bits", entry.get("teacher_losses", {}))
            for tname, tloss in sorted(tl.items()):
                row[f"tl_{tname}"] = round(tloss, 6)

            rs = entry.get("route_stats", {})
            if rs:
                row["jsd"] = round(rs.get("mean_jsd", 0), 6)
                row["route_entropy"] = round(rs.get("mean_route_entropy", 0), 6)
                row["n_routed"] = rs.get("n_routed", "")
                for tname, tw in rs.get("avg_teacher_weights", {}).items():
                    row[f"tw_{tname}"] = round(tw, 6)

            gb = entry.get("grad_budget", {})
            if gb:
                row["gb_ce_norm"] = round(gb.get("ce_grad_norm", 0), 6)
                row["gb_total_scale"] = round(gb.get("total_scale", 0), 6)

            rows.append(row)

    if not rows:
        print(f"No training entries in {log_path}", file=sys.stderr)
        sys.exit(1)

    all_keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} training entries to {output_path}")


def export_eval_csv(log_path: str, output_path: str):
    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "eval_bpb" not in entry and "eval_loss" not in entry:
                continue

            step = entry.get("step", 0)
            if "eval_bpb" in entry:
                bpb = entry["eval_bpb"]
            else:
                bpb = entry["eval_loss"] / math.log(2)
            row = {
                "step": step,
                "eval_bpb": round(bpb, 6),
                "phase": entry.get("phase", ""),
            }
            if "eval_byte_acc" in entry:
                row["eval_byte_acc"] = round(entry["eval_byte_acc"], 6)
            rows.append(row)

    if not rows:
        print(f"No eval entries in {log_path}", file=sys.stderr)
        sys.exit(1)

    all_keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} eval entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export JSONL training logs to CSV")
    parser.add_argument("--log", required=True, help="Path to JSONL log")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--eval-only", action="store_true",
                        help="Export only eval entries")
    args = parser.parse_args()

    if args.eval_only:
        export_eval_csv(args.log, args.output)
    else:
        export_train_csv(args.log, args.output)


if __name__ == "__main__":
    main()
