"""Ablation comparison tool — post-training analysis for E2 experiments.

Reads training JSONL logs from multiple ablation runs and produces
comparison tables, phase breakdowns, and routing analysis.

Usage:
    python compare_ablations.py \
        --logs A2=logs/e2_a2.jsonl A0=logs/e2_a0.jsonl A1=logs/e2_a1.jsonl
    python compare_ablations.py \
        --logs A2=logs/e2_a2.jsonl A0=logs/e2_a0.jsonl \
        --eval-results ablations/a2.json ablations/a0.json
    python compare_ablations.py \
        --logs A2=logs/e2_a2.jsonl --phase-breakdown
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunSummary:
    ablation_id: str
    log_path: str
    total_steps: int = 0
    final_ce_bpb: float = 0.0
    initial_ce_bpb: float = 0.0
    best_eval_bpb: float = float("inf")
    final_eval_bpb: float = 0.0
    phases_seen: list[str] = field(default_factory=list)
    phase_metrics: dict[str, dict] = field(default_factory=dict)
    route_stats: dict = field(default_factory=dict)
    grad_budget_stats: dict = field(default_factory=dict)
    teacher_loss_final: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def load_log(path: str) -> tuple[list[dict], list[dict]]:
    train, eval_ = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "HARD_FAIL" in entry:
                continue
            if "eval_bpb" in entry or "eval_loss" in entry:
                eval_.append(entry)
            elif "ce_loss" in entry or "bpb" in entry:
                train.append(entry)
    return train, eval_


def ce_to_bpb(ce_nats: float) -> float:
    return ce_nats / math.log(2) if ce_nats > 0 else 0.0


def analyze_run(ablation_id: str, log_path: str) -> RunSummary:
    train, eval_ = load_log(log_path)
    if not train:
        return RunSummary(ablation_id=ablation_id, log_path=log_path)

    summary = RunSummary(
        ablation_id=ablation_id,
        log_path=log_path,
        total_steps=train[-1].get("step", 0),
    )

    bpb_key = "bpb" if "bpb" in train[0] else None
    ce_key = "ce_loss" if "ce_loss" in train[0] else None

    if bpb_key:
        summary.initial_ce_bpb = train[0][bpb_key]
        summary.final_ce_bpb = train[-1][bpb_key]
    elif ce_key:
        summary.initial_ce_bpb = ce_to_bpb(train[0][ce_key])
        summary.final_ce_bpb = ce_to_bpb(train[-1][ce_key])

    if eval_:
        best = min(eval_, key=lambda e: e.get("eval_bpb", float("inf")))
        summary.best_eval_bpb = best.get("eval_bpb", float("inf"))
        summary.final_eval_bpb = eval_[-1].get("eval_bpb", 0)

    if train[-1].get("elapsed"):
        summary.elapsed_seconds = train[-1]["elapsed"]

    phases_seen = []
    current_phase = None
    phase_entries = defaultdict(list)
    for entry in train:
        p = entry.get("phase", "s0")
        if p != current_phase:
            if p not in phases_seen:
                phases_seen.append(p)
            current_phase = p
        phase_entries[p].append(entry)
    summary.phases_seen = phases_seen

    for phase, entries in phase_entries.items():
        if not entries:
            continue
        if bpb_key:
            bpbs = [e[bpb_key] for e in entries]
        elif ce_key:
            bpbs = [ce_to_bpb(e[ce_key]) for e in entries]
        else:
            bpbs = []

        phase_summary = {
            "n_steps": len(entries),
            "start_step": entries[0].get("step", 0),
            "end_step": entries[-1].get("step", 0),
        }
        if bpbs:
            phase_summary["entry_bpb"] = round(bpbs[0], 4)
            phase_summary["exit_bpb"] = round(bpbs[-1], 4)
            phase_summary["delta_bpb"] = round(bpbs[-1] - bpbs[0], 4)
            phase_summary["min_bpb"] = round(min(bpbs), 4)

        tl_entries = [e for e in entries
                      if e.get("teacher_losses_bits") or e.get("teacher_losses")]
        if tl_entries:
            all_teachers = set()
            for e in tl_entries:
                tl = e.get("teacher_losses_bits", e.get("teacher_losses", {}))
                all_teachers.update(tl.keys())
            teacher_means = {}
            for t in sorted(all_teachers):
                vals = []
                for e in tl_entries:
                    tl = e.get("teacher_losses_bits", e.get("teacher_losses", {}))
                    if t in tl:
                        vals.append(tl[t])
                if vals:
                    teacher_means[t] = round(sum(vals) / len(vals), 4)
            phase_summary["teacher_loss_means"] = teacher_means

        summary.phase_metrics[phase] = phase_summary

    tl = train[-1].get("teacher_losses_bits", train[-1].get("teacher_losses", {}))
    summary.teacher_loss_final = {k: round(v, 4) for k, v in tl.items()}

    route_entries = [e for e in train if e.get("route_stats")]
    if route_entries:
        jsds = [e["route_stats"].get("mean_jsd", 0) for e in route_entries]
        entropies = [e["route_stats"].get("mean_route_entropy", 0) for e in route_entries]
        n_routed = [e["route_stats"].get("n_routed", 0) for e in route_entries]
        summary.route_stats = {
            "mean_jsd": round(sum(jsds) / len(jsds), 4),
            "mean_entropy": round(sum(entropies) / len(entropies), 4),
            "mean_n_routed": round(sum(n_routed) / len(n_routed), 2),
            "n_entries": len(route_entries),
        }

        last_route = route_entries[-1]["route_stats"]
        if "avg_teacher_weights" in last_route:
            summary.route_stats["final_teacher_weights"] = {
                k: round(v, 4)
                for k, v in last_route["avg_teacher_weights"].items()
            }

    gb_entries = [e for e in train if e.get("grad_budget")]
    if gb_entries:
        ce_norms = [e["grad_budget"]["ce_grad_norm"] for e in gb_entries]
        total_scales = [e["grad_budget"]["total_scale"] for e in gb_entries]
        summary.grad_budget_stats = {
            "mean_ce_grad_norm": round(sum(ce_norms) / len(ce_norms), 6),
            "mean_total_scale": round(sum(total_scales) / len(total_scales), 4),
            "min_total_scale": round(min(total_scales), 4),
            "max_total_scale": round(max(total_scales), 4),
            "n_entries": len(gb_entries),
        }

    return summary


def print_comparison_table(summaries: list[RunSummary]):
    print(f"\n{'=' * 72}")
    print("  ABLATION COMPARISON TABLE")
    print(f"{'=' * 72}")

    header = f"{'Metric':<30s}"
    for s in summaries:
        header += f" {s.ablation_id:>12s}"
    print(f"\n{header}")
    print("-" * (30 + 13 * len(summaries)))

    def row(label: str, vals: list, fmt: str = ".4f"):
        line = f"  {label:<28s}"
        for v in vals:
            if v is None or v == float("inf"):
                line += f" {'N/A':>12s}"
            else:
                line += f" {v:>12{fmt}}"
        print(line)

    row("Total steps", [s.total_steps for s in summaries], "d")
    row("Initial CE BPB", [s.initial_ce_bpb for s in summaries])
    row("Final CE BPB", [s.final_ce_bpb for s in summaries])
    row("CE BPB delta", [s.final_ce_bpb - s.initial_ce_bpb for s in summaries])
    row("Best eval BPB", [s.best_eval_bpb for s in summaries])
    row("Final eval BPB", [s.final_eval_bpb for s in summaries])
    row("Elapsed (hours)",
        [s.elapsed_seconds / 3600 if s.elapsed_seconds else None for s in summaries],
        ".2f")

    if len(summaries) >= 2:
        ref = summaries[0]
        print(f"\n  Deltas vs {ref.ablation_id}:")
        for s in summaries[1:]:
            delta_final = s.final_ce_bpb - ref.final_ce_bpb
            delta_eval = (s.best_eval_bpb - ref.best_eval_bpb
                          if s.best_eval_bpb < float("inf")
                          and ref.best_eval_bpb < float("inf")
                          else None)
            print(f"    {s.ablation_id}: CE BPB {delta_final:+.4f}"
                  + (f", eval BPB {delta_eval:+.4f}" if delta_eval is not None else ""))


def print_phase_breakdown(summaries: list[RunSummary]):
    print(f"\n{'=' * 72}")
    print("  PHASE BREAKDOWN")
    print(f"{'=' * 72}")

    for s in summaries:
        print(f"\n  --- {s.ablation_id} ({s.log_path}) ---")
        if not s.phase_metrics:
            print("    No phase data.")
            continue
        for phase in s.phases_seen:
            pm = s.phase_metrics.get(phase, {})
            print(f"\n    {phase} (steps {pm.get('start_step', '?')}-{pm.get('end_step', '?')}, "
                  f"n={pm.get('n_steps', 0)})")
            if "entry_bpb" in pm:
                print(f"      BPB: {pm['entry_bpb']} -> {pm['exit_bpb']} "
                      f"(delta: {pm['delta_bpb']:+.4f}, min: {pm['min_bpb']})")
            if "teacher_loss_means" in pm:
                print("      Mean teacher losses (bits):")
                for t, v in pm["teacher_loss_means"].items():
                    print(f"        {t}: {v}")


def print_routing_analysis(summaries: list[RunSummary]):
    print(f"\n{'=' * 72}")
    print("  ROUTING ANALYSIS")
    print(f"{'=' * 72}")

    for s in summaries:
        rs = s.route_stats
        if not rs:
            print(f"\n  {s.ablation_id}: No routing data.")
            continue
        print(f"\n  {s.ablation_id}:")
        print(f"    Mean JSD: {rs.get('mean_jsd', 0):.4f}")
        print(f"    Mean route entropy: {rs.get('mean_entropy', 0):.4f}")
        print(f"    Mean n_routed: {rs.get('mean_n_routed', 0):.1f}")
        if "final_teacher_weights" in rs:
            print("    Final teacher weights:")
            for t, w in rs["final_teacher_weights"].items():
                bar_len = int(w * 40)
                bar = "#" * bar_len
                print(f"      {t:<30s} {w:.4f} |{bar}|")


def print_gradient_budget_analysis(summaries: list[RunSummary]):
    print(f"\n{'=' * 72}")
    print("  GRADIENT BUDGET ANALYSIS")
    print(f"{'=' * 72}")

    for s in summaries:
        gb = s.grad_budget_stats
        if not gb:
            print(f"\n  {s.ablation_id}: No gradient budget data.")
            continue
        print(f"\n  {s.ablation_id}:")
        print(f"    Mean CE grad norm: {gb.get('mean_ce_grad_norm', 0):.6f}")
        print(f"    Mean total scale: {gb.get('mean_total_scale', 0):.4f}")
        print(f"    Scale range: [{gb.get('min_total_scale', 0):.4f}, "
              f"{gb.get('max_total_scale', 0):.4f}]")

        if gb.get("mean_total_scale", 1) < 0.1:
            print("    *** WARNING: Teachers being aggressively clipped "
                  "(mean scale < 0.1) ***")


def export_csv(summaries: list[RunSummary], path: str):
    with open(path, "w") as f:
        cols = ["ablation_id", "total_steps", "initial_ce_bpb", "final_ce_bpb",
                "ce_bpb_delta", "best_eval_bpb", "final_eval_bpb",
                "elapsed_hours", "mean_jsd", "mean_route_entropy"]
        f.write(",".join(cols) + "\n")
        for s in summaries:
            vals = [
                s.ablation_id,
                str(s.total_steps),
                f"{s.initial_ce_bpb:.4f}",
                f"{s.final_ce_bpb:.4f}",
                f"{s.final_ce_bpb - s.initial_ce_bpb:.4f}",
                f"{s.best_eval_bpb:.4f}" if s.best_eval_bpb < float("inf") else "",
                f"{s.final_eval_bpb:.4f}",
                f"{s.elapsed_seconds / 3600:.2f}" if s.elapsed_seconds else "",
                f"{s.route_stats.get('mean_jsd', '')}" if s.route_stats else "",
                f"{s.route_stats.get('mean_entropy', '')}" if s.route_stats else "",
            ]
            f.write(",".join(vals) + "\n")
    print(f"\n  CSV exported: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare E2 ablation training logs")
    parser.add_argument(
        "--logs", nargs="+", required=True,
        help="Ablation logs as ID=path pairs (e.g. A2=logs/e2_a2.jsonl)")
    parser.add_argument(
        "--phase-breakdown", action="store_true",
        help="Show per-phase BPB breakdown")
    parser.add_argument(
        "--routing", action="store_true",
        help="Show routing statistics analysis")
    parser.add_argument(
        "--grad-budget", action="store_true",
        help="Show gradient budget analysis")
    parser.add_argument(
        "--csv", default="",
        help="Export comparison to CSV file")
    parser.add_argument(
        "--all", action="store_true",
        help="Show all analysis sections")
    args = parser.parse_args()

    summaries = []
    for spec in args.logs:
        if "=" in spec:
            ablation_id, path = spec.split("=", 1)
        else:
            ablation_id = Path(spec).stem
            path = spec
        if not Path(path).exists():
            print(f"WARNING: {path} not found, skipping.", file=sys.stderr)
            continue
        summaries.append(analyze_run(ablation_id, path))

    if not summaries:
        print("No valid logs found.", file=sys.stderr)
        sys.exit(1)

    print_comparison_table(summaries)

    if args.phase_breakdown or args.all:
        print_phase_breakdown(summaries)
    if args.routing or args.all:
        print_routing_analysis(summaries)
    if args.grad_budget or args.all:
        print_gradient_budget_analysis(summaries)
    if args.csv:
        export_csv(summaries, args.csv)


if __name__ == "__main__":
    main()
