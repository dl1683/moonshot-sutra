"""Compare E2 ablation runs and emit advisory decision-rule verdicts.

This utility reads frozen-eval JSON files and/or training JSONL logs,
summarizes the metrics needed by the E2 ablation gate, and prints
human-readable PASS/FAIL/REGRESS checks. It does not launch training and it
does not turn advisory comparisons into scientific claims by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    "bpb",
    "first_byte_acc",
    "bpb_high_nll",
    "bpb_high_entropy",
    "bpb_high_disagreement",
    "bpb_control",
    "first_byte_acc_high_nll",
    "first_byte_acc_high_entropy",
    "first_byte_acc_high_disagreement",
    "first_byte_acc_control",
)


DECISION_RULES = [
    ("A2", "A0", 0.02, "E2 beats CE-only"),
    ("A2", "A1", 0.02, "E2 beats anchor-only KD"),
    ("A2", "BLD", 0.02, "E2 beats raw byte-KL"),
    ("A2", "A5", 0.02, "Router beats uniform static mixing"),
    ("A2", "A6", 0.02, "Real teacher targets beat shuffled targets"),
    ("A9c", "A5a", 0.02, "Gold-free router beats prior static"),
    ("A9c", "A5b", 0.02, "Gold-free router beats tuned static"),
    ("A9c", "A5c", 0.02, "5-teacher routed beats X-Token-style 2-teacher"),
]


GOLDFREE_RULES = [
    ("A9c", "A2", 0.02, "A5b", "Gold-free router works"),
]


@dataclass
class RunSummary:
    ablation_id: str
    log_path: str
    eval_result: dict[str, Any] = field(default_factory=dict)
    train_entries: list[dict[str, Any]] = field(default_factory=list)
    hard_failures: list[dict[str, Any]] = field(default_factory=list)
    final_train_bpb: float | None = None
    grad_budget_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, Any]:
        metrics = self.eval_result.get("metrics") if self.eval_result else None
        return metrics if isinstance(metrics, dict) else {}

    def metric(self, key: str) -> float | None:
        raw = self.metrics.get(key)
        if raw is None and key == "bpb":
            raw = self.final_train_bpb
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return value


def _finite_number(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: {path}:{line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                entries.append(obj)
    return entries


def _grad_budget_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    usable = []
    for entry in entries:
        gb = entry.get("grad_budget")
        if not isinstance(gb, dict) or gb.get("enabled") is False:
            continue
        if "total_scale" not in gb and "ce_grad_norm" not in gb:
            continue
        usable.append(gb)
    if not usable:
        return {}

    total_scales = [
        float(gb["total_scale"]) for gb in usable
        if _finite_number(gb.get("total_scale"))
    ]
    ce_norms = [
        float(gb["ce_grad_norm"]) for gb in usable
        if _finite_number(gb.get("ce_grad_norm"))
    ]
    stats: dict[str, Any] = {"n_entries": len(usable)}
    if total_scales:
        stats["mean_total_scale"] = sum(total_scales) / len(total_scales)
        stats["min_total_scale"] = min(total_scales)
    if ce_norms:
        stats["mean_ce_grad_norm"] = sum(ce_norms) / len(ce_norms)
    return stats


def analyze_run(ablation_id: str, log_path: str) -> RunSummary:
    entries = _load_jsonl(log_path)
    train_entries = [
        e for e in entries
        if "ce_loss" in e or "bpb" in e or "teacher_losses_bits" in e
    ]
    hard_failures = [
        e for e in entries
        if "HARD_FAIL" in e or str(e.get("status", "")).lower() == "hard_fail"
    ]
    final_train_bpb = None
    for entry in reversed(train_entries):
        if _finite_number(entry.get("bpb")):
            final_train_bpb = float(entry["bpb"])
            break
    eval_entries = [
        e for e in entries
        if _finite_number(e.get("eval_bpb")) or _finite_number(e.get("bpb"))
    ]
    eval_result: dict[str, Any] = {}
    if eval_entries:
        latest = eval_entries[-1]
        metrics = {}
        if _finite_number(latest.get("eval_bpb")):
            metrics["bpb"] = float(latest["eval_bpb"])
        elif _finite_number(latest.get("bpb")):
            metrics["bpb"] = float(latest["bpb"])
        if _finite_number(latest.get("first_byte_acc")):
            metrics["first_byte_acc"] = float(latest["first_byte_acc"])
        eval_result = {"metrics": metrics}

    return RunSummary(
        ablation_id=ablation_id,
        log_path=log_path,
        eval_result=eval_result,
        train_entries=train_entries,
        hard_failures=hard_failures,
        final_train_bpb=final_train_bpb,
        grad_budget_stats=_grad_budget_stats(train_entries),
    )


def load_eval_results(paths: list[str]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"warning: eval result not found: {raw}", file=sys.stderr)
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: could not read {raw}: {exc}", file=sys.stderr)
            continue
        if not isinstance(result, dict):
            print(f"warning: eval result is not an object: {raw}", file=sys.stderr)
            continue
        ablation_id = result.get("ablation_id") or path.stem.upper()
        results[str(ablation_id)] = result
    return results


def _summary_by_id(summaries: list[RunSummary]) -> dict[str, RunSummary]:
    return {s.ablation_id: s for s in summaries}


def _bpb(summary: RunSummary | None) -> float | None:
    return summary.metric("bpb") if summary is not None else None


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _print_pair_rule(
    better: RunSummary,
    worse: RunSummary,
    margin: float,
    message: str,
) -> None:
    better_bpb = _bpb(better)
    worse_bpb = _bpb(worse)
    if better_bpb is None or worse_bpb is None:
        print(
            f"[VOID] {message}: missing BPB "
            f"({better.ablation_id}={_fmt(better_bpb)}, "
            f"{worse.ablation_id}={_fmt(worse_bpb)})"
        )
        return

    delta = worse_bpb - better_bpb
    if delta >= margin:
        status = "[PASS]"
        detail = f"delta={delta:.4f} >= {margin:.4f}"
    elif delta < 0:
        status = "[REGRESS]"
        detail = f"WORSE by {-delta:.4f}"
    else:
        status = "[FAIL]"
        detail = f"delta={delta:.4f} < {margin:.4f}"
    print(
        f"{status} {message}: {better.ablation_id}={better_bpb:.4f}, "
        f"{worse.ablation_id}={worse_bpb:.4f}, {detail}"
    )


def _print_goldfree_rule(
    candidate: RunSummary,
    oracle_ref: RunSummary,
    static_ref: RunSummary,
    margin: float,
    message: str,
) -> None:
    cand_bpb = _bpb(candidate)
    oracle_bpb = _bpb(oracle_ref)
    static_bpb = _bpb(static_ref)
    if cand_bpb is None or oracle_bpb is None or static_bpb is None:
        print(
            f"[VOID] {message}: missing BPB "
            f"({candidate.ablation_id}={_fmt(cand_bpb)}, "
            f"{oracle_ref.ablation_id}={_fmt(oracle_bpb)}, "
            f"{static_ref.ablation_id}={_fmt(static_bpb)})"
        )
        return
    close_to_oracle = cand_bpb <= oracle_bpb + margin
    beats_static = static_bpb - cand_bpb >= margin
    if close_to_oracle and beats_static:
        print(
            f"[PASS] {message}: {candidate.ablation_id}={cand_bpb:.4f}, "
            f"{oracle_ref.ablation_id}={oracle_bpb:.4f}, "
            f"{static_ref.ablation_id}={static_bpb:.4f}"
        )
    elif static_bpb < cand_bpb:
        print(
            f"[REGRESS] {message}: {candidate.ablation_id}={cand_bpb:.4f} "
            f"is WORSE than {static_ref.ablation_id}={static_bpb:.4f}"
        )
    else:
        print(
            f"[FAIL] {message}: close_to_oracle={close_to_oracle}, "
            f"beats_static={beats_static}"
        )


def evaluate_decision_rules(summaries: list[RunSummary]) -> None:
    by_id = _summary_by_id(summaries)
    evaluated = 0
    for better_id, worse_id, margin, message in DECISION_RULES:
        better = by_id.get(better_id)
        worse = by_id.get(worse_id)
        if better is None or worse is None:
            continue
        _print_pair_rule(better, worse, margin, message)
        evaluated += 1

    for candidate_id, oracle_id, margin, static_id, message in GOLDFREE_RULES:
        candidate = by_id.get(candidate_id)
        oracle = by_id.get(oracle_id)
        static = by_id.get(static_id) if static_id is not None else None
        if candidate is None or oracle is None or static is None:
            continue
        _print_goldfree_rule(candidate, oracle, static, margin, message)
        evaluated += 1

    if evaluated == 0:
        print("No decision rules could be evaluated")


def _format_csv_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(f):
        return ""
    return f"{f:.4f}"


def export_csv(summaries: list[RunSummary], output: str) -> None:
    fieldnames = [
        "ablation_id",
        "log_path",
        "final_train_bpb",
        "hard_fail_count",
        "grad_budget_entries",
    ]
    fieldnames.extend(f"eval_{key}" for key in METRIC_KEYS)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row: dict[str, Any] = {
                "ablation_id": summary.ablation_id,
                "log_path": summary.log_path,
                "final_train_bpb": _format_csv_float(summary.final_train_bpb),
                "hard_fail_count": len(summary.hard_failures),
                "grad_budget_entries": summary.grad_budget_stats.get("n_entries", 0),
            }
            for key in METRIC_KEYS:
                row[f"eval_{key}"] = _format_csv_float(summary.metric(key))
            writer.writerow(row)


def _infer_ablation_id(log_path: str) -> str:
    ablation_id = Path(log_path).stem.upper()
    if ablation_id.startswith("E2_"):
        ablation_id = ablation_id[3:]
    return ablation_id


def _build_summaries(logs: list[str], eval_paths: list[str]) -> list[RunSummary]:
    eval_results = load_eval_results(eval_paths)
    summaries: dict[str, RunSummary] = {}
    for log_path in logs:
        ablation_id = _infer_ablation_id(log_path)
        summaries[ablation_id] = analyze_run(ablation_id, log_path)
    for ablation_id, eval_result in eval_results.items():
        current = summaries.get(ablation_id)
        if current is None:
            summaries[ablation_id] = RunSummary(
                ablation_id=ablation_id,
                log_path="",
                eval_result=eval_result,
            )
        else:
            current.eval_result = eval_result
    return list(summaries.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare E2 ablation metrics")
    parser.add_argument("--logs", nargs="*", default=[],
                        help="Training JSONL logs to summarize")
    parser.add_argument("--eval-results", nargs="*", default=[],
                        help="Frozen eval JSON files from eval_e2.py")
    parser.add_argument("--csv", default=None,
                        help="Optional CSV summary output path")
    args = parser.parse_args(argv)

    summaries = _build_summaries(args.logs, args.eval_results)
    if not summaries:
        print("No runs loaded", file=sys.stderr)
        return 1
    evaluate_decision_rules(summaries)
    if args.csv:
        export_csv(summaries, args.csv)
        print(f"CSV saved: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
