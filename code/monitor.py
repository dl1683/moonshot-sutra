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
                    if "eval_bpb" not in entry and "eval_loss" in entry:
                        entry["eval_bpb"] = entry["eval_loss"] / math.log(2)
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

    route = latest.get("route_stats", {})
    if route:
        print("\n  Router stats:")
        if "mean_route_entropy" in route:
            print(f"    Route entropy: {route['mean_route_entropy']:.4f}")
        if "mean_jsd" in route:
            print(f"    Mean JSD: {route['mean_jsd']:.4f}")
        if "n_routed" in route:
            print(f"    Positions routed: {route['n_routed']}")
        if "avg_teacher_weights" in route:
            wt = route["avg_teacher_weights"]
            print(f"    Avg weights: {' '.join(f'{k}={v:.3f}' for k, v in sorted(wt.items()))}")

    gb = latest.get("grad_budget", {})
    if gb and gb.get("total_scale") is not None:
        print(f"\n  Gradient budget:")
        print(f"    CE grad norm: {gb.get('ce_grad_norm', 0):.4f}")
        print(f"    Total scale: {gb.get('total_scale', 0):.4f}")
        scales = gb.get("per_teacher_scales", {})
        if scales:
            print(f"    Per-teacher: {' '.join(f'{k}={v:.3f}' for k, v in sorted(scales.items()))}")
        cosines = gb.get("ce_teacher_cosines", {})
        if cosines:
            print(f"    CE-teacher cos: {' '.join(f'{k}={v:+.3f}' for k, v in sorted(cosines.items()))}")
        coherence = gb.get("pairwise_coherence")
        if coherence is not None:
            print(f"    Pairwise coherence: {coherence:+.4f}")

    if latest.get("gpu_mem_gb") is not None:
        print(f"\n  GPU memory (peak): {latest['gpu_mem_gb']:.2f} GB")

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


def _e2_anomalies(train: list[dict],
                  eval_: list[dict] | None = None) -> list[str]:
    """E2-specific anomaly detection."""
    anomalies = []

    nan_steps = [e["step"] for e in train
                 if not math.isfinite(e.get("ce_loss", 0))]
    if nan_steps:
        anomalies.append(
            f"Non-finite CE loss at step(s) {nan_steps[:5]} — "
            "model has diverged")

    nan_teacher_steps = []
    for e in train:
        tl = e.get("teacher_losses_nats", e.get("teacher_losses", {}))
        if isinstance(tl, dict):
            for v in tl.values():
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    nan_teacher_steps.append(e["step"])
                    break
    if nan_teacher_steps:
        anomalies.append(
            f"Non-finite teacher loss at step(s) {nan_teacher_steps[:5]} — "
            "check NaN guards in router/purifier")

    route_entropies = [
        e["route_stats"]["mean_route_entropy"]
        for e in train if e.get("route_stats", {}).get("mean_route_entropy") is not None
    ]
    if len(route_entropies) >= 10:
        recent = route_entropies[-10:]
        if max(recent) < 0.1:
            anomalies.append(
                f"Route entropy collapse: last 10 readings all < 0.1 "
                f"(max {max(recent):.4f}) — router locked to one teacher")

    grad_scales = [
        e["grad_budget"]["total_scale"]
        for e in train if e.get("grad_budget", {}).get("total_scale") is not None
    ]
    if len(grad_scales) >= 5:
        recent = grad_scales[-5:]
        if max(recent) < 0.01:
            anomalies.append(
                f"Gradient budget near zero: last 5 total_scale < 0.01 "
                f"(max {max(recent):.6f}) — teacher signal suppressed")

    zero_teacher_count = 0
    total_with_teachers = 0
    for e in train:
        tl = e.get("teacher_losses_bits", e.get("teacher_losses"))
        if isinstance(tl, dict) and len(tl) > 0:
            total_with_teachers += 1
            if all(v == 0 or v is None for v in tl.values()):
                zero_teacher_count += 1
    if total_with_teachers >= 10 and zero_teacher_count > total_with_teachers * 0.5:
        anomalies.append(
            f"Zero teacher signal in {zero_teacher_count}/{total_with_teachers} "
            f"steps ({zero_teacher_count/total_with_teachers*100:.0f}%) — "
            "cache coverage may be insufficient")

    _DISAGREEMENT_PHASES = ("E2.4_disagreement", "DISAGREEMENT", "E2.4")
    zero_route_count = sum(
        1 for e in train
        if e.get("route_stats", {}).get("n_routed", 0) == 0
        and e.get("phase") in _DISAGREEMENT_PHASES
    )
    disagreement_count = sum(
        1 for e in train
        if e.get("phase") in _DISAGREEMENT_PHASES
    )
    if disagreement_count >= 5 and zero_route_count > disagreement_count * 0.3:
        anomalies.append(
            f"No routed positions in {zero_route_count}/{disagreement_count} "
            f"disagreement steps — router may not be activating")

    anomalies.extend(_phase_boundary_checks(train, eval_))

    return anomalies


def _phase_boundary_checks(train: list[dict],
                           eval_: list[dict] | None = None) -> list[str]:
    """Phase-specific intervention checks from E2 Monitoring Protocol."""
    anomalies = []
    if eval_ is None:
        eval_ = []

    phase_entries: dict[str, list[dict]] = {}
    for e in train:
        p = e.get("phase", "")
        phase_entries.setdefault(p, []).append(e)

    _PORT = ("E2.1_port_warmup", "PORT_WARMUP", "E2.1")
    _CONS = ("E2.2_consensus", "CONSENSUS", "E2.2")
    _SEM = ("E2.3_semantic_landing", "SEMANTIC_LANDING", "E2.3")
    _DIS = ("E2.4_disagreement", "DISAGREEMENT", "E2.4")

    def _get_phase(names):
        for n in names:
            if n in phase_entries and phase_entries[n]:
                return phase_entries[n]
        return []

    port_entries = _get_phase(_PORT)
    cons_entries = _get_phase(_CONS)
    sem_entries = _get_phase(_SEM)
    dis_entries = _get_phase(_DIS)

    if port_entries and eval_:
        port_end = port_entries[-1]["step"]
        port_evals = [e for e in eval_ if e["step"] <= port_end]
        pre_evals = [e for e in eval_ if e["step"] <= port_entries[0]["step"]]
        if port_evals and pre_evals:
            baseline_bpb = pre_evals[0].get("eval_bpb", 0)
            port_end_bpb = port_evals[-1].get("eval_bpb", 0)
            if baseline_bpb > 0 and port_end_bpb - baseline_bpb > 0.02:
                anomalies.append(
                    f"PORT_WARMUP: eval BPB regressed {port_end_bpb - baseline_bpb:+.3f} "
                    f"(>{0.02}) — student should be frozen, check freeze config")

    if cons_entries:
        n_routed_vals = [
            e["route_stats"]["n_routed"]
            for e in cons_entries
            if e.get("route_stats", {}).get("n_routed") is not None
        ]
        if len(n_routed_vals) >= 5:
            recent = n_routed_vals[-5:]
            if max(recent) < 4:
                anomalies.append(
                    f"CONSENSUS: n_routed consistently <4 "
                    f"(max recent: {max(recent)}) — router underactivated")

        weights = []
        for e in cons_entries:
            rs = e.get("route_stats", {})
            atw = rs.get("avg_teacher_weights", {})
            if atw:
                weights.append(atw)
        if len(weights) >= 5:
            recent = weights[-5:]
            non_anchor = []
            for w in recent:
                total = sum(v for k, v in w.items()
                            if "anchor" not in k.lower())
                non_anchor.append(total)
            if non_anchor and max(non_anchor) < 0.05:
                anomalies.append(
                    "CONSENSUS: control teacher weight <0.05 in recent steps "
                    "— anchor dominates, control not contributing")

    if sem_entries:
        sem_loss_present = 0
        for e in sem_entries:
            tl = e.get("teacher_losses_bits", e.get("teacher_losses", {}))
            if isinstance(tl, dict):
                has_sem = any("semantic" in k.lower() for k in tl)
                if has_sem:
                    sem_loss_present += 1
        if len(sem_entries) >= 5 and sem_loss_present < len(sem_entries) * 0.8:
            anomalies.append(
                f"SEMANTIC_LANDING: semantic loss absent in "
                f"{len(sem_entries) - sem_loss_present}/{len(sem_entries)} "
                f"steps (>20%) — semantic teacher not reaching student")

    if len(dis_entries) >= 50:
        dis_entropies = [
            e["route_stats"]["mean_route_entropy"]
            for e in dis_entries
            if e.get("route_stats", {}).get("mean_route_entropy") is not None
        ]
        if len(dis_entropies) >= 50:
            recent = dis_entropies[-50:]
            if min(recent) > 1.30:
                anomalies.append(
                    f"DISAGREEMENT: route entropy >1.30 for 50+ readings "
                    f"(min: {min(recent):.3f}) — near-uniform routing, "
                    f"router not learning")
            elif max(recent) < 0.20:
                anomalies.append(
                    f"DISAGREEMENT: route entropy <0.20 for 50+ readings "
                    f"(max: {max(recent):.3f}) — routing collapse, "
                    f"one teacher dominates")

    return anomalies


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

    if mode == "e2":
        anomalies.extend(_e2_anomalies(train, eval_))

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
