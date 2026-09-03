"""Eklavya Embedding Pipeline — canonical single-command runner.

Runs the full extract → train → evaluate pipeline in one shot.

Usage:
  # Quick smoke test (toy data, small teachers, CPU)
  python code/run.py --mode smoke

  # Real run (real teachers, GPU)
  python code/run.py --mode real --device cuda

  # Extract only (save signatures for later)
  python code/run.py --mode extract --device cuda

  # Train from pre-extracted signatures
  python code/run.py --mode train --signatures data/embed_signatures.jsonl --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add code/ to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embed_tomography import (
    generate_toy_pairs,
    extract_signatures,
    save_signatures,
    load_signatures,
    load_model,
    eval_retrieval,
)
from data_loader import load_hard_toy, load_pairs
from train_student import train


SMOKE_TEACHERS = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "BAAI/bge-small-en-v1.5",
]

REAL_TEACHERS = [
    "sentence-transformers/all-MiniLM-L12-v2",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
]

STUDENT_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"


def run_smoke(args):
    """Quick smoke test on hard-negative toy data to verify pipeline works end-to-end."""
    print("=" * 60)
    print("SMOKE TEST — hard-negative toy data, small teachers, fast")
    print("=" * 60)

    pairs = load_hard_toy(n=30, n_docs=8)
    print(f"\nGenerated {len(pairs)} hard-negative toy pairs")

    sigs = extract_signatures(SMOKE_TEACHERS, pairs, device=args.device)
    sig_path = os.path.join(args.out_dir, "smoke_signatures.jsonl")
    save_signatures(sigs, sig_path)

    eval_pairs = [
        {"query": s.query, "documents": s.documents, "gold_idx": s.gold_idx}
        for s in sigs
    ]

    student = train(
        student_name=args.student or STUDENT_DEFAULT,
        signatures=sigs,
        out_dir=os.path.join(args.out_dir, "smoke"),
        steps=args.steps or 100,
        lr=args.lr or 2e-5,
        tau=args.tau or 0.05,
        device=args.device,
        eval_pairs=eval_pairs,
        log_every=10,
        save_every=50,
    )
    print("\nSmoke test complete.")


def run_extract(args):
    """Extract teacher signatures and save for later training runs."""
    teachers = args.teachers or REAL_TEACHERS
    print(f"Extracting signatures from {len(teachers)} teachers")

    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        n = args.n_pairs or 200
        print(f"No data file, generating {n} toy pairs")
        pairs = generate_toy_pairs(n=n)

    sigs = extract_signatures(teachers, pairs, device=args.device)
    sig_path = args.signatures or os.path.join(args.out_dir, "embed_signatures.jsonl")
    save_signatures(sigs, sig_path)
    print(f"Done. {len(sigs)} signatures saved.")


def run_train(args):
    """Train student from pre-extracted signatures."""
    sig_path = args.signatures
    if not sig_path or not os.path.exists(sig_path):
        print(f"Signatures file not found: {sig_path}")
        print("Run with --mode extract first, or --mode real for end-to-end")
        sys.exit(1)

    sigs = load_signatures(sig_path)
    eval_pairs = [
        {"query": s.query, "documents": s.documents, "gold_idx": s.gold_idx}
        for s in sigs
    ]

    train(
        student_name=args.student or STUDENT_DEFAULT,
        signatures=sigs,
        out_dir=args.out_dir,
        steps=args.steps or 500,
        lr=args.lr or 2e-5,
        tau=args.tau or 0.05,
        device=args.device,
        eval_pairs=eval_pairs,
    )


def run_real(args):
    """Full pipeline: extract from real teachers, train, evaluate."""
    print("=" * 60)
    print("REAL RUN — heterogeneous teachers, full training")
    print("=" * 60)

    teachers = args.teachers or REAL_TEACHERS

    if args.data:
        pairs = load_pairs(args.data, n=args.n_pairs or 200)
    else:
        n = args.n_pairs or 200
        print(f"No data file, generating {n} hard-negative toy pairs")
        pairs = load_hard_toy(n=n)

    # Extract
    sigs = extract_signatures(teachers, pairs, device=args.device)
    sig_path = os.path.join(args.out_dir, "embed_signatures.jsonl")
    save_signatures(sigs, sig_path)

    # Train
    eval_pairs = [
        {"query": s.query, "documents": s.documents, "gold_idx": s.gold_idx}
        for s in sigs
    ]

    train(
        student_name=args.student or STUDENT_DEFAULT,
        signatures=sigs,
        out_dir=args.out_dir,
        steps=args.steps or 500,
        lr=args.lr or 2e-5,
        tau=args.tau or 0.05,
        device=args.device,
        eval_pairs=eval_pairs,
    )

    # Control comparison: standard KD (no probes, just identity query)
    print("\n" + "=" * 60)
    print("CONTROL — standard KD (identity probe only)")
    print("=" * 60)

    control_sigs = []
    for sig in sigs:
        identity_only = {
            tname: {"identity": tsig.get("identity", list(tsig.values())[0])}
            for tname, tsig in sig.teacher_sigs.items()
        }
        from embed_tomography import EmbedSignature
        control_sigs.append(EmbedSignature(
            pair_id=sig.pair_id,
            query=sig.query,
            documents=sig.documents,
            probes=[p for p in sig.probes if p["probe_id"] == "identity"],
            teacher_sigs=identity_only,
            gold_idx=sig.gold_idx,
        ))

    control_dir = os.path.join(args.out_dir, "control_identity_only")
    train(
        student_name=args.student or STUDENT_DEFAULT,
        signatures=control_sigs,
        out_dir=control_dir,
        steps=args.steps or 500,
        lr=args.lr or 2e-5,
        tau=args.tau or 0.05,
        device=args.device,
        eval_pairs=eval_pairs,
    )


def run_eval(args):
    """Evaluate a trained student vs baseline."""
    model_path = args.model_path
    if not model_path:
        model_path = os.path.join(args.out_dir, "final")

    baseline_name = args.student or STUDENT_DEFAULT

    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        pairs = generate_toy_pairs(n=args.n_pairs or 100)

    print(f"Loading baseline: {baseline_name}")
    baseline = load_model(baseline_name, args.device)
    base_metrics = eval_retrieval(baseline, pairs)
    del baseline

    print(f"Loading trained student: {model_path}")
    student = load_model(model_path, args.device)
    student_metrics = eval_retrieval(student, pairs)

    print(f"\nRetained Gain Report")
    print(f"  Baseline Hit@5:  {base_metrics['hit_at_k']:.4f}")
    print(f"  Student  Hit@5:  {student_metrics['hit_at_k']:.4f}")
    print(f"  Owned gain:      {student_metrics['hit_at_k'] - base_metrics['hit_at_k']:+.4f}")
    print(f"  Baseline MRR:    {base_metrics['mrr']:.4f}")
    print(f"  Student  MRR:    {student_metrics['mrr']:.4f}")
    print(f"  Owned gain:      {student_metrics['mrr'] - base_metrics['mrr']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Eklavya Embedding Pipeline Runner")
    parser.add_argument("--mode", choices=["smoke", "extract", "train", "real", "eval"],
                        default="smoke")
    parser.add_argument("--teachers", nargs="+", default=None)
    parser.add_argument("--student", type=str, default=None)
    parser.add_argument("--signatures", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/eklavya_v0")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--n_pairs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    dispatch = {
        "smoke": run_smoke,
        "extract": run_extract,
        "train": run_train,
        "real": run_real,
        "eval": run_eval,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
