"""Eklavya Embedding Student Training — steal from teachers, own the invariants.

End-to-end pipeline:
  1. Load pre-extracted teacher signatures (or extract on the fly)
  2. Initialize a small student encoder
  3. Train on multi-teacher tomography loss
  4. Evaluate retained gain (student must own what it learned)

Usage:
  python code/train_student.py \
    --signatures data/embed_signatures.jsonl \
    --student sentence-transformers/all-MiniLM-L6-v2 \
    --out_dir outputs/eklavya_v0 \
    --steps 500 --lr 2e-5 --batch_size 8 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from embed_tomography import (
    EmbedSignature,
    generate_probes,
    generate_toy_pairs,
    extract_signatures,
    load_model,
    load_signatures,
    save_signatures,
    sims_to_distribution,
    eval_retrieval,
    Probe,
)


def prepare_targets(sig: EmbedSignature, tau: float = 0.05) -> dict:
    """Convert a signature into training targets: per-probe softmax distributions from each teacher."""
    targets = {}
    for probe_info in sig.probes:
        pid = probe_info["probe_id"]
        teacher_dists = []
        for tname, tsig in sig.teacher_sigs.items():
            if pid in tsig:
                teacher_dists.append(sims_to_distribution(tsig[pid], tau=tau))
        if teacher_dists:
            targets[pid] = {
                "text": probe_info["text"],
                "teacher_dists": teacher_dists,
            }
    return targets


def train_step(
    student,
    sig: EmbedSignature,
    targets: dict,
    tau: float = 0.05,
) -> torch.Tensor:
    """One training step: KL divergence between student and teacher ranking distributions."""
    doc_embs = student.encode(
        sig.documents, convert_to_tensor=True, normalize_embeddings=True
    )

    loss = torch.tensor(0.0, device=doc_embs.device, requires_grad=True)
    n = 0

    for pid, target_info in targets.items():
        q_emb = student.encode(
            [target_info["text"]], convert_to_tensor=True, normalize_embeddings=True
        )
        student_sims = (q_emb @ doc_embs.T).squeeze(0)
        student_log_dist = F.log_softmax(student_sims / tau, dim=0)

        for td in target_info["teacher_dists"]:
            td = td.to(student_log_dist.device)
            kl = F.kl_div(student_log_dist, td, reduction="batchmean", log_target=False)
            loss = loss + kl
            n += 1

    return loss / max(n, 1)


def train(
    student_name: str,
    signatures: list[EmbedSignature],
    out_dir: str,
    steps: int = 500,
    lr: float = 2e-5,
    tau: float = 0.05,
    device: str = "cpu",
    eval_pairs: list[dict] | None = None,
    log_every: int = 25,
    save_every: int = 100,
):
    print(f"Loading student: {student_name}")
    student = load_model(student_name, device=device)

    # Prepare all targets upfront
    all_targets = []
    for sig in signatures:
        t = prepare_targets(sig, tau=tau)
        all_targets.append(t)

    # Baseline eval before training
    if eval_pairs:
        base_metrics = eval_retrieval(student, eval_pairs)
        print(f"Baseline: Hit@5={base_metrics['hit_at_k']:.4f}  MRR={base_metrics['mrr']:.4f}")

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        idx = (step - 1) % len(signatures)
        sig = signatures[idx]
        targets = all_targets[idx]

        if not targets:
            continue

        optimizer.zero_grad()
        loss = train_step(student, sig, targets, tau=tau)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if step % log_every == 0:
            avg = running_loss / log_every
            elapsed = time.time() - t0
            entry = {"step": step, "loss": avg, "elapsed_s": round(elapsed, 1)}

            if eval_pairs and step % (log_every * 4) == 0:
                with torch.no_grad():
                    m = eval_retrieval(student, eval_pairs)
                entry["hit_at_5"] = m["hit_at_k"]
                entry["mrr"] = m["mrr"]

            print(f"  step {step:>5d}  loss={avg:.4f}" +
                  (f"  hit@5={entry.get('hit_at_5', '?')}  mrr={entry.get('mrr', '?')}" if "mrr" in entry else ""))
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

        if step % save_every == 0:
            ckpt_dir = os.path.join(out_dir, f"checkpoint-{step}")
            student.save(ckpt_dir)

    # Final save
    final_dir = os.path.join(out_dir, "final")
    student.save(final_dir)
    print(f"Student saved to {final_dir}")

    # Final eval — the retained gain test
    if eval_pairs:
        with torch.no_grad():
            final_metrics = eval_retrieval(student, eval_pairs)
        print(f"\nRetained gain report:")
        print(f"  Baseline Hit@5: {base_metrics['hit_at_k']:.4f}")
        print(f"  Final    Hit@5: {final_metrics['hit_at_k']:.4f}")
        print(f"  Gain:           {final_metrics['hit_at_k'] - base_metrics['hit_at_k']:+.4f}")
        print(f"  Baseline MRR:   {base_metrics['mrr']:.4f}")
        print(f"  Final    MRR:   {final_metrics['mrr']:.4f}")
        print(f"  Gain:           {final_metrics['mrr'] - base_metrics['mrr']:+.4f}")

        results = {
            "student": student_name,
            "steps": steps,
            "lr": lr,
            "tau": tau,
            "n_signatures": len(signatures),
            "baseline": base_metrics,
            "final": final_metrics,
            "owned_gain_hit5": final_metrics["hit_at_k"] - base_metrics["hit_at_k"],
            "owned_gain_mrr": final_metrics["mrr"] - base_metrics["mrr"],
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    log_f.close()
    return student


def main():
    parser = argparse.ArgumentParser(description="Train Eklavya embedding student")
    parser.add_argument("--signatures", type=str, default=None,
                        help="Path to pre-extracted signatures JSONL")
    parser.add_argument("--teachers", nargs="+", default=None,
                        help="Teacher model names (used if no signatures file)")
    parser.add_argument("--student", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out_dir", type=str, default="outputs/eklavya_v0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--toy", action="store_true", help="Use toy data for quick test")
    parser.add_argument("--n_toy", type=int, default=50)
    args = parser.parse_args()

    # Load or create signatures
    if args.signatures and os.path.exists(args.signatures):
        print(f"Loading signatures from {args.signatures}")
        signatures = load_signatures(args.signatures)
    else:
        teachers = args.teachers or [
            "sentence-transformers/all-MiniLM-L12-v2",
            "BAAI/bge-small-en-v1.5",
        ]
        print(f"Extracting signatures from {len(teachers)} teachers on toy data")
        pairs = generate_toy_pairs(n=args.n_toy)
        signatures = extract_signatures(teachers, pairs, device=args.device)

    # Eval pairs = same data without teacher sigs
    eval_pairs = [
        {"query": s.query, "documents": s.documents, "gold_idx": s.gold_idx}
        for s in signatures
    ]

    student = train(
        student_name=args.student,
        signatures=signatures,
        out_dir=args.out_dir,
        steps=args.steps,
        lr=args.lr,
        tau=args.tau,
        device=args.device,
        eval_pairs=eval_pairs,
    )


if __name__ == "__main__":
    main()
