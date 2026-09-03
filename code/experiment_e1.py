"""Eklavya Experiment E1 — Real embedding tomography vs standard KD.

Student: ModernBERT-base (149M, no embedding training)
Teachers: all-MiniLM-L12-v2 + bge-large-en-v1.5 (heterogeneous)
Data: Hard-negative retrieval pairs
Eval: Held-out retrieval accuracy + teacher-probe response matching

Arms:
  E1: Full tomography (multi-probe, multi-teacher KL on ranking distributions)
  B0: Contrastive-only baseline (InfoNCE, no teacher)
  B2: Standard single-teacher KD (identity probe only, best teacher)
  B3: Multi-teacher average KD (average teacher scores, identity probe only)

The decisive question: does multi-probe tomography transfer structure that
standard KD and averaging cannot?

Usage:
  python code/experiment_e1.py --device cuda --steps 1000 --out_dir outputs/E1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embed_tomography import (
    generate_probes,
    Probe,
    load_model as load_st_model,
)
from data_loader import load_hard_toy, load_msmarco_pairs


class ModernBERTEmbedder(nn.Module):
    """Wraps ModernBERT-base as a sentence embedder with mean pooling + projection."""

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", dim: int = 384):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768
        self.proj = nn.Linear(hidden, dim)
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(next(self.encoder.parameters()).device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        token_embs = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True, **kwargs):
        """sentence-transformers compatible interface."""
        embs = self.forward(texts)
        if not convert_to_tensor:
            return embs.cpu().numpy()
        return embs


@torch.no_grad()
def extract_teacher_scores(teacher, query: str, documents: list[str], probes: list[Probe]) -> dict[str, list[float]]:
    """Get teacher similarity scores for each probe."""
    doc_embs = teacher.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    scores = {}
    for probe in probes:
        q_emb = teacher.encode([probe.text], convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ doc_embs.T).squeeze(0).cpu().tolist()
        scores[probe.probe_id] = sims
    return scores


def compute_tomography_loss(
    student: ModernBERTEmbedder,
    query_probes: list[Probe],
    documents: list[str],
    teacher_scores: dict[str, dict[str, list[float]]],
    tau: float = 0.05,
) -> torch.Tensor:
    """KL divergence between student and teacher ranking distributions across all probes and teachers."""
    doc_embs = student.forward(documents)
    loss = torch.tensor(0.0, device=doc_embs.device)
    n = 0

    for probe in query_probes:
        q_emb = student.forward([probe.text])
        student_sims = (q_emb @ doc_embs.T).squeeze(0)
        student_log_dist = F.log_softmax(student_sims / tau, dim=0)

        for tname, tsig in teacher_scores.items():
            if probe.probe_id in tsig:
                target_sims = torch.tensor(tsig[probe.probe_id], dtype=torch.float32, device=doc_embs.device)
                target_dist = F.softmax(target_sims / tau, dim=0)
                kl = F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)
                loss = loss + kl
                n += 1

    return loss / max(n, 1)


def compute_kd_loss(
    student: ModernBERTEmbedder,
    query: str,
    documents: list[str],
    teacher_scores: list[float],
    tau: float = 0.05,
) -> torch.Tensor:
    """Standard KD: match one teacher's ranking distribution on identity query only."""
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    student_sims = (q_emb @ doc_embs.T).squeeze(0)
    student_log_dist = F.log_softmax(student_sims / tau, dim=0)
    target = torch.tensor(teacher_scores, dtype=torch.float32, device=doc_embs.device)
    target_dist = F.softmax(target / tau, dim=0)
    return F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)


def compute_contrastive_loss(
    student: ModernBERTEmbedder,
    query: str,
    documents: list[str],
    gold_idx: int,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss — no teacher, just (query, gold_doc) pairs."""
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    sims = (q_emb @ doc_embs.T).squeeze(0) / tau
    target = torch.tensor(gold_idx, device=sims.device)
    return F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))


@torch.no_grad()
def evaluate(student: ModernBERTEmbedder, pairs: list[dict]) -> dict:
    hits1 = hits5 = 0
    mrr_sum = 0.0
    for pair in pairs:
        q_emb = student.forward([pair["query"]])
        d_embs = student.forward(pair["documents"])
        sims = (q_emb @ d_embs.T).squeeze(0)
        ranked = sims.argsort(descending=True).tolist()
        gold = pair["gold_idx"]
        if ranked[0] == gold:
            hits1 += 1
        if gold in ranked[:5]:
            hits5 += 1
        rank = ranked.index(gold) + 1
        mrr_sum += 1.0 / rank
    n = len(pairs)
    return {"hit@1": hits1/n, "hit@5": hits5/n, "mrr": mrr_sum/n, "n": n}


def run_arm(
    arm_name: str,
    student: ModernBERTEmbedder,
    train_pairs: list[dict],
    eval_pairs: list[dict],
    teacher_data: dict | None,
    steps: int,
    lr: float,
    tau: float,
    out_dir: str,
    arm_type: str = "tomography",
):
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name} ({arm_type})")
    print(f"{'='*60}")

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    # Baseline
    base = evaluate(student, eval_pairs)
    print(f"  Baseline: Hit@1={base['hit@1']:.4f}  Hit@5={base['hit@5']:.4f}  MRR={base['mrr']:.4f}")

    arm_dir = os.path.join(out_dir, arm_name)
    Path(arm_dir).mkdir(parents=True, exist_ok=True)
    log_f = open(os.path.join(arm_dir, "log.jsonl"), "w")

    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        idx = (step - 1) % len(train_pairs)
        pair = train_pairs[idx]

        optimizer.zero_grad()

        if arm_type == "tomography":
            probes = generate_probes(pair["query"], seed=idx)
            loss = compute_tomography_loss(
                student, probes, pair["documents"],
                teacher_data[pair["id"]], tau=tau,
            )
        elif arm_type == "kd_single":
            # Use best teacher's identity scores
            tid = list(teacher_data[pair["id"]].keys())[0]
            loss = compute_kd_loss(
                student, pair["query"], pair["documents"],
                teacher_data[pair["id"]][tid]["identity"], tau=tau,
            )
        elif arm_type == "kd_avg":
            # Average teacher identity scores
            scores_lists = [t["identity"] for t in teacher_data[pair["id"]].values()]
            avg_scores = [sum(s)/len(s) for s in zip(*scores_lists)]
            loss = compute_kd_loss(
                student, pair["query"], pair["documents"],
                avg_scores, tau=tau,
            )
        elif arm_type == "contrastive":
            loss = compute_contrastive_loss(
                student, pair["query"], pair["documents"],
                pair["gold_idx"], tau=tau,
            )
        else:
            raise ValueError(f"Unknown arm type: {arm_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 50 == 0:
            avg = running_loss / 50
            entry = {"step": step, "loss": round(avg, 6), "elapsed_s": round(time.time()-t0, 1)}
            if step % 200 == 0:
                m = evaluate(student, eval_pairs)
                entry.update(m)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}")
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

    # Final eval
    final = evaluate(student, eval_pairs)
    result = {
        "arm": arm_name,
        "type": arm_type,
        "steps": steps,
        "baseline": base,
        "final": final,
        "gain_hit1": final["hit@1"] - base["hit@1"],
        "gain_mrr": final["mrr"] - base["mrr"],
    }
    print(f"\n  RESULT: Hit@1 {base['hit@1']:.4f} -> {final['hit@1']:.4f} ({result['gain_hit1']:+.4f})")
    print(f"          MRR   {base['mrr']:.4f} -> {final['mrr']:.4f} ({result['gain_mrr']:+.4f})")

    with open(os.path.join(arm_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log_f.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Eklavya E1 — Embedding Tomography Experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=150)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--out_dir", default="outputs/E1")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load data — MS MARCO for real retrieval difficulty
    print("Loading MS MARCO data...")
    all_pairs = load_msmarco_pairs(n=args.n_train + args.n_eval, seed=42)
    if len(all_pairs) < args.n_train + args.n_eval:
        print(f"Warning: only got {len(all_pairs)} pairs, falling back to hard toy data")
        all_pairs = load_hard_toy(n=args.n_train + args.n_eval, seed=42)
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]
    print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval pairs")

    # Extract teacher signatures for ALL probes on train data
    print("\nExtracting teacher signatures...")
    teachers = {}
    for tname in args.teachers:
        print(f"  Loading {tname}")
        teachers[tname] = load_st_model(tname, device=args.device)

    teacher_data = {}
    for pair in train_pairs:
        probes = generate_probes(pair["query"], seed=train_pairs.index(pair))
        td = {}
        for tname, tmodel in teachers.items():
            td[tname] = extract_teacher_scores(tmodel, pair["query"], pair["documents"], probes)
        teacher_data[pair["id"]] = td

    # Free teacher memory
    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted signatures for {len(teacher_data)} pairs")

    # Save experiment config
    config = {
        "student": args.student,
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "device": args.device,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    results = {}

    # ARM B0: Contrastive only (no teacher)
    student_b0 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B0_contrastive"] = run_arm(
        "B0_contrastive", student_b0, train_pairs, eval_pairs,
        teacher_data=None, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="contrastive",
    )
    del student_b0
    torch.cuda.empty_cache()

    # ARM B2: Single-teacher KD (best teacher, identity probe only)
    student_b2 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B2_kd_single"] = run_arm(
        "B2_kd_single", student_b2, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_single",
    )
    del student_b2
    torch.cuda.empty_cache()

    # ARM B3: Multi-teacher average KD (average scores, identity only)
    student_b3 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B3_kd_avg"] = run_arm(
        "B3_kd_avg", student_b3, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_avg",
    )
    del student_b3
    torch.cuda.empty_cache()

    # ARM E1: Full tomography (multi-probe, multi-teacher)
    student_e1 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["E1_tomography"] = run_arm(
        "E1_tomography", student_e1, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="tomography",
    )
    del student_e1
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT E1 SUMMARY")
    print("=" * 60)
    print(f"{'Arm':<20} {'Hit@1':>8} {'MRR':>8} {'Gain Hit@1':>12} {'Gain MRR':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['final']['hit@1']:>8.4f} {r['final']['mrr']:>8.4f} "
              f"{r['gain_hit1']:>+12.4f} {r['gain_mrr']:>+10.4f}")

    # Save summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Verdict
    tomo = results["E1_tomography"]
    best_baseline = max(results["B0_contrastive"]["final"]["mrr"],
                        results["B2_kd_single"]["final"]["mrr"],
                        results["B3_kd_avg"]["final"]["mrr"])
    margin = tomo["final"]["mrr"] - best_baseline
    print(f"\nTomography vs best baseline MRR margin: {margin:+.4f}")
    if margin > 0.01:
        print("VERDICT: Tomography shows signal. Proceed to E2.")
    elif margin > -0.01:
        print("VERDICT: Inconclusive. Need more data/training/harder eval.")
    else:
        print("VERDICT: Tomography absorbed. Investigate why.")


if __name__ == "__main__":
    main()
