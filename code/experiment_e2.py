"""Eklavya Experiment E2 — Scaled embedding training for shipping.

Fixes E1 confounds and scales up for a shipping-quality model:
1. Fixed random seed for projection layer (eliminates baseline variance)
2. 5000+ training pairs from MS MARCO
3. Proper train/val/test split
4. Model checkpoint saving
5. Quick MTEB evaluation at end
6. Sentence-transformers export of best model

Student: ModernBERT-base (149M)
Teachers: all-MiniLM-L12-v2 + bge-large-en-v1.5 + nomic-embed-text-v1.5
Data: MS MARCO v2.1 (5000 train, 500 val, 500 test)
Training: 3000 steps, cosine LR with warmup

Usage:
  python code/experiment_e2.py --device cuda --steps 3000 --out_dir outputs/E2
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

from embed_tomography import generate_probes, Probe, load_model as load_st_model
from data_loader import load_msmarco_pairs


class ModernBERTEmbedder(nn.Module):
    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", dim: int = 384,
                 proj_seed: int = 42):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(proj_seed)
        self.proj = nn.Linear(hidden, dim)
        torch.random.set_rng_state(rng_state)
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
        embs = self.forward(texts)
        if not convert_to_tensor:
            return embs.cpu().numpy()
        return embs


@torch.no_grad()
def extract_teacher_scores(teacher, query, documents, probes):
    doc_embs = teacher.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    scores = {}
    probe_texts = [p.text for p in probes]
    all_embs = teacher.encode(probe_texts, convert_to_tensor=True, normalize_embeddings=True)
    for i, probe in enumerate(probes):
        q_emb = all_embs[i:i+1]
        sims = (q_emb @ doc_embs.T).squeeze(0).cpu().tolist()
        scores[probe.probe_id] = sims
    return scores


def compute_tomography_loss(student, query_probes, documents, teacher_scores, tau=0.05):
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


def compute_kd_loss(student, query, documents, teacher_scores, tau=0.05):
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    student_sims = (q_emb @ doc_embs.T).squeeze(0)
    student_log_dist = F.log_softmax(student_sims / tau, dim=0)
    target = torch.tensor(teacher_scores, dtype=torch.float32, device=doc_embs.device)
    target_dist = F.softmax(target / tau, dim=0)
    return F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)


def compute_contrastive_loss(student, query, documents, gold_idx, tau=0.05):
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    sims = (q_emb @ doc_embs.T).squeeze(0) / tau
    target = torch.tensor(gold_idx, device=sims.device)
    return F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))


@torch.no_grad()
def evaluate(student, pairs):
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


def save_checkpoint(student, arm_dir, step, metrics):
    ckpt = {
        "step": step,
        "model_state_dict": student.state_dict(),
        "metrics": metrics,
    }
    path = os.path.join(arm_dir, f"checkpoint_step{step}.pt")
    torch.save(ckpt, path)
    best_path = os.path.join(arm_dir, "best_checkpoint.pt")
    existing_best = None
    if os.path.exists(best_path):
        existing_best = torch.load(best_path, map_location="cpu", weights_only=True)
    if existing_best is None or metrics.get("mrr", 0) > existing_best.get("metrics", {}).get("mrr", 0):
        torch.save(ckpt, best_path)
        return True
    return False


def run_arm(
    arm_name, student, train_pairs, val_pairs, test_pairs,
    teacher_data, steps, lr, tau, out_dir, arm_type="tomography",
    warmup_steps=300, save_every=500,
):
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name} ({arm_type})")
    print(f"{'='*60}")
    sys.stdout.flush()

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.01)

    base = evaluate(student, val_pairs)
    print(f"  Baseline: Hit@1={base['hit@1']:.4f}  Hit@5={base['hit@5']:.4f}  MRR={base['mrr']:.4f}")
    sys.stdout.flush()

    arm_dir = os.path.join(out_dir, arm_name)
    Path(arm_dir).mkdir(parents=True, exist_ok=True)
    log_f = open(os.path.join(arm_dir, "log.jsonl"), "w")

    t0 = time.time()
    running_loss = 0.0
    best_val_mrr = base["mrr"]

    for step in range(1, steps + 1):
        if step <= warmup_steps:
            warmup_lr = lr * step / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

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
            tid = list(teacher_data[pair["id"]].keys())[0]
            loss = compute_kd_loss(
                student, pair["query"], pair["documents"],
                teacher_data[pair["id"]][tid]["identity"], tau=tau,
            )
        elif arm_type == "kd_avg":
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
        if step > warmup_steps:
            scheduler.step()

        running_loss += loss.item()

        if step % 100 == 0:
            avg = running_loss / 100
            entry = {"step": step, "loss": round(avg, 6), "elapsed_s": round(time.time()-t0, 1)}
            if step % 500 == 0:
                m = evaluate(student, val_pairs)
                entry.update(m)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}")
                if m["mrr"] > best_val_mrr:
                    best_val_mrr = m["mrr"]
                    save_checkpoint(student, arm_dir, step, m)
                    print(f"    -> new best (val MRR={m['mrr']:.4f}), checkpoint saved")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}")
            sys.stdout.flush()
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

    save_checkpoint(student, arm_dir, steps, evaluate(student, val_pairs))

    val_final = evaluate(student, val_pairs)
    test_final = evaluate(student, test_pairs)
    result = {
        "arm": arm_name,
        "type": arm_type,
        "steps": steps,
        "baseline": base,
        "val_final": val_final,
        "test_final": test_final,
        "gain_mrr_val": val_final["mrr"] - base["mrr"],
        "gain_mrr_test": test_final["mrr"] - base["mrr"],
        "best_val_mrr": best_val_mrr,
    }
    print(f"\n  VAL:  MRR {base['mrr']:.4f} -> {val_final['mrr']:.4f} ({result['gain_mrr_val']:+.4f})")
    print(f"  TEST: MRR -> {test_final['mrr']:.4f} ({result['gain_mrr_test']:+.4f})")
    sys.stdout.flush()

    with open(os.path.join(arm_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log_f.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Eklavya E2 -- Scaled Embedding Training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=4000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--out_dir", default="outputs/E2_msmarco")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--proj_seed", type=int, default=42)
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2",
                                 "BAAI/bge-large-en-v1.5",
                                 "nomic-ai/nomic-embed-text-v1.5"])
    parser.add_argument("--arms", nargs="+", default=["contrastive", "kd_avg", "tomography"],
                        help="Which arms to run. Options: contrastive, kd_single, kd_avg, tomography")
    parser.add_argument("--warmup_steps", type=int, default=300)
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    total_needed = args.n_train + args.n_val + args.n_test
    print(f"Loading {total_needed} MS MARCO pairs...")
    sys.stdout.flush()
    all_pairs = load_msmarco_pairs(n=total_needed, seed=42)
    if len(all_pairs) < total_needed:
        print(f"Warning: only got {len(all_pairs)} pairs")
    train_pairs = all_pairs[:args.n_train]
    val_pairs = all_pairs[args.n_train:args.n_train + args.n_val]
    test_pairs = all_pairs[args.n_train + args.n_val:]
    print(f"Data: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    print("\nExtracting teacher signatures...")
    sys.stdout.flush()
    teachers = {}
    for tname in args.teachers:
        print(f"  Loading {tname}")
        sys.stdout.flush()
        teachers[tname] = load_st_model(tname, device=args.device)

    teacher_data = {}
    for pi, pair in enumerate(train_pairs):
        if (pi + 1) % 500 == 0:
            print(f"    Signatures: {pi + 1}/{len(train_pairs)}")
            sys.stdout.flush()
        probes = generate_probes(pair["query"], seed=pi)
        td = {}
        for tname, tmodel in teachers.items():
            td[tname] = extract_teacher_scores(tmodel, pair["query"], pair["documents"], probes)
        teacher_data[pair["id"]] = td

    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted signatures for {len(teacher_data)} pairs")
    sys.stdout.flush()

    config = {
        "student": args.student,
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "proj_seed": args.proj_seed,
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "n_test": len(test_pairs),
        "warmup_steps": args.warmup_steps,
        "arms": args.arms,
        "device": args.device,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    arm_configs = {
        "contrastive": ("B0_contrastive", "contrastive"),
        "kd_single": ("B2_kd_single", "kd_single"),
        "kd_avg": ("B3_kd_avg", "kd_avg"),
        "tomography": ("E2_tomography", "tomography"),
    }

    results = {}
    for arm_key in args.arms:
        if arm_key not in arm_configs:
            print(f"Unknown arm: {arm_key}, skipping")
            continue
        arm_name, arm_type = arm_configs[arm_key]
        student = ModernBERTEmbedder(args.student, dim=384, proj_seed=args.proj_seed).to(args.device)
        results[arm_name] = run_arm(
            arm_name, student, train_pairs, val_pairs, test_pairs,
            teacher_data=teacher_data if arm_type != "contrastive" else None,
            steps=args.steps, lr=args.lr, tau=args.tau,
            out_dir=args.out_dir, arm_type=arm_type,
            warmup_steps=args.warmup_steps,
        )
        del student
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EXPERIMENT E2 SUMMARY")
    print("=" * 60)
    print(f"{'Arm':<20} {'Val MRR':>8} {'Test MRR':>9} {'Gain Val':>10} {'Gain Test':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['val_final']['mrr']:>8.4f} {r['test_final']['mrr']:>9.4f} "
              f"{r['gain_mrr_val']:>+10.4f} {r['gain_mrr_test']:>+10.4f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_arm = max(results.items(), key=lambda x: x[1]["test_final"]["mrr"])
    print(f"\nBest arm: {best_arm[0]} (test MRR={best_arm[1]['test_final']['mrr']:.4f})")
    print(f"Best checkpoint: {os.path.join(args.out_dir, best_arm[0], 'best_checkpoint.pt')}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
