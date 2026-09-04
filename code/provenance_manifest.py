"""Provenance Manifest — score eval queries with both teachers.

Reconstructs E1.5 eval candidate pools (deterministic given seed/proj_seed),
then scores each pool with MiniLM-L12 and BGE-large to produce per-query
teacher rankings. Combined with B0/B2/B3 student rankings from result.json,
this creates the actual provenance data needed for donor-private analysis.

The manifest is hashed (SHA-256) to ensure immutability per Codex Round 2 §5.

CPU-only. Does not require GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def score_candidates_with_teacher(teacher, query: str, documents: list[str]) -> dict:
    """Score a candidate pool with a teacher model. Returns per-doc scores and gold ranking."""
    with torch.no_grad():
        q_emb = teacher.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        d_embs = teacher.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ d_embs.T).squeeze(0).cpu().tolist()
    ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    return {"scores": sims, "ranking": ranked}


def main():
    parser = argparse.ArgumentParser(description="Build provenance manifest")
    parser.add_argument("--e15_dir", default="outputs/E1_5_text")
    parser.add_argument("--out_file", default="outputs/provenance_manifest.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 271])
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_docs", type=int, default=32)
    parser.add_argument("--proj_seed", type=int, default=9999)
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"])
    args = parser.parse_args()

    from data_loader import load_msmarco_pairs, mine_hard_negatives
    from embed_tomography import load_model as load_st_model
    from experiment_e1 import ModernBERTEmbedder

    arms_to_load = ["B0_contrastive", "B2_kd_single", "B3_kd_avg_cal"]
    manifest = {"seeds": {}, "teachers": args.teachers, "arms": arms_to_load}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        t0 = time.time()

        import random as stdlib_random
        stdlib_random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Loading raw MSMARCO pairs...")
        raw_pairs = load_msmarco_pairs(
            n=args.n_train + args.n_eval, n_docs=10, seed=seed,
        )
        eval_raw = raw_pairs[args.n_train : args.n_train + args.n_eval]
        print(f"  {len(eval_raw)} eval pairs loaded")

        print("Loading raw student for hard-negative mining (CPU)...")
        raw_student = ModernBERTEmbedder(
            args.student, dim=384, proj_seed=args.proj_seed,
        ).to(args.device)
        raw_student.eval()

        print("Mining hard negatives (deterministic reconstruction)...")
        eval_pairs = mine_hard_negatives(
            eval_raw, raw_student, n_docs=args.n_docs,
        )
        del raw_student
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Verifying reconstruction against saved baseline...")
        seed_dir = os.path.join(args.e15_dir, f"seed_{seed}")
        b0_result = json.load(open(os.path.join(seed_dir, "B0_contrastive", "result.json")))
        saved_baseline_mrr = b0_result["baseline"]["mrr"]
        saved_baseline_n = b0_result["baseline"]["n"]
        saved_baseline_hit1 = b0_result["baseline"]["hit@1"]

        recon_student = ModernBERTEmbedder(
            args.student, dim=384, proj_seed=args.proj_seed,
        ).to(args.device)
        recon_student.eval()

        with torch.no_grad():
            recon_mrr = 0.0
            recon_hit1 = 0
            for pair in eval_pairs:
                q_emb = recon_student.forward([pair["query"]])
                d_embs = recon_student.forward(pair["documents"])
                sims = (q_emb @ d_embs.T).squeeze(0)
                ranked = sims.argsort(descending=True).tolist()
                gold = pair["gold_idx"]
                rank = ranked.index(gold) + 1
                recon_mrr += 1.0 / rank
                if ranked[0] == gold:
                    recon_hit1 += 1
            recon_mrr /= len(eval_pairs)
            recon_hit1 /= len(eval_pairs)

        del recon_student

        mrr_match = abs(recon_mrr - saved_baseline_mrr) < 0.001
        hit1_match = abs(recon_hit1 - saved_baseline_hit1) < 0.01
        print(f"  Saved baseline: MRR={saved_baseline_mrr:.6f}, hit@1={saved_baseline_hit1:.3f}")
        print(f"  Recon baseline: MRR={recon_mrr:.6f}, hit@1={recon_hit1:.3f}")
        print(f"  Match: MRR={'YES' if mrr_match else 'NO'}, hit@1={'YES' if hit1_match else 'NO'}")

        if not mrr_match:
            print(f"  WARNING: Baseline MRR mismatch — CPU mining may differ from GPU.")
            print(f"  Proceeding but flagging for GPU re-verification.")

        arm_results = {}
        for arm_name in arms_to_load:
            rpath = os.path.join(seed_dir, arm_name, "result.json")
            if os.path.exists(rpath):
                r = json.load(open(rpath))
                arm_results[arm_name] = {
                    q["id"]: q for q in r["final"]["per_query"]
                }
            else:
                print(f"  WARNING: {rpath} not found")
                arm_results[arm_name] = {}

        print("Scoring eval queries with teachers...")
        teachers = {}
        for tname in args.teachers:
            print(f"  Loading {tname} on {args.device}...")
            teachers[tname] = load_st_model(tname, device=args.device)

        seed_data = {
            "verified": mrr_match and hit1_match,
            "baseline_mrr_saved": saved_baseline_mrr,
            "baseline_mrr_recon": recon_mrr,
            "queries": [],
        }

        for i, pair in enumerate(eval_pairs):
            if (i + 1) % 50 == 0:
                print(f"  Scoring query {i+1}/{len(eval_pairs)}...")

            qdata = {
                "id": pair["id"],
                "query": pair["query"],
                "documents": pair["documents"],
                "n_candidates": len(pair["documents"]),
                "gold_idx": pair["gold_idx"],
                "teacher_scores": {},
                "student_ranks": {},
            }

            for tname, tmodel in teachers.items():
                result = score_candidates_with_teacher(tmodel, pair["query"], pair["documents"])
                gold_position = result["ranking"].index(pair["gold_idx"]) + 1
                qdata["teacher_scores"][tname] = {
                    "gold_rank": gold_position,
                    "rr": 1.0 / gold_position,
                    "scores": result["scores"],
                    "ranking": result["ranking"],
                }

            for arm_name in arms_to_load:
                if pair["id"] in arm_results[arm_name]:
                    ar = arm_results[arm_name][pair["id"]]
                    qdata["student_ranks"][arm_name] = {
                        "gold_rank": ar["gold_rank"],
                        "rr": ar["rr"],
                    }

            seed_data["queries"].append(qdata)

        del teachers
        elapsed = time.time() - t0
        print(f"  Seed {seed} complete in {elapsed:.1f}s")

        manifest["seeds"][str(seed)] = seed_data

    print(f"\n{'='*60}")
    print("COMPUTING PROVENANCE QUADRANTS")
    print(f"{'='*60}")

    for seed_key, seed_data in manifest["seeds"].items():
        for tname in args.teachers:
            short_name = tname.split("/")[-1]
            for arm_name in arms_to_load:
                if arm_name == "B0_contrastive":
                    continue

                q1 = q2 = q3 = q4 = 0
                q1_ids = []
                q2_ids = []
                for qdata in seed_data["queries"]:
                    ts = qdata["teacher_scores"].get(tname, {})
                    sr = qdata["student_ranks"].get(arm_name, {})
                    b0 = qdata["student_ranks"].get("B0_contrastive", {})

                    if not ts or not sr or not b0:
                        continue

                    teacher_correct = ts["gold_rank"] == 1
                    b0_correct = b0["gold_rank"] == 1

                    if teacher_correct and not b0_correct:
                        q1 += 1
                        q1_ids.append(qdata["id"])
                    elif b0_correct and not teacher_correct:
                        q2 += 1
                        q2_ids.append(qdata["id"])
                    elif teacher_correct and b0_correct:
                        q3 += 1
                    else:
                        q4 += 1

                total = q1 + q2 + q3 + q4
                print(f"\n  Seed {seed_key}, {short_name} vs B0:")
                print(f"    Q1 (teacher correct, B0 wrong) = {q1} ({100*q1/total:.1f}%) — DONOR-PRIVATE")
                print(f"    Q2 (B0 correct, teacher wrong)  = {q2} ({100*q2/total:.1f}%) — STUDENT-ONLY")
                print(f"    Q3 (both correct)               = {q3} ({100*q3/total:.1f}%) — SHARED")
                print(f"    Q4 (neither correct)            = {q4} ({100*q4/total:.1f}%) — UNKNOWN")

                if q1_ids:
                    q_lookup = {q["id"]: q for q in seed_data["queries"]}
                    rescued = sum(
                        1 for qid in q1_ids
                        if q_lookup[qid]["student_ranks"].get(arm_name, {}).get("gold_rank", 99) == 1
                    )
                    print(f"    Q1 rescued by {arm_name}: {rescued}/{q1}")

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    sha = hashlib.sha256(manifest_json.encode()).hexdigest()
    manifest["_sha256"] = sha
    manifest["_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"\n{'='*60}")
    print(f"MANIFEST SAVED: {args.out_file}")
    print(f"SHA-256: {sha[:16]}...")
    print(f"{'='*60}")

    summary = {
        "teacher_mrr": {},
        "b0_mrr": {},
    }
    for seed_key, seed_data in manifest["seeds"].items():
        for tname in args.teachers:
            short = tname.split("/")[-1]
            mrrs = [q["teacher_scores"][tname]["rr"] for q in seed_data["queries"]]
            summary["teacher_mrr"].setdefault(short, {})[seed_key] = np.mean(mrrs)
        b0_mrrs = [q["student_ranks"]["B0_contrastive"]["rr"]
                   for q in seed_data["queries"]
                   if "B0_contrastive" in q["student_ranks"]]
        summary["b0_mrr"][seed_key] = np.mean(b0_mrrs)

    print("\nTEACHER vs B0 SUMMARY (MRR on eval pools):")
    for tname_short, seed_mrrs in summary["teacher_mrr"].items():
        mean_mrr = np.mean(list(seed_mrrs.values()))
        print(f"  {tname_short}: mean MRR = {mean_mrr:.4f} (seeds: {seed_mrrs})")
    b0_mean = np.mean(list(summary["b0_mrr"].values()))
    print(f"  B0 (final):  mean MRR = {b0_mean:.4f} (seeds: {summary['b0_mrr']})")


if __name__ == "__main__":
    main()
