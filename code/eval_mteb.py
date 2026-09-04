"""MTEB evaluation for Eklavya embedding models.

Wraps a trained ModernBERT student (or any sentence-transformers model) and
runs a subset of MTEB tasks to get competitive benchmarks before shipping.

Quick eval (~10 min): 8 representative tasks across categories
Full eval (~2-4 hours): all 56 English MTEB tasks

Usage:
  python code/eval_mteb.py --model outputs/E2/best_model --mode quick
  python code/eval_mteb.py --model sentence-transformers/all-MiniLM-L6-v2 --mode quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

QUICK_TASKS = [
    ("STS", "STSBenchmark"),
    ("STS", "SICK-R"),
    ("Classification", "AmazonCounterfactualClassification"),
    ("Classification", "TweetSentimentExtractionClassification"),
    ("Clustering", "ArXivClusteringS2S"),
    ("PairClassification", "TwitterURLCorpus"),
    ("Reranking", "AskUbuntuDupQuestions"),
    ("Retrieval", "SciFact"),
]

CATEGORY_TASKS = {
    "STS": ["STSBenchmark", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16"],
    "Classification": [
        "AmazonCounterfactualClassification", "AmazonPolarityClassification",
        "AmazonReviewsClassification", "Banking77Classification",
        "EmotionClassification", "ImdbClassification",
        "MassiveIntentClassification", "MassiveScenarioClassification",
        "MTOPDomainClassification", "MTOPIntentClassification",
        "ToxicConversationsClassification", "TweetSentimentExtractionClassification",
    ],
    "Clustering": [
        "ArXivClusteringP2P", "ArXivClusteringS2S",
        "RedditClustering", "RedditClusteringP2P",
        "StackExchangeClustering", "StackExchangeClusteringP2P",
        "TwentyNewsgroupsClustering",
    ],
    "PairClassification": [
        "SprintDuplicateQuestions", "TwitterURLCorpus", "TwitterSemEval2015",
    ],
    "Reranking": [
        "AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "Retrieval": [
        "ArguAna", "ClimateFEVER", "CQADupstackTexRetrieval",
        "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO",
        "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact",
        "Touche2020", "TRECCOVID",
    ],
}


class ModernBERTWrapper:
    """Wrap a ModernBERT checkpoint for MTEB evaluation."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.device = device

        config_path = os.path.join(model_dir, "student_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_name = config.get("base_model", "answerdotai/ModernBERT-base")
            dim = config.get("dim", 384)
        else:
            model_name = "answerdotai/ModernBERT-base"
            dim = 384

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device).eval()
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Linear(hidden, dim).to(device)
        ckpt_path = os.path.join(model_dir, "student.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            if "proj.weight" in state:
                self.proj.load_state_dict({"weight": state["proj.weight"], "bias": state["proj.bias"]})
            elif "model_state_dict" in state:
                self.proj.load_state_dict({
                    k.replace("proj.", ""): v
                    for k, v in state["model_state_dict"].items()
                    if k.startswith("proj.")
                })

    @torch.no_grad()
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        import numpy as np

        all_embs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.encoder(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            projected = self.proj(pooled)
            if normalize_embeddings:
                projected = F.normalize(projected, p=2, dim=-1)
            all_embs.append(projected.cpu())

        result = torch.cat(all_embs, dim=0)
        if convert_to_tensor:
            return result
        return result.numpy()


def run_mteb_eval(model, task_name: str, output_dir: str) -> dict:
    """Run a single MTEB task and return results."""
    try:
        import mteb
    except ImportError:
        print("mteb not installed. Run: pip install mteb")
        return {}

    task = mteb.get_task(task_name)
    evaluation = mteb.MTEB(tasks=[task])
    results = evaluation.run(model, output_folder=output_dir, overwrite_results=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="MTEB evaluation for Eklavya models")
    parser.add_argument("--model", required=True, help="Model path or HF model name")
    parser.add_argument("--mode", choices=["quick", "full", "category"], default="quick")
    parser.add_argument("--category", type=str, help="Task category for category mode")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default="outputs/mteb_eval")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(args.model) and os.path.exists(os.path.join(args.model, "student.pt")):
        print(f"Loading Eklavya model from {args.model}")
        model = ModernBERTWrapper(args.model, device=args.device)
    else:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading sentence-transformers model: {args.model}")
            model = SentenceTransformer(args.model, device=args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    if args.mode == "quick":
        tasks = QUICK_TASKS
    elif args.mode == "category":
        if args.category not in CATEGORY_TASKS:
            print(f"Unknown category: {args.category}. Choose from: {list(CATEGORY_TASKS.keys())}")
            return
        tasks = [(args.category, t) for t in CATEGORY_TASKS[args.category]]
    else:
        tasks = [(cat, t) for cat, task_list in CATEGORY_TASKS.items() for t in task_list]

    print(f"\nRunning {len(tasks)} MTEB tasks ({args.mode} mode)...")
    sys.stdout.flush()

    results_summary = {}
    for cat, task_name in tasks:
        print(f"\n  [{cat}] {task_name}...", end=" ")
        sys.stdout.flush()
        try:
            res = run_mteb_eval(model, task_name, args.out_dir)
            if res:
                score = extract_main_score(res, task_name)
                results_summary[task_name] = {"category": cat, "score": score}
                print(f"score={score:.4f}")
            else:
                print("no result")
        except Exception as e:
            print(f"error: {e}")
            results_summary[task_name] = {"category": cat, "score": None, "error": str(e)}
        sys.stdout.flush()

    print("\n" + "=" * 60)
    print("MTEB RESULTS SUMMARY")
    print("=" * 60)

    by_category = {}
    for task_name, info in results_summary.items():
        cat = info["category"]
        by_category.setdefault(cat, [])
        if info.get("score") is not None:
            by_category[cat].append(info["score"])

    print(f"\n{'Category':<25} {'Avg Score':>10} {'N Tasks':>8}")
    print("-" * 45)
    all_scores = []
    for cat in sorted(by_category.keys()):
        scores = by_category[cat]
        if scores:
            avg = sum(scores) / len(scores)
            all_scores.extend(scores)
            print(f"{cat:<25} {avg:>10.4f} {len(scores):>8}")

    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"\n{'OVERALL':<25} {overall:>10.4f} {len(all_scores):>8}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"results": results_summary, "by_category": {
            cat: {"avg": sum(s)/len(s), "n": len(s)} for cat, s in by_category.items() if s
        }}, f, indent=2)

    print(f"\nResults saved to {args.out_dir}")
    sys.stdout.flush()


def extract_main_score(results, task_name: str) -> float:
    """Extract the primary score from MTEB results."""
    if isinstance(results, list) and results:
        result = results[0]
        if hasattr(result, "scores"):
            scores = result.scores
            if "test" in scores:
                test_scores = scores["test"]
                if isinstance(test_scores, list) and test_scores:
                    first = test_scores[0]
                    for key in ["main_score", "cos_sim.spearman", "accuracy", "ndcg_at_10", "map", "ap"]:
                        if key in first:
                            return first[key]
    return 0.0


if __name__ == "__main__":
    main()
