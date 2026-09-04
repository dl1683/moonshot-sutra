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
    ("Clustering", "ArxivClusteringS2S"),
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
        "ArxivClusteringP2P", "ArxivClusteringS2S",
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
    """Wrap a ModernBERT checkpoint for MTEB evaluation.

    Implements the MTEB AbsEncoder interface (DataLoader-based encode).
    """

    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.mteb_model_meta = None
        self.model_prompts = None

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, "student_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            model_name = config.get("student", config.get("base_model", "answerdotai/ModernBERT-base"))
            dim = config.get("dim", 384)
        else:
            model_name = "answerdotai/ModernBERT-base"
            dim = 384

        encoder_dir = os.path.join(model_dir, "encoder")
        if os.path.isdir(encoder_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
            self.encoder = AutoModel.from_pretrained(encoder_dir).to(device).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name).to(device).eval()
        hidden = self.encoder.config.hidden_size

        self.proj = nn.Linear(hidden, dim).to(device)

        proj_path = os.path.join(model_dir, "proj.pt")
        if os.path.exists(proj_path):
            proj_state = torch.load(proj_path, map_location=device, weights_only=True)
            self.proj.load_state_dict(proj_state)
        else:
            for name in ("model.pt", "student.pt"):
                ckpt_path = os.path.join(model_dir, name)
                if os.path.exists(ckpt_path):
                    state = torch.load(ckpt_path, map_location=device, weights_only=True)
                    if "proj.weight" in state:
                        self.proj.load_state_dict({"weight": state["proj.weight"], "bias": state["proj.bias"]})
                    break

    @torch.no_grad()
    def _encode_sentences(self, sentences: list[str], batch_size: int = 32,
                          normalize: bool = True):
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
            if normalize:
                projected = F.normalize(projected, p=2, dim=-1)
            all_embs.append(projected.cpu())
        return torch.cat(all_embs, dim=0)

    def encode(self, inputs, *, task_metadata=None, hf_split=None,
               hf_subset=None, prompt_type=None, **kwargs):
        sentences = [text for batch in inputs for text in batch["text"]]
        return self._encode_sentences(sentences).numpy()


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

    is_raw_checkpoint = os.path.isdir(args.model) and any(
        os.path.exists(os.path.join(args.model, f))
        for f in ("model.pt", "student.pt", "proj.pt")
    )

    from sentence_transformers import SentenceTransformer

    if is_raw_checkpoint:
        st_dir = os.path.join(args.out_dir, "_st_export")
        print(f"Exporting raw checkpoint to sentence-transformers format: {st_dir}")
        from export_model import build_sentence_transformer, load_checkpoint_weights
        proj_weights, encoder_path = load_checkpoint_weights(args.model)
        config_path = os.path.join(args.model, "config.json")
        base_model = "answerdotai/ModernBERT-base"
        dim = 384
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            base_model = cfg.get("student", base_model)
            dim = cfg.get("dim", dim)
        st_model = build_sentence_transformer(base_model, dim, proj_weights, encoder_path)
        st_model.save(st_dir)
        print(f"Loading exported model for MTEB...")
        model = SentenceTransformer(st_dir, device=args.device)
    else:
        try:
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
