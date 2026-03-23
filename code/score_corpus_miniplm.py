"""MiniPLM-style offline corpus scoring for P1b data curation.

Scores training corpus chunks with a teacher (Qwen3-0.6B-Base) and reference
(GPT-2 small) model. Computes importance weights per shard based on the
"difference sampling" principle: upweight chunks where the student/reference
struggles but the teacher finds easy (= learnable knowledge gaps).

Output: results/p1b_shard_weights.json — importance weights per shard.

Usage:
    # Full scoring (all shards, ~2-3 hours on CPU)
    python code/score_corpus_miniplm.py

    # Quick test (5 shards)
    python code/score_corpus_miniplm.py --max-shards 5

    # Use Sutra as reference instead of GPT-2 (slower but more accurate)
    python code/score_corpus_miniplm.py --reference sutra
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "shards"
OUTPUT_FILE = REPO / "results" / "p1b_shard_weights.json"

# Scoring config
CHUNKS_PER_SHARD = 20       # Number of random 512-token chunks per shard
CHUNK_LEN = 512             # Tokens per chunk (GPT-2 tokenization)
MIN_SHARD_TOKENS = 10000    # Skip tiny shards
MAX_SHARD_BYTES = 4 * 1024**3  # Skip >4GB shards (same as data_loader)
BATCH_CHUNKS = 4            # Process this many chunks at once


def load_teacher(device="cpu"):
    """Load Qwen3-0.6B-Base as the teacher model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading teacher: Qwen3-0.6B-Base...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        torch_dtype=torch.float32,  # CPU needs float32 for speed
        trust_remote_code=True,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Teacher loaded: {n_params:.0f}M params, {time.time()-t0:.1f}s")
    return model, tokenizer


def load_reference_gpt2(device="cpu"):
    """Load GPT-2 small as the reference model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading reference: GPT-2 small (124M)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        torch_dtype=torch.float32,
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Reference loaded: {n_params:.0f}M params, {time.time()-t0:.1f}s")
    return model, tokenizer


def load_reference_sutra(device="cpu"):
    """Load Sutra v0.6.0a as the reference model (more accurate, slower)."""
    sys.path.insert(0, str(REPO / "code"))
    from launch_v060a import create_v060a
    print("Loading reference: Sutra v0.6.0a (68M)...")
    t0 = time.time()

    ckpt_path = REPO / "results" / "checkpoints_v060a" / "best.pt"
    if not ckpt_path.exists():
        # Try rolling
        ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("No v0.6.0a checkpoint found")

    model = create_v060a()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Reference loaded: {n_params:.0f}M params, {time.time()-t0:.1f}s")
    return model, tokenizer


@torch.no_grad()
def score_chunks_transformer(model, tokenizer, texts, device="cpu"):
    """Score a list of text chunks with a HuggingFace causal LM.

    Returns per-character NLL for each text (normalized across tokenizers).
    """
    nlls = []
    for text in texts:
        if not text.strip():
            nlls.append(float("nan"))
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            nlls.append(float("nan"))
            continue
        outputs = model(input_ids)
        logits = outputs.logits[:, :-1, :]  # shift
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction="sum")
        n_chars = max(len(text), 1)
        nlls.append((loss.item()) / n_chars)  # per-character NLL
    return nlls


@torch.no_grad()
def score_chunks_sutra(model, texts, gpt2_tokenizer, device="cpu"):
    """Score text chunks with Sutra model (recurrent, 10 passes)."""
    nlls = []
    for text in texts:
        if not text.strip():
            nlls.append(float("nan"))
            continue
        enc = gpt2_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            nlls.append(float("nan"))
            continue
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        logits, _ = model(x, n_steps=10)  # D_infer=10
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               y.reshape(-1), reduction="sum")
        n_chars = max(len(text), 1)
        nlls.append(loss.item() / n_chars)
    return nlls


def sample_chunks_from_shard(shard_path, gpt2_tokenizer, n_chunks, chunk_len):
    """Load a shard, sample random chunks, decode to text."""
    try:
        tokens = torch.load(shard_path, weights_only=True)
    except Exception as e:
        print(f"  SKIP {shard_path.name}: {e}")
        return []

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
        return []

    n_tokens = tokens.numel()
    if n_tokens < chunk_len + 1:
        return []

    # Sample random start positions
    max_start = n_tokens - chunk_len - 1
    starts = torch.randint(0, max_start + 1, (n_chunks,))

    texts = []
    for s in starts:
        chunk_ids = tokens[s:s + chunk_len].tolist()
        try:
            text = gpt2_tokenizer.decode(chunk_ids, skip_special_tokens=False)
        except Exception:
            text = ""
        if text.strip():
            texts.append(text)

    return texts


def discover_shards(max_shards=None):
    """Discover valid shards, sorted by size (largest first)."""
    shard_files = sorted(SHARD_DIR.glob("*.pt"))
    shards = []
    for f in shard_files:
        sz = os.path.getsize(f)
        if sz > MAX_SHARD_BYTES:
            continue
        est_tokens = max(0, (sz - 200) // 8)
        if est_tokens < MIN_SHARD_TOKENS:
            continue
        shards.append({"path": f, "size_bytes": sz, "est_tokens": est_tokens})

    # Sort by size descending (score big shards first — they matter most)
    shards.sort(key=lambda x: x["size_bytes"], reverse=True)

    if max_shards is not None:
        shards = shards[:max_shards]

    return shards


def main():
    parser = argparse.ArgumentParser(description="MiniPLM offline corpus scoring")
    parser.add_argument("--max-shards", type=int, default=None, help="Limit shards to score")
    parser.add_argument("--reference", choices=["gpt2", "sutra"], default="gpt2",
                        help="Reference model (default: gpt2)")
    parser.add_argument("--chunks", type=int, default=CHUNKS_PER_SHARD,
                        help=f"Chunks per shard (default: {CHUNKS_PER_SHARD})")
    parser.add_argument("--resume", action="store_true", help="Resume from partial results")
    args = parser.parse_args()

    device = "cpu"
    print(f"\n{'='*60}")
    print(f"P1b MiniPLM Corpus Scoring")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Chunks/shard: {args.chunks}")
    print(f"  Chunk length: {CHUNK_LEN} tokens")
    print(f"  Reference: {args.reference}")

    # Discover shards
    shards = discover_shards(args.max_shards)
    print(f"  Shards to score: {len(shards)}")
    total_est_tokens = sum(s["est_tokens"] for s in shards)
    print(f"  Total corpus: ~{total_est_tokens/1e9:.2f}B tokens")

    # Load models
    teacher_model, teacher_tok = load_teacher(device)
    if args.reference == "gpt2":
        ref_model, ref_tok = load_reference_gpt2(device)
        use_sutra_ref = False
    else:
        ref_model, ref_tok = load_reference_sutra(device)
        use_sutra_ref = True

    # GPT-2 tokenizer for decoding shards (shards are GPT-2 encoded)
    from transformers import AutoTokenizer
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

    # Resume from partial results
    results = {}
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            old = json.load(f)
        if "shard_scores" in old:
            results = {s["shard"]: s for s in old["shard_scores"]}
            print(f"  Resuming: {len(results)} shards already scored")

    # Score each shard
    t_start = time.time()
    shard_scores = []
    for i, shard_info in enumerate(shards):
        shard_name = shard_info["path"].name
        if shard_name in results:
            shard_scores.append(results[shard_name])
            continue

        t0 = time.time()
        print(f"\n[{i+1}/{len(shards)}] Scoring {shard_name} "
              f"(~{shard_info['est_tokens']/1e6:.0f}M tokens)...")

        # Sample chunks
        texts = sample_chunks_from_shard(
            shard_info["path"], gpt2_tok, args.chunks, CHUNK_LEN)

        if len(texts) < 3:
            print(f"  SKIP: only {len(texts)} valid chunks")
            continue

        # Score with teacher
        t1 = time.time()
        teacher_nlls = score_chunks_transformer(teacher_model, teacher_tok, texts, device)
        t_teacher = time.time() - t1

        # Score with reference
        t2 = time.time()
        if use_sutra_ref:
            ref_nlls = score_chunks_sutra(ref_model, texts, ref_tok, device)
        else:
            ref_nlls = score_chunks_transformer(ref_model, ref_tok, texts, device)
        t_ref = time.time() - t2

        # Compute importance scores
        import math
        valid_pairs = [(r, t) for r, t in zip(ref_nlls, teacher_nlls)
                       if not (math.isnan(r) or math.isnan(t))]

        if len(valid_pairs) < 3:
            print(f"  SKIP: only {len(valid_pairs)} valid score pairs")
            continue

        ref_arr = [p[0] for p in valid_pairs]
        teacher_arr = [p[1] for p in valid_pairs]

        # MiniPLM importance: high means "teacher finds easy, reference finds hard"
        diffs = [r - t for r, t in valid_pairs]
        mean_diff = sum(diffs) / len(diffs)
        mean_ref = sum(ref_arr) / len(ref_arr)
        mean_teacher = sum(teacher_arr) / len(teacher_arr)

        score_entry = {
            "shard": shard_name,
            "est_tokens": shard_info["est_tokens"],
            "n_chunks_scored": len(valid_pairs),
            "mean_ref_nll_per_char": round(mean_ref, 6),
            "mean_teacher_nll_per_char": round(mean_teacher, 6),
            "mean_difficulty_gap": round(mean_diff, 6),  # ref - teacher
            "max_difficulty_gap": round(max(diffs), 6),
            "min_difficulty_gap": round(min(diffs), 6),
        }
        shard_scores.append(score_entry)

        elapsed = time.time() - t0
        print(f"  teacher={mean_teacher:.4f} ref={mean_ref:.4f} "
              f"gap={mean_diff:.4f} ({elapsed:.1f}s, "
              f"teacher={t_teacher:.1f}s ref={t_ref:.1f}s)")

        # Save intermediate results every 10 shards
        if (i + 1) % 10 == 0:
            _save_results(shard_scores, t_start, args)

    # Final save
    _save_results(shard_scores, t_start, args)
    print(f"\nDone! {len(shard_scores)} shards scored in {time.time()-t_start:.0f}s")
    print(f"Results: {OUTPUT_FILE}")


def _save_results(shard_scores, t_start, args):
    """Save current results with computed weights."""
    if not shard_scores:
        return

    # Compute normalized importance weights
    gaps = [s["mean_difficulty_gap"] for s in shard_scores]
    # Shift so minimum = small positive (no zero weights — every shard gets some probability)
    min_gap = min(gaps)
    shifted = [g - min_gap + 0.01 for g in gaps]
    total = sum(shifted)
    weights = [s / total for s in shifted]

    # Also compute per-source weights (aggregate shards from same source)
    source_scores = {}
    for s, w in zip(shard_scores, weights):
        name = s["shard"]
        parts = name.rsplit("_", 1)
        source = parts[0] if len(parts) == 2 and parts[1].replace(".pt", "").isdigit() else name.replace(".pt", "")
        if source not in source_scores:
            source_scores[source] = {"total_tokens": 0, "total_weight": 0.0,
                                     "n_shards": 0, "mean_gap": 0.0}
        source_scores[source]["total_tokens"] += s["est_tokens"]
        source_scores[source]["total_weight"] += w
        source_scores[source]["n_shards"] += 1
        source_scores[source]["mean_gap"] += s["mean_difficulty_gap"]

    for src in source_scores:
        n = source_scores[src]["n_shards"]
        source_scores[src]["mean_gap"] = round(source_scores[src]["mean_gap"] / n, 6)
        source_scores[src]["total_weight"] = round(source_scores[src]["total_weight"], 6)

    # Annotate shard scores with weights
    for s, w in zip(shard_scores, weights):
        s["importance_weight"] = round(w, 8)

    output = {
        "metadata": {
            "teacher": "Qwen3-0.6B-Base",
            "reference": args.reference,
            "chunks_per_shard": args.chunks,
            "chunk_len": CHUNK_LEN,
            "n_shards_scored": len(shard_scores),
            "elapsed_seconds": round(time.time() - t_start, 1),
            "method": "MiniPLM difference sampling (ref_NLL - teacher_NLL per character)",
        },
        "source_summary": dict(sorted(source_scores.items(),
                                       key=lambda x: x[1]["mean_gap"], reverse=True)),
        "shard_scores": sorted(shard_scores,
                               key=lambda x: x["mean_difficulty_gap"], reverse=True),
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
