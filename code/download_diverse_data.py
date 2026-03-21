"""Download diverse training data sources for Sutra.

Downloads and tokenizes multiple data sources to improve generation diversity.
Current training is too heavy on academic papers (MiniPile).

Each source is downloaded, tokenized with GPT-2 BPE, and saved as shards.
Sources are processed sequentially to avoid OOM.

Usage: python code/download_diverse_data.py [source_name]
  Without args: downloads all sources
  With arg: downloads only that source (wikipedia, math, stackexchange, etc.)
"""

import os, sys, time, json, torch
from pathlib import Path
from transformers import AutoTokenizer

REPO = Path(__file__).parent.parent
DATA = REPO / "data"
SHARD_DIR = DATA / "diverse_shards"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize_and_save(texts, source_name, max_tokens=None):
    """Tokenize texts and save as shards."""
    out_dir = SHARD_DIR / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tokens = []
    shard_num = 0
    total = 0
    t0 = time.time()
    SHARD_SIZE = 50_000_000  # 50M tokens per shard

    for i, text in enumerate(texts):
        if not text or len(text.strip()) < 50:
            continue
        toks = tokenizer.encode(text)
        all_tokens.extend(toks)
        total += len(toks)

        if len(all_tokens) >= SHARD_SIZE:
            shard_path = out_dir / f"shard_{shard_num:04d}.pt"
            torch.save(torch.tensor(all_tokens[:SHARD_SIZE], dtype=torch.long), shard_path)
            all_tokens = all_tokens[SHARD_SIZE:]
            shard_num += 1
            elapsed = time.time() - t0
            print(f"  [{source_name}] Shard {shard_num}: {total:,} tokens, {elapsed:.0f}s", flush=True)

        if max_tokens and total >= max_tokens:
            break

        if (i + 1) % 10000 == 0:
            print(f"  [{source_name}] {i+1} docs, {total:,} tokens", flush=True)

    # Save remaining
    if all_tokens:
        shard_path = out_dir / f"shard_{shard_num:04d}.pt"
        torch.save(torch.tensor(all_tokens, dtype=torch.long), shard_path)
        shard_num += 1

    elapsed = time.time() - t0
    print(f"  [{source_name}] DONE: {total:,} tokens in {shard_num} shards ({elapsed:.0f}s)")
    return total


def download_wikipedia(max_tokens=4_000_000_000):
    """Wikipedia English — encyclopedic, factual, well-structured."""
    print("\n=== WIKIPEDIA ENGLISH ===")
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "wikipedia", max_tokens=max_tokens)


def download_openwebmath(max_tokens=3_000_000_000):
    """OpenWebMath — math, LaTeX, proofs, reasoning."""
    print("\n=== OPENWEBMATH ===")
    from datasets import load_dataset
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "openwebmath", max_tokens=max_tokens)


def download_stackexchange(max_tokens=2_000_000_000):
    """StackExchange — Q&A across ALL domains."""
    print("\n=== STACKEXCHANGE ===")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceTB/stack-exchange-preferences", split="train", streaming=True)
    def format_qa(item):
        q = item.get("question", "")
        answers = item.get("answers", [])
        best = answers[0]["text"] if answers else ""
        return f"Question: {q}\nAnswer: {best}"
    texts = (format_qa(item) for item in ds)
    return tokenize_and_save(texts, "stackexchange", max_tokens=max_tokens)


def download_tinystories(max_tokens=500_000_000):
    """TinyStories — narrative coherence for small models."""
    print("\n=== TINYSTORIES ===")
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "tinystories", max_tokens=max_tokens)


def download_openassistant(max_tokens=100_000_000):
    """OpenAssistant — real human conversation."""
    print("\n=== OPENASSISTANT ===")
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst2", split="train")
    texts = [item["text"] for item in ds if item.get("text")]
    return tokenize_and_save(iter(texts), "openassistant", max_tokens=max_tokens)


def download_metamath(max_tokens=300_000_000):
    """MetaMathQA — math with step-by-step solutions."""
    print("\n=== METAMATHQA ===")
    from datasets import load_dataset
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    def format_math(item):
        q = item.get("query", "")
        a = item.get("response", "")
        return f"Problem: {q}\nSolution: {a}"
    texts = (format_math(item) for item in ds)
    return tokenize_and_save(texts, "metamathqa", max_tokens=max_tokens)


def download_gutenberg(max_tokens=3_000_000_000):
    """Project Gutenberg — fiction, literature, narrative."""
    print("\n=== PROJECT GUTENBERG ===")
    from datasets import load_dataset
    ds = load_dataset("manu/project_gutenberg", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "gutenberg", max_tokens=max_tokens)


SOURCES = {
    "wikipedia": download_wikipedia,
    "math": download_openwebmath,
    "stackexchange": download_stackexchange,
    "tinystories": download_tinystories,
    "openassistant": download_openassistant,
    "metamath": download_metamath,
    "gutenberg": download_gutenberg,
}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None

    if target and target in SOURCES:
        SOURCES[target]()
    elif target:
        print(f"Unknown source: {target}. Available: {list(SOURCES.keys())}")
    else:
        # Download all, smallest first (quick wins)
        order = ["openassistant", "metamath", "tinystories", "stackexchange", "wikipedia", "math", "gutenberg"]
        totals = {}
        for name in order:
            try:
                totals[name] = SOURCES[name]()
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
                totals[name] = 0

        print("\n=== SUMMARY ===")
        grand = 0
        for name in order:
            t = totals.get(name, 0)
            grand += t
            print(f"  {name:<20s}: {t/1e9:.2f}B tokens")
        print(f"  {'TOTAL':<20s}: {grand/1e9:.2f}B tokens")
