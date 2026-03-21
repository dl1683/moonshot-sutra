"""Download additional diverse data sources for maximum representation diversity.

Fills gaps: code, creative writing, news, instructions, structured data, adversarial.

Usage: python code/download_extra_data.py [source_name]
"""

import sys, time, torch, random
from pathlib import Path

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "diverse_shards"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_and_save(texts, source_name, max_tokens=None):
    out_dir = SHARD_DIR / source_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tokens = []
    shard_num = 0
    total = 0
    SHARD_SIZE = 50_000_000
    t0 = time.time()

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
            print(f"  [{source_name}] Shard {shard_num}: {total:,} tokens, {time.time()-t0:.0f}s", flush=True)
        if max_tokens and total >= max_tokens:
            break
        if (i + 1) % 10000 == 0:
            print(f"  [{source_name}] {i+1} docs, {total:,} tokens", flush=True)

    if all_tokens:
        shard_path = out_dir / f"shard_{shard_num:04d}.pt"
        torch.save(torch.tensor(all_tokens, dtype=torch.long), shard_path)
        shard_num += 1
    print(f"  [{source_name}] DONE: {total:,} tokens in {shard_num} shards ({time.time()-t0:.0f}s)")
    return total


def download_code_python(max_tokens=1_000_000_000):
    """Python code from The Stack."""
    print("\n=== PYTHON CODE (The Stack) ===")
    from datasets import load_dataset
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/python",
                       split="train", streaming=True, trust_remote_code=True)
    texts = (item["content"] for item in ds)
    return tokenize_and_save(texts, "code_python", max_tokens=max_tokens)


def download_code_javascript(max_tokens=500_000_000):
    """JavaScript code."""
    print("\n=== JAVASCRIPT CODE ===")
    from datasets import load_dataset
    ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/javascript",
                       split="train", streaming=True, trust_remote_code=True)
    texts = (item["content"] for item in ds)
    return tokenize_and_save(texts, "code_javascript", max_tokens=max_tokens)


def download_gutenberg(max_tokens=2_000_000_000):
    """Project Gutenberg — fiction, poetry, creative writing."""
    print("\n=== PROJECT GUTENBERG ===")
    from datasets import load_dataset
    ds = load_dataset("manu/project_gutenberg", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "gutenberg", max_tokens=max_tokens)


def download_poems(max_tokens=100_000_000):
    """Poetry — diverse creative forms."""
    print("\n=== POETRY ===")
    from datasets import load_dataset
    ds = load_dataset("merve/poetry", split="train")
    texts = [f"{item.get('poem name', '')}\n{item.get('content', '')}" for item in ds]
    return tokenize_and_save(iter(texts), "poetry", max_tokens=max_tokens)


def download_instructions(max_tokens=500_000_000):
    """Alpaca-style instructions — how-to, task completion."""
    print("\n=== INSTRUCTIONS (Alpaca) ===")
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    def fmt(item):
        inp = item.get("input", "")
        return f"Instruction: {item['instruction']}\n{('Input: ' + inp + chr(10)) if inp else ''}Output: {item['output']}"
    texts = [fmt(item) for item in ds]
    return tokenize_and_save(iter(texts), "instructions", max_tokens=max_tokens)


def download_news(max_tokens=1_000_000_000):
    """CC-News — journalism and reporting."""
    print("\n=== CC-NEWS ===")
    from datasets import load_dataset
    ds = load_dataset("cc_news", split="train", streaming=True)
    texts = (item["text"] for item in ds)
    return tokenize_and_save(texts, "cc_news", max_tokens=max_tokens)


def generate_adversarial(n_tokens=50_000_000):
    """Synthetic adversarial/noisy data for robustness.

    Includes: shuffled tokens, repeated patterns, random chars,
    near-miss spelling, mixed languages, format chaos.
    """
    print("\n=== ADVERSARIAL/NOISE DATA ===")
    out_dir = SHARD_DIR / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    all_tokens = []
    patterns = [
        # Repeated tokens (tests handling of repetition)
        lambda: [random.randint(0, 50256)] * random.randint(5, 50),
        # Random token sequences (pure noise baseline)
        lambda: [random.randint(0, 50256) for _ in range(random.randint(10, 100))],
        # Counting patterns
        lambda: tokenizer.encode(" ".join(str(i) for i in range(random.randint(1, 100)))),
        # Reversed text
        lambda: tokenizer.encode("".join(reversed("The quick brown fox jumps over the lazy dog. " * random.randint(1, 5)))),
        # Mixed case chaos
        lambda: tokenizer.encode("".join(c.upper() if random.random() > 0.5 else c.lower()
                                          for c in "This is a normal sentence that has been corrupted. " * 3)),
        # Partial JSON/structured data
        lambda: tokenizer.encode('{"key": "value", "number": ' + str(random.randint(0, 99999)) + ', "list": [1,2,3]}'),
        # Simple arithmetic expressions
        lambda: tokenizer.encode(f"{random.randint(0,999)} + {random.randint(0,999)} = {random.randint(0,1998)}"),
        # Table-like data
        lambda: tokenizer.encode(f"Name: Person{random.randint(1,100)}, Age: {random.randint(18,90)}, City: City{random.randint(1,50)}"),
    ]

    total = 0
    while total < n_tokens:
        pattern = random.choice(patterns)
        toks = pattern()
        all_tokens.extend(toks)
        total += len(toks)
        if len(all_tokens) >= 50_000_000:
            shard_path = out_dir / f"shard_0000.pt"
            torch.save(torch.tensor(all_tokens[:50_000_000], dtype=torch.long), shard_path)
            all_tokens = all_tokens[50_000_000:]

    if all_tokens:
        shard_path = out_dir / f"shard_0000.pt"
        torch.save(torch.tensor(all_tokens, dtype=torch.long), shard_path)

    print(f"  [adversarial] DONE: {total:,} tokens")
    return total


SOURCES = {
    "adversarial": generate_adversarial,
    "instructions": download_instructions,
    "poems": download_poems,
    "gutenberg": download_gutenberg,
    "code_python": download_code_python,
    "code_javascript": download_code_javascript,
    "news": download_news,
}


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target in SOURCES:
        SOURCES[target]()
    elif target:
        print(f"Unknown: {target}. Available: {list(SOURCES.keys())}")
    else:
        # Quick wins first
        for name in ["adversarial", "instructions", "poems", "gutenberg", "code_python", "news"]:
            try:
                SOURCES[name]()
            except Exception as e:
                print(f"  [{name}] FAILED: {e}")
