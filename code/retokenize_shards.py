"""Re-tokenize GPT-2 shards to 16K BPE tokenizer.
Usage: python code/retokenize_shards.py [--reverse]
  --reverse: process shards in reverse alphabetical order (for parallel instances)
"""
import os, sys, torch
from pathlib import Path
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "shards"
DST_DIR = ROOT / "data" / "shards_16k"
TOK_PATH = ROOT / "data" / "tokenizer_16k" / "tokenizer.json"
CHUNK = 500_000  # tokens per decode chunk (GPT-2 side)

def retokenize_shard(src_path: Path, dst_path: Path, gpt2_tok, tok16k):
    """Decode GPT-2 tokens -> text -> re-encode with 16K tokenizer."""
    data = torch.load(src_path, weights_only=True)
    ids = data.tolist()
    n = len(ids)

    new_ids = []
    for start in range(0, n, CHUNK):
        chunk_ids = ids[start:start + CHUNK]
        text = gpt2_tok.decode(chunk_ids)
        enc = tok16k.encode(text)
        new_ids.extend(enc.ids)

    result = torch.tensor(new_ids, dtype=torch.int64)
    torch.save(result, dst_path)
    return len(ids), len(new_ids)

def main():
    reverse = "--reverse" in sys.argv

    DST_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    print("Loading tokenizers...")
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    tok16k = Tokenizer.from_file(str(TOK_PATH))

    # Get shard list, sorted by file size (smallest first for fast progress)
    all_shards = sorted(SRC_DIR.glob("*.pt"))
    done = set(p.name for p in DST_DIR.glob("*.pt"))
    remaining = [s for s in all_shards if s.name not in done]
    remaining.sort(key=lambda p: p.stat().st_size)

    if reverse:
        remaining = list(reversed(remaining))

    print(f"Total: {len(all_shards)}, Done: {len(done)}, Remaining: {len(remaining)}")

    for i, shard_path in enumerate(remaining):
        dst_path = DST_DIR / shard_path.name
        if dst_path.exists():
            continue  # race condition guard
        print(f"[{i+1}/{len(remaining)}] {shard_path.name}...", end=" ", flush=True)
        try:
            n_old, n_new = retokenize_shard(shard_path, dst_path, gpt2_tok, tok16k)
            ratio = n_new / n_old if n_old > 0 else 0
            print(f"OK ({n_old:,} -> {n_new:,}, ratio={ratio:.3f})")
        except Exception as e:
            print(f"FAILED: {e}")
            if dst_path.exists():
                dst_path.unlink()

    final_count = len(list(DST_DIR.glob("*.pt")))
    print(f"\nDone. {final_count}/{len(all_shards)} shards retokenized.")

if __name__ == "__main__":
    main()
