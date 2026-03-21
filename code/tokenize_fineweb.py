"""Tokenize FineWeb-Edu with GPT-2 BPE, shard-based for memory efficiency.

Writes token shards to disk (200 chunks each ~11M tokens = ~2.2B per shard).
Final step concatenates shards into one tensor. Peak RAM ~4GB instead of ~70GB.

Usage: python code/tokenize_fineweb.py [--resume]
Output: data/fineweb_tokens.pt (~10B tokens)
"""

import torch
import time
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "fineweb_shards"
CHUNKS_PER_SHARD = 200  # ~2.2B tokens per shard at ~11M tokens/chunk


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    src = REPO / "data" / "fineweb_edu_10bt.txt"
    out = REPO / "data" / "fineweb_tokens.pt"
    log = REPO / "results" / "fineweb_tokenize_log.txt"

    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    file_size = src.stat().st_size
    print(f"Tokenizing {file_size/1e9:.1f}GB FineWeb-Edu (shard-based)...")

    # Resume support: skip already-processed bytes
    existing_shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    resume_offset = 0
    shard_num = len(existing_shards)
    total_tokens = 0
    if existing_shards and "--resume" in sys.argv:
        for sp in existing_shards:
            s = torch.load(sp, weights_only=True)
            total_tokens += len(s)
        # Estimate bytes processed from token count
        # ~1.15 bytes per token for English text with GPT-2 BPE
        resume_offset = int(total_tokens * 1.15)
        print(f"  Resuming: {len(existing_shards)} shards, ~{total_tokens:,} tokens, "
              f"skipping ~{resume_offset/1e9:.1f}GB")

    chunk_size = 50_000_000  # 50MB text chunks
    offset = 0
    chunk_num = 0
    shard_tokens = []
    t0 = time.time()

    with open(src, "r", encoding="utf-8", errors="replace") as f:
        if resume_offset > 0:
            f.seek(resume_offset)
            offset = resume_offset

        while True:
            text = f.read(chunk_size)
            if not text:
                break
            tokens = tokenizer.encode(text)
            shard_tokens.extend(tokens)
            total_tokens += len(tokens)
            offset += len(text.encode("utf-8", errors="replace"))
            chunk_num += 1
            elapsed = time.time() - t0
            rate = offset / max(elapsed, 1) / 1e6

            msg = (f"  Chunk {chunk_num}: {len(tokens):,} tokens, "
                   f"total={total_tokens:,}, "
                   f"{offset/file_size*100:.1f}%, "
                   f"{rate:.1f}MB/s")
            print(msg, flush=True)
            with open(log, "a") as lf:
                lf.write(msg + "\n")

            # Write shard to disk when buffer is large enough
            if chunk_num % CHUNKS_PER_SHARD == 0:
                shard_path = SHARD_DIR / f"shard_{shard_num:04d}.pt"
                torch.save(torch.tensor(shard_tokens, dtype=torch.long), shard_path)
                print(f"  -> Saved shard {shard_num}: {len(shard_tokens):,} tokens", flush=True)
                shard_tokens = []
                shard_num += 1

    # Save final partial shard
    if shard_tokens:
        shard_path = SHARD_DIR / f"shard_{shard_num:04d}.pt"
        torch.save(torch.tensor(shard_tokens, dtype=torch.long), shard_path)
        print(f"  -> Saved final shard {shard_num}: {len(shard_tokens):,} tokens")

    # No concatenation — shards are used directly by the streaming data loader.
    # Symlink into data/shards/ via: python code/data_loader.py --migrate
    total_shards = len(list(SHARD_DIR.glob("shard_*.pt")))
    print(f"\nDone: {total_shards} shards in {SHARD_DIR}")
    print(f"Run 'python code/data_loader.py --migrate' to add to training.")


if __name__ == "__main__":
    main()
