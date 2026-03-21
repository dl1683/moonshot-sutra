"""Universal streaming data loader for Sutra training.

Auto-discovers ALL token shards in data/shards/ and streams from them.
Never loads more than 1 shard into RAM at a time. Handles 100B+ tokens.

Shard directory: data/shards/*.pt
Any .pt file = a source. Naming: {source}_{shard_number}.pt or {source}.pt

Usage:
    from data_loader import ShardedDataset
    dataset = ShardedDataset()  # auto-discovers shards
    x, y = dataset.sample_batch(batch_size=8, seq_len=512, device='cuda', split='train')
"""

import os
import random
import torch
from pathlib import Path

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "shards"


class ShardedDataset:
    """Streaming dataset that samples from random shards without loading all into RAM.

    On init: scans shard directory, builds shard metadata.
    On sample: picks a random shard, loads it, samples a batch, discards it.
    Keeps one "hot" shard cached for consecutive samples.

    Peak RAM: one hot shard at a time.
    """

    def __init__(self, shard_dir=None, test_tokens=50000):
        self.shard_dir = Path(shard_dir) if shard_dir else SHARD_DIR
        self.test_tokens = test_tokens
        self.index = []  # list of shard metadata dicts
        self.sources = {}  # source_name -> total tokens
        self._hot_shard = None  # cached shard tensor
        self._hot_path = None   # path of cached shard
        self._test_tokens = None

        self._build_index()

    def _build_index(self):
        """Scan shard directory and build index."""
        if not self.shard_dir.exists():
            self.shard_dir.mkdir(parents=True)
            # Fallback
            old = REPO / "data" / "minipile_full_tokens.pt"
            if old.exists():
                size = os.path.getsize(old) // 8  # int64 = 8 bytes
                held_out = min(self.test_tokens, max(size - 1, 0))
                self.index.append({"path": old, "n_tokens": size, "held_out_tokens": held_out})
                self.sources["minipile_fallback"] = size
                self._total = size
                print(f"  No shards dir, falling back to {old.name} (~{size/1e9:.2f}B tokens)")
                return

        shard_files = sorted(self.shard_dir.glob("*.pt"))
        if not shard_files:
            old = REPO / "data" / "minipile_full_tokens.pt"
            if old.exists():
                size = os.path.getsize(old) // 8
                held_out = min(self.test_tokens, max(size - 1, 0))
                self.index.append({"path": old, "n_tokens": size, "held_out_tokens": held_out})
                self.sources["minipile_fallback"] = size
                self._total = size
                print(f"  Empty shards dir, falling back to {old.name}")
                return
            raise FileNotFoundError(f"No data in {self.shard_dir}")

        MAX_SHARD_BYTES = 4 * 1024 * 1024 * 1024  # 4GB max shard — skip giant unsplit originals
        total = 0
        skipped_giant = 0
        for f in shard_files:
            file_bytes = os.path.getsize(f)
            if file_bytes > MAX_SHARD_BYTES:
                skipped_giant += 1
                continue  # Skip giant shards — use the split versions instead
            # Estimate token count from file size WITHOUT loading into RAM
            size = max(0, (file_bytes - 200) // 8)  # conservative header estimate
            if size < 2:
                continue

            meta = {"path": f, "n_tokens": size}
            self.index.append(meta)
            total += size

            # Track source
            name = f.stem
            parts = name.rsplit("_", 1)
            source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
            self.sources[source] = self.sources.get(source, 0) + size

        if not self.index:
            raise FileNotFoundError(f"No valid token shards found in {self.shard_dir}")

        # Pick the LARGEST shard for held-out test (not the last/smallest)
        test_meta = max(self.index, key=lambda m: m["n_tokens"])
        held_out = min(self.test_tokens, test_meta["n_tokens"] // 10)  # max 10% of shard
        if held_out < 1000:
            raise ValueError(f"No shard large enough for held-out eval")
        test_meta["held_out_tokens"] = held_out
        for meta in self.index:
            if meta is not test_meta:
                meta["held_out_tokens"] = 0

        # Print summary
        for source in sorted(self.sources.keys()):
            t = self.sources[source]
            print(f"  {source}: ~{t/1e9:.2f}B tokens")
        if skipped_giant:
            print(f"  Skipped {skipped_giant} giant shards (>{MAX_SHARD_BYTES//1e9:.0f}GB) — use split versions")
        print(f"  TOTAL: ~{total/1e9:.2f}B tokens from {len(self.sources)} sources, {len(self.index)} shards")
        self._total = total

    def _load_shard(self, path):
        """Load a shard, caching for repeated access."""
        if self._hot_path == path:
            return self._hot_shard
        try:
            shard = torch.load(path, weights_only=True)
            if not isinstance(shard, torch.Tensor):
                raise TypeError(f"expected Tensor, got {type(shard).__name__}")
            if shard.ndim != 1:
                raise ValueError(f"expected 1D tensor, got shape {tuple(shard.shape)}")
            if shard.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"expected int64/int32 tokens, got {shard.dtype}")
            self._hot_shard = shard.to(dtype=torch.int64)
            self._hot_path = path
            return self._hot_shard
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")
            self._hot_shard = None
            self._hot_path = None
            return None

    def _split_span(self, meta, split):
        n_tokens = meta["n_tokens"]
        held_out = meta.get("held_out_tokens", 0)

        if split == "train":
            return 0, n_tokens - held_out
        if split == "test":
            if held_out <= 0:
                return None
            return n_tokens - held_out, n_tokens
        raise ValueError(f"Unknown split: {split}")

    def _split_candidates(self, seq_len, split):
        candidates = []
        weights = []
        for meta in self.index:
            span = self._split_span(meta, split)
            if span is None:
                continue
            start, end = span
            valid_starts = end - start - seq_len
            if valid_starts <= 0:
                continue
            candidates.append((meta, start, end))
            weights.append(valid_starts)
        if not candidates:
            raise RuntimeError(f"No valid {split} shards available for seq_len={seq_len}")
        return candidates, weights

    def sample_batch(self, batch_size, seq_len, device='cpu', split='train'):
        """Sample a random batch from the requested split.

        Sticky shard: reuse the same shard for multiple batches to avoid
        I/O thrash. Re-sample shard every _sticky_budget batches.
        """
        # Sticky shard logic: reuse current shard for N batches before resampling
        if not hasattr(self, '_sticky_count'):
            self._sticky_count = 0
            self._sticky_meta = None
            self._sticky_budget = 64  # batches per shard before resampling

        candidates, weights = self._split_candidates(seq_len, split)

        # Retry loop in case estimated n_tokens doesn't match actual shard size
        for _ in range(10):
            # Reuse sticky shard if budget remains and shard is still valid
            if (self._sticky_meta is not None and self._sticky_count < self._sticky_budget
                    and self._hot_path == self._sticky_meta["path"]):
                meta = self._sticky_meta
                span = self._split_span(meta, split)
                if span is not None:
                    span_start, span_end = span
                    tokens = self._load_shard(meta["path"])
                    self._sticky_count += 1
                else:
                    self._sticky_meta = None
                    continue
            else:
                meta, span_start, span_end = random.choices(candidates, weights=weights, k=1)[0]
                tokens = self._load_shard(meta["path"])
                self._sticky_meta = meta
                self._sticky_count = 0
            if tokens is None:
                continue

            # Clamp span to actual tensor size (file-size estimate can be off)
            actual_size = tokens.numel()
            span_end = min(span_end, actual_size)
            span_start = min(span_start, actual_size)

            # Update meta with actual size on first load (self-correcting)
            if meta["n_tokens"] != actual_size:
                meta["n_tokens"] = actual_size

            max_start = span_end - seq_len - 1
            if max_start < span_start:
                continue  # Shard too small for this seq_len, try another

            idx = torch.randint(span_start, max_start + 1, (batch_size,))
            x = torch.stack([tokens[i:i + seq_len] for i in idx])
            y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
            return x.to(device), y.to(device)

        raise RuntimeError(f"Failed to sample valid batch after 10 retries (seq_len={seq_len})")

    def get_test_tokens(self, n_tokens=None):
        """Get the fixed held-out tail from the test shard."""
        if self._test_tokens is not None:
            if n_tokens is None or self._test_tokens.numel() <= n_tokens:
                return self._test_tokens
            return self._test_tokens[-n_tokens:]

        # Find the shard with held-out tokens
        meta = next((m for m in self.index if m.get("held_out_tokens", 0) > 0), None)
        if meta is None:
            raise RuntimeError("No held-out test tokens configured")
        held_out = meta["held_out_tokens"]
        if held_out <= 0:
            raise RuntimeError("No held-out test tokens configured")

        tokens = self._load_shard(meta["path"])
        if tokens is None:
            raise RuntimeError(f"Failed to load held-out shard {meta['path']}")

        self._test_tokens = tokens[-held_out:].clone()
        if n_tokens is not None and self._test_tokens.numel() > n_tokens:
            return self._test_tokens[-n_tokens:]
        return self._test_tokens

    @property
    def total_tokens(self):
        return self._total


def migrate_shards():
    """Move existing shards from old locations into data/shards/."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # MiniPile
    mp = REPO / "data" / "minipile_full_tokens.pt"
    dst = SHARD_DIR / "minipile.pt"
    if mp.exists() and not dst.exists():
        print(f"  Linking {mp.name} -> shards/minipile.pt")
        try:
            dst.symlink_to(mp.resolve())
        except OSError:
            import shutil
            shutil.copy2(mp, dst)

    # Diverse shards
    diverse = REPO / "data" / "diverse_shards"
    if diverse.exists():
        for source_dir in sorted(diverse.iterdir()):
            if not source_dir.is_dir():
                continue
            name = source_dir.name
            for shard in sorted(source_dir.glob("shard_*.pt")):
                num = shard.stem.split("_")[1]
                dst = SHARD_DIR / f"{name}_{num}.pt"
                if not dst.exists():
                    try:
                        dst.symlink_to(shard.resolve())
                    except OSError:
                        import shutil
                        shutil.copy2(shard, dst)

    # FineWeb shards
    fineweb = REPO / "data" / "fineweb_shards"
    if fineweb.exists():
        for shard in sorted(fineweb.glob("shard_*.pt")):
            num = shard.stem.split("_")[1]
            dst = SHARD_DIR / f"fineweb_{num}.pt"
            if not dst.exists():
                try:
                    dst.symlink_to(shard.resolve())
                except OSError:
                    import shutil
                    shutil.copy2(shard, dst)

    total = len(list(SHARD_DIR.glob("*.pt")))
    print(f"  Migration complete: {total} shards in {SHARD_DIR}")


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv:
        print("=== MIGRATING SHARDS ===")
        migrate_shards()
    else:
        print("=== TESTING STREAMING LOADER ===")
        ds = ShardedDataset()
        x, y = ds.sample_batch(batch_size=4, seq_len=64, split="train")
        xt, yt = ds.sample_batch(batch_size=4, seq_len=64, split="test")
        print(f"\n  Train batch: x={x.shape}, y={y.shape}")
        print(f"  Test batch: x={xt.shape}, y={yt.shape}")
        print(f"  Held-out tokens: {ds.get_test_tokens().numel():,}")
