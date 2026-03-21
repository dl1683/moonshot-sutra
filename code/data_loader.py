"""Universal streaming data loader for Sutra training.

Auto-discovers ALL token shards in data/shards/ and streams from them.
Never loads more than 1 shard into RAM at a time. Handles 100B+ tokens.

Shard directory: data/shards/*.pt
Any .pt file = a source. Naming: {source}_{shard_number}.pt or {source}.pt

Usage:
    from data_loader import ShardedDataset
    dataset = ShardedDataset()  # auto-discovers shards
    x, y = dataset.sample_batch(batch_size=8, seq_len=512, device='cuda')
"""

import os
import random
import torch
from pathlib import Path

REPO = Path(__file__).parent.parent
SHARD_DIR = REPO / "data" / "shards"


class ShardedDataset:
    """Streaming dataset that samples from random shards without loading all into RAM.

    On init: scans shard directory, builds index of (path, length) pairs.
    On sample: picks a random shard, loads it, samples a batch, discards it.
    Keeps one "hot" shard cached for consecutive samples.

    Peak RAM: ~400MB (one shard of 50M tokens * 8 bytes).
    """

    def __init__(self, shard_dir=None, test_fraction=0.005):
        self.shard_dir = Path(shard_dir) if shard_dir else SHARD_DIR
        self.test_fraction = test_fraction
        self.index = []  # list of (path, n_tokens)
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
                self.index.append((old, size))
                self.sources["minipile_fallback"] = size
                print(f"  No shards dir, falling back to {old.name} (~{size/1e9:.2f}B tokens)")
                return

        shard_files = sorted(self.shard_dir.glob("*.pt"))
        if not shard_files:
            old = REPO / "data" / "minipile_full_tokens.pt"
            if old.exists():
                size = os.path.getsize(old) // 8
                self.index.append((old, size))
                self.sources["minipile_fallback"] = size
                print(f"  Empty shards dir, falling back to {old.name}")
                return
            raise FileNotFoundError(f"No data in {self.shard_dir}")

        total = 0
        for f in shard_files:
            # Estimate token count from file size (int64 = 8 bytes per token)
            # torch .pt files have overhead, but this is close enough for sampling weights
            size = os.path.getsize(f) // 8
            if size < 100:
                continue  # skip tiny/corrupt files
            self.index.append((f, size))
            total += size

            # Track source
            name = f.stem
            parts = name.rsplit("_", 1)
            source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
            self.sources[source] = self.sources.get(source, 0) + size

        # Print summary
        for source in sorted(self.sources.keys()):
            t = self.sources[source]
            print(f"  {source}: ~{t/1e9:.2f}B tokens")
        print(f"  TOTAL: ~{total/1e9:.2f}B tokens from {len(self.sources)} sources, {len(self.index)} shards")

        # Sampling weights proportional to shard size
        self._weights = [s for _, s in self.index]
        self._total = total

    def _load_shard(self, path):
        """Load a shard, caching for repeated access."""
        if self._hot_path == path:
            return self._hot_shard
        try:
            self._hot_shard = torch.load(path, weights_only=True)
            self._hot_path = path
            return self._hot_shard
        except Exception as e:
            print(f"  WARNING: Failed to load {path}: {e}")
            return None

    def sample_batch(self, batch_size, seq_len, device='cpu'):
        """Sample a random batch from a random shard.

        Picks shard weighted by size, loads it (cached), samples random positions.
        """
        # Pick a random shard (weighted by token count)
        path, _ = random.choices(self.index, weights=self._weights, k=1)[0]
        tokens = self._load_shard(path)
        if tokens is None or len(tokens) < seq_len + 1:
            # Retry with different shard
            path, _ = random.choice(self.index)
            tokens = self._load_shard(path)
            if tokens is None or len(tokens) < seq_len + 1:
                raise RuntimeError("No valid shard available")

        max_start = len(tokens) - seq_len - 1
        idx = torch.randint(0, max_start, (batch_size,))
        x = torch.stack([tokens[i:i + seq_len] for i in idx])
        y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
        return x.to(device), y.to(device)

    def get_test_tokens(self, n_tokens=50000):
        """Get a fixed test set (from last shard, deterministic)."""
        if self._test_tokens is not None:
            return self._test_tokens
        # Use last shard as test source
        path = self.index[-1][0]
        tokens = torch.load(path, weights_only=True)
        self._test_tokens = tokens[-n_tokens:]
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
        x, y = ds.sample_batch(batch_size=4, seq_len=64)
        print(f"\n  Batch: x={x.shape}, y={y.shape}")
        print(f"  Peak RAM: ~{ds._weights[0]*8/1e6:.0f}MB per shard (not {ds._total*8/1e9:.1f}GB)")
