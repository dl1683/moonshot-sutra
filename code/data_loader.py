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

    def __init__(self, shard_dir=None, test_tokens=50000, weight_file=None):
        self.shard_dir = Path(shard_dir) if shard_dir else SHARD_DIR
        self.test_tokens = test_tokens
        self.index = []  # list of shard metadata dicts
        self.sources = {}  # source_name -> total tokens
        self._shard_weights = {}  # shard_name -> importance weight (from MiniPLM scoring)
        self._hot_shard = None  # cached shard tensor
        self._hot_path = None   # path of cached shard
        self._test_tokens = None

        self._build_index()
        if weight_file is not None:
            self._load_weights(weight_file)

    def _load_weights(self, weight_file):
        """Load MiniPLM importance weights from JSON file."""
        import json
        wf = Path(weight_file)
        if not wf.exists():
            print(f"  WARNING: weight file {wf} not found, using uniform weights")
            return
        with open(wf) as f:
            data = json.load(f)
        for entry in data.get("shard_scores", []):
            name = entry.get("shard", "")
            w = entry.get("importance_weight", None)
            if name and w is not None:
                self._shard_weights[name] = w
        n_matched = sum(1 for m in self.index if m["path"].name in self._shard_weights)
        print(f"  Loaded importance weights: {len(self._shard_weights)} shards, "
              f"{n_matched}/{len(self.index)} matched to index")

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
            # Apply MiniPLM importance weight if available (train split only)
            base_weight = valid_starts
            if split == "train" and self._shard_weights:
                shard_name = meta["path"].name
                importance = self._shard_weights.get(shard_name, 1.0)
                base_weight *= importance
            weights.append(base_weight)
        if not candidates:
            raise RuntimeError(f"No valid {split} shards available for seq_len={seq_len}")
        return candidates, weights

    def sample_batch(self, batch_size, seq_len, device='cpu', split='train'):
        """Sample a random batch from the requested split.

        Sticky shard: reuse the same shard for multiple batches to avoid
        I/O thrash. Re-sample shard every _sticky_budget batches.
        """
        # Sticky shard logic: per-split state to avoid train/test contamination
        if not hasattr(self, '_sticky_state'):
            self._sticky_state = {}  # split -> {count, meta, budget}

        if split not in self._sticky_state:
            self._sticky_state[split] = {"count": 0, "meta": None, "budget": 64}

        ss = self._sticky_state[split]

        candidates, weights = self._split_candidates(seq_len, split)

        # Retry loop in case estimated n_tokens doesn't match actual shard size
        for _ in range(10):
            # Reuse sticky shard if budget remains and shard is still valid
            if (ss["meta"] is not None and ss["count"] < ss["budget"]
                    and self._hot_path == ss["meta"]["path"]):
                meta = ss["meta"]
                span = self._split_span(meta, split)
                if span is not None:
                    span_start, span_end = span
                    tokens = self._load_shard(meta["path"])
                    ss["count"] += 1
                else:
                    ss["meta"] = None
                    continue
            else:
                meta, span_start, span_end = random.choices(candidates, weights=weights, k=1)[0]
                tokens = self._load_shard(meta["path"])
                ss["meta"] = meta
                ss["count"] = 0
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

    def state_dict(self):
        """Capture sticky shard state for faithful resume."""
        ss = getattr(self, '_sticky_state', {})
        per_split = {}
        for split, state in ss.items():
            per_split[split] = {
                "count": state["count"],
                "path": str(state["meta"]["path"]) if state["meta"] else None,
                "budget": state["budget"],
            }
        return {"sticky_per_split": per_split}

    def load_state_dict(self, sd):
        """Restore sticky shard state from checkpoint."""
        if not hasattr(self, '_sticky_state'):
            self._sticky_state = {}
        # Support old format (single sticky state)
        if "sticky_per_split" not in sd:
            count = sd.get("sticky_count", 0)
            budget = sd.get("sticky_budget", 64)
            path = sd.get("sticky_path")
            meta = None
            if path is not None:
                meta = next((m for m in self.index if str(m["path"]) == str(path)), None)
                if meta is not None:
                    self._load_shard(meta["path"])
            self._sticky_state["train"] = {"count": count, "meta": meta, "budget": budget}
            return
        # New per-split format
        for split, state in sd["sticky_per_split"].items():
            path = state.get("path")
            meta = None
            if path is not None:
                meta = next((m for m in self.index if str(m["path"]) == str(path)), None)
                if meta is not None:
                    self._load_shard(meta["path"])  # Warm cache (mirrors old-format branch)
            self._sticky_state[split] = {
                "count": state.get("count", 0),
                "meta": meta,
                "budget": state.get("budget", 64),
            }

    @property
    def total_tokens(self):
        return self._total


def convert_shards_to_bytes(src_dir=None, dst_dir=None):
    """One-time conversion: BPE token shards -> raw UTF-8 byte shards (uint8 tensors).

    Reads each .pt token shard, decodes via tokenizer, saves as uint8 byte tensor.
    Pre-converted byte shards eliminate the 10-80s per-shard decode bottleneck.
    """
    import time as _time
    from tokenizers import Tokenizer

    src = Path(src_dir) if src_dir else (REPO / "data" / "shards_16k")
    dst = Path(dst_dir) if dst_dir else (REPO / "data" / "shards_bytes")
    dst.mkdir(parents=True, exist_ok=True)

    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    shard_files = sorted(src.glob("*.pt"))
    print(f"Converting {len(shard_files)} token shards -> byte shards")
    print(f"  src: {src}")
    print(f"  dst: {dst}")

    total_bytes = 0
    for i, sf in enumerate(shard_files):
        out_path = dst / sf.name
        if out_path.exists():
            total_bytes += os.path.getsize(out_path)
            continue  # already converted

        t0 = _time.time()
        tokens = torch.load(sf, weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            print(f"  SKIP {sf.name}: bad format")
            continue

        # Decode in chunks to limit memory
        chunk_size = 100_000
        byte_parts = []
        token_list = tokens.tolist()
        for j in range(0, len(token_list), chunk_size):
            chunk = token_list[j:j + chunk_size]
            text = tokenizer.decode(chunk, skip_special_tokens=False)
            byte_parts.append(text.encode('utf-8'))
        raw = b''.join(byte_parts)
        byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
        torch.save(byte_tensor, out_path)
        elapsed = _time.time() - t0
        total_bytes += len(raw)
        print(f"  [{i+1}/{len(shard_files)}] {sf.name}: {tokens.numel()/1e6:.1f}M tokens -> "
              f"{len(raw)/1e6:.1f}M bytes ({elapsed:.1f}s)")

    print(f"Done. Total: {total_bytes/1e9:.1f}B bytes in {dst}")
    return str(dst)


class ByteShardedDataset:
    """Streaming byte-level dataset. Uses pre-converted byte shards for speed.

    If data/shards_bytes/ exists with pre-converted uint8 .pt files, loads directly.
    Otherwise falls back to on-the-fly decode from data/shards_16k/ (slow).

    Run convert_shards_to_bytes() once to create the fast byte shards.

    Peak RAM: one hot shard at a time.
    """

    N_BYTES = 256  # byte vocabulary size (0-255)

    def __init__(self, shard_dir=None, test_bytes=1_000_000):
        # Resolve shard directory
        if shard_dir is not None:
            self.shard_dir = Path(shard_dir)
        else:
            # Auto-detect: prefer pre-converted byte shards if available
            byte_dir = REPO / "data" / "shards_bytes"
            if byte_dir.exists() and list(byte_dir.glob("*.pt")):
                self.shard_dir = byte_dir
            else:
                self.shard_dir = REPO / "data" / "shards_16k"

        # Detect format by sampling first shard's dtype
        self._preconverted = False
        sample_files = sorted(self.shard_dir.glob("*.pt"))[:1]
        if sample_files:
            try:
                sample = torch.load(sample_files[0], weights_only=True)
                if isinstance(sample, torch.Tensor) and sample.dtype == torch.uint8:
                    self._preconverted = True
            except Exception:
                pass

        self.test_bytes = test_bytes
        self._tokenizer = None  # lazy load only if needed

        self.index = []
        self.sources = {}
        self._hot_shard = None
        self._hot_path = None
        self._sticky_state = {}

        self._build_index()

    def _build_index(self):
        shard_files = sorted(self.shard_dir.glob("*.pt"))
        MAX_SHARD_BYTES = 4 * 1024 * 1024 * 1024
        total = 0
        skipped = 0
        for f in shard_files:
            file_bytes = os.path.getsize(f)
            if file_bytes > MAX_SHARD_BYTES:
                skipped += 1
                continue
            if self._preconverted:
                # Byte shards: file size ≈ actual byte count (uint8, 1 byte per element + header)
                est_bytes = max(0, file_bytes - 200)
            else:
                # Token shards: estimate ~4.5 bytes per token
                n_tokens = max(0, (file_bytes - 200) // 8)
                if n_tokens < 2:
                    continue
                est_bytes = int(n_tokens * 4.5)
            meta = {"path": f, "est_bytes": est_bytes}
            self.index.append(meta)
            total += est_bytes
            name = f.stem
            parts = name.rsplit("_", 1)
            source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
            self.sources[source] = self.sources.get(source, 0) + est_bytes

        if not self.index:
            raise FileNotFoundError(f"No valid shards in {self.shard_dir}")

        test_meta = max(self.index, key=lambda m: m["est_bytes"])
        held_out = min(self.test_bytes, test_meta["est_bytes"] // 10)
        test_meta["held_out_bytes"] = held_out
        for m in self.index:
            if m is not test_meta:
                m["held_out_bytes"] = 0

        self._total = total
        mode = "pre-converted bytes" if self._preconverted else "on-the-fly decode (SLOW)"
        print(f"  ByteShardedDataset [{mode}]: {len(self.index)} shards, "
              f"~{total/1e9:.1f}B est bytes from {len(self.sources)} sources")
        if skipped:
            print(f"  Skipped {skipped} giant shards")

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(tok_path))

    def _load_shard(self, path):
        """Load a shard, cache as hot shard."""
        if self._hot_path == path:
            return self._hot_shard
        try:
            data = torch.load(path, weights_only=True)
            if not isinstance(data, torch.Tensor) or data.ndim != 1:
                raise ValueError(f"Bad shard: {type(data)}, ndim={getattr(data,'ndim','?')}")

            if data.dtype == torch.uint8:
                # Pre-converted byte shard — load directly (fast path)
                self._hot_shard = data
            elif data.dtype in (torch.int64, torch.int32):
                # Token shard — on-the-fly decode (slow fallback)
                self._ensure_tokenizer()
                chunk_size = 100_000
                byte_parts = []
                token_list = data.tolist()
                for i in range(0, len(token_list), chunk_size):
                    chunk = token_list[i:i + chunk_size]
                    text = self._tokenizer.decode(chunk, skip_special_tokens=False)
                    byte_parts.append(text.encode('utf-8'))
                raw = b''.join(byte_parts)
                self._hot_shard = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
            else:
                raise ValueError(f"Unexpected dtype: {data.dtype}")

            self._hot_path = path
            return self._hot_shard
        except Exception as e:
            print(f"  WARNING: Failed to load shard {path}: {e}")
            self._hot_shard = None
            self._hot_path = None
            return None

    def _split_span(self, meta, split):
        n = meta.get("est_bytes", 0)
        held = meta.get("held_out_bytes", 0)
        if split == "train":
            return 0, n - held
        if split == "test":
            return (n - held, n) if held > 0 else None
        raise ValueError(f"Unknown split: {split}")

    def sample_batch(self, batch_size, seq_len, device='cpu', split='train'):
        """Sample a batch of byte sequences. Returns (x, y) both [B, seq_len] uint8->long."""
        if split not in self._sticky_state:
            self._sticky_state[split] = {"count": 0, "meta": None, "budget": 32}
        ss = self._sticky_state[split]

        # Build candidates
        candidates, weights = [], []
        for meta in self.index:
            span = self._split_span(meta, split)
            if span is None:
                continue
            start, end = span
            if end - start <= seq_len + 1:
                continue
            candidates.append((meta, start, end))
            weights.append(end - start - seq_len)
        if not candidates:
            raise RuntimeError(f"No valid byte shards for split={split}, seq_len={seq_len}")

        for _ in range(10):
            # Sticky shard reuse
            if (ss["meta"] is not None and ss["count"] < ss["budget"]
                    and self._hot_path == ss["meta"]["path"]):
                meta = ss["meta"]
                span = self._split_span(meta, split)
                if span is None:
                    ss["meta"] = None
                    continue
                span_start, span_end = span
                byte_data = self._hot_shard
                ss["count"] += 1
            else:
                meta, span_start, span_end = random.choices(candidates, weights=weights, k=1)[0]
                byte_data = self._load_shard(meta["path"])
                ss["meta"] = meta
                ss["count"] = 0

            if byte_data is None:
                continue

            # Clamp to actual decoded size
            actual = byte_data.numel()
            span_end = min(span_end, actual)
            span_start = min(span_start, actual)
            if span_end - span_start <= seq_len + 1:
                ss["meta"] = None
                continue

            # Sample batch_size windows
            xs, ys = [], []
            for _ in range(batch_size):
                idx = random.randint(span_start, span_end - seq_len - 1)
                window = byte_data[idx:idx + seq_len + 1]
                xs.append(window[:-1])
                ys.append(window[1:])

            x = torch.stack(xs).to(dtype=torch.long, device=device)
            y = torch.stack(ys).to(dtype=torch.long, device=device)
            return x, y

        raise RuntimeError("Failed to sample byte batch after 10 retries")

    @property
    def total_bytes(self):
        return self._total

    def state_dict(self):
        ss = self._sticky_state
        per_split = {}
        for split, state in ss.items():
            per_split[split] = {
                "count": state["count"],
                "path": str(state["meta"]["path"]) if state["meta"] else None,
                "budget": state["budget"],
            }
        return {"sticky_per_split": per_split}

    def load_state_dict(self, sd):
        for split, state in sd.get("sticky_per_split", {}).items():
            path = state.get("path")
            meta = None
            if path is not None:
                meta = next((m for m in self.index if str(m["path"]) == str(path)), None)
                if meta is not None:
                    self._load_shard(meta["path"])
            self._sticky_state[split] = {
                "count": state.get("count", 0),
                "meta": meta,
                "budget": state.get("budget", 32),
            }


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


def score_shards_teacher(shard_dir=None, teacher="qwen3:4b",
                         samples_per_source=2, output_path=None):
    """Score training shards with teacher model for importance weighting.

    Uses Ollama to score representative text samples from each data source.
    Scoring is per-SOURCE (not per-shard) for speed, then applied to all
    shards from that source.

    Scoring criteria (teacher-evaluated):
    - Quality: How well-written/informative is the text?
    - Informativeness: Does this teach useful knowledge?
    - Difficulty: How complex/challenging is the content?

    Combined into a single importance weight per shard, clipped to [0.5, 2.0].
    """
    import json
    import time
    import requests

    if shard_dir is None:
        shard_dir = REPO / "data" / "shards_16k"
    shard_dir = Path(shard_dir)

    # Load tokenizer for decoding
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    if not tok_path.exists():
        print(f"ERROR: Tokenizer not found at {tok_path}")
        return None

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Loaded tokenizer: {tok_path}")

    # Group shards by source
    shard_files = sorted(shard_dir.glob("*.pt"))
    sources = {}  # source_name -> list of shard paths
    for f in shard_files:
        name = f.stem
        parts = name.rsplit("_", 1)
        source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
        if source not in sources:
            sources[source] = []
        sources[source].append(f)

    print(f"Found {len(shard_files)} shards from {len(sources)} sources")

    # Score each source
    PROMPT_TEMPLATE = (
        "Rate this text on three criteria (1-10 each). "
        "Reply ONLY with three numbers separated by spaces.\n"
        "Quality (writing quality): ?\n"
        "Informativeness (teaches useful knowledge): ?\n"
        "Difficulty (complexity): ?\n\n"
        "Text:\n{text}\n\n"
        "Three numbers:"
    )

    source_scores = {}
    for source_name, shard_paths in sorted(sources.items()):
        print(f"\nScoring source: {source_name} ({len(shard_paths)} shards)")
        scores = []

        for sample_idx in range(min(samples_per_source, len(shard_paths))):
            shard_path = shard_paths[sample_idx % len(shard_paths)]
            try:
                tokens = torch.load(shard_path, weights_only=True)
                # Sample a random window
                start = random.randint(0, max(0, len(tokens) - 256))
                window = tokens[start:start + 256].tolist()
                text = tokenizer.decode(window)

                # Truncate for speed
                text = text[:500]

                # Call Ollama
                t0 = time.time()
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": teacher,
                        "prompt": PROMPT_TEMPLATE.format(text=text),
                        "stream": False,
                        "options": {"temperature": 0, "num_predict": 20}
                    },
                    timeout=60
                )
                elapsed = time.time() - t0

                if resp.status_code == 200:
                    reply = resp.json().get("response", "").strip()
                    # Parse three numbers
                    nums = [float(x) for x in reply.split() if x.replace('.', '').isdigit()]
                    if len(nums) >= 3:
                        quality, info, difficulty = nums[0], nums[1], nums[2]
                        # Combined score: weight informativeness highest
                        score = 0.2 * quality + 0.5 * info + 0.3 * difficulty
                        scores.append(score)
                        print(f"  Sample {sample_idx}: q={quality:.0f} i={info:.0f} "
                              f"d={difficulty:.0f} -> {score:.1f} ({elapsed:.1f}s)")
                    else:
                        print(f"  Sample {sample_idx}: Parse failed: '{reply}' ({elapsed:.1f}s)")
                else:
                    print(f"  Sample {sample_idx}: HTTP {resp.status_code}")

            except Exception as e:
                print(f"  Sample {sample_idx}: Error: {e}")

        if scores:
            avg_score = sum(scores) / len(scores)
            source_scores[source_name] = avg_score
            print(f"  -> Average score: {avg_score:.2f}")
        else:
            source_scores[source_name] = 5.0  # default middle score
            print(f"  -> No valid scores, using default 5.0")

    # Normalize to importance weights [0.5, 2.0]
    if source_scores:
        min_s = min(source_scores.values())
        max_s = max(source_scores.values())
        spread = max(max_s - min_s, 0.1)

        shard_scores = []
        for shard_path in shard_files:
            name = shard_path.stem
            parts = name.rsplit("_", 1)
            source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
            raw = source_scores.get(source, 5.0)
            # Map to [0.5, 2.0]
            weight = 0.5 + 1.5 * (raw - min_s) / spread
            weight = max(0.5, min(2.0, weight))
            shard_scores.append({
                "shard": shard_path.name,
                "source": source,
                "raw_score": round(raw, 2),
                "importance_weight": round(weight, 4),
            })

        result = {
            "teacher": teacher,
            "samples_per_source": samples_per_source,
            "source_scores": {k: round(v, 2) for k, v in sorted(source_scores.items())},
            "shard_scores": shard_scores,
        }

        if output_path is None:
            output_path = REPO / "results" / "teacher_shard_weights.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved weights: {output_path}")
        print(f"Source scores: {result['source_scores']}")
        return str(output_path)


def score_miniplm(teacher_name="Qwen/Qwen3-1.7B-Base",
                   reference_name="Qwen/Qwen3-0.6B-Base",
                   n_windows=500, seq_len=512, sample_pct=0.1,
                   output_path=None):
    """MiniPLM difference scoring: score corpus windows by teacher-reference NLL gap.

    For each sampled window, computes:
        score = NLL_reference(window) - NLL_teacher(window)

    High score = teacher knows this content much better than reference = high
    training value for small models.

    Args:
        teacher_name: HuggingFace model ID for teacher (larger model).
        reference_name: HuggingFace model ID for reference (smaller model).
        n_windows: Number of windows to score (for pilot validation).
        seq_len: Window length in our tokenizer's tokens (decoded to text).
        sample_pct: Fraction of corpus to sample from (0.1 = 10%).
        output_path: Where to save results JSON.

    Returns:
        Path to output JSON with per-window scores and histogram.
    """
    import json
    import time
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shard_dir = REPO / "data" / "shards_16k"

    # Load our tokenizer for decoding shards
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    our_tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Loaded shard tokenizer: {tok_path}")

    # Collect windows from shards (sample_pct of corpus)
    shard_files = sorted(shard_dir.glob("*.pt"))
    n_shards = max(1, int(len(shard_files) * sample_pct))
    sampled_shards = random.sample(shard_files, n_shards)
    print(f"Sampling from {n_shards}/{len(shard_files)} shards ({sample_pct*100:.0f}%)")

    # Gather text windows
    windows = []
    shard_sources = []
    for sf in sampled_shards:
        tokens = torch.load(sf, weights_only=True)
        # Extract source name
        name = sf.stem
        parts = name.rsplit("_", 1)
        source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name

        n_possible = max(1, len(tokens) // seq_len)
        n_sample = max(1, min(n_windows // n_shards + 1, n_possible))
        for _ in range(n_sample):
            start = random.randint(0, max(0, len(tokens) - seq_len))
            window_tokens = tokens[start:start + seq_len].tolist()
            text = our_tokenizer.decode(window_tokens)
            if len(text.strip()) > 50:  # skip near-empty windows
                windows.append(text)
                shard_sources.append(source)
            if len(windows) >= n_windows:
                break
        if len(windows) >= n_windows:
            break

    windows = windows[:n_windows]
    shard_sources = shard_sources[:n_windows]
    print(f"Collected {len(windows)} text windows")

    # Load teacher and reference tokenizer (same for Qwen3 family)
    print(f"\nLoading tokenizer from {teacher_name}...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(teacher_name)

    # Score with reference model first (smaller, faster)
    print(f"\nLoading reference model: {reference_name} (CPU, float32)...")
    t0 = time.time()
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_name, dtype=torch.float32, device_map="cpu"
    )
    ref_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in ref_model.parameters()):,}")

    ref_nlls = []
    print(f"Scoring {len(windows)} windows with reference...")
    for i, text in enumerate(windows):
        inputs = qwen_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = ref_model(**inputs, labels=inputs["input_ids"])
            ref_nlls.append(outputs.loss.item())
        if (i + 1) % 50 == 0:
            print(f"  Reference: {i+1}/{len(windows)} (avg NLL: {np.mean(ref_nlls):.4f})")

    del ref_model
    import gc; gc.collect()
    print(f"Reference scoring complete. Mean NLL: {np.mean(ref_nlls):.4f}")

    # Score with teacher model
    print(f"\nLoading teacher model: {teacher_name} (CPU, float32)...")
    t0 = time.time()
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name, dtype=torch.float32, device_map="cpu"
    )
    teacher_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, params: {sum(p.numel() for p in teacher_model.parameters()):,}")

    teacher_nlls = []
    print(f"Scoring {len(windows)} windows with teacher...")
    for i, text in enumerate(windows):
        inputs = qwen_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = teacher_model(**inputs, labels=inputs["input_ids"])
            teacher_nlls.append(outputs.loss.item())
        if (i + 1) % 50 == 0:
            print(f"  Teacher: {i+1}/{len(windows)} (avg NLL: {np.mean(teacher_nlls):.4f})")

    del teacher_model
    gc.collect()
    print(f"Teacher scoring complete. Mean NLL: {np.mean(teacher_nlls):.4f}")

    # Compute difference scores
    ref_nlls = np.array(ref_nlls)
    teacher_nlls = np.array(teacher_nlls)
    diff_scores = ref_nlls - teacher_nlls  # high = teacher knows more

    # Histogram bins
    hist_counts, hist_edges = np.histogram(diff_scores, bins=20)

    # Per-source stats
    source_stats = {}
    for i, src in enumerate(shard_sources):
        if src not in source_stats:
            source_stats[src] = []
        source_stats[src].append(diff_scores[i])
    source_summary = {
        src: {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
              "n": len(scores)}
        for src, scores in source_stats.items()
    }

    # Top-50% selection threshold
    threshold = float(np.median(diff_scores))
    n_selected = int(np.sum(diff_scores >= threshold))

    result = {
        "pilot": "miniplm_difference_scoring",
        "teacher": teacher_name,
        "reference": reference_name,
        "n_windows": len(windows),
        "seq_len": seq_len,
        "sample_pct": sample_pct,
        "stats": {
            "ref_nll_mean": float(np.mean(ref_nlls)),
            "ref_nll_std": float(np.std(ref_nlls)),
            "teacher_nll_mean": float(np.mean(teacher_nlls)),
            "teacher_nll_std": float(np.std(teacher_nlls)),
            "diff_mean": float(np.mean(diff_scores)),
            "diff_std": float(np.std(diff_scores)),
            "diff_min": float(np.min(diff_scores)),
            "diff_max": float(np.max(diff_scores)),
            "diff_median": float(np.median(diff_scores)),
            "threshold_top50": threshold,
            "n_selected": n_selected,
        },
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": [float(e) for e in hist_edges],
        },
        "source_summary": source_summary,
    }

    if output_path is None:
        output_path = REPO / "results" / "miniplm_pilot.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n{'='*60}")
    print(f"MiniPLM pilot complete!")
    print(f"  Windows scored: {len(windows)}")
    print(f"  Ref NLL: {np.mean(ref_nlls):.4f} +/- {np.std(ref_nlls):.4f}")
    print(f"  Teacher NLL: {np.mean(teacher_nlls):.4f} +/- {np.std(teacher_nlls):.4f}")
    print(f"  Diff score: {np.mean(diff_scores):.4f} +/- {np.std(diff_scores):.4f}")
    print(f"  Top-50% threshold: {threshold:.4f}")
    print(f"  Sources: {dict(sorted(source_summary.items(), key=lambda x: -x[1]['mean']))}")
    print(f"  Saved: {output_path}")
    return str(output_path)


def score_miniplm_gpu(student_ckpt, teacher_name="Qwen/Qwen3-1.7B-Base",
                      n_windows=10000, seq_len=512, sample_pct=0.25,
                      top_pct=0.20, output_dir=None):
    """GPU-accelerated MiniPLM scoring using our student as reference.

    Scores windows by student_NLL - teacher_NLL gap (high = teacher knows more).
    Creates curated shard files containing only top-% windows.

    Args:
        student_ckpt: Path to student checkpoint (.pt file).
        teacher_name: HuggingFace model ID for teacher.
        n_windows: Total windows to score.
        seq_len: Window length in our tokenizer's tokens.
        sample_pct: Fraction of shards to sample from.
        top_pct: Fraction of windows to keep (top by gap score).
        output_dir: Where to save curated shards. Default: data/shards_miniplm_top20.
    """
    import json, time, gc
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_dir = REPO / "data" / "shards_16k"

    # Load our tokenizer
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    our_tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Loaded shard tokenizer: {tok_path}")

    # --- Load student model from checkpoint ---
    print(f"\nLoading student from {Path(student_ckpt).name}...")
    t0 = time.time()
    sys.path.insert(0, str(REPO / "code"))
    from dense_baseline import DenseTransformer
    ckpt = torch.load(student_ckpt, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    student = DenseTransformer(
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 24),
        n_heads=cfg.get("n_heads", 12),
        ff_dim=cfg.get("ff_dim", 2304),
        exit_layers=cfg.get("exit_layers"),
        norm_type=cfg.get("norm_type", "rmsnorm"),
        block_schedule=cfg.get("block_schedule"),
        conv_kernel_size=cfg.get("conv_kernel_size", 4),
        head_dim=cfg.get("head_dim", 64),
        n_q_heads=cfg.get("n_q_heads"),
        n_kv_heads=cfg.get("n_kv_heads"),
    )
    init_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    student.load_state_dict(ckpt[init_key])
    student = student.to(device=device, dtype=torch.bfloat16)
    student.eval()
    n_params = sum(p.numel() for p in student.parameters())
    del ckpt; gc.collect()
    print(f"  Student loaded in {time.time()-t0:.1f}s ({n_params/1e6:.1f}M params)")

    # --- Load teacher ---
    print(f"\nLoading teacher: {teacher_name} (GPU, bf16)...")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, dtype=torch.bfloat16, device_map=device
    )
    teacher.eval()
    qwen_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    n_t_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher loaded in {time.time()-t0:.1f}s ({n_t_params/1e6:.1f}M params)")

    # --- Collect windows from shards ---
    shard_files = sorted(shard_dir.glob("*.pt"))
    n_shards = max(1, int(len(shard_files) * sample_pct))
    sampled_shards = random.sample(shard_files, n_shards)
    print(f"\nSampling from {n_shards}/{len(shard_files)} shards ({sample_pct*100:.0f}%)")

    windows = []       # (token_ids_list, text, shard_name, start_offset)
    for sf in sampled_shards:
        tokens = torch.load(sf, weights_only=True)
        name = sf.stem
        n_possible = max(1, len(tokens) // seq_len)
        n_sample = max(1, min(n_windows // n_shards + 1, n_possible))
        for _ in range(n_sample):
            start = random.randint(0, max(0, len(tokens) - seq_len))
            window_tok = tokens[start:start + seq_len]
            text = our_tokenizer.decode(window_tok.tolist())
            if len(text.strip()) > 50:
                windows.append((window_tok, text, name, start))
            if len(windows) >= n_windows:
                break
        if len(windows) >= n_windows:
            break
    windows = windows[:n_windows]
    print(f"Collected {len(windows)} text windows")

    # --- Score with student (our tokenizer) ---
    print(f"\nScoring {len(windows)} windows with student...")
    student_nlls = []
    t0 = time.time()
    for i, (tok_ids, text, sname, soff) in enumerate(windows):
        x = tok_ids[:-1].unsqueeze(0).to(device)
        y = tok_ids[1:].unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = student(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Handle potential extra dimensions from exit heads
            if logits.dim() == 3:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        student_nlls.append(loss.item())
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Student: {i+1}/{len(windows)} ({elapsed:.0f}s, "
                  f"avg NLL: {np.mean(student_nlls):.4f})")
    print(f"Student scoring done: {time.time()-t0:.0f}s, mean NLL={np.mean(student_nlls):.4f}")

    # --- Score with teacher (Qwen tokenizer) ---
    print(f"\nScoring {len(windows)} windows with teacher...")
    teacher_nlls = []
    t0 = time.time()
    for i, (tok_ids, text, sname, soff) in enumerate(windows):
        inputs = qwen_tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512).to(device)
        with torch.no_grad():
            outputs = teacher(**inputs, labels=inputs["input_ids"])
            teacher_nlls.append(outputs.loss.item())
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  Teacher: {i+1}/{len(windows)} ({elapsed:.0f}s, "
                  f"avg NLL: {np.mean(teacher_nlls):.4f})")
    print(f"Teacher scoring done: {time.time()-t0:.0f}s, mean NLL={np.mean(teacher_nlls):.4f}")

    # --- Compute gap scores ---
    student_nlls = np.array(student_nlls)
    teacher_nlls = np.array(teacher_nlls)
    gap_scores = student_nlls - teacher_nlls  # high = teacher knows much more

    # Select top windows
    threshold = np.percentile(gap_scores, (1.0 - top_pct) * 100)
    selected_mask = gap_scores >= threshold
    n_selected = int(selected_mask.sum())
    print(f"\n{'='*60}")
    print(f"Gap scores: mean={np.mean(gap_scores):.4f}, std={np.std(gap_scores):.4f}")
    print(f"  min={np.min(gap_scores):.4f}, max={np.max(gap_scores):.4f}")
    print(f"  threshold (top {top_pct*100:.0f}%): {threshold:.4f}")
    print(f"  Selected: {n_selected}/{len(windows)} windows")

    # --- Create curated shards ---
    if output_dir is None:
        output_dir = REPO / "data" / f"shards_miniplm_top{int(top_pct*100)}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group selected windows by source shard for efficient storage
    shard_windows = {}
    for i, (tok_ids, text, sname, soff) in enumerate(windows):
        if selected_mask[i]:
            if sname not in shard_windows:
                shard_windows[sname] = []
            shard_windows[sname].append((tok_ids, gap_scores[i]))

    total_tokens = 0
    shard_count = 0
    for sname, wlist in sorted(shard_windows.items()):
        # Sort by gap score (highest first) within each shard
        wlist.sort(key=lambda x: -x[1])
        # Concatenate all selected windows into one shard
        all_tokens = torch.cat([w[0] for w in wlist])
        out_path = output_dir / f"{sname}_curated.pt"
        torch.save(all_tokens, out_path)
        total_tokens += len(all_tokens)
        shard_count += 1

    print(f"\nCurated shards saved: {shard_count} files in {output_dir}")
    print(f"Total curated tokens: {total_tokens:,} ({total_tokens/1e6:.1f}M)")

    # --- Save scoring results ---
    hist_counts, hist_edges = np.histogram(gap_scores, bins=30)
    results = {
        "experiment": "miniplm_top20_ce_3k",
        "student_ckpt": str(student_ckpt),
        "teacher": teacher_name,
        "n_windows_scored": len(windows),
        "n_windows_selected": n_selected,
        "top_pct": top_pct,
        "seq_len": seq_len,
        "sample_pct": sample_pct,
        "stats": {
            "student_nll_mean": float(np.mean(student_nlls)),
            "student_nll_std": float(np.std(student_nlls)),
            "teacher_nll_mean": float(np.mean(teacher_nlls)),
            "teacher_nll_std": float(np.std(teacher_nlls)),
            "gap_mean": float(np.mean(gap_scores)),
            "gap_std": float(np.std(gap_scores)),
            "gap_min": float(np.min(gap_scores)),
            "gap_max": float(np.max(gap_scores)),
            "gap_median": float(np.median(gap_scores)),
            "threshold": float(threshold),
        },
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": [float(e) for e in hist_edges],
        },
        "curated_dir": str(output_dir),
        "curated_shards": shard_count,
        "curated_tokens": total_tokens,
    }
    results_path = REPO / "results" / "miniplm_gpu_scoring.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # Cleanup
    del student, teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(results_path)


def precompute_teacher_logits(teacher_name="Qwen/Qwen3-1.7B-Base",
                              curated_dir=None, seq_len=512, top_k=64,
                              output_dir=None, batch_size=8):
    """Pre-compute teacher top-K logits for curated shard windows.

    For each 512-token window in curated shards, runs teacher forward pass
    and caches top-K logit values, indices, and byte offsets. This enables
    offline KD training without loading the teacher during training.

    Output per shard: {shard_name}_logits.pt containing dict:
      - 'topk_vals': (N_windows, T_teacher, K) float16
      - 'topk_ids': (N_windows, T_teacher, K) int16
      - 'byte_offsets': (N_windows, T_teacher, 2) int32
      - 'attn_mask': (N_windows, T_teacher) bool
      - 'n_teacher_tokens': list of int (actual seq len per window)
    """
    import time, gc, json
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if curated_dir is None:
        curated_dir = REPO / "data" / "shards_miniplm_top20"
    else:
        curated_dir = Path(curated_dir)
    if output_dir is None:
        output_dir = REPO / "data" / "teacher_logits_q17"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load student tokenizer (for decoding tokens -> text)
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    our_tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Student tokenizer: {tok_path}")

    # Load teacher
    print(f"\nLoading teacher: {teacher_name}...")
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, dtype=torch.bfloat16, device_map=device
    )
    teacher.eval()
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Loaded in {time.time()-t0:.1f}s ({n_params/1e6:.1f}M params)")

    # Process each curated shard
    shard_files = sorted(curated_dir.glob("*.pt"))
    print(f"\nProcessing {len(shard_files)} curated shards...")

    total_windows = 0
    total_time = 0

    for si, shard_path in enumerate(shard_files):
        out_path = output_dir / f"{shard_path.stem}_logits.pt"
        if out_path.exists():
            # Skip already computed
            n = torch.load(shard_path, weights_only=True).numel() // seq_len
            total_windows += n
            print(f"  [{si+1}/{len(shard_files)}] {shard_path.stem}: SKIPPED (exists), {n} windows")
            continue

        t0 = time.time()
        tokens = torch.load(shard_path, weights_only=True)
        n_windows = len(tokens) // seq_len
        if n_windows == 0:
            continue

        # Split into windows
        windows = tokens[:n_windows * seq_len].view(n_windows, seq_len)

        # Decode each window to text using student tokenizer
        texts = []
        for w in range(n_windows):
            text = our_tokenizer.decode(windows[w].tolist(), skip_special_tokens=False)
            texts.append(text)

        # Batch teacher forward passes
        all_topk_vals = []
        all_topk_ids = []
        all_byte_offsets = []
        all_attn_masks = []
        all_n_teacher_tokens = []

        for batch_start in range(0, n_windows, batch_size):
            batch_texts = texts[batch_start:batch_start + batch_size]

            # Tokenize for teacher
            t_enc = teacher_tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=seq_len,
                return_offsets_mapping=True
            )

            input_ids = t_enc["input_ids"].to(device)
            attn_mask = t_enc["attention_mask"].to(device)
            offsets = t_enc["offset_mapping"]  # (B, T_t, 2) on CPU

            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
                out = teacher(input_ids=input_ids, attention_mask=attn_mask)
                logits = out.logits  # (B, T_t, V)

            # Extract top-K
            K = min(top_k, logits.shape[-1])
            topk = logits.topk(K, dim=-1)
            topk_vals = topk.values.half().cpu()  # (B, T_t, K)
            topk_ids = topk.indices.int().cpu()  # (B, T_t, K) — int32, NOT int16 (vocab > 32K)

            all_topk_vals.append(topk_vals)
            all_topk_ids.append(topk_ids)
            all_byte_offsets.append(offsets.int())
            all_attn_masks.append(attn_mask.cpu().bool())
            for b in range(len(batch_texts)):
                all_n_teacher_tokens.append(attn_mask[b].sum().item())

            del logits, out, topk
            torch.cuda.empty_cache()

        # Pad to uniform teacher seq len and save
        max_t_len = max(v.shape[1] for v in all_topk_vals)
        padded_vals = torch.zeros(n_windows, max_t_len, top_k, dtype=torch.float16)
        padded_ids = torch.zeros(n_windows, max_t_len, top_k, dtype=torch.int32)
        padded_offsets = torch.zeros(n_windows, max_t_len, 2, dtype=torch.int32)
        padded_mask = torch.zeros(n_windows, max_t_len, dtype=torch.bool)

        idx = 0
        for vals, ids, offs, masks in zip(all_topk_vals, all_topk_ids,
                                           all_byte_offsets, all_attn_masks):
            B, T = vals.shape[:2]
            padded_vals[idx:idx+B, :T] = vals
            padded_ids[idx:idx+B, :T] = ids
            padded_offsets[idx:idx+B, :T] = offs[:, :T]
            padded_mask[idx:idx+B, :T] = masks
            idx += B

        cache = {
            "topk_vals": padded_vals,
            "topk_ids": padded_ids,
            "byte_offsets": padded_offsets,
            "attn_mask": padded_mask,
            "n_teacher_tokens": all_n_teacher_tokens,
            "teacher": teacher_name,
            "top_k": top_k,
            "seq_len": seq_len,
        }
        torch.save(cache, out_path)

        dt = time.time() - t0
        total_windows += n_windows
        total_time += dt
        rate = n_windows / max(dt, 0.01)
        print(f"  [{si+1}/{len(shard_files)}] {shard_path.stem}: {n_windows} windows, "
              f"{dt:.1f}s ({rate:.1f} win/s)")

    # Summary
    cache_size = sum(f.stat().st_size for f in output_dir.glob("*.pt"))
    print(f"\n{'='*60}")
    print(f"Teacher logit cache complete!")
    print(f"  Windows: {total_windows}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Cache: {output_dir} ({cache_size/1e9:.2f}GB)")
    print(f"{'='*60}")

    # Save metadata
    meta = {
        "teacher": teacher_name,
        "top_k": top_k,
        "seq_len": seq_len,
        "total_windows": total_windows,
        "cache_dir": str(output_dir),
        "cache_size_gb": cache_size / 1e9,
        "curated_dir": str(curated_dir),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return str(output_dir)


def score_transport_gap(teacher_name="Qwen/Qwen3-1.7B-Base",
                        reference_name="Qwen/Qwen3-0.6B-Base",
                        n_windows=100000, seq_len=512, sample_pct=1.0,
                        output_path=None, batch_size=16):
    """Score corpus windows by teacher-reference NLL gap for data transport.

    Track 2: The teacher's value is in selecting which data the student sees.
    gap(x) = NLL_ref(x) - NLL_teacher(x), z-scored globally.
    High gap = teacher knows this much better than reference = high training value.

    Both models run on GPU in bf16 for speed. Processes windows in batches.
    Saves per-window data for 3-pool construction.
    """
    import json, time, gc
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    shard_dir = REPO / "data" / "shards_16k"

    # Load our tokenizer for decoding shards
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    from tokenizers import Tokenizer
    our_tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Loaded shard tokenizer: {tok_path}")

    # --- Collect windows from shards ---
    shard_files = sorted(shard_dir.glob("*.pt"))
    n_shards = max(1, int(len(shard_files) * sample_pct))
    sampled_shards = random.sample(shard_files, n_shards)
    print(f"Sampling from {n_shards}/{len(shard_files)} shards ({sample_pct*100:.0f}%)")

    windows = []  # list of (token_ids, text, source_name)
    for sf in sampled_shards:
        tokens = torch.load(sf, weights_only=True)
        name = sf.stem
        parts = name.rsplit("_", 1)
        source = parts[0] if len(parts) == 2 and parts[1].isdigit() else name
        n_possible = max(1, len(tokens) // seq_len)
        n_sample = max(1, min(n_windows // n_shards + 1, n_possible))
        for _ in range(n_sample):
            start = random.randint(0, max(0, len(tokens) - seq_len))
            window_tok = tokens[start:start + seq_len]
            text = our_tokenizer.decode(window_tok.tolist())
            if len(text.strip()) > 50:
                windows.append((window_tok, text, source))
            if len(windows) >= n_windows:
                break
        if len(windows) >= n_windows:
            break
    windows = windows[:n_windows]
    print(f"Collected {len(windows)} text windows")

    # --- Load reference model (smaller, Qwen3-0.6B) ---
    print(f"\nLoading reference: {reference_name} (GPU, bf16)...")
    t0 = time.time()
    ref_tokenizer = AutoTokenizer.from_pretrained(reference_name)
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_name, dtype=torch.bfloat16, device_map=device
    )
    ref_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, "
          f"params: {sum(p.numel() for p in ref_model.parameters()):,}")

    # --- Score with reference (batched) ---
    print(f"\nScoring {len(windows)} windows with reference...")
    ref_nlls = []
    t0 = time.time()
    for i in range(0, len(windows), batch_size):
        batch_texts = [w[1] for w in windows[i:i+batch_size]]
        inputs = ref_tokenizer(batch_texts, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = ref_model(**inputs)
            logits = outputs.logits
        # Compute per-sample NLL (can't use labels= with padding)
        for j in range(len(batch_texts)):
            mask = inputs["attention_mask"][j]
            n_tok = mask.sum().item()
            if n_tok <= 1:
                ref_nlls.append(10.0)  # degenerate
                continue
            sample_logits = logits[j, :n_tok-1]
            sample_labels = inputs["input_ids"][j, 1:n_tok]
            nll = torch.nn.functional.cross_entropy(
                sample_logits, sample_labels).item()
            ref_nlls.append(nll)
        if (i + batch_size) % (batch_size * 50) == 0 or i + batch_size >= len(windows):
            elapsed = time.time() - t0
            n_done = min(i + batch_size, len(windows))
            print(f"  Reference: {n_done}/{len(windows)} ({elapsed:.0f}s, "
                  f"avg NLL: {np.mean(ref_nlls):.4f})")
    print(f"Reference done: {time.time()-t0:.0f}s, mean NLL={np.mean(ref_nlls):.4f}")

    # Free reference model
    del ref_model, ref_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Load teacher model ---
    print(f"\nLoading teacher: {teacher_name} (GPU, bf16)...")
    t0 = time.time()
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_name, dtype=torch.bfloat16, device_map=device
    )
    teacher_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s, "
          f"params: {sum(p.numel() for p in teacher_model.parameters()):,}")

    # --- Score with teacher (batched) ---
    print(f"\nScoring {len(windows)} windows with teacher...")
    teacher_nlls = []
    t0 = time.time()
    for i in range(0, len(windows), batch_size):
        batch_texts = [w[1] for w in windows[i:i+batch_size]]
        inputs = teacher_tokenizer(batch_texts, return_tensors="pt", truncation=True,
                                   max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = teacher_model(**inputs)
            logits = outputs.logits
        for j in range(len(batch_texts)):
            mask = inputs["attention_mask"][j]
            n_tok = mask.sum().item()
            if n_tok <= 1:
                teacher_nlls.append(10.0)
                continue
            sample_logits = logits[j, :n_tok-1]
            sample_labels = inputs["input_ids"][j, 1:n_tok]
            nll = torch.nn.functional.cross_entropy(
                sample_logits, sample_labels).item()
            teacher_nlls.append(nll)
        if (i + batch_size) % (batch_size * 50) == 0 or i + batch_size >= len(windows):
            elapsed = time.time() - t0
            n_done = min(i + batch_size, len(windows))
            print(f"  Teacher: {n_done}/{len(windows)} ({elapsed:.0f}s, "
                  f"avg NLL: {np.mean(teacher_nlls):.4f})")
    print(f"Teacher done: {time.time()-t0:.0f}s, mean NLL={np.mean(teacher_nlls):.4f}")

    del teacher_model, teacher_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # --- Compute gap and z-scores ---
    ref_nlls = np.array(ref_nlls)
    teacher_nlls = np.array(teacher_nlls)
    gap = ref_nlls - teacher_nlls  # high = teacher knows much more
    z_scores = (gap - gap.mean()) / max(gap.std(), 1e-8)

    # Per-source stats
    source_stats = {}
    for i, (_, _, src) in enumerate(windows):
        if src not in source_stats:
            source_stats[src] = []
        source_stats[src].append(float(z_scores[i]))

    source_summary = {
        src: {"mean_z": float(np.mean(scores)), "std_z": float(np.std(scores)),
              "n": len(scores)}
        for src, scores in source_stats.items()
    }

    # --- Save per-window data for 3-pool construction ---
    # Save token IDs separately (large tensor) and metadata as JSON
    window_meta = []
    all_token_ids = []
    for i, (tok_ids, text, src) in enumerate(windows):
        window_meta.append({
            "idx": i,
            "source": src,
            "ref_nll": float(ref_nlls[i]),
            "teacher_nll": float(teacher_nlls[i]),
            "gap": float(gap[i]),
            "z_score": float(z_scores[i]),
            "n_tokens": len(tok_ids),
        })
        # Pad to seq_len for uniform stacking
        if len(tok_ids) < seq_len:
            tok_ids = torch.nn.functional.pad(tok_ids, (0, seq_len - len(tok_ids)))
        all_token_ids.append(tok_ids[:seq_len])

    # Thresholds for pool construction
    z_hard_raw = float(np.percentile(z_scores, 85))  # top 15%
    z_hard_rewrite = float(np.percentile(z_scores, 95))  # top 5%

    result = {
        "experiment": "transport_gap_scoring",
        "teacher": teacher_name,
        "reference": reference_name,
        "n_windows": len(windows),
        "seq_len": seq_len,
        "stats": {
            "ref_nll_mean": float(np.mean(ref_nlls)),
            "ref_nll_std": float(np.std(ref_nlls)),
            "teacher_nll_mean": float(np.mean(teacher_nlls)),
            "teacher_nll_std": float(np.std(teacher_nlls)),
            "gap_mean": float(np.mean(gap)),
            "gap_std": float(np.std(gap)),
            "gap_min": float(np.min(gap)),
            "gap_max": float(np.max(gap)),
            "z_threshold_top15": z_hard_raw,
            "z_threshold_top5": z_hard_rewrite,
        },
        "source_summary": source_summary,
        "windows": window_meta,
    }

    if output_path is None:
        output_path = REPO / "results" / "transport_gap_100k.json"
    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save token IDs as tensor
    tokens_path = output_path.with_suffix(".pt")
    torch.save(torch.stack(all_token_ids), tokens_path)

    print(f"\n{'='*60}")
    print(f"TRANSPORT GAP SCORING COMPLETE")
    print(f"  Windows: {len(windows)}")
    print(f"  Ref NLL: {np.mean(ref_nlls):.4f} +/- {np.std(ref_nlls):.4f}")
    print(f"  Teacher NLL: {np.mean(teacher_nlls):.4f} +/- {np.std(teacher_nlls):.4f}")
    print(f"  Gap: {np.mean(gap):.4f} +/- {np.std(gap):.4f}")
    print(f"  z threshold (top 15%): {z_hard_raw:.4f}")
    print(f"  z threshold (top 5%): {z_hard_rewrite:.4f}")
    print(f"  Top sources by mean z: "
          f"{sorted(source_summary.items(), key=lambda x: -x[1]['mean_z'])[:5]}")
    print(f"  Saved: {output_path} + {tokens_path}")
    print(f"{'='*60}")
    return str(output_path)


if __name__ == "__main__":
    import sys
    if "--migrate" in sys.argv:
        print("=== MIGRATING SHARDS ===")
        migrate_shards()
    elif "--score-shards" in sys.argv:
        teacher = "qwen3:4b"
        for arg in sys.argv:
            if arg.startswith("--teacher="):
                teacher = arg.split("=", 1)[1]
        print(f"=== TEACHER-GUIDED SHARD SCORING ({teacher}) ===")
        score_shards_teacher(teacher=teacher)
    elif "--miniplm" in sys.argv:
        n_windows = 500
        sample_pct = 0.1
        output_path = None
        for arg in sys.argv:
            if arg.startswith("--n-windows="):
                n_windows = int(arg.split("=", 1)[1])
            elif arg.startswith("--sample-pct="):
                sample_pct = float(arg.split("=", 1)[1])
            elif arg.startswith("--output="):
                output_path = arg.split("=", 1)[1]
        print(f"=== MINIPLM DIFFERENCE SCORING (n={n_windows}, sample={sample_pct*100:.0f}%) ===")
        score_miniplm(n_windows=n_windows, sample_pct=sample_pct,
                      output_path=output_path)
    elif "--precompute-logits" in sys.argv:
        teacher = "Qwen/Qwen3-1.7B-Base"
        batch_size = 8
        top_k = 64
        for arg in sys.argv:
            if arg.startswith("--teacher="):
                teacher = arg.split("=", 1)[1]
            elif arg.startswith("--batch-size="):
                batch_size = int(arg.split("=", 1)[1])
            elif arg.startswith("--top-k="):
                top_k = int(arg.split("=", 1)[1])
        print(f"=== PRE-COMPUTING TEACHER LOGITS ({teacher}, K={top_k}, bs={batch_size}) ===")
        precompute_teacher_logits(teacher_name=teacher, top_k=top_k, batch_size=batch_size)
    elif "--miniplm-gpu" in sys.argv:
        student_ckpt = None
        teacher = "Qwen/Qwen3-1.7B-Base"
        n_windows = 10000
        sample_pct = 0.25
        top_pct = 0.20
        for arg in sys.argv:
            if arg.startswith("--student="):
                student_ckpt = arg.split("=", 1)[1]
            elif arg.startswith("--teacher="):
                teacher = arg.split("=", 1)[1]
            elif arg.startswith("--n-windows="):
                n_windows = int(arg.split("=", 1)[1])
            elif arg.startswith("--sample-pct="):
                sample_pct = float(arg.split("=", 1)[1])
            elif arg.startswith("--top-pct="):
                top_pct = float(arg.split("=", 1)[1])
        if student_ckpt is None:
            print("ERROR: --student=<checkpoint.pt> required")
            sys.exit(1)
        print(f"=== GPU MINIPLM SCORING (n={n_windows}, top={top_pct*100:.0f}%) ===")
        score_miniplm_gpu(student_ckpt=student_ckpt, teacher_name=teacher,
                          n_windows=n_windows, sample_pct=sample_pct,
                          top_pct=top_pct)
    elif "--transport-gap" in sys.argv:
        n_windows = 100000
        batch_size = 16
        for arg in sys.argv:
            if arg.startswith("--n-windows="):
                n_windows = int(arg.split("=", 1)[1])
            elif arg.startswith("--batch-size="):
                batch_size = int(arg.split("=", 1)[1])
        print(f"=== TRANSPORT GAP SCORING (n={n_windows}, bs={batch_size}) ===")
        score_transport_gap(n_windows=n_windows, batch_size=batch_size)
    else:
        print("=== TESTING STREAMING LOADER ===")
        ds = ShardedDataset()
        x, y = ds.sample_batch(batch_size=4, seq_len=64, split="train")
        xt, yt = ds.sample_batch(batch_size=4, seq_len=64, split="test")
        print(f"\n  Train batch: x={x.shape}, y={y.shape}")
        print(f"  Test batch: x={xt.shape}, y={yt.shape}")
        print(f"  Held-out tokens: {ds.get_test_tokens().numel():,}")
