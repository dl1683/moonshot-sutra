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
    else:
        print("=== TESTING STREAMING LOADER ===")
        ds = ShardedDataset()
        x, y = ds.sample_batch(batch_size=4, seq_len=64, split="train")
        xt, yt = ds.sample_batch(batch_size=4, seq_len=64, split="test")
        print(f"\n  Train batch: x={x.shape}, y={y.shape}")
        print(f"  Test batch: x={xt.shape}, y={yt.shape}")
        print(f"  Held-out tokens: {ds.get_test_tokens().numel():,}")
