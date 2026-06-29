"""Eklavya E1 — Offline teacher signal cache builder.

Builds two types of cached records from a teacher model:
1. AlignRecords: token-span → patch mapping for embedding alignment
2. ByteKLRecords: first-byte marginals at selected patch positions

Cache is sparse: only selected high-value positions are stored.
Teacher runs offline with torch.no_grad(), quantized if needed.

Usage:
    python eklavya_cache.py \
        --teacher <HuggingFace-model-ID> \
        --data-dir data/shards_bytes_full \
        --output-dir eklavya_cache \
        --student-checkpoint checkpoints/s0/s0_best.pt \
        --max-shards 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class AlignRecord:
    shard_id: int
    seq_offset: int
    byte_start: int
    byte_len: int
    token_id: int


@dataclass
class ByteKLRecord:
    shard_id: int
    seq_offset: int
    patch_idx: int
    top_bytes: np.ndarray    # uint8[K]
    top_probs: np.ndarray    # float16[K]
    tail_prob: float
    entropy: float
    byte_pos: int = 0        # byte position within patch (0-3 for P=4)


def build_token_byte_table(tokenizer) -> dict[int, bytes]:
    """Build token_id → bytes mapping from vocabulary (cached once per tokenizer)."""
    table = {}
    vocab_size = getattr(tokenizer, 'vocab_size', None) or len(tokenizer)
    for tok_id in range(vocab_size):
        try:
            text = tokenizer.decode([tok_id], skip_special_tokens=False)
            table[tok_id] = text.encode("utf-8", errors="surrogatepass")
        except Exception:
            table[tok_id] = b""
    return table


def token_id_to_bytes(tokenizer, tok_id: int, _table: dict = None) -> bytes:
    if _table is not None:
        return _table.get(tok_id, b"")
    try:
        text = tokenizer.decode([tok_id], skip_special_tokens=False)
        return text.encode("utf-8", errors="surrogatepass")
    except Exception:
        return b""


def first_byte_marginal(logits: torch.Tensor, tokenizer, top_vocab: int = 4096,
                        K: int = 16, _byte_table: dict = None,
                        ) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute first-byte marginal from teacher token logits.

    Returns (top_bytes, top_probs, tail_prob, coverage) where coverage
    is the fraction of token probability mass captured by top_vocab.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    q = torch.zeros(256, device=probs.device, dtype=torch.float32)

    top_ids = torch.topk(probs, min(top_vocab, probs.shape[-1])).indices
    for tok_id in top_ids.tolist():
        bs = token_id_to_bytes(tokenizer, tok_id, _byte_table)
        if bs:
            q[bs[0]] += probs[tok_id].item()

    coverage = q.sum().item()
    if coverage > 0:
        uncovered = max(0.0, 1.0 - coverage)
        q = q + uncovered / 256.0
        q = q / q.sum()

    top_probs, top_bytes = torch.topk(q, min(K, 256))
    tail = 1.0 - top_probs.sum().item()

    return (
        top_bytes.cpu().numpy().astype(np.uint8),
        top_probs.cpu().numpy().astype(np.float16),
        max(0.0, tail),
        coverage,
    )


def multi_byte_marginal(logits: torch.Tensor, tokenizer, n_positions: int = 4,
                        top_vocab: int = 4096, K: int = 16,
                        _byte_table: dict = None,
                        ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Compute per-byte-position marginals from teacher token logits.

    Extracts teacher signal for ALL byte positions from a SINGLE forward pass
    by decomposing the token probability distribution. Each token maps to a
    byte sequence; accumulating probability mass per byte position gives
    marginals for positions 0 through n_positions-1.

    Coverage at position i = fraction of token probability mass on tokens
    with >= (i+1) bytes. For Qwen3 tokenizer: ~100%, 99.9%, 97%, 80%
    for positions 0-3.

    Returns list of (top_bytes, top_probs, tail_prob, coverage) tuples,
    one per byte position.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    top_ids = torch.topk(probs, min(top_vocab, probs.shape[-1])).indices

    q_per_pos = [torch.zeros(256, dtype=torch.float32) for _ in range(n_positions)]

    for tok_id in top_ids.tolist():
        bs = token_id_to_bytes(tokenizer, tok_id, _byte_table)
        if not bs:
            continue
        p = probs[tok_id].item()
        for pos in range(min(len(bs), n_positions)):
            q_per_pos[pos][bs[pos]] += p

    results = []
    for pos in range(n_positions):
        q = q_per_pos[pos]
        coverage = q.sum().item()
        if coverage > 0:
            uncovered = max(0.0, 1.0 - coverage)
            q = q + uncovered / 256.0
            q = q / q.sum()
        else:
            q = torch.ones(256, dtype=torch.float32) / 256.0

        top_probs_val, top_bytes_val = torch.topk(q, min(K, 256))
        tail = 1.0 - top_probs_val.sum().item()
        results.append((
            top_bytes_val.cpu().numpy().astype(np.uint8),
            top_probs_val.cpu().numpy().astype(np.float16),
            max(0.0, tail),
            coverage,
        ))

    return results


def compute_token_byte_spans(tokenizer, input_ids: list[int],
                             _byte_table: dict = None,
                             ) -> list[tuple[int, int]]:
    spans = []
    byte_offset = 0
    for tok_id in input_ids:
        tok_bytes = token_id_to_bytes(tokenizer, tok_id, _byte_table)
        n = len(tok_bytes)
        if n > 0:
            spans.append((byte_offset, byte_offset + n))
            byte_offset += n
        else:
            spans.append((byte_offset, byte_offset))
    return spans


def validate_token_byte_alignment(seq_bytes: bytes, token_spans: list[tuple[int, int]],
                                  input_ids: list[int], tokenizer,
                                  _byte_table: dict = None) -> list[bool]:
    """Validate each token's byte span matches the original shard bytes."""
    valid = []
    for tok_idx, (bs, be) in enumerate(token_spans):
        if be <= bs or be > len(seq_bytes):
            valid.append(False)
            continue
        expected = token_id_to_bytes(tokenizer, input_ids[tok_idx], _byte_table)
        actual = seq_bytes[bs:be]
        valid.append(actual == expected)
    return valid


def _stable_seed(base_seed: int, shard_id: int, seq_offset: int) -> int:
    import hashlib
    h = hashlib.blake2b(
        f"{base_seed}:{shard_id}:{seq_offset}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def select_uniform_kl_patches(n_patches: int, shard_id: int, seq_offset: int,
                               frac: float = 0.25, seed: int = 491) -> list[int]:
    candidates = np.arange(1, n_patches, dtype=np.uint16)
    k = max(1, round(frac * len(candidates)))
    rng = np.random.default_rng(_stable_seed(seed, shard_id, seq_offset))
    return sorted(rng.choice(candidates, size=k, replace=False).tolist())


def select_kl_patches(student_logits: torch.Tensor, byte_ids: torch.Tensor,
                      P: int = 4, nll_floor: float = 3.5,
                      control_frac: float = 0.15) -> list[int]:
    B, Nm1, Pp, V = student_logits.shape
    assert B == 1
    N = Nm1 + 1
    targets = byte_ids.reshape(1, N, P)[:, 1:]  # (1, N-1, P)

    nlls = []
    for patch_idx in range(Nm1):
        byte_0 = targets[0, patch_idx, 0].item()
        logit_0 = student_logits[0, patch_idx, 0]
        nll = -F.log_softmax(logit_0, dim=-1)[byte_0].item()
        if not math.isfinite(nll):
            nll = float("inf")
        nlls.append(nll)

    finite_nlls = [v for v in nlls if math.isfinite(v)]
    if finite_nlls:
        sorted_finite = sorted(finite_nlls)
        p90_idx = int(len(sorted_finite) * 0.90)
        p90_nll = sorted_finite[min(p90_idx, len(sorted_finite) - 1)]
        threshold = max(nll_floor, p90_nll)
    else:
        threshold = nll_floor

    selected = []
    for patch_idx in range(Nm1):
        if not math.isfinite(nlls[patch_idx]) or nlls[patch_idx] > threshold:
            selected.append(patch_idx + 1)
        elif torch.rand(1).item() < control_frac:
            selected.append(patch_idx + 1)

    return selected


@torch.no_grad()
def build_cache_for_shard(
    teacher,
    tokenizer,
    student_model,
    shard_path: Path,
    shard_id: int,
    seq_len: int = 4096,
    patch_size: int = 4,
    nll_threshold: float = 3.5,
    kl_top_k: int = 16,
    device: torch.device = torch.device("cuda"),
    byte_table: dict = None,
    selection_policy: str = "nll",
    sample_frac: float = 0.25,
    sample_seed: int = 491,
    skip_align: bool = False,
    byte_positions: str = "first",
) -> tuple[list[AlignRecord], list[ByteKLRecord]]:
    teacher.eval()
    if student_model is not None:
        student_model.eval()

    if byte_table is None:
        byte_table = build_token_byte_table(tokenizer)

    shard_data = np.fromfile(shard_path, dtype=np.uint8)
    n_seqs = len(shard_data) // seq_len
    align_records = []
    kl_records = []
    n_skipped_alignment = 0

    for seq_idx in range(n_seqs):
        offset = seq_idx * seq_len
        seq_bytes = shard_data[offset:offset + seq_len]
        byte_ids = torch.tensor(seq_bytes, dtype=torch.long, device=device).unsqueeze(0)

        seq_clean = bytes(b if b != 0xFF else 0x0A for b in seq_bytes)
        text = seq_clean.decode("utf-8", errors="replace")
        teacher_inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=min(tokenizer.model_max_length or 2048, 8192),
        ).to(device)

        input_ids = teacher_inputs.input_ids[0].tolist()

        if not skip_align:
            token_spans = compute_token_byte_spans(tokenizer, input_ids, byte_table)
            span_valid = validate_token_byte_alignment(
                bytes(seq_bytes), token_spans, input_ids, tokenizer, byte_table)

            for tok_idx, (bs, be) in enumerate(token_spans):
                if be <= bs or be > seq_len:
                    continue
                if not span_valid[tok_idx]:
                    n_skipped_alignment += 1
                    continue
                align_records.append(AlignRecord(
                    shard_id=shard_id,
                    seq_offset=offset,
                    byte_start=bs,
                    byte_len=be - bs,
                    token_id=input_ids[tok_idx],
                ))

        n_patches = seq_len // patch_size
        if selection_policy == "uniform":
            selected_patches = select_uniform_kl_patches(
                n_patches, shard_id, offset, frac=sample_frac, seed=sample_seed)
        else:
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(amp_device, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                student_out = student_model(byte_ids)
            selected_patches = select_kl_patches(
                student_out["logits"], byte_ids, patch_size, nll_threshold,
            )

        for patch_idx in selected_patches:
            t = patch_idx * patch_size
            prefix_bytes = seq_bytes[:t]
            prefix_clean = bytes(b if b != 0xFF else 0x0A for b in prefix_bytes)
            prefix_text = prefix_clean.decode("utf-8", errors="replace")

            prefix_ids = tokenizer(
                prefix_text, return_tensors="pt", truncation=True,
                max_length=min(tokenizer.model_max_length or 2048, 8192),
            ).input_ids.to(device)

            if prefix_ids.shape[1] == 0:
                continue

            t_logits = teacher(prefix_ids).logits[0, -1]

            if byte_positions == "all":
                _min_cov = [0.0, 0.50, 0.35, 0.25]
                marginals = multi_byte_marginal(
                    t_logits, tokenizer, n_positions=patch_size,
                    K=kl_top_k, _byte_table=byte_table)
                for bpos, (top_b, top_p, tail, cov) in enumerate(marginals):
                    min_cov = _min_cov[bpos] if bpos < len(_min_cov) else 0.25
                    if cov < min_cov:
                        continue
                    q = torch.zeros(256, dtype=torch.float32)
                    q[top_b] = torch.from_numpy(top_p.astype(np.float32))
                    q_sum = q.sum()
                    if q_sum > 0:
                        q = q / q_sum
                    ent = -(q * (q + 1e-10).log()).sum().item() / math.log(2)
                    kl_records.append(ByteKLRecord(
                        shard_id=shard_id,
                        seq_offset=offset,
                        patch_idx=patch_idx,
                        top_bytes=top_b,
                        top_probs=top_p,
                        tail_prob=tail,
                        entropy=ent,
                        byte_pos=bpos,
                    ))
            else:
                top_b, top_p, tail, coverage = first_byte_marginal(
                    t_logits, tokenizer, K=kl_top_k, _byte_table=byte_table)
                q = torch.zeros(256, dtype=torch.float32)
                q[top_b] = torch.from_numpy(top_p.astype(np.float32))
                q_sum = q.sum()
                if q_sum > 0:
                    q = q / q_sum
                ent = -(q * (q + 1e-10).log()).sum().item() / math.log(2)
                kl_records.append(ByteKLRecord(
                    shard_id=shard_id,
                    seq_offset=offset,
                    patch_idx=patch_idx,
                    top_bytes=top_b,
                    top_probs=top_p,
                    tail_prob=tail,
                    entropy=ent,
                ))

        if (seq_idx + 1) % 100 == 0:
            print(f"  shard {shard_id} seq {seq_idx+1}/{n_seqs}: "
                  f"{len(align_records)} align, {len(kl_records)} kl, "
                  f"{n_skipped_alignment} skipped (byte mismatch)")

    return align_records, kl_records


class StreamingCacheWriter:
    """Writes cache records incrementally to disk to avoid OOM on large datasets."""

    def __init__(self, output_dir: str, kl_top_k: int = 16,
                 cache_format_version: int = 1):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.kl_top_k = kl_top_k
        self.format_version = cache_format_version
        self.n_align = 0
        self.n_kl = 0

        self._align_path = os.path.join(output_dir, "align_records.bin")
        self._kl_path = os.path.join(output_dir, "kl_records.bin")

        self._align_f = open(self._align_path, "wb")
        self._align_f.write(struct.pack("<I", 0))  # placeholder

        self._kl_f = open(self._kl_path, "wb")
        self._kl_f.write(struct.pack("<II", 0, kl_top_k))  # placeholder

    def write_shard(self, align_records: list[AlignRecord],
                    kl_records: list[ByteKLRecord]):
        for r in align_records:
            self._align_f.write(struct.pack("<IqHHI", r.shard_id, r.seq_offset,
                                            r.byte_start, r.byte_len, r.token_id))
        self.n_align += len(align_records)

        for r in kl_records:
            self._kl_f.write(struct.pack("<IqH", r.shard_id, r.seq_offset, r.patch_idx))
            if self.format_version >= 2:
                self._kl_f.write(struct.pack("<B", r.byte_pos))
            self._kl_f.write(r.top_bytes.tobytes())
            self._kl_f.write(r.top_probs.tobytes())
            self._kl_f.write(struct.pack("<ee", r.tail_prob, r.entropy))
        self.n_kl += len(kl_records)

    def finalize(self, embedding_table: Optional[torch.Tensor] = None,
                 shard_range: Optional[tuple[int, int]] = None,
                 extra_manifest: Optional[dict] = None):
        self._align_f.seek(0)
        self._align_f.write(struct.pack("<I", self.n_align))
        self._align_f.flush()
        os.fsync(self._align_f.fileno())
        self._align_f.close()

        self._kl_f.seek(0)
        self._kl_f.write(struct.pack("<II", self.n_kl, self.kl_top_k))
        self._kl_f.flush()
        os.fsync(self._kl_f.fileno())
        self._kl_f.close()

        emb_path = os.path.join(self.output_dir, "teacher_embeddings.pt")
        if embedding_table is not None:
            torch.save(embedding_table.half(), emb_path)
        elif os.path.exists(emb_path):
            os.remove(emb_path)

        manifest = {
            "n_align": self.n_align,
            "n_kl": self.n_kl,
            "kl_top_k": self.kl_top_k,
            "has_embeddings": embedding_table is not None,
            "cache_format_version": self.format_version,
        }
        if shard_range is not None:
            manifest["shard_range"] = list(shard_range)
        if extra_manifest:
            manifest.update(extra_manifest)
        with open(os.path.join(self.output_dir, "cache_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Finalized cache: {self.n_align} align, {self.n_kl} kl records "
              f"(format v{self.format_version})")


class MappedByteKLCache:
    """Memory-mapped KL cache with indexed lookups by (shard_id, seq_offset).

    Unlike load_cache() which materializes all records, this uses mmap and
    builds a dict index for O(1) sequence lookups. Designed for Option C
    training where shuffled batches need random-access into the cache.
    """

    def __init__(self, cache_dir: str):
        import mmap as mmap_mod

        with open(os.path.join(cache_dir, "cache_manifest.json")) as f:
            self.manifest = json.load(f)

        self.format_version = self.manifest.get("cache_format_version", 1)

        kl_path = os.path.join(cache_dir, "kl_records.bin")
        self._kl_file = open(kl_path, "rb")
        self._kl_mm = mmap_mod.mmap(
            self._kl_file.fileno(), 0, access=mmap_mod.ACCESS_READ)

        hdr = struct.unpack("<II", self._kl_mm[:8])
        self.n_records = hdr[0]
        self.kl_top_k = hdr[1]
        self._header_size = 8
        # v2 adds 1 byte for byte_pos after patch_idx
        self._byte_pos_size = 1 if self.format_version >= 2 else 0
        self._rec_size = 14 + self._byte_pos_size + self.kl_top_k + self.kl_top_k * 2 + 4

        self._index: dict[tuple[int, int], tuple[int, int]] = {}
        self._build_index()

    def _build_index(self):
        current_key = None
        run_start = 0
        run_count = 0

        for i in range(self.n_records):
            offset = self._header_size + i * self._rec_size
            sid, soff, _ = struct.unpack("<IqH", self._kl_mm[offset:offset + 14])
            key = (sid, soff)

            if key == current_key:
                run_count += 1
            else:
                if current_key is not None:
                    self._index[current_key] = (run_start, run_count)
                current_key = key
                run_start = i
                run_count = 1

        if current_key is not None:
            self._index[current_key] = (run_start, run_count)

    def get_records(self, shard_id: int, seq_offset: int) -> list[ByteKLRecord]:
        key = (shard_id, seq_offset)
        if key not in self._index:
            return []

        start_idx, count = self._index[key]
        records = []
        K = self.kl_top_k

        for i in range(start_idx, start_idx + count):
            offset = self._header_size + i * self._rec_size
            sid, soff, pidx = struct.unpack(
                "<IqH", self._kl_mm[offset:offset + 14])
            pos = offset + 14
            if self.format_version >= 2:
                byte_pos = struct.unpack("<B", self._kl_mm[pos:pos + 1])[0]
                pos += 1
            else:
                byte_pos = 0
            top_b = np.frombuffer(
                self._kl_mm[pos:pos + K], dtype=np.uint8).copy()
            pos += K
            top_p = np.frombuffer(
                self._kl_mm[pos:pos + K * 2], dtype=np.float16).copy()
            pos += K * 2
            tail, ent = struct.unpack("<ee", self._kl_mm[pos:pos + 4])

            rec = ByteKLRecord(
                sid, soff, pidx, top_b, top_p, float(tail), float(ent),
                byte_pos=byte_pos)
            if _kl_record_is_valid(rec):
                records.append(rec)

        return records

    def has_sequence(self, shard_id: int, seq_offset: int) -> bool:
        return (shard_id, seq_offset) in self._index

    def close(self):
        self._kl_mm.close()
        self._kl_file.close()

    def __len__(self):
        return len(self._index)


def save_cache(output_dir: str, align_records: list[AlignRecord],
               kl_records: list[ByteKLRecord], embedding_table: Optional[torch.Tensor] = None,
               shard_range: Optional[tuple[int, int]] = None,
               cache_format_version: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    align_path = os.path.join(output_dir, "align_records.bin")
    with open(align_path, "wb") as f:
        f.write(struct.pack("<I", len(align_records)))
        for r in align_records:
            f.write(struct.pack("<IqHHI", r.shard_id, r.seq_offset,
                                r.byte_start, r.byte_len, r.token_id))
        f.flush()
        os.fsync(f.fileno())

    kl_path = os.path.join(output_dir, "kl_records.bin")
    K = kl_records[0].top_bytes.shape[0] if kl_records else 16
    with open(kl_path, "wb") as f:
        f.write(struct.pack("<II", len(kl_records), K))
        for r in kl_records:
            f.write(struct.pack("<IqH", r.shard_id, r.seq_offset, r.patch_idx))
            if cache_format_version >= 2:
                f.write(struct.pack("<B", r.byte_pos))
            f.write(r.top_bytes.tobytes())
            f.write(r.top_probs.tobytes())
            f.write(struct.pack("<ee", r.tail_prob, r.entropy))
        f.flush()
        os.fsync(f.fileno())

    emb_path = os.path.join(output_dir, "teacher_embeddings.pt")
    if embedding_table is not None:
        torch.save(embedding_table.half(), emb_path)
    elif os.path.exists(emb_path):
        os.remove(emb_path)

    manifest = {
        "n_align": len(align_records),
        "n_kl": len(kl_records),
        "kl_top_k": K,
        "has_embeddings": embedding_table is not None,
        "cache_format_version": cache_format_version,
    }
    if shard_range is not None:
        manifest["shard_range"] = list(shard_range)
    with open(os.path.join(output_dir, "cache_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved cache: {len(align_records)} align, {len(kl_records)} kl records "
          f"(format v{cache_format_version})")


def _kl_record_is_valid(rec: ByteKLRecord) -> bool:
    """Check if a ByteKLRecord has well-formed probability values."""
    if not np.all(np.isfinite(rec.top_probs)):
        return False
    if np.any(rec.top_probs < 0.0) or np.any(rec.top_probs > 1.0):
        return False
    tail = float(rec.tail_prob)
    if not math.isfinite(tail) or tail < 0.0:
        return False
    total = float(rec.top_probs.astype(np.float64).sum()) + tail
    if total < 0.9 or total > 1.1:
        return False
    return True


def load_cache(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "cache_manifest.json")) as f:
        manifest = json.load(f)

    align_path = os.path.join(cache_dir, "align_records.bin")
    align_records = []
    file_size = os.path.getsize(align_path)
    with open(align_path, "rb") as f:
        hdr_data = f.read(4)
        if len(hdr_data) < 4:
            n_align = 0
        else:
            n_align = struct.unpack("<I", hdr_data)[0]
            align_hdr = 4
            align_rec_size = 20  # IqHHI
            if file_size < align_hdr + n_align * align_rec_size:
                n_align = max(0, (file_size - align_hdr) // align_rec_size)
        for _ in range(n_align):
            sid, soff, bs, bl, tid = struct.unpack("<IqHHI", f.read(20))
            align_records.append(AlignRecord(sid, soff, bs, bl, tid))

    kl_path = os.path.join(cache_dir, "kl_records.bin")
    kl_records = []
    file_size = os.path.getsize(kl_path)
    with open(kl_path, "rb") as f:
        hdr_data = f.read(8)
        if len(hdr_data) < 8:
            n, K = 0, 16
        else:
            n, K = struct.unpack("<II", hdr_data)
            kl_hdr = 8
            fmt_version = manifest.get("cache_format_version", 1)
            byte_pos_size = 1 if fmt_version >= 2 else 0
            kl_rec_size = 14 + byte_pos_size + K + K * 2 + 4
            if file_size < kl_hdr + n * kl_rec_size:
                n = max(0, (file_size - kl_hdr) // kl_rec_size)
        for _ in range(n):
            sid, soff, pidx = struct.unpack("<IqH", f.read(14))
            if fmt_version >= 2:
                byte_pos = struct.unpack("<B", f.read(1))[0]
            else:
                byte_pos = 0
            top_b = np.frombuffer(f.read(K), dtype=np.uint8).copy()
            top_p = np.frombuffer(f.read(K * 2), dtype=np.float16).copy()
            tail, ent = struct.unpack("<ee", f.read(4))
            rec = ByteKLRecord(sid, soff, pidx, top_b, top_p, tail, ent,
                               byte_pos=byte_pos)
            if _kl_record_is_valid(rec):
                kl_records.append(rec)

    embedding_table = None
    if manifest.get("has_embeddings", False):
        emb_path = os.path.join(cache_dir, "teacher_embeddings.pt")
        if os.path.exists(emb_path):
            embedding_table = torch.load(emb_path, map_location="cpu", weights_only=True)

    return {
        "align_records": align_records,
        "kl_records": kl_records,
        "embedding_table": embedding_table,
        "manifest": manifest,
    }


def main():
    parser = argparse.ArgumentParser(description="Build Eklavya E1 teacher cache")
    parser.add_argument("--teacher", required=True,
                        help="HuggingFace model ID for the teacher")
    parser.add_argument("--data-dir", default="data/shards_bytes_full")
    parser.add_argument("--output-dir", default="eklavya_cache")
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--max-shards", type=int, default=50)
    parser.add_argument("--nll-threshold", type=float, default=3.5)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--selection-policy", choices=["nll", "uniform"],
                        default="nll")
    parser.add_argument("--sample-frac", type=float, default=0.25,
                        help="Fraction of patches to sample (uniform policy)")
    parser.add_argument("--sample-seed", type=int, default=491,
                        help="Seed for reproducible uniform sampling")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip alignment record building (KL-only cache)")
    parser.add_argument("--byte-positions", choices=["first", "all"],
                        default="first",
                        help="Which byte positions to cache: first (v1) or all (v2)")
    args = parser.parse_args()

    if args.selection_policy == "nll" and args.student_checkpoint is None:
        parser.error("--student-checkpoint required for NLL selection policy")

    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading teacher: {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16,
    ).to(device)
    teacher.eval()

    embedding_table = None
    if not args.no_align:
        embedding_table = teacher.get_input_embeddings().weight.detach().clone()
        print(f"Teacher embedding table: {embedding_table.shape}")

    student = None
    patch_size = 4
    if args.student_checkpoint is not None:
        from s0_architecture import SutraS0
        print(f"Loading student: {args.student_checkpoint}")
        ckpt = torch.load(args.student_checkpoint, map_location="cpu",
                          weights_only=False)
        student = SutraS0(ckpt["model_cfg"]).to(device)
        student.load_state_dict(ckpt["model"])
        student.eval()
        patch_size = ckpt["model_cfg"].patch_size
    else:
        print("No student checkpoint — using uniform selection (no NLL)")

    byte_table = build_token_byte_table(tokenizer)
    print(f"Built token→byte table: {len(byte_table)} entries")
    print(f"Selection policy: {args.selection_policy}")

    shards = sorted(Path(args.data_dir).glob("*.bin"))
    n_shards = min(len(shards), args.max_shards)
    print(f"Processing {n_shards} shards from {args.data_dir}")

    cache_fmt = 2 if args.byte_positions == "all" else 1
    writer = StreamingCacheWriter(args.output_dir, cache_format_version=cache_fmt)
    t0 = time.time()

    for i in range(n_shards):
        print(f"\nShard {i}/{n_shards}: {shards[i].name}")
        align, kl = build_cache_for_shard(
            teacher, tokenizer, student, shards[i], i,
            seq_len=args.seq_len,
            patch_size=patch_size,
            nll_threshold=args.nll_threshold,
            device=device,
            byte_table=byte_table,
            selection_policy=args.selection_policy,
            sample_frac=args.sample_frac,
            sample_seed=args.sample_seed,
            skip_align=args.no_align,
            byte_positions=args.byte_positions,
        )
        writer.write_shard(align, kl)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")

    manifest_extra = {
        "byte_positions": args.byte_positions,
    }
    if args.selection_policy == "uniform":
        manifest_extra.update({
            "selection_policy": "uniform",
            "sample_frac": args.sample_frac,
            "sample_seed": args.sample_seed,
        })

    writer.finalize(
        embedding_table, shard_range=(0, n_shards),
        extra_manifest=manifest_extra,
    )


if __name__ == "__main__":
    main()
