"""Eklavya E2 — Multi-teacher cache builder and binary record formats.

Extends E1 single-teacher cache with:
  - Per-teacher subdirectories with independent KL/align records
  - Shared position manifest (student-side gap selection)
  - Teacher registry with family/role/dimension metadata
  - Aggregate route + purified target records

Binary record formats use struct.pack for compact sequential streaming.
No parquet dependency — E1 binary style preserved.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from dataclasses import dataclass, asdict
from enum import IntEnum, IntFlag

import numpy as np
import torch



# ---------------------------------------------------------------------------
# Teacher registry
# ---------------------------------------------------------------------------

class TeacherFamily(IntEnum):
    DECODER = 0
    HYBRID = 1
    SSM = 2
    ENCODER = 3
    EMBEDDING = 4


class TeacherRole(IntEnum):
    ANCHOR = 0
    DIVERSITY = 1
    CONTROL = 2
    SEMANTIC = 3


class SelectionReason(IntFlag):
    HIGH_NLL = 1
    HIGH_ENTROPY = 2
    DISAGREEMENT = 4
    CONTROL = 8


@dataclass
class TeacherSpec:
    teacher_id: int
    name: str
    family: TeacherFamily
    role: TeacherRole
    hidden_dim: int
    vocab_size: int
    has_kl: bool
    has_align: bool
    has_semantic: bool
    prior: float
    vram_gb: float
    per_teacher_grad_cap: float = 0.10
    hf_name: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["family"] = self.family.name
        d["role"] = self.role.name
        if not d["hf_name"]:
            del d["hf_name"]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TeacherSpec:
        d = dict(d)
        d["family"] = TeacherFamily[d["family"]]
        d["role"] = TeacherRole[d["role"]]
        d.setdefault("hf_name", "")
        return cls(**d)


TEACHER_REGISTRY: list[TeacherSpec] = [
    TeacherSpec(
        teacher_id=0,
        name="t0_anchor_decoder",
        family=TeacherFamily.DECODER,
        role=TeacherRole.ANCHOR,
        hidden_dim=2048,
        vocab_size=151936,
        has_kl=True,
        has_align=True,
        has_semantic=False,
        prior=0.40,
        vram_gb=3.4,
    ),
    TeacherSpec(
        teacher_id=1,
        name="t1_diversity_hybrid",
        family=TeacherFamily.HYBRID,
        role=TeacherRole.DIVERSITY,
        hidden_dim=2048,
        vocab_size=65536,
        has_kl=True,
        has_align=True,
        has_semantic=False,
        prior=0.20,
        vram_gb=2.4,
    ),
    TeacherSpec(
        teacher_id=2,
        name="t2_control_decoder",
        family=TeacherFamily.DECODER,
        role=TeacherRole.CONTROL,
        hidden_dim=1024,
        vocab_size=151936,
        has_kl=True,
        has_align=True,
        has_semantic=False,
        prior=0.15,
        vram_gb=1.2,
    ),
    TeacherSpec(
        teacher_id=3,
        name="t3_semantic_embedding",
        family=TeacherFamily.EMBEDDING,
        role=TeacherRole.SEMANTIC,
        hidden_dim=768,
        vocab_size=256000,
        has_kl=False,
        has_align=False,
        has_semantic=True,
        prior=0.10,
        vram_gb=0.6,
        per_teacher_grad_cap=0.05,
    ),
    TeacherSpec(
        teacher_id=4,
        name="t4_diversity_ssm",
        family=TeacherFamily.SSM,
        role=TeacherRole.DIVERSITY,
        hidden_dim=1536,
        vocab_size=50277,
        has_kl=True,
        has_align=False,
        has_semantic=False,
        prior=0.15,
        vram_gb=1.6,
        per_teacher_grad_cap=0.05,
    ),
]


def get_teacher_by_name(name: str) -> TeacherSpec:
    for t in TEACHER_REGISTRY:
        if t.name == name:
            return t
    raise ValueError(f"Unknown teacher: {name}")


def get_teacher_by_id(tid: int) -> TeacherSpec:
    for t in TEACHER_REGISTRY:
        if t.teacher_id == tid:
            return t
    raise ValueError(f"Unknown teacher_id: {tid}")


def load_private_teacher_config(
    config_path: str,
    registry: list[TeacherSpec] | None = None,
) -> list[TeacherSpec]:
    """Merge private teacher config (HF names) into the public registry.

    The private config is a JSON file mapping alias → hf_name:
        {"t0_anchor_decoder": "Org/Model-Name", ...}

    Only needed at cache-build time when loading real models.
    """
    if registry is None:
        registry = TEACHER_REGISTRY
    with open(config_path) as f:
        private = json.load(f)
    result = []
    for spec in registry:
        hf = private.get(spec.name, spec.hf_name)
        updated = TeacherSpec(**{**asdict(spec), "hf_name": hf,
                                 "family": spec.family, "role": spec.role})
        result.append(updated)
    return result


# ---------------------------------------------------------------------------
# Binary record formats
# ---------------------------------------------------------------------------

@dataclass
class PositionRecord:
    """Shared position manifest entry — one per selected byte position."""
    position_id: int
    shard_id: int
    seq_offset: int
    patch_idx: int
    gold_byte: int
    student_nll: float
    student_entropy: float
    reason_mask: int

    _STRUCT = struct.Struct("<IIqHBffB")
    SIZE = _STRUCT.size

    def pack(self) -> bytes:
        return self._STRUCT.pack(
            self.position_id, self.shard_id, self.seq_offset,
            self.patch_idx, self.gold_byte,
            self.student_nll, self.student_entropy,
            self.reason_mask,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> PositionRecord:
        vals = cls._STRUCT.unpack(buf)
        return cls(*vals)


@dataclass
class E2KLRecord:
    """Per-teacher KL record at a selected position."""
    position_id: int
    patch_idx: int
    tail_prob: float
    entropy: float
    logp_gold: float
    top_bytes: np.ndarray    # uint8[K]
    top_probs: np.ndarray    # float16[K]

    _HEADER = struct.Struct("<IHeee")
    HEADER_SIZE = _HEADER.size

    def pack(self, K: int = 16) -> bytes:
        if len(self.top_bytes) < K:
            raise ValueError(f"top_bytes length {len(self.top_bytes)} < K={K}")
        if len(self.top_probs) < K:
            raise ValueError(f"top_probs length {len(self.top_probs)} < K={K}")
        if self.top_bytes.dtype != np.uint8:
            raise ValueError(f"top_bytes dtype must be uint8, got {self.top_bytes.dtype}")
        if self.top_probs.dtype != np.float16:
            raise ValueError(f"top_probs dtype must be float16, got {self.top_probs.dtype}")
        hdr = self._HEADER.pack(
            self.position_id, self.patch_idx,
            self.tail_prob, self.entropy, self.logp_gold,
        )
        return hdr + self.top_bytes[:K].tobytes() + self.top_probs[:K].tobytes()

    @classmethod
    def unpack(cls, buf: bytes, K: int = 16) -> E2KLRecord:
        hdr_vals = cls._HEADER.unpack(buf[:cls.HEADER_SIZE])
        offset = cls.HEADER_SIZE
        top_bytes = np.frombuffer(buf[offset:offset + K], dtype=np.uint8).copy()
        offset += K
        top_probs = np.frombuffer(buf[offset:offset + K * 2], dtype=np.float16).copy()
        return cls(
            position_id=hdr_vals[0],
            patch_idx=hdr_vals[1],
            tail_prob=hdr_vals[2],
            entropy=hdr_vals[3],
            logp_gold=hdr_vals[4],
            top_bytes=top_bytes,
            top_probs=top_probs,
        )

    @classmethod
    def record_size(cls, K: int = 16) -> int:
        return cls.HEADER_SIZE + K + K * 2


@dataclass
class E2AlignRecord:
    """Per-teacher alignment record at a selected position."""
    position_id: int
    byte_start: int
    byte_len: int
    token_id: int
    align_quality: float

    _STRUCT = struct.Struct("<IHHIe")
    SIZE = _STRUCT.size

    def pack(self) -> bytes:
        return self._STRUCT.pack(
            self.position_id, self.byte_start, self.byte_len,
            self.token_id, self.align_quality,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> E2AlignRecord:
        vals = cls._STRUCT.unpack(buf)
        return cls(*vals)


@dataclass
class RouteRecord:
    """Aggregate routing decision for a position across teachers."""
    position_id: int
    n_teachers: int
    jsd: float
    route_entropy: float
    teacher_ids: np.ndarray   # uint8[n_teachers]
    weights: np.ndarray       # float16[n_teachers]

    def pack(self) -> bytes:
        if len(self.teacher_ids) != self.n_teachers:
            raise ValueError(f"teacher_ids length {len(self.teacher_ids)} != n_teachers={self.n_teachers}")
        if len(self.weights) != self.n_teachers:
            raise ValueError(f"weights length {len(self.weights)} != n_teachers={self.n_teachers}")
        if self.teacher_ids.dtype != np.uint8:
            raise ValueError(f"teacher_ids dtype must be uint8, got {self.teacher_ids.dtype}")
        if self.weights.dtype != np.float16:
            raise ValueError(f"weights dtype must be float16, got {self.weights.dtype}")
        hdr = struct.pack("<IBee",
                          self.position_id, self.n_teachers,
                          self.jsd, self.route_entropy)
        return hdr + self.teacher_ids.tobytes() + self.weights.tobytes()

    @classmethod
    def unpack(cls, buf: bytes) -> RouteRecord:
        hdr_size = struct.calcsize("<IBee")
        pos_id, n, jsd, rent = struct.unpack("<IBee", buf[:hdr_size])
        offset = hdr_size
        tids = np.frombuffer(buf[offset:offset + n], dtype=np.uint8).copy()
        offset += n
        weights = np.frombuffer(buf[offset:offset + n * 2], dtype=np.float16).copy()
        return cls(pos_id, n, jsd, rent, tids, weights)


@dataclass
class SparseByteDist:
    """Sparse byte distribution: top-K bytes + probs + tail."""
    top_bytes: np.ndarray    # uint8[K]
    top_probs: np.ndarray    # float32[K]
    tail_prob: float


# ---------------------------------------------------------------------------
# Cache I/O — Position manifest
# ---------------------------------------------------------------------------

def write_position_manifest(output_path: str, records: list[PositionRecord]):
    """Write position manifest as binary file with count header."""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for r in records:
            f.write(r.pack())


def read_position_manifest(path: str) -> list[PositionRecord]:
    """Read position manifest from binary file."""
    records = []
    with open(path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError(f"Short read on manifest header in {path}")
        n = struct.unpack("<I", hdr)[0]
        for i in range(n):
            buf = f.read(PositionRecord.SIZE)
            if len(buf) < PositionRecord.SIZE:
                raise ValueError(
                    f"Short read at record {i}/{n} in {path}: "
                    f"got {len(buf)}, expected {PositionRecord.SIZE}")
            records.append(PositionRecord.unpack(buf))
    return records


# ---------------------------------------------------------------------------
# Cache I/O — Per-teacher KL records
# ---------------------------------------------------------------------------

def write_teacher_kl_records(output_path: str, records: list[E2KLRecord],
                              K: int = 16):
    """Write per-teacher KL records as binary file."""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<II", len(records), K))
        for r in records:
            f.write(r.pack(K))


def read_teacher_kl_records(path: str) -> tuple[list[E2KLRecord], int]:
    """Read per-teacher KL records. Returns (records, K)."""
    records = []
    with open(path, "rb") as f:
        hdr = f.read(8)
        if len(hdr) < 8:
            raise ValueError(f"Short read on KL header in {path}")
        n, K = struct.unpack("<II", hdr)
        rec_size = E2KLRecord.record_size(K)
        for i in range(n):
            buf = f.read(rec_size)
            if len(buf) < rec_size:
                raise ValueError(
                    f"Short read at KL record {i}/{n} in {path}: "
                    f"got {len(buf)}, expected {rec_size}")
            records.append(E2KLRecord.unpack(buf, K))
    return records, K


# ---------------------------------------------------------------------------
# Cache I/O — Per-teacher align records
# ---------------------------------------------------------------------------

def write_teacher_align_records(output_path: str,
                                 records: list[E2AlignRecord]):
    """Write per-teacher align records as binary file."""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for r in records:
            f.write(r.pack())


def read_teacher_align_records(path: str) -> list[E2AlignRecord]:
    """Read per-teacher align records."""
    records = []
    with open(path, "rb") as f:
        hdr = f.read(4)
        if len(hdr) < 4:
            raise ValueError(f"Short read on align header in {path}")
        n = struct.unpack("<I", hdr)[0]
        for i in range(n):
            buf = f.read(E2AlignRecord.SIZE)
            if len(buf) < E2AlignRecord.SIZE:
                raise ValueError(
                    f"Short read at align record {i}/{n} in {path}: "
                    f"got {len(buf)}, expected {E2AlignRecord.SIZE}")
            records.append(E2AlignRecord.unpack(buf))
    return records


# ---------------------------------------------------------------------------
# Cache I/O — Route records
# ---------------------------------------------------------------------------

def write_route_records(output_path: str, records: list[RouteRecord]):
    """Write aggregate route records."""
    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", len(records)))
        for r in records:
            f.write(r.pack())


def read_route_records(path: str) -> list[RouteRecord]:
    """Read aggregate route records."""
    records = []
    hdr_size = struct.calcsize("<IBee")
    with open(path, "rb") as f:
        count_buf = f.read(4)
        if len(count_buf) < 4:
            raise ValueError(f"Short read on route count header in {path}")
        n = struct.unpack("<I", count_buf)[0]
        for i in range(n):
            hdr_buf = f.read(hdr_size)
            if len(hdr_buf) < hdr_size:
                raise ValueError(
                    f"Short read at route header {i}/{n} in {path}: "
                    f"got {len(hdr_buf)}, expected {hdr_size}")
            _, n_t, _, _ = struct.unpack("<IBee", hdr_buf)
            rest_size = n_t + n_t * 2
            rest = f.read(rest_size)
            if len(rest) < rest_size:
                raise ValueError(
                    f"Short read at route body {i}/{n} in {path}: "
                    f"got {len(rest)}, expected {rest_size}")
            records.append(RouteRecord.unpack(hdr_buf + rest))
    return records


# ---------------------------------------------------------------------------
# Memory-mapped record access (streaming alternative to eager loading)
# ---------------------------------------------------------------------------

class MappedPositionRecords:
    """Memory-mapped position manifest — O(1) random access by index."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        hdr = self._fh.read(4)
        self._n = struct.unpack("<I", hdr)[0]
        self._data_offset = 4
        self._rec_size = PositionRecord.SIZE
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> PositionRecord:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")
        offset = self._data_offset + idx * self._rec_size
        buf = self._mm[offset:offset + self._rec_size]
        return PositionRecord.unpack(buf)

    def close(self):
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def to_list(self) -> list[PositionRecord]:
        return [self[i] for i in range(self._n)]

    def build_loc_index(self) -> dict[tuple[int, int], list[int]]:
        """Build (shard_id, seq_offset) → [record_index] from mmap headers."""
        idx: dict[tuple[int, int], list[int]] = {}
        for i in range(self._n):
            off = self._data_offset + i * self._rec_size
            sid = struct.unpack_from("<I", self._mm, off + 4)[0]
            soff = struct.unpack_from("<q", self._mm, off + 8)[0]
            idx.setdefault((sid, soff), []).append(i)
        return idx


class MappedKLRecords:
    """Memory-mapped KL records — O(1) random access by index."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        hdr = self._fh.read(8)
        self._n, self._K = struct.unpack("<II", hdr)
        self._data_offset = 8
        self._rec_size = E2KLRecord.record_size(self._K)
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def K(self) -> int:
        return self._K

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> E2KLRecord:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")
        offset = self._data_offset + idx * self._rec_size
        buf = self._mm[offset:offset + self._rec_size]
        return E2KLRecord.unpack(buf, self._K)

    def close(self):
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def to_list(self) -> list[E2KLRecord]:
        return [self[i] for i in range(self._n)]

    def build_pid_index(self) -> dict[int, int]:
        """Build position_id → record_index from mmap headers."""
        idx = {}
        for i in range(self._n):
            off = self._data_offset + i * self._rec_size
            pid = struct.unpack_from("<I", self._mm, off)[0]
            idx[pid] = i
        return idx


class MappedAlignRecords:
    """Memory-mapped align records — O(1) random access by index."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        hdr = self._fh.read(4)
        self._n = struct.unpack("<I", hdr)[0]
        self._data_offset = 4
        self._rec_size = E2AlignRecord.SIZE
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> E2AlignRecord:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")
        offset = self._data_offset + idx * self._rec_size
        buf = self._mm[offset:offset + self._rec_size]
        return E2AlignRecord.unpack(buf)

    def close(self):
        self._mm.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def to_list(self) -> list[E2AlignRecord]:
        return [self[i] for i in range(self._n)]

    def build_pid_index(self) -> dict[int, int]:
        """Build position_id → record_index from mmap headers."""
        idx = {}
        for i in range(self._n):
            off = self._data_offset + i * self._rec_size
            pid = struct.unpack_from("<I", self._mm, off)[0]
            idx[pid] = i
        return idx


# ---------------------------------------------------------------------------
# Teacher registry I/O
# ---------------------------------------------------------------------------

def save_teacher_registry(output_path: str, teachers: list[TeacherSpec]):
    """Save teacher registry as JSON."""
    with open(output_path, "w") as f:
        json.dump([t.to_dict() for t in teachers], f, indent=2)


def load_teacher_registry(path: str) -> list[TeacherSpec]:
    """Load teacher registry from JSON."""
    with open(path) as f:
        return [TeacherSpec.from_dict(d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# Multi-teacher cache manifest
# ---------------------------------------------------------------------------

def save_e2_manifest(cache_dir: str, teachers: list[TeacherSpec],
                      n_positions: int, K: int = 16,
                      shard_range: tuple[int, int] = (0, 0),
                      provenance: dict | None = None):
    """Write E2 cache manifest.json."""
    manifest = {
        "version": "e2.0",
        "n_positions": n_positions,
        "kl_top_k": K,
        "shard_range": list(shard_range),
        "teachers": [t.name for t in teachers],
        "teacher_count": len(teachers),
    }
    if provenance:
        manifest["provenance"] = provenance
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    save_teacher_registry(os.path.join(cache_dir, "teacher_registry.json"),
                          teachers)


def load_e2_manifest(cache_dir: str) -> dict:
    """Load E2 cache manifest."""
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    manifest["teacher_specs"] = load_teacher_registry(
        os.path.join(cache_dir, "teacher_registry.json"))
    return manifest


# ---------------------------------------------------------------------------
# Multi-teacher cache loader
# ---------------------------------------------------------------------------

def load_e2_cache(cache_dir: str) -> dict:
    """Load full E2 cache: positions + per-teacher records + routes."""
    manifest = load_e2_manifest(cache_dir)

    positions = read_position_manifest(
        os.path.join(cache_dir, "positions.bin"))

    teacher_data = {}
    for spec in manifest["teacher_specs"]:
        tdir = os.path.join(cache_dir, "teachers", spec.name)
        if not os.path.isdir(tdir):
            continue
        td = {"spec": spec}

        kl_path = os.path.join(tdir, "kl_records.bin")
        if os.path.exists(kl_path):
            td["kl_records"], td["kl_K"] = read_teacher_kl_records(kl_path)
        else:
            td["kl_records"], td["kl_K"] = [], 16

        align_path = os.path.join(tdir, "align_records.bin")
        if os.path.exists(align_path):
            td["align_records"] = read_teacher_align_records(align_path)
        else:
            td["align_records"] = []

        emb_path = os.path.join(tdir, "teacher_embeddings.pt")
        if os.path.exists(emb_path):
            td["embedding_table"] = torch.load(emb_path, map_location="cpu",
                                                weights_only=True)
        else:
            td["embedding_table"] = None

        teacher_data[spec.name] = td

    routes = []
    route_path = os.path.join(cache_dir, "aggregate", "route_records.bin")
    if os.path.exists(route_path):
        routes = read_route_records(route_path)

    return {
        "manifest": manifest,
        "positions": positions,
        "teachers": teacher_data,
        "routes": routes,
    }


# ---------------------------------------------------------------------------
# Index memory estimation
# ---------------------------------------------------------------------------

# CPython dict overhead per int→int entry, empirically measured:
# sys.getsizeof(dict()) = 64, each resize costs ~56 bytes/slot.
_DICT_ENTRY_BYTES = 56
_DICT_BASE_BYTES = 64
_TUPLE2_ENTRY_BYTES = 120  # tuple key (shard_id, seq_offset) → list value


def estimate_index_memory(
    n_positions: int,
    n_teachers: int,
    n_kl_per_teacher: int | None = None,
    n_align_per_teacher: int | None = None,
    n_unique_locs: int | None = None,
) -> dict[str, int]:
    """Estimate RAM for E2CacheView's in-memory indices.

    Returns byte counts for each index component and total. Call before
    loading the cache to check if indices will fit in available RAM.

    Args:
        n_positions: Total positions in manifest.
        n_teachers: Number of teachers.
        n_kl_per_teacher: KL records per teacher (default: n_positions).
        n_align_per_teacher: Align records per teacher (default: n_positions).
        n_unique_locs: Unique (shard_id, seq_offset) combos (estimated from
            n_positions / 20 if not given — typical gap density).
    """
    if n_kl_per_teacher is None:
        n_kl_per_teacher = n_positions
    if n_align_per_teacher is None:
        n_align_per_teacher = n_positions
    if n_unique_locs is None:
        n_unique_locs = max(1, n_positions // 20)

    loc_index = (_DICT_BASE_BYTES
                 + n_unique_locs * _TUPLE2_ENTRY_BYTES)

    kl_indices = n_teachers * (
        _DICT_BASE_BYTES + n_kl_per_teacher * _DICT_ENTRY_BYTES)

    align_indices = n_teachers * (
        _DICT_BASE_BYTES + n_align_per_teacher * _DICT_ENTRY_BYTES)

    total = loc_index + kl_indices + align_indices

    return {
        "loc_index_bytes": loc_index,
        "kl_indices_bytes": kl_indices,
        "align_indices_bytes": align_indices,
        "total_bytes": total,
        "total_mb": round(total / 1e6, 1),
        "total_gb": round(total / 1e9, 2),
    }


# ---------------------------------------------------------------------------
# Memory-mapped cache view (streaming alternative to load_e2_cache)
# ---------------------------------------------------------------------------

class E2CacheView:
    """Memory-mapped E2 cache with O(1) record access.

    Replaces load_e2_cache() for production training. Record data stays on
    disk via mmap; only compact int→int indices live in RAM. Records are
    decoded on demand — only records actually used in a training step get
    unpacked.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest = load_e2_manifest(cache_dir)

        self._positions = MappedPositionRecords(
            os.path.join(cache_dir, "positions.bin"))
        self._loc_index = self._positions.build_loc_index()

        self._kl_readers: dict[str, MappedKLRecords] = {}
        self._kl_pid_idx: dict[str, dict[int, int]] = {}
        self._align_readers: dict[str, MappedAlignRecords] = {}
        self._align_pid_idx: dict[str, dict[int, int]] = {}
        self._embedding_tables: dict[str, torch.Tensor | None] = {}
        self._teacher_names: set[str] = set()

        for spec in self.manifest["teacher_specs"]:
            tdir = os.path.join(cache_dir, "teachers", spec.name)
            if not os.path.isdir(tdir):
                continue
            self._teacher_names.add(spec.name)

            kl_path = os.path.join(tdir, "kl_records.bin")
            if os.path.exists(kl_path):
                kl = MappedKLRecords(kl_path)
                self._kl_readers[spec.name] = kl
                self._kl_pid_idx[spec.name] = kl.build_pid_index()

            align_path = os.path.join(tdir, "align_records.bin")
            if os.path.exists(align_path):
                al = MappedAlignRecords(align_path)
                self._align_readers[spec.name] = al
                self._align_pid_idx[spec.name] = al.build_pid_index()

            emb_path = os.path.join(tdir, "teacher_embeddings.pt")
            if os.path.exists(emb_path):
                self._embedding_tables[spec.name] = torch.load(
                    emb_path, map_location="cpu", weights_only=True)
            else:
                self._embedding_tables[spec.name] = None

    def positions_for_loc(
        self, shard_id: int, seq_offset: int,
    ) -> list[PositionRecord]:
        indices = self._loc_index.get((shard_id, seq_offset), [])
        return [self._positions[i] for i in indices]

    def kl_record(
        self, teacher_name: str, position_id: int,
    ) -> E2KLRecord | None:
        pid_idx = self._kl_pid_idx.get(teacher_name)
        if pid_idx is None:
            return None
        idx = pid_idx.get(position_id)
        if idx is None:
            return None
        return self._kl_readers[teacher_name][idx]

    def align_record(
        self, teacher_name: str, position_id: int,
    ) -> E2AlignRecord | None:
        pid_idx = self._align_pid_idx.get(teacher_name)
        if pid_idx is None:
            return None
        idx = pid_idx.get(position_id)
        if idx is None:
            return None
        return self._align_readers[teacher_name][idx]

    def embedding_table(self, teacher_name: str) -> torch.Tensor | None:
        return self._embedding_tables.get(teacher_name)

    def has_teacher(self, teacher_name: str) -> bool:
        return teacher_name in self._teacher_names

    def kl_pids_ordered(self, teacher_name: str) -> list[int]:
        """Position_ids in file order for shuffle support."""
        idx = self._kl_pid_idx.get(teacher_name)
        return list(idx.keys()) if idx else []

    def align_pids_ordered(self, teacher_name: str) -> list[int]:
        """Position_ids in file order for shuffle support."""
        idx = self._align_pid_idx.get(teacher_name)
        return list(idx.keys()) if idx else []

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    def memory_report(self) -> dict[str, int]:
        """Report actual in-memory index sizes using sys.getsizeof."""
        import sys as _sys
        loc_bytes = _sys.getsizeof(self._loc_index)
        for v in self._loc_index.values():
            loc_bytes += _sys.getsizeof(v)

        kl_bytes = sum(
            _sys.getsizeof(d) for d in self._kl_pid_idx.values())
        align_bytes = sum(
            _sys.getsizeof(d) for d in self._align_pid_idx.values())
        total = loc_bytes + kl_bytes + align_bytes
        return {
            "loc_index_bytes": loc_bytes,
            "kl_indices_bytes": kl_bytes,
            "align_indices_bytes": align_bytes,
            "total_bytes": total,
            "total_mb": round(total / 1e6, 1),
            "n_positions": len(self._positions),
            "n_teachers": len(self._teacher_names),
        }

    def validate(self, data_dir: str | None = None,
                 min_coverage: float = 0.5) -> list[str]:
        """Check cache integrity. Returns list of error strings (empty = OK).

        Checks:
          - Position count matches manifest
          - Teacher count matches manifest
          - Per-teacher record counts > 0 where has_kl/has_align
          - Per-teacher KL coverage >= min_coverage of total positions
          - Duplicate position IDs within KL/align records
          - Embedding table shapes match TeacherSpec
          - Embedding table rank is 2D
          - Shard coverage (if data_dir given)
        """
        errors: list[str] = []
        manifest = self.manifest
        n_positions = manifest["n_positions"]

        if len(self._positions) != n_positions:
            errors.append(
                f"Position count mismatch: file has {len(self._positions)}, "
                f"manifest says {n_positions}")

        if len(self._teacher_names) != manifest["teacher_count"]:
            errors.append(
                f"Teacher count mismatch: found {len(self._teacher_names)}, "
                f"manifest says {manifest['teacher_count']}")

        all_pids = set()
        for i in range(len(self._positions)):
            pid = self._positions[i].position_id
            if pid in all_pids:
                errors.append(f"Duplicate position_id {pid} in positions.bin")
                break
            all_pids.add(pid)

        for spec in manifest["teacher_specs"]:
            if spec.name not in self._teacher_names:
                errors.append(f"Teacher {spec.name} listed in registry but "
                              "has no directory")
                continue

            kl_reader = self._kl_readers.get(spec.name)
            kl_pid_idx = self._kl_pid_idx.get(spec.name, {})
            if spec.has_kl and kl_reader is None:
                errors.append(f"Teacher {spec.name} has_kl=True but no "
                              "kl_records.bin")
            elif spec.has_kl and kl_reader is not None and len(kl_reader) == 0:
                errors.append(f"Teacher {spec.name} has_kl=True but "
                              "kl_records.bin is empty")
            elif spec.has_kl and kl_reader is not None:
                n_kl = len(kl_reader)
                if n_positions > 0 and n_kl / n_positions < min_coverage:
                    errors.append(
                        f"Teacher {spec.name} KL coverage too low: "
                        f"{n_kl}/{n_positions} = "
                        f"{n_kl / n_positions:.1%} < {min_coverage:.0%}")
                if len(kl_pid_idx) != n_kl:
                    errors.append(
                        f"Teacher {spec.name} has {n_kl} KL records but "
                        f"{len(kl_pid_idx)} unique PIDs (duplicates)")

            align_reader = self._align_readers.get(spec.name)
            align_pid_idx = self._align_pid_idx.get(spec.name, {})
            if spec.has_align and align_reader is None:
                errors.append(f"Teacher {spec.name} has_align=True but no "
                              "align_records.bin")
            elif spec.has_align and align_reader is not None and len(align_reader) == 0:
                errors.append(f"Teacher {spec.name} has_align=True but "
                              "align_records.bin is empty")
            elif spec.has_align and align_reader is not None:
                if len(align_pid_idx) != len(align_reader):
                    errors.append(
                        f"Teacher {spec.name} has {len(align_reader)} align "
                        f"records but {len(align_pid_idx)} unique PIDs "
                        "(duplicates)")

            emb = self._embedding_tables.get(spec.name)
            if (spec.has_align or spec.has_semantic) and emb is None:
                errors.append(f"Teacher {spec.name} needs embeddings but "
                              "teacher_embeddings.pt missing")
            elif emb is not None:
                if emb.ndim != 2:
                    errors.append(
                        f"Teacher {spec.name} embedding table has "
                        f"{emb.ndim} dims, expected 2")
                elif emb.shape[1] != spec.hidden_dim:
                    errors.append(
                        f"Teacher {spec.name} embedding dim {emb.shape[1]} "
                        f"!= spec hidden_dim {spec.hidden_dim}")

        if data_dir is not None:
            import glob as _glob
            shard_files = set()
            for p in _glob.glob(os.path.join(data_dir, "*.bin")):
                try:
                    idx = int(os.path.basename(p).split("_")[-1].split(".")[0])
                    shard_files.add(idx)
                except ValueError:
                    pass

            shard_range = manifest.get("shard_range", [0, 0])
            for sid in range(shard_range[0], shard_range[1]):
                if sid not in shard_files:
                    errors.append(
                        f"Shard {sid} in cache range but missing from "
                        f"{data_dir}")

        return errors

    def close(self):
        self._positions.close()
        for r in self._kl_readers.values():
            r.close()
        for r in self._align_readers.values():
            r.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
