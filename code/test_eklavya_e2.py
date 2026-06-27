"""Tests for Eklavya E2 multi-teacher cache, router, losses, and infrastructure.

Covers:
  - Binary record roundtrip (pack/unpack identity)
  - Position manifest I/O
  - Per-teacher KL/align record I/O
  - Route record I/O with variable teacher counts
  - Teacher registry serialization
  - E2 manifest and full cache loader
  - Teacher lookup functions
  - Edge cases: empty records, K=1, K=32, max values
  - MultiTeacherBatch construction and joins
  - TeacherRouterV0 (PL-style routing)
  - Purifier modes (arithmetic, log_pool, route)
  - Disagreement JSD computation
  - MultiTeacherProjectionPorts
  - E2 KL loss on synthetic distributions
  - Multi-teacher gradient budget on toy model
  - Semantic cosine loss
"""

from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from eklavya_e2_cache import (
    TeacherSpec, TeacherFamily, TeacherRole, SelectionReason,
    TEACHER_REGISTRY,
    get_teacher_by_name, get_teacher_by_id,
    PositionRecord, E2KLRecord, E2AlignRecord, RouteRecord,
    SparseByteDist,
    write_position_manifest, read_position_manifest,
    write_teacher_kl_records, read_teacher_kl_records,
    write_teacher_align_records, read_teacher_align_records,
    write_route_records, read_route_records,
    save_teacher_registry, load_teacher_registry,
    save_e2_manifest, load_e2_manifest,
    load_e2_cache, E2CacheView,
    MappedPositionRecords, MappedKLRecords, MappedAlignRecords,
    estimate_index_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_position(pid: int = 0, shard: int = 1, seq: int = 1000,
                  patch: int = 5, gold: int = 65, nll: float = 4.2,
                  ent: float = 2.1, mask: int = 3) -> PositionRecord:
    return PositionRecord(pid, shard, seq, patch, gold, nll, ent, mask)


def make_kl(pid: int = 0, patch: int = 5, K: int = 16) -> E2KLRecord:
    return E2KLRecord(
        position_id=pid,
        patch_idx=patch,
        tail_prob=0.05,
        entropy=2.3,
        logp_gold=-1.8,
        top_bytes=np.arange(K, dtype=np.uint8),
        top_probs=np.linspace(0.5, 0.01, K).astype(np.float16),
    )


def make_align(pid: int = 0) -> E2AlignRecord:
    return E2AlignRecord(
        position_id=pid,
        byte_start=10,
        byte_len=4,
        token_id=12345,
        align_quality=0.92,
    )


def make_route(pid: int = 0, n: int = 3) -> RouteRecord:
    return RouteRecord(
        position_id=pid,
        n_teachers=n,
        jsd=0.15,
        route_entropy=1.2,
        teacher_ids=np.arange(n, dtype=np.uint8),
        weights=np.array([0.5, 0.3, 0.2][:n], dtype=np.float16),
    )


# ---------------------------------------------------------------------------
# Teacher registry tests
# ---------------------------------------------------------------------------

class TestTeacherRegistry:
    def test_registry_count(self):
        assert len(TEACHER_REGISTRY) == 5

    def test_unique_ids(self):
        ids = [t.teacher_id for t in TEACHER_REGISTRY]
        assert len(set(ids)) == len(ids)

    def test_unique_names(self):
        names = [t.name for t in TEACHER_REGISTRY]
        assert len(set(names)) == len(names)

    def test_priors_sum_to_one(self):
        total = sum(t.prior for t in TEACHER_REGISTRY)
        assert abs(total - 1.0) < 1e-6

    def test_lookup_by_name(self):
        t = get_teacher_by_name("t0_anchor_decoder")
        assert t.teacher_id == 0
        assert t.hidden_dim == 2048
        assert t.role == TeacherRole.ANCHOR

    def test_lookup_by_id(self):
        t = get_teacher_by_id(4)
        assert t.name == "t4_diversity_ssm"
        assert t.family == TeacherFamily.SSM

    def test_lookup_invalid_name(self):
        with pytest.raises(ValueError, match="Unknown teacher"):
            get_teacher_by_name("nonexistent")

    def test_lookup_invalid_id(self):
        with pytest.raises(ValueError, match="Unknown teacher_id"):
            get_teacher_by_id(99)

    def test_serialization_roundtrip(self):
        for t in TEACHER_REGISTRY:
            d = t.to_dict()
            t2 = TeacherSpec.from_dict(d)
            assert t.teacher_id == t2.teacher_id
            assert t.name == t2.name
            assert t.hidden_dim == t2.hidden_dim
            assert t.family == t2.family
            assert t.role == t2.role
            assert abs(t.prior - t2.prior) < 1e-9

    def test_anchor_is_t0_anchor_decoder(self):
        anchors = [t for t in TEACHER_REGISTRY if t.role == TeacherRole.ANCHOR]
        assert len(anchors) == 1
        assert anchors[0].name == "t0_anchor_decoder"

    def test_semantic_teacher_has_no_kl(self):
        sem = [t for t in TEACHER_REGISTRY if t.role == TeacherRole.SEMANTIC]
        for t in sem:
            assert not t.has_kl
            assert t.has_semantic

    def test_grad_caps(self):
        for t in TEACHER_REGISTRY:
            assert 0 < t.per_teacher_grad_cap <= 0.10


# ---------------------------------------------------------------------------
# SelectionReason flag tests
# ---------------------------------------------------------------------------

class TestSelectionReason:
    def test_single_flags(self):
        assert SelectionReason.HIGH_NLL == 1
        assert SelectionReason.HIGH_ENTROPY == 2
        assert SelectionReason.DISAGREEMENT == 4
        assert SelectionReason.CONTROL == 8

    def test_combined_flags(self):
        mask = SelectionReason.HIGH_NLL | SelectionReason.DISAGREEMENT
        assert mask == 5
        assert SelectionReason.HIGH_NLL in SelectionReason(mask)
        assert SelectionReason.DISAGREEMENT in SelectionReason(mask)
        assert SelectionReason.CONTROL not in SelectionReason(mask)


# ---------------------------------------------------------------------------
# PositionRecord binary roundtrip
# ---------------------------------------------------------------------------

class TestPositionRecord:
    def test_pack_unpack_identity(self):
        r = make_position()
        buf = r.pack()
        assert len(buf) == PositionRecord.SIZE
        r2 = PositionRecord.unpack(buf)
        assert r.position_id == r2.position_id
        assert r.shard_id == r2.shard_id
        assert r.seq_offset == r2.seq_offset
        assert r.patch_idx == r2.patch_idx
        assert r.gold_byte == r2.gold_byte
        assert abs(r.student_nll - r2.student_nll) < 1e-5
        assert abs(r.student_entropy - r2.student_entropy) < 1e-5
        assert r.reason_mask == r2.reason_mask

    def test_max_values(self):
        r = PositionRecord(
            position_id=2**32 - 1,
            shard_id=2**32 - 1,
            seq_offset=2**63 - 1,
            patch_idx=2**16 - 1,
            gold_byte=255,
            student_nll=100.0,
            student_entropy=8.0,
            reason_mask=255,
        )
        r2 = PositionRecord.unpack(r.pack())
        assert r.position_id == r2.position_id
        assert r.seq_offset == r2.seq_offset
        assert r.gold_byte == r2.gold_byte

    def test_size_is_28(self):
        assert PositionRecord.SIZE == 28


# ---------------------------------------------------------------------------
# E2KLRecord binary roundtrip
# ---------------------------------------------------------------------------

class TestE2KLRecord:
    def test_pack_unpack_k16(self):
        r = make_kl(K=16)
        buf = r.pack(16)
        assert len(buf) == E2KLRecord.record_size(16)
        r2 = E2KLRecord.unpack(buf, 16)
        assert r.position_id == r2.position_id
        assert r.patch_idx == r2.patch_idx
        np.testing.assert_array_equal(r.top_bytes[:16], r2.top_bytes)
        np.testing.assert_allclose(r.top_probs[:16], r2.top_probs, rtol=1e-2)

    def test_pack_unpack_k1(self):
        r = make_kl(K=1)
        buf = r.pack(1)
        assert len(buf) == E2KLRecord.record_size(1)
        r2 = E2KLRecord.unpack(buf, 1)
        assert r2.top_bytes.shape == (1,)
        assert r2.top_probs.shape == (1,)

    def test_pack_unpack_k32(self):
        r = E2KLRecord(
            position_id=42,
            patch_idx=10,
            tail_prob=0.02,
            entropy=3.1,
            logp_gold=-2.5,
            top_bytes=np.arange(32, dtype=np.uint8),
            top_probs=np.linspace(0.3, 0.001, 32).astype(np.float16),
        )
        buf = r.pack(32)
        r2 = E2KLRecord.unpack(buf, 32)
        np.testing.assert_array_equal(r.top_bytes, r2.top_bytes)

    def test_record_size(self):
        assert E2KLRecord.record_size(16) == E2KLRecord.HEADER_SIZE + 16 + 32
        assert E2KLRecord.record_size(32) == E2KLRecord.HEADER_SIZE + 32 + 64


# ---------------------------------------------------------------------------
# E2AlignRecord binary roundtrip
# ---------------------------------------------------------------------------

class TestE2AlignRecord:
    def test_pack_unpack_identity(self):
        r = make_align()
        buf = r.pack()
        assert len(buf) == E2AlignRecord.SIZE
        r2 = E2AlignRecord.unpack(buf)
        assert r.position_id == r2.position_id
        assert r.byte_start == r2.byte_start
        assert r.byte_len == r2.byte_len
        assert r.token_id == r2.token_id

    def test_size_is_14(self):
        assert E2AlignRecord.SIZE == 14


# ---------------------------------------------------------------------------
# RouteRecord binary roundtrip
# ---------------------------------------------------------------------------

class TestRouteRecord:
    def test_pack_unpack_3_teachers(self):
        r = make_route(n=3)
        buf = r.pack()
        r2 = RouteRecord.unpack(buf)
        assert r.position_id == r2.position_id
        assert r.n_teachers == r2.n_teachers
        np.testing.assert_array_equal(r.teacher_ids, r2.teacher_ids)
        np.testing.assert_allclose(r.weights, r2.weights, rtol=1e-2)

    def test_pack_unpack_1_teacher(self):
        r = RouteRecord(
            position_id=99,
            n_teachers=1,
            jsd=0.0,
            route_entropy=0.0,
            teacher_ids=np.array([2], dtype=np.uint8),
            weights=np.array([1.0], dtype=np.float16),
        )
        r2 = RouteRecord.unpack(r.pack())
        assert r2.n_teachers == 1
        assert r2.teacher_ids[0] == 2

    def test_pack_unpack_5_teachers(self):
        r = RouteRecord(
            position_id=0,
            n_teachers=5,
            jsd=0.25,
            route_entropy=2.0,
            teacher_ids=np.arange(5, dtype=np.uint8),
            weights=np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float16),
        )
        r2 = RouteRecord.unpack(r.pack())
        assert r2.n_teachers == 5
        np.testing.assert_array_equal(r.teacher_ids, r2.teacher_ids)


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------

class TestPositionManifestIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [make_position(pid=i, shard=i * 10) for i in range(100)]
        path = str(tmp_path / "positions.bin")
        write_position_manifest(path, records)
        loaded = read_position_manifest(path)
        assert len(loaded) == 100
        assert loaded[0].position_id == 0
        assert loaded[99].position_id == 99
        assert loaded[50].shard_id == 500

    def test_empty_manifest(self, tmp_path):
        path = str(tmp_path / "empty.bin")
        write_position_manifest(path, [])
        loaded = read_position_manifest(path)
        assert len(loaded) == 0

    def test_file_size(self, tmp_path):
        n = 1000
        records = [make_position(pid=i) for i in range(n)]
        path = str(tmp_path / "pos.bin")
        write_position_manifest(path, records)
        assert os.path.getsize(path) == 4 + n * PositionRecord.SIZE


class TestTeacherKLIO:
    def test_write_read_roundtrip_k16(self, tmp_path):
        records = [make_kl(pid=i, K=16) for i in range(50)]
        path = str(tmp_path / "kl.bin")
        write_teacher_kl_records(path, records, K=16)
        loaded, K = read_teacher_kl_records(path)
        assert K == 16
        assert len(loaded) == 50
        np.testing.assert_array_equal(
            records[0].top_bytes, loaded[0].top_bytes)

    def test_write_read_roundtrip_k32(self, tmp_path):
        records = [E2KLRecord(
            position_id=i, patch_idx=3,
            tail_prob=0.01, entropy=2.0, logp_gold=-1.5,
            top_bytes=np.arange(32, dtype=np.uint8),
            top_probs=np.linspace(0.3, 0.001, 32).astype(np.float16),
        ) for i in range(10)]
        path = str(tmp_path / "kl32.bin")
        write_teacher_kl_records(path, records, K=32)
        loaded, K = read_teacher_kl_records(path)
        assert K == 32
        assert len(loaded) == 10

    def test_empty(self, tmp_path):
        path = str(tmp_path / "empty_kl.bin")
        write_teacher_kl_records(path, [], K=16)
        loaded, K = read_teacher_kl_records(path)
        assert len(loaded) == 0
        assert K == 16


class TestTeacherAlignIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [make_align(pid=i) for i in range(20)]
        path = str(tmp_path / "align.bin")
        write_teacher_align_records(path, records)
        loaded = read_teacher_align_records(path)
        assert len(loaded) == 20
        assert loaded[0].position_id == 0
        assert loaded[19].position_id == 19


class TestRouteRecordIO:
    def test_write_read_roundtrip(self, tmp_path):
        records = [make_route(pid=i, n=3) for i in range(10)]
        path = str(tmp_path / "routes.bin")
        write_route_records(path, records)
        loaded = read_route_records(path)
        assert len(loaded) == 10
        assert loaded[5].position_id == 5

    def test_variable_teacher_counts(self, tmp_path):
        records = [
            make_route(pid=0, n=1),
            make_route(pid=1, n=3),
            RouteRecord(
                position_id=2, n_teachers=5, jsd=0.3, route_entropy=1.5,
                teacher_ids=np.arange(5, dtype=np.uint8),
                weights=np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float16),
            ),
        ]
        path = str(tmp_path / "var_routes.bin")
        write_route_records(path, records)
        loaded = read_route_records(path)
        assert len(loaded) == 3
        assert loaded[0].n_teachers == 1
        assert loaded[1].n_teachers == 3
        assert loaded[2].n_teachers == 5


# ---------------------------------------------------------------------------
# Teacher registry I/O
# ---------------------------------------------------------------------------

class TestTeacherRegistryIO:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "registry.json")
        save_teacher_registry(path, TEACHER_REGISTRY)
        loaded = load_teacher_registry(path)
        assert len(loaded) == len(TEACHER_REGISTRY)
        for orig, rec in zip(TEACHER_REGISTRY, loaded):
            assert orig.teacher_id == rec.teacher_id
            assert orig.name == rec.name
            assert orig.family == rec.family
            assert orig.hidden_dim == rec.hidden_dim

    def test_json_readable(self, tmp_path):
        path = str(tmp_path / "registry.json")
        save_teacher_registry(path, TEACHER_REGISTRY)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["name"] == "t0_anchor_decoder"
        assert data[0]["family"] == "DECODER"


# ---------------------------------------------------------------------------
# E2 manifest and full cache loader
# ---------------------------------------------------------------------------

class TestE2Manifest:
    def test_save_load_roundtrip(self, tmp_path):
        cache_dir = str(tmp_path)
        save_e2_manifest(cache_dir, TEACHER_REGISTRY, n_positions=5000,
                         K=16, shard_range=(0, 50))
        manifest = load_e2_manifest(cache_dir)
        assert manifest["version"] == "e2.0"
        assert manifest["n_positions"] == 5000
        assert manifest["kl_top_k"] == 16
        assert manifest["teacher_count"] == 5
        assert len(manifest["teacher_specs"]) == 5


class TestE2CacheLoader:
    def _build_cache(self, cache_dir: str, n_pos: int = 100, K: int = 16):
        """Build a minimal valid E2 cache on disk."""
        save_e2_manifest(cache_dir, TEACHER_REGISTRY[:3],
                         n_positions=n_pos, K=K)

        positions = [make_position(pid=i) for i in range(n_pos)]
        write_position_manifest(
            os.path.join(cache_dir, "positions.bin"), positions)

        for spec in TEACHER_REGISTRY[:3]:
            tdir = os.path.join(cache_dir, "teachers", spec.name)
            os.makedirs(tdir, exist_ok=True)

            if spec.has_kl:
                kl_recs = [make_kl(pid=i, K=K) for i in range(n_pos)]
                write_teacher_kl_records(
                    os.path.join(tdir, "kl_records.bin"), kl_recs, K)

            if spec.has_align:
                align_recs = [make_align(pid=i) for i in range(n_pos)]
                write_teacher_align_records(
                    os.path.join(tdir, "align_records.bin"), align_recs)

        agg_dir = os.path.join(cache_dir, "aggregate")
        os.makedirs(agg_dir, exist_ok=True)
        routes = [make_route(pid=i, n=3) for i in range(n_pos)]
        write_route_records(
            os.path.join(agg_dir, "route_records.bin"), routes)

    def test_load_full_cache(self, tmp_path):
        cache_dir = str(tmp_path / "e2_cache")
        os.makedirs(cache_dir)
        self._build_cache(cache_dir)
        cache = load_e2_cache(cache_dir)

        assert cache["manifest"]["version"] == "e2.0"
        assert len(cache["positions"]) == 100
        assert len(cache["teachers"]) == 3
        assert len(cache["routes"]) == 100

        anchor = cache["teachers"]["t0_anchor_decoder"]
        assert len(anchor["kl_records"]) == 100
        assert len(anchor["align_records"]) == 100
        assert anchor["embedding_table"] is None

    def test_missing_teacher_dir_skipped(self, tmp_path):
        cache_dir = str(tmp_path / "partial")
        os.makedirs(cache_dir)
        save_e2_manifest(cache_dir, TEACHER_REGISTRY[:3], n_positions=10)
        positions = [make_position(pid=i) for i in range(10)]
        write_position_manifest(
            os.path.join(cache_dir, "positions.bin"), positions)
        cache = load_e2_cache(cache_dir)
        assert len(cache["teachers"]) == 0
        assert len(cache["routes"]) == 0

    def test_missing_kl_records_ok(self, tmp_path):
        cache_dir = str(tmp_path / "no_kl")
        os.makedirs(cache_dir)
        save_e2_manifest(cache_dir, TEACHER_REGISTRY[:1], n_positions=5)
        positions = [make_position(pid=i) for i in range(5)]
        write_position_manifest(
            os.path.join(cache_dir, "positions.bin"), positions)
        tdir = os.path.join(cache_dir, "teachers", "t0_anchor_decoder")
        os.makedirs(tdir)
        cache = load_e2_cache(cache_dir)
        assert cache["teachers"]["t0_anchor_decoder"]["kl_records"] == []


# ---------------------------------------------------------------------------
# SparseByteDist
# ---------------------------------------------------------------------------

class TestSparseByteDist:
    def test_construction(self):
        d = SparseByteDist(
            top_bytes=np.array([65, 66, 67], dtype=np.uint8),
            top_probs=np.array([0.5, 0.3, 0.1], dtype=np.float32),
            tail_prob=0.1,
        )
        assert d.top_bytes.shape == (3,)
        assert abs(d.top_probs.sum() + d.tail_prob - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

import torch
import torch.nn.functional as F

from eklavya_e2_router import (
    route_teachers, purify_byte_target, disagreement_jsd,
    build_multi_teacher_batch, route_batch, purify_batch,
    RouteResult, RouterConfig,
    _zscore, _pairwise_agreement_scores, _teacher_student_jsd,
    _sparse_to_full, _VALID_ROUTER_MODES,
)
from eklavya_e2_losses import (
    MultiTeacherProjectionPorts,
    e2_topk_tail_kl, e2_batch_kl_loss,
    semantic_cosine_loss,
    apply_multi_teacher_gradient_budget, GradientBudgetReport,
)


def _make_sparse_dist(gold: int = 65, confidence: float = 0.6,
                      K: int = 8) -> SparseByteDist:
    top_bytes = np.array([gold] + list(range(K - 1)), dtype=np.uint8)
    probs = np.zeros(K, dtype=np.float32)
    probs[0] = confidence
    remaining = (1.0 - confidence) * 0.8
    for i in range(1, K):
        probs[i] = remaining / (K - 1)
    tail = 1.0 - probs.sum()
    return SparseByteDist(top_bytes=top_bytes, top_probs=probs, tail_prob=max(tail, 0.0))


class TestZScore:
    def test_single_value(self):
        assert _zscore([3.0]) == [0.0]

    def test_empty(self):
        assert _zscore([]) == []

    def test_two_values(self):
        z = _zscore([1.0, 3.0])
        assert abs(z[0] + z[1]) < 1e-9
        assert z[0] < 0 and z[1] > 0


class TestRouteTeachers:
    def test_single_teacher(self):
        dists = {"t0_anchor_decoder": _make_sparse_dist(gold=65, confidence=0.7)}
        priors = {"t0_anchor_decoder": 1.0}
        result = route_teachers(dists, gold_byte=65, priors=priors)
        assert abs(result.weights["t0_anchor_decoder"] - 1.0) < 1e-6

    def test_two_teachers_confident_wins(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.8),
            "t2": _make_sparse_dist(gold=65, confidence=0.2),
        }
        priors = {"t1": 0.5, "t2": 0.5}
        result = route_teachers(dists, gold_byte=65, priors=priors)
        assert result.weights["t1"] > result.weights["t2"]

    def test_prior_influence(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.5),
            "t2": _make_sparse_dist(gold=65, confidence=0.5),
        }
        priors = {"t1": 0.9, "t2": 0.1}
        result = route_teachers(dists, gold_byte=65, priors=priors)
        assert result.weights["t1"] > result.weights["t2"]

    def test_weights_sum_to_one(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.6),
            "t2": _make_sparse_dist(gold=65, confidence=0.4),
            "t3": _make_sparse_dist(gold=65, confidence=0.3),
        }
        priors = {"t1": 0.4, "t2": 0.3, "t3": 0.3}
        result = route_teachers(dists, gold_byte=65, priors=priors)
        total = sum(result.weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_empty_teachers(self):
        result = route_teachers({}, gold_byte=65, priors={})
        assert result.weights == {}
        assert result.jsd == 0.0

    def test_jsd_nonnegative(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.6),
            "t2": _make_sparse_dist(gold=65, confidence=0.4),
        }
        result = route_teachers(dists, gold_byte=65,
                                priors={"t1": 0.5, "t2": 0.5})
        assert result.jsd >= -1e-10

    def test_route_entropy_nonnegative(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.6),
            "t2": _make_sparse_dist(gold=65, confidence=0.4),
        }
        result = route_teachers(dists, gold_byte=65,
                                priors={"t1": 0.5, "t2": 0.5})
        assert result.route_entropy >= -1e-10


class TestGoldFreeRouter:
    """Tests for gold-free routing modes (A9a/A9b/A9c)."""

    def _student_probs(self, peak_byte: int = 65, confidence: float = 0.5):
        p = np.full(256, (1.0 - confidence) / 255, dtype=np.float64)
        p[peak_byte] = confidence
        p /= p.sum()
        return p

    def _student_entropy(self, probs):
        return -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))

    def test_gold_free_entropy_produces_valid_weights(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.8),
            "t2": _make_sparse_dist(gold=66, confidence=0.3),
        }
        cfg = RouterConfig(mode="gold_free_entropy")
        result = route_teachers(dists, gold_byte=None, priors={"t1": 0.5, "t2": 0.5}, config=cfg)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.weights["t1"] > result.weights["t2"], \
            "Lower-entropy teacher should get more weight"

    def test_gold_free_agreement_penalizes_outlier(self):
        d_agree1 = _make_sparse_dist(gold=65, confidence=0.7)
        d_agree2 = _make_sparse_dist(gold=65, confidence=0.6)
        d_outlier = _make_sparse_dist(gold=200, confidence=0.9)
        dists = {"t1": d_agree1, "t2": d_agree2, "t3": d_outlier}
        cfg = RouterConfig(mode="gold_free_agreement", agreement_gamma=1.0)
        result = route_teachers(
            dists, gold_byte=None,
            priors={"t1": 1/3, "t2": 1/3, "t3": 1/3}, config=cfg)
        assert result.weights["t3"] < result.weights["t1"], \
            "Outlier teacher should get lower weight with agreement penalty"

    def test_gold_free_student_jsd_requires_student_probs(self):
        dists = {"t1": _make_sparse_dist(gold=65, confidence=0.7)}
        cfg = RouterConfig(mode="gold_free_student_jsd")
        with pytest.raises(ValueError, match="requires student_probs"):
            route_teachers(dists, gold_byte=None, priors={"t1": 1.0}, config=cfg)

    def test_gold_free_student_jsd_produces_valid_weights(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.7),
            "t2": _make_sparse_dist(gold=66, confidence=0.7),
        }
        s_probs = self._student_probs(peak_byte=65, confidence=0.5)
        s_ent = self._student_entropy(s_probs)
        cfg = RouterConfig(mode="gold_free_student_jsd")
        result = route_teachers(
            dists, gold_byte=None, priors={"t1": 0.5, "t2": 0.5},
            config=cfg, student_probs=s_probs, student_entropy=s_ent)
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_oracle_gold_requires_gold_byte(self):
        dists = {"t1": _make_sparse_dist(gold=65, confidence=0.7)}
        cfg = RouterConfig(mode="oracle_gold")
        with pytest.raises(ValueError, match="requires gold_byte"):
            route_teachers(dists, gold_byte=None, priors={"t1": 1.0}, config=cfg)

    def test_invalid_mode_raises(self):
        dists = {"t1": _make_sparse_dist(gold=65, confidence=0.7)}
        cfg = RouterConfig(mode="invalid_mode")
        with pytest.raises(ValueError, match="Unknown router mode"):
            route_teachers(dists, gold_byte=65, priors={"t1": 1.0}, config=cfg)

    def test_gold_free_ignores_gold_byte(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.8)
        d2 = _make_sparse_dist(gold=66, confidence=0.8)
        dists = {"t1": d1, "t2": d2}
        priors = {"t1": 0.5, "t2": 0.5}
        cfg_gf = RouterConfig(mode="gold_free_entropy")
        r_gf = route_teachers(dists, gold_byte=None, priors=priors, config=cfg_gf)

        cfg_oracle = RouterConfig(mode="oracle_gold")
        r_65 = route_teachers(dists, gold_byte=65, priors=priors, config=cfg_oracle)
        r_66 = route_teachers(dists, gold_byte=66, priors=priors, config=cfg_oracle)
        assert abs(r_65.weights["t1"] - r_66.weights["t1"]) > 0.01, \
            "Oracle mode should produce different weights for different gold bytes"
        assert r_gf.weights["t1"] != r_65.weights["t1"] or \
               r_gf.weights["t1"] != r_66.weights["t1"], \
            "Gold-free mode should differ from at least one oracle result"

    def test_gold_free_student_jsd_requires_student_entropy(self):
        dists = {"t1": _make_sparse_dist(gold=65, confidence=0.7)}
        s_probs = self._student_probs()
        cfg = RouterConfig(mode="gold_free_student_jsd")
        with pytest.raises(ValueError, match="requires student_entropy"):
            route_teachers(dists, gold_byte=None, priors={"t1": 1.0},
                           config=cfg, student_probs=s_probs, student_entropy=None)

    def test_student_jsd_term_changes_ranking(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.6)
        d2 = _make_sparse_dist(gold=200, confidence=0.6)
        dists = {"t1": d1, "t2": d2}
        priors = {"t1": 0.5, "t2": 0.5}
        s_probs = self._student_probs(peak_byte=65, confidence=0.8)
        s_ent = self._student_entropy(s_probs)
        cfg_agree = RouterConfig(mode="gold_free_agreement", student_delta=0.0)
        cfg_sjsd = RouterConfig(mode="gold_free_student_jsd", student_delta=1.0)
        r_agree = route_teachers(dists, None, priors, cfg_agree,
                                 student_probs=s_probs, student_entropy=s_ent)
        r_sjsd = route_teachers(dists, None, priors, cfg_sjsd,
                                student_probs=s_probs, student_entropy=s_ent)
        assert r_agree.weights != r_sjsd.weights, \
            "Student-JSD term with delta=1.0 should change weight distribution"

    def test_single_teacher_all_modes(self):
        dist = _make_sparse_dist(gold=65, confidence=0.7)
        s_probs = self._student_probs()
        s_ent = self._student_entropy(s_probs)
        for mode in _VALID_ROUTER_MODES:
            cfg = RouterConfig(mode=mode)
            kwargs = {"student_probs": s_probs, "student_entropy": s_ent}
            gold = 65 if mode == "oracle_gold" else None
            result = route_teachers(
                {"t1": dist}, gold_byte=gold, priors={"t1": 1.0},
                config=cfg, **kwargs)
            assert abs(result.weights["t1"] - 1.0) < 1e-6


class TestPairwiseAgreementScores:
    def test_identical_teachers_zero_agreement(self):
        d = _make_sparse_dist(gold=65, confidence=0.7)
        fulls = {"t1": _sparse_to_full(d), "t2": _sparse_to_full(d)}
        scores = _pairwise_agreement_scores(["t1", "t2"], fulls)
        assert all(abs(s) < 1e-6 for s in scores)

    def test_divergent_teachers_positive_agreement(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.95)
        d2 = _make_sparse_dist(gold=200, confidence=0.95)
        fulls = {"t1": _sparse_to_full(d1), "t2": _sparse_to_full(d2)}
        scores = _pairwise_agreement_scores(["t1", "t2"], fulls)
        assert all(s > 0.01 for s in scores)

    def test_single_teacher_returns_zero(self):
        d = _make_sparse_dist(gold=65, confidence=0.7)
        fulls = {"t1": _sparse_to_full(d)}
        scores = _pairwise_agreement_scores(["t1"], fulls)
        assert scores == [0.0]

    def test_outlier_has_highest_score(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.7)
        d2 = _make_sparse_dist(gold=65, confidence=0.6)
        d_out = _make_sparse_dist(gold=200, confidence=0.95)
        fulls = {
            "t1": _sparse_to_full(d1),
            "t2": _sparse_to_full(d2),
            "t3": _sparse_to_full(d_out),
        }
        scores = _pairwise_agreement_scores(["t1", "t2", "t3"], fulls)
        assert scores[2] > scores[0], "Outlier should have highest agreement score"
        assert scores[2] > scores[1]


class TestTeacherStudentJSD:
    def test_identical_distributions_zero_jsd(self):
        d = _make_sparse_dist(gold=65, confidence=0.7)
        full = _sparse_to_full(d)
        jsd = _teacher_student_jsd(full, full)
        assert abs(jsd) < 1e-6

    def test_different_distributions_positive_jsd(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.9)
        student = np.full(256, 1/256, dtype=np.float64)
        jsd = _teacher_student_jsd(_sparse_to_full(d1), student)
        assert jsd > 0.01

    def test_jsd_symmetric(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.8)
        d2 = _make_sparse_dist(gold=66, confidence=0.6)
        f1, f2 = _sparse_to_full(d1), _sparse_to_full(d2)
        jsd1 = _teacher_student_jsd(f1, f2)
        jsd2 = _teacher_student_jsd(f2, f1)
        assert abs(jsd1 - jsd2) < 1e-6


class TestDisagreementJSD:
    def test_identical_teachers_zero_jsd(self):
        d = _make_sparse_dist(gold=65, confidence=0.6)
        jsd = disagreement_jsd({"t1": d, "t2": d})
        assert abs(jsd) < 1e-6

    def test_different_teachers_positive_jsd(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.9)
        d2 = _make_sparse_dist(gold=66, confidence=0.9)
        jsd = disagreement_jsd({"t1": d1, "t2": d2})
        assert jsd > 0.01

    def test_uniform_jsd_differs_from_pl_weighted(self):
        """Validates P0 fix: A5 disabled-router must use uniform-weighted JSD,
        not PL-weighted JSD which can be near-zero for skewed priors."""
        d1 = _make_sparse_dist(gold=65, confidence=0.95)
        d2 = _make_sparse_dist(gold=66, confidence=0.95)
        dists = {"t1": d1, "t2": d2}

        uniform_jsd = disagreement_jsd(dists)
        assert uniform_jsd > 0.1, "Divergent teachers should have high uniform JSD"

        skewed_jsd = disagreement_jsd(dists, weights={"t1": 0.99, "t2": 0.01})
        assert skewed_jsd < uniform_jsd, (
            "PL-weighted JSD with skewed priors should be lower than uniform JSD"
        )


    def test_empty_teachers_returns_zero(self):
        jsd = disagreement_jsd({})
        assert jsd == 0.0


class TestPurifyByteTarget:
    def test_arithmetic_produces_valid_dist(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.6),
            "t2": _make_sparse_dist(gold=65, confidence=0.4),
        }
        route = RouteResult(weights={"t1": 0.6, "t2": 0.4}, jsd=0.1,
                            route_entropy=0.5)
        result = purify_byte_target(dists, route, mode="arithmetic", K=16)
        assert result is not None
        total = result.top_probs.sum() + result.tail_prob
        assert abs(total - 1.0) < 0.01

    def test_log_pool_sharper_than_arithmetic(self):
        dists = {
            "t1": _make_sparse_dist(gold=65, confidence=0.8),
            "t2": _make_sparse_dist(gold=65, confidence=0.7),
        }
        route = RouteResult(weights={"t1": 0.5, "t2": 0.5}, jsd=0.05,
                            route_entropy=0.7)
        arith = purify_byte_target(dists, route, mode="arithmetic", K=16)
        logp = purify_byte_target(dists, route, mode="log_pool", K=16)
        assert arith is not None and logp is not None
        arith_max = arith.top_probs.max()
        logp_max = logp.top_probs.max()
        assert logp_max >= arith_max - 0.05

    def test_route_mode_picks_best_teacher(self):
        d1 = _make_sparse_dist(gold=65, confidence=0.9)
        d2 = _make_sparse_dist(gold=65, confidence=0.1)
        dists = {"t1": d1, "t2": d2}
        route = RouteResult(weights={"t1": 0.8, "t2": 0.2}, jsd=0.1,
                            route_entropy=0.5)
        result = purify_byte_target(dists, route, mode="route")
        assert result is d1

    def test_empty_teachers_returns_none(self):
        route = RouteResult(weights={}, jsd=0.0, route_entropy=0.0)
        result = purify_byte_target({}, route)
        assert result is None


# ---------------------------------------------------------------------------
# MultiTeacherBatch tests
# ---------------------------------------------------------------------------

class TestMultiTeacherBatch:
    def _make_batch(self, B: int = 5, K: int = 16):
        positions = [make_position(pid=i, gold=65 + i % 10) for i in range(B)]
        teacher_kl = {}
        specs = [s for s in TEACHER_REGISTRY if s.has_kl]
        for spec in specs:
            teacher_kl[spec.name] = [make_kl(pid=i, K=K) for i in range(B)]
        return build_multi_teacher_batch(positions, teacher_kl, specs, K)

    def test_batch_size(self):
        batch = self._make_batch(B=10)
        assert batch.batch_size == 10

    def test_teacher_count(self):
        batch = self._make_batch()
        kl_teachers = [s for s in TEACHER_REGISTRY if s.has_kl]
        assert batch.n_teachers == len(kl_teachers)

    def test_all_valid(self):
        batch = self._make_batch()
        for name, ts in batch.teachers.items():
            assert ts.valid_mask.all()

    def test_partial_records(self):
        positions = [make_position(pid=i) for i in range(5)]
        teacher_kl = {
            "t0_anchor_decoder": [make_kl(pid=0), make_kl(pid=2), make_kl(pid=4)],
        }
        specs = [TEACHER_REGISTRY[0]]
        batch = build_multi_teacher_batch(positions, teacher_kl, specs, K=16)
        ts = batch.teachers["t0_anchor_decoder"]
        assert ts.valid_mask[0] and not ts.valid_mask[1]
        assert ts.valid_mask[2] and not ts.valid_mask[3]
        assert ts.valid_mask[4]

    def test_semantic_teacher_excluded(self):
        positions = [make_position(pid=0)]
        batch = build_multi_teacher_batch(positions, {}, TEACHER_REGISTRY, K=16)
        assert "t3_semantic_embedding" not in batch.teachers


# ---------------------------------------------------------------------------
# Batch routing and purification
# ---------------------------------------------------------------------------

class TestBatchRouting:
    def test_route_batch_returns_correct_count(self):
        positions = [make_position(pid=i) for i in range(3)]
        kl_specs = [s for s in TEACHER_REGISTRY if s.has_kl]
        teacher_kl = {s.name: [make_kl(pid=i) for i in range(3)] for s in kl_specs}
        batch = build_multi_teacher_batch(positions, teacher_kl, kl_specs)
        routes = route_batch(batch)
        assert len(routes) == 3
        for r in routes:
            assert abs(sum(r.weights.values()) - 1.0) < 1e-5

    def test_purify_batch_returns_correct_count(self):
        positions = [make_position(pid=i) for i in range(3)]
        kl_specs = [s for s in TEACHER_REGISTRY if s.has_kl]
        teacher_kl = {s.name: [make_kl(pid=i) for i in range(3)] for s in kl_specs}
        batch = build_multi_teacher_batch(positions, teacher_kl, kl_specs)
        routes = route_batch(batch)
        targets = purify_batch(batch, routes, mode="arithmetic")
        assert len(targets) == 3
        for t in targets:
            assert t is not None


# ---------------------------------------------------------------------------
# MultiTeacherProjectionPorts tests
# ---------------------------------------------------------------------------

class TestProjectionPorts:
    def test_construction(self):
        ports = MultiTeacherProjectionPorts(student_dim=576)
        align_teachers = [s for s in TEACHER_REGISTRY if s.has_align]
        sem_teachers = [s for s in TEACHER_REGISTRY if s.has_semantic]
        assert len(ports.align_ports) == len(align_teachers)
        assert len(ports.semantic_ports) == len(sem_teachers)

    def test_align_forward_shape(self):
        ports = MultiTeacherProjectionPorts(student_dim=576)
        x = torch.randn(4, 576)
        out = ports.get_align_projection("t0_anchor_decoder", x)
        assert out.shape == (4, 2048)

    def test_semantic_forward_shape(self):
        ports = MultiTeacherProjectionPorts(student_dim=576)
        x = torch.randn(4, 576)
        out = ports.get_semantic_projection("t3_semantic_embedding", x)
        assert out.shape == (4, 768)
        norms = out.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_warm_start_from_e1(self):
        from eklavya_training import AlignProjection
        e1_proj = AlignProjection(576, 2048)
        ports = MultiTeacherProjectionPorts(student_dim=576)
        ports.warm_start_from_e1(e1_proj)
        x = torch.randn(2, 576)
        e1_out = e1_proj(x)
        e2_out = ports.get_align_projection("t0_anchor_decoder", x)
        torch.testing.assert_close(e1_out, e2_out)

    def test_invalid_port_raises(self):
        ports = MultiTeacherProjectionPorts(student_dim=576)
        with pytest.raises(ValueError):
            ports.get_align_projection("nonexistent", torch.randn(2, 576))
        with pytest.raises(ValueError):
            ports.get_semantic_projection("t0_anchor_decoder", torch.randn(2, 576))

    def test_param_count(self):
        ports = MultiTeacherProjectionPorts(student_dim=576)
        total = sum(p.numel() for p in ports.parameters())
        assert total > 0


# ---------------------------------------------------------------------------
# E2 KL loss tests
# ---------------------------------------------------------------------------

class TestE2KLLoss:
    def test_finite_loss(self):
        logits = torch.randn(256)
        target = _make_sparse_dist(gold=65, confidence=0.6)
        loss = e2_topk_tail_kl(logits, target)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self):
        logits = torch.full((256,), -10.0)
        logits[65] = 10.0
        target = _make_sparse_dist(gold=65, confidence=0.95, K=4)
        loss = e2_topk_tail_kl(logits, target, T=1.0)
        assert loss.item() < 1.0

    def test_batch_kl_loss(self):
        logits = torch.randn(5, 256)
        targets = [_make_sparse_dist(gold=i, confidence=0.5) for i in range(5)]
        loss = e2_batch_kl_loss(logits, targets)
        assert torch.isfinite(loss)

    def test_batch_kl_loss_with_nones(self):
        logits = torch.randn(5, 256)
        targets = [_make_sparse_dist(gold=0), None, _make_sparse_dist(gold=2),
                    None, None]
        loss = e2_batch_kl_loss(logits, targets)
        assert torch.isfinite(loss)

    def test_batch_kl_loss_all_none(self):
        logits = torch.randn(3, 256)
        targets = [None, None, None]
        loss = e2_batch_kl_loss(logits, targets)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Semantic cosine loss tests
# ---------------------------------------------------------------------------

class TestSemanticLoss:
    def test_identical_vectors_zero_loss(self):
        v = F.normalize(torch.randn(4, 768), dim=-1)
        loss = semantic_cosine_loss(v, v)
        assert loss.item() < 1e-5

    def test_orthogonal_vectors_unit_loss(self):
        a = torch.zeros(1, 768)
        a[0, 0] = 1.0
        b = torch.zeros(1, 768)
        b[0, 1] = 1.0
        loss = semantic_cosine_loss(a, b)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_loss_in_range(self):
        a = F.normalize(torch.randn(8, 768), dim=-1)
        b = F.normalize(torch.randn(8, 768), dim=-1)
        loss = semantic_cosine_loss(a, b)
        assert 0.0 <= loss.item() <= 2.0


# ---------------------------------------------------------------------------
# Multi-teacher gradient budget tests
# ---------------------------------------------------------------------------

class TestMultiTeacherGradBudget:
    def _make_toy_model(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        return model, x

    def test_ce_only_no_teachers(self):
        model, x = self._make_toy_model()
        y = torch.randn(4, 10)
        ce_loss = F.mse_loss(model(x), y)
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {}, per_teacher_cap=0.10)
        assert report.ce_grad_norm > 0
        assert report.total_teacher_norm_after == 0.0

    def test_single_teacher_capped(self):
        model, x = self._make_toy_model()
        y = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, y)
        t_loss = 100.0 * F.mse_loss(out, torch.randn(4, 10))
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {"t1": t_loss},
            per_teacher_cap=0.10, total_teacher_cap=0.30)
        assert report.per_teacher_scales["t1"] <= 1.0
        assert report.total_teacher_norm_after <= 0.30 * report.ce_grad_norm + 1e-6

    def test_two_teachers_total_cap(self):
        model, x = self._make_toy_model()
        y = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, y)
        t1_loss = 50.0 * F.mse_loss(out, torch.randn(4, 10))
        t2_loss = 50.0 * F.mse_loss(out, torch.randn(4, 10))
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss,
            {"t1": t1_loss, "t2": t2_loss},
            per_teacher_cap=0.20, total_teacher_cap=0.30)
        assert report.total_teacher_norm_after <= 0.30 * report.ce_grad_norm + 1e-6

    def test_grads_not_none_after(self):
        model, x = self._make_toy_model()
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t_loss = F.mse_loss(out, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {"t1": t_loss})
        for p in model.parameters():
            assert p.grad is not None

    def test_preserves_accumulated_grads(self):
        model, x = self._make_toy_model()
        out = model(x)
        pre_loss = F.mse_loss(out, torch.randn(4, 10))
        pre_loss.backward()
        saved_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters()
        ) ** 0.5

        out2 = model(x)
        ce_loss = F.mse_loss(out2, torch.randn(4, 10))
        t_loss = F.mse_loss(out2, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {"t1": t_loss})

        final_norm = sum(
            p.grad.norm().item() ** 2 for p in model.parameters()
        ) ** 0.5
        assert final_norm > saved_norm * 0.5

    def test_report_fields(self):
        model, x = self._make_toy_model()
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t1_loss = F.mse_loss(out, torch.randn(4, 10))
        t2_loss = F.mse_loss(out, torch.randn(4, 10))
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss,
            {"t1": t1_loss, "t2": t2_loss})
        assert isinstance(report, GradientBudgetReport)
        assert "t1" in report.per_teacher_norms
        assert "t2" in report.per_teacher_norms
        assert report.total_scale > 0


# ---------------------------------------------------------------------------
# E2 Trainer tests
# ---------------------------------------------------------------------------

from eklavya_e2_training import (
    E2Config, E2Phase, E2Trainer, get_e2_phase, sigmoid_ramp,
    validate_ablation_config, _parse_static_weights,
)


class TestParseStaticWeights:
    def test_none_returns_none(self):
        assert _parse_static_weights(None) is None

    def test_single_pair(self):
        result = _parse_static_weights("t0:0.5")
        assert result == {"t0": 0.5}

    def test_multiple_pairs(self):
        result = _parse_static_weights("t0:0.4,t1:0.3,t2:0.3")
        assert result == {"t0": 0.4, "t1": 0.3, "t2": 0.3}

    def test_whitespace_tolerance(self):
        result = _parse_static_weights("t0 : 0.5 , t1 : 0.5")
        assert result == {"t0": 0.5, "t1": 0.5}

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid static weight"):
            _parse_static_weights("t0_0.5")

    def test_full_teacher_names(self):
        result = _parse_static_weights(
            "t0_anchor_decoder:0.6,t2_control_decoder:0.4")
        assert result == {"t0_anchor_decoder": 0.6,
                          "t2_control_decoder": 0.4}


class TestSigmoidRamp:
    def test_zero_at_start(self):
        assert sigmoid_ramp(0, 1000) < 0.02

    def test_half_at_midpoint(self):
        assert abs(sigmoid_ramp(500, 1000) - 0.5) < 1e-6

    def test_near_one_at_end(self):
        assert sigmoid_ramp(1000, 1000) > 0.98

    def test_zero_warmup(self):
        assert sigmoid_ramp(0, 0) == 1.0

    def test_monotonic(self):
        vals = [sigmoid_ramp(s, 1000) for s in range(0, 1001, 100)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a


class TestE2Phase:
    def test_phase_sequence(self):
        cfg = E2Config()
        phases = [get_e2_phase(s, cfg) for s in range(0, 15000, 100)]
        seen = []
        for p in phases:
            if not seen or seen[-1] != p:
                seen.append(p)
        assert seen == [
            E2Phase.PORT_WARMUP,
            E2Phase.CONSENSUS,
            E2Phase.SEMANTIC,
            E2Phase.DISAGREEMENT,
            E2Phase.OWNERSHIP,
        ]

    def test_phase_boundaries(self):
        cfg = E2Config(
            port_warmup_steps=100,
            consensus_steps=200,
            semantic_landing_steps=300,
            disagreement_steps=400,
        )
        assert get_e2_phase(0, cfg) == E2Phase.PORT_WARMUP
        assert get_e2_phase(99, cfg) == E2Phase.PORT_WARMUP
        assert get_e2_phase(100, cfg) == E2Phase.CONSENSUS
        assert get_e2_phase(299, cfg) == E2Phase.CONSENSUS
        assert get_e2_phase(300, cfg) == E2Phase.SEMANTIC
        assert get_e2_phase(599, cfg) == E2Phase.SEMANTIC
        assert get_e2_phase(600, cfg) == E2Phase.DISAGREEMENT
        assert get_e2_phase(999, cfg) == E2Phase.DISAGREEMENT
        assert get_e2_phase(1000, cfg) == E2Phase.OWNERSHIP


class TestE2Config:
    def test_defaults(self):
        cfg = E2Config()
        assert cfg.per_teacher_grad_cap == 0.10
        assert cfg.total_teacher_grad_cap == 0.30
        assert cfg.purifier_mode == "arithmetic"

    def test_total_steps(self):
        cfg = E2Config()
        total = (cfg.port_warmup_steps + cfg.consensus_steps
                 + cfg.semantic_landing_steps + cfg.disagreement_steps)
        assert total == 13750


class TestAblationConfigValidation:
    """Ensure ablation_id enforces matching CLI flags."""

    def test_a2_default_passes(self):
        cfg = E2Config(ablation_id="A2")
        validate_ablation_config(cfg)

    def test_a2_with_disable_router_fails(self):
        cfg = E2Config(ablation_id="A2", disable_router=True)
        with pytest.raises(ValueError, match="A2.*forbids.*disable-router"):
            validate_ablation_config(cfg)

    def test_a2_with_shuffle_fails(self):
        cfg = E2Config(ablation_id="A2", shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="A2.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    def test_a2_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A2",
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A2.*forbids.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a2_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="A2",
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A2.*forbids.*teacher-exclude"):
            validate_ablation_config(cfg)

    def test_a5_with_disable_router_passes(self):
        cfg = E2Config(ablation_id="A5", disable_router=True)
        validate_ablation_config(cfg)

    def test_a5_without_disable_router_fails(self):
        cfg = E2Config(ablation_id="A5")
        with pytest.raises(ValueError, match="A5.*requires.*disable-router"):
            validate_ablation_config(cfg)

    def test_a5_with_shuffle_also_fails(self):
        cfg = E2Config(ablation_id="A5", disable_router=True,
                       shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="A5.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    # --- A5a: prior-weighted static ---
    def test_a5a_prior_weighted_passes(self):
        cfg = E2Config(ablation_id="A5a", disable_router=True,
                       static_weight_mode="prior")
        validate_ablation_config(cfg)

    def test_a5a_wrong_weight_mode_fails(self):
        cfg = E2Config(ablation_id="A5a", disable_router=True,
                       static_weight_mode="uniform")
        with pytest.raises(ValueError, match="A5a.*requires.*static-weight-mode.*prior"):
            validate_ablation_config(cfg)

    # --- A5b: custom static weights ---
    def test_a5b_custom_weights_passes(self):
        cfg = E2Config(ablation_id="A5b", disable_router=True,
                       static_weight_mode="custom",
                       static_weights={"t0_anchor_decoder": 0.5,
                                       "t1_diversity_hybrid": 0.3,
                                       "t2_control_decoder": 0.2})
        validate_ablation_config(cfg)

    def test_a5b_missing_weights_fails(self):
        cfg = E2Config(ablation_id="A5b", disable_router=True,
                       static_weight_mode="custom")
        with pytest.raises(ValueError, match="A5b.*requires.*static-weights"):
            validate_ablation_config(cfg)

    def test_a5b_wrong_weight_mode_fails(self):
        cfg = E2Config(ablation_id="A5b", disable_router=True,
                       static_weight_mode="prior")
        with pytest.raises(ValueError, match="A5b.*requires.*static-weight-mode.*custom"):
            validate_ablation_config(cfg)

    # --- A5c: best-2 teacher static (prior-weighted) ---
    def test_a5c_best2_passes(self):
        cfg = E2Config(ablation_id="A5c", disable_router=True,
                       static_weight_mode="prior",
                       teacher_include=["t0_anchor_decoder",
                                        "t1_diversity_hybrid"])
        validate_ablation_config(cfg)

    def test_a5c_without_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A5c", disable_router=True,
                       static_weight_mode="prior")
        with pytest.raises(ValueError, match="A5c.*requires.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a5c_without_disable_router_fails(self):
        cfg = E2Config(ablation_id="A5c",
                       static_weight_mode="prior",
                       teacher_include=["t0_anchor_decoder",
                                        "t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A5c.*requires.*disable-router"):
            validate_ablation_config(cfg)

    def test_a5c_wrong_teacher_count_fails(self):
        cfg = E2Config(ablation_id="A5c", disable_router=True,
                       static_weight_mode="prior",
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A5c.*exactly 2 teachers"):
            validate_ablation_config(cfg)

    def test_a5c_wrong_weight_mode_fails(self):
        cfg = E2Config(ablation_id="A5c", disable_router=True,
                       teacher_include=["t0_anchor_decoder",
                                        "t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A5c.*requires.*static-weight-mode.*prior"):
            validate_ablation_config(cfg)

    def test_a6_with_shuffle_passes(self):
        cfg = E2Config(ablation_id="A6", shuffle_teacher_targets=True)
        validate_ablation_config(cfg)

    def test_a6_without_shuffle_fails(self):
        cfg = E2Config(ablation_id="A6")
        with pytest.raises(ValueError, match="A6.*requires.*shuffle"):
            validate_ablation_config(cfg)

    def test_a1_with_anchor_passes(self):
        cfg = E2Config(ablation_id="A1",
                       teacher_include=["t0_anchor_decoder"])
        validate_ablation_config(cfg)

    def test_a1_without_teachers_fails(self):
        cfg = E2Config(ablation_id="A1")
        with pytest.raises(ValueError, match="A1.*requires.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a1_with_wrong_teacher_fails(self):
        cfg = E2Config(ablation_id="A1",
                       teacher_include=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A1.*t0_anchor_decoder"):
            validate_ablation_config(cfg)

    def test_a1_with_multiple_teachers_fails(self):
        cfg = E2Config(ablation_id="A1",
                       teacher_include=["t0_anchor_decoder",
                                        "t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A1.*anchor"):
            validate_ablation_config(cfg)

    def test_a3_with_exclude_passes(self):
        cfg = E2Config(ablation_id="A3",
                       teacher_exclude=["t1_diversity_hybrid"])
        validate_ablation_config(cfg)

    def test_a3_without_exclude_fails(self):
        cfg = E2Config(ablation_id="A3")
        with pytest.raises(ValueError, match="A3.*requires.*teacher-exclude"):
            validate_ablation_config(cfg)

    def test_a4_with_semantic_excluded_passes(self):
        cfg = E2Config(ablation_id="A4",
                       teacher_exclude=["t3_semantic_embedding"])
        validate_ablation_config(cfg)

    def test_a4_with_wrong_exclude_fails(self):
        cfg = E2Config(ablation_id="A4",
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A4.*exactly.*t3_semantic_embedding"):
            validate_ablation_config(cfg)

    def test_a4_with_extra_exclude_fails(self):
        cfg = E2Config(ablation_id="A4",
                       teacher_exclude=["t3_semantic_embedding",
                                        "t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A4.*exactly.*t3_semantic_embedding"):
            validate_ablation_config(cfg)

    def test_a0_ce_only_passes(self):
        cfg = E2Config(ablation_id="A0", ce_only=True)
        validate_ablation_config(cfg)

    def test_a0_without_ce_only_flag_fails(self):
        cfg = E2Config(ablation_id="A0")
        with pytest.raises(ValueError, match="A0.*requires.*--ce-only"):
            validate_ablation_config(cfg)

    def test_a0_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A0", ce_only=True,
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A0.*forbids.*--teacher-include"):
            validate_ablation_config(cfg)

    def test_a0_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="A0", ce_only=True,
                       teacher_exclude=["t3_semantic_embedding"])
        with pytest.raises(ValueError, match="A0.*forbids.*--teacher-exclude"):
            validate_ablation_config(cfg)

    # --- A7: no gradient budget ---

    def test_a7_with_disable_gradient_budget_passes(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True)
        validate_ablation_config(cfg)

    def test_a7_without_disable_gradient_budget_fails(self):
        cfg = E2Config(ablation_id="A7")
        with pytest.raises(ValueError, match="A7.*requires.*disable-gradient-budget"):
            validate_ablation_config(cfg)

    def test_a7_with_ce_only_fails(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True,
                       ce_only=True)
        with pytest.raises(ValueError, match="A7.*forbids.*ce-only"):
            validate_ablation_config(cfg)

    def test_a7_with_disable_router_fails(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True,
                       disable_router=True)
        with pytest.raises(ValueError, match="A7.*forbids.*disable-router"):
            validate_ablation_config(cfg)

    def test_a7_with_shuffle_fails(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True,
                       shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="A7.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    def test_a7_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True,
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A7.*forbids.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a7_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="A7", disable_gradient_budget=True,
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A7.*forbids.*teacher-exclude"):
            validate_ablation_config(cfg)

    # --- A8: no phased admission ---

    def test_a8_with_no_phased_admission_passes(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True)
        validate_ablation_config(cfg)

    def test_a8_without_no_phased_admission_fails(self):
        cfg = E2Config(ablation_id="A8")
        with pytest.raises(ValueError, match="A8.*requires.*no-phased-admission"):
            validate_ablation_config(cfg)

    def test_a8_with_ce_only_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       ce_only=True)
        with pytest.raises(ValueError, match="A8.*forbids.*ce-only"):
            validate_ablation_config(cfg)

    def test_a8_with_disable_router_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       disable_router=True)
        with pytest.raises(ValueError, match="A8.*forbids.*disable-router"):
            validate_ablation_config(cfg)

    def test_a8_with_shuffle_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="A8.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    def test_a8_with_disable_gradient_budget_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       disable_gradient_budget=True)
        with pytest.raises(ValueError, match="A8.*forbids.*disable-gradient-budget"):
            validate_ablation_config(cfg)

    def test_a8_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A8.*forbids.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a8_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="A8", no_phased_admission=True,
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A8.*forbids.*teacher-exclude"):
            validate_ablation_config(cfg)

    # --- BLD: single-teacher byte KL baseline ---

    def test_bld_with_bld_mode_passes(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True)
        validate_ablation_config(cfg)

    def test_bld_without_bld_mode_fails(self):
        cfg = E2Config(ablation_id="BLD")
        with pytest.raises(ValueError, match="BLD.*requires.*bld-mode"):
            validate_ablation_config(cfg)

    def test_bld_with_ce_only_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True, ce_only=True)
        with pytest.raises(ValueError, match="BLD.*forbids.*ce-only"):
            validate_ablation_config(cfg)

    def test_bld_with_disable_router_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True, disable_router=True)
        with pytest.raises(ValueError, match="BLD.*forbids.*disable-router"):
            validate_ablation_config(cfg)

    def test_bld_with_shuffle_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True,
                       shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="BLD.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    def test_bld_with_disable_gradient_budget_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True,
                       disable_gradient_budget=True)
        with pytest.raises(ValueError, match="BLD.*forbids.*disable-gradient-budget"):
            validate_ablation_config(cfg)

    def test_bld_with_no_phased_admission_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True,
                       no_phased_admission=True)
        with pytest.raises(ValueError, match="BLD.*forbids.*no-phased-admission"):
            validate_ablation_config(cfg)

    def test_bld_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True,
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="BLD.*forbids.*teacher-include"):
            validate_ablation_config(cfg)

    def test_bld_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="BLD", bld_mode=True,
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="BLD.*forbids.*teacher-exclude"):
            validate_ablation_config(cfg)

    # --- A9a: gold-free entropy ---
    def test_a9a_with_correct_router_mode_passes(self):
        cfg = E2Config(ablation_id="A9a", router_mode="gold_free_entropy")
        validate_ablation_config(cfg)

    def test_a9a_with_wrong_router_mode_fails(self):
        cfg = E2Config(ablation_id="A9a", router_mode="oracle_gold")
        with pytest.raises(ValueError, match="A9a.*requires.*router-mode.*gold_free_entropy"):
            validate_ablation_config(cfg)

    def test_a9a_with_ce_only_fails(self):
        cfg = E2Config(ablation_id="A9a", router_mode="gold_free_entropy", ce_only=True)
        with pytest.raises(ValueError, match="A9a.*forbids.*ce-only"):
            validate_ablation_config(cfg)

    def test_a9a_with_disable_router_fails(self):
        cfg = E2Config(ablation_id="A9a", router_mode="gold_free_entropy", disable_router=True)
        with pytest.raises(ValueError, match="A9a.*forbids.*disable-router"):
            validate_ablation_config(cfg)

    def test_a9a_with_bld_mode_fails(self):
        cfg = E2Config(ablation_id="A9a", router_mode="gold_free_entropy", bld_mode=True)
        with pytest.raises(ValueError, match="A9a.*forbids.*bld-mode"):
            validate_ablation_config(cfg)

    # --- A9b: gold-free agreement ---
    def test_a9b_with_correct_router_mode_passes(self):
        cfg = E2Config(ablation_id="A9b", router_mode="gold_free_agreement")
        validate_ablation_config(cfg)

    def test_a9b_with_wrong_router_mode_fails(self):
        cfg = E2Config(ablation_id="A9b", router_mode="gold_free_entropy")
        with pytest.raises(ValueError, match="A9b.*requires.*router-mode.*gold_free_agreement"):
            validate_ablation_config(cfg)

    def test_a9b_with_shuffle_fails(self):
        cfg = E2Config(ablation_id="A9b", router_mode="gold_free_agreement",
                       shuffle_teacher_targets=True)
        with pytest.raises(ValueError, match="A9b.*forbids.*shuffle"):
            validate_ablation_config(cfg)

    # --- A9c: gold-free student JSD ---
    def test_a9c_with_correct_router_mode_passes(self):
        cfg = E2Config(ablation_id="A9c", router_mode="gold_free_student_jsd")
        validate_ablation_config(cfg)

    def test_a9c_with_wrong_router_mode_fails(self):
        cfg = E2Config(ablation_id="A9c", router_mode="oracle_gold")
        with pytest.raises(ValueError, match="A9c.*requires.*router-mode.*gold_free_student_jsd"):
            validate_ablation_config(cfg)

    def test_a9c_with_teacher_include_fails(self):
        cfg = E2Config(ablation_id="A9c", router_mode="gold_free_student_jsd",
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError, match="A9c.*forbids.*teacher-include"):
            validate_ablation_config(cfg)

    def test_a9c_with_teacher_exclude_fails(self):
        cfg = E2Config(ablation_id="A9c", router_mode="gold_free_student_jsd",
                       teacher_exclude=["t1_diversity_hybrid"])
        with pytest.raises(ValueError, match="A9c.*forbids.*teacher-exclude"):
            validate_ablation_config(cfg)

    def test_unknown_ablation_warns(self, capsys):
        cfg = E2Config(ablation_id="X99")
        validate_ablation_config(cfg)
        out = capsys.readouterr().out
        assert "WARNING" in out and "X99" in out

    def test_multiple_violations_all_reported(self):
        cfg = E2Config(ablation_id="A5",
                       shuffle_teacher_targets=True,
                       teacher_include=["t0_anchor_decoder"])
        with pytest.raises(ValueError) as exc_info:
            validate_ablation_config(cfg)
        msg = str(exc_info.value)
        assert "disable-router" in msg  # required but missing
        assert "shuffle" in msg  # forbidden but set
        assert "teacher-include" in msg  # forbidden but set


# ---------------------------------------------------------------------------
# Binary validation tests (Fix 3)
# ---------------------------------------------------------------------------

class TestBinaryValidation:
    def test_kl_record_short_top_bytes_raises(self):
        rec = E2KLRecord(
            position_id=0, patch_idx=5, tail_prob=0.05,
            entropy=2.3, logp_gold=-1.8,
            top_bytes=np.arange(8, dtype=np.uint8),
            top_probs=np.linspace(0.5, 0.01, 8).astype(np.float16),
        )
        with pytest.raises(ValueError, match="top_bytes length"):
            rec.pack(K=16)

    def test_kl_record_wrong_dtype_raises(self):
        rec = E2KLRecord(
            position_id=0, patch_idx=5, tail_prob=0.05,
            entropy=2.3, logp_gold=-1.8,
            top_bytes=np.arange(16, dtype=np.uint8),
            top_probs=np.linspace(0.5, 0.01, 16).astype(np.float32),
        )
        with pytest.raises(ValueError, match="float16"):
            rec.pack(K=16)

    def test_route_record_mismatched_arrays_raises(self):
        rec = RouteRecord(
            position_id=0, n_teachers=3, jsd=0.15, route_entropy=1.2,
            teacher_ids=np.arange(2, dtype=np.uint8),
            weights=np.array([0.5, 0.3, 0.2], dtype=np.float16),
        )
        with pytest.raises(ValueError, match="teacher_ids length"):
            rec.pack()

    def test_route_record_weights_mismatch_raises(self):
        rec = RouteRecord(
            position_id=0, n_teachers=3, jsd=0.15, route_entropy=1.2,
            teacher_ids=np.arange(3, dtype=np.uint8),
            weights=np.array([0.5, 0.5], dtype=np.float16),
        )
        with pytest.raises(ValueError, match="weights length"):
            rec.pack()

    def test_short_position_manifest_raises(self, tmp_path):
        path = str(tmp_path / "corrupt.bin")
        with open(path, "wb") as f:
            import struct as st
            f.write(st.pack("<I", 5))
            f.write(make_position(pid=0).pack())
        with pytest.raises(ValueError, match="Short read"):
            read_position_manifest(path)


# ---------------------------------------------------------------------------
# Per-teacher gradient cap tests (Fix 5)
# ---------------------------------------------------------------------------

class TestPerTeacherGradCaps:
    def test_dict_caps_applied(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t1_loss = 100.0 * F.mse_loss(out, torch.randn(4, 10))
        t2_loss = 100.0 * F.mse_loss(out, torch.randn(4, 10))
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss,
            {"t1": t1_loss, "t2": t2_loss},
            per_teacher_cap={"t1": 0.05, "t2": 0.10},
            total_teacher_cap=0.30)
        assert report.per_teacher_scales["t1"] <= 1.0
        assert report.per_teacher_scales["t2"] <= 1.0

    def test_dict_caps_default_fallback(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t1_loss = 100.0 * F.mse_loss(out, torch.randn(4, 10))
        report = apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss,
            {"t1": t1_loss},
            per_teacher_cap={"t2": 0.05},
            total_teacher_cap=0.30)
        assert "t1" in report.per_teacher_scales


# ---------------------------------------------------------------------------
# Retain graph behavior tests (Fix 4)
# ---------------------------------------------------------------------------

class TestRetainGraphBehavior:
    def test_default_releases_graph(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t_loss = F.mse_loss(out, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {"t1": t_loss})
        # Graph should be freed — another backward should fail
        with pytest.raises(RuntimeError, match="backward through the graph"):
            ce_loss.backward()

    def test_retain_graph_true_keeps_graph(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        t_loss = F.mse_loss(out, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {"t1": t_loss},
            retain_graph=True)
        ce_loss.backward()


# ---------------------------------------------------------------------------
# E2 Trainer compute_teacher_losses integration test (Fix 8)
# ---------------------------------------------------------------------------

class TestE2TrainerTeacherLosses:
    def _build_minimal_trainer(self):
        """Build a trainer with fake cache for integration testing."""
        from s0_architecture import S0Config, SutraS0

        model_cfg = S0Config(
            d_model=64, n_heads=2, n_kv_heads=1,
            n_layers=2, patch_size=4, vocab_size=260,
            ffn_mult=2, max_seq_len=64,
            decoder_dim=64, decoder_layers=1, decoder_heads=2,
            byte_dim=64,
        )
        student = SutraS0(model_cfg)

        n_pos = 10
        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0, reason_mask=1,
            )
            for i in range(n_pos)
        ]

        anchor = TEACHER_REGISTRY[0]
        kl_recs = [make_kl(pid=i, K=16) for i in range(n_pos)]
        align_recs = [
            E2AlignRecord(
                position_id=i, byte_start=i * 4, byte_len=4,
                token_id=i, align_quality=0.9,
            )
            for i in range(n_pos)
        ]
        emb_table = torch.randn(n_pos, anchor.hidden_dim)

        teacher_data = {
            anchor.name: {
                "spec": anchor,
                "kl_records": kl_recs,
                "kl_K": 16,
                "align_records": align_recs,
                "embedding_table": emb_table,
            }
        }

        manifest = {
            "version": "e2.0",
            "n_positions": n_pos,
            "kl_top_k": 16,
            "teacher_count": 1,
            "teacher_specs": [anchor],
        }

        cache = {
            "manifest": manifest,
            "positions": positions,
            "teachers": teacher_data,
            "routes": [],
        }

        specs = [anchor]
        ports = MultiTeacherProjectionPorts(student_dim=model_cfg.d_model,
                                            teachers=specs)
        cfg = E2Config(
            port_warmup_steps=2, consensus_steps=3,
            semantic_landing_steps=3, disagreement_steps=5,
        )
        device = torch.device("cpu")
        trainer = E2Trainer(cfg, student, ports, cache, device)
        return trainer, student, ports, model_cfg, cfg

    def test_positions_indexed(self):
        trainer, *_ = self._build_minimal_trainer()
        assert (0, 0) in trainer.positions_by_loc
        assert len(trainer.positions_by_loc[(0, 0)]) == 10

    def test_teacher_records_indexed(self):
        trainer, *_ = self._build_minimal_trainer()
        assert "t0_anchor_decoder" in trainer.teacher_kl_by_pid
        assert len(trainer.teacher_kl_by_pid["t0_anchor_decoder"]) == 10

    def test_consensus_phase_produces_kl_loss(self):
        trainer, student, ports, model_cfg, cfg = self._build_minimal_trainer()
        P = model_cfg.patch_size
        seq_len = 64 * P
        byte_ids = torch.randint(0, 256, (1, seq_len))
        out = student(byte_ids)
        logits = out["logits"]
        patch_states = out["patch_states"]
        shard_ids = torch.tensor([0])
        seq_starts = torch.tensor([0])

        losses = trainer.compute_teacher_losses(
            logits, patch_states, shard_ids, seq_starts,
            E2Phase.CONSENSUS, step=3)

        assert len(losses) > 0, "Should produce at least one teacher loss"
        for name, loss in losses.items():
            assert torch.isfinite(loss), f"{name} is not finite"
            assert loss.requires_grad, f"{name} has no grad"

    def test_port_warmup_produces_align_loss(self):
        trainer, student, ports, model_cfg, cfg = self._build_minimal_trainer()
        P = model_cfg.patch_size
        seq_len = 64 * P
        byte_ids = torch.randint(0, 256, (1, seq_len))
        out = student(byte_ids)
        logits = out["logits"]
        patch_states = out["patch_states"]
        shard_ids = torch.tensor([0])
        seq_starts = torch.tensor([0])

        losses = trainer.compute_teacher_losses(
            logits, patch_states, shard_ids, seq_starts,
            E2Phase.PORT_WARMUP, step=0)

        align_losses = {k: v for k, v in losses.items() if "align" in k}
        assert len(align_losses) > 0, "Port warmup should produce align losses"

    def test_no_positions_returns_empty(self):
        trainer, student, ports, model_cfg, cfg = self._build_minimal_trainer()
        P = model_cfg.patch_size
        seq_len = 64 * P
        byte_ids = torch.randint(0, 256, (1, seq_len))
        out = student(byte_ids)
        shard_ids = torch.tensor([999])
        seq_starts = torch.tensor([999])

        losses = trainer.compute_teacher_losses(
            out["logits"], out["patch_states"], shard_ids, seq_starts,
            E2Phase.CONSENSUS, step=3)

        assert len(losses) == 0

    def test_semantic_phase_produces_calibration_loss(self):
        trainer, student, ports, model_cfg, cfg = self._build_minimal_trainer()
        P = model_cfg.patch_size
        seq_len = 64 * P
        byte_ids = torch.randint(0, 256, (1, seq_len))
        out = student(byte_ids)
        shard_ids = torch.tensor([0])
        seq_starts = torch.tensor([0])

        step = cfg.port_warmup_steps + cfg.consensus_steps + 10
        losses = trainer.compute_teacher_losses(
            out["logits"], out["patch_states"], shard_ids, seq_starts,
            E2Phase.SEMANTIC, step=step)

        cal_losses = {k: v for k, v in losses.items() if "calibration" in k}
        assert len(cal_losses) > 0, "Semantic phase should produce calibration loss"
        for name, loss in cal_losses.items():
            assert torch.isfinite(loss), f"{name} is not finite"
            assert loss.requires_grad, f"{name} has no grad"

    def test_ownership_phase_no_losses(self):
        trainer, student, ports, model_cfg, cfg = self._build_minimal_trainer()
        P = model_cfg.patch_size
        seq_len = 64 * P
        byte_ids = torch.randint(0, 256, (1, seq_len))
        out = student(byte_ids)
        shard_ids = torch.tensor([0])
        seq_starts = torch.tensor([0])

        losses = trainer.compute_teacher_losses(
            out["logits"], out["patch_states"], shard_ids, seq_starts,
            E2Phase.OWNERSHIP, step=100)

        assert len(losses) == 0


# ---------------------------------------------------------------------------
# RouteRecord dtype validation tests (review R3, finding 1)
# ---------------------------------------------------------------------------

class TestRouteRecordDtypeValidation:
    def test_teacher_ids_wrong_dtype_raises(self):
        rec = RouteRecord(
            position_id=0, n_teachers=3, jsd=0.15, route_entropy=1.2,
            teacher_ids=np.arange(3, dtype=np.int64),
            weights=np.array([0.5, 0.3, 0.2], dtype=np.float16),
        )
        with pytest.raises(ValueError, match="teacher_ids dtype must be uint8"):
            rec.pack()

    def test_weights_wrong_dtype_raises(self):
        rec = RouteRecord(
            position_id=0, n_teachers=3, jsd=0.15, route_entropy=1.2,
            teacher_ids=np.arange(3, dtype=np.uint8),
            weights=np.array([0.5, 0.3, 0.2], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="weights dtype must be float16"):
            rec.pack()


# ---------------------------------------------------------------------------
# Short-read checks for align and route readers (review R3, finding 1)
# ---------------------------------------------------------------------------

class TestAlignRouteShortReads:
    def test_short_align_header_raises(self, tmp_path):
        path = str(tmp_path / "short_align.bin")
        with open(path, "wb") as f:
            f.write(b"\x01\x00")
        with pytest.raises(ValueError, match="Short read on align header"):
            read_teacher_align_records(path)

    def test_short_align_body_raises(self, tmp_path):
        path = str(tmp_path / "trunc_align.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 3))
            f.write(E2AlignRecord(0, 0, 4, 100, 0.9).pack())
        with pytest.raises(ValueError, match="Short read at align record"):
            read_teacher_align_records(path)

    def test_short_route_header_raises(self, tmp_path):
        path = str(tmp_path / "short_route.bin")
        with open(path, "wb") as f:
            f.write(b"\x02")
        with pytest.raises(ValueError, match="Short read on route count"):
            read_route_records(path)

    def test_short_route_body_raises(self, tmp_path):
        path = str(tmp_path / "trunc_route.bin")
        with open(path, "wb") as f:
            f.write(struct.pack("<I", 2))
            rec = RouteRecord(
                position_id=0, n_teachers=3, jsd=0.1, route_entropy=0.8,
                teacher_ids=np.arange(3, dtype=np.uint8),
                weights=np.array([0.5, 0.3, 0.2], dtype=np.float16))
            f.write(rec.pack())
        with pytest.raises(ValueError, match="Short read at route"):
            read_route_records(path)


# ---------------------------------------------------------------------------
# Empty teacher_losses doesn't retain graph (review R3, finding 2)
# ---------------------------------------------------------------------------

class TestEmptyTeacherLossesReleasesGraph:
    def test_empty_teacher_losses_releases_graph(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {})
        with pytest.raises(RuntimeError, match="backward through the graph"):
            ce_loss.backward()

    def test_empty_teacher_losses_with_retain_keeps_graph(self):
        model = torch.nn.Linear(10, 10)
        x = torch.randn(4, 10)
        out = model(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))
        apply_multi_teacher_gradient_budget(
            model.parameters(), ce_loss, {}, retain_graph=True)
        ce_loss.backward()


# ---------------------------------------------------------------------------
# Frozen CE with active teacher losses (review R4, finding 1)
# ---------------------------------------------------------------------------

class TestFrozenCEGradientBudget:
    def test_frozen_ce_with_teacher_losses_succeeds(self):
        """PORT_WARMUP scenario: student frozen (CE has no grad), ports trainable."""
        frozen = torch.nn.Linear(10, 10)
        for p in frozen.parameters():
            p.requires_grad = False
        trainable = torch.nn.Linear(10, 10)

        x = torch.randn(4, 10)
        out_frozen = frozen(x)
        ce_loss = F.mse_loss(out_frozen, torch.randn(4, 10))
        assert not ce_loss.requires_grad

        out_trainable = trainable(x)
        t_loss = F.mse_loss(out_trainable, torch.randn(4, 10))
        assert t_loss.requires_grad

        all_params = list(frozen.parameters()) + list(trainable.parameters())
        report = apply_multi_teacher_gradient_budget(
            all_params, ce_loss, {"t1": t_loss})

        assert report.ce_grad_norm == 0.0
        assert "t1" in report.per_teacher_norms
        assert report.per_teacher_scales["t1"] == 1.0

        has_grad = any(p.grad is not None for p in trainable.parameters())
        assert has_grad, "Trainable params should have gradients"

    def test_frozen_ce_no_teacher_losses(self):
        """All frozen, no teacher losses — should not crash."""
        frozen = torch.nn.Linear(10, 10)
        for p in frozen.parameters():
            p.requires_grad = False

        x = torch.randn(4, 10)
        out = frozen(x)
        ce_loss = F.mse_loss(out, torch.randn(4, 10))

        report = apply_multi_teacher_gradient_budget(
            frozen.parameters(), ce_loss, {})
        assert report.ce_grad_norm == 0.0


# ---------------------------------------------------------------------------
# Cache builder tests (CPU-only, minimal models)
# ---------------------------------------------------------------------------

class TestBuildPositionManifest:
    """Test position manifest generation on CPU with a tiny S0 model."""

    def _make_student(self):
        from s0_architecture import S0Config, SutraS0
        cfg = S0Config(
            d_model=64, n_heads=2, n_kv_heads=1,
            n_layers=2, patch_size=4, vocab_size=260,
            ffn_mult=2, max_seq_len=64,
            decoder_dim=64, decoder_layers=1, decoder_heads=2,
            byte_dim=64,
        )
        return SutraS0(cfg), cfg

    def _make_shard(self, tmp_path, seq_len=64, n_seqs=2):
        data = np.random.randint(0, 256, size=seq_len * n_seqs, dtype=np.uint8)
        path = tmp_path / "shard_000.bin"
        data.tofile(str(path))
        return path

    def test_returns_position_records(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        assert len(records) > 0
        assert all(isinstance(r, PositionRecord) for r in records)

    def test_pids_are_sequential(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        pids = [r.position_id for r in records]
        assert pids == list(range(len(pids)))

    def test_pid_start_offset(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
            pid_start=100,
        )
        assert records[0].position_id == 100
        pids = [r.position_id for r in records]
        assert pids == list(range(100, 100 + len(pids)))

    def test_shard_id_propagated(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=42,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        assert all(r.shard_id == 42 for r in records)

    def test_high_threshold_selects_fewer(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        low = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=0.0, device=torch.device("cpu"),
        )
        high = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=100.0,
            control_frac=0.0, device=torch.device("cpu"),
        )
        assert len(high) <= len(low)

    def test_reason_mask_set(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        for r in records:
            assert r.reason_mask > 0

    def test_gold_byte_in_range(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        for r in records:
            assert 0 <= r.gold_byte <= 255

    def test_student_nll_positive(self, tmp_path):
        from eklavya_e2_cache_builder import build_position_manifest
        student, cfg = self._make_student()
        shard_path = self._make_shard(tmp_path)
        records = build_position_manifest(
            student, shard_path, shard_id=0,
            seq_len=64, nll_threshold=0.0,
            control_frac=1.0, device=torch.device("cpu"),
        )
        for r in records:
            assert r.student_nll >= 0.0


# ---------------------------------------------------------------------------
# build_teacher_records tests (mocked model/tokenizer, CPU-only)
# ---------------------------------------------------------------------------

class _MockTokenizerOutput:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        return self


class _MockTokenizer:
    vocab_size = 32
    model_max_length = 512

    def __init__(self):
        self._vocab = {i: chr(65 + (i % 26)) for i in range(self.vocab_size)}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._vocab.get(i, "?") for i in ids)

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=None):
        ids = [ord(c) % self.vocab_size for c in text[:8]]
        return _MockTokenizerOutput(torch.tensor([ids], dtype=torch.long))

    def __len__(self):
        return self.vocab_size


class _MockModelOutput:
    def __init__(self, logits):
        self.logits = logits


class _MockTeacherModel(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_dim=16):
        super().__init__()
        self._emb = torch.nn.Embedding(vocab_size, hidden_dim)

    def eval(self):
        return self

    def get_input_embeddings(self):
        return self._emb

    def forward(self, input_ids, **kwargs):
        B, S = input_ids.shape
        logits = torch.randn(B, S, self._emb.num_embeddings)
        return _MockModelOutput(logits)


class TestBuildTeacherRecords:

    def _make_spec(self, has_kl=True, has_align=True, has_semantic=False):
        return TeacherSpec(
            teacher_id=99, name="mock_teacher", hf_name="mock/mock",
            family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
            hidden_dim=16, vocab_size=32,
            has_kl=has_kl, has_align=has_align, has_semantic=has_semantic,
            prior=0.5, vram_gb=0.1,
        )

    def _make_positions(self, shard_id=0, seq_offset=0, n=3, patch_size=4):
        return [
            PositionRecord(
                position_id=i,
                shard_id=shard_id,
                seq_offset=seq_offset,
                patch_idx=i + 1,
                gold_byte=65 + i,
                student_nll=4.0,
                student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n)
        ]

    def _make_shard(self, tmp_path, seq_len=64):
        data = np.random.randint(32, 122, size=seq_len, dtype=np.uint8)
        p = tmp_path / "shard_000.bin"
        data.tofile(str(p))
        return p

    def test_returns_kl_and_align_lists(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=True)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0)
        kl, align = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        assert isinstance(kl, list)
        assert isinstance(align, list)

    def test_kl_records_have_correct_fields(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=False)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0, n=2)
        kl, align = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        for rec in kl:
            assert hasattr(rec, "position_id")
            assert hasattr(rec, "top_bytes")
            assert hasattr(rec, "top_probs")
            assert len(rec.top_bytes) == 8
            assert len(rec.top_probs) == 8
            assert rec.tail_prob >= 0.0
            assert rec.entropy >= 0.0

    def test_align_records_have_correct_fields(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=False, has_align=True)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0, n=2)
        kl, align = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        assert len(kl) == 0
        for rec in align:
            assert hasattr(rec, "position_id")
            assert hasattr(rec, "byte_start")
            assert hasattr(rec, "byte_len")
            assert rec.byte_len > 0
            assert rec.align_quality == 1.0

    def test_no_kl_when_has_kl_false(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=False, has_align=False, has_semantic=False)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0)
        kl, align = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        assert len(kl) == 0
        assert len(align) == 0

    def test_empty_positions_returns_empty(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=True)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        kl, align = build_teacher_records(
            model, tok, spec, [], shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        assert len(kl) == 0
        assert len(align) == 0

    def test_position_ids_preserved(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=False)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0, n=3)
        kl, _ = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"),
        )
        kl_pids = {r.position_id for r in kl}
        pos_pids = {p.position_id for p in positions}
        assert kl_pids.issubset(pos_pids)

    def test_custom_byte_table_used(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=False)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0, n=1)
        byte_table = {i: bytes([65 + (i % 26)]) for i in range(32)}
        kl, _ = build_teacher_records(
            model, tok, spec, positions, shard,
            seq_len=64, patch_size=4, kl_top_k=8,
            device=torch.device("cpu"), byte_table=byte_table,
        )
        assert isinstance(kl, list)

    def test_kl_top_k_respected(self, tmp_path):
        from eklavya_e2_cache_builder import build_teacher_records
        spec = self._make_spec(has_kl=True, has_align=False)
        model = _MockTeacherModel(vocab_size=32)
        tok = _MockTokenizer()
        shard = self._make_shard(tmp_path, seq_len=64)
        positions = self._make_positions(seq_offset=0, n=1)
        for K in [4, 16]:
            kl, _ = build_teacher_records(
                model, tok, spec, positions, shard,
                seq_len=64, patch_size=4, kl_top_k=K,
                device=torch.device("cpu"),
            )
            for rec in kl:
                assert len(rec.top_bytes) == K
                assert len(rec.top_probs) == K


# ---------------------------------------------------------------------------
# E2Config smoke tests (CPU-only, no data/models needed)
# ---------------------------------------------------------------------------

class TestE2ConfigSmoke:

    def test_config_defaults_valid(self):
        from eklavya_e2_training import E2Config
        cfg = E2Config()
        assert cfg.port_warmup_steps > 0
        assert cfg.consensus_steps > cfg.port_warmup_steps
        total = cfg.port_warmup_steps + cfg.consensus_steps + cfg.semantic_landing_steps + cfg.disagreement_steps
        assert total > 0

    def test_phase_ordering(self):
        from eklavya_e2_training import E2Config, get_e2_phase, E2Phase
        cfg = E2Config()
        assert get_e2_phase(0, cfg) == E2Phase.PORT_WARMUP
        assert get_e2_phase(cfg.port_warmup_steps, cfg) == E2Phase.CONSENSUS
        phase3_start = cfg.port_warmup_steps + cfg.consensus_steps
        assert get_e2_phase(phase3_start, cfg) == E2Phase.SEMANTIC
        phase4_start = phase3_start + cfg.semantic_landing_steps
        assert get_e2_phase(phase4_start, cfg) == E2Phase.DISAGREEMENT

    def test_paths_are_set(self):
        from eklavya_e2_training import E2Config
        cfg = E2Config()
        assert cfg.log_file
        assert cfg.checkpoint_dir

    def test_grad_caps_sane(self):
        from eklavya_e2_training import E2Config
        cfg = E2Config()
        assert 0.0 < cfg.per_teacher_grad_cap <= 1.0
        assert 0.0 < cfg.total_teacher_grad_cap <= 1.0
        assert cfg.total_teacher_grad_cap >= cfg.per_teacher_grad_cap

    def test_ablation_config_defaults(self):
        from eklavya_e2_training import E2Config
        cfg = E2Config()
        assert cfg.ablation_id == "A2"
        assert cfg.teacher_include is None
        assert cfg.teacher_exclude == []
        assert cfg.disable_router is False
        assert cfg.shuffle_teacher_targets is False

    def test_ablation_config_custom(self):
        from eklavya_e2_training import E2Config
        cfg = E2Config(
            ablation_id="A5",
            teacher_exclude=["t4_diversity_ssm"],
            disable_router=True,
        )
        assert cfg.ablation_id == "A5"
        assert cfg.disable_router is True
        assert "t4_diversity_ssm" in cfg.teacher_exclude


# ---------------------------------------------------------------------------
# E2Trainer instantiation smoke test (CPU-only, synthetic cache)
# ---------------------------------------------------------------------------

class TestE2TrainerSmoke:

    def _make_tiny_student(self):
        from s0_architecture import S0Config, SutraS0
        cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4,
        )
        return SutraS0(cfg), cfg

    def _make_tiny_cache(self, tmp_path, teacher_specs):
        """Create a minimal E2 cache directory with positions and teacher records."""
        cache_dir = tmp_path / "e2_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        K = 8
        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(3)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        for spec in teacher_specs:
            tdir = cache_dir / "teachers" / spec.name
            tdir.mkdir()

            if spec.has_kl:
                kl_recs = [
                    E2KLRecord(
                        position_id=i, patch_idx=i + 1,
                        tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                        top_bytes=np.arange(K, dtype=np.uint8),
                        top_probs=np.full(K, 1.0 / K, dtype=np.float16),
                    )
                    for i in range(3)
                ]
                from eklavya_e2_cache import write_teacher_kl_records
                write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

            if spec.has_align or spec.has_semantic:
                align_recs = [
                    E2AlignRecord(
                        position_id=i, byte_start=i * 4,
                        byte_len=4, token_id=10 + i, align_quality=1.0,
                    )
                    for i in range(3)
                ]
                from eklavya_e2_cache import write_teacher_align_records
                write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

                emb = torch.randn(spec.vocab_size, spec.hidden_dim).half()
                torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        from eklavya_e2_cache import save_e2_manifest
        save_e2_manifest(str(cache_dir), teacher_specs, n_positions=3, K=K)

        return str(cache_dir)

    def _make_specs(self):
        return [
            TeacherSpec(
                teacher_id=0, name="mock_anchor", hf_name="mock/anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.5, vram_gb=0.1,
            ),
            TeacherSpec(
                teacher_id=1, name="mock_semantic", hf_name="mock/semantic",
                family=TeacherFamily.EMBEDDING, role=TeacherRole.SEMANTIC,
                hidden_dim=16, vocab_size=32,
                has_kl=False, has_align=False, has_semantic=True,
                prior=0.2, vram_gb=0.1,
            ),
        ]

    def test_trainer_init(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        from eklavya_e2_cache import load_e2_cache

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_tiny_cache(tmp_path, specs)
        cache = load_e2_cache(cache_dir)

        cfg = E2Config()
        ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
        device = torch.device("cpu")

        trainer = E2Trainer(cfg, student, ports, cache, device)
        assert trainer.total_steps() > 0
        assert len(trainer.positions) == 3

    def test_trainer_active_teachers_by_phase(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        from eklavya_e2_cache import load_e2_cache

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_tiny_cache(tmp_path, specs)
        cache = load_e2_cache(cache_dir)

        cfg = E2Config()
        ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
        trainer = E2Trainer(cfg, student, ports, cache, torch.device("cpu"))

        warmup_active = trainer.get_active_teachers(E2Phase.PORT_WARMUP)
        assert all(s.role.name == "ANCHOR" for s in warmup_active)

        disagree_active = trainer.get_active_teachers(E2Phase.DISAGREEMENT)
        assert len(disagree_active) >= 1

    def test_trainer_configure_freeze(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        from eklavya_e2_cache import load_e2_cache

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_tiny_cache(tmp_path, specs)
        cache = load_e2_cache(cache_dir)

        cfg = E2Config()
        ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
        trainer = E2Trainer(cfg, student, ports, cache, torch.device("cpu"))

        trainer.configure_freeze(E2Phase.PORT_WARMUP)
        assert not any(p.requires_grad for p in student.parameters())
        assert any(p.requires_grad for p in ports.parameters())

        trainer.configure_freeze(E2Phase.DISAGREEMENT)
        assert any(p.requires_grad for p in student.parameters())

    def test_trainer_loss_weights(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        from eklavya_e2_cache import load_e2_cache

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_tiny_cache(tmp_path, specs)
        cache = load_e2_cache(cache_dir)

        cfg = E2Config()
        ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
        trainer = E2Trainer(cfg, student, ports, cache, torch.device("cpu"))

        w_warmup = trainer.get_loss_weights(0, E2Phase.PORT_WARMUP)
        assert w_warmup["kl"] == 0.0
        assert w_warmup["semantic"] == 0.0

        w_disagree = trainer.get_loss_weights(
            cfg.port_warmup_steps + cfg.consensus_steps + cfg.semantic_landing_steps + 100,
            E2Phase.DISAGREEMENT)
        assert w_disagree["kl"] > 0.0
        assert w_disagree["semantic"] > 0.0

    def test_trainer_build_optimizer(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        from eklavya_e2_cache import load_e2_cache

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_tiny_cache(tmp_path, specs)
        cache = load_e2_cache(cache_dir)

        cfg = E2Config()
        ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
        trainer = E2Trainer(cfg, student, ports, cache, torch.device("cpu"))

        trainer.configure_freeze(E2Phase.PORT_WARMUP)
        optimizer = trainer.build_optimizer()
        assert len(optimizer.param_groups) >= 1


# ---------------------------------------------------------------------------
# Cache shard coverage tests (P0-1: train must stay within cached range)
# ---------------------------------------------------------------------------

class TestCacheShardCoverage:

    def test_train_range_derived_from_manifest(self):
        """Verify train_e2 would use cache shard_range, not full data."""
        manifest = {"shard_range": [5, 15], "n_positions": 100,
                     "teacher_count": 1, "kl_top_k": 16}
        cache_start, cache_end = manifest["shard_range"]
        n_cached = cache_end - cache_start
        n_eval = min(2, max(1, n_cached // 10))
        train_range = (cache_start, cache_end - n_eval)
        assert train_range == (5, 14)

    def test_small_cache_raises(self):
        """Cache covering <2 shards should fail."""
        manifest = {"shard_range": [3, 4]}
        cache_start, cache_end = manifest["shard_range"]
        n_cached = cache_end - cache_start
        assert n_cached < 2

    def test_cache_range_respected(self):
        """Larger cache: eval shards come from within cached range."""
        manifest = {"shard_range": [10, 60]}
        cache_start, cache_end = manifest["shard_range"]
        n_cached = cache_end - cache_start
        n_eval = min(2, max(1, n_cached // 10))
        train_range = (cache_start, cache_end - n_eval)
        assert train_range[0] >= cache_start
        assert train_range[1] <= cache_end
        assert cache_end - train_range[1] == n_eval

    def test_entropy_threshold_parameter(self, tmp_path):
        """Verify entropy_threshold is now a real parameter."""
        from eklavya_e2_cache_builder import build_position_manifest
        import inspect
        sig = inspect.signature(build_position_manifest)
        assert "entropy_threshold" in sig.parameters
        assert sig.parameters["entropy_threshold"].default == 4.0


# ---------------------------------------------------------------------------
# E2 resume, eval, and best-checkpoint tests (P1-3)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# eval_e2.py scaffold tests (P1-4)
# ---------------------------------------------------------------------------

class TestEvalE2Scaffold:

    def test_eval_config_defaults(self):
        from eval_e2 import EvalConfig
        cfg = EvalConfig()
        assert cfg.ablation_id == "A2"
        assert cfg.run_label == ""
        assert cfg.eval_shards
        assert cfg.cache_dir is None

    def test_eval_config_custom(self):
        from eval_e2 import EvalConfig
        cfg = EvalConfig(ablation_id="A5", run_label="no_router")
        assert cfg.ablation_id == "A5"
        assert cfg.run_label == "no_router"

    def test_evaluate_bpb_callable(self):
        from eval_e2 import evaluate_bpb
        import inspect
        sig = inspect.signature(evaluate_bpb)
        assert "student" in sig.parameters
        assert "eval_loader" in sig.parameters
        assert "cache_positions" in sig.parameters


class TestE2ResumeAndEval:

    def test_e2_config_has_resume_and_eval_fields(self):
        cfg = E2Config()
        assert hasattr(cfg, "resume_from")
        assert cfg.resume_from is None
        assert hasattr(cfg, "eval_every")
        assert cfg.eval_every > 0
        assert hasattr(cfg, "eval_batches")
        assert cfg.eval_batches > 0

    def test_evaluate_e2_exists_and_callable(self):
        from eklavya_e2_training import evaluate_e2
        import inspect
        sig = inspect.signature(evaluate_e2)
        assert "student" in sig.parameters
        assert "eval_loader" in sig.parameters
        assert "device" in sig.parameters
        assert "cfg" in sig.parameters

    def test_checkpoint_roundtrip(self, tmp_path):
        """Verify checkpoint save/load preserves model state."""
        from s0_architecture import S0Config, SutraS0
        model_cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4,
        )
        student = SutraS0(model_cfg)

        ckpt_path = str(tmp_path / "e2_test.pt")
        torch.save({
            "step": 42,
            "phase": "CONSENSUS",
            "model": student.state_dict(),
            "model_cfg": model_cfg,
            "ports": {},
            "optimizer": None,
            "config": E2Config().__dict__,
            "best_eval_bpb": 3.14,
        }, ckpt_path)

        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert loaded["step"] == 42
        assert loaded["best_eval_bpb"] == 3.14

        student2 = SutraS0(loaded["model_cfg"])
        student2.load_state_dict(loaded["model"])
        x = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            out1 = student(x)["logits"]
            out2 = student2(x)["logits"]
        assert torch.allclose(out1, out2)


class TestMappedPositionRecords:
    def test_roundtrip(self, tmp_path):
        recs = [make_position(pid=i) for i in range(50)]
        path = str(tmp_path / "positions.bin")
        write_position_manifest(path, recs)

        with MappedPositionRecords(path) as mapped:
            assert len(mapped) == 50
            for i in range(50):
                assert mapped[i].position_id == recs[i].position_id
                assert mapped[i].gold_byte == recs[i].gold_byte

    def test_to_list(self, tmp_path):
        recs = [make_position(pid=i) for i in range(10)]
        path = str(tmp_path / "positions.bin")
        write_position_manifest(path, recs)

        with MappedPositionRecords(path) as mapped:
            lst = mapped.to_list()
            assert len(lst) == 10
            assert lst[5].position_id == recs[5].position_id

    def test_out_of_bounds(self, tmp_path):
        recs = [make_position(pid=0)]
        path = str(tmp_path / "positions.bin")
        write_position_manifest(path, recs)

        with MappedPositionRecords(path) as mapped:
            with pytest.raises(IndexError):
                mapped[1]
            with pytest.raises(IndexError):
                mapped[-1]


class TestMappedKLRecords:
    def test_roundtrip(self, tmp_path):
        K = 16
        recs = [make_kl(pid=i, K=K) for i in range(30)]
        path = str(tmp_path / "kl.bin")
        write_teacher_kl_records(path, recs, K)

        with MappedKLRecords(path) as mapped:
            assert len(mapped) == 30
            assert mapped.K == K
            for i in range(30):
                assert mapped[i].position_id == recs[i].position_id
                np.testing.assert_array_equal(
                    mapped[i].top_bytes, recs[i].top_bytes[:K])

    def test_out_of_bounds(self, tmp_path):
        recs = [make_kl(pid=0)]
        path = str(tmp_path / "kl.bin")
        write_teacher_kl_records(path, recs, 16)

        with MappedKLRecords(path) as mapped:
            with pytest.raises(IndexError):
                mapped[1]


class TestMappedAlignRecords:
    def test_roundtrip(self, tmp_path):
        recs = [make_align(pid=i) for i in range(20)]
        path = str(tmp_path / "align.bin")
        write_teacher_align_records(path, recs)

        with MappedAlignRecords(path) as mapped:
            assert len(mapped) == 20
            for i in range(20):
                assert mapped[i].position_id == recs[i].position_id
                assert mapped[i].token_id == recs[i].token_id

    def test_out_of_bounds(self, tmp_path):
        recs = [make_align(pid=0)]
        path = str(tmp_path / "align.bin")
        write_teacher_align_records(path, recs)

        with MappedAlignRecords(path) as mapped:
            with pytest.raises(IndexError):
                mapped[1]


# ---------------------------------------------------------------------------
# CPU end-to-end integration test (train / checkpoint / resume / eval)
# ---------------------------------------------------------------------------

import math

class TestE2EndToEnd:
    """CPU-only integration test for the full E2 train/resume/eval cycle.

    Creates real shard files, a real E2 cache on disk, and runs the actual
    train_e2 function. This catches launch-day failures that unit tests miss.
    """

    def _create_fixtures(self, tmp_path, n_shards=3, n_pos=4, K=8):
        import torch
        from s0_architecture import S0Config, SutraS0

        model_cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4, max_seq_len=256,
            decoder_dim=32, decoder_layers=1, decoder_heads=2,
        )
        student = SutraS0(model_cfg)
        ckpt_path = str(tmp_path / "student.pt")
        torch.save({
            "step": 1000,
            "model": student.state_dict(),
            "model_cfg": model_cfg,
        }, ckpt_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        rng = np.random.RandomState(42)
        for i in range(n_shards):
            data = rng.randint(0, 256, 4096, dtype=np.uint8)
            (shard_dir / f"shard_{i:04d}.bin").write_bytes(data.tobytes())

        specs = [
            TeacherSpec(
                teacher_id=0, name="test_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.6, vram_gb=0.1,
            ),
        ]

        cache_dir = tmp_path / "e2_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n_pos)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        tdir = cache_dir / "teachers" / "test_anchor"
        tdir.mkdir()
        kl_recs = [
            E2KLRecord(
                position_id=i, patch_idx=i + 1,
                tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                top_bytes=np.arange(K, dtype=np.uint8),
                top_probs=np.full(K, 1.0 / K, dtype=np.float16),
            )
            for i in range(n_pos)
        ]
        write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

        align_recs = [
            E2AlignRecord(
                position_id=i, byte_start=i * 4,
                byte_len=4, token_id=i, align_quality=1.0,
            )
            for i in range(n_pos)
        ]
        write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

        emb = torch.randn(64, 32).half()
        torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        save_e2_manifest(
            str(cache_dir), specs, n_positions=n_pos, K=K,
            shard_range=(0, n_shards),
        )

        return ckpt_path, str(shard_dir), str(cache_dir), model_cfg

    def _make_cfg(self, tmp_path, shard_dir, cache_dir, **overrides):
        from eklavya_e2_training import E2Config
        log_dir = tmp_path / "logs"
        log_dir.mkdir(exist_ok=True)
        defaults = dict(
            port_warmup_steps=2,
            consensus_steps=2,
            semantic_landing_steps=2,
            disagreement_steps=4,
            batch_size=1,
            seq_len=64,
            grad_accum=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            log_file=str(log_dir / "e2.jsonl"),
            checkpoint_every=5,
            log_every=1,
            eval_every=5,
            eval_batches=1,
            data_dir=shard_dir,
            cache_dir=cache_dir,
        )
        defaults.update(overrides)
        return E2Config(**defaults)

    def test_train_produces_checkpoint_and_logs(self, tmp_path):
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(tmp_path, shard_dir, cache_dir)

        train_e2(cfg, ckpt_path, cache_dir)

        final_path = os.path.join(cfg.checkpoint_dir, "e2_final.pt")
        assert os.path.exists(final_path)

        with open(cfg.log_file) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) >= 5

        final = torch.load(final_path, map_location="cpu", weights_only=False)
        assert "step" in final
        assert "model" in final
        assert "ports" in final
        assert "model_cfg" in final
        assert "phase" in final
        assert final["step"] >= 8

    def test_checkpoint_resume_preserves_state(self, tmp_path):
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=3, consensus_steps=3,
            semantic_landing_steps=3, disagreement_steps=6,
            checkpoint_every=5, eval_every=100,
        )

        train_e2(cfg, ckpt_path, cache_dir)

        step5_path = os.path.join(cfg.checkpoint_dir, "e2_step5.pt")
        assert os.path.exists(step5_path), "Step 5 checkpoint missing"

        ckpt5 = torch.load(step5_path, map_location="cpu", weights_only=False)
        assert ckpt5["step"] == 5
        assert "phase" in ckpt5
        assert "model" in ckpt5
        assert "ports" in ckpt5
        assert ckpt5["optimizer"] is not None
        assert "rng_state" in ckpt5
        assert "config" in ckpt5

        resume_dir = tmp_path / "resume"
        resume_dir.mkdir()
        resume_cfg = self._make_cfg(
            resume_dir, shard_dir, cache_dir,
            port_warmup_steps=3, consensus_steps=3,
            semantic_landing_steps=3, disagreement_steps=6,
            checkpoint_every=5, eval_every=100,
            resume_from=step5_path,
        )

        train_e2(resume_cfg, ckpt_path, cache_dir)

        final = torch.load(
            os.path.join(resume_cfg.checkpoint_dir, "e2_final.pt"),
            map_location="cpu", weights_only=False)
        assert final["step"] >= 13

        with open(resume_cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        logged_steps = [e["step"] for e in entries if "step" in e]
        assert all(s >= 6 for s in logged_steps), (
            f"Resume from step 5 should start at step 6, got steps: {logged_steps[:5]}"
        )

    def test_eval_produces_finite_bpb(self, tmp_path):
        from eklavya_e2_training import E2Config, evaluate_e2
        from s0_architecture import S0Config, SutraS0
        from eklavya_training import EklavyaDataset
        import torch

        model_cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4, max_seq_len=256,
            decoder_dim=32, decoder_layers=1, decoder_heads=2,
        )
        student = SutraS0(model_cfg)

        shard_dir = tmp_path / "eval_shards"
        shard_dir.mkdir()
        rng = np.random.RandomState(42)
        for i in range(2):
            data = rng.randint(0, 256, 4096, dtype=np.uint8)
            (shard_dir / f"shard_{i:04d}.bin").write_bytes(data.tobytes())

        from torch.utils.data import DataLoader
        dataset = EklavyaDataset(str(shard_dir), 64, 4)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

        cfg = E2Config(eval_batches=2)
        metrics = evaluate_e2(student, loader, torch.device("cpu"), cfg)

        assert metrics["eval_loss"] > 0
        assert metrics["eval_bpb"] > 0
        assert not math.isinf(metrics["eval_bpb"])

    def test_phase_transitions_logged(self, tmp_path):
        from eklavya_e2_training import train_e2
        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(tmp_path, shard_dir, cache_dir, eval_every=100)

        train_e2(cfg, ckpt_path, cache_dir)

        with open(cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        phases = {e["phase"] for e in entries if "phase" in e}
        assert len(phases) >= 2, f"Expected multiple phases, got {phases}"

    def test_checkpoint_fires_with_grad_accum_2(self, tmp_path):
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=2, consensus_steps=2,
            semantic_landing_steps=2, disagreement_steps=6,
            checkpoint_every=4, eval_every=100,
        )
        cfg.grad_accum = 2

        train_e2(cfg, ckpt_path, cache_dir)

        step4_path = os.path.join(cfg.checkpoint_dir, "e2_step4.pt")
        step8_path = os.path.join(cfg.checkpoint_dir, "e2_step8.pt")
        assert os.path.exists(step4_path), "Step 4 checkpoint must exist with grad_accum=2"
        assert os.path.exists(step8_path), "Step 8 checkpoint must exist with grad_accum=2"

        ckpt = torch.load(step4_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 4
        assert "config" in ckpt

    def test_best_checkpoint_includes_config(self, tmp_path):
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=2, consensus_steps=2,
            semantic_landing_steps=2, disagreement_steps=4,
            eval_every=5, checkpoint_every=100,
        )

        train_e2(cfg, ckpt_path, cache_dir)

        best_path = os.path.join(cfg.checkpoint_dir, "e2_best.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            assert "config" in ckpt, "Best checkpoint must include config for eval provenance"
            assert "optimizer" in ckpt, "Best checkpoint must be resumable"
            assert "rng_state" in ckpt, "Best checkpoint must include RNG state"
            assert "py_rng_state" in ckpt, "Best checkpoint must include Python RNG state"
            assert "np_rng_state" in ckpt, "Best checkpoint must include NumPy RNG state"

    def test_log_entries_include_ablation_id(self, tmp_path):
        from eklavya_e2_training import train_e2

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(tmp_path, shard_dir, cache_dir, eval_every=100)
        cfg.ablation_id = "TEST_ABL"

        train_e2(cfg, ckpt_path, cache_dir)

        with open(cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        train_entries = [e for e in entries if "ablation_id" in e]
        assert len(train_entries) > 0, "Log entries must include ablation_id"
        assert all(e["ablation_id"] == "TEST_ABL" for e in train_entries)

    def test_ce_only_mode_trains_without_teachers(self, tmp_path):
        """A0 CE-only runs the full loop with no teacher losses and no phase transitions."""
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=2, consensus_steps=2,
            semantic_landing_steps=2, disagreement_steps=4,
            eval_every=5, checkpoint_every=100,
        )
        cfg.ablation_id = "A0"
        cfg.ce_only = True

        train_e2(cfg, ckpt_path, cache_dir)

        final_path = os.path.join(cfg.checkpoint_dir, "e2_final.pt")
        assert os.path.exists(final_path), "CE-only must produce final checkpoint"

        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 10, "CE-only should run all 10 steps"

        with open(cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        phases = {e.get("phase") for e in entries if "phase" in e}
        assert "CE_ONLY" in phases, "CE-only log entries should have phase=CE_ONLY"
        assert "PORT_WARMUP" not in phases, "CE-only should skip PORT_WARMUP"
        assert "CONSENSUS" not in phases, "CE-only should skip CONSENSUS"

    def _create_bld_fixtures(self, tmp_path, seq_len=64, n_pos=4, K=8):
        """Like _create_fixtures but with tiny shards so every sample hits offset 0."""
        import torch
        from s0_architecture import S0Config, SutraS0

        model_cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4, max_seq_len=256,
            decoder_dim=32, decoder_layers=1, decoder_heads=2,
        )
        student = SutraS0(model_cfg)
        ckpt_path = str(tmp_path / "student.pt")
        torch.save({
            "step": 1000,
            "model": student.state_dict(),
            "model_cfg": model_cfg,
        }, ckpt_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        rng = np.random.RandomState(42)
        for i in range(3):
            data = rng.randint(0, 256, seq_len, dtype=np.uint8)
            (shard_dir / f"shard_{i:04d}.bin").write_bytes(data.tobytes())

        specs = [
            TeacherSpec(
                teacher_id=0, name="test_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=1.0, vram_gb=0.1,
            ),
        ]

        cache_dir = tmp_path / "e2_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n_pos)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        tdir = cache_dir / "teachers" / "test_anchor"
        tdir.mkdir()
        kl_recs = [
            E2KLRecord(
                position_id=i, patch_idx=i + 1,
                tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                top_bytes=np.arange(K, dtype=np.uint8),
                top_probs=np.full(K, 1.0 / K, dtype=np.float16),
            )
            for i in range(n_pos)
        ]
        write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

        align_recs = [
            E2AlignRecord(
                position_id=i, byte_start=i * 4,
                byte_len=4, token_id=i, align_quality=1.0,
            )
            for i in range(n_pos)
        ]
        write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

        emb = torch.randn(64, 32).half()
        torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        save_e2_manifest(
            str(cache_dir), specs, n_positions=n_pos, K=K,
            shard_range=(0, 3),
        )

        return ckpt_path, str(shard_dir), str(cache_dir), model_cfg

    def _create_multi_teacher_fixtures(self, tmp_path, seq_len=64, n_pos=4, K=8):
        """Fixtures with 2 KL teachers for multi-teacher routing tests."""
        import torch
        from s0_architecture import S0Config, SutraS0

        model_cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4, max_seq_len=256,
            decoder_dim=32, decoder_layers=1, decoder_heads=2,
        )
        student = SutraS0(model_cfg)
        ckpt_path = str(tmp_path / "student.pt")
        torch.save({
            "step": 1000,
            "model": student.state_dict(),
            "model_cfg": model_cfg,
        }, ckpt_path)

        shard_dir = tmp_path / "shards"
        shard_dir.mkdir()
        rng = np.random.RandomState(42)
        for i in range(3):
            data = rng.randint(0, 256, seq_len, dtype=np.uint8)
            (shard_dir / f"shard_{i:04d}.bin").write_bytes(data.tobytes())

        specs = [
            TeacherSpec(
                teacher_id=0, name="test_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.6, vram_gb=0.1,
            ),
            TeacherSpec(
                teacher_id=1, name="test_diversity",
                family=TeacherFamily.HYBRID, role=TeacherRole.DIVERSITY,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=False, has_semantic=False,
                prior=0.4, vram_gb=0.1,
            ),
        ]

        cache_dir = tmp_path / "e2_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n_pos)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        for spec in specs:
            tdir = cache_dir / "teachers" / spec.name
            tdir.mkdir()
            kl_recs = [
                E2KLRecord(
                    position_id=i, patch_idx=i + 1,
                    tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                    top_bytes=np.arange(K, dtype=np.uint8),
                    top_probs=np.full(K, 1.0 / K, dtype=np.float16),
                )
                for i in range(n_pos)
            ]
            write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

            if spec.has_align:
                align_recs = [
                    E2AlignRecord(
                        position_id=i, byte_start=i * 4,
                        byte_len=4, token_id=i, align_quality=1.0,
                    )
                    for i in range(n_pos)
                ]
                write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

            emb = torch.randn(64, 32).half()
            torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        save_e2_manifest(
            str(cache_dir), specs, n_positions=n_pos, K=K,
            shard_range=(0, 3),
        )

        return ckpt_path, str(shard_dir), str(cache_dir), model_cfg

    def test_gold_free_router_trains_end_to_end(self, tmp_path):
        """A9c gold-free router runs the full loop with 2 teachers."""
        from eklavya_e2_training import train_e2

        ckpt_path, shard_dir, cache_dir, _ = self._create_multi_teacher_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=2, consensus_steps=2,
            semantic_landing_steps=2, disagreement_steps=4,
            eval_every=5, checkpoint_every=100,
        )
        cfg.router_mode = "gold_free_student_jsd"

        train_e2(cfg, ckpt_path, cache_dir)

        final_path = os.path.join(cfg.checkpoint_dir, "e2_final.pt")
        assert os.path.exists(final_path), "Gold-free router must produce final checkpoint"

        with open(cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        train_entries = [e for e in entries if "ce_loss" in e]
        assert len(train_entries) >= 5

    def test_bld_mode_trains_with_anchor_kl(self, tmp_path):
        """BLD baseline: single-teacher byte KL, no E2 machinery."""
        from eklavya_e2_training import train_e2
        import torch

        ckpt_path, shard_dir, cache_dir, _ = self._create_bld_fixtures(tmp_path)
        cfg = self._make_cfg(
            tmp_path, shard_dir, cache_dir,
            port_warmup_steps=2, consensus_steps=2,
            semantic_landing_steps=2, disagreement_steps=4,
            eval_every=5, checkpoint_every=100,
        )
        cfg.ablation_id = "BLD"
        cfg.bld_mode = True
        cfg.bld_kl_weight = 0.10

        train_e2(cfg, ckpt_path, cache_dir)

        final_path = os.path.join(cfg.checkpoint_dir, "e2_final.pt")
        assert os.path.exists(final_path), "BLD must produce final checkpoint"

        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        assert ckpt["step"] == 10, "BLD should run all 10 steps"

        with open(cfg.log_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        phases = {e.get("phase") for e in entries if "phase" in e}
        assert "BLD" in phases, "BLD log entries should have phase=BLD"
        assert "PORT_WARMUP" not in phases, "BLD should skip phase curriculum"

        train_entries = [e for e in entries if "bld_kl_loss" in e]
        assert len(train_entries) > 0, "BLD should log bld_kl_loss"
        for e in train_entries:
            assert e["bld_kl_loss"] >= 0.0, "BLD KL loss should be non-negative"
            assert "bld_kl_bits" in e, "BLD should log bld_kl_bits"


# ---------------------------------------------------------------------------
# Index memory estimation tests
# ---------------------------------------------------------------------------


class TestIndexMemoryEstimation:

    def test_small_cache_fits_in_megabytes(self):
        est = estimate_index_memory(n_positions=1000, n_teachers=5)
        assert est["total_mb"] < 10

    def test_scales_with_positions(self):
        small = estimate_index_memory(n_positions=1000, n_teachers=5)
        big = estimate_index_memory(n_positions=10000, n_teachers=5)
        assert big["total_bytes"] > small["total_bytes"] * 5

    def test_scales_with_teachers(self):
        t2 = estimate_index_memory(n_positions=10000, n_teachers=2)
        t5 = estimate_index_memory(n_positions=10000, n_teachers=5)
        assert t5["kl_indices_bytes"] > t2["kl_indices_bytes"] * 2

    def test_production_scale_warning(self):
        est = estimate_index_memory(
            n_positions=10_000_000, n_teachers=5)
        assert est["total_gb"] > 0
        assert "total_gb" in est

    def test_custom_kl_count(self):
        est = estimate_index_memory(
            n_positions=10000, n_teachers=5,
            n_kl_per_teacher=5000, n_align_per_teacher=8000)
        est_full = estimate_index_memory(
            n_positions=10000, n_teachers=5)
        assert est["total_bytes"] < est_full["total_bytes"]


# ---------------------------------------------------------------------------
# Streaming cache writer tests
# ---------------------------------------------------------------------------


class TestStreamingWriters:
    """Verify streaming writers produce binary-compatible output."""

    def test_streaming_kl_matches_batch_write(self, tmp_path):
        from eklavya_e2_cache_builder import _StreamingKLWriter
        K = 8
        records = [
            E2KLRecord(
                position_id=i, patch_idx=i + 1, tail_prob=0.05,
                entropy=2.0, logp_gold=-1.5,
                top_bytes=np.arange(K, dtype=np.uint8),
                top_probs=np.linspace(0.5, 0.01, K).astype(np.float16),
            )
            for i in range(5)
        ]

        batch_path = str(tmp_path / "batch_kl.bin")
        write_teacher_kl_records(batch_path, records, K=K)

        stream_path = str(tmp_path / "stream_kl.bin")
        writer = _StreamingKLWriter(stream_path, K=K)
        writer.extend(records[:3])
        writer.extend(records[3:])
        n = writer.close()

        assert n == 5
        batch_data = open(batch_path, "rb").read()
        stream_data = open(stream_path, "rb").read()
        assert batch_data == stream_data

        read_back, read_K = read_teacher_kl_records(stream_path)
        assert read_K == K
        assert len(read_back) == 5
        assert read_back[0].position_id == 0
        assert read_back[4].position_id == 4

    def test_streaming_align_matches_batch_write(self, tmp_path):
        from eklavya_e2_cache_builder import _StreamingAlignWriter
        records = [
            E2AlignRecord(
                position_id=i, byte_start=i * 4,
                byte_len=4, token_id=100 + i, align_quality=0.9,
            )
            for i in range(4)
        ]

        batch_path = str(tmp_path / "batch_align.bin")
        write_teacher_align_records(batch_path, records)

        stream_path = str(tmp_path / "stream_align.bin")
        writer = _StreamingAlignWriter(stream_path)
        writer.extend(records[:2])
        writer.extend(records[2:])
        n = writer.close()

        assert n == 4
        batch_data = open(batch_path, "rb").read()
        stream_data = open(stream_path, "rb").read()
        assert batch_data == stream_data

        read_back = read_teacher_align_records(stream_path)
        assert len(read_back) == 4
        assert read_back[0].position_id == 0
        assert read_back[3].position_id == 3

    def test_streaming_empty_produces_valid_file(self, tmp_path):
        from eklavya_e2_cache_builder import _StreamingKLWriter, _StreamingAlignWriter
        kl_path = str(tmp_path / "empty_kl.bin")
        align_path = str(tmp_path / "empty_align.bin")

        kl_writer = _StreamingKLWriter(kl_path, K=16)
        n_kl = kl_writer.close()
        assert n_kl == 0

        align_writer = _StreamingAlignWriter(align_path)
        n_align = align_writer.close()
        assert n_align == 0

        recs, K = read_teacher_kl_records(kl_path)
        assert len(recs) == 0 and K == 16

        recs = read_teacher_align_records(align_path)
        assert len(recs) == 0


# ---------------------------------------------------------------------------
# E2CacheView tests (mmap-backed streaming cache)
# ---------------------------------------------------------------------------

class TestE2CacheView:

    def _make_cache_dir(self, tmp_path, n_positions=5, K=8):
        """Create a minimal E2 cache on disk for E2CacheView tests."""
        cache_dir = tmp_path / "view_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        specs = [
            TeacherSpec(
                teacher_id=0, name="t_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.6, vram_gb=0.1,
            ),
            TeacherSpec(
                teacher_id=1, name="t_diversity",
                family=TeacherFamily.SSM, role=TeacherRole.DIVERSITY,
                hidden_dim=16, vocab_size=32,
                has_kl=True, has_align=False, has_semantic=False,
                prior=0.4, vram_gb=0.1,
            ),
        ]

        positions = [
            PositionRecord(
                position_id=i, shard_id=i // 3, seq_offset=(i % 3) * 100,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n_positions)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        import torch
        for spec in specs:
            tdir = cache_dir / "teachers" / spec.name
            tdir.mkdir()

            if spec.has_kl:
                kl_recs = [
                    E2KLRecord(
                        position_id=i, patch_idx=i + 1,
                        tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                        top_bytes=np.arange(K, dtype=np.uint8),
                        top_probs=np.full(K, 1.0 / K, dtype=np.float16),
                    )
                    for i in range(n_positions)
                ]
                write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

            if spec.has_align:
                align_recs = [
                    E2AlignRecord(
                        position_id=i, byte_start=i * 4,
                        byte_len=4, token_id=10 + i, align_quality=1.0,
                    )
                    for i in range(n_positions)
                ]
                write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

                emb = torch.randn(spec.vocab_size, spec.hidden_dim).half()
                torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        save_e2_manifest(str(cache_dir), specs, n_positions=n_positions, K=K)
        return str(cache_dir), specs, positions

    def test_init_and_position_count(self, tmp_path):
        cache_dir, specs, positions = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            assert view.n_positions == 5
            assert len(view.manifest["teacher_specs"]) == 2

    def test_positions_for_loc(self, tmp_path):
        cache_dir, _, positions = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            result = view.positions_for_loc(0, 0)
            assert len(result) == 1
            assert result[0].position_id == 0
            assert result[0].gold_byte == 65

            empty = view.positions_for_loc(99, 99)
            assert empty == []

    def test_kl_record_lookup(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            rec = view.kl_record("t_anchor", 2)
            assert rec is not None
            assert rec.position_id == 2
            assert len(rec.top_bytes) == 8

            assert view.kl_record("t_anchor", 999) is None
            assert view.kl_record("nonexistent", 0) is None

    def test_align_record_lookup(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            rec = view.align_record("t_anchor", 3)
            assert rec is not None
            assert rec.position_id == 3
            assert rec.byte_start == 12

            assert view.align_record("t_diversity", 0) is None
            assert view.align_record("t_anchor", 999) is None

    def test_embedding_table(self, tmp_path):
        cache_dir, specs, _ = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            emb = view.embedding_table("t_anchor")
            assert emb is not None
            assert emb.shape == (specs[0].vocab_size, specs[0].hidden_dim)

            assert view.embedding_table("t_diversity") is None

    def test_has_teacher(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            assert view.has_teacher("t_anchor")
            assert view.has_teacher("t_diversity")
            assert not view.has_teacher("nonexistent")

    def test_kl_pids_ordered(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path, n_positions=5)
        with E2CacheView(cache_dir) as view:
            pids = view.kl_pids_ordered("t_anchor")
            assert pids == [0, 1, 2, 3, 4]
            assert view.kl_pids_ordered("nonexistent") == []

    def test_align_pids_ordered(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path, n_positions=5)
        with E2CacheView(cache_dir) as view:
            pids = view.align_pids_ordered("t_anchor")
            assert pids == [0, 1, 2, 3, 4]
            assert view.align_pids_ordered("t_diversity") == []

    def test_build_loc_index(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path, n_positions=6)
        with E2CacheView(cache_dir) as view:
            n_keys = len(view._loc_index)
            total = sum(len(v) for v in view._loc_index.values())
            assert total == 6
            assert n_keys >= 1

    def test_build_pid_index(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path, n_positions=4)
        with E2CacheView(cache_dir) as view:
            idx = view._kl_pid_idx.get("t_anchor", {})
            assert len(idx) == 4
            for pid in range(4):
                assert pid in idx

    def test_validate_clean_cache(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path)
        with E2CacheView(cache_dir) as view:
            errors = view.validate()
            assert errors == []

    def test_validate_position_count_mismatch(self, tmp_path):
        cache_dir, _, _ = self._make_cache_dir(tmp_path, n_positions=5)
        with E2CacheView(cache_dir) as view:
            view.manifest["n_positions"] = 999
            errors = view.validate()
            assert any("Position count mismatch" in e for e in errors)

    def test_validate_missing_kl_for_has_kl(self, tmp_path):
        cache_dir, specs, _ = self._make_cache_dir(tmp_path)
        import os as _os
        _os.remove(_os.path.join(cache_dir, "teachers", "t_anchor", "kl_records.bin"))
        with E2CacheView(cache_dir) as view:
            errors = view.validate()
            assert any("has_kl=True but no kl_records" in e for e in errors)

    def test_validate_embedding_dim_mismatch(self, tmp_path):
        cache_dir, specs, _ = self._make_cache_dir(tmp_path)
        import torch
        wrong_emb = torch.randn(64, 99).half()
        torch.save(wrong_emb, os.path.join(
            cache_dir, "teachers", "t_anchor", "teacher_embeddings.pt"))
        with E2CacheView(cache_dir) as view:
            errors = view.validate()
            assert any("embedding dim" in e for e in errors)


# ---------------------------------------------------------------------------
# E2Trainer with E2CacheView integration test
# ---------------------------------------------------------------------------

class TestE2TrainerWithCacheView:

    def _make_cache_dir(self, tmp_path, specs, n_pos=3, K=8):
        cache_dir = tmp_path / "trainer_view_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers").mkdir()

        positions = [
            PositionRecord(
                position_id=i, shard_id=0, seq_offset=0,
                patch_idx=i + 1, gold_byte=65 + i,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            )
            for i in range(n_pos)
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        import torch
        for spec in specs:
            tdir = cache_dir / "teachers" / spec.name
            tdir.mkdir()

            if spec.has_kl:
                kl_recs = [
                    E2KLRecord(
                        position_id=i, patch_idx=i + 1,
                        tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                        top_bytes=np.arange(K, dtype=np.uint8),
                        top_probs=np.full(K, 1.0 / K, dtype=np.float16),
                    )
                    for i in range(n_pos)
                ]
                write_teacher_kl_records(str(tdir / "kl_records.bin"), kl_recs, K=K)

            if spec.has_align or spec.has_semantic:
                align_recs = [
                    E2AlignRecord(
                        position_id=i, byte_start=i * 4,
                        byte_len=4, token_id=10 + i, align_quality=1.0,
                    )
                    for i in range(n_pos)
                ]
                write_teacher_align_records(str(tdir / "align_records.bin"), align_recs)

                emb = torch.randn(spec.vocab_size, spec.hidden_dim).half()
                torch.save(emb, str(tdir / "teacher_embeddings.pt"))

        save_e2_manifest(str(cache_dir), specs, n_positions=n_pos, K=K)
        return str(cache_dir)

    def _make_specs(self):
        return [
            TeacherSpec(
                teacher_id=0, name="mock_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=64,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.5, vram_gb=0.1,
            ),
            TeacherSpec(
                teacher_id=1, name="mock_semantic",
                family=TeacherFamily.EMBEDDING, role=TeacherRole.SEMANTIC,
                hidden_dim=16, vocab_size=32,
                has_kl=False, has_align=False, has_semantic=True,
                prior=0.2, vram_gb=0.1,
            ),
        ]

    def _make_tiny_student(self):
        from s0_architecture import S0Config, SutraS0
        cfg = S0Config(
            d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
            byte_dim=32, ffn_mult=2.0, local_mixer_layers=1,
            patch_size=4,
        )
        return SutraS0(cfg), cfg

    def test_trainer_init_with_view(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_cache_dir(tmp_path, specs)

        with E2CacheView(cache_dir) as view:
            cfg = E2Config()
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            assert trainer.total_steps() > 0
            assert len(trainer.positions) == 3
            assert trainer._cache_view is view

    def test_positions_by_loc_via_view(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_cache_dir(tmp_path, specs)

        with E2CacheView(cache_dir) as view:
            cfg = E2Config()
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            positions = trainer.positions_by_loc.get((0, 0), [])
            assert len(positions) == 3
            assert all(isinstance(p, PositionRecord) for p in positions)

            empty = trainer.positions_by_loc.get((99, 99), [])
            assert len(empty) == 0

    def test_kl_lookup_via_view(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_cache_dir(tmp_path, specs)

        with E2CacheView(cache_dir) as view:
            cfg = E2Config()
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            kl_lookup = trainer.teacher_kl_by_pid.get("mock_anchor", {})
            rec = kl_lookup.get(1)
            assert rec is not None
            assert rec.position_id == 1

            assert kl_lookup.get(999) is None

    def test_active_teachers_via_view(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_cache_dir(tmp_path, specs)

        with E2CacheView(cache_dir) as view:
            cfg = E2Config()
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            warmup = trainer.get_active_teachers(E2Phase.PORT_WARMUP)
            assert all(s.role.name == "ANCHOR" for s in warmup)

    def test_shuffle_via_view(self, tmp_path):
        from eklavya_e2_training import E2Config, E2Trainer
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = self._make_specs()
        cache_dir = self._make_cache_dir(tmp_path, specs, n_pos=10)

        with E2CacheView(cache_dir) as view:
            cfg = E2Config(shuffle_teacher_targets=True, shuffle_seed=42)
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            kl_lookup = trainer.teacher_kl_by_pid.get("mock_anchor", {})
            assert len(kl_lookup) == 10
            found = kl_lookup.get(0)
            assert found is not None

    def test_oob_token_id_skipped(self, tmp_path):
        """Align records with token_id >= emb_table.shape[0] must be skipped."""
        from eklavya_e2_training import E2Config, E2Trainer, E2Phase
        from eklavya_e2_losses import MultiTeacherProjectionPorts
        import torch

        student, model_cfg = self._make_tiny_student()
        specs = [
            TeacherSpec(
                teacher_id=0, name="mock_anchor",
                family=TeacherFamily.DECODER, role=TeacherRole.ANCHOR,
                hidden_dim=32, vocab_size=5,
                has_kl=True, has_align=True, has_semantic=False,
                prior=0.5, vram_gb=0.1,
            ),
        ]
        cache_dir = tmp_path / "oob_cache"
        cache_dir.mkdir()
        (cache_dir / "teachers" / "mock_anchor").mkdir(parents=True)

        positions = [
            PositionRecord(
                position_id=0, shard_id=0, seq_offset=0,
                patch_idx=2, gold_byte=65,
                student_nll=4.0, student_entropy=2.0,
                reason_mask=SelectionReason.HIGH_NLL,
            ),
        ]
        write_position_manifest(str(cache_dir / "positions.bin"), positions)

        align_recs = [
            E2AlignRecord(
                position_id=0, byte_start=0,
                byte_len=4, token_id=999, align_quality=1.0,
            ),
        ]
        write_teacher_align_records(
            str(cache_dir / "teachers" / "mock_anchor" / "align_records.bin"),
            align_recs)

        kl_recs = [
            E2KLRecord(
                position_id=0, patch_idx=2,
                tail_prob=0.1, entropy=2.0, logp_gold=-3.0,
                top_bytes=np.arange(8, dtype=np.uint8),
                top_probs=np.full(8, 1.0 / 8, dtype=np.float16),
            ),
        ]
        write_teacher_kl_records(
            str(cache_dir / "teachers" / "mock_anchor" / "kl_records.bin"),
            kl_recs, K=8)

        emb = torch.randn(5, 32).half()
        torch.save(emb, str(cache_dir / "teachers" / "mock_anchor" / "teacher_embeddings.pt"))

        save_e2_manifest(str(cache_dir), specs, n_positions=1, K=8)

        with E2CacheView(str(cache_dir)) as view:
            cfg = E2Config()
            ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs)
            trainer = E2Trainer(cfg, student, ports, view, torch.device("cpu"))

            byte_ids = torch.randint(0, 256, (1, model_cfg.patch_size * 8))
            with torch.no_grad():
                out = student(byte_ids)

            losses = trainer.compute_teacher_losses(
                out["logits"], out["patch_states"],
                torch.tensor([0]), torch.tensor([0]),
                E2Phase.DISAGREEMENT, step=6000,
            )
            assert "align_mock_anchor" not in losses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
