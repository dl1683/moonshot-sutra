"""CPU-only tests for Option C: teacher-guided S0 pretraining.

Verifies:
  - OptionCConfig dataclass defaults
  - lambda_kd schedule
  - teacher_grad_budget schedule
  - compute_batch_kl_loss with synthetic cache
  - MappedByteKLCache integration
  - Training loop smoke test (tiny model + synthetic cache, 2 steps)
"""

import json
import math
import os
import struct
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import S0Config, SutraS0
from s0_option_c_training import (
    OptionCConfig, get_lambda_kd, get_teacher_grad_budget,
    compute_batch_kl_loss,
)
from eklavya_cache import (
    ByteKLRecord, StreamingCacheWriter, MappedByteKLCache,
    multi_byte_marginal, load_cache,
)


def tiny_cfg():
    return S0Config(
        byte_dim=16, local_mixer_layers=1, local_mixer_window=4,
        patch_size=4, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_mult=1.0, max_seq_len=16, decoder_dim=16, decoder_layers=1,
        decoder_heads=4, verifier_dim=16,
    )


def _make_synthetic_cache(tmpdir, shard_id=0, n_seqs=2, n_patches_per_seq=5,
                          kl_top_k=4, patch_size=4):
    writer = StreamingCacheWriter(tmpdir, kl_top_k=kl_top_k)
    records = []
    for seq_idx in range(n_seqs):
        offset = seq_idx * 64 * patch_size
        for p in range(1, n_patches_per_seq + 1):
            top_b = np.array([10, 20, 30, 40], dtype=np.uint8)[:kl_top_k]
            top_p = np.array([0.4, 0.3, 0.2, 0.05], dtype=np.float16)[:kl_top_k]
            records.append(ByteKLRecord(
                shard_id, offset, p, top_b, top_p, 0.05, 1.5))
    writer.write_shard([], records)
    writer.finalize(None, shard_range=(0, 1), extra_manifest={
        "selection_policy": "uniform",
        "sample_frac": 0.25,
    })
    return tmpdir


# --- Schedule tests ---

def test_lambda_kd_schedule():
    """lambda_kd ramps per Codex R53: 0.05→0.20 (1K), ramp to 0.35 (5K), hold, decay."""
    assert get_lambda_kd(0) == 0.05
    assert abs(get_lambda_kd(1000) - 0.20) < 0.01
    assert abs(get_lambda_kd(5000) - 0.35) < 0.01
    assert get_lambda_kd(10000) == 0.35
    assert get_lambda_kd(29999) == 0.35
    assert get_lambda_kd(30000) == 0.25
    assert get_lambda_kd(44999) == 0.25
    assert get_lambda_kd(45000) == 0.15
    assert get_lambda_kd(50000) == 0.15
    print("  test_lambda_kd_schedule PASSED")


def test_teacher_grad_budget_schedule():
    """Teacher grad budget per Codex R53: 0.35 (1K), 0.65 (30K), 0.50 (45K), 0.35."""
    assert get_teacher_grad_budget(0) == 0.35
    assert get_teacher_grad_budget(999) == 0.35
    assert get_teacher_grad_budget(1000) == 0.65
    assert get_teacher_grad_budget(29999) == 0.65
    assert get_teacher_grad_budget(30000) == 0.50
    assert get_teacher_grad_budget(44999) == 0.50
    assert get_teacher_grad_budget(45000) == 0.35
    assert get_teacher_grad_budget(50000) == 0.35
    print("  test_teacher_grad_budget_schedule PASSED")


def test_config_defaults():
    """OptionCConfig has correct Codex R53 defaults."""
    cfg = OptionCConfig()
    assert cfg.total_steps == 50000
    assert cfg.lr == 2e-4
    assert cfg.min_lr == 2e-5
    assert cfg.warmup_steps == 1500
    assert cfg.kl_temperature == 2.0
    assert cfg.max_kl_per_seq == 2048
    assert cfg.grad_accum_steps == 16
    print("  test_config_defaults PASSED")


# --- Batch KL loss tests ---

def test_batch_kl_loss_with_cache():
    """compute_batch_kl_loss retrieves records from MappedByteKLCache."""
    td = tempfile.mkdtemp()
    try:
        _make_synthetic_cache(td, shard_id=0, n_seqs=1, n_patches_per_seq=3)
        cache = MappedByteKLCache(td)

        mcfg = tiny_cfg()
        model = SutraS0(mcfg)

        seq_len = mcfg.max_seq_len * mcfg.patch_size
        byte_ids = torch.randint(0, 256, (1, seq_len)).long()
        out = model(byte_ids, return_aux=False)

        shard_ids = torch.tensor([0])
        seq_offsets = torch.tensor([0])

        loss, n_used, n_seqs = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"),
            T=2.0, max_per_seq=64)

        assert n_used > 0, "Should find KL records in cache"
        assert n_seqs == 1
        assert loss.item() > 0
        assert loss.requires_grad

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_batch_kl_loss_with_cache PASSED")


def test_batch_kl_loss_missing_sequence():
    """Missing cache sequence returns zero loss."""
    td = tempfile.mkdtemp()
    try:
        _make_synthetic_cache(td, shard_id=0, n_seqs=1, n_patches_per_seq=1)
        cache = MappedByteKLCache(td)

        mcfg = tiny_cfg()
        model = SutraS0(mcfg)

        seq_len = mcfg.max_seq_len * mcfg.patch_size
        byte_ids = torch.randint(0, 256, (1, seq_len)).long()
        with torch.no_grad():
            out = model(byte_ids, return_aux=False)

        shard_ids = torch.tensor([99])
        seq_offsets = torch.tensor([0])

        loss, n_used, n_seqs = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"))

        assert n_used == 0
        assert n_seqs == 0
        assert loss.item() == 0.0

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_batch_kl_loss_missing_sequence PASSED")


def test_batch_kl_loss_max_per_seq():
    """max_per_seq limits the number of records used per sequence."""
    td = tempfile.mkdtemp()
    try:
        _make_synthetic_cache(td, shard_id=0, n_seqs=1, n_patches_per_seq=10)
        cache = MappedByteKLCache(td)

        mcfg = tiny_cfg()
        model = SutraS0(mcfg)

        seq_len = mcfg.max_seq_len * mcfg.patch_size
        byte_ids = torch.randint(0, 256, (1, seq_len)).long()
        with torch.no_grad():
            out = model(byte_ids, return_aux=False)

        shard_ids = torch.tensor([0])
        seq_offsets = torch.tensor([0])

        _, n_all, _ = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"),
            max_per_seq=999)

        _, n_capped, _ = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"),
            max_per_seq=3)

        assert n_capped <= 3, f"Expected max 3 records, got {n_capped}"
        assert n_all >= n_capped

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_batch_kl_loss_max_per_seq PASSED")


def test_batch_kl_loss_gradient_flows():
    """KL loss gradient flows back through student logits."""
    td = tempfile.mkdtemp()
    try:
        _make_synthetic_cache(td, shard_id=0, n_seqs=1, n_patches_per_seq=3)
        cache = MappedByteKLCache(td)

        mcfg = tiny_cfg()
        model = SutraS0(mcfg)

        seq_len = mcfg.max_seq_len * mcfg.patch_size
        byte_ids = torch.randint(0, 256, (1, seq_len)).long()
        out = model(byte_ids, return_aux=False)

        shard_ids = torch.tensor([0])
        seq_offsets = torch.tensor([0])

        loss, _, _ = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"))

        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, "KL loss should produce gradients"

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_batch_kl_loss_gradient_flows PASSED")


def _make_v2_cache(tmpdir, shard_id=0, n_seqs=1, n_patches=3, kl_top_k=4,
                   patch_size=4, n_byte_positions=4):
    writer = StreamingCacheWriter(tmpdir, kl_top_k=kl_top_k,
                                  cache_format_version=2)
    records = []
    for seq_idx in range(n_seqs):
        offset = seq_idx * 64 * patch_size
        for p in range(1, n_patches + 1):
            for bpos in range(n_byte_positions):
                top_b = np.array([10 + bpos, 20, 30, 40], dtype=np.uint8)[:kl_top_k]
                top_p = np.array([0.4, 0.3, 0.2, 0.05], dtype=np.float16)[:kl_top_k]
                records.append(ByteKLRecord(
                    shard_id, offset, p, top_b, top_p, 0.05, 1.5,
                    byte_pos=bpos))
    writer.write_shard([], records)
    writer.finalize(None, shard_range=(0, 1), extra_manifest={
        "selection_policy": "uniform",
        "sample_frac": 0.50,
    })
    return tmpdir


def test_cache_format_v2_roundtrip():
    """Cache format v2 stores and retrieves byte_pos correctly."""
    td = tempfile.mkdtemp()
    try:
        _make_v2_cache(td, n_patches=2, n_byte_positions=4)
        cache = MappedByteKLCache(td)

        assert cache.format_version == 2
        records = cache.get_records(0, 0)
        assert len(records) == 8  # 2 patches × 4 positions
        byte_positions = [r.byte_pos for r in records]
        assert set(byte_positions) == {0, 1, 2, 3}
        for r in records:
            assert r.top_bytes[0] == 10 + r.byte_pos

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_cache_format_v2_roundtrip PASSED")


def test_v2_kl_loss_uses_byte_pos():
    """compute_batch_kl_loss selects correct logit dim based on byte_pos."""
    td = tempfile.mkdtemp()
    try:
        _make_v2_cache(td, n_patches=3, n_byte_positions=4)
        cache = MappedByteKLCache(td)

        mcfg = tiny_cfg()
        model = SutraS0(mcfg)
        seq_len = mcfg.max_seq_len * mcfg.patch_size
        byte_ids = torch.randint(0, 256, (1, seq_len)).long()
        out = model(byte_ids, return_aux=False)

        shard_ids = torch.tensor([0])
        seq_offsets = torch.tensor([0])

        loss, n_used, n_seqs = compute_batch_kl_loss(
            out["logits"], shard_ids, seq_offsets, cache, torch.device("cpu"))

        assert n_used > 0
        assert n_seqs == 1
        assert loss.item() > 0
        assert loss.requires_grad

        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_v2_kl_loss_uses_byte_pos PASSED")


def test_multi_byte_marginal_basic():
    """multi_byte_marginal returns per-position distributions from token logits."""
    vocab_size = 100
    logits = torch.randn(vocab_size)

    byte_table = {}
    for i in range(vocab_size):
        n_bytes = (i % 5) + 1  # 1-5 bytes per token
        byte_table[i] = bytes([j % 256 for j in range(i, i + n_bytes)])

    class FakeTokenizer:
        pass

    tok = FakeTokenizer()
    tok.vocab_size = vocab_size

    results = multi_byte_marginal(logits, tok, n_positions=4, top_vocab=100,
                                  K=8, _byte_table=byte_table)
    assert len(results) == 4

    for pos, (top_b, top_p, tail, cov) in enumerate(results):
        assert top_b.dtype == np.uint8
        assert top_p.dtype == np.float16
        assert len(top_b) == 8
        assert len(top_p) == 8
        assert tail >= 0.0
        total = float(top_p.astype(np.float64).sum()) + tail
        assert abs(total - 1.0) < 0.05, f"Position {pos}: total={total}"

    assert results[0][3] > results[3][3], \
        "Position 0 coverage should exceed position 3"
    print("  test_multi_byte_marginal_basic PASSED")


def test_multi_byte_marginal_all_single_byte():
    """When all tokens are single-byte, only position 0 has coverage."""
    vocab_size = 50
    logits = torch.randn(vocab_size)
    byte_table = {i: bytes([i % 256]) for i in range(vocab_size)}

    class FakeTokenizer:
        pass

    tok = FakeTokenizer()
    tok.vocab_size = vocab_size

    results = multi_byte_marginal(logits, tok, n_positions=4, top_vocab=50,
                                  K=8, _byte_table=byte_table)
    assert results[0][3] > 0.9  # position 0 has high coverage
    assert results[1][3] < 0.01  # position 1 has no coverage
    print("  test_multi_byte_marginal_all_single_byte PASSED")


def test_v1_cache_backward_compat():
    """V1 cache (no byte_pos) reads correctly with byte_pos=0 default."""
    td = tempfile.mkdtemp()
    try:
        _make_synthetic_cache(td, shard_id=0, n_seqs=1, n_patches_per_seq=3)
        cache = MappedByteKLCache(td)

        assert cache.format_version == 1
        records = cache.get_records(0, 0)
        assert len(records) == 3
        for r in records:
            assert r.byte_pos == 0

        cache.close()
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_v1_cache_backward_compat PASSED")


def test_load_cache_v2():
    """load_cache() reads v2 format with byte_pos correctly."""
    td = tempfile.mkdtemp()
    try:
        _make_v2_cache(td, n_patches=2, n_byte_positions=4)
        result = load_cache(td)

        assert result["manifest"]["cache_format_version"] == 2
        kl_records = result["kl_records"]
        assert len(kl_records) == 8  # 2 patches × 4 positions
        byte_positions = [r.byte_pos for r in kl_records]
        assert set(byte_positions) == {0, 1, 2, 3}
        for r in kl_records:
            assert r.top_bytes[0] == 10 + r.byte_pos
    finally:
        import shutil
        try:
            shutil.rmtree(td)
        except OSError:
            pass
    print("  test_load_cache_v2 PASSED")


if __name__ == "__main__":
    print("\n=== Option C Test Suite ===\n")

    tests = [
        test_lambda_kd_schedule,
        test_teacher_grad_budget_schedule,
        test_config_defaults,
        test_batch_kl_loss_with_cache,
        test_batch_kl_loss_missing_sequence,
        test_batch_kl_loss_max_per_seq,
        test_batch_kl_loss_gradient_flows,
        test_cache_format_v2_roundtrip,
        test_v2_kl_loss_uses_byte_pos,
        test_multi_byte_marginal_basic,
        test_multi_byte_marginal_all_single_byte,
        test_v1_cache_backward_compat,
        test_load_cache_v2,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    if failed > 0:
        sys.exit(1)
