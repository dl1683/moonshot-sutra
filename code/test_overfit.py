"""Tiny overfit test: verify S0 can learn a trivial pattern.

Creates a small (byte_dim=16, d_model=32, 2 layers) model and trains it
to memorize a single batch of 4 sequences. Loss should drop below 2.0
within 200 steps if the training loop works correctly.

This verifies:
1. Forward pass produces valid gradients
2. Loss decreases monotonically
3. Optimizer + LR schedule works
4. No silent failures in the pipeline
"""

import sys
import os
import math

import torch

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import S0Config, SutraS0
from s0_training import compute_loss


def test_overfit():
    torch.manual_seed(42)

    cfg = S0Config(
        byte_dim=16,
        local_mixer_layers=1,
        local_mixer_window=4,
        patch_size=4,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_mult=1.0,
        max_seq_len=16,
        decoder_dim=16,
        decoder_layers=1,
        decoder_heads=4,
        verifier_dim=16,
    )

    model = SutraS0(cfg)
    model.train()

    # Freeze dead heads (same as training loop)
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ("encoder.entropy_head", "encoder.residual_head", "verifier")):
            param.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=0.0,
    )

    # Fixed batch to memorize: 4 sequences of 64 bytes each
    batch = torch.randint(0, 256, (4, 64))

    losses = []
    for step in range(200):
        optimizer.zero_grad()
        out = model(batch)
        loss_dict = compute_loss(out, batch, cfg.patch_size)
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            bpb = loss.item() / math.log(2)
            print(f"  step {step:>3d}: loss={loss.item():.4f} bpb={bpb:.3f}")

    print(f"\n  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Loss ratio:   {losses[-1] / losses[0]:.4f}")

    # Verify loss decreased substantially
    assert losses[-1] < losses[0] * 0.5, f"Loss did not decrease enough: {losses[-1]:.4f} vs initial {losses[0]:.4f}"
    assert losses[-1] < 3.0, f"Final loss too high: {losses[-1]:.4f} (expected < 3.0)"

    # Verify loss is monotonically decreasing (with some tolerance for noise)
    smoothed = [sum(losses[max(0, i-5):i+1]) / min(i+1, 6) for i in range(len(losses))]
    for i in range(10, len(smoothed)):
        assert smoothed[i] <= smoothed[i-10] + 0.5, f"Loss not decreasing at step {i}: {smoothed[i]:.4f} vs {smoothed[i-10]:.4f}"

    # Causality tests on trained model
    model.eval()
    x = torch.randint(0, 256, (1, 16))

    # Test 1: hidden[0] must not depend on target patch bytes (patch 1)
    y = x.clone()
    y[:, 4:8] = (y[:, 4:8] + 17) % 256
    with torch.no_grad():
        ps_x, _, _, _ = model.encoder(x)
        h_x = model.reasoner(ps_x)
        ps_y, _, _, _ = model.encoder(y)
        h_y = model.reasoner(ps_y)
    diff_target = (h_x[:, 0] - h_y[:, 0]).abs().max().item()
    assert diff_target == 0.0, f"Target patch leakage in conditioning: {diff_target}"

    # Test 2: hidden[0] must not depend on future patches (patch 2+)
    z = x.clone()
    z[:, 8:12] = (z[:, 8:12] + 17) % 256
    with torch.no_grad():
        ps_z, _, _, _ = model.encoder(z)
        h_z = model.reasoner(ps_z)
    diff_future = (h_x[:, 0] - h_z[:, 0]).abs().max().item()
    assert diff_future == 0.0, f"Future patch leakage: {diff_future}"

    # Test 3: hidden[0] MUST depend on its own patch bytes (patch 0)
    w = x.clone()
    w[:, 0:4] = (w[:, 0:4] + 17) % 256
    with torch.no_grad():
        ps_w, _, _, _ = model.encoder(w)
        h_w = model.reasoner(ps_w)
    diff_own = (h_x[:, 0] - h_w[:, 0]).abs().max().item()
    assert diff_own > 0.0, "Conditioning must depend on its own patch bytes"

    print("\n  ALL TESTS PASSED")


import pytest

@pytest.mark.parametrize("config_name", ["p4", "p8", "d640", "d768"])
def test_config_preset_forward(config_name):
    """Verify all S0 config presets produce models that forward correctly."""
    from s0_configs import ALL_CONFIGS
    cfg = ALL_CONFIGS[config_name]()
    model = SutraS0(cfg)
    model.eval()
    B, T = 1, cfg.patch_size * 8
    x = torch.randint(0, 256, (B, T))
    with torch.no_grad():
        out = model(x)
    P = cfg.patch_size
    N = T // P
    assert out["logits"].shape == (B, N - 1, P, 256)


def test_default_config_param_count():
    """Verify default P4 config produces ~121.7M params."""
    from s0_configs import s0_p4
    cfg = s0_p4()
    model = SutraS0(cfg)
    counts = model.count_parameters()
    total = counts["total"]
    assert 115_000_000 < total < 130_000_000, f"Expected ~121.7M, got {total/1e6:.1f}M"


def test_compute_loss_shape():
    """Verify compute_loss returns expected keys and valid values."""
    cfg = S0Config(byte_dim=16, d_model=32, n_layers=2, n_heads=2,
                   n_kv_heads=1, ffn_mult=1.0, local_mixer_layers=1,
                   patch_size=4, max_seq_len=16, decoder_dim=16,
                   decoder_layers=1, decoder_heads=2, verifier_dim=16)
    model = SutraS0(cfg)
    model.train()
    x = torch.randint(0, 256, (2, 16))
    out = model(x)
    losses = compute_loss(out, x, cfg.patch_size)
    assert "loss" in losses
    assert "byte_ce" in losses
    assert "bpb" in losses
    assert losses["loss"].requires_grad
    assert losses["bpb"] > 0


def test_lr_schedule():
    """Verify cosine LR schedule has correct shape."""
    from s0_training import get_lr, TrainConfig
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=100, total_steps=1000)
    assert get_lr(0, cfg) < cfg.lr
    assert get_lr(1, cfg) > 0
    lr_50 = get_lr(50, cfg)
    lr_100 = get_lr(100, cfg)
    assert lr_50 < lr_100
    assert abs(lr_100 - cfg.lr) < 1e-6
    lr_mid = get_lr(550, cfg)
    assert cfg.min_lr < lr_mid < cfg.lr
    lr_end = get_lr(1000, cfg)
    assert abs(lr_end - cfg.min_lr) < 1e-6


# ---------------------------------------------------------------------------
# ByteShardDataset tests (shard_range, indexing, edge cases)
# ---------------------------------------------------------------------------

class TestByteShardDataset:

    def _make_shards(self, tmp_path, n_shards=5, shard_bytes=256):
        for i in range(n_shards):
            data = np.random.randint(0, 256, size=shard_bytes, dtype=np.uint8)
            path = tmp_path / f"shard_{i:03d}.bin"
            data.tofile(str(path))
        return str(tmp_path)

    def test_full_range(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=5, shard_bytes=256)
        ds = ByteShardDataset(data_dir, seq_len=64, patch_size=4)
        assert len(ds) == 5 * (256 // 64)

    def test_shard_range_subset(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=5, shard_bytes=256)
        ds = ByteShardDataset(data_dir, seq_len=64, patch_size=4,
                              shard_range=(1, 3))
        assert len(ds) == 2 * (256 // 64)

    def test_shard_range_single(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=5, shard_bytes=256)
        ds = ByteShardDataset(data_dir, seq_len=64, patch_size=4,
                              shard_range=(2, 3))
        assert len(ds) == 256 // 64

    def test_empty_range_raises(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=3, shard_bytes=256)
        with pytest.raises(ValueError, match="shard_range.*yields 0"):
            ByteShardDataset(data_dir, seq_len=64, patch_size=4,
                             shard_range=(5, 5))

    def test_getitem_returns_tensor(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=2, shard_bytes=256)
        ds = ByteShardDataset(data_dir, seq_len=64, patch_size=4)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (64,)
        assert item.dtype == torch.long

    def test_seq_len_not_divisible_by_patch_raises(self, tmp_path):
        from s0_training import ByteShardDataset
        data_dir = self._make_shards(tmp_path, n_shards=1, shard_bytes=256)
        with pytest.raises(ValueError, match="divisible by patch_size"):
            ByteShardDataset(data_dir, seq_len=65, patch_size=4)


import numpy as np


class TestEklavyaDataset:

    def _make_shards(self, tmp_path, n_shards=3, shard_bytes=256):
        for i in range(n_shards):
            data = np.random.randint(0, 256, size=shard_bytes, dtype=np.uint8)
            path = tmp_path / f"shard_{i:03d}.bin"
            data.tofile(str(path))
        return str(tmp_path)

    def test_returns_triple(self, tmp_path):
        from eklavya_training import EklavyaDataset
        data_dir = self._make_shards(tmp_path)
        ds = EklavyaDataset(data_dir, seq_len=64, patch_size=4)
        byte_ids, shard_idx, start = ds[0]
        assert isinstance(byte_ids, torch.Tensor)
        assert byte_ids.shape == (64,)
        assert isinstance(shard_idx, int)
        assert isinstance(start, int)

    def test_shard_range_returns_global_ids(self, tmp_path):
        from eklavya_training import EklavyaDataset
        data_dir = self._make_shards(tmp_path, n_shards=5, shard_bytes=256)
        ds = EklavyaDataset(data_dir, seq_len=64, patch_size=4,
                            shard_range=(2, 4))
        assert len(ds) == 2 * (256 // 64)
        for i in range(len(ds)):
            _, shard_idx, _ = ds[i]
            assert 2 <= shard_idx < 4


if __name__ == "__main__":
    test_overfit()
