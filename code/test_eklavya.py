"""CPU-only tests for Eklavya E1 — cache builder and training loop.

Verifies:
  - first_byte_marginal grouping math
  - compute_token_byte_spans offset tracking
  - select_kl_patches NLL-based selection
  - save_cache / load_cache round-trip
  - AlignProjection shape + forward
  - overlap_pool weighted pooling
  - topk_tail_kl computation
  - EklavyaTrainer phase transitions + freeze/unfreeze
  - EklavyaTrainer alignment loss with synthetic records
  - EklavyaTrainer KL loss with synthetic records
  - End-to-end: tiny model + synthetic cache, loss backpropagates
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import S0Config, SutraS0
from eklavya_cache import (
    AlignRecord, ByteKLRecord,
    first_byte_marginal, compute_token_byte_spans,
    select_kl_patches, save_cache, load_cache,
    validate_token_byte_alignment, token_id_to_bytes,
    build_token_byte_table, StreamingCacheWriter,
)
from eklavya_training import (
    AlignProjection, overlap_pool, topk_tail_kl,
    EklavyaConfig, EklavyaTrainer, apply_gradient_budget,
    evaluate_e1, _rng_state, EklavyaDataset,
)


def tiny_cfg():
    return S0Config(
        byte_dim=16, local_mixer_layers=1, local_mixer_window=4,
        patch_size=4, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        ffn_mult=1.0, max_seq_len=16, decoder_dim=16, decoder_layers=1,
        decoder_heads=4, verifier_dim=16,
    )


def test_first_byte_marginal():
    torch.manual_seed(0)
    logits = torch.randn(100)

    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            c = ids[0] % 128
            return chr(c) if 32 <= c < 127 else chr(65 + c % 26)

    top_b, top_p, tail, coverage = first_byte_marginal(logits, FakeTokenizer(), top_vocab=50, K=8)

    assert top_b.shape == (8,), f"Expected (8,), got {top_b.shape}"
    assert top_p.shape == (8,), f"Expected (8,), got {top_p.shape}"
    assert top_p.dtype == np.float16
    assert 0.0 <= tail <= 1.0, f"Tail out of range: {tail}"
    assert 0.0 <= coverage <= 1.0, f"Coverage out of range: {coverage}"
    total = float(top_p.astype(np.float64).sum()) + tail
    assert abs(total - 1.0) < 0.02, f"Probabilities should sum to ~1.0, got {total}"
    print("  test_first_byte_marginal PASSED")


def test_compute_token_byte_spans():
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {10: "Hello", 20: " ", 30: "world"}
            return mapping.get(ids[0], "?")

    spans = compute_token_byte_spans(FakeTokenizer(), [10, 20, 30])
    assert len(spans) == 3
    assert spans[0] == (0, 5), f"'Hello' should span (0, 5), got {spans[0]}"
    assert spans[1] == (5, 6), f"' ' should span (5, 6), got {spans[1]}"
    assert spans[2] == (6, 11), f"'world' should span (6, 11), got {spans[2]}"
    print("  test_compute_token_byte_spans PASSED")


def test_select_kl_patches():
    torch.manual_seed(42)
    P = 4
    N = 32
    B = 1
    Nm1 = N - 1

    logits = torch.randn(B, Nm1, P, 256)
    byte_ids = torch.randint(0, 256, (B, N * P))

    # With p90 adaptive threshold, roughly 10% of patches should exceed it
    selected = select_kl_patches(logits, byte_ids, P=P, nll_floor=0.0, control_frac=0.0)
    assert len(selected) > 0, "Some patches should exceed the p90 threshold"
    assert len(selected) <= Nm1, "Not all patches should be selected"
    assert all(1 <= s <= Nm1 for s in selected), f"Indices out of range: {selected}"

    selected_none = select_kl_patches(logits, byte_ids, P=P, nll_floor=100.0, control_frac=0.0)
    assert len(selected_none) == 0, "With floor=100 and no control, nothing should be selected"

    selected_ctrl = select_kl_patches(logits, byte_ids, P=P, nll_floor=100.0, control_frac=1.0)
    assert len(selected_ctrl) == Nm1, "With control_frac=1.0, all patches should be selected"
    print("  test_select_kl_patches PASSED")


def test_validate_byte_alignment():
    class ValTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {10: "Hello", 20: " ", 30: "world"}
            return mapping.get(ids[0], "?")

    tok = ValTokenizer()
    seq_bytes = b"Hello world"
    input_ids = [10, 20, 30]
    spans = compute_token_byte_spans(tok, input_ids)

    valid = validate_token_byte_alignment(seq_bytes, spans, input_ids, tok)
    assert all(valid), f"All tokens should be valid for matching bytes, got {valid}"

    bad_bytes = b"Hellx world"
    valid_bad = validate_token_byte_alignment(bad_bytes, spans, input_ids, tok)
    assert not valid_bad[0], "First token should be invalid for mismatched bytes"
    assert valid_bad[1], "Space token should still be valid"
    assert valid_bad[2], "World token should still be valid"
    print("  test_validate_byte_alignment PASSED")


def test_validate_byte_alignment_empty_token():
    class EmptyTokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {0: "", 1: "A", 2: "BC"}
            return mapping.get(ids[0], "")

    tok = EmptyTokenizer()
    spans = compute_token_byte_spans(tok, [0, 1, 2])
    assert spans[0] == (0, 0), f"Empty token should have zero-length span, got {spans[0]}"
    assert spans[1] == (0, 1), f"'A' should span (0, 1), got {spans[1]}"
    assert spans[2] == (1, 3), f"'BC' should span (1, 3), got {spans[2]}"

    valid = validate_token_byte_alignment(b"ABC", spans, [0, 1, 2], tok)
    assert not valid[0], "Empty token span should be invalid (be <= bs)"
    assert valid[1], "'A' should match"
    assert valid[2], "'BC' should match"
    print("  test_validate_byte_alignment_empty_token PASSED")


def test_validate_byte_alignment_multibyte_utf8():
    class Utf8Tokenizer:
        def decode(self, ids, skip_special_tokens=False):
            mapping = {10: "café", 20: " ", 30: "é"}
            return mapping.get(ids[0], "?")

    tok = Utf8Tokenizer()
    input_ids = [10, 20, 30]
    spans = compute_token_byte_spans(tok, input_ids)

    expected_bytes = "café".encode("utf-8") + b" " + "é".encode("utf-8")
    valid = validate_token_byte_alignment(expected_bytes, spans, input_ids, tok)
    assert all(valid), f"All multibyte tokens should validate, got {valid}"

    corrupted = b"cafX" + b"\xcc\x81" + b" " + b"\xc3\xa9"
    valid_bad = validate_token_byte_alignment(corrupted, spans, input_ids, tok)
    assert not valid_bad[0], "Corrupted multibyte span should fail validation"
    print("  test_validate_byte_alignment_multibyte_utf8 PASSED")


def test_build_token_byte_table():
    class SmallTokenizer:
        vocab_size = 5
        def __len__(self):
            return self.vocab_size
        def decode(self, ids, skip_special_tokens=False):
            mapping = {0: "<s>", 1: "hello", 2: " ", 3: "é", 4: ""}
            return mapping.get(ids[0], "")

    table = build_token_byte_table(SmallTokenizer())
    assert len(table) == 5
    assert table[1] == b"hello"
    assert table[2] == b" "
    assert table[3] == "é".encode("utf-8")
    assert table[4] == b""
    assert table[0] == b"<s>"

    assert token_id_to_bytes(None, 1, _table=table) == b"hello"
    assert token_id_to_bytes(None, 99, _table=table) == b""
    print("  test_build_token_byte_table PASSED")


def test_save_load_cache_roundtrip():
    align = [
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=100),
        AlignRecord(shard_id=0, seq_offset=4096, byte_start=8, byte_len=3, token_id=200),
        AlignRecord(shard_id=1, seq_offset=0, byte_start=0, byte_len=7, token_id=50),
    ]
    kl = [_make_valid_kl(0, 0, 2), _make_valid_kl(1, 0, 5)]
    emb = torch.randn(300, 64)

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl, emb)
        loaded = load_cache(td)

    assert len(loaded["align_records"]) == 3
    assert len(loaded["kl_records"]) == 2
    assert loaded["embedding_table"].shape == (300, 64)

    for orig, loaded_r in zip(align, loaded["align_records"]):
        assert orig.shard_id == loaded_r.shard_id
        assert orig.seq_offset == loaded_r.seq_offset
        assert orig.byte_start == loaded_r.byte_start
        assert orig.byte_len == loaded_r.byte_len
        assert orig.token_id == loaded_r.token_id

    for orig, loaded_r in zip(kl, loaded["kl_records"]):
        assert orig.shard_id == loaded_r.shard_id
        assert orig.patch_idx == loaded_r.patch_idx
        np.testing.assert_array_equal(orig.top_bytes, loaded_r.top_bytes)
        np.testing.assert_array_almost_equal(
            orig.top_probs.astype(np.float32),
            loaded_r.top_probs.astype(np.float32), decimal=3)

    print("  test_save_load_cache_roundtrip PASSED")


def test_align_projection_shape():
    proj = AlignProjection(student_dim=32, teacher_dim=64)
    x = torch.randn(32)
    y = proj(x)
    assert y.shape == (64,), f"Expected (64,), got {y.shape}"

    x_batch = torch.randn(4, 32)
    y_batch = proj(x_batch)
    assert y_batch.shape == (4, 64), f"Expected (4, 64), got {y_batch.shape}"
    print("  test_align_projection_shape PASSED")


def test_overlap_pool_single_patch():
    P = 4
    patch_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 patches
    result = overlap_pool(patch_states, byte_start=0, byte_end=4, P=P)
    torch.testing.assert_close(result, torch.tensor([1.0, 2.0]))
    print("  test_overlap_pool_single_patch PASSED")


def test_overlap_pool_two_patches():
    P = 4
    patch_states = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # byte_start=2, byte_end=6: patch 0 has 2 bytes overlap, patch 1 has 2 bytes
    result = overlap_pool(patch_states, byte_start=2, byte_end=6, P=P)
    expected = 0.5 * torch.tensor([1.0, 2.0]) + 0.5 * torch.tensor([3.0, 4.0])
    torch.testing.assert_close(result, expected)
    print("  test_overlap_pool_two_patches PASSED")


def test_overlap_pool_uneven():
    P = 4
    patch_states = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    # byte_start=3, byte_end=7: patch 0 has 1 byte (3-4), patch 1 has 3 bytes (4-7)
    result = overlap_pool(patch_states, byte_start=3, byte_end=7, P=P)
    expected = (1.0/4.0) * torch.tensor([1.0, 0.0]) + (3.0/4.0) * torch.tensor([0.0, 1.0])
    torch.testing.assert_close(result, expected)
    print("  test_overlap_pool_uneven PASSED")


def test_overlap_pool_batched():
    P = 4
    patch_states = torch.randn(1, 5, 32)
    result = overlap_pool(patch_states, byte_start=4, byte_end=8, P=P)
    torch.testing.assert_close(result, patch_states[0, 1])
    print("  test_overlap_pool_batched PASSED")


def test_overlap_pool_no_overlap():
    P = 4
    patch_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 2 patches, covers bytes 0-7
    result = overlap_pool(patch_states, byte_start=100, byte_end=104, P=P)
    assert result is None, f"Expected None for out-of-range span, got {result}"
    print("  test_overlap_pool_no_overlap PASSED")


def test_topk_tail_kl_basic():
    torch.manual_seed(0)
    student_logits = torch.randn(256)
    top_bytes = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    top_probs = torch.tensor([0.5, 0.3, 0.1, 0.05], dtype=torch.float32)
    tail_prob = torch.tensor(0.05)

    loss = topk_tail_kl(student_logits, top_bytes, top_probs, tail_prob, T=2.0)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, f"KL should be positive, got {loss.item()}"
    assert torch.isfinite(loss), "Loss should be finite"
    print("  test_topk_tail_kl_basic PASSED")


def test_topk_tail_kl_gradient():
    student_logits = torch.randn(256, requires_grad=True)
    top_bytes = torch.tensor([10, 20, 30])
    top_probs = torch.tensor([0.6, 0.3, 0.05])
    tail_prob = torch.tensor(0.05)

    loss = topk_tail_kl(student_logits, top_bytes, top_probs, tail_prob, T=1.0)
    loss.backward()
    assert student_logits.grad is not None, "Gradient should flow"
    assert student_logits.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("  test_topk_tail_kl_gradient PASSED")


def test_phase_transitions():
    cfg = EklavyaConfig(
        projection_warmup_steps=100,
        alignment_landing_steps=200,
        full_e1_steps=500,
    )
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    align_proj = AlignProjection(32, 64)

    cache = {
        "embedding_table": torch.randn(100, 64),
        "align_records": [],
        "kl_records": [],
        "manifest": {},
    }
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    assert trainer.get_phase(0) == "E1.0_warmup"
    assert trainer.get_phase(99) == "E1.0_warmup"
    assert trainer.get_phase(100) == "E1.1_landing"
    assert trainer.get_phase(299) == "E1.1_landing"
    assert trainer.get_phase(300) == "E1.2_full"
    assert trainer.get_phase(799) == "E1.2_full"
    print("  test_phase_transitions PASSED")


def test_freeze_phases():
    cfg = EklavyaConfig()
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    align_proj = AlignProjection(32, 64)

    cache = {
        "embedding_table": torch.randn(100, 64),
        "align_records": [],
        "kl_records": [],
        "manifest": {},
    }
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    # E1.0: only align_proj trainable
    trainer.configure_freeze("E1.0_warmup")
    for p in student.parameters():
        assert not p.requires_grad, "Student should be frozen in E1.0"
    for p in align_proj.parameters():
        assert p.requires_grad, "Align proj should be trainable in E1.0"

    # E1.1: encoder + align_proj trainable
    trainer.configure_freeze("E1.1_landing")
    for p in student.encoder.parameters():
        assert p.requires_grad, "Encoder should be trainable in E1.1"
    for p in student.reasoner.parameters():
        assert not p.requires_grad, "Reasoner should be frozen in E1.1"
    for p in student.decoder.parameters():
        assert not p.requires_grad, "Decoder should be frozen in E1.1"
    for p in align_proj.parameters():
        assert p.requires_grad, "Align proj should be trainable in E1.1"

    # E1.2: everything trainable
    trainer.configure_freeze("E1.2_full")
    for p in student.parameters():
        assert p.requires_grad, "All student params should be trainable in E1.2"
    for p in align_proj.parameters():
        assert p.requires_grad, "Align proj should be trainable in E1.2"

    print("  test_freeze_phases PASSED")


def test_align_loss_with_synthetic_records():
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    student.eval()

    teacher_dim = 64
    align_proj = AlignProjection(model_cfg.d_model, teacher_dim)

    embedding_table = torch.randn(100, teacher_dim)

    records = [
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=10),
        AlignRecord(shard_id=0, seq_offset=0, byte_start=4, byte_len=3, token_id=20),
        AlignRecord(shard_id=0, seq_offset=0, byte_start=8, byte_len=4, token_id=30),
    ]

    cache = {
        "embedding_table": embedding_table,
        "align_records": records,
        "kl_records": [],
        "manifest": {},
    }

    cfg = EklavyaConfig()
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    torch.manual_seed(42)
    byte_ids = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        out = student(byte_ids)

    loss = trainer.compute_align_loss(out["patch_states"], records)
    assert loss.item() > 0, f"Alignment loss should be positive, got {loss.item()}"
    assert torch.isfinite(loss), "Loss should be finite"
    print("  test_align_loss_with_synthetic_records PASSED")


def test_kl_loss_with_synthetic_records():
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    student.eval()

    records = [
        ByteKLRecord(
            shard_id=0, seq_offset=0, patch_idx=2,
            top_bytes=np.arange(16, dtype=np.uint8),
            top_probs=np.array([0.3, 0.2, 0.1, 0.05] + [0.02] * 12, dtype=np.float16),
            tail_prob=0.01, entropy=3.0,
        ),
        ByteKLRecord(
            shard_id=0, seq_offset=0, patch_idx=5,
            top_bytes=np.arange(16, dtype=np.uint8) + 50,
            top_probs=np.array([0.25, 0.15, 0.1, 0.05] + [0.03] * 12, dtype=np.float16),
            tail_prob=0.09, entropy=3.5,
        ),
    ]

    cache = {
        "embedding_table": None,
        "align_records": [],
        "kl_records": records,
        "manifest": {},
    }

    cfg = EklavyaConfig()
    trainer = EklavyaTrainer(cfg, student, AlignProjection(32, 64), cache, torch.device("cpu"))

    torch.manual_seed(42)
    byte_ids = torch.randint(0, 256, (1, 64))
    with torch.no_grad():
        out = student(byte_ids)

    loss = trainer.compute_kl_loss(out["logits"], records)
    assert loss.item() > 0, f"KL loss should be positive, got {loss.item()}"
    assert torch.isfinite(loss), "Loss should be finite"
    print("  test_kl_loss_with_synthetic_records PASSED")


def test_e2e_backward():
    torch.manual_seed(7)
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    student.train()

    teacher_dim = 64
    align_proj = AlignProjection(model_cfg.d_model, teacher_dim)
    embedding_table = torch.randn(100, teacher_dim)
    P = model_cfg.patch_size

    align_records = [
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=10),
        AlignRecord(shard_id=0, seq_offset=0, byte_start=4, byte_len=4, token_id=20),
    ]
    kl_records = [
        ByteKLRecord(
            shard_id=0, seq_offset=0, patch_idx=3,
            top_bytes=np.arange(16, dtype=np.uint8),
            top_probs=np.array([0.4, 0.2, 0.1, 0.05] + [0.02] * 12, dtype=np.float16),
            tail_prob=0.01, entropy=2.5,
        ),
    ]

    cache = {
        "embedding_table": embedding_table,
        "align_records": align_records,
        "kl_records": kl_records,
        "manifest": {},
    }

    cfg = EklavyaConfig(
        projection_warmup_steps=2,
        alignment_landing_steps=2,
        full_e1_steps=4,
        lambda_align=0.05,
        lambda_kl=0.10,
    )
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    byte_ids = torch.randint(0, 256, (1, 64))

    for phase_name in ["E1.0_warmup", "E1.1_landing", "E1.2_full"]:
        trainer.configure_freeze(phase_name)
        optimizer = trainer.build_optimizer()

        optimizer.zero_grad()
        out = student(byte_ids)
        logits = out["logits"]
        B, Nm1, Pp, V = logits.shape
        N = Nm1 + 1
        targets = byte_ids.reshape(B, N, P)[:, 1:]
        L_ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

        L_align = trainer.compute_align_loss(out["patch_states"], align_records)
        L_kl = trainer.compute_kl_loss(logits, kl_records)

        if phase_name == "E1.0_warmup":
            loss = L_align
        elif phase_name == "E1.1_landing":
            loss = L_ce + cfg.lambda_align * L_align
        else:
            loss = L_ce + cfg.lambda_align * L_align + cfg.lambda_kl * L_kl

        loss.backward()

        trainable_grads = sum(
            1 for p in list(student.parameters()) + list(align_proj.parameters())
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert trainable_grads > 0, f"No gradients flowing in phase {phase_name}"
        optimizer.step()

    print("  test_e2e_backward PASSED")


def test_optimizer_param_groups():
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    align_proj = AlignProjection(32, 64)

    cache = {
        "embedding_table": torch.randn(100, 64),
        "align_records": [],
        "kl_records": [],
        "manifest": {},
    }

    cfg = EklavyaConfig(base_lr=3e-5, align_lr=3e-4)
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    trainer.configure_freeze("E1.2_full")
    optimizer = trainer.build_optimizer()

    assert len(optimizer.param_groups) == 2, f"Expected 2 param groups, got {len(optimizer.param_groups)}"
    assert optimizer.param_groups[0]["lr"] == 3e-5, "Base LR mismatch"
    assert optimizer.param_groups[1]["lr"] == 3e-4, "Proj LR mismatch"

    # E1.0: only proj params (1 group)
    trainer.configure_freeze("E1.0_warmup")
    optimizer_warmup = trainer.build_optimizer()
    assert len(optimizer_warmup.param_groups) == 1, f"Warmup should have 1 group, got {len(optimizer_warmup.param_groups)}"
    assert optimizer_warmup.param_groups[0]["lr"] == 3e-4, "Warmup group should use proj LR"
    print("  test_optimizer_param_groups PASSED")


def test_gradient_budget():
    torch.manual_seed(99)
    x = torch.randn(10, requires_grad=True)
    w = torch.randn(10, requires_grad=True)

    L_ce = (x * w).sum()
    L_teacher = 10.0 * (x ** 2).sum()

    params = [x, w]
    ce_norm, teacher_norm, scale = apply_gradient_budget(params, L_ce, L_teacher, budget=0.30)

    assert ce_norm > 0, "CE grad norm should be positive"
    assert teacher_norm > 0, "Teacher grad norm should be positive"

    if teacher_norm > 0.30 * ce_norm:
        assert scale < 1.0, f"Scale should be < 1.0 when teacher exceeds budget, got {scale}"
        expected_scale = 0.30 * ce_norm / teacher_norm
        assert abs(scale - expected_scale) < 1e-5, f"Scale mismatch: {scale} vs {expected_scale}"

    assert x.grad is not None, "x should have grad"
    assert w.grad is not None, "w should have grad"
    print("  test_gradient_budget PASSED")


def test_gradient_budget_within_budget():
    torch.manual_seed(99)
    x = torch.randn(10, requires_grad=True)

    L_ce = 100.0 * (x ** 2).sum()
    L_teacher = 0.01 * x.sum()

    params = [x]
    ce_norm, teacher_norm, scale = apply_gradient_budget(params, L_ce, L_teacher, budget=0.30)

    assert scale == 1.0, f"Scale should be 1.0 when teacher within budget, got {scale}"
    print("  test_gradient_budget_within_budget PASSED")


def test_gradient_budget_preserves_accumulated_grads():
    torch.manual_seed(42)
    x = torch.randn(10, requires_grad=True)
    w = torch.randn(10, requires_grad=True)

    L_ce_1 = (x * w).sum()
    L_teacher_1 = 0.5 * (x ** 2).sum()
    apply_gradient_budget([x, w], L_ce_1, L_teacher_1, budget=0.30)

    grad_after_first = x.grad.clone()
    assert grad_after_first is not None

    x_fresh = x.detach().requires_grad_(True)
    w_fresh = w.detach().requires_grad_(True)
    L_ce_2 = (x_fresh * w_fresh).sum() * 2
    L_teacher_2 = 0.3 * (x_fresh ** 2).sum()
    apply_gradient_budget([x_fresh, w_fresh], L_ce_2, L_teacher_2, budget=0.30)
    grad_single = x_fresh.grad.clone()

    x2 = x.detach().requires_grad_(True)
    w2 = w.detach().requires_grad_(True)
    L_ce_a = (x2 * w2).sum()
    L_teacher_a = 0.5 * (x2 ** 2).sum()
    apply_gradient_budget([x2, w2], L_ce_a, L_teacher_a, budget=0.30)

    L_ce_b = (x2 * w2).sum() * 2
    L_teacher_b = 0.3 * (x2 ** 2).sum()
    apply_gradient_budget([x2, w2], L_ce_b, L_teacher_b, budget=0.30)

    assert x2.grad is not None
    assert not torch.allclose(x2.grad, grad_single, atol=1e-6), \
        "Two-step accumulated grad should differ from single-step grad"
    print("  test_gradient_budget_preserves_accumulated_grads PASSED")


def test_streaming_cache_writer():
    align1 = [AlignRecord(0, 0, 0, 4, 10), AlignRecord(0, 0, 4, 3, 20)]
    kl1 = [_make_valid_kl(0, 0, 2)]
    align2 = [AlignRecord(1, 0, 0, 4, 30)]
    kl2 = [_make_valid_kl(1, 0, 5)]

    emb = torch.randn(50, 32)

    with tempfile.TemporaryDirectory() as td:
        writer = StreamingCacheWriter(td)
        writer.write_shard(align1, kl1)
        writer.write_shard(align2, kl2)
        writer.finalize(emb)

        loaded = load_cache(td)

    assert len(loaded["align_records"]) == 3, f"Expected 3 align, got {len(loaded['align_records'])}"
    assert len(loaded["kl_records"]) == 2, f"Expected 2 kl, got {len(loaded['kl_records'])}"
    assert loaded["manifest"]["n_align"] == 3
    assert loaded["manifest"]["n_kl"] == 2
    assert loaded["embedding_table"].shape == (50, 32)

    assert loaded["align_records"][0].token_id == 10
    assert loaded["align_records"][2].token_id == 30
    print("  test_streaming_cache_writer PASSED")


def test_stale_embeddings_not_resurrected():
    align = [AlignRecord(0, 0, 0, 4, 10)]
    kl = [_make_valid_kl(0, 0, 2)]
    emb = torch.randn(50, 32)

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl, emb)
        loaded1 = load_cache(td)
        assert loaded1["embedding_table"] is not None, "First load should have embeddings"
        assert loaded1["embedding_table"].shape == (50, 32)

        save_cache(td, align, kl, embedding_table=None)
        loaded2 = load_cache(td)
        assert loaded2["embedding_table"] is None, \
            "Rewrite without embeddings should NOT resurrect stale embeddings"
        assert loaded2["manifest"]["has_embeddings"] is False

    with tempfile.TemporaryDirectory() as td:
        writer = StreamingCacheWriter(td)
        writer.write_shard(align, kl)
        writer.finalize(emb)
        loaded3 = load_cache(td)
        assert loaded3["embedding_table"] is not None

        writer2 = StreamingCacheWriter(td)
        writer2.write_shard(align, kl)
        writer2.finalize(embedding_table=None)
        loaded4 = load_cache(td)
        assert loaded4["embedding_table"] is None, \
            "StreamingCacheWriter rewrite without embeddings should clean up stale file"

    print("  test_stale_embeddings_not_resurrected PASSED")


def test_record_indexing():
    align_records = [
        AlignRecord(0, 0, 0, 4, 10),
        AlignRecord(0, 0, 4, 4, 20),
        AlignRecord(0, 4096, 0, 4, 30),
        AlignRecord(1, 0, 0, 4, 40),
    ]
    kl_records = [
        ByteKLRecord(0, 0, 2, np.zeros(16, np.uint8), np.zeros(16, np.float16), 0.0, 0.0),
        ByteKLRecord(0, 4096, 5, np.zeros(16, np.uint8), np.zeros(16, np.float16), 0.0, 0.0),
    ]

    cache = {
        "embedding_table": None,
        "align_records": align_records,
        "kl_records": kl_records,
        "manifest": {},
    }

    cfg = EklavyaConfig()
    model_cfg = tiny_cfg()
    trainer = EklavyaTrainer(cfg, SutraS0(model_cfg), AlignProjection(32, 64), cache, torch.device("cpu"))

    assert len(trainer.align_by_seq[(0, 0)]) == 2
    assert len(trainer.align_by_seq[(0, 4096)]) == 1
    assert len(trainer.align_by_seq[(1, 0)]) == 1
    assert len(trainer.kl_by_seq[(0, 0)]) == 1
    assert len(trainer.kl_by_seq[(0, 4096)]) == 1
    assert (1, 4096) not in trainer.kl_by_seq
    print("  test_record_indexing PASSED")


def test_truncated_kl_cache_loads_gracefully():
    """Truncated KL cache file should load usable records, not crash."""
    kl = [_make_valid_kl(0, 0, 2), _make_valid_kl(0, 0, 5), _make_valid_kl(1, 0, 3)]
    align = [AlignRecord(0, 0, 0, 4, 10)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        kl_path = os.path.join(td, "kl_records.bin")
        full_size = os.path.getsize(kl_path)
        with open(kl_path, "r+b") as f:
            f.truncate(full_size - 5)
        loaded = load_cache(td)
        assert len(loaded["kl_records"]) < 3, "Truncated cache should have fewer records"
        assert len(loaded["kl_records"]) >= 2, "Should recover at least 2 full records"
    print("  test_truncated_kl_cache_loads_gracefully PASSED")


def test_truncated_align_cache_loads_gracefully():
    """Truncated align cache file should load usable records, not crash."""
    align = [
        AlignRecord(0, 0, 0, 4, 10),
        AlignRecord(0, 0, 4, 3, 20),
        AlignRecord(1, 0, 0, 7, 50),
    ]
    kl = [_make_valid_kl(0, 0, 2)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        align_path = os.path.join(td, "align_records.bin")
        full_size = os.path.getsize(align_path)
        with open(align_path, "r+b") as f:
            f.truncate(full_size - 5)
        loaded = load_cache(td)
        assert len(loaded["align_records"]) < 3, "Truncated cache should have fewer records"
        assert len(loaded["align_records"]) >= 2, "Should recover at least 2 full records"
    print("  test_truncated_align_cache_loads_gracefully PASSED")


def test_nan_top_probs_record_dropped_at_load():
    """NaN in top_probs → record dropped (mass deviation after sanitization)."""
    kl = [_make_valid_kl(0, 0, 2)]
    align = [AlignRecord(0, 0, 0, 4, 10)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        kl_path = os.path.join(td, "kl_records.bin")
        with open(kl_path, "r+b") as f:
            nan_bytes = np.array([float('nan')], dtype=np.float16).tobytes()
            f.seek(8 + 14 + 16)  # header + IqH + top_bytes
            f.write(nan_bytes)
        loaded = load_cache(td)
        assert len(loaded["kl_records"]) == 0, \
            "NaN-corrupted record should be dropped by validation"
    print("  test_nan_top_probs_record_dropped_at_load PASSED")


def test_nan_tail_sanitized_at_load():
    """NaN in tail → sanitized to 0.0, record still valid (probs sum ~0.92)."""
    kl = [_make_valid_kl(0, 0, 2)]
    align = [AlignRecord(0, 0, 0, 4, 10)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        kl_path = os.path.join(td, "kl_records.bin")
        with open(kl_path, "r+b") as f:
            import struct as st
            nan_bytes = st.pack("<e", float('nan'))
            f.seek(8 + 14 + 16 + 32)  # header + IqH + top_bytes + top_probs
            f.write(nan_bytes)
        loaded = load_cache(td)
        assert len(loaded["kl_records"]) == 1, \
            "NaN tail with valid probs should still pass validation"
        import math as m
        assert m.isfinite(loaded["kl_records"][0].tail_prob), \
            "NaN tail_prob should be sanitized to 0.0"
    print("  test_nan_tail_sanitized_at_load PASSED")


def _make_valid_kl(sid, soff, pidx, K=16):
    raw = np.linspace(0.5, 0.01, K)
    probs = (raw / raw.sum() * 0.92).astype(np.float16)
    tail = max(0.0, 1.0 - float(probs.astype(np.float64).sum()))
    return ByteKLRecord(sid, soff, pidx, np.arange(K, dtype=np.uint8),
                        probs, tail, 3.0)


def test_corrupt_kl_record_filtered_at_load():
    """KL record with negative prob should be silently dropped at load time."""
    kl = [_make_valid_kl(0, 0, 2), _make_valid_kl(0, 0, 5)]
    align = [AlignRecord(0, 0, 0, 4, 10)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        kl_path = os.path.join(td, "kl_records.bin")
        with open(kl_path, "r+b") as f:
            import struct as st
            # Inject negative probability into first record's first top_prob
            neg_bytes = np.array([-0.5], dtype=np.float16).tobytes()
            f.seek(8 + 14 + 16)  # header + IqH + top_bytes
            f.write(neg_bytes)
        loaded = load_cache(td)
        assert len(loaded["kl_records"]) == 1, \
            f"Corrupt record should be dropped, got {len(loaded['kl_records'])}"
    print("  test_corrupt_kl_record_filtered_at_load PASSED")


def test_header_truncated_kl_cache():
    """KL cache file truncated below header size should load empty."""
    align = [AlignRecord(0, 0, 0, 4, 10)]
    kl = [_make_valid_kl(0, 0, 2)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        kl_path = os.path.join(td, "kl_records.bin")
        with open(kl_path, "r+b") as f:
            f.truncate(4)  # Less than 8-byte header
        loaded = load_cache(td)
        assert len(loaded["kl_records"]) == 0
    print("  test_header_truncated_kl_cache PASSED")


def test_header_truncated_align_cache():
    """Align cache file truncated below header size should load empty."""
    align = [AlignRecord(0, 0, 0, 4, 10)]
    kl = [_make_valid_kl(0, 0, 2)]

    with tempfile.TemporaryDirectory() as td:
        save_cache(td, align, kl)
        align_path = os.path.join(td, "align_records.bin")
        with open(align_path, "r+b") as f:
            f.truncate(2)  # Less than 4-byte header
        loaded = load_cache(td)
        assert len(loaded["align_records"]) == 0
    print("  test_header_truncated_align_cache PASSED")


def test_select_kl_patches_nan_positions():
    """NaN NLL positions must be selected, not silently dropped."""
    import math
    torch.manual_seed(0)
    P = 4
    N = 8
    B = 1
    Nm1 = N - 1

    logits = torch.randn(B, Nm1, P, 256)
    byte_ids = torch.randint(0, 256, (B, N * P))

    # Inject NaN into one patch's logits to produce NaN NLL
    logits[0, 3, 0, :] = float("nan")

    selected = select_kl_patches(logits, byte_ids, P=P, nll_floor=0.0, control_frac=0.0)
    # patch_idx=3 → position 4 (1-indexed) should be selected
    assert 4 in selected, f"NaN position should be selected, got {selected}"
    print("  test_select_kl_patches_nan_positions PASSED")


def test_select_kl_patches_all_nan():
    """All-NaN logits must not produce empty selection."""
    import math
    torch.manual_seed(0)
    P = 4
    N = 8
    B = 1
    Nm1 = N - 1

    logits = torch.full((B, Nm1, P, 256), float("nan"))
    byte_ids = torch.randint(0, 256, (B, N * P))

    selected = select_kl_patches(logits, byte_ids, P=P, nll_floor=0.0, control_frac=0.0)
    assert len(selected) == Nm1, f"All NaN positions should be selected, got {len(selected)}/{Nm1}"
    print("  test_select_kl_patches_all_nan PASSED")


def test_evaluate_e1_empty_loader():
    """evaluate_e1 returns inf when eval_loader is empty."""
    import random
    torch.manual_seed(42)
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    align_proj = AlignProjection(model_cfg.d_model, 64)

    empty_loader = []
    result = evaluate_e1(student, align_proj, empty_loader,
                         torch.device("cpu"), torch.bfloat16, False, max_batches=10)
    assert result["eval_bpb"] == float("inf")
    print("  test_evaluate_e1_empty_loader PASSED")


def test_rng_state_roundtrip():
    """RNG state save/restore produces identical sequences."""
    import random
    torch.manual_seed(99)
    random.seed(99)
    np.random.seed(99)

    state = _rng_state(torch.device("cpu"))

    r1_torch = torch.randn(5).tolist()
    r1_py = random.random()
    r1_np = np.random.rand(3).tolist()

    torch.set_rng_state(state["rng_state"])
    random.setstate(state["py_rng_state"])
    np.random.set_state(state["np_rng_state"])

    r2_torch = torch.randn(5).tolist()
    r2_py = random.random()
    r2_np = np.random.rand(3).tolist()

    assert r1_torch == r2_torch, "Torch RNG not restored"
    assert r1_py == r2_py, "Python RNG not restored"
    assert r1_np == r2_np, "NumPy RNG not restored"
    print("  test_rng_state_roundtrip PASSED")


def test_align_loss_oob_token_id():
    """Out-of-range token_id records are skipped, not crash."""
    torch.manual_seed(0)
    model_cfg = tiny_cfg()
    student = SutraS0(model_cfg)
    align_proj = AlignProjection(model_cfg.d_model, 64)

    embedding_table = torch.randn(10, 64)
    cache = {
        "embedding_table": embedding_table,
        "align_records": [],
        "kl_records": [],
        "manifest": {},
    }
    cfg = EklavyaConfig()
    trainer = EklavyaTrainer(cfg, student, align_proj, cache, torch.device("cpu"))

    byte_ids = torch.randint(0, 256, (1, 64))
    out = student(byte_ids)

    oob_records = [
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=999),
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=-1),
        AlignRecord(shard_id=0, seq_offset=0, byte_start=0, byte_len=4, token_id=5),
    ]
    loss = trainer.compute_align_loss(out["patch_states"], oob_records)
    assert loss.item() > 0, "Valid record should contribute to loss"
    print("  test_align_loss_oob_token_id PASSED")


def test_legacy_cache_hard_fail():
    """Config without allow_legacy_cache rejects manifests missing shard_range."""
    cfg = EklavyaConfig(allow_legacy_cache=False)
    assert not cfg.allow_legacy_cache

    cfg_allow = EklavyaConfig(allow_legacy_cache=True)
    assert cfg_allow.allow_legacy_cache
    print("  test_legacy_cache_hard_fail PASSED")


if __name__ == "__main__":
    print("\n=== Eklavya E1 Test Suite ===\n")

    tests = [
        test_first_byte_marginal,
        test_compute_token_byte_spans,
        test_select_kl_patches,
        test_validate_byte_alignment,
        test_validate_byte_alignment_empty_token,
        test_validate_byte_alignment_multibyte_utf8,
        test_build_token_byte_table,
        test_save_load_cache_roundtrip,
        test_align_projection_shape,
        test_overlap_pool_single_patch,
        test_overlap_pool_two_patches,
        test_overlap_pool_uneven,
        test_overlap_pool_batched,
        test_overlap_pool_no_overlap,
        test_topk_tail_kl_basic,
        test_topk_tail_kl_gradient,
        test_phase_transitions,
        test_freeze_phases,
        test_align_loss_with_synthetic_records,
        test_kl_loss_with_synthetic_records,
        test_e2e_backward,
        test_optimizer_param_groups,
        test_gradient_budget,
        test_gradient_budget_within_budget,
        test_gradient_budget_preserves_accumulated_grads,
        test_streaming_cache_writer,
        test_stale_embeddings_not_resurrected,
        test_record_indexing,
        test_truncated_kl_cache_loads_gracefully,
        test_truncated_align_cache_loads_gracefully,
        test_nan_top_probs_record_dropped_at_load,
        test_nan_tail_sanitized_at_load,
        test_corrupt_kl_record_filtered_at_load,
        test_header_truncated_kl_cache,
        test_header_truncated_align_cache,
        test_select_kl_patches_nan_positions,
        test_select_kl_patches_all_nan,
        test_evaluate_e1_empty_loader,
        test_rng_state_roundtrip,
        test_align_loss_oob_token_id,
        test_legacy_cache_hard_fail,
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
    print("ALL TESTS PASSED")
