"""Tests for burnin_verdict.py — pure-logic checks, no GPU needed.

Covers: check_hard_fails, check_soft_concerns, check_trajectory, load_log.
"""

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from burnin_verdict import check_hard_fails, check_soft_concerns, check_trajectory, load_log


# ── Helpers ──────────────────────────────────────────────────────────────

def _train(step, bpb=4.0, loss=None, grad_norm=0.5, **kw):
    entry = {"step": step, "bpb": bpb, "loss": loss if loss is not None else bpb, "grad_norm": grad_norm}
    entry.update(kw)
    return entry


def _eval(step, eval_bpb=4.0, **kw):
    entry = {"step": step, "eval_bpb": eval_bpb}
    entry.update(kw)
    return entry


def _healthy_train(n=50):
    return [_train(i * 10, bpb=7.5 - i * 0.06) for i in range(n)]


def _healthy_eval(n=5):
    return [_eval(i * 100, eval_bpb=7.8 - i * 0.6) for i in range(n)]


# ═══ check_hard_fails ════════════════════════════════════════════════════

class TestHardFails:
    def test_empty_train(self):
        fails = check_hard_fails([], [])
        assert any("No training steps" in f for f in fails)

    def test_nan_loss(self):
        train = [_train(10, loss=float("nan"))]
        fails = check_hard_fails(train, [])
        assert any("NaN/Inf loss" in f and "step 10" in f for f in fails)

    def test_inf_loss(self):
        train = [_train(20, loss=float("inf"))]
        fails = check_hard_fails(train, [])
        assert any("NaN/Inf loss" in f for f in fails)

    def test_nan_grad_norm(self):
        train = [_train(30, grad_norm=float("nan"))]
        fails = check_hard_fails(train, [])
        assert any("NaN/Inf grad_norm" in f for f in fails)

    def test_inf_grad_norm(self):
        train = [_train(40, grad_norm=float("inf"))]
        fails = check_hard_fails(train, [])
        assert any("NaN/Inf grad_norm" in f for f in fails)

    def test_eval_bpb_above_hard_fail(self):
        train = _healthy_train(10)
        evals = [_eval(500, eval_bpb=7.5)]
        fails = check_hard_fails(train, evals)
        assert any("7.0" in f and "Eval BPB" in f for f in fails)

    def test_eval_bpb_at_threshold_no_fail(self):
        train = _healthy_train(10)
        evals = [_eval(500, eval_bpb=7.0)]
        fails = check_hard_fails(train, evals)
        assert not any("Eval BPB" in f and "7.0" in f for f in fails)

    def test_data_leak_detection(self):
        train = [_train(500, bpb=3.0)]
        evals = [_eval(100, eval_bpb=7.0), _eval(500, eval_bpb=7.8)]
        fails = check_hard_fails(train, evals)
        assert any("data leak" in f.lower() or "eval bug" in f.lower() for f in fails)

    def test_no_data_leak_when_both_drop(self):
        train = [_train(500, bpb=4.0)]
        evals = [_eval(100, eval_bpb=6.0), _eval(500, eval_bpb=4.5)]
        fails = check_hard_fails(train, evals)
        assert not any("data leak" in f.lower() for f in fails)

    def test_pos0_near_random_while_others_learn(self):
        train = _healthy_train(5)
        evals = [_eval(500, eval_pos_acc=[0.005, 0.10, 0.08, 0.12])]
        fails = check_hard_fails(train, evals)
        assert any("Position 0" in f and "global context" in f for f in fails)

    def test_pos0_below_minimum(self):
        train = _healthy_train(5)
        evals = [_eval(500, eval_pos_acc=[0.02, 0.02, 0.02, 0.02])]
        fails = check_hard_fails(train, evals)
        assert any("Position 0 acc" in f and "minimum" in f for f in fails)

    def test_pos0_healthy_no_fail(self):
        train = _healthy_train(5)
        evals = [_eval(500, eval_pos_acc=[0.05, 0.08, 0.10, 0.12])]
        fails = check_hard_fails(train, evals)
        assert not any("Position 0" in f for f in fails)

    def test_byte_acc_below_1pct(self):
        train = _healthy_train(5)
        evals = [_eval(500, eval_byte_acc=0.005)]
        fails = check_hard_fails(train, evals)
        assert any("Byte accuracy" in f and "1%" in f for f in fails)

    def test_byte_acc_above_1pct_no_fail(self):
        train = _healthy_train(5)
        evals = [_eval(500, eval_byte_acc=0.02)]
        fails = check_hard_fails(train, evals)
        assert not any("Byte accuracy" in f for f in fails)

    def test_high_grad_norm_frequency(self):
        # >30% of steps with grad_norm > 100
        train = [_train(i, grad_norm=150) for i in range(40)]
        train += [_train(i + 40, grad_norm=0.5) for i in range(60)]
        fails = check_hard_fails(train, [])
        assert any("Grad norm > 100" in f for f in fails)

    def test_low_grad_norm_frequency_no_fail(self):
        train = [_train(i, grad_norm=150) for i in range(10)]
        train += [_train(i + 10, grad_norm=0.5) for i in range(90)]
        fails = check_hard_fails(train, [])
        assert not any("Grad norm > 100" in f for f in fails)

    def test_all_passing(self):
        train = _healthy_train(50)
        evals = [_eval(500, eval_bpb=4.0, eval_byte_acc=0.05,
                        eval_pos_acc=[0.05, 0.08, 0.10, 0.12])]
        fails = check_hard_fails(train, evals)
        assert fails == []

    def test_multiple_simultaneous_failures(self):
        train = [_train(10, loss=float("nan"), grad_norm=float("inf"))]
        evals = [_eval(500, eval_bpb=8.0, eval_byte_acc=0.001)]
        fails = check_hard_fails(train, evals)
        assert len(fails) >= 3

    def test_missing_optional_fields_no_crash(self):
        train = [{"step": 10, "bpb": 4.0}]
        evals = [{"step": 500, "eval_bpb": 4.5}]
        fails = check_hard_fails(train, evals)
        assert isinstance(fails, list)


# ═══ check_soft_concerns ═════════════════════════════════════════════════

class TestSoftConcerns:
    def test_eval_bpb_above_healthy_range(self):
        evals = [_eval(500, eval_bpb=5.3)]
        concerns = check_soft_concerns([], evals)
        assert any("above healthy" in c.lower() for c in concerns)

    def test_eval_bpb_in_healthy_range_no_concern(self):
        evals = [_eval(500, eval_bpb=4.2)]
        concerns = check_soft_concerns([], evals)
        assert not any("above healthy" in c.lower() for c in concerns)

    def test_moderate_train_eval_gap(self):
        train = [_train(500, bpb=4.0)]
        evals = [_eval(500, eval_bpb=4.5)]
        concerns = check_soft_concerns(train, evals)
        assert any("moderate" in c.lower() and "gap" in c.lower() for c in concerns)

    def test_high_train_eval_gap(self):
        train = [_train(500, bpb=3.0)]
        evals = [_eval(500, eval_bpb=4.0)]
        concerns = check_soft_concerns(train, evals)
        assert any("high" in c.lower() and "overfitting" in c.lower() for c in concerns)

    def test_small_gap_no_concern(self):
        train = [_train(500, bpb=4.0)]
        evals = [_eval(500, eval_bpb=4.2)]
        concerns = check_soft_concerns(train, evals)
        assert not any("gap" in c.lower() for c in concerns)

    def test_grad_clipping_frequent(self):
        train = [_train(i, grad_norm=1.5) for i in range(60)]
        train += [_train(i + 60, grad_norm=0.3) for i in range(40)]
        concerns = check_soft_concerns(train, [])
        assert any("clipping" in c.lower() for c in concerns)

    def test_grad_clipping_infrequent_no_concern(self):
        train = [_train(i, grad_norm=0.3) for i in range(80)]
        train += [_train(i + 80, grad_norm=1.5) for i in range(20)]
        concerns = check_soft_concerns(train, [])
        assert not any("clipping" in c.lower() for c in concerns)

    def test_eval_jitter_high(self):
        evals = [_eval(100, eval_bpb=6.0),
                 _eval(200, eval_bpb=5.5),
                 _eval(300, eval_bpb=5.9)]
        concerns = check_soft_concerns([], evals)
        assert any("jitter" in c.lower() for c in concerns)

    def test_eval_jitter_low_no_concern(self):
        evals = [_eval(100, eval_bpb=6.0),
                 _eval(200, eval_bpb=5.9),
                 _eval(300, eval_bpb=5.85)]
        concerns = check_soft_concerns([], evals)
        assert not any("jitter" in c.lower() for c in concerns)

    def test_no_concerns_healthy_run(self):
        train = [_train(i * 10, bpb=5.0 - i * 0.02, grad_norm=0.3) for i in range(50)]
        evals = [_eval(500, eval_bpb=4.2)]
        concerns = check_soft_concerns(train, evals)
        assert concerns == []

    def test_empty_inputs_no_crash(self):
        concerns = check_soft_concerns([], [])
        assert isinstance(concerns, list)


# ═══ check_trajectory ════════════════════════════════════════════════════

class TestTrajectory:
    def test_steep_to_smooth_shape(self):
        evals = [_eval(100, eval_bpb=7.5),
                 _eval(200, eval_bpb=5.5),
                 _eval(300, eval_bpb=4.8),
                 _eval(400, eval_bpb=4.5),
                 _eval(500, eval_bpb=4.3)]
        notes = check_trajectory([], evals)
        assert any("steep" in n.lower() and "smooth" in n.lower() for n in notes)

    def test_bpb_not_decreasing_first_half(self):
        evals = [_eval(100, eval_bpb=6.0),
                 _eval(200, eval_bpb=6.2),
                 _eval(300, eval_bpb=6.1),
                 _eval(400, eval_bpb=5.0)]
        notes = check_trajectory([], evals)
        assert any("not decreasing" in n.lower() for n in notes)

    def test_bpb_increasing_second_half(self):
        evals = [_eval(100, eval_bpb=7.0),
                 _eval(200, eval_bpb=5.0),
                 _eval(300, eval_bpb=5.5),
                 _eval(400, eval_bpb=6.0)]
        notes = check_trajectory([], evals)
        assert any("increasing" in n.lower() for n in notes)

    def test_smoothed_reversals_detected(self):
        evals = [_eval(i * 100, eval_bpb=bpb)
                 for i, bpb in enumerate([7.0, 5.5, 4.5, 4.0, 6.0, 5.8, 4.0])]
        notes = check_trajectory([], evals)
        assert any("reversal" in n.lower() for n in notes)

    def test_insufficient_evals_no_notes(self):
        evals = [_eval(100, eval_bpb=7.0), _eval(200, eval_bpb=5.0)]
        notes = check_trajectory([], evals)
        assert notes == []

    def test_single_eval_no_notes(self):
        notes = check_trajectory([], [_eval(100, eval_bpb=6.0)])
        assert notes == []

    def test_empty_evals_no_notes(self):
        notes = check_trajectory([], [])
        assert notes == []

    def test_monotonically_decreasing_no_reversals(self):
        evals = [_eval(i * 100, eval_bpb=7.0 - i * 0.3)
                 for i in range(8)]
        notes = check_trajectory([], evals)
        assert not any("reversal" in n.lower() for n in notes)


# ═══ load_log ════════════════════════════════════════════════════════════

class TestLoadLog:
    def test_splits_train_and_eval(self):
        entries = [
            json.dumps({"step": 10, "bpb": 6.5, "loss": 4.5, "grad_norm": 0.3}),
            json.dumps({"step": 50, "eval_bpb": 6.2}),
            json.dumps({"step": 100, "bpb": 5.0, "loss": 3.5, "grad_norm": 0.4}),
            json.dumps({"step": 100, "eval_bpb": 5.8}),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(entries) + "\n")
            path = f.name
        try:
            train, eval_ = load_log(path)
            assert len(train) == 2
            assert len(eval_) == 2
            assert train[0]["step"] == 10
            assert eval_[0]["eval_bpb"] == 6.2
        finally:
            os.unlink(path)

    def test_skips_empty_lines(self):
        entries = [
            "",
            json.dumps({"step": 10, "bpb": 6.0, "loss": 4.0, "grad_norm": 0.3}),
            "",
            json.dumps({"step": 50, "eval_bpb": 5.5}),
            "",
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(entries) + "\n")
            path = f.name
        try:
            train, eval_ = load_log(path)
            assert len(train) == 1
            assert len(eval_) == 1
        finally:
            os.unlink(path)

    def test_hard_fail_entry_exits(self):
        entries = [
            json.dumps({"step": 10, "bpb": 6.0, "loss": 4.0, "grad_norm": 0.3}),
            json.dumps({"step": 20, "HARD_FAIL": "non-finite CE loss", "phase": "s0"}),
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(entries) + "\n")
            path = f.name
        try:
            with pytest.raises(SystemExit) as exc_info:
                load_log(path)
            assert exc_info.value.code == 1
        finally:
            os.unlink(path)

    def test_empty_log(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("")
            path = f.name
        try:
            train, eval_ = load_log(path)
            assert train == []
            assert eval_ == []
        finally:
            os.unlink(path)


# ═══ Integration: verdict logic ══════════════════════════════════════════

class TestVerdictIntegration:
    """Simulate realistic logs and verify correct overall verdict."""

    def test_healthy_500step_burnin(self):
        train = [_train(i * 10, bpb=7.8 - i * 0.07, grad_norm=0.5) for i in range(50)]
        evals = [_eval(i * 100, eval_bpb=7.9 - i * 0.7,
                        eval_byte_acc=0.01 + i * 0.01,
                        eval_pos_acc=[0.03 + i * 0.01] * 4)
                 for i in range(1, 6)]
        hard = check_hard_fails(train, evals)
        soft = check_soft_concerns(train, evals)
        traj = check_trajectory(train, evals)
        assert hard == []
        assert isinstance(soft, list)
        assert isinstance(traj, list)

    def test_diverged_model(self):
        train = [_train(i, bpb=8.0, loss=float("nan")) for i in range(5)]
        evals = [_eval(500, eval_bpb=8.0, eval_byte_acc=0.003)]
        hard = check_hard_fails(train, evals)
        assert len(hard) >= 2
