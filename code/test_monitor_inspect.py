"""Tests for monitor, inspect_checkpoint, and s0_configs. No GPU needed."""

import io
import json
import math
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))


# ═══ monitor.format_time ════════════════════════════════════════════════

from monitor import (format_time, load_entries, detect_mode, display,
                     _e2_anomalies, _phase_boundary_checks)


class TestFormatTime:
    def test_seconds(self):
        assert format_time(45) == "45s"

    def test_minutes(self):
        assert format_time(150) == "2.5m"

    def test_hours(self):
        assert format_time(7200) == "2.0h"

    def test_boundary_60(self):
        assert format_time(60) == "1.0m"

    def test_boundary_3600(self):
        assert format_time(3600) == "1.0h"

    def test_zero(self):
        assert format_time(0) == "0s"


# ═══ monitor.load_entries ═══════════════════════════════════════════════

def _write_log(entries):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                    encoding="utf-8")
    for e in entries:
        if isinstance(e, str):
            f.write(e + "\n")
        else:
            f.write(json.dumps(e) + "\n")
    f.close()
    return f.name


class TestLoadEntries:
    def test_splits_train_eval(self):
        path = _write_log([
            {"step": 1, "bpb": 7.0},
            {"step": 100, "eval_bpb": 6.5},
            {"step": 2, "bpb": 6.8},
        ])
        try:
            train, eval_ = load_entries(path)
            assert len(train) == 2
            assert len(eval_) == 1
        finally:
            os.unlink(path)

    def test_hard_fail_goes_to_train(self):
        path = _write_log([
            {"step": 1, "bpb": 7.0},
            {"step": 50, "HARD_FAIL": "non-finite loss", "loss": float("nan")},
        ])
        try:
            train, eval_ = load_entries(path)
            assert len(train) == 2
            assert train[1]["HARD_FAIL"] == "non-finite loss"
            assert len(eval_) == 0
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        train, eval_ = load_entries("/nonexistent/log.jsonl")
        assert train == []
        assert eval_ == []

    def test_empty_lines_skipped(self):
        path = _write_log([
            "",
            {"step": 1, "bpb": 7.0},
            "",
        ])
        try:
            train, eval_ = load_entries(path)
            assert len(train) == 1
        finally:
            os.unlink(path)

    def test_ce_loss_entry_is_train(self):
        path = _write_log([
            {"step": 1, "ce_loss": 4.8, "phase": "PORT_WARMUP"},
        ])
        try:
            train, eval_ = load_entries(path)
            assert len(train) == 1
        finally:
            os.unlink(path)

    def test_eval_loss_entry_is_eval(self):
        path = _write_log([
            {"step": 500, "eval_loss": 4.2},
        ])
        try:
            train, eval_ = load_entries(path)
            assert len(eval_) == 1
        finally:
            os.unlink(path)

    def test_eval_loss_converted_to_bpb(self):
        path = _write_log([
            {"step": 500, "eval_loss": 4.5},
        ])
        try:
            _, eval_ = load_entries(path)
            assert "eval_bpb" in eval_[0]
            assert eval_[0]["eval_bpb"] == pytest.approx(4.5 / math.log(2))
        finally:
            os.unlink(path)

    def test_eval_bpb_not_overwritten(self):
        path = _write_log([
            {"step": 500, "eval_bpb": 6.0, "eval_loss": 4.0},
        ])
        try:
            _, eval_ = load_entries(path)
            assert eval_[0]["eval_bpb"] == pytest.approx(6.0)
        finally:
            os.unlink(path)


# ═══ monitor.detect_mode ═══════════════════════════════════════════════

class TestDetectMode:
    def test_s0_mode(self):
        train = [{"step": 1, "bpb": 7.0}, {"step": 2, "bpb": 6.9}]
        assert detect_mode(train) == "s0"

    def test_e2_mode_by_phase(self):
        train = [{"step": 1, "phase": "PORT_WARMUP", "ce_loss": 4.8}]
        assert detect_mode(train) == "e2"

    def test_e2_mode_by_teacher_losses(self):
        train = [{"step": 1, "teacher_losses_bits": {"t0": 3.5}}]
        assert detect_mode(train) == "e2"

    def test_e2_mode_by_ce_loss(self):
        train = [{"step": 1, "ce_loss": 4.8}]
        assert detect_mode(train) == "e2"

    def test_empty_train_defaults_s0(self):
        assert detect_mode([]) == "s0"


# ═══ monitor.display ═══════════════════════════════════════════════════

class TestDisplay:
    def test_s0_display_has_expected_sections(self, capsys):
        path = _write_log([
            {"step": 1, "bpb": 7.0, "grad_norm": 0.5},
            {"step": 100, "bpb": 6.5, "grad_norm": 0.4},
            {"step": 100, "eval_bpb": 6.8, "eval_byte_acc": 0.12},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "S0 Training Monitor" in out
            assert "7.000" in out
            assert "6.500" in out
            assert "Eval BPB" in out
        finally:
            os.unlink(path)

    def test_e2_display_has_expected_sections(self, capsys):
        path = _write_log([
            {"step": 1, "ce_loss": 4.8, "phase": "PORT_WARMUP",
             "teacher_losses_bits": {"t0_anchor": 3.5}, "elapsed": 60},
            {"step": 100, "ce_loss": 4.2, "phase": "CONSENSUS",
             "teacher_losses_bits": {"t0_anchor": 3.0}, "elapsed": 600},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "E2 Multi-Teacher KD Monitor" in out
            assert "Phase: CONSENSUS" in out
            assert "t0_anchor" in out
            assert "Phase transitions:" in out
        finally:
            os.unlink(path)

    def test_hard_fail_displayed(self, capsys):
        path = _write_log([
            {"step": 1, "bpb": 7.0},
            {"step": 50, "HARD_FAIL": "non-finite loss"},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "HARD FAIL" in out
            assert "non-finite loss" in out
        finally:
            os.unlink(path)

    def test_empty_log_no_crash(self, capsys):
        path = _write_log([])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "No entries" in out
        finally:
            os.unlink(path)

    def test_loss_spike_detected(self, capsys):
        path = _write_log([
            {"step": 1, "bpb": 7.0},
            {"step": 2, "bpb": 6.8},
            {"step": 3, "bpb": 7.5},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "Loss spike" in out
        finally:
            os.unlink(path)

    def test_high_grad_norm_anomaly(self, capsys):
        entries = [{"step": i, "bpb": 7.0 - i * 0.01, "grad_norm": 15.0}
                   for i in range(10)]
        path = _write_log(entries)
        try:
            display(path)
            out = capsys.readouterr().out
            assert "High grad norm" in out
        finally:
            os.unlink(path)

    def test_throughput_stats(self, capsys):
        path = _write_log([
            {"step": 1, "bpb": 7.0, "elapsed": 0},
            {"step": 100, "bpb": 6.5, "elapsed": 200},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "s/step" in out
        finally:
            os.unlink(path)

    def test_e2_route_stats_displayed(self, capsys):
        path = _write_log([
            {"step": 1, "ce_loss": 4.8, "phase": "DISAGREEMENT",
             "route_stats": {"mean_route_entropy": 0.7, "mean_jsd": 0.15,
                             "n_routed": 10}},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "Route entropy" in out
            assert "0.7" in out
        finally:
            os.unlink(path)

    def test_e2_grad_budget_displayed(self, capsys):
        path = _write_log([
            {"step": 1, "ce_loss": 4.8, "phase": "CONSENSUS",
             "grad_budget": {"ce_grad_norm": 1.2, "total_scale": 0.45}},
        ])
        try:
            display(path)
            out = capsys.readouterr().out
            assert "Gradient budget" in out
            assert "0.45" in out
        finally:
            os.unlink(path)


    def test_phase_boundary_anomaly_shows_in_display(self, capsys):
        entries = [
            {"step": i, "ce_loss": 4.0, "phase": "E2.4_disagreement",
             "route_stats": {"mean_route_entropy": 1.5}}
            for i in range(60)
        ]
        path = _write_log(entries)
        try:
            display(path)
            out = capsys.readouterr().out
            assert "DISAGREEMENT" in out
            assert "near-uniform" in out
        finally:
            os.unlink(path)


class TestE2Anomalies:
    def test_clean_log_no_anomalies(self):
        entries = [{"step": i, "ce_loss": 4.0 - i * 0.1} for i in range(20)]
        assert _e2_anomalies(entries) == []

    def test_nan_ce_loss_detected(self):
        entries = [{"step": i, "ce_loss": 4.0} for i in range(5)]
        entries.append({"step": 5, "ce_loss": float("nan")})
        anomalies = _e2_anomalies(entries)
        assert any("Non-finite CE loss" in a for a in anomalies)

    def test_nan_teacher_loss_detected(self):
        entries = [{"step": i, "ce_loss": 4.0,
                    "teacher_losses_nats": {"kl_purified": 0.5}}
                   for i in range(5)]
        entries.append({"step": 5, "ce_loss": 4.0,
                        "teacher_losses_nats": {"kl_purified": float("inf")}})
        anomalies = _e2_anomalies(entries)
        assert any("Non-finite teacher loss" in a for a in anomalies)

    def test_route_entropy_collapse(self):
        entries = [{"step": i, "ce_loss": 4.0,
                    "route_stats": {"mean_route_entropy": 0.01}}
                   for i in range(15)]
        anomalies = _e2_anomalies(entries)
        assert any("Route entropy collapse" in a for a in anomalies)

    def test_grad_budget_suppressed(self):
        entries = [{"step": i, "ce_loss": 4.0,
                    "grad_budget": {"total_scale": 0.001}}
                   for i in range(10)]
        anomalies = _e2_anomalies(entries)
        assert any("Gradient budget near zero" in a for a in anomalies)

    def test_zero_teacher_signal(self):
        entries = [{"step": i, "ce_loss": 4.0,
                    "teacher_losses_bits": {"kl_purified": 0.0}}
                   for i in range(15)]
        anomalies = _e2_anomalies(entries)
        assert any("Zero teacher signal" in a for a in anomalies)


# ═══ Phase boundary checks ═════════════════════════════════════════════

class TestPhaseBoundaryChecks:
    def test_clean_phases_no_anomalies(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.1_port_warmup"}
                 for i in range(10)]
        assert _phase_boundary_checks(train) == []

    def test_port_warmup_bpb_regression(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.1_port_warmup"}
                 for i in range(10)]
        eval_ = [
            {"step": 0, "eval_bpb": 5.0},
            {"step": 9, "eval_bpb": 5.1},
        ]
        anomalies = _phase_boundary_checks(train, eval_)
        assert any("PORT_WARMUP" in a and "regressed" in a for a in anomalies)

    def test_port_warmup_no_regression_when_small(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.1_port_warmup"}
                 for i in range(10)]
        eval_ = [
            {"step": 0, "eval_bpb": 5.0},
            {"step": 9, "eval_bpb": 5.01},
        ]
        anomalies = _phase_boundary_checks(train, eval_)
        assert not any("PORT_WARMUP" in a for a in anomalies)

    def test_consensus_low_n_routed(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.2_consensus",
                  "route_stats": {"n_routed": 2, "mean_route_entropy": 0.5}}
                 for i in range(10)]
        anomalies = _phase_boundary_checks(train)
        assert any("CONSENSUS" in a and "n_routed" in a for a in anomalies)

    def test_consensus_healthy_n_routed(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.2_consensus",
                  "route_stats": {"n_routed": 8, "mean_route_entropy": 0.5}}
                 for i in range(10)]
        anomalies = _phase_boundary_checks(train)
        assert not any("n_routed" in a for a in anomalies)

    def test_consensus_anchor_dominance(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.2_consensus",
                  "route_stats": {"avg_teacher_weights": {
                      "t0_anchor_decoder": 0.98, "t2_control_decoder": 0.02}}}
                 for i in range(10)]
        anomalies = _phase_boundary_checks(train)
        assert any("CONSENSUS" in a and "control" in a.lower() for a in anomalies)

    def test_semantic_loss_absent(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.3_semantic_landing",
                  "teacher_losses_bits": {"kl_purified_anchor": 3.0}}
                 for i in range(10)]
        anomalies = _phase_boundary_checks(train)
        assert any("SEMANTIC_LANDING" in a and "semantic" in a.lower()
                    for a in anomalies)

    def test_semantic_loss_present(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.3_semantic_landing",
                  "teacher_losses_bits": {
                      "kl_purified_anchor": 3.0,
                      "align_t3_semantic_embedding": 0.5}}
                 for i in range(10)]
        anomalies = _phase_boundary_checks(train)
        assert not any("SEMANTIC_LANDING" in a for a in anomalies)

    def test_disagreement_entropy_too_high(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.4_disagreement",
                  "route_stats": {"mean_route_entropy": 1.5}}
                 for i in range(60)]
        anomalies = _phase_boundary_checks(train)
        assert any("DISAGREEMENT" in a and "1.30" in a for a in anomalies)

    def test_disagreement_entropy_collapse(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.4_disagreement",
                  "route_stats": {"mean_route_entropy": 0.05}}
                 for i in range(60)]
        anomalies = _phase_boundary_checks(train)
        assert any("DISAGREEMENT" in a and "collapse" in a for a in anomalies)

    def test_disagreement_healthy_entropy(self):
        train = [{"step": i, "ce_loss": 4.0, "phase": "E2.4_disagreement",
                  "route_stats": {"mean_route_entropy": 0.8}}
                 for i in range(60)]
        anomalies = _phase_boundary_checks(train)
        assert not any("DISAGREEMENT" in a for a in anomalies)

    def test_insufficient_data_skips_checks(self):
        train = [{"step": 0, "ce_loss": 4.0, "phase": "E2.4_disagreement",
                  "route_stats": {"mean_route_entropy": 1.5}}]
        anomalies = _phase_boundary_checks(train)
        assert anomalies == []


# ═══ s0_configs ═════════════════════════════════════════════════════════

from s0_configs import s0_p4, s0_p8, s0_d640, s0_d768, ALL_CONFIGS
from s0_architecture import S0Config


class TestS0Configs:
    def test_all_configs_dict_has_four_entries(self):
        assert set(ALL_CONFIGS.keys()) == {"p4", "p8", "d640", "d768"}

    def test_all_configs_return_s0config(self):
        for name, fn in ALL_CONFIGS.items():
            cfg = fn()
            assert isinstance(cfg, S0Config), f"{name} did not return S0Config"

    def test_p4_defaults(self):
        cfg = s0_p4()
        assert cfg.patch_size == 4
        assert cfg.d_model == 576
        assert cfg.n_layers == 30

    def test_p8_patch_size(self):
        cfg = s0_p8()
        assert cfg.patch_size == 8
        assert cfg.max_seq_len == 512

    def test_d640_dimensions(self):
        cfg = s0_d640()
        assert cfg.d_model == 640
        assert cfg.n_heads == 10
        assert cfg.n_kv_heads == 2

    def test_d768_dimensions(self):
        cfg = s0_d768()
        assert cfg.d_model == 768
        assert cfg.n_layers == 22
        assert cfg.n_heads == 12
        assert cfg.n_kv_heads == 4

    def test_all_configs_have_valid_head_ratio(self):
        for name, fn in ALL_CONFIGS.items():
            cfg = fn()
            assert cfg.n_heads % cfg.n_kv_heads == 0, (
                f"{name}: n_heads ({cfg.n_heads}) not divisible by "
                f"n_kv_heads ({cfg.n_kv_heads})")

    def test_p4_and_p8_share_d_model(self):
        assert s0_p4().d_model == s0_p8().d_model


# ═══ inspect_checkpoint ═════════════════════════════════════════════════

from inspect_checkpoint import inspect_checkpoint


class TestInspectCheckpoint:
    def _save_ckpt(self, tmp_path, name, data):
        path = str(tmp_path / name)
        torch.save(data, path)
        return path

    def test_basic_checkpoint(self, tmp_path, capsys):
        ckpt = {
            "step": 500,
            "phase": "CONSENSUS",
            "model": {"layer.weight": torch.randn(16, 16)},
        }
        path = self._save_ckpt(tmp_path, "basic.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "Step: 500" in out
        assert "Phase: CONSENSUS" in out
        assert "Model state:" in out

    def test_nan_detection(self, tmp_path, capsys):
        w = torch.randn(4, 4)
        w[0, 0] = float("nan")
        ckpt = {"model": {"bad_layer": w}}
        path = self._save_ckpt(tmp_path, "nan.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "NAN" in out
        assert "bad_layer" in out

    def test_inf_detection(self, tmp_path, capsys):
        w = torch.randn(4, 4)
        w[1, 1] = float("inf")
        ckpt = {"model": {"inf_layer": w}}
        path = self._save_ckpt(tmp_path, "inf.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "INF" in out
        assert "inf_layer" in out

    def test_empty_tensor_detection(self, tmp_path, capsys):
        ckpt = {"model": {"empty_param": torch.empty(0)}}
        path = self._save_ckpt(tmp_path, "empty.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "EMPTY" in out

    def test_nonexistent_file(self, capsys):
        inspect_checkpoint("/nonexistent/ckpt.pt")
        out = capsys.readouterr().out
        assert "NOT FOUND" in out

    def test_optimizer_state(self, tmp_path, capsys):
        ckpt = {
            "step": 100,
            "model": {"w": torch.randn(4, 4)},
            "optimizer": {
                "param_groups": [
                    {"lr": 3e-5, "weight_decay": 0.01, "params": [0, 1]},
                    {"lr": 3e-4, "weight_decay": 0.0, "params": [2]},
                ],
            },
        }
        path = self._save_ckpt(tmp_path, "opt.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "2 param groups" in out
        assert "lr=3e-05" in out

    def test_config_dict(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "config": {"lr": 3e-5, "batch_size": 4},
        }
        path = self._save_ckpt(tmp_path, "cfg.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "config.lr" in out
        assert "config.batch_size" in out

    def test_model_cfg_dataclass(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "model_cfg": S0Config(),
        }
        path = self._save_ckpt(tmp_path, "mcfg.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "model_cfg.d_model" in out
        assert "model_cfg.patch_size" in out

    def test_ports_state(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "ports": {"proj.weight": torch.randn(16, 8)},
        }
        path = self._save_ckpt(tmp_path, "ports.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "Projection ports" in out

    def test_best_eval_bpb(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "best_eval_bpb": 5.1234,
        }
        path = self._save_ckpt(tmp_path, "best.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "Best eval BPB: 5.1234" in out

    def test_rng_state(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "rng_state": torch.get_rng_state(),
        }
        path = self._save_ckpt(tmp_path, "rng.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "rng_state present" in out

    def test_scaler_state(self, tmp_path, capsys):
        ckpt = {
            "step": 10,
            "model": {"w": torch.randn(2, 2)},
            "scaler": {"scale": 65536.0},
        }
        path = self._save_ckpt(tmp_path, "scaler.pt", ckpt)
        inspect_checkpoint(path)
        out = capsys.readouterr().out
        assert "GradScaler state present" in out


# ═══ s0_eval.bytes_to_text ══════════════════════════════════════════════

from s0_eval import bytes_to_text


class TestBytesToText:
    def test_ascii_roundtrip(self):
        text = "hello"
        t = torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)
        assert bytes_to_text(t) == "hello"

    def test_utf8_multibyte(self):
        text = "é"  # é
        raw = list(text.encode("utf-8"))
        t = torch.tensor(raw, dtype=torch.uint8)
        assert bytes_to_text(t) == "é"

    def test_invalid_bytes_replaced(self):
        t = torch.tensor([0xFF, 0xFE, 0x41], dtype=torch.uint8)
        result = bytes_to_text(t)
        assert "A" in result
        assert "�" in result

    def test_empty_tensor(self):
        t = torch.tensor([], dtype=torch.uint8)
        assert bytes_to_text(t) == ""

    def test_null_bytes(self):
        t = torch.tensor([0, 0, 65, 0], dtype=torch.uint8)
        result = bytes_to_text(t)
        assert "A" in result
        assert len(result) == 4
