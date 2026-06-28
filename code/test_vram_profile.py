"""Tests for vram_profile.py — VRAM estimation logic, runs on CPU."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from s0_architecture import S0Config
from vram_profile import estimate_vram, estimate_e2_overhead


class TestEstimateVram:
    def test_default_config_fits_24gb(self):
        est = estimate_vram(S0Config(), batch_size=4)
        assert est["fits_24gb"]
        assert est["total_mem_gb"] < 24.0

    def test_returns_expected_keys(self):
        est = estimate_vram(S0Config())
        expected = {
            "total_params", "trainable_params", "dead_params",
            "param_mem_mb", "grad_mem_mb", "optimizer_mem_mb",
            "activation_mem_mb", "misc_mem_mb",
            "total_mem_mb", "total_mem_gb", "fits_24gb",
        }
        assert expected.issubset(set(est.keys()))

    def test_trainable_plus_dead_equals_total(self):
        est = estimate_vram(S0Config())
        assert est["trainable_params"] + est["dead_params"] == est["total_params"]

    def test_total_is_sum_of_parts(self):
        est = estimate_vram(S0Config())
        parts_mb = (est["param_mem_mb"] + est["grad_mem_mb"] +
                    est["optimizer_mem_mb"] + est["activation_mem_mb"] +
                    est["misc_mem_mb"])
        assert est["total_mem_mb"] == pytest.approx(parts_mb, rel=1e-4)

    def test_total_gb_consistent_with_mb(self):
        est = estimate_vram(S0Config())
        assert est["total_mem_gb"] == pytest.approx(est["total_mem_mb"] / 1e3, rel=1e-4)

    def test_larger_batch_uses_more_memory(self):
        est_small = estimate_vram(S0Config(), batch_size=2)
        est_large = estimate_vram(S0Config(), batch_size=8)
        assert est_large["activation_mem_mb"] > est_small["activation_mem_mb"]
        assert est_large["total_mem_mb"] > est_small["total_mem_mb"]

    def test_param_mem_unchanged_by_batch(self):
        est_b2 = estimate_vram(S0Config(), batch_size=2)
        est_b8 = estimate_vram(S0Config(), batch_size=8)
        assert est_b2["param_mem_mb"] == est_b8["param_mem_mb"]
        assert est_b2["optimizer_mem_mb"] == est_b8["optimizer_mem_mb"]

    def test_higher_checkpoint_every_reduces_stored_activations(self):
        est_frequent = estimate_vram(S0Config(), checkpoint_every=2)
        est_rare = estimate_vram(S0Config(), checkpoint_every=30)
        assert est_rare["activation_mem_mb"] < est_frequent["activation_mem_mb"]

    def test_larger_model_uses_more_memory(self):
        est_small = estimate_vram(S0Config())
        est_large = estimate_vram(S0Config(d_model=768, n_layers=22,
                                           n_heads=12, n_kv_heads=4))
        assert est_large["total_params"] > est_small["total_params"]
        assert est_large["total_mem_mb"] > est_small["total_mem_mb"]

    def test_p8_config(self):
        est = estimate_vram(S0Config(patch_size=8, max_seq_len=512))
        assert est["total_params"] > 0
        assert est["fits_24gb"]


class TestEstimateE2Overhead:
    def test_returns_expected_keys(self):
        e2 = estimate_e2_overhead()
        expected = {
            "port_params", "port_details", "port_mem_mb",
            "port_grad_mb", "port_optimizer_mb", "router_mem_mb",
            "total_overhead_mb", "total_overhead_gb",
        }
        assert expected.issubset(set(e2.keys()))

    def test_overhead_is_small(self):
        e2 = estimate_e2_overhead()
        assert e2["total_overhead_gb"] < 1.0

    def test_port_details_nonempty(self):
        e2 = estimate_e2_overhead()
        assert len(e2["port_details"]) > 0

    def test_total_is_sum_of_parts(self):
        e2 = estimate_e2_overhead()
        parts = (e2["port_mem_mb"] + e2["port_grad_mb"] +
                 e2["port_optimizer_mb"] + e2["router_mem_mb"] +
                 e2["emb_tables_mb"])
        assert e2["total_overhead_mb"] == pytest.approx(parts, rel=1e-4)

    def test_gb_consistent_with_mb(self):
        e2 = estimate_e2_overhead()
        assert e2["total_overhead_gb"] == pytest.approx(
            e2["total_overhead_mb"] / 1e3, rel=1e-4)

    def test_s0_plus_e2_fits_24gb(self):
        s0 = estimate_vram(S0Config(), batch_size=4)
        e2 = estimate_e2_overhead()
        combined = s0["total_mem_gb"] + e2["total_overhead_gb"]
        assert combined < 24.0

    def test_port_details_have_required_fields(self):
        e2 = estimate_e2_overhead()
        for name, detail in e2["port_details"].items():
            assert "params" in detail
            assert "mem_mb" in detail
            assert "hidden_dim" in detail
            assert detail["params"] > 0

    def test_custom_student_dim(self):
        e2_small = estimate_e2_overhead(student_dim=256)
        e2_large = estimate_e2_overhead(student_dim=1024)
        assert e2_large["port_params"] > e2_small["port_params"]
