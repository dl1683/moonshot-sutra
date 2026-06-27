"""Tests for export_log_csv.py — CSV export from JSONL logs, no GPU needed."""

import csv
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from export_log_csv import export_train_csv, export_eval_csv


# ── Helpers ──────────────────────────────────────────────────────────────

def _write_jsonl(entries):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for e in entries:
        if isinstance(e, str):
            f.write(e + "\n")
        else:
            f.write(json.dumps(e) + "\n")
    f.close()
    return f.name


def _read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def _tmp_csv():
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    f.close()
    return f.name


# ═══ export_train_csv ════════════════════════════════════════════════════

class TestExportTrainCSV:
    def test_basic_train_export(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "grad_norm": 0.5, "lr": 1e-4, "elapsed": 2.5, "phase": "s0"},
            {"step": 20, "bpb": 6.0, "grad_norm": 0.4, "lr": 2e-4, "elapsed": 5.1, "phase": "s0"},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, fields = _read_csv(out)
            assert len(rows) == 2
            assert float(rows[0]["ce_bpb"]) == pytest.approx(6.5)
            assert float(rows[1]["ce_bpb"]) == pytest.approx(6.0)
            assert "step" in fields
            assert "phase" in fields
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_ce_loss_fallback(self):
        import math
        log = _write_jsonl([
            {"step": 10, "ce_loss": 4.5, "phase": "e2"},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, _ = _read_csv(out)
            expected_bpb = 4.5 / math.log(2)
            assert float(rows[0]["ce_bpb"]) == pytest.approx(expected_bpb, rel=1e-4)
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_skips_eval_entries(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "s0"},
            {"step": 50, "eval_bpb": 6.2},
            {"step": 20, "bpb": 6.0, "phase": "s0"},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, _ = _read_csv(out)
            assert len(rows) == 2
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_skips_hard_fail_entries(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "s0"},
            {"step": 20, "HARD_FAIL": "non-finite CE loss", "phase": "s0"},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, _ = _read_csv(out)
            assert len(rows) == 1
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_skips_empty_lines(self):
        log = _write_jsonl([
            "",
            {"step": 10, "bpb": 6.5, "phase": "s0"},
            "",
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, _ = _read_csv(out)
            assert len(rows) == 1
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_teacher_losses_columns(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "e2",
             "teacher_losses_bits": {"t0_anchor": 5.2, "t1_hybrid": 4.8}},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, fields = _read_csv(out)
            assert "tl_t0_anchor" in fields
            assert "tl_t1_hybrid" in fields
            assert float(rows[0]["tl_t0_anchor"]) == pytest.approx(5.2)
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_route_stats_columns(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "e2",
             "route_stats": {
                 "mean_jsd": 0.15,
                 "mean_route_entropy": 0.85,
                 "n_routed": 12,
                 "avg_teacher_weights": {"t0": 0.6, "t1": 0.4},
             }},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, fields = _read_csv(out)
            assert "jsd" in fields
            assert "route_entropy" in fields
            assert "n_routed" in fields
            assert "tw_t0" in fields
            assert float(rows[0]["jsd"]) == pytest.approx(0.15)
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_grad_budget_columns(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "e2",
             "grad_budget": {"ce_grad_norm": 1.2, "total_scale": 0.8}},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, fields = _read_csv(out)
            assert "gb_ce_norm" in fields
            assert "gb_total_scale" in fields
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_no_train_entries_exits(self):
        log = _write_jsonl([
            {"step": 50, "eval_bpb": 6.2},
        ])
        out = _tmp_csv()
        try:
            with pytest.raises(SystemExit):
                export_train_csv(log, out)
        finally:
            os.unlink(log)
            if os.path.exists(out):
                os.unlink(out)

    def test_field_ordering_stable(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5, "phase": "e2",
             "teacher_losses_bits": {"t0": 5.0}},
            {"step": 20, "bpb": 6.0, "phase": "e2",
             "teacher_losses_bits": {"t0": 4.8, "t1": 4.5}},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, fields = _read_csv(out)
            assert fields.index("step") < fields.index("ce_bpb")
            assert "tl_t0" in fields
            assert "tl_t1" in fields
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_missing_optional_fields_empty(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5},
        ])
        out = _tmp_csv()
        try:
            export_train_csv(log, out)
            rows, _ = _read_csv(out)
            assert rows[0]["lr"] == ""
            assert rows[0]["elapsed_s"] == ""
        finally:
            os.unlink(log)
            os.unlink(out)


# ═══ export_eval_csv ═════════════════════════════════════════════════════

class TestExportEvalCSV:
    def test_basic_eval_export(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5},
            {"step": 50, "eval_bpb": 6.2, "phase": "s0"},
            {"step": 100, "eval_bpb": 5.5, "phase": "s0", "eval_byte_acc": 0.03},
        ])
        out = _tmp_csv()
        try:
            export_eval_csv(log, out)
            rows, fields = _read_csv(out)
            assert len(rows) == 2
            assert float(rows[0]["eval_bpb"]) == pytest.approx(6.2)
            assert "eval_byte_acc" in fields
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_no_eval_entries_exits(self):
        log = _write_jsonl([
            {"step": 10, "bpb": 6.5},
        ])
        out = _tmp_csv()
        try:
            with pytest.raises(SystemExit):
                export_eval_csv(log, out)
        finally:
            os.unlink(log)
            if os.path.exists(out):
                os.unlink(out)

    def test_eval_loss_fallback(self):
        log = _write_jsonl([
            {"step": 50, "eval_loss": 4.5},
        ])
        out = _tmp_csv()
        try:
            export_eval_csv(log, out)
            rows, _ = _read_csv(out)
            assert len(rows) == 1
            import math
            expected_bpb = round(4.5 / math.log(2), 6)
            assert float(rows[0]["eval_bpb"]) == expected_bpb
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_skips_empty_lines(self):
        log = _write_jsonl([
            "",
            {"step": 50, "eval_bpb": 6.2},
            "",
        ])
        out = _tmp_csv()
        try:
            export_eval_csv(log, out)
            rows, _ = _read_csv(out)
            assert len(rows) == 1
        finally:
            os.unlink(log)
            os.unlink(out)

    def test_dynamic_columns(self):
        log = _write_jsonl([
            {"step": 50, "eval_bpb": 6.2},
            {"step": 100, "eval_bpb": 5.5, "eval_byte_acc": 0.03},
        ])
        out = _tmp_csv()
        try:
            export_eval_csv(log, out)
            rows, fields = _read_csv(out)
            assert "eval_byte_acc" in fields
            assert rows[0]["eval_byte_acc"] == ""
            assert float(rows[1]["eval_byte_acc"]) == pytest.approx(0.03)
        finally:
            os.unlink(log)
            os.unlink(out)
