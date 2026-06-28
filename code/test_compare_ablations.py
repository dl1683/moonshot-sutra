"""Tests for compare_ablations.py — ablation analysis logic, no GPU needed."""

import json
import math
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from compare_ablations import (
    ce_to_bpb, load_log, load_eval_results, analyze_run, export_csv,
    RunSummary, evaluate_decision_rules, DECISION_RULES, GOLDFREE_RULES,
)


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


def _train(step, bpb=None, ce_loss=None, phase="s0", grad_norm=0.5, **kw):
    entry = {"step": step, "phase": phase, "grad_norm": grad_norm}
    if bpb is not None:
        entry["bpb"] = bpb
    if ce_loss is not None:
        entry["ce_loss"] = ce_loss
    entry.update(kw)
    return entry


def _eval(step, eval_bpb=4.0, **kw):
    entry = {"step": step, "eval_bpb": eval_bpb}
    entry.update(kw)
    return entry


# ═══ ce_to_bpb ══════════════════════════════════════════════════════════

class TestCeToBpb:
    def test_basic_conversion(self):
        assert ce_to_bpb(math.log(2)) == pytest.approx(1.0)

    def test_zero_returns_zero(self):
        assert ce_to_bpb(0.0) == 0.0

    def test_negative_returns_zero(self):
        assert ce_to_bpb(-1.0) == 0.0

    def test_large_value(self):
        assert ce_to_bpb(5.0) == pytest.approx(5.0 / math.log(2))


# ═══ load_log ════════════════════════════════════════════════════════════

class TestLoadLog:
    def test_splits_train_eval(self):
        path = _write_jsonl([
            _train(10, bpb=6.5),
            _eval(50, eval_bpb=6.0),
            _train(20, bpb=6.0),
        ])
        try:
            train, eval_, had_hard_fail = load_log(path)
            assert len(train) == 2
            assert len(eval_) == 1
            assert not had_hard_fail
        finally:
            os.unlink(path)

    def test_skips_hard_fail(self):
        path = _write_jsonl([
            _train(10, bpb=6.5),
            {"step": 20, "HARD_FAIL": "NaN", "phase": "s0"},
            _train(30, bpb=5.5),
        ])
        try:
            train, eval_, had_hard_fail = load_log(path)
            assert len(train) == 2
            assert had_hard_fail
        finally:
            os.unlink(path)

    def test_skips_empty_lines(self):
        path = _write_jsonl(["", _train(10, bpb=6.5), ""])
        try:
            train, _, _ = load_log(path)
            assert len(train) == 1
        finally:
            os.unlink(path)

    def test_ce_loss_entries_parsed(self):
        path = _write_jsonl([_train(10, ce_loss=4.5)])
        try:
            train, _, _ = load_log(path)
            assert len(train) == 1
            assert "ce_loss" in train[0]
        finally:
            os.unlink(path)

    def test_eval_loss_entries_parsed(self):
        path = _write_jsonl([{"step": 50, "eval_loss": 3.0, "eval_bpb": 4.33}])
        try:
            _, eval_, _ = load_log(path)
            assert len(eval_) == 1
        finally:
            os.unlink(path)


# ═══ analyze_run ═════════════════════════════════════════════════════════

class TestAnalyzeRun:
    def test_empty_log(self):
        path = _write_jsonl([])
        try:
            s = analyze_run("A0", path)
            assert s.ablation_id == "A0"
            assert s.total_steps == 0
        finally:
            os.unlink(path)

    def test_basic_bpb_tracking(self):
        path = _write_jsonl([
            _train(0, bpb=7.5),
            _train(100, bpb=6.0),
            _train(500, bpb=4.5, elapsed=300.0),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.total_steps == 500
            assert s.initial_ce_bpb == pytest.approx(7.5)
            assert s.final_ce_bpb == pytest.approx(4.5)
            assert s.elapsed_seconds == pytest.approx(300.0)
        finally:
            os.unlink(path)

    def test_ce_loss_conversion(self):
        ce = 4.0
        path = _write_jsonl([
            _train(0, ce_loss=6.0),
            _train(100, ce_loss=ce),
        ])
        try:
            s = analyze_run("A1", path)
            assert s.final_ce_bpb == pytest.approx(ce / math.log(2))
        finally:
            os.unlink(path)

    def test_eval_tracking(self):
        path = _write_jsonl([
            _train(0, bpb=7.0),
            _eval(100, eval_bpb=6.5),
            _train(200, bpb=5.0),
            _eval(200, eval_bpb=5.0),
            _train(500, bpb=4.0),
            _eval(500, eval_bpb=4.5),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.best_eval_bpb == pytest.approx(4.5)
            assert s.final_eval_bpb == pytest.approx(4.5)
        finally:
            os.unlink(path)

    def test_best_eval_is_minimum(self):
        path = _write_jsonl([
            _train(0, bpb=7.0),
            _eval(100, eval_bpb=6.5),
            _eval(200, eval_bpb=4.2),
            _eval(300, eval_bpb=4.8),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.best_eval_bpb == pytest.approx(4.2)
            assert s.final_eval_bpb == pytest.approx(4.8)
        finally:
            os.unlink(path)

    def test_phase_tracking(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, phase="E2.1_port_warmup"),
            _train(100, bpb=6.5, phase="E2.1_port_warmup"),
            _train(200, bpb=6.0, phase="E2.2_consensus"),
            _train(500, bpb=5.0, phase="E2.2_consensus"),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.phases_seen == ["E2.1_port_warmup", "E2.2_consensus"]
            assert "E2.1_port_warmup" in s.phase_metrics
            assert "E2.2_consensus" in s.phase_metrics
        finally:
            os.unlink(path)

    def test_phase_bpb_metrics(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, phase="E2.1_port_warmup"),
            _train(100, bpb=6.0, phase="E2.1_port_warmup"),
            _train(200, bpb=5.5, phase="E2.1_port_warmup"),
        ])
        try:
            s = analyze_run("A2", path)
            pm = s.phase_metrics["E2.1_port_warmup"]
            assert pm["entry_bpb"] == 7.0
            assert pm["exit_bpb"] == 5.5
            assert pm["delta_bpb"] == pytest.approx(-1.5)
            assert pm["min_bpb"] == 5.5
            assert pm["n_steps"] == 3
            assert pm["start_step"] == 0
            assert pm["end_step"] == 200
        finally:
            os.unlink(path)

    def test_teacher_loss_means(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, teacher_losses_bits={"t0": 5.0, "t1": 6.0}),
            _train(100, bpb=6.0, teacher_losses_bits={"t0": 4.0, "t1": 5.0}),
        ])
        try:
            s = analyze_run("A2", path)
            pm = s.phase_metrics["s0"]
            assert pm["teacher_loss_means"]["t0"] == pytest.approx(4.5)
            assert pm["teacher_loss_means"]["t1"] == pytest.approx(5.5)
        finally:
            os.unlink(path)

    def test_teacher_loss_final(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, teacher_losses_bits={"t0": 5.0}),
            _train(100, bpb=6.0, teacher_losses_bits={"t0": 3.5, "t1": 4.0}),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.teacher_loss_final["t0"] == pytest.approx(3.5)
            assert s.teacher_loss_final["t1"] == pytest.approx(4.0)
        finally:
            os.unlink(path)

    def test_route_stats_aggregation(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, route_stats={
                "mean_jsd": 0.10, "mean_route_entropy": 0.80, "n_routed": 10,
                "avg_teacher_weights": {"t0": 0.7, "t1": 0.3},
            }),
            _train(100, bpb=6.0, route_stats={
                "mean_jsd": 0.20, "mean_route_entropy": 0.90, "n_routed": 14,
                "avg_teacher_weights": {"t0": 0.6, "t1": 0.4},
            }),
        ])
        try:
            s = analyze_run("A2", path)
            rs = s.route_stats
            assert rs["mean_jsd"] == pytest.approx(0.15)
            assert rs["mean_entropy"] == pytest.approx(0.85)
            assert rs["mean_n_routed"] == pytest.approx(12.0)
            assert rs["n_entries"] == 2
            assert rs["final_teacher_weights"]["t0"] == pytest.approx(0.6)
        finally:
            os.unlink(path)

    def test_grad_budget_stats(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, grad_budget={
                "ce_grad_norm": 1.0, "total_scale": 0.8}),
            _train(100, bpb=6.0, grad_budget={
                "ce_grad_norm": 2.0, "total_scale": 0.4}),
        ])
        try:
            s = analyze_run("A2", path)
            gb = s.grad_budget_stats
            assert gb["mean_ce_grad_norm"] == pytest.approx(1.5)
            assert gb["mean_total_scale"] == pytest.approx(0.6)
            assert gb["min_total_scale"] == pytest.approx(0.4)
            assert gb["max_total_scale"] == pytest.approx(0.8)
        finally:
            os.unlink(path)

    def test_no_route_stats_when_absent(self):
        path = _write_jsonl([_train(0, bpb=7.0), _train(100, bpb=6.0)])
        try:
            s = analyze_run("A0", path)
            assert s.route_stats == {}
        finally:
            os.unlink(path)


# ═══ export_csv ══════════════════════════════════════════════════════════

class TestExportCSV:
    def test_basic_export(self):
        s = RunSummary(
            ablation_id="A2", log_path="fake.jsonl",
            total_steps=500, initial_ce_bpb=7.0, final_ce_bpb=4.5,
            best_eval_bpb=4.3, final_eval_bpb=4.5,
            elapsed_seconds=3600.0,
        )
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                lines = f.readlines()
            assert len(lines) == 2
            header = lines[0].strip().split(",")
            assert "ablation_id" in header
            assert "total_steps" in header
            vals = lines[1].strip().split(",")
            assert vals[0] == "A2"
        finally:
            os.unlink(out.name)

    def test_multiple_ablations(self):
        summaries = [
            RunSummary(ablation_id="A2", log_path="a.jsonl",
                       total_steps=500, initial_ce_bpb=7.0, final_ce_bpb=4.5),
            RunSummary(ablation_id="A0", log_path="b.jsonl",
                       total_steps=500, initial_ce_bpb=7.0, final_ce_bpb=5.0),
        ]
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv(summaries, out.name)
            with open(out.name) as f:
                lines = f.readlines()
            assert len(lines) == 3
        finally:
            os.unlink(out.name)

    def test_inf_best_eval_exports_empty(self):
        s = RunSummary(ablation_id="A0", log_path="x.jsonl",
                       best_eval_bpb=float("inf"))
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                lines = f.readlines()
            vals = lines[1].strip().split(",")
            idx = lines[0].strip().split(",").index("best_eval_bpb")
            assert vals[idx] == ""
        finally:
            os.unlink(out.name)

    def test_eval_result_metrics_exported(self):
        s = RunSummary(ablation_id="A2", log_path="x.jsonl")
        s.eval_result = {"metrics": {"bpb": 4.2, "first_byte_acc": 0.05}}
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                content = f.read()
            assert "4.2" in content
        finally:
            os.unlink(out.name)


# ═══ evaluate_decision_rules ═════════════════════════════════════════════

class TestDecisionRules:
    def _make_summary(self, aid, eval_bpb):
        s = RunSummary(ablation_id=aid, log_path="x.jsonl")
        s.eval_result = {"metrics": {"bpb": eval_bpb}}
        return s

    def test_a2_beats_a0_passes(self, capsys):
        summaries = [
            self._make_summary("A2", 4.0),
            self._make_summary("A0", 4.1),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "PASS" in out
        assert "E2 beats CE-only" in out

    def test_a2_does_not_beat_a0_fails(self, capsys):
        summaries = [
            self._make_summary("A2", 4.0),
            self._make_summary("A0", 4.01),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "FAIL" in out
        assert "abandon" in out.lower()

    def test_no_eval_results_returns_silently(self, capsys):
        summaries = [
            RunSummary(ablation_id="A2", log_path="x.jsonl"),
            RunSummary(ablation_id="A0", log_path="y.jsonl"),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert out.strip() == ""

    def test_partial_coverage_skips_missing(self, capsys):
        summaries = [
            self._make_summary("A2", 4.0),
            self._make_summary("A0", 4.1),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "A1" not in out

    def test_decision_rules_table_not_empty(self):
        assert len(DECISION_RULES) > 0
        for rule in DECISION_RULES:
            assert len(rule) == 5
            better, worse, margin, fail_msg, pass_msg = rule
            assert isinstance(better, str)
            assert isinstance(worse, str)
            assert isinstance(margin, float)
            assert margin > 0


# ═══ Integration ═════════════════════════════════════════════════════════

class TestAnalyzeRunIntegration:
    def test_realistic_e2_log(self):
        entries = []
        for i in range(10):
            entries.append(_train(i * 75, bpb=7.0 - i * 0.2,
                                  phase="E2.1_port_warmup"))
        for i in range(20):
            step = 750 + i * 100
            entries.append(_train(step, bpb=5.0 - i * 0.05,
                                  phase="E2.2_consensus",
                                  teacher_losses_bits={"t0": 4.5 - i * 0.03},
                                  route_stats={
                                      "mean_jsd": 0.1 + i * 0.005,
                                      "mean_route_entropy": 0.7 + i * 0.01,
                                      "n_routed": 10 + i,
                                      "avg_teacher_weights": {"t0": 0.8 - i * 0.01},
                                  },
                                  grad_budget={
                                      "ce_grad_norm": 1.0 + i * 0.05,
                                      "total_scale": 0.5 + i * 0.02,
                                  }))
        entries.append(_eval(500, eval_bpb=5.5))
        entries.append(_eval(2000, eval_bpb=4.0))
        entries.append(_eval(2750, eval_bpb=3.8))

        path = _write_jsonl(entries)
        try:
            s = analyze_run("A2", path)
            assert s.total_steps == 2650
            assert len(s.phases_seen) == 2
            assert s.best_eval_bpb == pytest.approx(3.8)
            assert s.route_stats["n_entries"] == 20
            assert s.grad_budget_stats["n_entries"] == 20
            assert "t0" in s.teacher_loss_final
        finally:
            os.unlink(path)


# ═══ Gold-free routing rules ════════════════════════════════════════════

class TestGoldFreeRules:
    def _make_summary(self, aid, eval_bpb):
        s = RunSummary(ablation_id=aid, log_path="x.jsonl")
        s.eval_result = {"metrics": {"bpb": eval_bpb}}
        return s

    def test_goldfree_rules_table_not_empty(self):
        assert len(GOLDFREE_RULES) > 0
        for rule in GOLDFREE_RULES:
            assert len(rule) == 6

    def test_goldfree_pass_triple_comparison(self, capsys):
        summaries = [
            self._make_summary("A9c", 4.0),
            self._make_summary("A2", 4.005),
            self._make_summary("A5b", 4.05),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "Gold-free router works" in out

    def test_goldfree_fail_too_far_from_oracle(self, capsys):
        summaries = [
            self._make_summary("A9c", 4.5),
            self._make_summary("A2", 4.0),
            self._make_summary("A5b", 4.6),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "Oracle routing is material" in out

    def test_goldfree_unproven_matches_static(self, capsys):
        summaries = [
            self._make_summary("A9c", 4.1),
            self._make_summary("A2", 3.95),
            self._make_summary("A5b", 4.0),
        ]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "unproven" in out.lower()

    def test_goldfree_missing_ablation_skips(self, capsys):
        summaries = [self._make_summary("A9c", 4.0)]
        evaluate_decision_rules(summaries)
        out = capsys.readouterr().out
        assert "Gold-free" not in out


# ═══ load_eval_results ══════════════════════════════════════════════════

class TestLoadEvalResults:
    def test_loads_valid_file(self):
        data = {"ablation_id": "A2", "metrics": {"bpb": 4.0}}
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        try:
            results = load_eval_results([f.name])
            assert "A2" in results
            assert results["A2"]["metrics"]["bpb"] == 4.0
        finally:
            os.unlink(f.name)

    def test_missing_file_skipped(self):
        results = load_eval_results(["nonexistent_file_12345.json"])
        assert results == {}

    def test_uses_stem_when_no_ablation_id(self):
        data = {"metrics": {"bpb": 4.0}}
        f = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, prefix="a3_results_")
        json.dump(data, f)
        f.close()
        try:
            results = load_eval_results([f.name])
            stem = os.path.splitext(os.path.basename(f.name))[0]
            assert stem in results
        finally:
            os.unlink(f.name)

    def test_multiple_files(self):
        files = []
        for aid in ("A0", "A1"):
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
            json.dump({"ablation_id": aid, "metrics": {"bpb": 4.0 + len(files)}}, f)
            f.close()
            files.append(f.name)
        try:
            results = load_eval_results(files)
            assert "A0" in results
            assert "A1" in results
        finally:
            for p in files:
                os.unlink(p)


# ═══ Alternate key paths ════════════════════════════════════════════════

class TestAlternateKeyPaths:
    def test_teacher_losses_key_not_bits(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, teacher_losses={"t0": 5.0, "t1": 6.0}),
            _train(100, bpb=6.0, teacher_losses={"t0": 3.0, "t1": 4.0}),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.teacher_loss_final["t0"] == pytest.approx(3.0)
            pm = s.phase_metrics["s0"]
            assert pm["teacher_loss_means"]["t0"] == pytest.approx(4.0)
        finally:
            os.unlink(path)

    def test_route_stats_without_teacher_weights(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, route_stats={
                "mean_jsd": 0.1, "mean_route_entropy": 0.8, "n_routed": 10,
            }),
        ])
        try:
            s = analyze_run("A2", path)
            assert "final_teacher_weights" not in s.route_stats
            assert s.route_stats["mean_jsd"] == pytest.approx(0.1)
        finally:
            os.unlink(path)

    def test_grad_budget_without_ce_grad_norm_skipped(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, grad_budget={"some_other_key": 1.0}),
        ])
        try:
            s = analyze_run("A2", path)
            assert s.grad_budget_stats == {}
        finally:
            os.unlink(path)

    def test_eval_entry_with_eval_loss_only(self):
        path = _write_jsonl([
            _train(0, bpb=7.0),
            {"step": 100, "eval_loss": 3.5},
        ])
        try:
            train, eval_, _ = load_log(path)
            assert len(eval_) == 1
            assert eval_[0]["eval_loss"] == 3.5
        finally:
            os.unlink(path)


# ═══ CSV edge cases ═════════════════════════════════════════════════════

class TestExportCSVEdgeCases:
    def test_route_stats_missing_key_uses_get_default(self):
        s = RunSummary(ablation_id="A0", log_path="x.jsonl")
        s.route_stats = {"mean_jsd": 0.1}
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                lines = f.readlines()
            header = lines[0].strip().split(",")
            vals = lines[1].strip().split(",")
            jsd_idx = header.index("mean_jsd")
            assert vals[jsd_idx] == "0.1"
        finally:
            os.unlink(out.name)

    def test_no_eval_result_exports_empty_fields(self):
        s = RunSummary(ablation_id="A0", log_path="x.jsonl",
                       total_steps=100, initial_ce_bpb=7.0, final_ce_bpb=6.0)
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                lines = f.readlines()
            header = lines[0].strip().split(",")
            vals = lines[1].strip().split(",")
            bpb_idx = header.index("eval_bpb")
            assert vals[bpb_idx] == ""
        finally:
            os.unlink(out.name)

    def test_no_elapsed_exports_empty(self):
        s = RunSummary(ablation_id="A0", log_path="x.jsonl",
                       elapsed_seconds=0.0)
        out = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        out.close()
        try:
            export_csv([s], out.name)
            with open(out.name) as f:
                lines = f.readlines()
            header = lines[0].strip().split(",")
            vals = lines[1].strip().split(",")
            idx = header.index("elapsed_hours")
            assert vals[idx] == ""
        finally:
            os.unlink(out.name)


class TestGCGCoherenceExtraction:
    def test_pairwise_coherence_extracted(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, grad_budget={
                "ce_grad_norm": 1.0, "total_scale": 0.5,
                "pairwise_coherence": 0.25,
                "ce_teacher_cosines": {"t0": 0.8, "t1": -0.1}}),
            _train(10, bpb=6.5, grad_budget={
                "ce_grad_norm": 1.5, "total_scale": 0.4,
                "pairwise_coherence": 0.35,
                "ce_teacher_cosines": {"t0": 0.7, "t1": 0.0}}),
        ])
        try:
            s = analyze_run("A2", path)
            gb = s.grad_budget_stats
            assert gb["mean_pairwise_coherence"] == pytest.approx(0.3)
            assert "mean_ce_teacher_cosines" in gb
            assert gb["mean_ce_teacher_cosines"]["t0"] == pytest.approx(0.75)
            assert gb["mean_ce_teacher_cosines"]["t1"] == pytest.approx(-0.05)
        finally:
            os.unlink(path)

    def test_no_coherence_when_absent(self):
        path = _write_jsonl([
            _train(0, bpb=7.0, grad_budget={
                "ce_grad_norm": 1.0, "total_scale": 0.5}),
        ])
        try:
            s = analyze_run("A2", path)
            gb = s.grad_budget_stats
            assert "mean_pairwise_coherence" not in gb
            assert "mean_ce_teacher_cosines" not in gb
        finally:
            os.unlink(path)
