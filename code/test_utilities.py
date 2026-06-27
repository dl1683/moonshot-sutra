"""Tests for utility scripts — preflight checks, opsec scan. No GPU needed."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from preflight import check_onedrive_path, check_checkpoints
from check_opsec import scan_file, COMPILED


# ═══ check_onedrive_path ═════════════════════════════════════════════════

class TestCheckOneDrivePath:
    def test_onedrive_path_fails(self):
        ok, msg = check_onedrive_path("C:/Users/dev/OneDrive/Desktop/checkpoints")
        assert not ok
        assert "OneDrive" in msg

    def test_onedrive_lowercase_fails(self):
        ok, msg = check_onedrive_path("C:/Users/dev/onedrive/stuff")
        assert not ok

    def test_non_onedrive_path_passes(self):
        ok, msg = check_onedrive_path("C:/sutra_fast/checkpoints/e2")
        assert ok

    def test_local_path_passes(self):
        ok, msg = check_onedrive_path("D:/training/checkpoints")
        assert ok

    def test_relative_path_without_onedrive_passes(self):
        ok, msg = check_onedrive_path("checkpoints/e2")
        assert ok


# ═══ check_checkpoints ══════════════════════════════════════════════════

class TestCheckCheckpoints:
    def test_writable_dir(self):
        with tempfile.TemporaryDirectory() as d:
            ok, msg = check_checkpoints(d)
            assert ok
            assert "writable" in msg

    def test_creates_missing_dir(self):
        with tempfile.TemporaryDirectory() as parent:
            target = os.path.join(parent, "subdir", "ckpts")
            ok, msg = check_checkpoints(target)
            assert ok
            assert os.path.isdir(target)

    def test_nonexistent_drive_fails(self):
        ok, msg = check_checkpoints("Z:/impossible/path/that/does/not/exist")
        assert not ok


# ═══ scan_file (check_opsec) ════════════════════════════════════════════

class TestScanFile:
    def _write_temp(self, content):
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                        encoding="utf-8")
        f.write(content)
        f.close()
        return f.name

    def test_clean_file_no_violations(self):
        path = self._write_temp("model = load_model('t0_anchor_decoder')\n")
        try:
            assert scan_file(path) == []
        finally:
            os.unlink(path)

    def test_detects_qwen3_model_name(self):
        path = self._write_temp("teacher = 'Qwen3-1.7B'\n")
        try:
            violations = scan_file(path)
            assert len(violations) == 1
            assert violations[0][0] == 1
        finally:
            os.unlink(path)

    def test_detects_lfm2_model_name(self):
        path = self._write_temp("model = 'LFM2.5-1.2B'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_detects_embeddinggemma(self):
        path = self._write_temp("embed = 'EmbeddingGemma'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_detects_mamba2_780m(self):
        path = self._write_temp("ssm = 'Mamba2-780M'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_detects_qwen_hub_path(self):
        path = self._write_temp("path = 'Qwen/Qwen3-0.6B'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_detects_state_spaces_mamba(self):
        path = self._write_temp("repo = 'state-spaces/mamba'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_detects_liquid_ai_hub(self):
        path = self._write_temp("hub = 'Liquid AI/lfm'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_multiple_violations(self):
        content = "a = 'Qwen3-1.7B'\nb = 'Mamba2-780M'\nc = clean\n"
        path = self._write_temp(content)
        try:
            violations = scan_file(path)
            assert len(violations) == 2
            assert violations[0][0] == 1
            assert violations[1][0] == 2
        finally:
            os.unlink(path)

    def test_role_aliases_clean(self):
        content = "\n".join([
            "t0 = 't0_anchor_decoder'",
            "t1 = 't1_diversity_hybrid'",
            "t2 = 't2_control_decoder'",
            "t3 = 't3_semantic_embedding'",
            "t4 = 't4_diversity_ssm'",
        ]) + "\n"
        path = self._write_temp(content)
        try:
            assert scan_file(path) == []
        finally:
            os.unlink(path)

    def test_nonexistent_file_no_crash(self):
        violations = scan_file("/nonexistent/file.py")
        assert violations == []

    def test_internal_aliases_clean(self):
        content = "model_id = 'qwen3_0p6b'\n"
        path = self._write_temp(content)
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)

    def test_google_embedding_path(self):
        path = self._write_temp("path = 'google/embedding-gecko'\n")
        try:
            violations = scan_file(path)
            assert len(violations) >= 1
        finally:
            os.unlink(path)
