"""Tests for check_opsec scanner. No GPU needed."""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from check_opsec import scan_file, COMPILED


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
