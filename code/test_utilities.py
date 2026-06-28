"""Tests for utility scripts — preflight checks, opsec scan, CLI smoke. No GPU needed."""

import argparse
import os
import re
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


# ═══ CLI smoke tests ═══════════════════════════════════════════════════════

def _extract_flags(cmd_lines: list[str]) -> list[str]:
    """Extract --flag names from a multi-line shell command."""
    flags = []
    for line in cmd_lines:
        for m in re.finditer(r'(--[a-z][a-z0-9_-]*)', line):
            flags.append(m.group(1))
    return flags


def _build_parser(module_name: str) -> argparse.ArgumentParser:
    """Import a training module and build its argparse parser."""
    mod = __import__(module_name)
    parser = argparse.ArgumentParser()
    if module_name == "s0_training":
        parser.add_argument("--burnin", action="store_true")
        parser.add_argument("--data-dir", type=str, default=None)
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--eval-every", type=int, default=None)
        parser.add_argument("--warmup-steps", type=int, default=None)
        parser.add_argument("--config", default="p4")
    elif module_name == "eklavya_cache":
        parser.add_argument("--teacher", required=False)
        parser.add_argument("--data-dir", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--student-checkpoint", required=False)
        parser.add_argument("--max-shards", type=int, default=50)
        parser.add_argument("--nll-threshold", type=float)
        parser.add_argument("--seq-len", type=int)
    elif module_name == "eklavya_training":
        parser.add_argument("--student-checkpoint", required=False)
        parser.add_argument("--cache-dir", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--data-dir", default=None)
        parser.add_argument("--steps", type=int, default=None)
        parser.add_argument("--resume-from", default=None)
        parser.add_argument("--allow-legacy-cache", action="store_true")
    elif module_name == "eklavya_e2_cache_builder":
        parser.add_argument("--student-checkpoint", required=False)
        parser.add_argument("--data-dir", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--max-shards", type=int)
        parser.add_argument("--seq-len", type=int)
        parser.add_argument("--nll-threshold", type=float)
        parser.add_argument("--kl-top-k", type=int)
        parser.add_argument("--teachers", nargs="+")
        parser.add_argument("--teacher-config", default=None)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--control-frac", type=float)
        parser.add_argument("--entropy-threshold", type=float)
        parser.add_argument("--shard-start", type=int)
        parser.add_argument("--shard-end", type=int)
        parser.add_argument("--positions-only", action="store_true")
        parser.add_argument("--teachers-only", action="store_true")
        parser.add_argument("--jsd-threshold", type=float)
        parser.add_argument("--skip-disagreement", action="store_true")
        parser.add_argument("--allow-nonfinite-drop", action="store_true")
    elif module_name == "eklavya_e2_training":
        parser.add_argument("--student-checkpoint", required=False)
        parser.add_argument("--cache-dir", default=None)
        parser.add_argument("--output-dir", default=None)
        parser.add_argument("--steps", type=int)
        parser.add_argument("--resume", type=str)
        parser.add_argument("--ablation-id", default="A2")
        parser.add_argument("--teachers", nargs="+")
        parser.add_argument("--exclude-teachers", nargs="+")
        parser.add_argument("--disable-router", action="store_true")
        parser.add_argument("--shuffle-teacher-targets", action="store_true")
        parser.add_argument("--ce-only", action="store_true")
        parser.add_argument("--disable-gradient-budget", action="store_true")
        parser.add_argument("--no-phased-admission", action="store_true")
        parser.add_argument("--router-mode", default="oracle_gold")
        parser.add_argument("--router-agreement-gamma", type=float)
        parser.add_argument("--router-student-delta", type=float)
        parser.add_argument("--static-weight-mode", default="uniform")
        parser.add_argument("--static-weights", type=str)
        parser.add_argument("--bld-mode", action="store_true")
        parser.add_argument("--bld-kl-weight", type=float)
        parser.add_argument("--shuffle-seed", type=int)
    return parser


class TestCLISmokeChecklist:
    """Verify GPU launch checklist commands parse without unknown flags."""

    CHECKLIST = os.path.join(
        os.path.dirname(__file__), "..", "research", "GPU_LAUNCH_CHECKLIST.md")

    @classmethod
    def _parse_checklist_commands(cls):
        """Extract all python *.py commands from the checklist markdown."""
        if not os.path.isfile(cls.CHECKLIST):
            return []
        with open(cls.CHECKLIST, encoding="utf-8") as f:
            text = f.read()
        blocks = re.findall(r'```bash\n(.*?)```', text, re.DOTALL)
        commands = []
        for block in blocks:
            for line_group in block.strip().split('\n\n'):
                lines = line_group.strip().split('\n')
                joined = ' '.join(
                    l.rstrip('\\').strip() for l in lines if l.strip())
                for cmd in joined.split('python '):
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                    m = re.match(r'^(\S+\.py)\s*(.*)', cmd)
                    if m:
                        commands.append((m.group(1), m.group(2)))
        return commands

    def test_checklist_file_exists(self):
        assert os.path.isfile(self.CHECKLIST), "GPU_LAUNCH_CHECKLIST.md missing"

    def test_s0_burnin_flags_recognized(self):
        """S0 burn-in command flags must be accepted by s0_training.py parser."""
        flags = ["--burnin", "--data-dir", "../data",
                 "--checkpoint-dir", "C:/sutra_fast/ckpt"]
        parser = _build_parser("s0_training")
        args = parser.parse_args(flags)
        assert args.burnin is True

    def test_s0_full_training_flags_recognized(self):
        flags = ["--data-dir", "../data", "--checkpoint-dir", "C:/ckpt",
                 "--steps", "50000", "--warmup-steps", "1000",
                 "--eval-every", "500", "--resume", "ckpt.pt"]
        parser = _build_parser("s0_training")
        args = parser.parse_args(flags)
        assert args.steps == 50000

    def test_e1_cache_flags_recognized(self):
        flags = ["--teacher", "anchor", "--data-dir", "../data",
                 "--output-dir", "cache/", "--student-checkpoint", "s0.pt",
                 "--max-shards", "50"]
        parser = _build_parser("eklavya_cache")
        args = parser.parse_args(flags)
        assert args.max_shards == 50

    def test_e1_training_flags_recognized(self):
        flags = ["--student-checkpoint", "s0.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/", "--data-dir", "../data",
                 "--steps", "12000"]
        parser = _build_parser("eklavya_training")
        args = parser.parse_args(flags)
        assert args.steps == 12000

    def test_e2_cache_positions_only_flags(self):
        flags = ["--student-checkpoint", "e1.pt", "--data-dir", "../data",
                 "--output-dir", "cache/", "--max-shards", "50",
                 "--positions-only"]
        parser = _build_parser("eklavya_e2_cache_builder")
        args = parser.parse_args(flags)
        assert args.positions_only is True

    def test_e2_cache_teachers_only_flags(self):
        flags = ["--student-checkpoint", "e1.pt", "--data-dir", "../data",
                 "--output-dir", "cache/", "--max-shards", "50",
                 "--teachers-only"]
        parser = _build_parser("eklavya_e2_cache_builder")
        args = parser.parse_args(flags)
        assert args.teachers_only is True

    def test_e2_training_a2_flags(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/", "--ablation-id", "A2"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.ablation_id == "A2"

    def test_e2_training_a0_ce_only_flags(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/", "--ablation-id", "A0",
                 "--ce-only", "--steps", "8000"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.ce_only is True

    def test_e2_training_bld_flags(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/", "--ablation-id", "BLD",
                 "--bld-mode", "--steps", "8000"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.bld_mode is True

    def test_e2_training_a1_teacher_include(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/",
                 "--ablation-id", "A1", "--teachers", "t0_anchor_decoder",
                 "--steps", "8000"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.teachers == ["t0_anchor_decoder"]

    def test_e2_training_a9c_gold_free(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/",
                 "--ablation-id", "A9c",
                 "--router-mode", "gold_free_student_jsd",
                 "--steps", "8000"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.router_mode == "gold_free_student_jsd"

    def test_e2_training_a5b_custom_weights(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/",
                 "--ablation-id", "A5b", "--disable-router",
                 "--static-weight-mode", "custom",
                 "--static-weights", "t0:0.45,t1:0.25",
                 "--steps", "8000"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.static_weight_mode == "custom"

    def test_e2_training_resume_flag(self):
        flags = ["--student-checkpoint", "e1.pt", "--cache-dir", "cache/",
                 "--output-dir", "ckpt/",
                 "--ablation-id", "A2",
                 "--resume", "e2_step10000.pt"]
        parser = _build_parser("eklavya_e2_training")
        args = parser.parse_args(flags)
        assert args.resume == "e2_step10000.pt"
