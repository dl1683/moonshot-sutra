"""Tests for benchmark_harness.py — scoring functions and data loading."""

import math
import torch
import pytest
import numpy as np

from s0_architecture import S0Config, SutraS0
from benchmark_harness import (
    score_completion,
    score_multiple_choice,
    ScoredChoice,
)


def _datasets_available():
    try:
        import datasets
        return True
    except ImportError:
        return False


@pytest.fixture
def tiny_model():
    cfg = S0Config(
        d_model=64, n_layers=2, n_heads=2, n_kv_heads=1,
        byte_dim=32, local_mixer_layers=1,
        decoder_dim=48, decoder_layers=1, decoder_heads=2,
        decoder_cross_attn=False,
        ffn_mult=2.0, patch_size=4,
    )
    model = SutraS0(cfg)
    model.eval()
    return model


class TestScoreCompletion:
    def test_returns_scored_choice(self, tiny_model):
        ctx = list(b"Hello world ")
        comp = list(b"test")
        result = score_completion(tiny_model, ctx, comp, torch.device("cpu"))
        assert isinstance(result, ScoredChoice)
        assert result.n_bytes > 0
        assert result.bpb > 0
        assert math.isfinite(result.bpb)
        assert math.isfinite(result.total_nll)

    def test_empty_completion_returns_inf(self, tiny_model):
        ctx = list(b"Hello")
        result = score_completion(tiny_model, ctx, [], torch.device("cpu"))
        assert result.bpb == float("inf")
        assert result.n_bytes == 0

    def test_short_context_works(self, tiny_model):
        ctx = list(b"Hi")
        comp = list(b"there")
        result = score_completion(tiny_model, ctx, comp, torch.device("cpu"))
        assert result.n_bytes > 0
        assert math.isfinite(result.bpb)

    def test_longer_completion_has_more_scored_bytes(self, tiny_model):
        ctx = list(b"The cat sat on the ")
        short = list(b"mat")
        long = list(b"comfortable sofa")
        r_short = score_completion(tiny_model, ctx, short, torch.device("cpu"))
        r_long = score_completion(tiny_model, ctx, long, torch.device("cpu"))
        assert r_long.n_bytes > r_short.n_bytes

    def test_different_completions_different_scores(self, tiny_model):
        ctx = list(b"The answer is ")
        comp1 = list(b"yes")
        comp2 = list(b"no")
        r1 = score_completion(tiny_model, ctx, comp1, torch.device("cpu"))
        r2 = score_completion(tiny_model, ctx, comp2, torch.device("cpu"))
        assert r1.total_nll != r2.total_nll or r1.n_bytes != r2.n_bytes

    def test_patch_boundary_alignment(self, tiny_model):
        for ctx_len in [3, 4, 5, 7, 8, 12, 15, 16]:
            ctx = list(range(32, 32 + ctx_len))
            comp = list(range(65, 73))
            result = score_completion(tiny_model, ctx, comp, torch.device("cpu"))
            assert result.n_bytes > 0, f"Failed for ctx_len={ctx_len}"
            assert math.isfinite(result.bpb), f"Non-finite BPB for ctx_len={ctx_len}"

    def test_unicode_context_and_completion(self, tiny_model):
        ctx = list("Café au ".encode("utf-8"))
        comp = list("lait".encode("utf-8"))
        result = score_completion(tiny_model, ctx, comp, torch.device("cpu"))
        assert result.n_bytes > 0
        assert math.isfinite(result.bpb)


class TestScoreMultipleChoice:
    def test_returns_best_index_and_scores(self, tiny_model):
        ctx = "The capital of France is"
        choices = [" Paris", " London", " Berlin", " Madrid"]
        best_idx, scored = score_multiple_choice(
            tiny_model, ctx, choices, torch.device("cpu")
        )
        assert 0 <= best_idx < len(choices)
        assert len(scored) == len(choices)
        assert all(isinstance(s, ScoredChoice) for s in scored)

    def test_normalized_vs_raw_can_differ(self, tiny_model):
        ctx = "Question: What is 2+2? Answer:"
        choices = [" 4", " The answer to this question is four"]
        raw_idx, _ = score_multiple_choice(
            tiny_model, ctx, choices, torch.device("cpu"),
            length_normalize=False
        )
        norm_idx, _ = score_multiple_choice(
            tiny_model, ctx, choices, torch.device("cpu"),
            length_normalize=True
        )
        # These CAN differ (long completion penalized differently)
        assert 0 <= raw_idx < 2
        assert 0 <= norm_idx < 2

    def test_two_choices(self, tiny_model):
        ctx = "Is the sky blue?"
        choices = [" Yes", " No"]
        best_idx, scored = score_multiple_choice(
            tiny_model, ctx, choices, torch.device("cpu")
        )
        assert best_idx in (0, 1)
        assert len(scored) == 2


class TestEvalMultipleChoice:
    def test_single_pass_scoring(self, tiny_model):
        from benchmark_harness import eval_multiple_choice
        examples = [
            {"context": "The cat sat on the", "choices": [" mat", " hat", " bat", " rat"], "label": 0},
            {"context": "Water is", "choices": [" wet", " dry"], "label": 0},
        ]
        result = eval_multiple_choice(tiny_model, examples, torch.device("cpu"), "test_bench")
        assert result.n_examples == 2
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.accuracy_norm <= 1.0
        assert result.mean_bpb > 0
        assert result.elapsed_s >= 0


class TestEvalLambada:
    def test_lambada_single_example(self, tiny_model):
        from benchmark_harness import eval_lambada
        examples = [
            {"context": "The capital of France is", "target": " Paris", "full_text": "The capital of France is Paris"},
        ]
        result = eval_lambada(tiny_model, examples, torch.device("cpu"))
        assert result.n_examples == 1
        assert result.accuracy in (0.0, 1.0)
        assert result.mean_bpb > 0

    def test_predicted_bytes_populated(self, tiny_model):
        ctx = list(b"Hello world ")
        comp = list(b"test")
        result = score_completion(tiny_model, ctx, comp, torch.device("cpu"))
        assert len(result.predicted_bytes) == result.n_bytes
        assert all(0 <= b < 256 for b in result.predicted_bytes)


class TestEvalWinoGrande:
    def test_winogrande_shared_completion(self, tiny_model):
        from benchmark_harness import eval_winogrande
        examples = [
            {
                "context": ["The trophy doesn't fit in the suitcase because the trophy", "The trophy doesn't fit in the suitcase because the suitcase"],
                "completion": "is too big.",
                "label": 0,
            },
        ]
        result = eval_winogrande(tiny_model, examples, torch.device("cpu"))
        assert result.n_examples == 1
        assert result.accuracy in (0.0, 1.0)
        assert result.mean_bpb > 0


class TestNoisify:
    def test_noise_changes_text(self):
        from benchmark_harness import _add_noise
        text = "The quick brown fox jumps over the lazy dog"
        noised = _add_noise(text, noise_rate=0.2, seed=42)
        assert noised != text
        assert len(noised) > 0

    def test_noise_deterministic(self):
        from benchmark_harness import _add_noise
        text = "Hello world"
        a = _add_noise(text, noise_rate=0.15, seed=123)
        b = _add_noise(text, noise_rate=0.15, seed=123)
        assert a == b

    def test_noisify_examples(self):
        from benchmark_harness import noisify_examples
        examples = [
            {"context": "The cat sat on the", "choices": [" mat", " hat"], "label": 0},
        ]
        noised = noisify_examples(examples, noise_rate=0.15)
        assert len(noised) == 1
        assert noised[0]["label"] == 0
        assert noised[0]["context"] != examples[0]["context"]


class TestDataLoaders:
    @pytest.mark.skipif(
        not _datasets_available(),
        reason="HuggingFace datasets not available"
    )
    def test_hellaswag_format(self):
        from benchmark_harness import load_hellaswag
        examples = load_hellaswag()[:3]
        for ex in examples:
            assert "context" in ex
            assert "choices" in ex
            assert "label" in ex
            assert len(ex["choices"]) == 4
            assert 0 <= ex["label"] < 4

    @pytest.mark.skipif(
        not _datasets_available(),
        reason="HuggingFace datasets not available"
    )
    def test_piqa_format(self):
        from benchmark_harness import load_piqa
        examples = load_piqa()[:3]
        for ex in examples:
            assert "context" in ex
            assert ex["context"].startswith("Question: ")
            assert "choices" in ex
            assert "label" in ex
            assert len(ex["choices"]) == 2
