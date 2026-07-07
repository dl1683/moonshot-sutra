"""Coordinate-inheritance prototype for byte-wrapped Qwen geometry.

Stage 1 trains a <=2M calibration adapter from frozen codec encoder states into
raw Qwen embedding gauge, then tests copied Qwen layers against random and
coordinate-disrupted controls under token-space next-token loss.

Stage 2 benchmark mode is intentionally a prototype scorer: bytes enter through
the codec, the inherited Qwen-shaped core scores continuations with the Qwen
LM head, and no byte decoder/BPB claim is made.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from semantic_codec import RMSNorm  # noqa: E402
from tier3_brainseed_chart_probe import ByteShardSampler, _token_spans_for_bytes, load_codec, load_tokenizer  # noqa: E402

DEFAULT_CODEC = "C:/sutra_fast/codec_phase1.5/codec_final.pt"
DEFAULT_DATA_DIR = "C:/sutra_fast/data/shards_diverse"
DEFAULT_OUTPUT_DIR = "C:/sutra_fast/coordinate_inheritance"
DEFAULT_TEACHER = "Qwen/Qwen3-0.6B"

WIDE7_BASELINE = {
    "hellaswag": 0.266,
    "piqa": 0.509,
    "arc_easy": 0.277,
    "arc_challenge": 0.228,
    "bpb": 1.293,
}

READOUT_NAMES = ("token_end", "patch_boundary")
READOUT_TO_ID = {name: i for i, name in enumerate(READOUT_NAMES)}


@dataclass
class SequenceSample:
    codec_hidden: torch.Tensor
    labels: torch.Tensor
    positions: torch.Tensor
    readout: str


@dataclass
class ScoredCompletion:
    total_nll: float
    n_tokens: int

    @property
    def nll_per_token(self) -> float:
        return float("inf") if self.n_tokens <= 0 else self.total_nll / self.n_tokens

    def to_json(self) -> dict:
        return {
            "total_nll": None if not math.isfinite(self.total_nll) else self.total_nll,
            "n_tokens": self.n_tokens,
            "nll_per_token": None if not math.isfinite(self.nll_per_token) else self.nll_per_token,
        }


class CalibrationAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 1024,
        kind: str = "rms_linear",
        rank: int = 256,
        conditioning: str = "shared",
        readouts: tuple[str, ...] = READOUT_NAMES,
    ):
        super().__init__()
        self.kind = kind
        self.conditioning = conditioning
        self.output_dim = output_dim
        self.readouts = tuple(readouts)
        if conditioning == "shared":
            self.net = self._make_net(input_dim, output_dim, kind, rank)
        elif conditioning == "readout":
            self.nets = nn.ModuleDict({readout: self._make_net(input_dim, output_dim, kind, rank) for readout in self.readouts})
        else:
            raise ValueError(conditioning)

    @staticmethod
    def _make_net(input_dim: int, output_dim: int, kind: str, rank: int) -> nn.Module:
        if kind == "linear":
            return nn.Linear(input_dim, output_dim)
        if kind == "rms_linear":
            return nn.Sequential(RMSNorm(input_dim), nn.Linear(input_dim, output_dim))
        if kind == "low_rank":
            return nn.Sequential(RMSNorm(input_dim), nn.Linear(input_dim, rank, bias=False), nn.GELU(), nn.Linear(rank, output_dim))
        raise ValueError(kind)

    def forward(self, x: torch.Tensor, readout: str | None = None, readout_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        if self.conditioning == "shared":
            return self.net(x)
        if readout is not None:
            return self.nets[readout](x)
        if readout_ids is None:
            raise ValueError("readout-conditioned adapter requires readout or readout_ids")
        out = torch.zeros((*x.shape[:-1], self.output_dim), dtype=torch.float32, device=x.device)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_ids = readout_ids.reshape(-1)
        flat_out = out.reshape(-1, self.output_dim)
        for readout_name in self.readouts:
            readout_id = READOUT_TO_ID[readout_name]
            mask = flat_ids.eq(readout_id)
            if bool(mask.any().item()):
                flat_out[mask] = self.nets[readout_name](flat_x[mask])
        return out


def ensure_offline(allow_downloads: bool) -> None:
    if not allow_downloads:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def load_teacher(name: str, device: torch.device, allow_downloads: bool):
    ensure_offline(allow_downloads)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, local_files_only=not allow_downloads)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def embedding_mean_norm(teacher) -> float:
    return float(teacher.model.embed_tokens.weight.detach().float().norm(dim=-1).mean().item())


def copy_head_tensors(dst: dict[str, torch.Tensor], teacher_state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    copied = 0
    for key, value in list(dst.items()):
        if key.startswith("model.embed_tokens") or key.startswith("model.norm") or key.startswith("lm_head"):
            if key in teacher_state and tuple(teacher_state[key].shape) == tuple(value.shape):
                dst[key] = teacher_state[key].detach().to(device=value.device, dtype=value.dtype)
                copied += 1
    return dst, copied


def build_qwen_variant(teacher, layers: int, variant: str, device: torch.device, seed: int, source_layer_start: int = 0):
    total_layers = int(teacher.config.num_hidden_layers)
    if layers < 1 or layers > total_layers:
        raise ValueError(f"layers must be 1..{teacher.config.num_hidden_layers}")
    if variant != "random" and (source_layer_start < 0 or source_layer_start + layers > total_layers):
        raise ValueError(f"source layer range {source_layer_start}:{source_layer_start + layers} is outside 0..{total_layers}")
    cfg = copy.deepcopy(teacher.config)
    cfg.num_hidden_layers = layers
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype).eval()
    teacher_state = teacher.state_dict()
    state, copied_head = copy_head_tensors(model.state_dict(), teacher_state)
    layer_order = None
    copied_layers = 0
    if variant in {"copied", "shuffled", "generic_pretrained"}:
        layer_order = list(range(source_layer_start, source_layer_start + layers))
        if variant == "shuffled":
            rng = np.random.default_rng(seed)
            rng.shuffle(layer_order)
        for dst_layer, src_layer in enumerate(layer_order):
            dst_prefix = f"model.layers.{dst_layer}."
            src_prefix = f"model.layers.{src_layer}."
            for key, value in list(state.items()):
                if not key.startswith(dst_prefix):
                    continue
                src_key = src_prefix + key[len(dst_prefix):]
                if src_key in teacher_state and tuple(teacher_state[src_key].shape) == tuple(value.shape):
                    state[key] = teacher_state[src_key].detach().to(device=value.device, dtype=value.dtype)
                    copied_layers += 1
    elif variant != "random":
        raise ValueError(variant)
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {
        "variant": variant,
        "layers": layers,
        "source_layer_start": None if variant == "random" else source_layer_start,
        "source_layer_end_exclusive": None if variant == "random" else source_layer_start + layers,
        "copied_head_tensors": copied_head,
        "copied_layer_tensors": copied_layers,
        "layer_order": layer_order,
    }


def collect_sequence_sample(codec, tokenizer, byte_row: torch.Tensor, readout: str, device: torch.device, max_positions: int, rng: np.random.Generator) -> SequenceSample | None:
    spans = _token_spans_for_bytes(byte_row, tokenizer)
    if len(spans) < 3:
        return None
    records: list[tuple[int, int]] = []
    if readout == "token_end":
        records = [(end, tid) for _, end, tid in spans]
    elif readout == "patch_boundary":
        span_idx = 0
        for pos in range(codec.cfg.patch_size - 1, int(byte_row.shape[0]), codec.cfg.patch_size):
            while span_idx < len(spans) and spans[span_idx][1] < pos:
                span_idx += 1
            if span_idx >= len(spans):
                break
            start, end, tid = spans[span_idx]
            if start <= pos <= end:
                records.append((pos, tid))
    else:
        raise ValueError(readout)
    if len(records) < 3:
        return None
    if max_positions > 0 and len(records) > max_positions:
        chosen = np.sort(rng.choice(len(records), size=max_positions, replace=False))
        records = [records[int(i)] for i in chosen]
    byte_ids = byte_row.unsqueeze(0).to(device)
    positions = torch.tensor([p for p, _ in records], dtype=torch.long, device=device)
    labels = torch.tensor([tid for _, tid in records], dtype=torch.long)
    with torch.no_grad():
        hidden = codec.encoder(byte_ids)[0].float().index_select(0, positions)
    return SequenceSample(hidden.cpu(), labels.cpu(), positions.cpu(), readout)


def collect_samples(codec, tokenizer, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, list[SequenceSample]], dict]:
    rng = np.random.default_rng(args.seed)
    sampler = ByteShardSampler(args.data_dir, seq_len=args.seq_len)
    out = {readout: [] for readout in args.eval_readouts}
    attempts = 0
    max_attempts = max(100, args.num_sequences * len(args.eval_readouts) * 20)
    while any(len(v) < args.num_sequences for v in out.values()) and attempts < max_attempts:
        attempts += 1
        row = sampler.sample(1, rng)[0]
        for readout in args.eval_readouts:
            if len(out[readout]) >= args.num_sequences:
                continue
            sample = collect_sequence_sample(codec, tokenizer, row, readout, device, args.max_positions_per_sequence, rng)
            if sample is not None:
                out[readout].append(sample)
    missing = {k: args.num_sequences - len(v) for k, v in out.items() if len(v) < args.num_sequences}
    if missing:
        raise RuntimeError(f"sample collection failed after {attempts} attempts: {missing}")
    manifest = {
        "data_dir": args.data_dir,
        "seq_len": args.seq_len,
        "num_sequences_per_readout": args.num_sequences,
        "attempts": attempts,
        "total_shard_bytes": sampler.total_bytes,
        "readouts": {k: {"n": len(v), "mean_len": float(np.mean([len(s.labels) for s in v])), "min_len": min(len(s.labels) for s in v), "max_len": max(len(s.labels) for s in v)} for k, v in out.items()},
    }
    return out, manifest


def split_samples(samples: list[SequenceSample], train_fraction: float, seed: int) -> tuple[list[SequenceSample], list[SequenceSample]]:
    if len(samples) <= 1:
        return samples, samples
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_train = max(1, min(len(samples) - 1, int(round(len(samples) * train_fraction))))
    train_set = set(int(i) for i in idx[:n_train])
    train = [s for i, s in enumerate(samples) if i in train_set]
    evals = [s for i, s in enumerate(samples) if i not in train_set]
    return train, evals or train


def flatten_samples(samples_by_readout: dict[str, list[SequenceSample]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, labels, readout_ids = [], [], []
    for readout, samples in samples_by_readout.items():
        readout_id = READOUT_TO_ID[readout]
        for sample in samples:
            hidden.append(sample.codec_hidden)
            labels.append(sample.labels)
            readout_ids.append(torch.full_like(sample.labels, readout_id))
    return torch.cat(hidden, dim=0), torch.cat(labels, dim=0), torch.cat(readout_ids, dim=0)


def train_adapter(adapter: CalibrationAdapter, teacher, train_samples: dict[str, list[SequenceSample]], device: torch.device, args: argparse.Namespace) -> dict:
    adapter.to(device).train()
    hidden_cpu, labels_cpu, readout_ids_cpu = flatten_samples(train_samples)
    n = int(labels_cpu.numel())
    if n == 0:
        raise RuntimeError("no adapter anchors")
    rng = np.random.default_rng(args.seed + 101)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.adapter_lr, weight_decay=0.01)
    losses = []
    for step in range(1, args.adapter_steps + 1):
        idx = rng.integers(0, n, size=min(args.adapter_batch_anchors, n))
        x = hidden_cpu[idx].to(device)
        y = labels_cpu[idx].to(device)
        readout_ids = readout_ids_cpu[idx].to(device)
        with torch.no_grad():
            target = teacher.model.embed_tokens(y).float()
        pred = adapter(x, readout_ids=readout_ids)
        mse = F.mse_loss(pred, target)
        cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
        norm_loss = F.smooth_l1_loss(pred.norm(dim=-1), target.norm(dim=-1))
        loss = mse + 0.10 * cosine + 0.01 * norm_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if args.progress and (step == 1 or step % max(1, args.adapter_steps // 10) == 0):
            print(f"adapter step {step}/{args.adapter_steps}: loss={loss.item():.5f} mse={mse.item():.5f}", flush=True)
    adapter.eval()
    return {"steps": args.adapter_steps, "batch_anchors": args.adapter_batch_anchors, "lr": args.adapter_lr, "anchors": n, "loss_first": losses[0], "loss_last": losses[-1], "loss_min": min(losses)}


def pad_samples(samples: list[SequenceSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(s.labels.numel()) for s in samples)
    hidden_dim = int(samples[0].codec_hidden.shape[-1])
    hidden = torch.zeros((len(samples), max_len, hidden_dim), dtype=torch.float32, device=device)
    labels = torch.full((len(samples), max_len), -100, dtype=torch.long, device=device)
    attention = torch.zeros((len(samples), max_len), dtype=torch.long, device=device)
    readout_ids = torch.zeros((len(samples), max_len), dtype=torch.long, device=device)
    for i, sample in enumerate(samples):
        n = int(sample.labels.numel())
        hidden[i, :n] = sample.codec_hidden.to(device)
        labels[i, :n] = sample.labels.to(device)
        attention[i, :n] = 1
        readout_ids[i, :n] = READOUT_TO_ID[sample.readout]
    return hidden, labels, attention, readout_ids


def make_inputs(hidden: torch.Tensor, labels: torch.Tensor, input_kind: str, teacher, codec, adapter: CalibrationAdapter | None, scale: float, dtype: torch.dtype, transform: Callable[[torch.Tensor], torch.Tensor] | None = None, readout_ids: torch.Tensor | None = None) -> torch.Tensor:
    if input_kind == "calibrated":
        if adapter is None:
            raise ValueError("calibrated input requires adapter")
        inputs = adapter(hidden, readout_ids=readout_ids)
    elif input_kind == "raw_codec":
        inputs = codec.alignment_head(hidden.float()) * scale
    elif input_kind == "true_embedding":
        input_labels = labels.clamp_min(0)
        inputs = teacher.model.embed_tokens(input_labels).float()
        inputs = inputs.masked_fill(labels.eq(-100).unsqueeze(-1), 0.0)
    else:
        raise ValueError(input_kind)
    if transform is not None:
        inputs = transform(inputs.float())
    return inputs.to(dtype=dtype)


def shifted_ce_per_sequence(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        zeros = torch.zeros(labels.shape[0], dtype=torch.float32, device=logits.device)
        return zeros, zeros.long(), 0, 0
    loss_flat = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
    loss_sum = (loss_flat * mask.float()).sum(dim=1)
    token_count = mask.sum(dim=1)
    pred = shift_logits.argmax(dim=-1)
    correct = int((pred[mask] == shift_labels[mask]).sum().item())
    total = int(mask.sum().item())
    return loss_sum, token_count, correct, total


@torch.no_grad()
def evaluate_nll(model, samples: list[SequenceSample], teacher, codec, adapter: CalibrationAdapter | None, scale: float, batch_size: int, device: torch.device, input_kind: str, transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> dict:
    dtype = next(model.parameters()).dtype
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    per_sequence = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        hidden, labels, attention, readout_ids = pad_samples(chunk, device)
        inputs = make_inputs(hidden, labels, input_kind, teacher, codec, adapter, scale, dtype, transform, readout_ids)
        out = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
        loss_sum, token_count, correct, total = shifted_ce_per_sequence(out.logits, labels)
        total_loss += float(loss_sum.sum().item())
        total_tokens += total
        total_correct += correct
        for loss_i, tok_i in zip(loss_sum.detach().cpu().tolist(), token_count.detach().cpu().tolist()):
            if int(tok_i) > 0:
                per_sequence.append(float(loss_i) / int(tok_i))
    nll = total_loss / max(1, total_tokens)
    return {"nll": nll, "ppl_capped": float(math.exp(min(20.0, nll))), "next_token_acc": total_correct / max(1, total_tokens), "tokens": total_tokens, "sequences": len(per_sequence), "per_sequence_nll": per_sequence}


def paired_bootstrap_delta(a: list[float], b: list[float], samples: int, seed: int) -> dict:
    n = min(len(a), len(b))
    if n == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    delta = np.asarray(a[:n], dtype=np.float64) - np.asarray(b[:n], dtype=np.float64)
    if samples <= 0 or n == 1:
        mean = float(delta.mean())
        return {"mean": mean, "ci95": [mean, mean], "n": n}
    rng = np.random.default_rng(seed)
    boot = [float(delta[rng.integers(0, n, size=n)].mean()) for _ in range(samples)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(delta.mean()), "ci95": [float(lo), float(hi)], "n": n}


def random_orthogonal(dim: int, device: torch.device, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn(dim, dim, generator=gen, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    return (q * signs).to(device)


def random_permutation(dim: int, device: torch.device, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.permutation(dim), dtype=torch.long, device=device)


def random_half_mask(dim: int, device: torch.device, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    mask = np.zeros(dim, dtype=np.float32)
    keep = rng.choice(dim, size=max(1, dim // 2), replace=False)
    mask[keep] = 1.0
    return torch.tensor(mask, dtype=torch.float32, device=device)


def inherited_lift_fraction(random_nll: float, copied_nll: float, variant_nll: float) -> float | None:
    denom = random_nll - copied_nll
    if denom <= 0:
        return None
    return (random_nll - variant_nll) / denom


def summarize_stage1_readout(results: dict, bootstrap_samples: int, seed: int, readout: str) -> dict:
    copied = results["copied_calibrated"]
    random_res = results["random_calibrated"]
    true_res = results["copied_true_embedding"]
    noinv = results["copied_rotated_no_inverse"]
    inv = results["copied_rotated_with_inverse"]
    generic = results.get("generic_pretrained_calibrated")
    random_minus_copied = random_res["nll"] - copied["nll"]
    random_minus_true = random_res["nll"] - true_res["nll"]
    gap_to_true = copied["nll"] - true_res["nll"]
    closure = random_minus_copied / random_minus_true if random_minus_true > 0 else None
    noinv_remaining = inherited_lift_fraction(random_res["nll"], copied["nll"], noinv["nll"])
    inv_recovery = inherited_lift_fraction(random_res["nll"], copied["nll"], inv["nll"])
    generic_minus_copied = generic["nll"] - copied["nll"] if generic is not None else None
    strong_disruption_fractions = {}
    for name, result_key in {
        "dim_permutation": "copied_dim_permuted",
        "zeroed_50pct": "copied_zeroed_50pct",
        "gaussian_norm_noise": "copied_gaussian_norm_noise",
    }.items():
        if result_key in results:
            strong_disruption_fractions[name] = inherited_lift_fraction(random_res["nll"], copied["nll"], results[result_key]["nll"])
    primary_strong_fraction = strong_disruption_fractions.get("gaussian_norm_noise")
    gap_limit = 2.0 if readout == "patch_boundary" else 1.5
    ci = paired_bootstrap_delta(random_res["per_sequence_nll"], copied["per_sequence_nll"], bootstrap_samples, seed)
    gates = {
        "copied_advantage_ge_2_nats": random_minus_copied >= 2.0,
        "readout_specific_gap_gate": gap_to_true <= gap_limit or (closure is not None and closure >= 0.60),
        "generic_pretrained_gap_ge_0_75_nats": generic_minus_copied is not None and generic_minus_copied >= 0.75,
        "rotation_no_inverse_collapses": noinv_remaining is not None and noinv_remaining <= 0.30,
        "rotation_inverse_recovers_ge_80pct": inv_recovery is not None and inv_recovery >= 0.80,
        "strong_disruption_gaussian_norm_noise_le_20pct": primary_strong_fraction is not None and primary_strong_fraction <= 0.20,
        "frozen_core_gain_ge_70pct": False,
    }
    return {
        "metrics": {
            "random_minus_copied_nll": random_minus_copied,
            "random_minus_true_nll": random_minus_true,
            "gap_to_true_nll": gap_to_true,
            "gap_closure": closure,
            "generic_pretrained_minus_copied_nll": generic_minus_copied,
            "rotated_no_inverse_lift_fraction": noinv_remaining,
            "rotated_with_inverse_recovery_fraction": inv_recovery,
            "strong_disruption_lift_fractions": strong_disruption_fractions,
            "primary_strong_disruption": "gaussian_norm_noise",
            "advantage_ci_random_minus_copied": ci,
            "frozen_core_gain_fraction_of_finetuned": None,
        },
        "gates": gates,
    }


def finetune_core(model, adapter: CalibrationAdapter, train_samples: dict[str, list[SequenceSample]], teacher, device: torch.device, args: argparse.Namespace):
    ft_model = copy.deepcopy(model).float()
    ft_adapter = copy.deepcopy(adapter).float()
    ft_model.train()
    ft_adapter.train()
    trainable_model_params = []
    for name, p in ft_model.named_parameters():
        trainable = name.startswith("model.layers") or name.startswith("model.norm")
        p.requires_grad_(trainable)
        if trainable:
            trainable_model_params.append(p)
    for p in ft_adapter.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([
        {"params": trainable_model_params, "lr": args.finetune_lr},
        {"params": ft_adapter.parameters(), "lr": args.finetune_lr * 5.0},
    ], weight_decay=0.01)
    all_samples = [s for samples in train_samples.values() for s in samples]
    rng = np.random.default_rng(args.seed + 404)
    dtype = next(ft_model.parameters()).dtype
    losses = []
    for step in range(1, args.finetune_core_steps + 1):
        idx = rng.integers(0, len(all_samples), size=min(args.finetune_batch_sequences, len(all_samples)))
        chunk = [all_samples[int(i)] for i in idx]
        hidden, labels, attention, readout_ids = pad_samples(chunk, device)
        inputs = make_inputs(hidden, labels, "calibrated", teacher, None, ft_adapter, 1.0, dtype, readout_ids=readout_ids)
        logits = ft_model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            print(f"core finetune stopped at step {step}: non-finite loss={loss.item()}", flush=True)
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_model_params + list(ft_adapter.parameters()), 0.5)
        opt.step()
        losses.append(float(loss.item()))
        if args.progress and (step == 1 or step % max(1, args.finetune_core_steps // 5) == 0):
            print(f"core finetune step {step}/{args.finetune_core_steps}: loss={loss.item():.5f}", flush=True)
    ft_model.eval()
    ft_adapter.eval()
    for p in ft_model.parameters():
        p.requires_grad_(False)
    for p in ft_adapter.parameters():
        p.requires_grad_(False)
    return ft_model, ft_adapter, {"steps": args.finetune_core_steps, "batch_sequences": args.finetune_batch_sequences, "lr": args.finetune_lr, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None}


def run_layer_depth_curve(teacher, codec, adapter: CalibrationAdapter, train_by_readout: dict[str, list[SequenceSample]], eval_by_readout: dict[str, list[SequenceSample]], scale: float, device: torch.device, args: argparse.Namespace) -> dict:
    curve: dict[str, dict] = {}
    for depth in args.depth_curve_layers:
        depth = int(depth)
        copied_model, copied_manifest = build_qwen_variant(teacher, depth, "copied", device, args.seed)
        random_model, random_manifest = build_qwen_variant(teacher, depth, "random", device, args.seed)
        ft_model = None
        ft_adapter = None
        ft_manifest = None
        if args.finetune_core_steps > 0:
            ft_model, ft_adapter, ft_manifest = finetune_core(copied_model, adapter, train_by_readout, teacher, device, args)
        readouts = {}
        for readout in args.eval_readouts:
            eval_samples = eval_by_readout[readout]
            copied_res = evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated")
            random_res = evaluate_nll(random_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated")
            finetuned_res = None
            ratio = None
            if ft_model is not None and ft_adapter is not None:
                finetuned_res = evaluate_nll(ft_model, eval_samples, teacher, codec, ft_adapter, scale, args.batch_size, device, "calibrated")
                frozen_gain = random_res["nll"] - copied_res["nll"]
                finetuned_gain = random_res["nll"] - finetuned_res["nll"]
                if finetuned_gain > 0:
                    ratio = frozen_gain / finetuned_gain
                elif frozen_gain > 0:
                    ratio = 1.0
            readouts[readout] = {
                "copied_calibrated": {k: v for k, v in copied_res.items() if k != "per_sequence_nll"},
                "random_calibrated": {k: v for k, v in random_res.items() if k != "per_sequence_nll"},
                "finetuned_core_calibrated": None if finetuned_res is None else {k: v for k, v in finetuned_res.items() if k != "per_sequence_nll"},
                "metrics": {
                    "random_minus_copied_nll": random_res["nll"] - copied_res["nll"],
                    "frozen_core_gain_fraction_of_finetuned": ratio,
                },
                "gates": {"frozen_core_gain_ge_70pct": ratio is not None and ratio >= 0.70},
            }
        curve[str(depth)] = {
            "copy_manifests": {"copied": copied_manifest, "random": random_manifest},
            "core_finetune": ft_manifest,
            "readouts": readouts,
        }
        del copied_model, random_model, ft_model, ft_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return curve


def run_preflight(args: argparse.Namespace) -> dict:
    started = time.time()
    set_seed(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)
    scale = embedding_mean_norm(teacher) if args.scale == "auto" else float(args.scale)

    samples_by_readout, data_manifest = collect_samples(codec, tokenizer, args, device)
    train_by_readout, eval_by_readout = {}, {}
    for readout, samples in samples_by_readout.items():
        train, evals = split_samples(samples, args.train_fraction, args.seed + 17)
        train_by_readout[readout] = train
        eval_by_readout[readout] = evals

    adapter = CalibrationAdapter(kind=args.adapter_kind, rank=args.adapter_rank, conditioning=args.adapter_conditioning)
    adapter_manifest = {
        "kind": args.adapter_kind,
        "rank": args.adapter_rank,
        "conditioning": args.adapter_conditioning,
        "readouts": list(READOUT_NAMES),
        "params": param_count(adapter),
        "param_gate_le_2m": param_count(adapter) <= 2_000_000,
    }
    adapter_training = train_adapter(adapter, teacher, train_by_readout, device, args)

    copied_model, copied_manifest = build_qwen_variant(teacher, args.layers, "copied", device, args.seed)
    random_model, random_manifest = build_qwen_variant(teacher, args.layers, "random", device, args.seed)
    shuffled_model, shuffled_manifest = build_qwen_variant(teacher, args.layers, "shuffled", device, args.seed + 23)
    generic_model, generic_manifest = build_qwen_variant(teacher, args.layers, "generic_pretrained", device, args.seed, source_layer_start=args.generic_layer_start)
    rot = random_orthogonal(int(teacher.config.hidden_size), device, args.seed + 303)
    dim_perm = random_permutation(int(teacher.config.hidden_size), device, args.seed + 505)
    half_mask = random_half_mask(int(teacher.config.hidden_size), device, args.seed + 606).view(1, 1, -1)

    def rotate_no_inverse(x: torch.Tensor) -> torch.Tensor:
        return x @ rot

    def rotate_with_inverse(x: torch.Tensor) -> torch.Tensor:
        return (x @ rot) @ rot.t()

    def dim_permutation(x: torch.Tensor) -> torch.Tensor:
        return x.index_select(-1, dim_perm)

    def zero_half_dims(x: torch.Tensor) -> torch.Tensor:
        return x * half_mask.to(device=x.device, dtype=x.dtype)

    def gaussian_norm_noise(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        noise = torch.randn_like(x_float)
        noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_norm = x_float.norm(dim=-1, keepdim=True)
        return noise * (x_norm / noise_norm)

    ft_model = None
    ft_adapter = None
    ft_manifest = None
    if args.finetune_core_steps > 0:
        ft_model, ft_adapter, ft_manifest = finetune_core(copied_model, adapter, train_by_readout, teacher, device, args)

    readout_payloads = {}
    for readout in args.eval_readouts:
        eval_samples = eval_by_readout[readout]
        results = {
            "copied_raw_codec": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "raw_codec"),
            "copied_calibrated": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated"),
            "random_calibrated": evaluate_nll(random_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated"),
            "shuffled_calibrated": evaluate_nll(shuffled_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated"),
            "generic_pretrained_calibrated": evaluate_nll(generic_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated"),
            "copied_true_embedding": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "true_embedding"),
            "copied_rotated_no_inverse": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated", rotate_no_inverse),
            "copied_rotated_with_inverse": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated", rotate_with_inverse),
            "copied_dim_permuted": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated", dim_permutation),
            "copied_zeroed_50pct": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated", zero_half_dims),
            "copied_gaussian_norm_noise": evaluate_nll(copied_model, eval_samples, teacher, codec, adapter, scale, args.batch_size, device, "calibrated", gaussian_norm_noise),
        }
        if ft_model is not None and ft_adapter is not None:
            results["finetuned_core_calibrated"] = evaluate_nll(ft_model, eval_samples, teacher, codec, ft_adapter, scale, args.batch_size, device, "calibrated")
        summary = summarize_stage1_readout(results, args.bootstrap_samples, args.seed + 700, readout)
        if "finetuned_core_calibrated" in results:
            random_nll = results["random_calibrated"]["nll"]
            frozen_gain = random_nll - results["copied_calibrated"]["nll"]
            finetuned_gain = random_nll - results["finetuned_core_calibrated"]["nll"]
            if finetuned_gain > 0:
                ratio = frozen_gain / finetuned_gain
            elif frozen_gain > 0:
                ratio = 1.0
            else:
                ratio = None
            summary["metrics"]["frozen_core_gain_fraction_of_finetuned"] = ratio
            summary["gates"]["frozen_core_gain_ge_70pct"] = ratio is not None and ratio >= 0.70
        readout_payloads[readout] = {
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "per_sequence_nll"} for k, v in results.items()},
            "summary": summary,
        }

    stage1_gates = {"adapter_params_le_2m": adapter_manifest["param_gate_le_2m"]}
    for readout, payload in readout_payloads.items():
        gates = payload["summary"]["gates"]
        stage1_gates[f"{readout}_copied_advantage_ge_2_nats"] = gates["copied_advantage_ge_2_nats"]
        stage1_gates[f"{readout}_gap_gate"] = gates["readout_specific_gap_gate"]
        stage1_gates[f"{readout}_generic_pretrained_gap_ge_0_75_nats"] = gates["generic_pretrained_gap_ge_0_75_nats"]
        stage1_gates[f"{readout}_rotation_inverse_recovers_ge_80pct"] = gates["rotation_inverse_recovers_ge_80pct"]
        stage1_gates[f"{readout}_strong_disruption_gaussian_norm_noise_le_20pct"] = gates["strong_disruption_gaussian_norm_noise_le_20pct"]
        stage1_gates[f"{readout}_frozen_core_gain_ge_70pct"] = gates["frozen_core_gain_ge_70pct"]
    depth_curve = run_layer_depth_curve(teacher, codec, adapter, train_by_readout, eval_by_readout, scale, device, args) if args.depth_curve_layers else {}
    stage1_pass = all(stage1_gates.values())

    payload = {
        "mode": "preflight",
        "run": {"seed": args.seed, "device": str(device), "teacher": args.teacher, "layers": args.layers, "generic_layer_start": args.generic_layer_start, "depth_curve_layers": args.depth_curve_layers, "readouts": args.eval_readouts, "num_sequences_per_readout": args.num_sequences, "seq_len": args.seq_len, "scale": scale, "elapsed_s": round(time.time() - started, 3)},
        "codec": codec_manifest,
        "qwen_config": {"hidden_size": teacher.config.hidden_size, "num_hidden_layers": teacher.config.num_hidden_layers, "num_attention_heads": teacher.config.num_attention_heads, "num_key_value_heads": teacher.config.num_key_value_heads, "vocab_size": teacher.config.vocab_size},
        "adapter": adapter_manifest,
        "adapter_training": adapter_training,
        "core_finetune": ft_manifest,
        "data": data_manifest,
        "copy_manifests": {"copied": copied_manifest, "random": random_manifest, "shuffled": shuffled_manifest, "generic_pretrained": generic_manifest},
        "readouts": readout_payloads,
        "layer_depth_curve": depth_curve,
        "precommitted_verdict_tokens": {
            "stage1_pass": "PASS_STAGE1_V1_PREFLIGHT__PROCEED_TO_STAGE2",
            "stage1_fail": "FAIL_STAGE1_V1_PREFLIGHT__DO_NOT_RUN_STAGE2",
            "stage2_pass": "PASS_STAGE2_UNCOMPRESSED_BYTEIFIED_INHERITANCE",
            "stage2_fail": "FAIL_STAGE2__DEMOTE_COORDINATE_INHERITANCE_TO_CODEC_DIAGNOSTIC",
        },
        "stage1_gates": stage1_gates,
        "stage1_pass": stage1_pass,
        "verdict": "PASS_STAGE1_V1_PREFLIGHT" if stage1_pass else "FAIL_STAGE1_V1_PREFLIGHT",
        "limitations": [
            "NLL is token-space next-token loss through a Qwen head, not byte BPB.",
            "Patch-boundary sequences can repeat token labels when multiple patch boundaries fall inside one token.",
            "Rotation sanity is input-gauge disruption/recovery, not a full model-weight basis transform.",
            "The stronger disruption gate uses same-norm Gaussian replacement as the precommitted primary control and separately reports hidden-dimension permutation and zeroed-50% dimensions.",
            "Generic pretrained control uses a different Qwen layer range, not a separately trained non-Qwen architecture.",
            "Frozen-core gain uses the optional short fp32 copied-core finetune path when --finetune-core-steps is set.",
        ],
    }
    if not args.no_artifacts:
        out_dir = Path(args.output_dir)
        write_json(out_dir / "preflight_metrics.json", payload)
        torch.save({"adapter_state_dict": adapter.cpu().state_dict(), "adapter_manifest": adapter_manifest, "teacher": args.teacher, "codec_checkpoint": args.codec_checkpoint}, out_dir / "calibration_adapter.pt")
        adapter.to(device)
    return payload


def cached_arrow_path(name: str, split: str) -> Path | None:
    home = Path.home()
    candidates = {
        ("hellaswag", "train"): home / ".cache" / "huggingface" / "datasets" / "Rowan___hellaswag" / "default" / "0.0.0" / "218ec52e09a7e7462a5400043bb9a69a41d06b76" / "hellaswag-train.arrow",
        ("hellaswag", "validation"): home / ".cache" / "huggingface" / "datasets" / "Rowan___hellaswag" / "default" / "0.0.0" / "218ec52e09a7e7462a5400043bb9a69a41d06b76" / "hellaswag-validation.arrow",
        ("piqa", "train"): home / ".cache" / "huggingface" / "datasets" / "baber___piqa" / "default" / "0.0.0" / "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1" / "piqa-train.arrow",
        ("piqa", "validation"): home / ".cache" / "huggingface" / "datasets" / "baber___piqa" / "default" / "0.0.0" / "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1" / "piqa-validation.arrow",
        ("arc_easy", "train"): home / ".cache" / "huggingface" / "datasets" / "allenai___ai2_arc" / "ARC-Easy" / "0.0.0" / "210d026faf9955653af8916fad021475a3f00453" / "ai2_arc-train.arrow",
        ("arc_easy", "validation"): home / ".cache" / "huggingface" / "datasets" / "allenai___ai2_arc" / "ARC-Easy" / "0.0.0" / "210d026faf9955653af8916fad021475a3f00453" / "ai2_arc-validation.arrow",
        ("arc_challenge", "train"): home / ".cache" / "huggingface" / "datasets" / "allenai___ai2_arc" / "ARC-Challenge" / "0.0.0" / "210d026faf9955653af8916fad021475a3f00453" / "ai2_arc-train.arrow",
        ("arc_challenge", "validation"): home / ".cache" / "huggingface" / "datasets" / "allenai___ai2_arc" / "ARC-Challenge" / "0.0.0" / "210d026faf9955653af8916fad021475a3f00453" / "ai2_arc-validation.arrow",
    }
    path = candidates.get((name, split))
    return path if path is not None and path.exists() else None


def hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_limited_benchmark(name: str, count: int, split: str, seed: int, allow_downloads: bool) -> list[dict]:
    ensure_offline(allow_downloads)
    from datasets import Dataset, load_dataset
    arrow = cached_arrow_path(name, split)
    if arrow is not None:
        ds = Dataset.from_file(str(arrow))
    elif name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split=split)
    elif name == "piqa":
        ds = load_dataset("baber/piqa", split=split)
    elif name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    elif name == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    else:
        raise ValueError(name)
    n = min(count, len(ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist() if n < len(ds) else list(range(n))
    ds = ds.select(indices)
    examples = []
    if name == "hellaswag":
        for local_idx, row in enumerate(ds):
            ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
            examples.append({
                "id": f"{name}:{split}:{indices[local_idx]}",
                "source_index": int(indices[local_idx]),
                "context": hellaswag_preprocess(row["activity_label"] + ": " + ctx),
                "choices": [hellaswag_preprocess(e) for e in row["endings"]],
                "label": int(row["label"]),
            })
    elif name == "piqa":
        for local_idx, row in enumerate(ds):
            examples.append({
                "id": f"{name}:{split}:{indices[local_idx]}",
                "source_index": int(indices[local_idx]),
                "context": f"Question: {row['goal']}\nAnswer:",
                "choices": [row["sol1"], row["sol2"]],
                "label": int(row["label"]),
            })
    else:
        for local_idx, row in enumerate(ds):
            label_map = {str(k): i for i, k in enumerate(row["choices"]["label"])}
            examples.append({
                "id": f"{name}:{split}:{indices[local_idx]}",
                "source_index": int(indices[local_idx]),
                "context": f"Question: {row['question']}\nAnswer:",
                "choices": row["choices"]["text"],
                "label": int(label_map.get(str(row["answerKey"]), 0)),
            })
    return examples


def finite_json(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else value


def rank_choice_scores(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (scores[i], i))


def build_choice_prediction_record(example: dict, scored: list[ScoredCompletion], teacher_record: dict | None = None) -> dict:
    scores = [s.nll_per_token for s in scored]
    ranking = rank_choice_scores(scores)
    pred = int(ranking[0]) if ranking else -1
    label = int(example["label"])
    gold_score = scores[label] if 0 <= label < len(scores) else float("inf")
    wrong = [(i, score) for i, score in enumerate(scores) if i != label]
    best_wrong_idx, best_wrong_score = min(wrong, key=lambda item: (item[1], item[0])) if wrong else (-1, float("inf"))
    margin = best_wrong_score - gold_score if math.isfinite(gold_score) and math.isfinite(best_wrong_score) else None
    record = {
        "id": example.get("id"),
        "source_index": example.get("source_index"),
        "label": label,
        "pred": pred,
        "correct": int(pred == label),
        "ranking": [int(i) for i in ranking],
        "gold_nll_per_token": finite_json(gold_score),
        "best_wrong_index": int(best_wrong_idx),
        "best_wrong_nll_per_token": finite_json(best_wrong_score),
        "margin_best_wrong_minus_gold_nll": finite_json(margin),
        "choice_scores": [s.to_json() for s in scored],
    }
    if teacher_record is not None:
        record["qwen_teacher_pred"] = int(teacher_record["pred"])
        record["qwen_teacher_ranking"] = [int(i) for i in teacher_record["ranking"]]
        record["qwen_top1_agreement"] = int(pred == int(teacher_record["pred"]))
        record["qwen_full_ranking_agreement"] = int(ranking == teacher_record["ranking"])
    return record


def text_to_choice_sample(codec, tokenizer, text: str, context_byte_len: int, readout: str, device: torch.device, max_bytes: int) -> tuple[SequenceSample, torch.Tensor] | None:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) > max_bytes:
        trim = len(raw) - max_bytes
        raw = raw[trim:]
        context_byte_len = max(0, context_byte_len - trim)
    byte_arr = np.frombuffer(raw, dtype=np.uint8).copy()
    byte_arr[byte_arr == 0xFF] = 32
    byte_row = torch.from_numpy(byte_arr.astype(np.int64))
    spans = _token_spans_for_bytes(byte_row, tokenizer)
    if len(spans) < 3:
        return None
    records: list[tuple[int, int, bool]] = []
    if readout == "token_end":
        records = [(end, tid, end >= context_byte_len) for _, end, tid in spans]
    elif readout == "patch_boundary":
        span_idx = 0
        for pos in range(codec.cfg.patch_size - 1, int(byte_row.shape[0]), codec.cfg.patch_size):
            while span_idx < len(spans) and spans[span_idx][1] < pos:
                span_idx += 1
            if span_idx >= len(spans):
                break
            start, end, tid = spans[span_idx]
            if start <= pos <= end:
                records.append((pos, tid, pos >= context_byte_len))
    else:
        raise ValueError(readout)
    if len(records) < 3 or not any(r[2] for r in records[1:]):
        return None
    byte_ids = byte_row.unsqueeze(0).to(device)
    positions = torch.tensor([p for p, _, _ in records], dtype=torch.long, device=device)
    labels = torch.tensor([tid for _, tid, _ in records], dtype=torch.long)
    choice_mask = torch.tensor([is_choice for _, _, is_choice in records], dtype=torch.bool)
    with torch.no_grad():
        hidden = codec.encoder(byte_ids)[0].float().index_select(0, positions)
    return SequenceSample(hidden.cpu(), labels.cpu(), positions.cpu(), readout), choice_mask.cpu()


@torch.no_grad()
def score_completion_token_space(
    model,
    adapter: CalibrationAdapter | None,
    codec,
    tokenizer,
    context: str,
    choice: str,
    teacher,
    device: torch.device,
    readout: str,
    max_bytes: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    input_kind: str = "calibrated",
) -> ScoredCompletion:
    context_bytes = context.encode("utf-8", errors="replace")
    out = text_to_choice_sample(codec, tokenizer, context + choice, len(context_bytes), readout, device, max_bytes)
    if out is None:
        return ScoredCompletion(float("inf"), 0)
    sample, choice_mask = out
    hidden, labels, attention, readout_ids = pad_samples([sample], device)
    dtype = next(model.parameters()).dtype
    inputs = make_inputs(hidden, labels, input_kind, teacher, codec, adapter, 1.0, dtype, transform, readout_ids)
    logits = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False).logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    shifted_choice = choice_mask[1:].to(device).unsqueeze(0) & shift_labels.ne(-100)
    if int(shifted_choice.sum().item()) == 0:
        return ScoredCompletion(float("inf"), 0)
    loss_flat = F.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
    return ScoredCompletion(float(loss_flat[shifted_choice].sum().item()), int(shifted_choice.sum().item()))


@torch.no_grad()
def score_teacher_completion(teacher, tokenizer, context: str, choice: str, device: torch.device) -> ScoredCompletion:
    ctx_ids = tokenizer(context, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = tokenizer(context + choice, add_special_tokens=False, return_tensors="pt").input_ids
    if int(full_ids.numel()) < 2:
        return ScoredCompletion(float("inf"), 0)
    start = min(int(ctx_ids.shape[1]), int(full_ids.shape[1]) - 1)
    input_ids = full_ids.to(device)
    labels = input_ids.clone()
    labels[:, :start] = -100
    logits = teacher(input_ids=input_ids, use_cache=False).logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        return ScoredCompletion(float("inf"), 0)
    loss_flat = F.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
    return ScoredCompletion(float(loss_flat[mask].sum().item()), int(mask.sum().item()))


def evaluate_teacher_rankings(teacher, tokenizer, examples: list[dict], device: torch.device, progress: bool, name: str) -> dict:
    correct = 0
    predictions = []
    started = time.time()
    for i, ex in enumerate(examples):
        scored = [score_teacher_completion(teacher, tokenizer, ex["context"], choice, device) for choice in ex["choices"]]
        record = build_choice_prediction_record(ex, scored)
        correct += int(record["correct"])
        predictions.append(record)
        if progress and (i + 1) % 25 == 0:
            print(f"  [{name}] {i + 1}/{len(examples)} acc={correct/(i+1):.3f}", flush=True)
    summary = summarize_prediction_records(predictions)
    summary["elapsed_s"] = round(time.time() - started, 3)
    summary["predictions"] = predictions
    return summary


def summarize_prediction_records(predictions: list[dict]) -> dict:
    n = len(predictions)
    margins = [float(p["margin_best_wrong_minus_gold_nll"]) for p in predictions if p.get("margin_best_wrong_minus_gold_nll") is not None]
    top1_agreements = [int(p["qwen_top1_agreement"]) for p in predictions if "qwen_top1_agreement" in p]
    full_agreements = [int(p["qwen_full_ranking_agreement"]) for p in predictions if "qwen_full_ranking_agreement" in p]
    return {
        "accuracy": float(np.mean([int(p["correct"]) for p in predictions])) if n else 0.0,
        "n_examples": n,
        "mean_margin_best_wrong_minus_gold_nll": float(np.mean(margins)) if margins else None,
        "median_margin_best_wrong_minus_gold_nll": float(np.median(margins)) if margins else None,
        "positive_margin_fraction": float(np.mean([m > 0.0 for m in margins])) if margins else None,
        "qwen_top1_agreement": float(np.mean(top1_agreements)) if top1_agreements else None,
        "qwen_full_ranking_agreement": float(np.mean(full_agreements)) if full_agreements else None,
    }


def evaluate_benchmark_variant(
    model,
    adapter,
    codec,
    tokenizer,
    teacher,
    examples: list[dict],
    teacher_predictions: list[dict],
    device: torch.device,
    readout: str,
    max_bytes: int,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
    progress: bool,
    name: str,
    input_kind: str = "calibrated",
) -> dict:
    predictions = []
    started = time.time()
    for i, ex in enumerate(examples):
        scored = [
            score_completion_token_space(model, adapter, codec, tokenizer, ex["context"], choice, teacher, device, readout, max_bytes, transform, input_kind)
            for choice in ex["choices"]
        ]
        record = build_choice_prediction_record(ex, scored, teacher_predictions[i])
        predictions.append(record)
        if progress and (i + 1) % 25 == 0:
            acc = float(np.mean([int(p["correct"]) for p in predictions]))
            print(f"  [{name}] {i + 1}/{len(examples)} acc={acc:.3f}", flush=True)
    summary = summarize_prediction_records(predictions)
    summary["elapsed_s"] = round(time.time() - started, 3)
    summary["predictions"] = predictions
    return summary


def bootstrap_scalar_delta(main: list[dict], control: list[dict], key: str, samples: int, seed: int) -> dict:
    n = min(len(main), len(control))
    deltas = []
    for i in range(n):
        a = main[i].get(key)
        b = control[i].get(key)
        if a is not None and b is not None:
            deltas.append(float(a) - float(b))
    if not deltas:
        return {"mean": None, "ci95": [None, None], "n": 0}
    delta = np.asarray(deltas, dtype=np.float64)
    if samples <= 0 or len(delta) == 1:
        mean = float(delta.mean())
        return {"mean": mean, "ci95": [mean, mean], "n": int(len(delta))}
    rng = np.random.default_rng(seed)
    boot = [float(delta[rng.integers(0, len(delta), size=len(delta))].mean()) for _ in range(samples)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(delta.mean()), "ci95": [float(lo), float(hi)], "n": int(len(delta))}


def bootstrap_accuracy_delta(main: list[dict], control: list[dict], samples: int, seed: int) -> dict:
    return bootstrap_scalar_delta(main, control, "correct", samples, seed)


def load_adapter(path: str, device: torch.device) -> tuple[CalibrationAdapter, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload.get("adapter_manifest", {})
    adapter = CalibrationAdapter(
        kind=manifest.get("kind", "rms_linear"),
        rank=int(manifest.get("rank", 256)),
        conditioning=manifest.get("conditioning", "shared"),
        readouts=tuple(manifest.get("readouts", READOUT_NAMES)),
    )
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.to(device).eval()
    return adapter, manifest


def functional_margin_shadow_verdict(results: dict) -> dict:
    threshold = 0.01
    benchmark_rows = {}
    pass_count = 0
    for bench, bench_results in results.items():
        main = bench_results.get("main_inherited")
        random_control = bench_results.get("random_core")
        gaussian_control = bench_results.get("gaussian_destroyed_input")
        if main is None or random_control is None or gaussian_control is None:
            continue
        delta_random = float(main["accuracy"] - random_control["accuracy"])
        delta_gaussian = float(main["accuracy"] - gaussian_control["accuracy"])
        passed = delta_random >= threshold and delta_gaussian >= threshold
        pass_count += int(passed)
        benchmark_rows[bench] = {
            "main_accuracy": main["accuracy"],
            "random_accuracy": random_control["accuracy"],
            "gaussian_destroyed_accuracy": gaussian_control["accuracy"],
            "delta_main_minus_random_accuracy": delta_random,
            "delta_main_minus_gaussian_destroyed_accuracy": delta_gaussian,
            "passes_plus_1pp_over_both_controls": passed,
        }
    n_benchmarks = len(benchmark_rows)
    if pass_count >= 2:
        verdict = "PASS_FUNCTIONAL_MARGIN_SHADOW"
        story = "candidate_discriminative_signal_present"
    elif pass_count == 0 and n_benchmarks > 0:
        verdict = "FAIL_FUNCTIONAL_MARGIN_SHADOW"
        story = "SURFACE_COMPATIBILITY_ONLY"
    else:
        verdict = "MARGINAL_FUNCTIONAL_MARGIN"
        story = "ambiguous_candidate_discriminative_signal"
    return {
        "threshold_accuracy_delta": threshold,
        "required_pass_benchmarks": 2,
        "passed_benchmarks": pass_count,
        "n_benchmarks": n_benchmarks,
        "benchmarks": benchmark_rows,
        "verdict": verdict,
        "causal_story": story,
    }


def strip_predictions(value):
    if isinstance(value, dict):
        return {k: strip_predictions(v) for k, v in value.items() if k != "predictions"}
    if isinstance(value, list):
        return [strip_predictions(v) for v in value]
    return value


def run_benchmark(args: argparse.Namespace) -> dict:
    started = time.time()
    set_seed(args.seed)
    ensure_offline(args.allow_downloads)
    device = choose_device(args.device)
    if not args.adapter_checkpoint:
        raise ValueError("--adapter-checkpoint is required for benchmark mode")
    if args.functional_margin_shadow and args.benchmark_split != "train":
        args.benchmark_split = "train"
    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)
    adapter, adapter_manifest = load_adapter(args.adapter_checkpoint, device)
    copied_model, copied_manifest = build_qwen_variant(teacher, args.layers, "copied", device, args.seed)
    random_model, random_manifest = build_qwen_variant(teacher, args.layers, "random", device, args.seed)
    shuffled_model, shuffled_manifest = build_qwen_variant(teacher, args.layers, "shuffled", device, args.seed + 23)
    generic_model, generic_manifest = build_qwen_variant(teacher, args.layers, "generic_pretrained", device, args.seed, source_layer_start=args.generic_layer_start)
    rot = random_orthogonal(int(teacher.config.hidden_size), device, args.seed + 303)

    def rotate_with_inverse(x: torch.Tensor) -> torch.Tensor:
        return (x @ rot) @ rot.t()

    def gaussian_norm_noise(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        noise = torch.randn_like(x_float)
        noise_norm = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x_norm = x_float.norm(dim=-1, keepdim=True)
        return noise * (x_norm / noise_norm)

    variants = {
        "main_inherited": {"model": copied_model, "transform": None, "input_kind": "calibrated", "description": "inherited copied core"},
        "random_core": {"model": random_model, "transform": None, "input_kind": "calibrated", "description": "adapter plus random core"},
        "shuffled_core": {"model": shuffled_model, "transform": None, "input_kind": "calibrated", "description": "adapter plus shuffled copied core"},
        "generic_pretrained_core": {"model": generic_model, "transform": None, "input_kind": "calibrated", "description": "adapter plus Qwen middle-layer core"},
        "gaussian_destroyed_input": {"model": copied_model, "transform": gaussian_norm_noise, "input_kind": "calibrated", "description": "adapter plus same-norm Gaussian destroyed input"},
        "inverse_recovered_rotation": {"model": copied_model, "transform": rotate_with_inverse, "input_kind": "calibrated", "description": "adapter plus rotate-then-inverse recovery"},
        "true_embedding_truncated_qwen": {"model": copied_model, "transform": None, "input_kind": "true_embedding", "description": "true-embedding truncated Qwen upper bound"},
    }
    all_results: dict[str, dict] = {}
    for bench in args.benchmarks:
        examples = load_limited_benchmark(bench, args.benchmark_examples, args.benchmark_split, args.seed + len(bench), args.allow_downloads)
        all_results[bench] = {
            "metadata": {
                "split": args.benchmark_split,
                "train_safe": args.benchmark_split == "train",
                "n_examples": len(examples),
                "readout": args.benchmark_readout,
            }
        }
        teacher_result = evaluate_teacher_rankings(teacher, tokenizer, examples, device, args.progress, f"{bench}:qwen_teacher_full")
        all_results[bench]["qwen_teacher_full"] = teacher_result
        teacher_predictions = teacher_result["predictions"]
        for variant_name, spec in variants.items():
            all_results[bench][variant_name] = evaluate_benchmark_variant(
                spec["model"],
                adapter,
                codec,
                tokenizer,
                teacher,
                examples,
                teacher_predictions,
                device,
                args.benchmark_readout,
                args.benchmark_max_bytes,
                spec["transform"],
                args.progress,
                f"{bench}:{variant_name}",
                spec["input_kind"],
            )
        main_preds = all_results[bench]["main_inherited"]["predictions"]
        for control in [name for name in variants if name != "main_inherited"]:
            control_preds = all_results[bench][control]["predictions"]
            all_results[bench][f"delta_main_minus_{control}"] = {
                "accuracy": bootstrap_accuracy_delta(main_preds, control_preds, args.bootstrap_samples, args.seed + 900),
                "margin_best_wrong_minus_gold_nll": bootstrap_scalar_delta(main_preds, control_preds, "margin_best_wrong_minus_gold_nll", args.bootstrap_samples, args.seed + 901),
            }

    shadow_verdict = functional_margin_shadow_verdict(all_results) if args.functional_margin_shadow else None
    summary_benchmarks = {bench: strip_predictions(result) for bench, result in all_results.items()}
    payload = {
        "mode": "functional_margin_shadow" if args.functional_margin_shadow else "benchmark",
        "run": {
            "seed": args.seed,
            "device": str(device),
            "teacher": args.teacher,
            "layers": args.layers,
            "generic_layer_start": args.generic_layer_start,
            "benchmark_readout": args.benchmark_readout,
            "benchmark_examples": args.benchmark_examples,
            "benchmark_split": args.benchmark_split,
            "functional_margin_shadow": bool(args.functional_margin_shadow),
            "elapsed_s": round(time.time() - started, 3),
        },
        "codec": codec_manifest,
        "adapter": adapter_manifest,
        "copy_manifests": {"copied": copied_manifest, "random": random_manifest, "shuffled": shuffled_manifest, "generic_pretrained": generic_manifest},
        "variant_descriptions": {name: spec["description"] for name, spec in variants.items()},
        "wide7_reference": WIDE7_BASELINE,
        "precommitted_verdict_tokens": {
            "pass": "PASS_FUNCTIONAL_MARGIN_SHADOW",
            "fail": "FAIL_FUNCTIONAL_MARGIN_SHADOW",
            "marginal": "MARGINAL_FUNCTIONAL_MARGIN",
        },
        "functional_margin_shadow": shadow_verdict,
        "benchmarks": summary_benchmarks,
        "benchmark_details": all_results if (args.functional_margin_shadow or args.save_benchmark_predictions) else {},
        "limitations": [
            "Benchmark mode is token-space candidate scoring through byte-derived codec states.",
            "Functional-margin shadow uses train split examples only when --functional-margin-shadow is active.",
            "Qwen preference agreement uses full Qwen/Qwen3-0.6B continuation scoring as the teacher ranking.",
            "It does not implement a byte decoder or byte BPB.",
            "Generic-pretrained control uses a different Qwen layer range, not a separately trained non-Qwen architecture.",
            "Tokenized sibling controls are not implemented in this prototype.",
        ],
    }
    if not args.no_artifacts:
        out_dir = Path(args.output_dir)
        write_json(out_dir / ("functional_margin_shadow.json" if args.functional_margin_shadow else "benchmark_metrics.json"), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "benchmark"], default="preflight")
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adapter-checkpoint", default="")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--generic-layer-start", type=int, default=14)
    parser.add_argument("--depth-curve-layers", nargs="*", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--num-sequences", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-positions-per-sequence", type=int, default=128)
    parser.add_argument("--eval-readouts", nargs="+", choices=["token_end", "patch_boundary"], default=["token_end", "patch_boundary"])
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--adapter-kind", choices=["linear", "rms_linear", "low_rank"], default="rms_linear")
    parser.add_argument("--adapter-conditioning", choices=["shared", "readout"], default="readout")
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--adapter-steps", type=int, default=1200)
    parser.add_argument("--adapter-batch-anchors", type=int, default=4096)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-core-steps", type=int, default=0)
    parser.add_argument("--finetune-batch-sequences", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=1e-5)
    parser.add_argument("--scale", default="auto")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--benchmarks", nargs="+", choices=["hellaswag", "piqa", "arc_easy", "arc_challenge"], default=["hellaswag", "piqa", "arc_easy", "arc_challenge"])
    parser.add_argument("--benchmark-examples", type=int, default=256)
    parser.add_argument("--benchmark-split", choices=["train", "validation"], default="validation")
    parser.add_argument("--benchmark-readout", choices=["token_end", "patch_boundary"], default="token_end")
    parser.add_argument("--benchmark-max-bytes", type=int, default=1536)
    parser.add_argument("--functional-margin-shadow", action="store_true")
    parser.add_argument("--save-benchmark-predictions", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_preflight(args) if args.mode == "preflight" else run_benchmark(args)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.mode == "preflight":
        print(json.dumps({"verdict": payload["verdict"], "stage1_pass": payload["stage1_pass"], "stage1_gates": payload["stage1_gates"]}, indent=2, sort_keys=True))
    else:
        print(json.dumps({"mode": payload["mode"], "functional_margin_shadow": payload.get("functional_margin_shadow"), "benchmarks": payload["benchmarks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()




