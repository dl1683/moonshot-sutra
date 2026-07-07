"""Minimal byte-native chain-init via codec prototype.

The prototype treats the frozen semantic codec as a byte-to-Qwen-embedding adapter.
It then compares a small Qwen-shaped reasoner initialized from Qwen3-0.6B layers
against the same reasoner with random layers, using the same frozen Qwen output
head. The metric is next-token cross-entropy on byte-derived sequences.

This is not a trained Sutra model and not a byte decoder. It is a compatibility
probe for direct weight transfer through the codec interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from tier3_brainseed_chart_probe import ByteShardSampler, _token_spans_for_bytes, load_codec, load_tokenizer


DEFAULT_CODEC = "C:/sutra_fast/codec_phase1.5/codec_final.pt"
DEFAULT_DATA_DIR = "C:/sutra_fast/data/shards_diverse"
DEFAULT_OUTPUT_DIR = "C:/sutra_fast/chain_init_probe"


def ensure_offline(allow_downloads: bool) -> None:
    if allow_downloads:
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_teacher(name: str, device: torch.device, allow_downloads: bool):
    ensure_offline(allow_downloads)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, local_files_only=not allow_downloads)
    model.to(device)
    model.eval()
    return model


def make_small_qwen_pair(teacher, layers: int, device: torch.device):
    cfg = copy.deepcopy(teacher.config)
    cfg.num_hidden_layers = layers
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    chain = AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype).eval()
    random_model = AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype).eval()

    teacher_sd = teacher.state_dict()
    chain_sd = chain.state_dict()
    copied = []
    for key, value in list(chain_sd.items()):
        if key in teacher_sd and tuple(teacher_sd[key].shape) == tuple(value.shape):
            chain_sd[key] = teacher_sd[key].detach().to(device=value.device, dtype=value.dtype)
            copied.append(key)
    chain.load_state_dict(chain_sd)

    random_sd = random_model.state_dict()
    copied_head = []
    for key, value in list(random_sd.items()):
        if key.startswith("model.embed_tokens") or key.startswith("model.norm") or key.startswith("lm_head"):
            if key in teacher_sd and tuple(teacher_sd[key].shape) == tuple(value.shape):
                random_sd[key] = teacher_sd[key].detach().to(device=value.device, dtype=value.dtype)
                copied_head.append(key)
    random_model.load_state_dict(random_sd)
    return chain, random_model, {"chain_copied_tensors": len(copied), "random_copied_head_tensors": len(copied_head)}


def codec_sequence_for_row(codec, tokenizer, byte_row: torch.Tensor, readout: str, device: torch.device, max_positions: int) -> tuple[torch.Tensor, torch.Tensor] | None:
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
        records = records[:max_positions]

    byte_ids = byte_row.unsqueeze(0).to(device)
    positions = torch.tensor([p for p, _ in records], dtype=torch.long, device=device)
    labels = torch.tensor([tid for _, tid in records], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = codec.encoder(byte_ids)[0].float().index_select(0, positions)
        embeds = codec.alignment_head(hidden).float()
    return embeds.cpu(), labels.cpu()


def collect_samples(codec, tokenizer, args: argparse.Namespace, device: torch.device) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    sampler = ByteShardSampler(args.data_dir, seq_len=args.seq_len)
    samples = []
    attempts = 0
    while len(samples) < args.num_sequences and attempts < args.num_sequences * 10:
        attempts += 1
        rows = sampler.sample(1, rng)
        out = codec_sequence_for_row(codec, tokenizer, rows[0], args.readout, device, args.max_positions_per_sequence)
        if out is None:
            continue
        embeds, labels = out
        samples.append({"codec_embeds": embeds, "labels": labels})
    if len(samples) < args.num_sequences:
        raise RuntimeError(f"only collected {len(samples)} usable samples after {attempts} attempts")
    return samples


def pad_batch(samples: list[dict], start: int, batch_size: int, device: torch.device, dtype: torch.dtype):
    chunk = samples[start : start + batch_size]
    max_len = max(int(s["labels"].shape[0]) for s in chunk)
    embeds = torch.zeros((len(chunk), max_len, chunk[0]["codec_embeds"].shape[-1]), dtype=dtype, device=device)
    labels = torch.full((len(chunk), max_len), -100, dtype=torch.long, device=device)
    attention = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
    for i, sample in enumerate(chunk):
        n = int(sample["labels"].shape[0])
        embeds[i, :n] = sample["codec_embeds"].to(device=device, dtype=dtype)
        labels[i, :n] = sample["labels"].to(device=device)
        attention[i, :n] = 1
    return embeds, labels, attention


def shifted_ce_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)
    if int(mask.sum().item()) == 0:
        return logits.new_tensor(0.0), 0, 0
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100, reduction="sum")
    pred = shift_logits.argmax(dim=-1)
    correct = int((pred[mask] == shift_labels[mask]).sum().item())
    total = int(mask.sum().item())
    return loss, correct, total


def evaluate_reasoner(model, samples: list[dict], teacher, scale: float, batch_size: int, device: torch.device, use_teacher_embeds: bool = False) -> dict:
    dtype = next(model.parameters()).dtype
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            codec_embeds, labels, attention = pad_batch(samples, start, batch_size, device, dtype)
            if use_teacher_embeds:
                input_labels = labels.clamp_min(0)
                inputs = teacher.model.embed_tokens(input_labels).to(dtype)
                inputs = inputs.masked_fill(labels.eq(-100).unsqueeze(-1), 0.0)
            else:
                inputs = codec_embeds * scale
            out = model(inputs_embeds=inputs, attention_mask=attention, use_cache=False)
            loss, correct, total = shifted_ce_from_logits(out.logits, labels)
            total_loss += float(loss.item())
            total_tokens += total
            total_correct += correct
    nll = total_loss / max(1, total_tokens)
    return {
        "nll": nll,
        "ppl": float(math.exp(min(20.0, nll))),
        "next_token_acc": total_correct / max(1, total_tokens),
        "tokens": total_tokens,
    }


def evaluate_identity(samples: list[dict], teacher, scale: float, batch_size: int, device: torch.device) -> dict:
    dtype = next(teacher.parameters()).dtype
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            codec_embeds, labels, _ = pad_batch(samples, start, batch_size, device, dtype)
            hidden = teacher.model.norm(codec_embeds * scale)
            logits = teacher.lm_head(hidden)
            loss, correct, total = shifted_ce_from_logits(logits, labels)
            total_loss += float(loss.item())
            total_tokens += total
            total_correct += correct
    nll = total_loss / max(1, total_tokens)
    return {
        "nll": nll,
        "ppl": float(math.exp(min(20.0, nll))),
        "next_token_acc": total_correct / max(1, total_tokens),
        "tokens": total_tokens,
    }


def embedding_mean_norm(teacher) -> float:
    weight = teacher.model.embed_tokens.weight.detach().float()
    return float(weight.norm(dim=-1).mean().item())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--readout", choices=["token_end", "patch_boundary"], default="token_end")
    parser.add_argument("--num-sequences", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-positions-per-sequence", type=int, default=128)
    parser.add_argument("--scale", default="auto", help="auto uses Qwen embedding mean norm; otherwise float scale")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_offline(args.allow_downloads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    started = time.time()

    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)
    teacher = load_teacher(args.teacher, device, args.allow_downloads)
    scale = embedding_mean_norm(teacher) if args.scale == "auto" else float(args.scale)

    samples = collect_samples(codec, tokenizer, args, device)
    chain, random_model, copy_manifest = make_small_qwen_pair(teacher, args.layers, device)

    results = {
        "identity_codec_to_lm_head": evaluate_identity(samples, teacher, scale, args.batch_size, device),
        "random_layers_codec_input": evaluate_reasoner(random_model, samples, teacher, scale, args.batch_size, device),
        "chain_initialized_layers_codec_input": evaluate_reasoner(chain, samples, teacher, scale, args.batch_size, device),
        "chain_initialized_layers_teacher_embedding_input": evaluate_reasoner(chain, samples, teacher, scale, args.batch_size, device, use_teacher_embeds=True),
    }
    chain_nll = results["chain_initialized_layers_codec_input"]["nll"]
    random_nll = results["random_layers_codec_input"]["nll"]
    results["precommitted_check"] = {
        "chain_nll_lt_random_nll": chain_nll < random_nll,
        "nll_delta_chain_minus_random": chain_nll - random_nll,
    }

    cfg = teacher.config
    payload = {
        "run": {
            "seed": args.seed,
            "device": str(device),
            "layers": args.layers,
            "readout": args.readout,
            "num_sequences": args.num_sequences,
            "seq_len": args.seq_len,
            "scale": scale,
            "elapsed_s": round(time.time() - started, 3),
        },
        "qwen_config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": cfg.intermediate_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_attention_heads": cfg.num_attention_heads,
            "num_key_value_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size,
            "tie_word_embeddings": cfg.tie_word_embeddings,
        },
        "codec": codec_manifest,
        "copy_manifest": copy_manifest,
        "sample_stats": {
            "n": len(samples),
            "min_len": min(int(s["labels"].shape[0]) for s in samples),
            "max_len": max(int(s["labels"].shape[0]) for s in samples),
            "mean_len": float(np.mean([int(s["labels"].shape[0]) for s in samples])),
        },
        "results": results,
        "limitations": [
            "token_end readout uses oracle Qwen tokenizer boundaries for the prototype",
            "metric is token-space next-token loss through Qwen lm_head, not byte decoder perplexity",
            "copied first-N Qwen layers are truncated, so absolute perplexity is not a full-model quality claim",
        ],
    }
    if not args.no_artifacts:
        write_json(Path(args.output_dir) / f"chain_init_{args.readout}_layers{args.layers}.json", payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()