"""Batch 5 diagnostics for Brainseed final adjudication.

This script runs the diagnostics requested after Batch 4:

1. zero-cost/readout-only patch chart baselines;
2. offset/length/frequency slices for the Phase 1.5 patch chart;
3. stronger frozen scorer variants over the same codec pair features.

It deliberately does not train or mutate the codec.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_codec import SemanticCodec
from tier3_brainseed_chart_probe import (
    ByteShardSampler,
    codec_only_accuracy,
    find_anchor_sets,
    grouped_accuracy,
    load_codec,
    load_examples,
    load_teacher_embeddings,
    load_teacher_model,
    load_tokenizer,
    pooled_codec_text_feature,
    set_seed,
    teacher_completion_score,
    topk_retrieval,
    _token_spans_for_bytes,
)


DEFAULT_CODEC = "C:/sutra_fast/codec_phase1.5/codec_final.pt"
DEFAULT_DATA_DIR = "C:/sutra_fast/data/shards_diverse"
DEFAULT_TEACHER_EMB = "C:/sutra_fast/teacher_embeddings.pt"
DEFAULT_OUTPUT_DIR = "C:/sutra_fast/brainseed_batch5"


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


@dataclass
class PatchBatch:
    current_query: torch.Tensor
    prev_token_query: torch.Tensor
    prev_token_ids: torch.Tensor
    prev_token_mask: torch.Tensor
    mean4_query: torch.Tensor
    mean8_query: torch.Tensor
    max4_query: torch.Tensor
    max8_query: torch.Tensor
    current_hidden: torch.Tensor
    token_ids: torch.Tensor
    offsets: torch.Tensor
    reverse_offsets: torch.Tensor
    token_lengths: torch.Tensor
    token_starts: torch.Tensor
    token_ends: torch.Tensor
    patch_positions: torch.Tensor
    token_id_frequency: dict[int, int]
    collection: dict


def _project_hidden(codec: SemanticCodec, hidden_rows: torch.Tensor) -> torch.Tensor:
    if hidden_rows.numel() == 0:
        return torch.empty((0, codec.cfg.teacher_dim), dtype=torch.float32, device=hidden_rows.device)
    return codec.alignment_head(hidden_rows).float()


def _patch_records_for_row(
    byte_row: torch.Tensor,
    tokenizer,
    patch_size: int,
) -> tuple[list[dict], dict[int, int]]:
    spans = _token_spans_for_bytes(byte_row, tokenizer)
    freq: dict[int, int] = {}
    for _, _, tid in spans:
        freq[tid] = freq.get(tid, 0) + 1

    records: list[dict] = []
    span_idx = 0
    for pos in range(patch_size - 1, int(byte_row.shape[0]), patch_size):
        while span_idx < len(spans) and spans[span_idx][1] < pos:
            span_idx += 1
        if span_idx >= len(spans):
            break
        start, end, tid = spans[span_idx]
        if not (start <= pos <= end):
            continue
        prev_idx = span_idx if end <= pos else span_idx - 1
        prev_pos = spans[prev_idx][1] if prev_idx >= 0 else -1
        prev_tid = spans[prev_idx][2] if prev_idx >= 0 else -1
        records.append(
            {
                "patch_pos": int(pos),
                "target_tid": int(tid),
                "start": int(start),
                "end": int(end),
                "offset": int(pos - start),
                "reverse_offset": int(end - pos),
                "length": int(end - start + 1),
                "prev_pos": int(prev_pos),
                "prev_tid": int(prev_tid),
            }
        )
    return records, freq


def collect_patch_batch(
    codec: SemanticCodec,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> PatchBatch:
    sampler = ByteShardSampler(args.data_dir, seq_len=args.seq_len)
    current_q: list[torch.Tensor] = []
    prev_q: list[torch.Tensor] = []
    prev_ids: list[int] = []
    prev_mask: list[bool] = []
    mean4_q: list[torch.Tensor] = []
    mean8_q: list[torch.Tensor] = []
    max4_q: list[torch.Tensor] = []
    max8_q: list[torch.Tensor] = []
    current_h: list[torch.Tensor] = []
    token_ids: list[int] = []
    offsets: list[int] = []
    reverse_offsets: list[int] = []
    lengths: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    positions: list[int] = []
    freq: dict[int, int] = {}
    started = time.time()

    n_done = 0
    while n_done < args.num_sequences:
        bs = min(args.batch_size, args.num_sequences - n_done)
        byte_ids_cpu = sampler.sample(bs, rng)
        byte_ids = byte_ids_cpu.to(device)
        with torch.no_grad():
            hidden = codec.encoder(byte_ids).float()

            for b in range(bs):
                records, row_freq = _patch_records_for_row(byte_ids_cpu[b], tokenizer, codec.cfg.patch_size)
                for tid, count in row_freq.items():
                    freq[tid] = freq.get(tid, 0) + count
                for rec in records:
                    pos = rec["patch_pos"]
                    h_cur = hidden[b, pos]
                    current_h.append(h_cur.cpu())
                    current_q.append(_project_hidden(codec, h_cur.unsqueeze(0)).squeeze(0).cpu())

                    if rec["prev_pos"] >= 0:
                        h_prev = hidden[b, rec["prev_pos"]]
                        prev_q.append(_project_hidden(codec, h_prev.unsqueeze(0)).squeeze(0).cpu())
                        prev_ids.append(rec["prev_tid"])
                        prev_mask.append(True)
                    else:
                        prev_q.append(torch.zeros(codec.cfg.teacher_dim, dtype=torch.float32))
                        prev_ids.append(-1)
                        prev_mask.append(False)

                    for window, mean_store, max_store in (
                        (4, mean4_q, max4_q),
                        (8, mean8_q, max8_q),
                    ):
                        lo = max(0, pos - window + 1)
                        local = hidden[b, lo : pos + 1]
                        mean_store.append(_project_hidden(codec, local.mean(dim=0, keepdim=True)).squeeze(0).cpu())
                        max_store.append(_project_hidden(codec, local.max(dim=0).values.unsqueeze(0)).squeeze(0).cpu())

                    token_ids.append(rec["target_tid"])
                    offsets.append(rec["offset"])
                    reverse_offsets.append(rec["reverse_offset"])
                    lengths.append(rec["length"])
                    starts.append(rec["start"])
                    ends.append(rec["end"])
                    positions.append(pos)
        n_done += bs

    if not token_ids:
        raise RuntimeError("No patch records collected")

    collection = {
        "data_dir": args.data_dir,
        "num_sequences": args.num_sequences,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "patch_records": len(token_ids),
        "total_shard_bytes": sampler.total_bytes,
        "elapsed_s": round(time.time() - started, 3),
    }
    return PatchBatch(
        current_query=torch.stack(current_q),
        prev_token_query=torch.stack(prev_q),
        prev_token_ids=torch.tensor(prev_ids, dtype=torch.long),
        prev_token_mask=torch.tensor(prev_mask, dtype=torch.bool),
        mean4_query=torch.stack(mean4_q),
        mean8_query=torch.stack(mean8_q),
        max4_query=torch.stack(max4_q),
        max8_query=torch.stack(max8_q),
        current_hidden=torch.stack(current_h),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        offsets=torch.tensor(offsets, dtype=torch.long),
        reverse_offsets=torch.tensor(reverse_offsets, dtype=torch.long),
        token_lengths=torch.tensor(lengths, dtype=torch.long),
        token_starts=torch.tensor(starts, dtype=torch.long),
        token_ends=torch.tensor(ends, dtype=torch.long),
        patch_positions=torch.tensor(positions, dtype=torch.long),
        token_id_frequency=freq,
        collection=collection,
    )


def retrieval_for_query(
    name: str,
    queries: torch.Tensor,
    ids: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    device: torch.device,
) -> dict:
    out = topk_retrieval(queries, ids, ids, teacher_embeddings, device)
    out["name"] = name
    return out


def fit_concat_linear_readout(
    batch: PatchBatch,
    teacher_embeddings: torch.Tensor,
    device: torch.device,
    seed: int,
    ridge: float,
    train_fraction: float,
    train_cap: int,
) -> dict:
    mask = batch.prev_token_mask
    idx_all = torch.nonzero(mask, as_tuple=False).flatten()
    if idx_all.numel() < 16:
        return {"skipped": True, "reason": "not enough previous-token anchors"}
    g = torch.Generator()
    g.manual_seed(seed)
    idx_all = idx_all[torch.randperm(idx_all.numel(), generator=g)]
    n_train = max(8, int(round(idx_all.numel() * train_fraction)))
    train_idx = idx_all[:n_train]
    eval_idx = idx_all[n_train:]
    if eval_idx.numel() == 0:
        eval_idx = train_idx
    if train_cap > 0 and train_idx.numel() > train_cap:
        train_idx = train_idx[:train_cap]

    x_prev = batch.prev_token_query
    x_cur = batch.current_hidden
    x = torch.cat([x_prev, x_cur], dim=1).float()
    y = teacher_embeddings.index_select(0, batch.token_ids.to(device)).cpu().float()

    x_train = x.index_select(0, train_idx)
    y_train = y.index_select(0, train_idx)
    x_eval = x.index_select(0, eval_idx)
    ids_eval = batch.token_ids.index_select(0, eval_idx)

    ones_train = torch.ones((x_train.shape[0], 1), dtype=x_train.dtype)
    x_aug = torch.cat([x_train, ones_train], dim=1)
    eye = torch.eye(x_aug.shape[1], dtype=x_aug.dtype)
    eye[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + ridge * eye
    rhs = x_aug.T @ y_train
    weights = torch.linalg.solve(lhs, rhs)

    ones_eval = torch.ones((x_eval.shape[0], 1), dtype=x_eval.dtype)
    pred = torch.cat([x_eval, ones_eval], dim=1) @ weights
    pred = F.normalize(pred, dim=-1)
    metrics = topk_retrieval(pred, ids_eval, ids_eval, teacher_embeddings, device)
    return {
        "skipped": False,
        "train_n": int(train_idx.numel()),
        "eval_n": int(eval_idx.numel()),
        "input_dim": int(x.shape[1]),
        "ridge": ridge,
        "metrics": metrics,
    }


def run_zero_cost(args: argparse.Namespace, batch: PatchBatch, teacher_embeddings: torch.Tensor, device: torch.device) -> dict:
    ids = batch.token_ids
    results = {
        "current_patch_state": retrieval_for_query("current_patch_state", batch.current_query, ids, teacher_embeddings, device),
        "nearest_preceding_token_end_against_patch_target": topk_retrieval(
            batch.prev_token_query[batch.prev_token_mask],
            batch.token_ids[batch.prev_token_mask],
            batch.token_ids[batch.prev_token_mask],
            teacher_embeddings,
            device,
        ),
        "nearest_preceding_token_end_own_token_control": topk_retrieval(
            batch.prev_token_query[batch.prev_token_mask],
            batch.prev_token_ids[batch.prev_token_mask],
            batch.prev_token_ids[batch.prev_token_mask],
            teacher_embeddings,
            device,
        ),
        "local_mean_last4": retrieval_for_query("local_mean_last4", batch.mean4_query, ids, teacher_embeddings, device),
        "local_mean_last8": retrieval_for_query("local_mean_last8", batch.mean8_query, ids, teacher_embeddings, device),
        "local_max_last4": retrieval_for_query("local_max_last4", batch.max4_query, ids, teacher_embeddings, device),
        "local_max_last8": retrieval_for_query("local_max_last8", batch.max8_query, ids, teacher_embeddings, device),
        "prev_token_end_plus_current_hidden_linear": fit_concat_linear_readout(
            batch,
            teacher_embeddings,
            device,
            seed=args.seed + 500,
            ridge=args.linear_ridge,
            train_fraction=args.linear_train_fraction,
            train_cap=args.linear_train_cap,
        ),
    }
    patch_target_methods = {
        "current_patch_state",
        "nearest_preceding_token_end_against_patch_target",
        "local_mean_last4",
        "local_mean_last8",
        "local_max_last4",
        "local_max_last8",
        "prev_token_end_plus_current_hidden_linear",
    }
    top1s = []
    for name, block in results.items():
        if name not in patch_target_methods:
            continue
        metrics = block.get("metrics", block)
        if metrics.get("top1") is not None:
            top1s.append((name, float(metrics["top1"])))
    best_name, best_top1 = max(top1s, key=lambda x: x[1])
    return {
        "methods": results,
        "best_method": best_name,
        "best_top1": best_top1,
        "precommitted_check": {
            "any_zero_cost_patch_top1_ge_40": best_top1 >= 0.40,
            "threshold": 0.40,
        },
    }


def _slice_metric(
    name: str,
    mask: torch.Tensor,
    queries: torch.Tensor,
    ids: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    device: torch.device,
) -> dict:
    mask = mask.bool()
    n = int(mask.sum().item())
    if n == 0:
        return {"name": name, "n": 0, "top1": None, "top5": None, "top10": None}
    metrics = topk_retrieval(queries[mask], ids[mask], ids[mask], teacher_embeddings, device)
    metrics["name"] = name
    return metrics


def run_offset_slices(batch: PatchBatch, teacher_embeddings: torch.Tensor, device: torch.device) -> dict:
    q = batch.current_query
    ids = batch.token_ids
    slices: dict[str, dict] = {}

    offset_masks = {
        "offset_0_first_byte": batch.offsets == 0,
        "offset_1": batch.offsets == 1,
        "offset_2": batch.offsets == 2,
        "offset_3plus": batch.offsets >= 3,
        "last_byte": batch.reverse_offsets == 0,
        "second_to_last": batch.reverse_offsets == 1,
    }
    length_masks = {
        "len_1": batch.token_lengths == 1,
        "len_2": batch.token_lengths == 2,
        "len_3": batch.token_lengths == 3,
        "len_4": batch.token_lengths == 4,
        "len_5plus": batch.token_lengths >= 5,
    }
    # Qwen token IDs are only a frequency proxy, not a calibrated corpus rank.
    frequency_proxy_masks = {
        "token_id_lt_1000_proxy": batch.token_ids < 1000,
        "token_id_1000_to_9999_proxy": (batch.token_ids >= 1000) & (batch.token_ids < 10000),
        "token_id_ge_10000_proxy": batch.token_ids >= 10000,
    }

    for group_name, masks in (
        ("token_offset", offset_masks),
        ("token_length", length_masks),
        ("token_frequency_proxy", frequency_proxy_masks),
    ):
        slices[group_name] = {
            name: _slice_metric(name, mask, q, ids, teacher_embeddings, device)
            for name, mask in masks.items()
        }

    early = slices["token_offset"].get("offset_0_first_byte", {}).get("top1")
    offset1 = slices["token_offset"].get("offset_1", {}).get("top1")
    early_values = [x for x in (early, offset1) if x is not None]
    early_mean = float(sum(early_values) / len(early_values)) if early_values else None
    if early_mean is None:
        verdict = "OFFSET_SLICE_INCONCLUSIVE_NO_EARLY_POSITIONS"
    elif early_mean < 0.15:
        verdict = "EARLY_TOKEN_POSITIONS_FAIL_HARD_PART"
    elif early_mean > 0.25:
        verdict = "EARLY_TOKEN_POSITIONS_SHOW_MIDTOKEN_LEARNING"
    else:
        verdict = "EARLY_TOKEN_POSITIONS_AMBIGUOUS"

    return {
        "slices": slices,
        "precommitted_early_offset_top1_mean": early_mean,
        "precommitted_verdict": verdict,
        "frequency_note": "frequency slices use token-id rank as a proxy because no tokenizer frequency table is available in the repo",
    }


def standardize(y: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    mean = float(y.mean().item())
    std = float(y.std(unbiased=False).clamp_min(1e-6).item())
    return (y - mean) / std, mean, std


class MLPScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class BilinearRankScorer(nn.Module):
    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.u = nn.Parameter(torch.empty(dim, rank))
        self.v = nn.Parameter(torch.empty(dim, rank))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def forward(self, pair_x: torch.Tensor) -> torch.Tensor:
        d = pair_x.shape[1] // 4
        ctx = pair_x[:, :d]
        cand = pair_x[:, d : 2 * d]
        return ((ctx @ self.u) * (cand @ self.v)).sum(dim=1) / math.sqrt(self.u.shape[1]) + self.bias


class WeightedCosineScorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(dim))
        self.scale = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, pair_x: torch.Tensor) -> torch.Tensor:
        d = pair_x.shape[1] // 4
        ctx = pair_x[:, :d]
        cand = pair_x[:, d : 2 * d]
        weights = F.softplus(self.log_weights)
        return self.scale * (weights * ctx * cand).sum(dim=1) / weights.sum().clamp_min(1e-6) + self.bias


def train_regressor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    seed: int,
    batch_size: int = 256,
) -> dict:
    torch.manual_seed(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = x.shape[0]
    losses = []
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pred = model(x.index_select(0, idx))
            loss = F.mse_loss(pred, y.index_select(0, idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item()) * idx.numel()
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 4) == 0:
            losses.append({"epoch": epoch, "mse": epoch_loss / n})
    return {"losses": losses}


def pair_features_for_examples(
    codec: SemanticCodec,
    tokenizer,
    examples: list[dict],
    max_bytes: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    rows: list[torch.Tensor] = []
    group_sizes: list[int] = []
    cache: dict[str, torch.Tensor] = {}
    for ex in examples:
        context = ex["context"]
        if context not in cache:
            cache[context] = pooled_codec_text_feature(codec, tokenizer, context, max_bytes, device)
        ctx = cache[context]
        group_sizes.append(len(ex["choices"]))
        for choice in ex["choices"]:
            key = "choice:" + choice
            if key not in cache:
                cache[key] = pooled_codec_text_feature(codec, tokenizer, choice, max_bytes, device)
            cand = cache[key]
            rows.append(torch.cat([ctx, cand, ctx * cand, torch.abs(ctx - cand)], dim=0))
    return torch.stack(rows), group_sizes


def _hellaswag_preprocess(text: str) -> str:
    import re
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _cached_arrow_path(name: str, split: str) -> str | None:
    home = Path.home()
    candidates = {
        ("hellaswag", "train"): home / ".cache" / "huggingface" / "datasets" / "Rowan___hellaswag" / "default" / "0.0.0" / "218ec52e09a7e7462a5400043bb9a69a41d06b76" / "hellaswag-train.arrow",
        ("hellaswag", "validation"): home / ".cache" / "huggingface" / "datasets" / "Rowan___hellaswag" / "default" / "0.0.0" / "218ec52e09a7e7462a5400043bb9a69a41d06b76" / "hellaswag-validation.arrow",
        ("piqa", "train"): home / ".cache" / "huggingface" / "datasets" / "baber___piqa" / "default" / "0.0.0" / "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1" / "piqa-train.arrow",
        ("piqa", "validation"): home / ".cache" / "huggingface" / "datasets" / "baber___piqa" / "default" / "0.0.0" / "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1" / "piqa-validation.arrow",
    }
    path = candidates.get((name, split))
    if path is not None and path.exists():
        return str(path)
    return None


def load_limited_examples(name: str, split: str, count: int, seed: int, allow_downloads: bool) -> list[dict]:
    if count <= 0:
        return []
    if not allow_downloads:
        import os
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import Dataset, load_dataset

    arrow_path = _cached_arrow_path(name, split)
    if arrow_path is not None:
        ds = Dataset.from_file(arrow_path)
    else:
        dataset_name = "Rowan/hellaswag" if name == "hellaswag" else "baber/piqa"
        ds = load_dataset(dataset_name, split=split)
    n = min(count, len(ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist() if n < len(ds) else list(range(n))
    ds = ds.select(indices)
    examples: list[dict] = []
    if name == "hellaswag":
        for row in ds:
            ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
            query = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
            choices = [_hellaswag_preprocess(e) for e in row["endings"]]
            examples.append({"context": query, "choices": choices, "label": int(row["label"])})
    elif name == "piqa":
        for row in ds:
            examples.append({
                "context": f"Question: {row['goal']}\nAnswer:",
                "choices": [row["sol1"], row["sol2"]],
                "label": int(row["label"]),
            })
    else:
        raise ValueError(name)
    return examples

def build_or_load_scorer_cache(
    args: argparse.Namespace,
    codec: SemanticCodec,
    codec_tokenizer,
    device: torch.device,
) -> dict:
    cache_path = Path(args.output_dir) / "scorer_cache.pt"
    if args.reuse_scorer_cache and cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    teacher_device = choose_device(args.teacher_device)
    teacher, teacher_tokenizer = load_teacher_model(args.teacher, teacher_device, args.allow_downloads)

    hs_train = load_limited_examples("hellaswag", "train", args.extract_hellaswag, args.seed, args.allow_downloads)
    piqa_train = load_limited_examples("piqa", "train", args.extract_piqa, args.seed + 1, args.allow_downloads)
    hs_eval = load_limited_examples("hellaswag", "validation", args.eval_hellaswag, args.seed + 2, args.allow_downloads)
    piqa_eval = load_limited_examples("piqa", "validation", args.eval_piqa, args.seed + 3, args.allow_downloads)
    train_examples = hs_train + piqa_train

    scores: list[float] = []
    for i, ex in enumerate(train_examples, start=1):
        ex_scores = [teacher_completion_score(teacher, teacher_tokenizer, ex["context"], c, teacher_device) for c in ex["choices"]]
        scores.extend(ex_scores)
        if args.progress and (i == 1 or i % 100 == 0):
            print(f"teacher-scored {i}/{len(train_examples)} train examples")

    x_train, train_groups = pair_features_for_examples(codec, codec_tokenizer, train_examples, args.scorer_max_bytes, device)
    eval_sets = {"hellaswag": hs_eval, "piqa": piqa_eval}
    x_eval = {}
    groups_eval = {}
    for name, examples in eval_sets.items():
        x_eval[name], groups_eval[name] = pair_features_for_examples(codec, codec_tokenizer, examples, args.scorer_max_bytes, device)

    payload = {
        "train_examples": train_examples,
        "eval_sets": eval_sets,
        "x_train": x_train,
        "y_train": torch.tensor(scores, dtype=torch.float32),
        "train_groups": train_groups,
        "x_eval": x_eval,
        "groups_eval": groups_eval,
        "counts": {
            "extract_hellaswag": len(hs_train),
            "extract_piqa": len(piqa_train),
            "eval_hellaswag": len(hs_eval),
            "eval_piqa": len(piqa_eval),
            "train_pairs": len(scores),
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def pca_project_train_eval(x_train: torch.Tensor, x_eval: dict[str, torch.Tensor], rank: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict]:
    x = x_train.float()
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    q = min(rank, x.shape[0] - 1, x.shape[1])
    _, _, v = torch.pca_lowrank(xc, q=q, center=False)
    basis = v[:, :q].contiguous()
    train_z = xc @ basis
    eval_z = {name: (xe.float() - mean) @ basis for name, xe in x_eval.items()}
    return train_z, eval_z, {"rank": int(q), "input_dim": int(x.shape[1])}


def evaluate_scores(pred_by_set: dict[str, torch.Tensor], cache: dict) -> dict:
    out = {}
    for name, pred in pred_by_set.items():
        out[name] = grouped_accuracy(pred.detach().cpu(), cache["eval_sets"][name], cache["groups_eval"][name])
    return out


def run_scorers(args: argparse.Namespace, codec: SemanticCodec, tokenizer, device: torch.device) -> dict:
    started = time.time()
    cache = build_or_load_scorer_cache(args, codec, tokenizer, device)
    x_train = cache["x_train"].float()
    y_raw = cache["y_train"].float()
    y, y_mean, y_std = standardize(y_raw)
    d_pair = x_train.shape[1]
    d = d_pair // 4

    baseline = {
        name: codec_only_accuracy(xe, cache["eval_sets"][name], cache["groups_eval"][name])
        for name, xe in cache["x_eval"].items()
    }

    results: dict[str, dict] = {
        "codec_only": baseline,
        "target_standardization": {"mean": y_mean, "std": y_std},
        "counts": cache["counts"],
    }

    train_z, eval_z, pca_meta = pca_project_train_eval(x_train, cache["x_eval"], args.mlp_input_rank)
    mlp = MLPScorer(train_z.shape[1])
    mlp_train = train_regressor(mlp, train_z, y, args.mlp_epochs, args.mlp_lr, args.seed + 1000)
    mlp.eval()
    with torch.no_grad():
        pred = {name: mlp(z.float()) for name, z in eval_z.items()}
    results["mlp_pca256"] = {
        "train": mlp_train,
        "pca": pca_meta,
        "accuracy": evaluate_scores(pred, cache),
    }

    for rank in args.bilinear_ranks:
        model = BilinearRankScorer(d, rank)
        train = train_regressor(model, x_train, y, args.bilinear_epochs, args.bilinear_lr, args.seed + 2000 + rank)
        model.eval()
        with torch.no_grad():
            pred = {name: model(xe.float()) for name, xe in cache["x_eval"].items()}
        results[f"bilinear_rank{rank}"] = {"train": train, "accuracy": evaluate_scores(pred, cache)}

    weighted = WeightedCosineScorer(d)
    wc_train = train_regressor(weighted, x_train, y, args.cosine_epochs, args.cosine_lr, args.seed + 3000)
    weighted.eval()
    with torch.no_grad():
        pred = {name: weighted(xe.float()) for name, xe in cache["x_eval"].items()}
    results["learned_weighted_cosine"] = {"train": wc_train, "accuracy": evaluate_scores(pred, cache)}

    best_lifts = {}
    for bench in baseline:
        best_name = None
        best_acc = -1.0
        for name, block in results.items():
            if name in {"codec_only", "target_standardization", "counts"}:
                continue
            acc = block["accuracy"][bench]
            if acc > best_acc:
                best_name = name
                best_acc = acc
        best_lifts[bench] = {
            "best_method": best_name,
            "best_acc": best_acc,
            "codec_only_acc": baseline[bench],
            "delta_pp": 100.0 * (best_acc - baseline[bench]),
        }

    results["precommitted_check"] = {
        "hellaswag_any_scorer_beats_codec_only_by_ge_3pp": best_lifts.get("hellaswag", {}).get("delta_pp", -999.0) >= 3.0,
        "best_lifts": best_lifts,
    }
    results["elapsed_s"] = round(time.time() - started, 3)
    return results


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["zero_cost", "slices", "scorers", "all"], default="all")
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-embeddings", default=DEFAULT_TEACHER_EMB)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--teacher-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--linear-ridge", type=float, default=1.0)
    parser.add_argument("--linear-train-fraction", type=float, default=0.7)
    parser.add_argument("--linear-train-cap", type=int, default=4096)
    parser.add_argument("--extract-hellaswag", type=int, default=512)
    parser.add_argument("--extract-piqa", type=int, default=512)
    parser.add_argument("--eval-hellaswag", type=int, default=1024)
    parser.add_argument("--eval-piqa", type=int, default=1024)
    parser.add_argument("--scorer-max-bytes", type=int, default=384)
    parser.add_argument("--reuse-scorer-cache", action="store_true")
    parser.add_argument("--mlp-input-rank", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--bilinear-ranks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--bilinear-epochs", type=int, default=120)
    parser.add_argument("--bilinear-lr", type=float, default=2e-3)
    parser.add_argument("--cosine-epochs", type=int, default=120)
    parser.add_argument("--cosine-lr", type=float, default=3e-3)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = set_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    teacher_embeddings, emb_manifest = load_teacher_embeddings(args.teacher_embeddings, device)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)

    payload: dict = {
        "run": {
            "mode": args.mode,
            "seed": args.seed,
            "device": str(device),
            "started_at_unix": time.time(),
        },
        "codec": codec_manifest,
        "teacher_embeddings": emb_manifest,
        "args": {k: v for k, v in vars(args).items() if k not in {"json"}},
    }

    patch_batch = None
    if args.mode in {"zero_cost", "slices", "all"}:
        patch_batch = collect_patch_batch(codec, tokenizer, args, device, rng)
        payload["collection"] = patch_batch.collection
    if args.mode in {"zero_cost", "all"}:
        assert patch_batch is not None
        payload["zero_cost"] = run_zero_cost(args, patch_batch, teacher_embeddings, device)
    if args.mode in {"slices", "all"}:
        assert patch_batch is not None
        payload["offset_slices"] = run_offset_slices(patch_batch, teacher_embeddings, device)
    if args.mode in {"scorers", "all"}:
        payload["scorers"] = run_scorers(args, codec, tokenizer, device)

    payload["run"]["finished_at_unix"] = time.time()
    payload["run"]["elapsed_s"] = round(payload["run"]["finished_at_unix"] - payload["run"]["started_at_unix"], 3)
    payload = make_jsonable(payload)
    if not args.no_artifacts:
        out_path = Path(args.output_dir) / f"batch5_{args.mode}_diagnostics.json"
        write_json(out_path, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if "zero_cost" in payload:
            z = payload["zero_cost"]
            print(f"zero_cost best={z['best_method']} top1={z['best_top1']:.4f}")
        if "offset_slices" in payload:
            o = payload["offset_slices"]
            print(f"offset verdict={o['precommitted_verdict']} early_mean={o['precommitted_early_offset_top1_mean']}")
        if "scorers" in payload:
            s = payload["scorers"]["precommitted_check"]["best_lifts"]
            print(json.dumps(s, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
