"""Tier 3.0 Brainseed chart probe.

This script is a minimal real-model falsifier for the trained byte-to-token
codec. The first gate is chart quality at both token-end anchors and 4-byte
patch-boundary anchors. Teacher candidate forwards and frozen scorer extraction
are deliberately kept behind that gate.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from semantic_codec import CodecConfig, SemanticCodec


DEFAULT_CODEC = "C:/sutra_fast/codec_phase1/codec_final.pt"
DEFAULT_TEACHER_EMB = "C:/sutra_fast/teacher_embeddings.pt"
DEFAULT_DATA_DIR = "C:/sutra_fast/data/shards_diverse"
DEFAULT_OUTPUT_DIR = "C:/sutra_fast/brainseed_v0"


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def ensure_offline(allow_downloads: bool) -> None:
    if allow_downloads:
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_tokenizer(name: str, allow_downloads: bool):
    ensure_offline(allow_downloads)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name, local_files_only=not allow_downloads)


def load_teacher_embeddings(path: str, device: torch.device) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    emb = payload["embeddings"].float()
    emb = F.normalize(emb, dim=-1)
    return emb.to(device), {
        "path": path,
        "shape": list(emb.shape),
        "dtype": str(emb.dtype),
        "tokenizer_name": payload.get("tokenizer_name", ""),
    }


def load_codec(path: str, device: torch.device) -> tuple[SemanticCodec, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg_payload = ckpt.get("config", {})
    cfg = CodecConfig(
        codec_dim=int(cfg_payload.get("codec_dim", 256)),
        codec_layers=int(cfg_payload.get("codec_layers", 4)),
        window_size=int(cfg_payload.get("window_size", 256)),
    )
    d_model = int(cfg_payload.get("d_model", 1152))
    codec = SemanticCodec(cfg, d_model=d_model)
    codec.load_state_dict(ckpt["codec_state_dict"])
    codec.to(device)
    codec.eval()
    manifest = {
        "path": path,
        "config": {**asdict(cfg), "d_model": d_model},
        "best_acc": ckpt.get("best_acc"),
        "param_counts": codec.count_params(),
    }
    return codec, manifest


class ByteShardSampler:
    def __init__(self, data_dir: str, seq_len: int):
        self.seq_len = seq_len
        self.shards = sorted(Path(data_dir).glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No .bin shards found in {data_dir}")
        self.shard_sizes = [p.stat().st_size for p in self.shards]
        self._handles: dict[int, object] = {}
        self._mmaps: dict[int, mmap.mmap] = {}

    @property
    def total_bytes(self) -> int:
        return int(sum(self.shard_sizes))

    def _mmap(self, idx: int) -> mmap.mmap:
        if idx not in self._mmaps:
            handle = open(self.shards[idx], "rb")
            self._handles[idx] = handle
            self._mmaps[idx] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmaps[idx]

    def sample(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        rows = []
        for _ in range(batch_size):
            shard_idx = int(rng.integers(0, len(self.shards)))
            max_start = max(0, self.shard_sizes[shard_idx] - self.seq_len)
            start = int(rng.integers(0, max_start + 1)) if max_start else 0
            raw = self._mmap(shard_idx)[start:start + self.seq_len]
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
            if len(arr) < self.seq_len:
                arr = np.pad(arr, (0, self.seq_len - len(arr)), constant_values=32)
            arr[arr == 0xFF] = 32
            rows.append(arr)
        return torch.from_numpy(np.stack(rows)).long()


@dataclass
class AnchorSet:
    positions: list[list[int]]
    token_ids: list[list[int]]

    @property
    def n(self) -> int:
        return int(sum(len(x) for x in self.positions))


def _token_spans_for_bytes(byte_row: torch.Tensor, tokenizer) -> list[tuple[int, int, int]]:
    byte_np = byte_row.cpu().numpy().astype(np.uint8)
    text = bytes(byte_np).decode("utf-8", errors="replace")
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    spans: list[tuple[int, int, int]] = []
    byte_offset = 0
    limit = len(byte_np)

    for tid in token_ids:
        token_text = tokenizer.decode([int(tid)])
        token_bytes = token_text.encode("utf-8", errors="replace")
        if not token_bytes:
            continue
        start = byte_offset
        end = byte_offset + len(token_bytes) - 1
        byte_offset = end + 1
        if start >= limit:
            break
        if end < limit:
            spans.append((start, end, int(tid)))
        else:
            break
    return spans


def _cap_list(values: list[int], cap: int, rng: np.random.Generator) -> list[int]:
    if cap <= 0 or len(values) <= cap:
        return values
    idx = np.sort(rng.choice(len(values), size=cap, replace=False))
    return [values[int(i)] for i in idx]


def find_anchor_sets(
    byte_ids: torch.Tensor,
    tokenizer,
    patch_size: int,
    max_token_anchors_per_seq: int,
    max_patch_anchors_per_seq: int,
    rng: np.random.Generator,
) -> tuple[AnchorSet, AnchorSet, Counter]:
    token_positions: list[list[int]] = []
    token_ids: list[list[int]] = []
    patch_positions: list[list[int]] = []
    patch_token_ids: list[list[int]] = []
    freq: Counter = Counter()

    for b in range(byte_ids.shape[0]):
        spans = _token_spans_for_bytes(byte_ids[b], tokenizer)
        for _, _, tid in spans:
            freq[tid] += 1

        tok_pos = [end for _, end, _ in spans]
        tok_ids = [tid for _, _, tid in spans]
        if max_token_anchors_per_seq > 0 and len(tok_pos) > max_token_anchors_per_seq:
            keep = np.sort(rng.choice(len(tok_pos), size=max_token_anchors_per_seq, replace=False))
            tok_pos = [tok_pos[int(i)] for i in keep]
            tok_ids = [tok_ids[int(i)] for i in keep]
        token_positions.append(tok_pos)
        token_ids.append(tok_ids)

        p_pos: list[int] = []
        p_ids: list[int] = []
        span_idx = 0
        for pos in range(patch_size - 1, byte_ids.shape[1], patch_size):
            while span_idx < len(spans) and spans[span_idx][1] < pos:
                span_idx += 1
            if span_idx >= len(spans):
                break
            start, end, tid = spans[span_idx]
            if start <= pos <= end:
                p_pos.append(pos)
                p_ids.append(tid)
        if max_patch_anchors_per_seq > 0 and len(p_pos) > max_patch_anchors_per_seq:
            keep = np.sort(rng.choice(len(p_pos), size=max_patch_anchors_per_seq, replace=False))
            p_pos = [p_pos[int(i)] for i in keep]
            p_ids = [p_ids[int(i)] for i in keep]
        patch_positions.append(p_pos)
        patch_token_ids.append(p_ids)

    return AnchorSet(token_positions, token_ids), AnchorSet(patch_positions, patch_token_ids), freq


def gather_anchor_hidden(hidden: torch.Tensor, anchors: AnchorSet) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    ids = []
    for b, (positions, token_ids) in enumerate(zip(anchors.positions, anchors.token_ids)):
        if not positions:
            continue
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        rows.append(hidden[b].index_select(0, pos_tensor))
        ids.extend(token_ids)
    if not rows:
        return hidden.new_zeros((0, hidden.shape[-1])), torch.empty(0, dtype=torch.long, device=hidden.device)
    return torch.cat(rows, dim=0), torch.tensor(ids, dtype=torch.long, device=hidden.device)


def signed_permutation_rotation(x: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    perm = torch.randperm(x.shape[-1], generator=generator, device=x.device)
    signs = torch.randint(0, 2, (x.shape[-1],), generator=generator, device=x.device, dtype=torch.long)
    signs = signs.float().mul_(2.0).sub_(1.0).to(x.dtype)
    return x.index_select(-1, perm) * signs


@dataclass
class ChartFeatures:
    token_query: torch.Tensor
    token_random_query: torch.Tensor
    token_ids: torch.Tensor
    patch_query: torch.Tensor
    patch_random_query: torch.Tensor
    patch_ids: torch.Tensor
    token_frequency: Counter
    collection: dict


def collect_chart_features(
    codec: SemanticCodec,
    random_codec: SemanticCodec,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> ChartFeatures:
    sampler = ByteShardSampler(args.data_dir, seq_len=args.seq_len)
    token_q: list[torch.Tensor] = []
    token_rq: list[torch.Tensor] = []
    token_ids: list[torch.Tensor] = []
    patch_q: list[torch.Tensor] = []
    patch_rq: list[torch.Tensor] = []
    patch_ids: list[torch.Tensor] = []
    token_freq: Counter = Counter()
    started = time.time()

    n_done = 0
    while n_done < args.num_sequences:
        bs = min(args.batch_size, args.num_sequences - n_done)
        byte_ids_cpu = sampler.sample(bs, rng)
        token_set, patch_set, freq = find_anchor_sets(
            byte_ids_cpu,
            tokenizer,
            patch_size=codec.cfg.patch_size,
            max_token_anchors_per_seq=args.max_token_anchors_per_seq,
            max_patch_anchors_per_seq=args.max_patch_anchors_per_seq,
            rng=rng,
        )
        token_freq.update(freq)
        byte_ids = byte_ids_cpu.to(device)
        with torch.no_grad():
            hidden = codec.encoder(byte_ids)
            random_hidden = random_codec.encoder(byte_ids)
            tok_hidden, tok_ids = gather_anchor_hidden(hidden, token_set)
            tok_random_hidden, _ = gather_anchor_hidden(random_hidden, token_set)
            pat_hidden, pat_ids = gather_anchor_hidden(hidden, patch_set)
            pat_random_hidden, _ = gather_anchor_hidden(random_hidden, patch_set)
            if tok_hidden.numel():
                token_q.append(codec.alignment_head(tok_hidden).cpu())
                token_rq.append(random_codec.alignment_head(tok_random_hidden).cpu())
                token_ids.append(tok_ids.cpu())
            if pat_hidden.numel():
                patch_q.append(codec.alignment_head(pat_hidden).cpu())
                patch_rq.append(random_codec.alignment_head(pat_random_hidden).cpu())
                patch_ids.append(pat_ids.cpu())
        n_done += bs

    if not token_q or not patch_q:
        raise RuntimeError("No valid anchors were collected")

    return ChartFeatures(
        token_query=torch.cat(token_q, dim=0),
        token_random_query=torch.cat(token_rq, dim=0),
        token_ids=torch.cat(token_ids, dim=0),
        patch_query=torch.cat(patch_q, dim=0),
        patch_random_query=torch.cat(patch_rq, dim=0),
        patch_ids=torch.cat(patch_ids, dim=0),
        token_frequency=token_freq,
        collection={
            "data_dir": args.data_dir,
            "num_sequences": args.num_sequences,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "total_shard_bytes": sampler.total_bytes,
            "elapsed_s": round(time.time() - started, 3),
        },
    )


def topk_retrieval(
    queries: torch.Tensor,
    key_token_ids: torch.Tensor,
    correct_token_ids: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    device: torch.device,
    topks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    n = int(queries.shape[0])
    if n == 0:
        return {f"top{k}": None for k in topks} | {"n": 0, "chance_diag": None, "chance_token_id": None}
    kmax = min(max(topks), n)
    queries_d = F.normalize(queries.to(device).float(), dim=-1)
    key_ids_d = key_token_ids.to(device).long()
    keys = teacher_embeddings.index_select(0, key_ids_d)
    sim = queries_d @ keys.T
    _, idx = torch.topk(sim, k=kmax, dim=1)
    idx_cpu = idx.cpu()
    key_ids_cpu = key_token_ids.cpu()
    correct_cpu = correct_token_ids.cpu()
    out = {"n": n}
    for k in topks:
        kk = min(k, n)
        retrieved = key_ids_cpu[idx_cpu[:, :kk]]
        out[f"top{k}"] = float((retrieved == correct_cpu[:, None]).any(dim=1).float().mean().item())
    counts = Counter(int(x) for x in key_ids_cpu.tolist())
    out["chance_diag"] = 1.0 / max(1, n)
    out["chance_token_id"] = float(sum((c / n) ** 2 for c in counts.values()))
    out["unique_targets"] = int(len(counts))
    return out


def frequency_lookup_metrics(token_ids: torch.Tensor, freq: Counter, topks: tuple[int, ...] = (1, 5, 10)) -> dict:
    ids = [int(x) for x in token_ids.cpu().tolist()]
    n = len(ids)
    if n == 0:
        return {f"top{k}": None for k in topks} | {"n": 0}
    unique_ids = sorted(set(ids), key=lambda tid: (-freq.get(tid, 0), tid))
    out = {"n": n}
    for k in topks:
        top = set(unique_ids[: min(k, len(unique_ids))])
        out[f"top{k}"] = float(sum(tid in top for tid in ids) / n)
    out["unique_targets"] = int(len(unique_ids))
    return out


def subset_by_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return x[:0]
    return x[mask]


def rarity_masks(token_ids: torch.Tensor, freq: Counter) -> dict[str, torch.Tensor]:
    vals = np.array([freq.get(int(t), 0) for t in token_ids.cpu().tolist()], dtype=np.float64)
    if vals.size == 0:
        return {"rare": torch.zeros(0, dtype=torch.bool), "frequent": torch.zeros(0, dtype=torch.bool)}
    rare_cut = max(1.0, float(np.quantile(vals, 0.25)))
    freq_cut = float(np.quantile(vals, 0.75))
    return {
        "rare": torch.from_numpy(vals <= rare_cut),
        "frequent": torch.from_numpy(vals >= freq_cut),
    }


def evaluate_anchor_set(
    name: str,
    queries: torch.Tensor,
    random_queries: torch.Tensor,
    token_ids: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    freq: Counter,
    device: torch.device,
    seed: int,
) -> dict:
    n = int(token_ids.shape[0])
    generator = torch.Generator()
    generator.manual_seed(seed)
    vocab = int(teacher_embeddings.shape[0])
    random_ids = torch.randint(0, vocab, token_ids.shape, generator=generator, dtype=torch.long)
    perm = torch.randperm(vocab, generator=generator)
    shuffled_ids = perm[token_ids.long()]
    rotated = signed_permutation_rotation(queries, seed + 991)

    controls = {
        "per_occurrence_random_target": topk_retrieval(queries, random_ids, random_ids, teacher_embeddings, device),
        "fixed_shuffled_target": topk_retrieval(queries, shuffled_ids, shuffled_ids, teacher_embeddings, device),
        "random_codec": topk_retrieval(random_queries, token_ids, token_ids, teacher_embeddings, device),
        "rotated_chart": topk_retrieval(rotated, token_ids, token_ids, teacher_embeddings, device),
        "frequency_lookup": frequency_lookup_metrics(token_ids, freq),
    }
    real = topk_retrieval(queries, token_ids, token_ids, teacher_embeddings, device)

    masks = rarity_masks(token_ids, freq)
    slices = {}
    for slice_name, mask in masks.items():
        mask = mask.bool()
        if int(mask.sum()) == 0:
            continue
        q_s = subset_by_mask(queries, mask)
        rq_s = subset_by_mask(random_queries, mask)
        ids_s = subset_by_mask(token_ids, mask)
        random_s = subset_by_mask(random_ids, mask)
        shuffled_s = subset_by_mask(shuffled_ids, mask)
        rotated_s = subset_by_mask(rotated, mask)
        slices[slice_name] = {
            "real": topk_retrieval(q_s, ids_s, ids_s, teacher_embeddings, device),
            "controls": {
                "per_occurrence_random_target": topk_retrieval(q_s, random_s, random_s, teacher_embeddings, device),
                "fixed_shuffled_target": topk_retrieval(q_s, shuffled_s, shuffled_s, teacher_embeddings, device),
                "random_codec": topk_retrieval(rq_s, ids_s, ids_s, teacher_embeddings, device),
                "rotated_chart": topk_retrieval(rotated_s, ids_s, ids_s, teacher_embeddings, device),
                "frequency_lookup": frequency_lookup_metrics(ids_s, freq),
            },
        }

    best_control_top1 = max(v["top1"] for v in controls.values() if v.get("top1") is not None)
    return {
        "name": name,
        "n": n,
        "real": real,
        "controls": controls,
        "slices": slices,
        "best_control_top1": float(best_control_top1),
        "real_vs_best_control_gap_pp": float(100.0 * (real["top1"] - best_control_top1)),
    }


def gate_a(metrics: dict) -> dict:
    token = metrics["token_end"]
    patch = metrics["patch_boundary"]
    token_fixed = token["controls"]["fixed_shuffled_target"]["top1"]
    token_gap_fixed = token["real"]["top1"] - token_fixed
    per_random = token["controls"]["per_occurrence_random_target"]["top1"]
    chance = token["real"]["chance_diag"]
    patch_gap = patch["real"]["top1"] - patch["best_control_top1"]

    rare = patch["slices"].get("rare", {})
    rare_real = rare.get("real", {}).get("top1")
    rare_controls = rare.get("controls", {})
    rare_best = max((v["top1"] for v in rare_controls.values() if v.get("top1") is not None), default=None)
    rare_gap = None if rare_real is None or rare_best is None else rare_real - rare_best

    checks = {
        "token_end_top1_ge_50": token["real"]["top1"] >= 0.50,
        "token_end_real_vs_fixed_shuffled_gap_ge_25pp": token_gap_fixed >= 0.25,
        "per_occurrence_random_within_2x_chance": per_random <= 2.0 * chance + 1e-12,
        "patch_boundary_top1_ge_30_or_top10_ge_65": patch["real"]["top1"] >= 0.30 or patch["real"]["top10"] >= 0.65,
        "patch_boundary_real_vs_best_control_gap_ge_15pp": patch_gap >= 0.15,
        "rare_patch_top1_ge_15": rare_real is not None and rare_real >= 0.15,
        "rare_patch_beats_best_control_ge_8pp": rare_gap is not None and rare_gap >= 0.08,
    }
    kill_checks = {
        "patch_boundary_top1_lt_15": patch["real"]["top1"] < 0.15,
        "patch_boundary_gap_lt_8pp": patch_gap < 0.08,
        "controls_explain_most_signal": patch["best_control_top1"] >= 0.75 * max(patch["real"]["top1"], 1e-12),
    }
    if all(checks.values()):
        verdict = "PASS_GATE_A"
    elif checks["token_end_top1_ge_50"] and not checks["patch_boundary_top1_ge_30_or_top10_ge_65"]:
        verdict = "ROUTE_PHASE_1_5_DENSE_PATCH_SUPERVISION"
    elif any(kill_checks.values()):
        verdict = "KILL_PROCEED_TO_TRANSPLANT"
    else:
        verdict = "FAIL_GATE_A_INCONCLUSIVE"
    return {
        "checks": checks,
        "kill_checks": kill_checks,
        "token_end_real_vs_fixed_gap_pp": round(100.0 * token_gap_fixed, 4),
        "patch_real_vs_best_control_gap_pp": round(100.0 * patch_gap, 4),
        "rare_patch_real_vs_best_control_gap_pp": None if rare_gap is None else round(100.0 * rare_gap, 4),
        "verdict": verdict,
    }


def run_synthetic_smoke(args: argparse.Namespace) -> dict:
    rng = set_seed(args.seed)
    vocab = 4096
    dim = 128
    n = args.synthetic_n
    device = torch.device("cpu")
    teacher = F.normalize(torch.randn(vocab, dim), dim=-1)
    ids = torch.from_numpy(rng.integers(0, vocab, size=n)).long()
    queries = teacher[ids] + 0.03 * torch.randn(n, dim)
    queries = F.normalize(queries, dim=-1)
    random_queries = F.normalize(torch.randn(n, dim), dim=-1)
    freq = Counter(int(x) for x in ids.tolist())
    metrics = {
        "token_end": evaluate_anchor_set("token_end", queries, random_queries, ids, teacher, freq, device, args.seed),
        "patch_boundary": evaluate_anchor_set("patch_boundary", queries, random_queries, ids, teacher, freq, device, args.seed + 1),
    }
    metrics["gate_a"] = gate_a(metrics)
    return metrics


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def maybe_write_artifacts(args: argparse.Namespace, payload: dict, codec_manifest: dict | None = None) -> None:
    if args.no_artifacts:
        return
    if payload.get("gate_a", {}).get("verdict") != "PASS_GATE_A" and not args.write_failed_artifacts:
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "chart_metrics.json", payload)
    if codec_manifest is not None:
        write_json(out / "codec_manifest.json", codec_manifest)


def load_examples(name: str, split: str, count: int, seed: int, allow_downloads: bool) -> list[dict]:
    if count <= 0:
        return []
    ensure_offline(allow_downloads)
    import sys

    code_dir = str(Path(__file__).resolve().parent)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    from benchmark_harness import load_hellaswag, load_piqa

    loader = load_hellaswag if name == "hellaswag" else load_piqa
    examples = loader(split)
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples[:count]


def load_teacher_model(name: str, device: torch.device, allow_downloads: bool):
    ensure_offline(allow_downloads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=not allow_downloads)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        local_files_only=not allow_downloads,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def teacher_completion_score(model, tokenizer, context: str, choice: str, device: torch.device) -> float:
    text = context + " " + choice
    prefix = tokenizer(context, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if full.numel() <= prefix.numel():
        return -1e9
    input_ids = full.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = F.log_softmax(logits.float(), dim=-1)
    start = max(0, int(prefix.numel()) - 1)
    target_logp = logp[0, start:, :].gather(1, labels[0, start:].unsqueeze(1)).squeeze(1)
    return float(target_logp.mean().item())


def pooled_codec_text_feature(
    codec: SemanticCodec,
    tokenizer,
    text: str,
    max_bytes: int,
    device: torch.device,
) -> torch.Tensor:
    raw = list(text.encode("utf-8", errors="replace"))[:max_bytes]
    if not raw:
        raw = [32]
    byte_ids = torch.tensor(raw, dtype=torch.long).unsqueeze(0)
    token_set, patch_set, _ = find_anchor_sets(
        byte_ids,
        tokenizer,
        patch_size=codec.cfg.patch_size,
        max_token_anchors_per_seq=0,
        max_patch_anchors_per_seq=0,
        rng=np.random.default_rng(0),
    )
    byte_ids = byte_ids.to(device)
    with torch.no_grad():
        hidden = codec.encoder(byte_ids)
        anchor_hidden, _ = gather_anchor_hidden(hidden, patch_set if patch_set.n else token_set)
        if anchor_hidden.numel() == 0:
            anchor_hidden = hidden[:, -1, :]
        projected = codec.alignment_head(anchor_hidden)
    return F.normalize(projected.float().mean(dim=0), dim=0).cpu()


def pair_feature(ctx: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    return torch.cat([ctx, cand, ctx * cand, torch.abs(ctx - cand)], dim=0)


def build_pair_features(
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
            rows.append(pair_feature(ctx, cache[key]))
    return torch.stack(rows), group_sizes


def ridge_fit_ranked(x: torch.Tensor, y: torch.Tensor, rank: int, ridge: float) -> dict:
    x = x.float()
    y = y.float()
    mean = x.mean(dim=0, keepdim=True)
    xc = x - mean
    u, s, v = torch.pca_lowrank(xc, q=rank, center=False)
    basis = v[:, :rank].contiguous()
    z = xc @ basis
    eye = torch.eye(rank)
    w = torch.linalg.solve(z.T @ z + ridge * eye, z.T @ y)
    bias = y.mean() - (z.mean(dim=0) @ w)
    return {"mean": mean.squeeze(0), "basis": basis, "weight": w, "bias": bias}


def ridge_predict(model: dict, x: torch.Tensor) -> torch.Tensor:
    z = (x.float() - model["mean"]) @ model["basis"]
    return z @ model["weight"] + model["bias"]


def grouped_accuracy(scores: torch.Tensor, examples: list[dict], group_sizes: list[int]) -> float:
    offset = 0
    correct = 0
    for ex, size in zip(examples, group_sizes):
        group = scores[offset:offset + size]
        pred = int(torch.argmax(group).item())
        correct += int(pred == int(ex["label"]))
        offset += size
    return correct / max(1, len(examples))


def codec_only_accuracy(x: torch.Tensor, examples: list[dict], group_sizes: list[int]) -> float:
    # pair_feature layout: [ctx, cand, product, absdiff]
    d = x.shape[1] // 4
    scores = (x[:, :d] * x[:, d:2 * d]).sum(dim=1)
    return grouped_accuracy(scores, examples, group_sizes)


def run_frozen_scorer(
    args: argparse.Namespace,
    codec: SemanticCodec,
    codec_tokenizer,
    device: torch.device,
    gate_payload: dict,
) -> dict:
    if gate_payload["gate_a"]["verdict"] != "PASS_GATE_A":
        return {"skipped": True, "reason": gate_payload["gate_a"]["verdict"]}

    teacher_device = choose_device(args.teacher_device)
    teacher, teacher_tokenizer = load_teacher_model(args.teacher, teacher_device, args.allow_downloads)
    hs_train = load_examples("hellaswag", "train", args.extract_hellaswag, args.seed, args.allow_downloads)
    piqa_train = load_examples("piqa", "train", args.extract_piqa, args.seed + 1, args.allow_downloads)
    hs_eval = load_examples("hellaswag", "validation", args.eval_hellaswag, args.seed + 2, args.allow_downloads)
    piqa_eval = load_examples("piqa", "validation", args.eval_piqa, args.seed + 3, args.allow_downloads)

    train_examples = hs_train + piqa_train
    extraction_scores: list[float] = []
    for ex in train_examples:
        scores = [teacher_completion_score(teacher, teacher_tokenizer, ex["context"], c, teacher_device) for c in ex["choices"]]
        extraction_scores.extend(scores)
    y = torch.tensor(extraction_scores, dtype=torch.float32)

    x_train, train_groups = build_pair_features(codec, codec_tokenizer, train_examples, args.scorer_max_bytes, device)
    results = {
        "extraction_examples": {"hellaswag": len(hs_train), "piqa": len(piqa_train)},
        "eval_examples": {"hellaswag": len(hs_eval), "piqa": len(piqa_eval)},
        "ranks": {},
    }

    eval_sets = {
        "hellaswag": hs_eval,
        "piqa": piqa_eval,
    }
    x_eval = {}
    groups_eval = {}
    for name, examples in eval_sets.items():
        x_eval[name], groups_eval[name] = build_pair_features(codec, codec_tokenizer, examples, args.scorer_max_bytes, device)

    for rank in args.ranks:
        model = ridge_fit_ranked(x_train, y, rank=rank, ridge=args.ridge)
        rank_result = {}
        for name, examples in eval_sets.items():
            pred = ridge_predict(model, x_eval[name])
            rank_result[name] = {
                "brainseed_acc": grouped_accuracy(pred, examples, groups_eval[name]),
                "codec_only_acc": codec_only_accuracy(x_eval[name], examples, groups_eval[name]),
            }
        results["ranks"][str(rank)] = rank_result
        if not args.no_artifacts:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model["basis"].half(), out / f"basis_B_rank{rank}.fp16.pt")
            torch.save(model["weight"].half(), out / f"energy_E_rank{rank}.fp16.pt")

    return results


def run_chart_probe(args: argparse.Namespace) -> tuple[dict, dict]:
    rng = set_seed(args.seed)
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    teacher_embeddings, emb_manifest = load_teacher_embeddings(args.teacher_embeddings, device)
    codec, codec_manifest = load_codec(args.codec_checkpoint, device)
    random_codec = SemanticCodec(codec.cfg, d_model=codec.d_model).to(device)
    random_codec.eval()

    features = collect_chart_features(codec, random_codec, tokenizer, args, device, rng)
    metrics = {
        "run": {
            "seed": args.seed,
            "device": str(device),
            "chart_only": args.chart_only,
            "teacher": args.teacher,
            "started_at_unix": time.time(),
        },
        "teacher_embeddings": emb_manifest,
        "codec": codec_manifest,
        "collection": features.collection,
        "token_end": evaluate_anchor_set(
            "token_end",
            features.token_query,
            features.token_random_query,
            features.token_ids,
            teacher_embeddings,
            features.token_frequency,
            device,
            args.seed + 10,
        ),
        "patch_boundary": evaluate_anchor_set(
            "patch_boundary",
            features.patch_query,
            features.patch_random_query,
            features.patch_ids,
            teacher_embeddings,
            features.token_frequency,
            device,
            args.seed + 20,
        ),
    }
    metrics["gate_a"] = gate_a(metrics)
    metrics["run"]["finished_at_unix"] = time.time()
    metrics["run"]["elapsed_s"] = round(metrics["run"]["finished_at_unix"] - metrics["run"]["started_at_unix"], 3)
    return metrics, codec_manifest


def compact_print(payload: dict) -> None:
    if "gate_a" not in payload:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print("Tier 3.0 Brainseed Chart Probe")
    for key in ("token_end", "patch_boundary"):
        block = payload[key]
        print(f"\n[{key}] n={block['n']}")
        print(
            "  real: "
            f"top1={block['real']['top1']:.4f} "
            f"top5={block['real']['top5']:.4f} "
            f"top10={block['real']['top10']:.4f}"
        )
        for cname, cm in block["controls"].items():
            print(f"  {cname}: top1={cm['top1']:.4f} top10={cm['top10']:.4f}")
        if "rare" in block["slices"]:
            rare = block["slices"]["rare"]["real"]
            print(f"  rare real: n={rare['n']} top1={rare['top1']:.4f} top10={rare['top10']:.4f}")
        print(f"  real-vs-best-control gap: {block['real_vs_best_control_gap_pp']:.2f}pp")
    gate = payload["gate_a"]
    print(f"\nGate A verdict: {gate['verdict']}")
    for name, ok in gate["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-embeddings", default=DEFAULT_TEACHER_EMB)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract-hellaswag", type=int, default=512)
    parser.add_argument("--extract-piqa", type=int, default=512)
    parser.add_argument("--eval-hellaswag", type=int, default=1024)
    parser.add_argument("--eval-piqa", type=int, default=1024)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--chart-only", action="store_true")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--write-failed-artifacts", action="store_true")
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--teacher-device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--num-sequences", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-token-anchors-per-seq", type=int, default=128)
    parser.add_argument("--max-patch-anchors-per-seq", type=int, default=128)
    parser.add_argument("--scorer-max-bytes", type=int, default=384)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-n", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_smoke:
        payload = run_synthetic_smoke(args)
        print(json.dumps(payload, indent=2, sort_keys=True) if args.json else "", end="") if args.json else compact_print(payload)
        return

    payload, codec_manifest = run_chart_probe(args)
    if not args.chart_only:
        payload["frozen_scorer"] = run_frozen_scorer(args, load_codec(args.codec_checkpoint, choose_device(args.device))[0], load_tokenizer(args.teacher, args.allow_downloads), choose_device(args.device), payload)
    maybe_write_artifacts(args, payload, codec_manifest)
    print(json.dumps(payload, indent=2, sort_keys=True) if args.json else "", end="") if args.json else compact_print(payload)


if __name__ == "__main__":
    main()
