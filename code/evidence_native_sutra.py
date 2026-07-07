"""Batch 6 evidence-native Sutra prototype.

This is the first retrieval-born candidate judge, not a production S0 trainer.
It builds a hashed evidence corpus, retrieves passages for HellaSwag/PIQA,
serializes (context, evidence, candidate) as bytes with section separators,
and trains a small byte-native judgment model to score candidates.

The default run is intentionally small enough for a fast falsification pass.
The architecture knobs can be raised toward the 121M target once the controls
show a real evidence-use signal.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s0_architecture import GlobalReasoner, S0Config
from semantic_codec import CodecConfig, PatchProjection, SemanticCodec


DEFAULT_OUTPUT_DIR = "C:/sutra_fast/evidence_native"
DEFAULT_CODEC = "C:/sutra_fast/codec_phase1.5/codec_final.pt"
DEFAULT_DATA_DIR = "C:/sutra_fast/data/shards_diverse"
DEFAULT_TEACHER_CACHE = "C:/sutra_fast/brainseed_batch5/scorer_cache.pt"

SEP_CONTEXT_EVIDENCE = 256
SEP_EVIDENCE_CANDIDATE = 257
EVIDENCE_PASSAGE_SEP = 258
PAD_BYTE = 259
PASSAGE_SEPARATOR_TEXT = "\n[PASSAGE]\n"


def ensure_offline(allow_downloads: bool) -> None:
    if allow_downloads:
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def text_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def sha256_texts(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(normalize_text(text).encode("utf-8", errors="replace"))
        h.update(b"\n")
    return h.hexdigest()


def _hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def cached_arrow_path(name: str, split: str) -> str | None:
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


def shuffle_choices(
    choices: list[str],
    label: int,
    rng: random.Random,
    enabled: bool,
) -> tuple[list[str], int, list[int]]:
    order = list(range(len(choices)))
    if enabled and len(order) > 1:
        rng.shuffle(order)
    shuffled = [choices[i] for i in order]
    return shuffled, order.index(int(label)), order


def load_limited_examples(
    name: str,
    split: str,
    count: int,
    seed: int,
    allow_downloads: bool,
    randomize_choices: bool,
) -> list[dict]:
    if count <= 0:
        return []
    ensure_offline(allow_downloads)
    from datasets import Dataset, load_dataset

    arrow_path = cached_arrow_path(name, split)
    if arrow_path is not None:
        ds = Dataset.from_file(arrow_path)
    else:
        dataset_name = "Rowan/hellaswag" if name == "hellaswag" else "baber/piqa"
        ds = load_dataset(dataset_name, split=split)

    n = min(count, len(ds))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n, replace=False).tolist() if n < len(ds) else list(range(n))
    ds = ds.select(indices)
    choice_rng = random.Random(seed + 10_003)

    examples: list[dict] = []
    if name == "hellaswag":
        for row in ds:
            ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
            query = _hellaswag_preprocess(row["activity_label"] + ": " + ctx)
            choices = [_hellaswag_preprocess(e) for e in row["endings"]]
            choices, label, order = shuffle_choices(choices, int(row["label"]), choice_rng, randomize_choices)
            examples.append({
                "dataset": name,
                "split": split,
                "context": query,
                "choices": choices,
                "label": label,
                "choice_order": order,
            })
    elif name == "piqa":
        for row in ds:
            choices, label, order = shuffle_choices([row["sol1"], row["sol2"]], int(row["label"]), choice_rng, randomize_choices)
            examples.append({
                "dataset": name,
                "split": split,
                "context": f"Question: {row['goal']}\nAnswer:",
                "choices": choices,
                "label": label,
                "choice_order": order,
            })
    else:
        raise ValueError(name)
    return examples


def example_key(ex: dict) -> str:
    payload = {
        "dataset": ex.get("dataset", ""),
        "context": normalize_text(ex["context"]),
        "choices": [normalize_text(c) for c in ex["choices"]],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def example_key_without_dataset(ex: dict) -> str:
    payload = {
        "context": normalize_text(ex["context"]),
        "choices": [normalize_text(c) for c in ex["choices"]],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def load_teacher_score_map(path: str) -> dict[str, list[float]]:
    if not path or not Path(path).exists():
        return {}
    cache = torch.load(path, map_location="cpu", weights_only=False)
    examples = cache.get("train_examples", [])
    scores = cache.get("y_train")
    if scores is None:
        return {}
    scores = scores.detach().cpu().float().tolist()
    out: dict[str, list[float]] = {}
    offset = 0
    for ex in examples:
        n = len(ex["choices"])
        ex_scores = [float(x) for x in scores[offset : offset + n]]
        out[example_key(ex)] = ex_scores
        out[example_key_without_dataset(ex)] = ex_scores
        offset += n
    return out


def load_codec_checkpoint(path: str, device: torch.device) -> tuple[SemanticCodec, dict]:
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
    return codec, {
        "path": path,
        "config": {**asdict(cfg), "d_model": d_model},
        "best_acc": ckpt.get("best_acc"),
        "param_counts": codec.count_params(),
    }


@dataclass
class CorpusDoc:
    doc_id: int
    source: str
    text: str


def docs_from_examples(examples: list[dict], source_prefix: str) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    for ex in examples:
        pieces = [ex["context"]]
        pieces.extend(ex["choices"])
        text = " ".join(pieces)
        if len(text_tokens(text)) >= 5:
            docs.append(CorpusDoc(-1, f"{source_prefix}:{ex['dataset']}:{ex['split']}", text))
    return docs


def docs_from_diverse_shards(data_dir: str, max_docs: int, max_bytes: int) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    root = Path(data_dir)
    if max_docs <= 0 or not root.exists():
        return docs
    consumed = 0
    for shard in sorted(root.glob("*.bin")):
        if len(docs) >= max_docs or consumed >= max_bytes:
            break
        take = min(max_bytes - consumed, shard.stat().st_size)
        if take <= 0:
            break
        raw = shard.open("rb").read(take)
        consumed += len(raw)
        for part in raw.split(b"\xff"):
            if len(docs) >= max_docs:
                break
            text = part.decode("utf-8", errors="replace")
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) < 120:
                continue
            if len(text) > 700:
                text = text[:700]
            if len(text_tokens(text)) < 12:
                continue
            docs.append(CorpusDoc(-1, "diverse_shard_public_text", text))
    return docs


class BM25Retriever:
    def __init__(self, docs: list[CorpusDoc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = [text_tokens(d.text) for d in docs]
        self.doc_terms: list[set[str]] = [set(toks) for toks in self.doc_tokens]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = float(sum(self.doc_lens) / max(1, len(self.doc_lens)))
        self.term_freqs: list[dict[str, int]] = []
        self.postings: dict[str, list[int]] = {}
        df: dict[str, int] = {}
        for idx, toks in enumerate(self.doc_tokens):
            tf: dict[str, int] = {}
            for tok in toks:
                tf[tok] = tf.get(tok, 0) + 1
            self.term_freqs.append(tf)
            for tok in tf:
                df[tok] = df.get(tok, 0) + 1
                self.postings.setdefault(tok, []).append(idx)
        n_docs = max(1, len(docs))
        self.idf = {
            tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }

    def score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        score = 0.0
        dl = self.doc_lens[doc_idx]
        tf = self.term_freqs[doc_idx]
        denom_const = self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-6))
        for tok in set(query_tokens):
            f = tf.get(tok, 0)
            if f <= 0:
                continue
            score += self.idf.get(tok, 0.0) * f * (self.k1 + 1.0) / (f + denom_const)
        return score

    def top_k(self, query: str, k: int = 3, banned_ids: set[int] | None = None) -> list[tuple[int, float]]:
        q = text_tokens(query)
        if not q:
            return []
        banned_ids = banned_ids or set()
        scores: dict[int, float] = {}
        for tok in set(q):
            for doc_idx in self.postings.get(tok, []):
                if doc_idx in banned_ids:
                    continue
                scores.setdefault(doc_idx, 0.0)
        for doc_idx in list(scores):
            scores[doc_idx] = self.score_doc(q, doc_idx)
        if not scores:
            return []
        best = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
        return [(int(i), float(s)) for i, s in best]

    def low_overlap_random(self, query: str, k: int, rng: random.Random) -> list[int]:
        q = set(text_tokens(query))
        if not self.docs:
            return []
        candidates = [i for i, terms in enumerate(self.doc_terms) if not (q & terms)]
        if len(candidates) < k:
            candidates = list(range(len(self.docs)))
        return rng.sample(candidates, k=min(k, len(candidates)))

    def random_docs(self, k: int, rng: random.Random) -> list[int]:
        if not self.docs:
            return []
        return rng.sample(range(len(self.docs)), k=min(k, len(self.docs)))


def build_corpus(
    train_examples: list[dict],
    data_dir: str,
    max_corpus_docs: int,
    shard_docs: int,
    shard_bytes: int,
) -> tuple[list[CorpusDoc], dict]:
    docs = docs_from_examples(train_examples, "benchmark_train")
    remaining = max(0, max_corpus_docs - len(docs))
    docs.extend(docs_from_diverse_shards(data_dir, min(shard_docs, remaining), shard_bytes))
    docs = docs[:max_corpus_docs]
    for idx, doc in enumerate(docs):
        doc.doc_id = idx
    texts = [d.text for d in docs]
    manifest = {
        "n_docs": len(docs),
        "sha256_normalized": sha256_texts(texts),
        "sources": sorted({d.source for d in docs}),
        "max_corpus_docs": max_corpus_docs,
        "shard_docs": shard_docs,
        "shard_bytes": shard_bytes,
        "data_dir": data_dir,
    }
    return docs, manifest


def evidence_text(doc_ids: list[int], docs: list[CorpusDoc]) -> str:
    chunks = []
    for doc_id in doc_ids:
        if 0 <= doc_id < len(docs):
            chunks.append(docs[doc_id].text)
    return PASSAGE_SEPARATOR_TEXT.join(chunks)


def assign_shuffled_retrieved_evidence(
    records: list[dict],
    docs: list[CorpusDoc],
    top_k: int,
    rng: random.Random,
) -> None:
    by_dataset: dict[str, list[int]] = {}
    for idx, rec in enumerate(records):
        by_dataset.setdefault(rec.get("dataset", ""), []).append(idx)

    for indices in by_dataset.values():
        if len(indices) > 1:
            receivers = indices[:]
            rng.shuffle(receivers)
            donors = receivers[1:] + receivers[:1]
            for receiver_idx, donor_idx in zip(receivers, donors):
                donor_docs = records[donor_idx]["evidence_doc_ids"].get("retrieved", [])
                records[receiver_idx]["evidence_doc_ids"]["shuffled"] = list(donor_docs)[:top_k]
        elif indices:
            pool = list(range(len(docs)))
            fallback = rng.sample(pool, k=min(top_k, len(pool))) if pool else []
            records[indices[0]]["evidence_doc_ids"]["shuffled"] = fallback


def leakage_audit(eval_examples: list[dict], docs: list[CorpusDoc]) -> dict:
    corpus_norm = "\n".join(normalize_text(d.text) for d in docs)
    exact_context_hits = 0
    exact_choice_hits = 0
    long_ngram_hits = 0
    hit_examples: list[dict] = []
    for i, ex in enumerate(eval_examples):
        ctx = normalize_text(ex["context"])
        ctx_hit = bool(ctx and ctx in corpus_norm)
        choice_hit = any(normalize_text(c) in corpus_norm for c in ex["choices"] if len(normalize_text(c)) >= 24)
        toks = text_tokens(ex["context"])
        ngram_hit = False
        if len(toks) >= 12:
            for start in range(0, len(toks) - 11):
                phrase = " ".join(toks[start : start + 12])
                if phrase in corpus_norm:
                    ngram_hit = True
                    break
        exact_context_hits += int(ctx_hit)
        exact_choice_hits += int(choice_hit)
        long_ngram_hits += int(ngram_hit)
        if (ctx_hit or choice_hit or ngram_hit) and len(hit_examples) < 10:
            hit_examples.append({
                "idx": i,
                "dataset": ex.get("dataset"),
                "ctx_hit": ctx_hit,
                "choice_hit": choice_hit,
                "long_ngram_hit": ngram_hit,
                "context_preview": ex["context"][:160],
            })
    return {
        "eval_examples": len(eval_examples),
        "exact_context_hits": exact_context_hits,
        "exact_choice_hits": exact_choice_hits,
        "long_12gram_hits": long_ngram_hits,
        "sample_hits": hit_examples,
    }


def prepare_records(args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    randomize_choices = not args.no_randomize_choices
    train_hs = load_limited_examples("hellaswag", "train", args.train_hellaswag, args.seed, args.allow_downloads, randomize_choices)
    train_piqa = load_limited_examples("piqa", "train", args.train_piqa, args.seed + 1, args.allow_downloads, randomize_choices)
    eval_hs = load_limited_examples("hellaswag", "validation", args.eval_hellaswag, args.seed + 2, args.allow_downloads, randomize_choices)
    eval_piqa = load_limited_examples("piqa", "validation", args.eval_piqa, args.seed + 3, args.allow_downloads, randomize_choices)
    train_examples = train_hs + train_piqa
    eval_examples = eval_hs + eval_piqa

    docs, corpus_manifest = build_corpus(
        train_examples,
        args.data_dir,
        args.max_corpus_docs,
        args.shard_docs,
        args.shard_bytes,
    )
    retriever = BM25Retriever(docs)
    rng = random.Random(args.seed)
    teacher_scores = load_teacher_score_map(args.teacher_cache)

    def enrich(examples: list[dict], split_name: str) -> list[dict]:
        records: list[dict] = []
        for idx, ex in enumerate(examples):
            retrieved = [doc_id for doc_id, _ in retriever.top_k(ex["context"], args.top_k)]
            gold_query = ex["context"] + " " + ex["choices"][int(ex["label"])]
            gold = [doc_id for doc_id, _ in retriever.top_k(gold_query, args.top_k)]
            wrong = retriever.low_overlap_random(ex["context"], args.top_k, rng)
            if len(retrieved) < args.top_k:
                retrieved.extend(retriever.random_docs(args.top_k - len(retrieved), rng))
            if len(gold) < args.top_k:
                gold.extend(retriever.random_docs(args.top_k - len(gold), rng))
            rec = {
                "id": f"{split_name}-{idx:06d}",
                "dataset": ex["dataset"],
                "split": ex["split"],
                "context": ex["context"],
                "choices": ex["choices"],
                "label": int(ex["label"]),
                "evidence_doc_ids": {
                    "retrieved": retrieved[: args.top_k],
                    "shuffled": [],
                    "wrong_topic": wrong[: args.top_k],
                    "gold": gold[: args.top_k],
                    "none": [],
                },
                "teacher_scores": teacher_scores.get(example_key(ex)) or teacher_scores.get(example_key_without_dataset(ex)),
            }
            records.append(rec)
        return records

    train_records = enrich(train_examples, "train")
    eval_records = enrich(eval_examples, "eval")
    assign_shuffled_retrieved_evidence(train_records, docs, args.top_k, rng)
    assign_shuffled_retrieved_evidence(eval_records, docs, args.top_k, rng)
    leak = leakage_audit(eval_examples, docs)
    teacher_attached = sum(1 for r in train_records if r.get("teacher_scores") is not None)

    payload = {
        "created_at_unix": time.time(),
        "args": vars(args),
        "corpus_manifest": corpus_manifest,
        "leakage_audit": leak,
        "controls": {
            "candidate_order_randomized": randomize_choices,
            "shuffled_evidence": "retrieved_doc_ids_deranged_within_dataset",
            "wrong_topic_evidence": "low_query_overlap_random_docs_with_random_fallback",
            "passage_separator": PASSAGE_SEPARATOR_TEXT.strip(),
        },
        "teacher_cache": {
            "path": args.teacher_cache,
            "train_records_with_teacher_scores": teacher_attached,
            "train_records_total": len(train_records),
        },
        "docs": [asdict(d) for d in docs],
        "train_records": train_records,
        "eval_records": eval_records,
    }
    path = out_dir / "evidence_records.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "corpus_manifest.json").write_text(json.dumps({
        "corpus_manifest": corpus_manifest,
        "leakage_audit": leak,
        "controls": payload["controls"],
        "teacher_cache": payload["teacher_cache"],
    }, indent=2), encoding="utf-8")
    return payload


def load_or_prepare_records(args: argparse.Namespace) -> dict:
    path = Path(args.output_dir) / "evidence_records.json"
    if args.reuse_records and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return prepare_records(args)


def truncate_bytes(values: list[int], max_len: int, keep_tail: bool = False) -> list[int]:
    if len(values) <= max_len:
        return values
    if keep_tail:
        return values[-max_len:]
    return values[:max_len]


def serialize_triple(
    context: str,
    evidence: str,
    candidate: str,
    max_context_bytes: int,
    max_evidence_bytes: int,
    max_candidate_bytes: int,
    max_total_bytes: int,
) -> tuple[list[int], int]:
    ctx = truncate_bytes(list(context.encode("utf-8", errors="replace")), max_context_bytes, keep_tail=True)
    ev = truncate_bytes(list(evidence.encode("utf-8", errors="replace")), max_evidence_bytes)
    cand = truncate_bytes(list(candidate.encode("utf-8", errors="replace")), max_candidate_bytes)
    ids = ctx + [SEP_CONTEXT_EVIDENCE] + ev + [SEP_EVIDENCE_CANDIDATE] + cand
    if len(ids) > max_total_bytes:
        overflow = len(ids) - max_total_bytes
        if overflow < len(ev):
            ev = ev[:-overflow]
        else:
            ev = []
            overflow = len(ctx) + 2 + len(cand) - max_total_bytes
            if overflow > 0:
                ctx = ctx[overflow:]
        ids = ctx + [SEP_CONTEXT_EVIDENCE] + ev + [SEP_EVIDENCE_CANDIDATE] + cand
    return ids, len(ids)


class EvidenceChoiceDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        docs: list[CorpusDoc],
        evidence_kind: str,
        max_context_bytes: int,
        max_evidence_bytes: int,
        max_candidate_bytes: int,
        max_total_bytes: int,
    ):
        self.records = records
        self.docs = docs
        self.evidence_kind = evidence_kind
        self.max_context_bytes = max_context_bytes
        self.max_evidence_bytes = max_evidence_bytes
        self.max_candidate_bytes = max_candidate_bytes
        self.max_total_bytes = max_total_bytes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        doc_ids = rec["evidence_doc_ids"].get(self.evidence_kind, [])
        ev = evidence_text(doc_ids, self.docs)
        encoded = [
            serialize_triple(
                rec["context"],
                ev,
                choice,
                self.max_context_bytes,
                self.max_evidence_bytes,
                self.max_candidate_bytes,
                self.max_total_bytes,
            )
            for choice in rec["choices"]
        ]
        return {
            "id": rec["id"],
            "dataset": rec["dataset"],
            "input_ids": [x[0] for x in encoded],
            "lengths": [x[1] for x in encoded],
            "label": int(rec["label"]),
            "teacher_scores": rec.get("teacher_scores"),
            "n_choices": len(rec["choices"]),
            "choices": rec["choices"],
        }


def collate_choice_batch(batch: list[dict]) -> dict:
    rows: list[list[int]] = []
    lengths: list[int] = []
    group_sizes: list[int] = []
    labels: list[int] = []
    teacher_scores: list[list[float] | None] = []
    datasets: list[str] = []
    ids: list[str] = []
    max_len = 0
    for item in batch:
        group_sizes.append(item["n_choices"])
        labels.append(item["label"])
        teacher_scores.append(item.get("teacher_scores"))
        datasets.append(item["dataset"])
        ids.append(item["id"])
        for row, length in zip(item["input_ids"], item["lengths"]):
            rows.append(row)
            lengths.append(length)
            max_len = max(max_len, len(row))
    # Keep patch sampling aligned.
    if max_len % 4:
        max_len += 4 - (max_len % 4)
    x = torch.full((len(rows), max_len), PAD_BYTE, dtype=torch.long)
    for i, row in enumerate(rows):
        x[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return {
        "input_ids": x,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "group_sizes": group_sizes,
        "labels": torch.tensor(labels, dtype=torch.long),
        "teacher_scores": teacher_scores,
        "datasets": datasets,
        "ids": ids,
    }


class EvidenceNativeJudge(nn.Module):
    def __init__(
        self,
        codec: SemanticCodec,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_mult: float,
        freeze_codec: bool,
        max_patches: int,
    ):
        super().__init__()
        self.codec = codec
        self.freeze_codec = freeze_codec
        if freeze_codec:
            for param in self.codec.encoder.parameters():
                param.requires_grad_(False)
            for param in self.codec.alignment_head.parameters():
                param.requires_grad_(False)
        self.projection = PatchProjection(codec.cfg.codec_dim, d_model)
        cfg = S0Config(
            patch_size=codec.cfg.patch_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_mult=ffn_mult,
            max_seq_len=max_patches,
            dropout=0.0,
        )
        self.reasoner = GlobalReasoner(cfg)
        self.pool_norm = nn.LayerNorm(d_model * 2)
        self.judgment_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, byte_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.freeze_codec:
            with torch.no_grad():
                patch_hidden = self.codec.encoder.get_patch_states(byte_ids)
        else:
            patch_hidden = self.codec.encoder.get_patch_states(byte_ids)
        patch_states = self.projection(patch_hidden)
        hidden = self.reasoner(patch_states)
        patch_lens = torch.div(lengths + self.codec.cfg.patch_size - 1, self.codec.cfg.patch_size, rounding_mode="floor")
        patch_lens = patch_lens.clamp(min=1, max=hidden.shape[1])
        idx = (patch_lens - 1).to(hidden.device)
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        last = hidden[batch_idx, idx]
        mask = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0) < patch_lens.unsqueeze(1).to(hidden.device)
        masked = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        mean = masked.sum(dim=1) / patch_lens.to(hidden.device).unsqueeze(-1).clamp(min=1)
        pooled = self.pool_norm(torch.cat([last, mean], dim=-1))
        return self.judgment_head(pooled).squeeze(-1)

    def count_parameters(self) -> dict[str, int]:
        return {
            "codec_encoder": sum(p.numel() for p in self.codec.encoder.parameters()),
            "projection": sum(p.numel() for p in self.projection.parameters()),
            "reasoner": sum(p.numel() for p in self.reasoner.parameters()),
            "head": sum(p.numel() for p in self.judgment_head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[EvidenceNativeJudge, dict]:
    if args.codec_checkpoint and Path(args.codec_checkpoint).exists():
        codec, manifest = load_codec_checkpoint(args.codec_checkpoint, device)
    else:
        cfg = CodecConfig()
        codec = SemanticCodec(cfg, d_model=args.d_model).to(device)
        manifest = {"path": None, "warning": "random codec: checkpoint not found"}
    model = EvidenceNativeJudge(
        codec=codec,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        ffn_mult=args.ffn_mult,
        freeze_codec=not args.finetune_codec,
        max_patches=max(16, math.ceil(args.max_total_bytes / codec.cfg.patch_size)),
    ).to(device)
    return model, manifest


def group_ce_loss(
    scores: torch.Tensor,
    group_sizes: list[int],
    labels: torch.Tensor,
    teacher_scores: list[list[float] | None] | None = None,
    teacher_alpha: float = 0.0,
    teacher_temperature: float = 1.0,
) -> tuple[torch.Tensor, int, int]:
    losses = []
    correct = 0
    offset = 0
    for i, g in enumerate(group_sizes):
        logits = scores[offset : offset + g].unsqueeze(0)
        label = labels[i].view(1).to(scores.device)
        gold_loss = F.cross_entropy(logits, label)
        loss_i = gold_loss
        if teacher_alpha > 0.0 and teacher_scores is not None and teacher_scores[i] is not None:
            raw = torch.tensor(teacher_scores[i], dtype=torch.float32, device=scores.device)
            if raw.numel() == g and torch.isfinite(raw).all():
                target = F.softmax(raw / max(teacher_temperature, 1e-6), dim=0)
                log_probs = F.log_softmax(logits.squeeze(0), dim=0)
                teacher_loss = F.kl_div(log_probs, target, reduction="sum")
                loss_i = (1.0 - teacher_alpha) * gold_loss + teacher_alpha * teacher_loss
        losses.append(loss_i)
        correct += int(logits.argmax(dim=1).item() == int(label.item()))
        offset += g
    loss = torch.stack(losses).mean() if losses else scores.sum() * 0.0
    return loss, correct, len(group_sizes)


def train_one_model(
    args: argparse.Namespace,
    model: EvidenceNativeJudge,
    train_records: list[dict],
    docs: list[CorpusDoc],
    device: torch.device,
    evidence_kind: str,
) -> dict:
    dataset = EvidenceChoiceDataset(
        train_records,
        docs,
        evidence_kind=evidence_kind,
        max_context_bytes=args.max_context_bytes,
        max_evidence_bytes=args.max_evidence_bytes,
        max_candidate_bytes=args.max_candidate_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_choice_batch)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "float16"))
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model.train()
    logs: list[dict] = []
    global_step = 0
    total_steps = max(1, args.epochs * len(loader))
    started = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch in loader:
            global_step += 1
            progress = min(1.0, global_step / max(1, args.warmup_steps))
            lr = args.lr * progress
            if global_step > args.warmup_steps:
                decay = (global_step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
                lr = args.min_lr + 0.5 * (1.0 + math.cos(math.pi * decay)) * (args.lr - args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            x = batch["input_ids"].to(device, non_blocking=True)
            lengths = batch["lengths"].to(device, non_blocking=True)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                scores = model(x, lengths)
                loss, correct, total = group_ce_loss(scores, batch["group_sizes"], labels, batch.get("teacher_scores"), args.teacher_alpha, args.teacher_temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not math.isfinite(float(grad_norm.item())):
                raise RuntimeError(f"non-finite grad norm at step {global_step}: {grad_norm}")
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item()) * total
            epoch_correct += correct
            epoch_total += total
        logs.append({
            "epoch": epoch + 1,
            "loss": epoch_loss / max(1, epoch_total),
            "train_accuracy": epoch_correct / max(1, epoch_total),
            "elapsed_s": round(time.time() - started, 1),
        })
        if args.progress:
            row = logs[-1]
            print(f"epoch {row['epoch']} | loss {row['loss']:.4f} | train_acc {row['train_accuracy']:.3f}")
    return {"evidence_kind": evidence_kind, "logs": logs}


@torch.no_grad()
def evaluate_model(
    args: argparse.Namespace,
    model: EvidenceNativeJudge,
    records: list[dict],
    docs: list[CorpusDoc],
    device: torch.device,
    evidence_kind: str,
) -> dict:
    dataset = EvidenceChoiceDataset(
        records,
        docs,
        evidence_kind=evidence_kind,
        max_context_bytes=args.max_context_bytes,
        max_evidence_bytes=args.max_evidence_bytes,
        max_candidate_bytes=args.max_candidate_bytes,
        max_total_bytes=args.max_total_bytes,
    )
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_choice_batch)
    model.eval()
    correct_by_dataset: dict[str, int] = {}
    total_by_dataset: dict[str, int] = {}
    predictions: list[dict] = []
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    for batch in loader:
        x = batch["input_ids"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            scores = model(x, lengths).float().cpu()
        offset = 0
        for i, g in enumerate(batch["group_sizes"]):
            group_scores = scores[offset : offset + g]
            pred = int(group_scores.argmax().item())
            gold = int(batch["labels"][i].item())
            ds = batch["datasets"][i]
            correct_by_dataset[ds] = correct_by_dataset.get(ds, 0) + int(pred == gold)
            total_by_dataset[ds] = total_by_dataset.get(ds, 0) + 1
            if len(predictions) < args.prediction_samples:
                predictions.append({
                    "id": batch["ids"][i],
                    "dataset": ds,
                    "pred": pred,
                    "label": gold,
                    "scores": [round(float(s), 4) for s in group_scores.tolist()],
                })
            offset += g
    total_correct = sum(correct_by_dataset.values())
    total = sum(total_by_dataset.values())
    return {
        "evidence_kind": evidence_kind,
        "overall": total_correct / max(1, total),
        "by_dataset": {
            ds: {
                "accuracy": correct_by_dataset.get(ds, 0) / max(1, n),
                "correct": correct_by_dataset.get(ds, 0),
                "total": n,
            }
            for ds, n in sorted(total_by_dataset.items())
        },
        "prediction_samples": predictions,
    }


def unigram_frequency_scores(records: list[dict], docs: list[CorpusDoc]) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for doc in docs:
        for tok in text_tokens(doc.text):
            counts[tok] = counts.get(tok, 0) + 1
            total += 1
    vocab = max(1, len(counts))
    out: dict[str, float] = {}
    for rec in records:
        scores = []
        for choice in rec["choices"]:
            toks = text_tokens(choice)
            if not toks:
                scores.append(-1e9)
                continue
            score = sum(math.log((counts.get(tok, 0) + 1) / (total + vocab)) for tok in toks) / len(toks)
            scores.append(score)
        out[rec["id"]] = float(max(range(len(scores)), key=lambda i: scores[i]))
    return out


def overlap_score(a: str, b: str) -> float:
    ta = set(text_tokens(a))
    tb = set(text_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


def nearest_neighbor_predictions(train_records: list[dict], eval_records: list[dict]) -> dict[str, int]:
    train_tokens = [(set(text_tokens(r["context"])), int(r["label"]), len(r["choices"])) for r in train_records]
    preds: dict[str, int] = {}
    for rec in eval_records:
        q = set(text_tokens(rec["context"]))
        best_score = -1.0
        best_label = 0
        for toks, label, n_choices in train_tokens:
            if label >= len(rec["choices"]):
                continue
            denom = len(q | toks) or 1
            score = len(q & toks) / denom
            if score > best_score:
                best_score = score
                best_label = label
        preds[rec["id"]] = int(best_label)
    return preds


def score_prediction_map(records: list[dict], pred_map: dict[str, int]) -> dict:
    by_ds: dict[str, list[int]] = {}
    for rec in records:
        pred = int(pred_map.get(rec["id"], 0))
        by_ds.setdefault(rec["dataset"], []).append(int(pred == int(rec["label"])))
    total = sum(sum(v) for v in by_ds.values())
    n = sum(len(v) for v in by_ds.values())
    return {
        "overall": total / max(1, n),
        "by_dataset": {
            ds: {"accuracy": sum(vals) / max(1, len(vals)), "correct": sum(vals), "total": len(vals)}
            for ds, vals in sorted(by_ds.items())
        },
    }


def dumb_baselines(train_records: list[dict], eval_records: list[dict], docs: list[CorpusDoc]) -> dict:
    baselines: dict[str, dict] = {}
    majority_by_ds: dict[str, int] = {}
    for ds in sorted({r["dataset"] for r in train_records}):
        labels = [int(r["label"]) for r in train_records if r["dataset"] == ds]
        majority_by_ds[ds] = max(set(labels), key=labels.count) if labels else 0
    baselines["majority_label"] = score_prediction_map(
        eval_records,
        {r["id"]: majority_by_ds.get(r["dataset"], 0) for r in eval_records},
    )
    baselines["shortest_candidate"] = score_prediction_map(
        eval_records,
        {r["id"]: min(range(len(r["choices"])), key=lambda i: len(r["choices"][i])) for r in eval_records},
    )
    baselines["unigram_frequency"] = score_prediction_map(eval_records, unigram_frequency_scores(eval_records, docs))
    baselines["nearest_neighbor_train_label"] = score_prediction_map(
        eval_records,
        nearest_neighbor_predictions(train_records, eval_records),
    )

    bm25_preds = {}
    for rec in eval_records:
        ev = evidence_text(rec["evidence_doc_ids"].get("retrieved", []), docs)
        scores = [overlap_score(choice, ev) + 0.25 * overlap_score(rec["context"] + " " + choice, ev) for choice in rec["choices"]]
        bm25_preds[rec["id"]] = int(max(range(len(scores)), key=lambda i: scores[i]))
    baselines["bm25_evidence_overlap_ranker"] = score_prediction_map(eval_records, bm25_preds)
    return baselines


def run_all(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)
    payload = load_or_prepare_records(args)
    docs = [CorpusDoc(**d) for d in payload["docs"]]
    train_records = payload["train_records"]
    eval_records = payload["eval_records"]
    model, codec_manifest = build_model(args, device)
    model_manifest = {
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "kv_heads": args.kv_heads,
        "ffn_mult": args.ffn_mult,
        "freeze_codec": not args.finetune_codec,
        "param_counts": model.count_parameters(),
        "codec": codec_manifest,
    }
    if args.progress:
        print(json.dumps(model_manifest["param_counts"], indent=2))

    train_result = train_one_model(args, model, train_records, docs, device, evidence_kind=args.train_evidence_kind)
    evals = {}
    for kind in ["retrieved", "none", "shuffled", "wrong_topic", "gold"]:
        evals[kind] = evaluate_model(args, model, eval_records, docs, device, kind)
    baselines = dumb_baselines(train_records, eval_records, docs)

    retrieved = evals["retrieved"]["overall"]
    none = evals["none"]["overall"]
    best_dumb = max(v["overall"] for v in baselines.values()) if baselines else 0.0
    shuffled = evals["shuffled"]["overall"]
    gates = {
        "evidence_native_beats_closed_book_by_ge_5pp": (retrieved - none) >= 0.05,
        "evidence_native_beats_dumb_baselines_by_ge_3pp": (retrieved - best_dumb) >= 0.03,
        "shuffled_evidence_much_worse_than_retrieved": (retrieved - shuffled) >= 0.03,
        "retrieved_minus_no_evidence_pp": round((retrieved - none) * 100, 2),
        "retrieved_minus_best_dumb_pp": round((retrieved - best_dumb) * 100, 2),
        "retrieved_minus_shuffled_pp": round((retrieved - shuffled) * 100, 2),
        "best_dumb_baseline": max(baselines.items(), key=lambda kv: kv[1]["overall"])[0] if baselines else None,
    }
    out = {
        "run": {
            "seed": args.seed,
            "device": str(device),
            "output_dir": args.output_dir,
            "train_evidence_kind": args.train_evidence_kind,
        },
        "data": {
            "train_records": len(train_records),
            "eval_records": len(eval_records),
            "corpus_manifest": payload["corpus_manifest"],
            "leakage_audit": payload["leakage_audit"],
            "controls": payload.get("controls"),
            "teacher_cache": payload["teacher_cache"],
        },
        "model": model_manifest,
        "train": train_result,
        "eval": evals,
        "dumb_baselines": baselines,
        "gates": gates,
        "chain_init_baseline_note": {
            "status": "not_rerun_in_this_script",
            "batch5_artifacts": [
                "C:/sutra_fast/chain_init_probe/chain_init_token_end_layers4.json",
                "C:/sutra_fast/chain_init_probe/chain_init_patch_boundary_layers4.json",
            ],
            "reason": "Batch 5 chain-init probe measures token-space NLL compatibility, not evidence-conditioned MCQ accuracy.",
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    ckpt_path = out_dir / "evidence_judge.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_manifest": model_manifest,
        "args": vars(args),
        "metrics_path": str(metrics_path),
    }, ckpt_path)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "train_eval"], default="train_eval")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codec-checkpoint", default=DEFAULT_CODEC)
    parser.add_argument("--teacher-cache", default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--reuse-records", action="store_true")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--no-randomize-choices", action="store_true")

    parser.add_argument("--train-hellaswag", type=int, default=512)
    parser.add_argument("--train-piqa", type=int, default=512)
    parser.add_argument("--eval-hellaswag", type=int, default=1024)
    parser.add_argument("--eval-piqa", type=int, default=1024)
    parser.add_argument("--max-corpus-docs", type=int, default=2200)
    parser.add_argument("--shard-docs", type=int, default=1200)
    parser.add_argument("--shard-bytes", type=int, default=32_000_000)
    parser.add_argument("--top-k", type=int, default=3)

    parser.add_argument("--max-context-bytes", type=int, default=192)
    parser.add_argument("--max-evidence-bytes", type=int, default=448)
    parser.add_argument("--max-candidate-bytes", type=int, default=128)
    parser.add_argument("--max-total-bytes", type=int, default=768)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ffn-mult", type=float, default=2.0)
    parser.add_argument("--finetune-codec", action="store_true")

    parser.add_argument("--train-evidence-kind", choices=["retrieved", "gold", "none"], default="retrieved")
    parser.add_argument("--teacher-alpha", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prediction-samples", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        payload = load_or_prepare_records(args)
        print(json.dumps({
            "train_records": len(payload["train_records"]),
            "eval_records": len(payload["eval_records"]),
            "corpus_manifest": payload["corpus_manifest"],
            "leakage_audit": payload["leakage_audit"],
            "controls": payload.get("controls"),
            "teacher_cache": payload["teacher_cache"],
        }, indent=2))
        return
    metrics = run_all(args)
    print(json.dumps({
        "retrieved_overall": metrics["eval"]["retrieved"]["overall"],
        "none_overall": metrics["eval"]["none"]["overall"],
        "shuffled_overall": metrics["eval"]["shuffled"]["overall"],
        "wrong_topic_overall": metrics["eval"]["wrong_topic"]["overall"],
        "gold_overall": metrics["eval"]["gold"]["overall"],
        "best_dumb": metrics["gates"]["best_dumb_baseline"],
        "gates": metrics["gates"],
        "metrics_path": str(Path(args.output_dir) / "metrics.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
