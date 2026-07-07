"""Evidence-Native v1: factorized evidence-conditioned judge.

Separate context, evidence, and candidate encoding; candidate/context cross-attend
to evidence; matched M_evidence and M_none controls are run by the suite mode.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_native_sutra import (  # noqa: E402
    BM25Retriever,
    CorpusDoc,
    DEFAULT_CODEC,
    DEFAULT_DATA_DIR,
    PAD_BYTE,
    PASSAGE_SEPARATOR_TEXT,
    choose_device,
    docs_from_diverse_shards,
    ensure_offline,
    evidence_text,
    load_codec_checkpoint,
    load_limited_examples,
    normalize_text,
    overlap_score,
    set_seed,
    sha256_texts,
    text_tokens,
)
from s0_architecture import GlobalReasoner, S0Config  # noqa: E402
from semantic_codec import CodecConfig, PatchProjection, SemanticCodec  # noqa: E402

DEFAULT_OUTPUT_DIR = "C:/sutra_fast/evidence_native_v1"
DEFAULT_FALLBACK_OUTPUT_DIR = "tmp_evidence_native_v1"
DEFAULT_SEEDS = "20260707,42,12345"
EVAL_KINDS = ["retrieved", "none", "shuffled", "wrong_topic", "gold"]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_output_dir(requested: str, fallback: str) -> tuple[Path, dict]:
    requested_path = Path(requested)
    meta = {"requested_output_dir": requested, "fallback_used": False, "write_error": None}
    try:
        requested_path.mkdir(parents=True, exist_ok=True)
        probe = requested_path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        meta["actual_output_dir"] = str(requested_path)
        return requested_path, meta
    except Exception as exc:  # noqa: BLE001
        actual = Path(fallback)
        actual.mkdir(parents=True, exist_ok=True)
        meta.update({"fallback_used": True, "write_error": f"{type(exc).__name__}: {exc}", "actual_output_dir": str(actual)})
        return actual, meta


def parse_seeds(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def enc(text: str, max_bytes: int, keep_tail: bool = False) -> tuple[list[int], int]:
    vals = list(text.encode("utf-8", errors="replace"))
    if len(vals) > max_bytes:
        vals = vals[-max_bytes:] if keep_tail else vals[:max_bytes]
    return vals, len(vals)


def rationale(ex: dict, compact: bool = False) -> str:
    label = int(ex["label"])
    correct = ex["choices"][label]
    wrongs = "; ".join(c for i, c in enumerate(ex["choices"]) if i != label)[:360]
    if compact:
        return f"Correct answer: {correct}. Context: {ex['context']} Less supported alternatives: {wrongs}."
    return (
        f"The supported answer is: {correct}. It best continues or solves the situation in the context: "
        f"{ex['context']} The alternatives are less supported: {wrongs}."
    )


def gold_paraphrase(rec: dict) -> str:
    return f"This passage favors '{rec['choices'][int(rec['label'])]}' as the coherent answer for the described situation."


def gold_masked(rec: dict) -> str:
    text = rec.get("gold_evidence", "")
    correct = rec["choices"][int(rec["label"])]
    masked = re.sub(re.escape(correct), "[MASKED_CORRECT_OPTION]", text, flags=re.IGNORECASE)
    return masked if masked != text else text.replace(correct.split()[0], "[MASKED_TOKEN]", 1)


def counterfactual(rec: dict) -> tuple[str, int]:
    label = int(rec["label"])
    wrong = next((i for i in range(len(rec["choices"])) if i != label), label)
    return f"Counterfactual evidence: the supported answer is '{rec['choices'][wrong]}', not the original answer.", wrong


def benchmark_ngrams(examples: list[dict], n: int = 12) -> set[str]:
    grams: set[str] = set()
    for ex in examples:
        for text in [ex["context"], *ex["choices"]]:
            toks = text_tokens(text)
            for i in range(max(0, len(toks) - n + 1)):
                grams.add(" ".join(toks[i : i + n]))
    return grams


def contaminated(doc: CorpusDoc, grams: set[str], n: int = 12) -> bool:
    toks = text_tokens(doc.text)
    if len(toks) < n:
        return False
    for i in range(len(toks) - n + 1):
        if " ".join(toks[i : i + n]) in grams:
            return True
    return False


def clean_external_docs(docs: list[CorpusDoc], train: list[dict], evals: list[dict]) -> tuple[list[CorpusDoc], dict]:
    grams = benchmark_ngrams(train + evals)
    kept: list[CorpusDoc] = []
    samples: list[dict] = []
    removed = 0
    for doc in docs:
        if contaminated(doc, grams):
            removed += 1
            if len(samples) < 10:
                samples.append({"source": doc.source, "preview": doc.text[:180]})
        else:
            doc.source = "external_shard:common-pile/wikimedia_stackexchange_gutenberg_news"
            kept.append(doc)
    for i, doc in enumerate(kept):
        doc.doc_id = i
    return kept, {"input_docs": len(docs), "kept_docs": len(kept), "removed_docs": removed, "rule": "12-gram set intersection against train and eval examples", "removed_samples": samples}
def load_examples(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    rand = not args.no_randomize_choices
    train = []
    train += load_limited_examples("hellaswag", "train", args.train_hellaswag, args.seed, args.allow_downloads, rand)
    train += load_limited_examples("piqa", "train", args.train_piqa, args.seed + 1, args.allow_downloads, rand)
    evals = []
    evals += load_limited_examples("hellaswag", "validation", args.eval_hellaswag, args.seed + 2, args.allow_downloads, rand)
    evals += load_limited_examples("piqa", "validation", args.eval_piqa, args.seed + 3, args.allow_downloads, rand)
    return train, evals


def build_corpus(train: list[dict], evals: list[dict], args: argparse.Namespace) -> tuple[list[CorpusDoc], dict]:
    raw = docs_from_diverse_shards(args.data_dir, args.shard_docs, args.shard_bytes)
    clean, decon = clean_external_docs(raw, train, evals)
    docs = clean[: args.max_external_docs]
    rationales = []
    if args.include_train_rationales_in_corpus:
        for ex in train[: args.max_rationale_docs]:
            rationales.append(CorpusDoc(-1, f"teacher_rationale:{ex['dataset']}:train", rationale(ex, args.rationale_mode == "compact")))
    docs.extend(rationales)
    docs = docs[: args.max_corpus_docs]
    for i, doc in enumerate(docs):
        doc.doc_id = i
    manifest = {
        "n_docs": len(docs),
        "sha256_normalized": sha256_texts(d.text for d in docs),
        "sources": sorted({d.source for d in docs}),
        "external_docs_kept_before_limit": len(clean),
        "rationale_docs_added": len(rationales),
        "max_corpus_docs": args.max_corpus_docs,
        "data_dir": args.data_dir,
        "shard_docs": args.shard_docs,
        "shard_bytes": args.shard_bytes,
        "decontamination": decon,
        "benchmark_train_as_corpus": False,
        "benchmark_choices_as_corpus": False,
    }
    return docs, manifest


def teacher_scores(rec: dict) -> list[float]:
    ev = rec.get("gold_evidence", "")
    vals = []
    for i, choice in enumerate(rec["choices"]):
        score = overlap_score(choice, ev) + 0.15 * overlap_score(rec["context"] + " " + choice, ev)
        if i == int(rec["label"]):
            score += 1.0
        vals.append(float(score))
    return vals


def assign_shuffled(records: list[dict], top_k: int, rng: random.Random, n_docs: int) -> None:
    by_ds: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        by_ds.setdefault(rec["dataset"], []).append(i)
    for inds in by_ds.values():
        if len(inds) > 1:
            recv = inds[:]
            rng.shuffle(recv)
            donors = recv[1:] + recv[:1]
            for r, d in zip(recv, donors):
                records[r]["evidence_doc_ids"]["shuffled"] = list(records[d]["evidence_doc_ids"].get("retrieved", []))[:top_k]
        elif inds:
            records[inds[0]]["evidence_doc_ids"]["shuffled"] = rng.sample(range(n_docs), k=min(top_k, n_docs)) if n_docs else []


def prepare_records(args: argparse.Namespace, out_dir: Path) -> dict:
    ensure_offline(args.allow_downloads)
    train, evals = load_examples(args)
    docs, manifest = build_corpus(train, evals, args)
    retriever = BM25Retriever(docs)
    rng = random.Random(args.seed + 910)

    def enrich(examples: list[dict], split: str) -> list[dict]:
        records = []
        for i, ex in enumerate(examples):
            ret = [doc_id for doc_id, _ in retriever.top_k(ex["context"], args.top_k)]
            if len(ret) < args.top_k:
                ret.extend(retriever.random_docs(args.top_k - len(ret), rng))
            wrong = retriever.low_overlap_random(ex["context"], args.top_k, rng)
            rec = {"id": f"{split}-{i:06d}", "dataset": ex["dataset"], "split": ex["split"], "context": ex["context"], "choices": ex["choices"], "label": int(ex["label"]), "choice_order": ex.get("choice_order"), "evidence_doc_ids": {"retrieved": ret[: args.top_k], "shuffled": [], "wrong_topic": wrong[: args.top_k], "none": []}, "gold_evidence": rationale(ex, args.rationale_mode == "compact")}
            rec["gold_paraphrase"] = gold_paraphrase(rec)
            rec["gold_masked"] = gold_masked(rec)
            cf, cf_label = counterfactual(rec)
            rec["counterfactual_evidence"] = cf
            rec["counterfactual_label"] = cf_label
            rec["teacher_scores"] = teacher_scores(rec)
            records.append(rec)
        return records

    train_records = enrich(train, "train")
    eval_records = enrich(evals, "eval")
    assign_shuffled(train_records, args.top_k, rng, len(docs))
    assign_shuffled(eval_records, args.top_k, rng, len(docs))
    controls = {"candidate_order_randomized": not args.no_randomize_choices, "corpus_policy": "external shards plus train teacher rationales, no raw benchmark train examples", "shuffled_evidence": "retrieved doc ids deranged within dataset", "wrong_topic_evidence": "low-overlap random docs", "gold_evidence": "oracle teacher/template rationale, not label-conditioned BM25", "teacher_scores": "available for all records", "passage_separator": PASSAGE_SEPARATOR_TEXT.strip()}
    payload = {"created_at_unix": time.time(), "args": vars(args), "corpus_manifest": manifest, "leakage_audit": {"eval_examples": len(evals), "note": "external docs decontaminated against train and eval before adding train-only rationale docs"}, "controls": controls, "docs": [asdict(d) for d in docs], "train_records": train_records, "eval_records": eval_records}
    dump_json(out_dir / "evidence_records.json", payload)
    dump_json(out_dir / "corpus_manifest.json", {"corpus_manifest": manifest, "controls": controls, "leakage_audit": payload["leakage_audit"]})
    return payload


def load_or_prepare(args: argparse.Namespace, out_dir: Path) -> dict:
    path = out_dir / "evidence_records.json"
    if args.reuse_records and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return prepare_records(args, out_dir)


def ev_text(rec: dict, docs: list[CorpusDoc], kind: str) -> str:
    if kind == "none":
        return ""
    if kind == "gold":
        return rec.get("gold_evidence", "")
    if kind == "gold_paraphrase":
        return rec.get("gold_paraphrase", "")
    if kind == "gold_masked":
        return rec.get("gold_masked", "")
    if kind == "counterfactual":
        return rec.get("counterfactual_evidence", "")
    return evidence_text(rec["evidence_doc_ids"].get(kind, []), docs)


class FactDataset(Dataset):
    def __init__(self, records: list[dict], docs: list[CorpusDoc], kind: str, args: argparse.Namespace, train_mode: bool = False, seed: int = 0):
        self.records, self.docs, self.kind, self.args = records, docs, kind, args
        self.train_mode = train_mode
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        kind = self.kind
        if self.train_mode and kind != "none" and self.args.evidence_dropout > 0 and self.rng.random() < self.args.evidence_dropout:
            kind = "none"
        cids, clen = enc(rec["context"], self.args.max_context_bytes, keep_tail=True)
        eids, elen = enc(ev_text(rec, self.docs, kind), self.args.max_evidence_bytes)
        choices = []
        for choice in rec["choices"]:
            ids, ln = enc(choice, self.args.max_candidate_bytes)
            choices.append((ids, ln))
        return {"id": rec["id"], "dataset": rec["dataset"], "context_ids": cids, "context_len": clen, "evidence_ids": eids, "evidence_len": elen, "choices": choices, "label": int(rec["label"]), "teacher_scores": rec.get("teacher_scores"), "raw_choices": rec["choices"], "counterfactual_label": rec.get("counterfactual_label")}


def round4(n: int) -> int:
    n = max(4, n)
    return n + ((4 - n % 4) % 4)


def pad(rows: list[list[int]]) -> torch.Tensor:
    max_len = max((round4(len(r)) for r in rows), default=4)
    x = torch.full((len(rows), max_len), PAD_BYTE, dtype=torch.long)
    for i, row in enumerate(rows):
        if row:
            x[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return x


def collate(batch: list[dict]) -> dict:
    ctx_rows, ev_rows, cand_rows = [], [], []
    ctx_lens, ev_lens, cand_lens = [], [], []
    groups, labels, t_scores, ids, dsets = [], [], [], [], []
    for item in batch:
        groups.append(len(item["choices"]))
        labels.append(item["label"])
        t_scores.append(item.get("teacher_scores"))
        ids.append(item["id"])
        dsets.append(item["dataset"])
        for ids_cand, len_cand in item["choices"]:
            ctx_rows.append(item["context_ids"]); ctx_lens.append(item["context_len"])
            ev_rows.append(item["evidence_ids"]); ev_lens.append(item["evidence_len"])
            cand_rows.append(ids_cand); cand_lens.append(len_cand)
    return {"context_ids": pad(ctx_rows), "evidence_ids": pad(ev_rows), "candidate_ids": pad(cand_rows), "context_lens": torch.tensor(ctx_lens), "evidence_lens": torch.tensor(ev_lens), "candidate_lens": torch.tensor(cand_lens), "group_sizes": groups, "labels": torch.tensor(labels), "teacher_scores": t_scores, "ids": ids, "datasets": dsets}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp(min=1.0)


class SegmentEncoder(nn.Module):
    def __init__(self, codec: SemanticCodec, args: argparse.Namespace, max_patches: int):
        super().__init__()
        self.codec = codec
        self.freeze_codec = not args.finetune_codec
        if self.freeze_codec:
            for p in self.codec.encoder.parameters():
                p.requires_grad_(False)
            for p in self.codec.alignment_head.parameters():
                p.requires_grad_(False)
        self.proj = PatchProjection(codec.cfg.codec_dim, args.d_model)
        cfg = S0Config(patch_size=codec.cfg.patch_size, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, n_kv_heads=args.kv_heads, ffn_mult=args.ffn_mult, max_seq_len=max_patches, dropout=0.0)
        self.reasoner = GlobalReasoner(cfg)
        self.null_state = nn.Parameter(torch.zeros(args.d_model))

    def forward(self, ids: torch.Tensor, lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.freeze_codec:
            with torch.no_grad():
                ph = self.codec.encoder.get_patch_states(ids)
        else:
            ph = self.codec.encoder.get_patch_states(ids)
        h = self.reasoner(self.proj(ph))
        plens = torch.div(lens + self.codec.cfg.patch_size - 1, self.codec.cfg.patch_size, rounding_mode="floor").clamp(min=0, max=h.shape[1])
        pos = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
        mask = pos < plens.to(h.device).unsqueeze(1)
        empty = plens.to(h.device) == 0
        if empty.any():
            h = h.clone(); mask = mask.clone()
            h[empty, 0, :] = self.null_state.to(h.dtype)
            mask[empty, 0] = True
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        return h, mask, masked_mean(h, mask)


class FactorizedJudge(nn.Module):
    def __init__(self, codec: SemanticCodec, args: argparse.Namespace):
        super().__init__()
        max_bytes = max(args.max_context_bytes, args.max_evidence_bytes, args.max_candidate_bytes, 4)
        self.segment = SegmentEncoder(codec, args, max(16, math.ceil(max_bytes / codec.cfg.patch_size)))
        d = args.d_model
        self.cand_ev = nn.MultiheadAttention(d, args.heads, batch_first=True)
        self.ctx_ev = nn.MultiheadAttention(d, args.heads, batch_first=True)
        self.cand_ctx = nn.MultiheadAttention(d, args.heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(d * 10), nn.Linear(d * 10, d * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, batch: dict) -> torch.Tensor:
        ch, cm, cp = self.segment(batch["context_ids"], batch["context_lens"])
        eh, em, ep = self.segment(batch["evidence_ids"], batch["evidence_lens"])
        ah, am, ap = self.segment(batch["candidate_ids"], batch["candidate_lens"])
        ae, _ = self.cand_ev(ah, eh, eh, key_padding_mask=~em)
        ce, _ = self.ctx_ev(ch, eh, eh, key_padding_mask=~em)
        ac, _ = self.cand_ctx(ah, ch, ch, key_padding_mask=~cm)
        aep = masked_mean(ae, am)
        cep = masked_mean(ce, cm)
        acp = masked_mean(ac, am)
        feat = torch.cat([cp, ep, ap, aep, cep, acp, torch.abs(ap - cp), ap * cp, aep * ap, aep * ep], dim=-1)
        return self.head(feat).squeeze(-1)

    def count_parameters(self) -> dict[str, int]:
        codec = self.segment.codec
        cross = sum(p.numel() for p in self.cand_ev.parameters()) + sum(p.numel() for p in self.ctx_ev.parameters()) + sum(p.numel() for p in self.cand_ctx.parameters())
        return {"codec_encoder": sum(p.numel() for p in codec.encoder.parameters()), "segment_projection": sum(p.numel() for p in self.segment.proj.parameters()), "segment_reasoner": sum(p.numel() for p in self.segment.reasoner.parameters()), "cross_attention": cross, "head": sum(p.numel() for p in self.head.parameters()), "total": sum(p.numel() for p in self.parameters()), "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}


def build_model(args: argparse.Namespace, device: torch.device) -> tuple[FactorizedJudge, dict]:
    if args.codec_checkpoint and Path(args.codec_checkpoint).exists():
        codec, manifest = load_codec_checkpoint(args.codec_checkpoint, device)
    else:
        codec = SemanticCodec(CodecConfig(), d_model=args.d_model).to(device)
        manifest = {"path": None, "warning": "random codec: checkpoint not found"}
    model = FactorizedJudge(codec, args).to(device)
    return model, manifest


def to_device(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ["context_ids", "evidence_ids", "candidate_ids", "context_lens", "evidence_lens", "candidate_lens", "labels"]:
        out[key] = batch[key].to(device, non_blocking=True)
    return out


def group_loss(scores: torch.Tensor, groups: list[int], labels: torch.Tensor, teacher: list[list[float] | None], args: argparse.Namespace) -> tuple[torch.Tensor, int, int]:
    losses, correct, offset = [], 0, 0
    for i, g in enumerate(groups):
        logits = scores[offset : offset + g].unsqueeze(0)
        label = labels[i].view(1).to(scores.device)
        loss = F.cross_entropy(logits, label)
        if args.teacher_alpha > 0 and teacher[i] is not None:
            raw = torch.tensor(teacher[i], dtype=torch.float32, device=scores.device)
            if raw.numel() == g and torch.isfinite(raw).all():
                target = F.softmax(raw / max(args.teacher_temperature, 1e-6), dim=0)
                loss_t = F.kl_div(F.log_softmax(logits.squeeze(0), dim=0), target, reduction="sum")
                loss = (1 - args.teacher_alpha) * loss + args.teacher_alpha * loss_t
        losses.append(loss)
        correct += int(logits.argmax(1).item() == int(label.item()))
        offset += g
    return torch.stack(losses).mean(), correct, len(groups)


def train_model(args: argparse.Namespace, model: FactorizedJudge, records: list[dict], docs: list[CorpusDoc], device: torch.device, kind: str, seed: int) -> dict:
    ds = FactDataset(records, docs, kind, args, train_mode=True, seed=seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "float16"))
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    total_steps = max(1, args.epochs * len(loader)); step = 0; logs = []; start = time.time(); model.train()
    for epoch in range(args.epochs):
        ep_loss = ep_correct = ep_total = 0
        for raw in loader:
            step += 1
            lr = args.lr * min(1.0, step / max(1, args.warmup_steps))
            if step > args.warmup_steps:
                decay = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
                lr = args.min_lr + 0.5 * (1 + math.cos(math.pi * decay)) * (args.lr - args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            batch = to_device(raw, device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                scores = model(batch)
                loss, c, n = group_loss(scores, raw["group_sizes"], batch["labels"], raw["teacher_scores"], args)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not math.isfinite(float(grad.item())):
                raise RuntimeError(f"non-finite grad norm at step {step}: {grad}")
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()) * n; ep_correct += c; ep_total += n
        row = {"epoch": epoch + 1, "loss": ep_loss / max(1, ep_total), "train_accuracy": ep_correct / max(1, ep_total), "elapsed_s": round(time.time() - start, 1)}
        logs.append(row)
        if args.progress:
            print(f"{kind} epoch {row['epoch']} | loss {row['loss']:.4f} | train_acc {row['train_accuracy']:.3f}")
    return {"evidence_kind": kind, "logs": logs}


@torch.no_grad()
def evaluate(args: argparse.Namespace, model: FactorizedJudge, records: list[dict], docs: list[CorpusDoc], device: torch.device, kind: str) -> dict:
    ds = FactDataset(records, docs, kind, args)
    loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    amp_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model.eval(); by_c: dict[str, int] = {}; by_n: dict[str, int] = {}; samples = []
    table = {"ids": [], "datasets": [], "labels": [], "preds": [], "correct": [], "scores": []}
    for raw in loader:
        batch = to_device(raw, device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            scores = model(batch).float().cpu()
        off = 0
        for i, g in enumerate(raw["group_sizes"]):
            gs = scores[off : off + g]
            pred = int(gs.argmax().item()); gold = int(raw["labels"][i].item()); ds_name = raw["datasets"][i]
            ok = int(pred == gold)
            by_c[ds_name] = by_c.get(ds_name, 0) + ok; by_n[ds_name] = by_n.get(ds_name, 0) + 1
            table["ids"].append(raw["ids"][i]); table["datasets"].append(ds_name); table["labels"].append(gold); table["preds"].append(pred); table["correct"].append(ok); table["scores"].append([float(x) for x in gs.tolist()])
            if len(samples) < args.prediction_samples:
                samples.append({"id": raw["ids"][i], "dataset": ds_name, "pred": pred, "label": gold, "scores": [round(float(x), 4) for x in gs.tolist()]})
            off += g
    total_c, total_n = sum(by_c.values()), sum(by_n.values())
    return {"evidence_kind": kind, "overall": total_c / max(1, total_n), "by_dataset": {ds: {"accuracy": by_c.get(ds, 0) / max(1, n), "correct": by_c.get(ds, 0), "total": n} for ds, n in sorted(by_n.items())}, "prediction_samples": samples, "prediction_table": table}


def score_map(records: list[dict], preds: dict[str, int]) -> dict:
    by: dict[str, list[int]] = {}
    for rec in records:
        by.setdefault(rec["dataset"], []).append(int(int(preds.get(rec["id"], 0)) == int(rec["label"])))
    n = sum(len(v) for v in by.values()); c = sum(sum(v) for v in by.values())
    return {"overall": c / max(1, n), "by_dataset": {ds: {"accuracy": sum(v) / max(1, len(v)), "correct": sum(v), "total": len(v)} for ds, v in sorted(by.items())}}


def unigram_preds(records: list[dict], docs: list[CorpusDoc]) -> dict[str, int]:
    counts: dict[str, int] = {}; total = 0
    for doc in docs:
        for tok in text_tokens(doc.text):
            counts[tok] = counts.get(tok, 0) + 1; total += 1
    vocab = max(1, len(counts)); out = {}
    for rec in records:
        vals = []
        for choice in rec["choices"]:
            toks = text_tokens(choice)
            vals.append(-1e9 if not toks else sum(math.log((counts.get(t, 0) + 1) / (total + vocab)) for t in toks) / len(toks))
        out[rec["id"]] = int(max(range(len(vals)), key=lambda i: vals[i]))
    return out


def nn_preds(train: list[dict], evals: list[dict]) -> dict[str, int]:
    train_toks = [(set(text_tokens(r["context"])), int(r["label"])) for r in train]
    out = {}
    for rec in evals:
        q = set(text_tokens(rec["context"])); best = (-1.0, 0)
        for toks, label in train_toks:
            if label >= len(rec["choices"]):
                continue
            score = len(q & toks) / max(1, len(q | toks))
            if score > best[0]:
                best = (score, label)
        out[rec["id"]] = int(best[1])
    return out


def baselines(train: list[dict], evals: list[dict], docs: list[CorpusDoc]) -> dict[str, dict]:
    out = {}
    majority = {}
    for ds in sorted({r["dataset"] for r in train}):
        labs = [int(r["label"]) for r in train if r["dataset"] == ds]
        majority[ds] = max(set(labs), key=labs.count) if labs else 0
    out["majority_label"] = score_map(evals, {r["id"]: majority.get(r["dataset"], 0) for r in evals})
    out["shortest_candidate"] = score_map(evals, {r["id"]: min(range(len(r["choices"])), key=lambda i: len(r["choices"][i])) for r in evals})
    out["unigram_frequency"] = score_map(evals, unigram_preds(evals, docs))
    out["nearest_neighbor_train_label"] = score_map(evals, nn_preds(train, evals))
    bm25, gold = {}, {}
    for rec in evals:
        ev = ev_text(rec, docs, "retrieved")
        gev = ev_text(rec, docs, "gold")
        bm25[rec["id"]] = int(max(range(len(rec["choices"])), key=lambda i: overlap_score(rec["choices"][i], ev) + 0.25 * overlap_score(rec["context"] + " " + rec["choices"][i], ev)))
        gold[rec["id"]] = int(max(range(len(rec["choices"])), key=lambda i: overlap_score(rec["choices"][i], gev)))
    out["bm25_evidence_overlap_ranker"] = score_map(evals, bm25)
    out["gold_overlap_oracle_rule"] = score_map(evals, gold)
    return out


def correct(result: dict) -> list[int]:
    return [int(x) for x in result.get("prediction_table", {}).get("correct", [])]


def boot(a: list[int], b: list[int], samples: int, seed: int) -> dict:
    if len(a) != len(b) or not a:
        return {"delta": None, "ci95": [None, None], "n": 0}
    aa = np.asarray(a, dtype=np.float32); bb = np.asarray(b, dtype=np.float32)
    rng = np.random.default_rng(seed); vals = []; n = len(aa)
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        vals.append(float(aa[idx].mean() - bb[idx].mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return {"delta": float(aa.mean() - bb.mean()), "ci95": [float(lo), float(hi)], "n": n}


def strip_tables(obj: dict) -> dict:
    out = copy.deepcopy(obj)
    for runs in out.get("models", {}).values():
        if isinstance(runs, list):
            for run in runs:
                for ev in run.get("eval", {}).values():
                    ev.pop("prediction_table", None)
                for ev in run.get("probe_eval", {}).values():
                    ev.pop("prediction_table", None)
        else:
            for ev in runs.get("eval", {}).values():
                ev.pop("prediction_table", None)
    return out


def run_one(args: argparse.Namespace, docs: list[CorpusDoc], train: list[dict], evals: list[dict], device: torch.device, seed: int, name: str, train_kind: str, out_dir: Path) -> dict:
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, codec_manifest = build_model(args, device)
    model_manifest = {"d_model": args.d_model, "layers": args.layers, "heads": args.heads, "kv_heads": args.kv_heads, "ffn_mult": args.ffn_mult, "freeze_codec": not args.finetune_codec, "param_counts": model.count_parameters(), "codec": codec_manifest}
    if args.progress:
        print(f"{name} seed={seed} params={model_manifest['param_counts']}")
    train_result = train_model(args, model, train, docs, device, train_kind, seed)
    eval_result = {kind: evaluate(args, model, evals, docs, device, kind) for kind in EVAL_KINDS}
    probes = {}
    if args.run_probe_conditions:
        for kind in ["gold_paraphrase", "gold_masked", "counterfactual"]:
            probes[kind] = evaluate(args, model, evals, docs, device, kind)
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    result = {"seed": seed, "model_name": name, "train_evidence_kind": train_kind, "model": model_manifest, "train": train_result, "eval": eval_result, "probe_eval": probes, "peak_vram_bytes": peak}
    model_dir = out_dir / f"seed_{seed}" / name
    model_dir.mkdir(parents=True, exist_ok=True)
    dump_json(model_dir / "metrics_with_predictions.json", result)
    dump_json(model_dir / "metrics.json", strip_tables({"models": {name: result}})["models"][name])
    if args.save_checkpoints:
        torch.save({"model_state_dict": model.state_dict(), "model_manifest": model_manifest, "args": vars(args), "seed": seed, "model_name": name}, model_dir / "evidence_factorized_judge.pt")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def aggregate(models: dict[str, list[dict]], base: dict[str, dict], args: argparse.Namespace) -> dict:
    means = {}
    for name, runs in models.items():
        means[name] = {}
        for kind in EVAL_KINDS:
            vals = [r["eval"][kind]["overall"] for r in runs]
            means[name][kind] = {"mean": float(np.mean(vals)) if vals else None, "std": float(np.std(vals)) if vals else None, "per_seed": vals}
            for ds in ["hellaswag", "piqa"]:
                ds_vals = [r["eval"][kind]["by_dataset"].get(ds, {}).get("accuracy") for r in runs]
                ds_vals = [v for v in ds_vals if v is not None]
                means[name][kind][ds] = float(np.mean(ds_vals)) if ds_vals else None
    best_name, best_payload = max(base.items(), key=lambda kv: kv[1]["overall"])
    me, mn = means.get("M_evidence", {}), means.get("M_none", {})
    gates = {"internalization_pp": round((me.get("none", {}).get("mean", 0) - mn.get("none", {}).get("mean", 0)) * 100, 2), "evidence_use_pp": round((me.get("retrieved", {}).get("mean", 0) - mn.get("retrieved", {}).get("mean", 0)) * 100, 2), "baseline_pp": round((me.get("retrieved", {}).get("mean", 0) - best_payload["overall"]) * 100, 2), "sensitivity_pp": round((me.get("retrieved", {}).get("mean", 0) - me.get("shuffled", {}).get("mean", 0)) * 100, 2), "gold_ceiling_hellaswag": me.get("gold", {}).get("hellaswag"), "best_dumb_baseline": best_name, "best_dumb_overall": best_payload["overall"]}
    gates.update({"INTERNALIZATION": gates["internalization_pp"] >= 2.0, "EVIDENCE_USE": gates["evidence_use_pp"] >= 3.0, "BASELINE": gates["baseline_pp"] >= 5.0, "SENSITIVITY": gates["sensitivity_pp"] >= 3.0, "GOLD_CEILING": (gates["gold_ceiling_hellaswag"] or 0) >= 0.35})
    cis = {}
    if models.get("M_evidence") and models.get("M_none"):
        a_none=[]; b_none=[]; a_ret=[]; b_ret=[]; a_sens=[]; b_sens=[]
        for er, nr in zip(models["M_evidence"], models["M_none"]):
            a_none += correct(er["eval"]["none"]); b_none += correct(nr["eval"]["none"])
            a_ret += correct(er["eval"]["retrieved"]); b_ret += correct(nr["eval"]["retrieved"])
            a_sens += correct(er["eval"]["retrieved"]); b_sens += correct(er["eval"]["shuffled"])
        cis = {"internalization": boot(a_none, b_none, args.bootstrap_samples, args.seed + 1), "evidence_use": boot(a_ret, b_ret, args.bootstrap_samples, args.seed + 2), "sensitivity": boot(a_sens, b_sens, args.bootstrap_samples, args.seed + 3)}
    return {"means": means, "gates": gates, "paired_bootstrap": cis}


def run_suite(args: argparse.Namespace) -> dict:
    out_dir, out_meta = resolve_output_dir(args.output_dir, args.fallback_output_dir)
    args.actual_output_dir = str(out_dir)
    set_seed(args.seed)
    device = choose_device(args.device)
    payload = load_or_prepare(args, out_dir)
    docs = [CorpusDoc(**d) for d in payload["docs"]]
    train, evals = payload["train_records"], payload["eval_records"]
    base = baselines(train, evals, docs)
    dump_json(out_dir / "dumb_baselines.json", base)
    models = {"M_evidence": [], "M_none": []}
    for seed in parse_seeds(args.seeds):
        models["M_evidence"].append(run_one(args, docs, train, evals, device, seed, "M_evidence", args.train_evidence_kind, out_dir))
        models["M_none"].append(run_one(args, docs, train, evals, device, seed, "M_none", "none", out_dir))
    agg = aggregate(models, base, args)
    suite = {"run": {"mode": "suite", "seeds": parse_seeds(args.seeds), "device": str(device), "requested_output_dir": args.output_dir, "actual_output_dir": str(out_dir), "output_meta": out_meta, "date": "2026-07-07"}, "data": {"train_records": len(train), "eval_records": len(evals), "corpus_manifest": payload["corpus_manifest"], "leakage_audit": payload["leakage_audit"], "controls": payload["controls"]}, "dumb_baselines": base, "models": models, "aggregate": agg}
    dump_json(out_dir / "suite_metrics_with_predictions.json", suite)
    dump_json(out_dir / "suite_metrics.json", strip_tables(suite))
    return suite


def run_train_eval(args: argparse.Namespace) -> dict:
    out_dir, out_meta = resolve_output_dir(args.output_dir, args.fallback_output_dir)
    args.actual_output_dir = str(out_dir)
    device = choose_device(args.device)
    payload = load_or_prepare(args, out_dir)
    docs = [CorpusDoc(**d) for d in payload["docs"]]
    train, evals = payload["train_records"], payload["eval_records"]
    name = "M_none" if args.train_evidence_kind == "none" else "M_evidence"
    model = run_one(args, docs, train, evals, device, args.seed, name, args.train_evidence_kind, out_dir)
    base = baselines(train, evals, docs)
    metrics = {"run": {"mode": "train_eval", "seed": args.seed, "device": str(device), "requested_output_dir": args.output_dir, "actual_output_dir": str(out_dir), "output_meta": out_meta}, "data": {"train_records": len(train), "eval_records": len(evals), "corpus_manifest": payload["corpus_manifest"], "leakage_audit": payload["leakage_audit"], "controls": payload["controls"]}, "dumb_baselines": base, "models": {name: model}}
    dump_json(out_dir / "metrics_with_predictions.json", metrics)
    dump_json(out_dir / "metrics.json", strip_tables(metrics))
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["prepare", "train_eval", "suite", "aggregate"], default="suite")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR); p.add_argument("--fallback-output-dir", default=DEFAULT_FALLBACK_OUTPUT_DIR)
    p.add_argument("--codec-checkpoint", default=DEFAULT_CODEC); p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--allow-downloads", action="store_true"); p.add_argument("--reuse-records", action="store_true")
    p.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"]); p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--seed", type=int, default=20260707); p.add_argument("--seeds", default=DEFAULT_SEEDS); p.add_argument("--no-randomize-choices", action="store_true")
    p.add_argument("--train-hellaswag", type=int, default=2000); p.add_argument("--train-piqa", type=int, default=2000); p.add_argument("--eval-hellaswag", type=int, default=1024); p.add_argument("--eval-piqa", type=int, default=1024)
    p.add_argument("--max-corpus-docs", type=int, default=5000); p.add_argument("--max-external-docs", type=int, default=4500); p.add_argument("--max-rationale-docs", type=int, default=4000); p.add_argument("--shard-docs", type=int, default=7000); p.add_argument("--shard-bytes", type=int, default=96_000_000); p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--include-train-rationales-in-corpus", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--rationale-mode", choices=["template", "compact"], default="template")
    p.add_argument("--max-context-bytes", type=int, default=224); p.add_argument("--max-evidence-bytes", type=int, default=640); p.add_argument("--max-candidate-bytes", type=int, default=160)
    p.add_argument("--d-model", type=int, default=256); p.add_argument("--layers", type=int, default=2); p.add_argument("--heads", type=int, default=4); p.add_argument("--kv-heads", type=int, default=2); p.add_argument("--ffn-mult", type=float, default=2.0); p.add_argument("--finetune-codec", action="store_true")
    p.add_argument("--train-evidence-kind", choices=["retrieved", "gold", "none"], default="retrieved"); p.add_argument("--evidence-dropout", type=float, default=0.0); p.add_argument("--teacher-alpha", type=float, default=0.1); p.add_argument("--teacher-temperature", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--eval-batch-size", type=int, default=16); p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--min-lr", type=float, default=3e-5); p.add_argument("--warmup-steps", type=int, default=50); p.add_argument("--weight-decay", type=float, default=0.02); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--prediction-samples", type=int, default=20); p.add_argument("--bootstrap-samples", type=int, default=1000); p.add_argument("--run-probe-conditions", action="store_true"); p.add_argument("--save-checkpoints", action="store_true"); p.add_argument("--progress", action="store_true")
    return p.parse_args()


def compact(metrics: dict) -> dict:
    if metrics["run"]["mode"] == "suite":
        return {"mode": "suite", "actual_output_dir": metrics["run"]["actual_output_dir"], "train_records": metrics["data"]["train_records"], "eval_records": metrics["data"]["eval_records"], "gates": metrics["aggregate"]["gates"], "paired_bootstrap": metrics["aggregate"]["paired_bootstrap"]}
    model = next(iter(metrics["models"].values()))
    return {"mode": metrics["run"]["mode"], "actual_output_dir": metrics["run"]["actual_output_dir"], "model": model["model_name"], "retrieved": model["eval"]["retrieved"]["overall"], "none": model["eval"]["none"]["overall"], "shuffled": model["eval"]["shuffled"]["overall"], "gold": model["eval"]["gold"]["overall"], "peak_vram_bytes": model["peak_vram_bytes"]}


def main() -> None:
    args = parse_args()
    out_dir, out_meta = resolve_output_dir(args.output_dir, args.fallback_output_dir)
    args.actual_output_dir = str(out_dir)
    if args.mode == "prepare":
        payload = load_or_prepare(args, out_dir)
        print(json.dumps({"train_records": len(payload["train_records"]), "eval_records": len(payload["eval_records"]), "corpus_manifest": payload["corpus_manifest"], "controls": payload["controls"], "output_meta": out_meta}, indent=2))
        return
    metrics = run_train_eval(args) if args.mode == "train_eval" else run_suite(args)
    print(json.dumps(compact(metrics), indent=2))


if __name__ == "__main__":
    main()


