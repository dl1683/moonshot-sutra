"""Eklavya E2 — Multi-teacher cache builder.

Builds the E2 cache in two passes:
  Pass 1: Student forward → position manifest (gap selection)
  Pass 2: Per-teacher forward → KL + align records at selected positions

Each teacher runs sequentially to fit in VRAM. Cache is written
incrementally to avoid OOM on large datasets.

Usage:
    python eklavya_e2_cache_builder.py \
        --student-checkpoint checkpoints/s0/s0_best.pt \
        --data-dir data/shards_bytes_full \
        --output-dir eklavya_e2_cache \
        --max-shards 50
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(__file__))

from s0_architecture import SutraS0
from eklavya_cache import (
    build_token_byte_table, first_byte_marginal,
    compute_token_byte_spans, validate_token_byte_alignment,
)
from eklavya_e2_cache import (
    TeacherSpec, TEACHER_REGISTRY, get_teacher_by_name,
    load_private_teacher_config,
    PositionRecord, E2KLRecord, E2AlignRecord,
    SelectionReason,
    write_position_manifest, read_position_manifest,
    save_e2_manifest,
)


class _StreamingKLWriter:
    """Streams KL records to disk shard by shard, fixes header count on close."""

    def __init__(self, path: str, K: int = 16):
        self._path = path
        self._K = K
        self._count = 0
        self._fh = open(path, "wb")
        self._fh.write(struct.pack("<II", 0, K))

    def extend(self, records: list[E2KLRecord]):
        for r in records:
            self._fh.write(r.pack(self._K))
        self._count += len(records)

    def close(self) -> int:
        self._fh.seek(0)
        self._fh.write(struct.pack("<II", self._count, self._K))
        self._fh.close()
        return self._count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _StreamingAlignWriter:
    """Streams align records to disk shard by shard, fixes header count on close."""

    def __init__(self, path: str):
        self._path = path
        self._count = 0
        self._fh = open(path, "wb")
        self._fh.write(struct.pack("<I", 0))

    def extend(self, records: list[E2AlignRecord]):
        for r in records:
            self._fh.write(r.pack())
        self._count += len(records)

    def close(self) -> int:
        self._fh.seek(0)
        self._fh.write(struct.pack("<I", self._count))
        self._fh.close()
        return self._count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.no_grad()
def build_position_manifest(
    student: SutraS0,
    shard_path: Path,
    shard_id: int,
    seq_len: int = 4096,
    nll_threshold: float = 3.5,
    entropy_threshold: float = 4.0,
    control_frac: float = 0.05,
    device: torch.device = torch.device("cuda"),
    pid_start: int = 0,
) -> list[PositionRecord]:
    """Run student forward on a shard to select gap positions."""
    student.eval()
    P = student.cfg.patch_size

    shard_data = np.fromfile(shard_path, dtype=np.uint8)
    n_seqs = len(shard_data) // seq_len
    records = []
    pid_counter = pid_start

    for seq_idx in range(n_seqs):
        offset = seq_idx * seq_len
        seq_bytes = shard_data[offset:offset + seq_len]
        byte_ids = torch.tensor(seq_bytes, dtype=torch.long, device=device).unsqueeze(0)

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = student(byte_ids)

        logits = out["logits"]
        B, Nm1, Pp, V = logits.shape
        N = Nm1 + 1
        targets = byte_ids.reshape(1, N, P)[:, 1:]

        for patch_idx in range(Nm1):
            byte_0 = targets[0, patch_idx, 0].item()
            logit_0 = logits[0, patch_idx, 0]
            log_p = F.log_softmax(logit_0, dim=-1)
            nll = -log_p[byte_0].item()
            probs = F.softmax(logit_0, dim=-1)
            ent = -(probs * probs.clamp(min=1e-10).log()).sum().item()

            reason = 0
            if nll > nll_threshold:
                reason |= SelectionReason.HIGH_NLL
            if ent > entropy_threshold:
                reason |= SelectionReason.HIGH_ENTROPY

            if reason == 0:
                if torch.rand(1).item() < control_frac:
                    reason = SelectionReason.CONTROL
                else:
                    continue

            records.append(PositionRecord(
                position_id=pid_counter,
                shard_id=shard_id,
                seq_offset=offset,
                patch_idx=patch_idx + 1,
                gold_byte=byte_0,
                student_nll=nll,
                student_entropy=ent,
                reason_mask=reason,
            ))
            pid_counter += 1

    return records


@torch.no_grad()
def build_teacher_records(
    teacher_model,
    tokenizer,
    spec: TeacherSpec,
    positions: list[PositionRecord],
    shard_path: Path,
    seq_len: int = 4096,
    patch_size: int = 4,
    kl_top_k: int = 16,
    device: torch.device = torch.device("cuda"),
    byte_table: dict = None,
) -> tuple[list[E2KLRecord], list[E2AlignRecord]]:
    """Build per-teacher KL and align records at selected positions."""
    teacher_model.eval()

    needs_byte_table = spec.has_kl or spec.has_align or spec.has_semantic
    if byte_table is None and needs_byte_table:
        byte_table = build_token_byte_table(tokenizer)

    shard_data = np.fromfile(shard_path, dtype=np.uint8)

    pos_by_seq = {}
    for pos in positions:
        key = pos.seq_offset
        pos_by_seq.setdefault(key, []).append(pos)

    kl_records = []
    align_records = []
    needs_align = spec.has_align or spec.has_semantic

    for seq_offset, pos_list in pos_by_seq.items():
        seq_bytes = shard_data[seq_offset:seq_offset + seq_len]
        text = bytes(seq_bytes).decode("utf-8", errors="replace")

        teacher_inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=getattr(tokenizer, 'model_max_length', 2048) or 2048,
        ).to(device)

        input_ids = teacher_inputs.input_ids[0].tolist()

        if needs_align:
            token_spans = compute_token_byte_spans(tokenizer, input_ids, byte_table)
            span_valid = validate_token_byte_alignment(
                bytes(seq_bytes), token_spans, input_ids, tokenizer, byte_table)

            for tok_idx, (bs, be) in enumerate(token_spans):
                if be <= bs or be > seq_len:
                    continue
                if not span_valid[tok_idx]:
                    continue
                for pos in pos_list:
                    p_start = pos.patch_idx * patch_size
                    p_end = p_start + patch_size
                    if bs < p_end and be > p_start:
                        align_records.append(E2AlignRecord(
                            position_id=pos.position_id,
                            byte_start=bs,
                            byte_len=be - bs,
                            token_id=input_ids[tok_idx],
                            align_quality=1.0,
                        ))

        if spec.has_kl:
            for pos in pos_list:
                t = pos.patch_idx * patch_size
                prefix_bytes = seq_bytes[:t]
                prefix_text = bytes(prefix_bytes).decode("utf-8", errors="replace")

                max_len = getattr(tokenizer, 'model_max_length', 2048) or 2048
                prefix_ids = tokenizer(
                    prefix_text, return_tensors="pt",
                    truncation=False,
                ).input_ids.to(device)

                if prefix_ids.shape[1] == 0:
                    continue
                if prefix_ids.shape[1] > max_len:
                    prefix_ids = prefix_ids[:, -max_len:]

                t_logits = teacher_model(prefix_ids).logits[0, -1]
                top_b, top_p, tail = first_byte_marginal(
                    t_logits, tokenizer, K=kl_top_k, _byte_table=byte_table)

                q = torch.zeros(256, dtype=torch.float32)
                q[top_b.astype(np.int64)] = torch.from_numpy(top_p.astype(np.float32))
                q_sum = q.sum()
                if q_sum > 0:
                    q = q / q_sum
                ent = -(q * (q + 1e-10).log()).sum().item() / math.log(2)
                logp_gold = math.log(max(q[pos.gold_byte].item(), 1e-20))

                kl_records.append(E2KLRecord(
                    position_id=pos.position_id,
                    patch_idx=pos.patch_idx,
                    tail_prob=tail,
                    entropy=ent,
                    logp_gold=logp_gold,
                    top_bytes=top_b,
                    top_probs=top_p,
                ))

    return kl_records, align_records


def load_teacher_for_spec(
    spec: TeacherSpec,
    device: torch.device,
) -> tuple:
    """Load a teacher model and tokenizer for the given spec.

    Causal/hybrid/SSM teachers use AutoModelForCausalLM.
    Embedding teachers use AutoModel (encoder-only).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_name)

    if spec.has_semantic and not spec.has_kl:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            spec.hf_name, torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_name, torch_dtype=torch.bfloat16,
        ).to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Build Eklavya E2 multi-teacher cache")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/shards_bytes_full")
    parser.add_argument("--output-dir", default="eklavya_e2_cache")
    parser.add_argument("--max-shards", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--nll-threshold", type=float, default=3.5)
    parser.add_argument("--kl-top-k", type=int, default=16)
    parser.add_argument("--teachers", nargs="+", default=None,
                        help="Teacher names to include (default: all)")
    parser.add_argument("--teacher-config", default=None,
                        help="Path to private teacher config JSON (alias→HF name)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--control-frac", type=float, default=0.05,
                        help="Fraction of non-gap positions to include as controls")
    parser.add_argument("--entropy-threshold", type=float, default=4.0,
                        help="Student entropy threshold for HIGH_ENTROPY selection")
    parser.add_argument("--shard-start", type=int, default=0,
                        help="First shard index to process (for resuming)")
    parser.add_argument("--shard-end", type=int, default=None,
                        help="Last shard index (exclusive, default: --max-shards)")
    parser.add_argument("--positions-only", action="store_true",
                        help="Only build position manifest, skip teacher records")
    parser.add_argument("--teachers-only", action="store_true",
                        help="Skip position manifest (read existing), build teacher records only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Select teachers
    if args.teachers:
        specs = [get_teacher_by_name(n) for n in args.teachers]
    else:
        specs = list(TEACHER_REGISTRY)

    # Resolve private HF names from config
    config_path = args.teacher_config or os.environ.get("SUTRA_TEACHER_CONFIG")
    if config_path:
        specs = load_private_teacher_config(config_path, specs)
        print(f"Loaded private teacher config from {config_path}")

    missing_hf = [s.name for s in specs if not s.hf_name]
    if missing_hf and not args.positions_only:
        raise SystemExit(
            f"ERROR: Teachers without HF names: {missing_hf}\n"
            "  Use --teacher-config or set SUTRA_TEACHER_CONFIG\n"
            "  (use --positions-only to skip teacher records)"
        )
    elif missing_hf:
        print(f"WARNING: Teachers without HF names: {missing_hf} "
              "(OK for --positions-only)")
    print(f"Teachers: {[s.name for s in specs]}")

    # Get shards
    shards = sorted(Path(args.data_dir).glob("*.bin"))
    if not shards:
        raise SystemExit(f"ERROR: No .bin shards found in {args.data_dir}")

    # Detect gaps in shard file numbering (silent remapping risk)
    shard_nums = []
    for s in shards:
        m = re.search(r"(\d+)", s.stem)
        if m:
            shard_nums.append(int(m.group(1)))
    if shard_nums:
        expected = set(range(min(shard_nums), max(shard_nums) + 1))
        missing = expected - set(shard_nums)
        if missing:
            raise SystemExit(
                f"ERROR: Sparse shard numbering detected. Missing shard numbers: "
                f"{sorted(missing)[:20]}{'...' if len(missing) > 20 else ''}. "
                f"This causes silent index remapping. Re-number shards or remove gaps."
            )

    shard_start = args.shard_start
    if shard_start < 0:
        raise SystemExit(
            f"ERROR: --shard-start {shard_start} is negative")
    if shard_start >= len(shards):
        raise SystemExit(
            f"ERROR: --shard-start {shard_start} >= total shards {len(shards)}")

    shard_end = args.shard_end if args.shard_end is not None else min(len(shards), args.max_shards)
    if args.shard_end is not None and args.shard_end < 0:
        raise SystemExit(
            f"ERROR: --shard-end {args.shard_end} is negative")
    if args.shard_end is not None and args.shard_end > len(shards):
        raise SystemExit(
            f"ERROR: --shard-end {args.shard_end} > total shards {len(shards)}")
    shard_end = min(shard_end, len(shards))

    if shard_start >= shard_end:
        raise SystemExit(
            f"ERROR: Empty shard range [{shard_start}, {shard_end})")
    print(f"Processing shards [{shard_start}, {shard_end}) of {len(shards)} total")

    os.makedirs(args.output_dir, exist_ok=True)
    agg_dir = os.path.join(args.output_dir, "aggregate")
    os.makedirs(agg_dir, exist_ok=True)

    t0 = time.time()
    manifest_path = os.path.join(args.output_dir, "positions.bin")

    if args.teachers_only:
        print("\n=== Skipping Pass 1 (--teachers-only), reading existing manifest ===")
        all_positions = read_position_manifest(manifest_path)
        print(f"  Loaded {len(all_positions)} positions from existing manifest")
        P = 4
        ckpt = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
        P = ckpt["model_cfg"].patch_size
        del ckpt
    else:
        # Load student
        print(f"Loading student: {args.student_checkpoint}")
        ckpt = torch.load(args.student_checkpoint, map_location=device, weights_only=False)
        model_cfg = ckpt["model_cfg"]
        student = SutraS0(model_cfg).to(device)
        student.load_state_dict(ckpt["model"])
        student.eval()
        P = model_cfg.patch_size
        print(f"Student loaded: {model_cfg.d_model}d, {model_cfg.n_layers}L, P={P}")

        # Pass 1: Build position manifest
        print("\n=== Pass 1: Position manifest ===")
        all_positions = []
        next_pid = 0

        for i in range(shard_start, shard_end):
            shard_data = np.fromfile(shards[i], dtype=np.uint8)
            if len(shard_data) < args.seq_len:
                print(f"  Shard {i}: SKIP (too small: {len(shard_data)} bytes)")
                continue
            print(f"  Shard {i}/{shard_end}: {shards[i].name}")
            positions = build_position_manifest(
                student, shards[i], i,
                seq_len=args.seq_len,
                nll_threshold=args.nll_threshold,
                entropy_threshold=args.entropy_threshold,
                control_frac=args.control_frac,
                device=device,
                pid_start=next_pid,
            )
            next_pid += len(positions)
            all_positions.extend(positions)

        write_position_manifest(manifest_path, all_positions)
        elapsed = time.time() - t0
        print(f"  {len(all_positions)} positions selected in {elapsed:.0f}s")

        # Free student VRAM
        del student
        torch.cuda.empty_cache()
        gc.collect()

    if args.positions_only:
        print("\n=== Done (--positions-only) ===")
        return

    # Group positions by shard
    pos_by_shard = {}
    for pos in all_positions:
        pos_by_shard.setdefault(pos.shard_id, []).append(pos)

    # Pass 2: Per-teacher cache building (one at a time for VRAM)
    print("\n=== Pass 2: Per-teacher records ===")

    for spec in specs:
        print(f"\n--- Teacher: {spec.name} ---")
        t_start = time.time()

        teacher_dir = os.path.join(args.output_dir, "teachers", spec.name)
        os.makedirs(teacher_dir, exist_ok=True)

        model, tokenizer = load_teacher_for_spec(spec, device)
        byte_table = build_token_byte_table(tokenizer) if (spec.has_kl or spec.has_align or spec.has_semantic) else None

        emb_table = model.get_input_embeddings().weight.detach().cpu()
        print(f"  Embedding table: {emb_table.shape}")

        kl_path = os.path.join(teacher_dir, "kl_records.bin")
        align_path = os.path.join(teacher_dir, "align_records.bin")
        kl_writer = _StreamingKLWriter(kl_path, K=args.kl_top_k)
        align_writer = _StreamingAlignWriter(align_path)

        try:
            for shard_id in sorted(pos_by_shard.keys()):
                if shard_id >= shard_end:
                    continue
                positions = pos_by_shard[shard_id]
                print(f"  Shard {shard_id}: {len(positions)} positions")

                kl_recs, align_recs = build_teacher_records(
                    model, tokenizer, spec, positions, shards[shard_id],
                    seq_len=args.seq_len,
                    patch_size=P,
                    kl_top_k=args.kl_top_k,
                    device=device,
                    byte_table=byte_table,
                )
                kl_writer.extend(kl_recs)
                align_writer.extend(align_recs)
        finally:
            n_kl = kl_writer.close()
            n_align = align_writer.close()

        if n_kl > 0:
            print(f"  Wrote {n_kl} KL records (streamed)")
        else:
            os.remove(kl_path)

        if n_align > 0:
            print(f"  Wrote {n_align} align records (streamed)")
        else:
            os.remove(align_path)

        if spec.has_align or spec.has_semantic:
            torch.save(emb_table.half(),
                       os.path.join(teacher_dir, "teacher_embeddings.pt"))
            print(f"  Saved embedding table: {emb_table.shape}")

        elapsed = time.time() - t_start
        print(f"  Teacher {spec.name} done in {elapsed:.0f}s")

        # Free teacher VRAM before loading next
        del model, tokenizer, byte_table
        torch.cuda.empty_cache()
        gc.collect()

    # Build provenance
    import subprocess
    git_hash = "unknown"
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    provenance = {
        "seed": args.seed,
        "nll_threshold": args.nll_threshold,
        "entropy_threshold": args.entropy_threshold,
        "control_frac": args.control_frac,
        "student_checkpoint": args.student_checkpoint,
        "data_dir": args.data_dir,
        "seq_len": args.seq_len,
        "git_commit": git_hash,
        "teacher_aliases": [s.name for s in specs],
    }

    save_e2_manifest(
        args.output_dir, specs,
        n_positions=len(all_positions),
        K=args.kl_top_k,
        shard_range=(shard_start, shard_end),
        provenance=provenance,
    )

    total_elapsed = time.time() - t0
    print("\n=== E2 cache complete ===")
    print(f"  {len(all_positions)} positions, {len(specs)} teachers")
    print(f"  Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
