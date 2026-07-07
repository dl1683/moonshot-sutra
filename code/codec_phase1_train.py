"""Phase 1 / Phase 1.5: Semantic Codec Pre-Training.

Phase 1 trains the CausalByteTransformer to retrieve teacher token embeddings at
teacher token-end byte positions. Phase 1.5 keeps the same InfoNCE objective but
adds dense 4-byte patch-boundary supervision: each patch boundary is supervised
against the teacher token whose byte span contains that boundary.

Codex R63/R65 protocol:
- InfoNCE retrieval, not cosine alone.
- Phase 1.5 changes anchor positions, not architecture or objective.
- Keep token-end anchors alive while emphasizing patch-boundary anchors.
- Log anchor counts so Phase 1 vs Phase 1.5 supervision density is explicit.

Usage:
  python code/codec_phase1_train.py --phase 1 --data-dir C:/sutra_fast/data/shards_diverse --steps 5000
  python code/codec_phase1_train.py --phase 1.5 --resume-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt --output-dir C:/sutra_fast/codec_phase1.5 --data-dir C:/sutra_fast/data/shards_diverse --steps 5000
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from semantic_codec import CodecConfig, SemanticCodec


DEFAULT_TEACHER_EMB = "C:/sutra_fast/teacher_embeddings.pt"
DEFAULT_TEACHER = "Qwen/Qwen3-0.6B"


def ensure_offline(allow_downloads: bool) -> None:
    if allow_downloads:
        return
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_teacher_embeddings(cache_path: str = DEFAULT_TEACHER_EMB) -> tuple[torch.Tensor, str]:
    """Load or create teacher embedding table from Qwen3-0.6B."""
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        embeddings = F.normalize(data["embeddings"].float(), dim=-1)
        print(f"Loaded teacher embeddings from cache: {embeddings.shape}")
        return embeddings, data.get("tokenizer_name", DEFAULT_TEACHER)

    print("Extracting teacher embeddings from Qwen3-0.6B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = DEFAULT_TEACHER
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    embeddings = F.normalize(model.model.embed_tokens.weight.detach().float(), dim=-1)
    torch.save({"embeddings": embeddings, "tokenizer_name": model_name}, cache_path)
    print(f"Saved teacher embeddings: {embeddings.shape} -> {cache_path}")
    del tokenizer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, model_name


def load_tokenizer(model_name: str = DEFAULT_TEACHER, allow_downloads: bool = False):
    """Load Qwen3-0.6B tokenizer for boundary detection."""
    ensure_offline(allow_downloads)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_downloads)


class ByteShardLoader:
    """Load random byte sequences from shard files."""

    def __init__(self, data_dir: str, seq_len: int = 4096):
        self.seq_len = seq_len
        self.shards = sorted(Path(data_dir).glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No .bin shards in {data_dir}")
        self.shard_sizes = [s.stat().st_size for s in self.shards]
        self.total_bytes = sum(self.shard_sizes)
        self._mmap_cache: dict[int, mmap.mmap] = {}
        self._file_handles: dict[int, object] = {}
        print(f"ByteShardLoader: {len(self.shards)} shards, {self.total_bytes/1e9:.1f}GB")

    def _get_mmap(self, shard_idx: int) -> mmap.mmap:
        if shard_idx not in self._mmap_cache:
            handle = open(self.shards[shard_idx], "rb")
            self._file_handles[shard_idx] = handle
            self._mmap_cache[shard_idx] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap_cache[shard_idx]

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        """Get a batch of random byte sequences."""
        seqs = []
        for _ in range(batch_size):
            shard_idx = int(rng.integers(len(self.shards)))
            max_start = max(0, self.shard_sizes[shard_idx] - self.seq_len)
            start = int(rng.integers(max_start + 1)) if max_start else 0
            raw = self._get_mmap(shard_idx)[start:start + self.seq_len]
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
            if len(arr) < self.seq_len:
                arr = np.pad(arr, (0, self.seq_len - len(arr)), constant_values=32)
            arr[arr == 0xFF] = 32
            seqs.append(arr)
        return torch.from_numpy(np.stack(seqs)).long()


def token_spans_for_bytes(byte_row: torch.Tensor, tokenizer) -> list[tuple[int, int, int]]:
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


def sample_pairs(
    pairs: list[tuple[int, int, str]],
    cap: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, str]]:
    if cap <= 0 or len(pairs) <= cap:
        return pairs
    keep = np.sort(rng.choice(len(pairs), size=cap, replace=False))
    return [pairs[int(i)] for i in keep]


def find_codec_anchors(
    byte_ids: torch.Tensor,
    tokenizer,
    phase: str,
    max_anchors: int,
    patch_size: int,
    patch_fraction: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Find token-end anchors for Phase 1 or mixed token/patch anchors for Phase 1.5."""
    batch_positions: list[list[int]] = []
    batch_token_ids: list[list[int]] = []
    batch_masks: list[list[bool]] = []
    stats = {
        "token_found": 0,
        "patch_found": 0,
        "overlap_found": 0,
        "token_used": 0,
        "patch_used": 0,
        "anchors_used": 0,
    }

    patch_budget = int(round(max_anchors * patch_fraction)) if phase == "1.5" else 0
    patch_budget = min(max_anchors, max(0, patch_budget))
    token_budget = max_anchors - patch_budget if phase == "1.5" else max_anchors
    if phase == "1.5" and token_budget == 0 and max_anchors > 1:
        token_budget = 1
        patch_budget = max_anchors - 1

    for b in range(byte_ids.shape[0]):
        spans = token_spans_for_bytes(byte_ids[b], tokenizer)
        token_pairs = [(end, tid, "token") for _, end, tid in spans]

        patch_pairs: list[tuple[int, int, str]] = []
        span_idx = 0
        for pos in range(patch_size - 1, byte_ids.shape[1], patch_size):
            while span_idx < len(spans) and spans[span_idx][1] < pos:
                span_idx += 1
            if span_idx >= len(spans):
                break
            start, end, tid = spans[span_idx]
            if start <= pos <= end:
                patch_pairs.append((pos, tid, "patch"))

        token_keys = {(pos, tid) for pos, tid, _ in token_pairs}
        patch_keys = {(pos, tid) for pos, tid, _ in patch_pairs}
        stats["token_found"] += len(token_pairs)
        stats["patch_found"] += len(patch_pairs)
        stats["overlap_found"] += len(token_keys & patch_keys)

        if phase == "1":
            selected = sample_pairs(token_pairs, max_anchors, rng)
        elif phase == "1.5":
            # Patch anchors dominate because they are the failing downstream readout;
            # token-end anchors remain in-batch to prevent forgetting the proven chart.
            selected = sample_pairs(patch_pairs, patch_budget, rng) + sample_pairs(token_pairs, token_budget, rng)
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        deduped: list[tuple[int, int, str]] = []
        seen: set[tuple[int, int]] = set()
        for pos, tid, kind in selected:
            key = (pos, tid)
            if key in seen:
                continue
            seen.add(key)
            deduped.append((pos, tid, kind))
        if len(deduped) > max_anchors:
            deduped = sample_pairs(deduped, max_anchors, rng)

        stats["token_used"] += sum(1 for _, _, kind in deduped if kind == "token")
        stats["patch_used"] += sum(1 for _, _, kind in deduped if kind == "patch")
        stats["anchors_used"] += len(deduped)

        positions = [pos for pos, _, _ in deduped]
        token_ids = [tid for _, tid, _ in deduped]
        mask = [True] * len(deduped)
        pad = max_anchors - len(deduped)
        if pad > 0:
            positions.extend([0] * pad)
            token_ids.extend([0] * pad)
            mask.extend([False] * pad)
        batch_positions.append(positions)
        batch_token_ids.append(token_ids)
        batch_masks.append(mask)

    return (
        torch.tensor(batch_positions, dtype=torch.long),
        torch.tensor(batch_token_ids, dtype=torch.long),
        torch.tensor(batch_masks, dtype=torch.bool),
        stats,
    )


def infonce_loss_flat(
    projected: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, float]:
    """InfoNCE over a variable number of unpadded anchors."""
    if projected.ndim != 2 or teacher_embeddings.ndim != 2:
        raise ValueError("projected and teacher_embeddings must be flat (N, D) tensors")
    if projected.shape[0] == 0:
        raise ValueError("no anchors available for InfoNCE")
    queries = F.normalize(projected.float(), dim=-1)
    keys = F.normalize(teacher_embeddings.float(), dim=-1)
    sim = queries @ keys.T / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = F.cross_entropy(sim, labels)
    with torch.no_grad():
        acc = (sim.argmax(dim=1) == labels).float().mean().item()
    return loss, acc


def load_resume_checkpoint(path: str | None) -> dict | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    print(f"Warm-starting codec from: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_codec_from_args(args: argparse.Namespace, resume_ckpt: dict | None) -> tuple[SemanticCodec, CodecConfig, int]:
    cfg_payload = (resume_ckpt or {}).get("config", {})
    cfg = CodecConfig(
        codec_dim=int(cfg_payload.get("codec_dim", 256)),
        codec_layers=int(cfg_payload.get("codec_layers", 4)),
        window_size=int(cfg_payload.get("window_size", args.window_size)),
    )
    d_model = int(cfg_payload.get("d_model", args.d_model))
    codec = SemanticCodec(cfg, d_model=d_model)
    if resume_ckpt is not None:
        codec.load_state_dict(resume_ckpt["codec_state_dict"])
    return codec, cfg, d_model


def train_phase1(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")

    teacher_embs, _ = load_teacher_embeddings(args.teacher_embeddings)
    teacher_embs = teacher_embs.to(device)
    print(f"Teacher embeddings: {teacher_embs.shape}")

    tokenizer = load_tokenizer(args.teacher, args.allow_downloads)
    resume_ckpt = load_resume_checkpoint(args.resume_checkpoint)
    codec, cfg, d_model = build_codec_from_args(args, resume_ckpt)
    codec.to(device)
    params = codec.count_params()
    print(f"Codec params: {params['total']:,}")

    loader = ByteShardLoader(args.data_dir, seq_len=args.seq_len)
    rng = np.random.default_rng(args.seed)

    codec_params = list(codec.encoder.parameters()) + list(codec.patch_projection.parameters())
    align_params = list(codec.alignment_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": codec_params, "lr": args.lr_codec},
        {"params": align_params, "lr": args.lr_align},
    ], weight_decay=0.01)

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        decay_steps = max(1, args.steps - args.warmup_steps)
        decay = 1.0 - (step - args.warmup_steps) / decay_steps
        return max(0.1, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    safe_phase = args.phase.replace(".", "_")
    log_path = os.path.join(args.output_dir, f"phase{safe_phase}_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    best_acc = float((resume_ckpt or {}).get("best_acc", 0.0))
    t0 = time.time()
    running_stats = {"token_found": 0, "patch_found": 0, "overlap_found": 0, "token_used": 0, "patch_used": 0, "anchors_used": 0}
    running_batches = 0

    for step in range(1, args.steps + 1):
        byte_ids_cpu = loader.get_batch(args.batch_size, rng)
        anchor_pos, anchor_tids, anchor_mask, anchor_stats = find_codec_anchors(
            byte_ids_cpu,
            tokenizer,
            phase=args.phase,
            max_anchors=args.max_anchors,
            patch_size=cfg.patch_size,
            patch_fraction=args.patch_fraction,
            rng=rng,
        )
        for key in running_stats:
            running_stats[key] += anchor_stats[key]
        running_batches += 1

        byte_ids = byte_ids_cpu.to(device)
        anchor_pos = anchor_pos.to(device)
        anchor_tids = anchor_tids.to(device)
        anchor_mask = anchor_mask.to(device)
        target_embs = teacher_embs[anchor_tids]

        with autocast(dtype=torch.bfloat16, enabled=device.type == "cuda"):
            projected_all = codec.forward_phase1(byte_ids, anchor_pos)
            projected = projected_all[anchor_mask]
            targets = target_embs[anchor_mask]
            loss, acc = infonce_loss_flat(projected, targets, temperature=cfg.temperature)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if acc > best_acc:
            best_acc = acc

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t0
            denom = max(1, running_batches * args.batch_size)
            entry = {
                "step": step,
                "phase": args.phase,
                "loss": round(float(loss.item()), 4),
                "top1_acc": round(float(acc), 4),
                "anchors_in_batch": int(projected.shape[0]),
                "avg_token_found_per_seq": round(running_stats["token_found"] / denom, 2),
                "avg_patch_found_per_seq": round(running_stats["patch_found"] / denom, 2),
                "avg_overlap_found_per_seq": round(running_stats["overlap_found"] / denom, 2),
                "avg_token_used_per_seq": round(running_stats["token_used"] / denom, 2),
                "avg_patch_used_per_seq": round(running_stats["patch_used"] / denom, 2),
                "avg_anchors_used_per_seq": round(running_stats["anchors_used"] / denom, 2),
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "elapsed_s": round(elapsed, 1),
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(
                f"[{step}/{args.steps}] loss={loss.item():.4f} acc={acc:.4f} "
                f"anchors={projected.shape[0]} token_used={entry['avg_token_used_per_seq']} "
                f"patch_used={entry['avg_patch_used_per_seq']} lr={entry['lr']:.8f} ({elapsed:.0f}s)"
            )

        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"codec_step{step}.pt")
            save_checkpoint(ckpt_path, codec, optimizer, best_acc, cfg, d_model, args, step)
            print(f"  Saved checkpoint: {ckpt_path}")

    log_f.close()
    print(f"\nPhase {args.phase} complete. Best batch top-1 accuracy: {best_acc:.4f}")
    final_path = os.path.join(args.output_dir, "codec_final.pt")
    save_checkpoint(final_path, codec, optimizer, best_acc, cfg, d_model, args, args.steps, final=True)
    print(f"Final codec saved: {final_path}")


def save_checkpoint(
    path: str,
    codec: SemanticCodec,
    optimizer: torch.optim.Optimizer,
    best_acc: float,
    cfg: CodecConfig,
    d_model: int,
    args: argparse.Namespace,
    step: int,
    final: bool = False,
) -> None:
    payload = {
        "step": step,
        "phase": args.phase,
        "codec_state_dict": codec.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": {
            "codec_dim": cfg.codec_dim,
            "codec_layers": cfg.codec_layers,
            "window_size": cfg.window_size,
            "d_model": d_model,
            "phase": args.phase,
            "max_anchors": args.max_anchors,
            "patch_fraction": args.patch_fraction,
            "resume_checkpoint": args.resume_checkpoint,
        },
    }
    if final:
        payload.pop("optimizer_state_dict", None)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 / Phase 1.5 Semantic Codec Pre-Training")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="C:/sutra_fast/codec_phase1")
    parser.add_argument("--phase", choices=["1", "1.5"], default="1")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--teacher", default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-embeddings", default=DEFAULT_TEACHER_EMB)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--patch-fraction", type=float, default=0.75)
    parser.add_argument("--lr-codec", type=float, default=3e-4)
    parser.add_argument("--lr-align", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    if not (0.0 <= args.patch_fraction <= 1.0):
        raise ValueError("--patch-fraction must be in [0, 1]")
    train_phase1(args)


if __name__ == "__main__":
    main()
