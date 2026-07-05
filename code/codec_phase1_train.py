"""Phase 1: Semantic Codec Pre-Training.

Train the CausalByteTransformer to retrieve correct teacher token embeddings
at token boundary positions, using InfoNCE loss.

Codex R63 protocol:
- InfoNCE retrieval (not cosine alone)
- Shuffled-target control must fail
- bf16 training, AdamW
- LR: 3e-4 codec, 1e-3 alignment head
- Warmup: 500-1000 steps, grad clip 1.0
- Gate: top-1 retrieval >>chance, shuffled ≈ chance

Usage:
  python codec_phase1_train.py --data-dir C:/sutra_fast/data/shards_diverse --steps 5000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import mmap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from semantic_codec import SemanticCodec, CodecConfig, infonce_loss


def load_teacher_embeddings(cache_path: str = "C:/sutra_fast/teacher_embeddings.pt"):
    """Load or create teacher embedding table from Qwen3-0.6B."""
    if os.path.exists(cache_path):
        data = torch.load(cache_path, weights_only=True)
        print(f"Loaded teacher embeddings from cache: {data['embeddings'].shape}")
        return data["embeddings"], data["tokenizer_name"]

    print("Extracting teacher embeddings from Qwen3-0.6B...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    embeddings = model.model.embed_tokens.weight.detach().float()
    embeddings = F.normalize(embeddings, dim=-1)

    torch.save({"embeddings": embeddings, "tokenizer_name": model_name}, cache_path)
    print(f"Saved teacher embeddings: {embeddings.shape} -> {cache_path}")

    del model
    torch.cuda.empty_cache()

    return embeddings, model_name


def load_tokenizer():
    """Load Qwen3-0.6B tokenizer for boundary detection."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


class ByteShardLoader:
    """Load random byte sequences from shard files."""
    def __init__(self, data_dir: str, seq_len: int = 4096):
        self.seq_len = seq_len
        self.shards = sorted(Path(data_dir).glob("*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No .bin shards in {data_dir}")
        self.shard_sizes = [s.stat().st_size for s in self.shards]
        self.total_bytes = sum(self.shard_sizes)
        self._mmap_cache = {}
        self._file_handles = {}
        print(f"ByteShardLoader: {len(self.shards)} shards, {self.total_bytes/1e9:.1f}GB")

    def _get_mmap(self, shard_idx: int):
        if shard_idx not in self._mmap_cache:
            f = open(self.shards[shard_idx], "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._file_handles[shard_idx] = f
            self._mmap_cache[shard_idx] = mm
        return self._mmap_cache[shard_idx]

    def get_batch(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        """Get a batch of random byte sequences."""
        seqs = []
        for _ in range(batch_size):
            shard_idx = rng.integers(len(self.shards))
            max_start = self.shard_sizes[shard_idx] - self.seq_len
            if max_start <= 0:
                start = 0
            else:
                start = rng.integers(max_start)
            mm = self._get_mmap(shard_idx)
            raw = mm[start:start + self.seq_len]
            arr = np.frombuffer(raw, dtype=np.uint8).copy()
            # Replace doc separator (0xff) with space to avoid confusion
            arr[arr == 0xff] = 32
            seqs.append(arr)
        return torch.from_numpy(np.stack(seqs)).long()


def find_token_anchors(
    byte_ids: torch.Tensor,
    tokenizer,
    max_anchors: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find token boundary positions and corresponding token IDs.

    Args:
        byte_ids: (B, T) byte sequences
        tokenizer: HuggingFace tokenizer
        max_anchors: maximum anchors per sequence (pad shorter)
    Returns:
        anchor_positions: (B, max_anchors) — byte positions where tokens end
        anchor_token_ids: (B, max_anchors) — token IDs at those positions
    """
    B, T = byte_ids.shape
    all_positions = []
    all_token_ids = []

    for b in range(B):
        bytes_np = byte_ids[b].numpy().astype(np.uint8)
        text = bytes(bytes_np).decode("utf-8", errors="replace")

        tokens = tokenizer.encode(text, add_special_tokens=False)

        positions = []
        token_ids = []
        byte_offset = 0

        for tid in tokens:
            token_str = tokenizer.decode([tid])
            token_bytes = token_str.encode("utf-8", errors="replace")
            byte_offset += len(token_bytes)

            if byte_offset <= T:
                positions.append(byte_offset - 1)  # last byte of this token
                token_ids.append(tid)

        # Pad or truncate to max_anchors
        n = len(positions)
        if n >= max_anchors:
            positions = positions[:max_anchors]
            token_ids = token_ids[:max_anchors]
        else:
            positions = positions + [0] * (max_anchors - n)
            token_ids = token_ids + [0] * (max_anchors - n)

        all_positions.append(positions)
        all_token_ids.append(token_ids)

    return (
        torch.tensor(all_positions, dtype=torch.long),
        torch.tensor(all_token_ids, dtype=torch.long),
    )


def train_phase1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load teacher embeddings
    teacher_embs, _ = load_teacher_embeddings()
    teacher_embs = teacher_embs.to(device)  # (vocab_size, 1024)
    print(f"Teacher embeddings: {teacher_embs.shape}")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Build codec
    cfg = CodecConfig(window_size=args.window_size)
    codec = SemanticCodec(cfg, d_model=args.d_model).to(device)
    params = codec.count_params()
    print(f"Codec params: {params['total']:,}")

    # Data loader
    loader = ByteShardLoader(args.data_dir, seq_len=args.seq_len)
    rng = np.random.default_rng(args.seed)

    # Optimizer (separate LR per Codex R63)
    codec_params = list(codec.encoder.parameters()) + list(codec.patch_projection.parameters())
    align_params = list(codec.alignment_head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": codec_params, "lr": args.lr_codec},
        {"params": align_params, "lr": args.lr_align},
    ], weight_decay=0.01)

    # LR scheduler with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        decay = 1.0 - (step - args.warmup_steps) / (args.steps - args.warmup_steps)
        return max(0.1, decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Training
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "phase1_log.jsonl")
    log_f = open(log_path, "w")

    best_acc = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        # Get batch
        byte_ids = loader.get_batch(args.batch_size, rng)

        # Find anchors (CPU-bound tokenization)
        anchor_pos, anchor_tids = find_token_anchors(
            byte_ids, tokenizer, max_anchors=args.max_anchors
        )

        byte_ids = byte_ids.to(device)
        anchor_pos = anchor_pos.to(device)
        anchor_tids = anchor_tids.to(device)

        # Look up teacher embeddings for anchor tokens
        target_embs = teacher_embs[anchor_tids]  # (B, N, 1024)

        # Forward
        with autocast(dtype=torch.bfloat16):
            projected = codec.forward_phase1(byte_ids, anchor_pos)  # (B, N, 1024)
            loss, acc = infonce_loss(projected, target_embs, temperature=cfg.temperature)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Log
        if step % args.log_every == 0:
            elapsed = time.time() - t0
            entry = {
                "step": step,
                "loss": round(loss.item(), 4),
                "top1_acc": round(acc, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 6),
                "elapsed_s": round(elapsed, 1),
            }
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            print(f"[{step}/{args.steps}] loss={loss.item():.4f} acc={acc:.4f} "
                  f"lr={entry['lr']:.6f} ({elapsed:.0f}s)")

            if acc > best_acc:
                best_acc = acc

        # Checkpoint
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"codec_step{step}.pt")
            torch.save({
                "step": step,
                "codec_state_dict": codec.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": {
                    "codec_dim": cfg.codec_dim,
                    "codec_layers": cfg.codec_layers,
                    "window_size": cfg.window_size,
                    "d_model": args.d_model,
                },
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    log_f.close()
    print(f"\nPhase 1 complete. Best top-1 accuracy: {best_acc:.4f}")
    print(f"  (Chance = {1/(args.batch_size * args.max_anchors):.6f})")

    # Save final codec
    final_path = os.path.join(args.output_dir, "codec_final.pt")
    torch.save({
        "codec_state_dict": codec.state_dict(),
        "best_acc": best_acc,
        "config": {
            "codec_dim": cfg.codec_dim,
            "codec_layers": cfg.codec_layers,
            "window_size": cfg.window_size,
            "d_model": args.d_model,
        },
    }, final_path)
    print(f"Final codec saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Semantic Codec Pre-Training")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="C:/sutra_fast/codec_phase1")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--lr-codec", type=float, default=3e-4)
    parser.add_argument("--lr-align", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    args = parser.parse_args()

    train_phase1(args)


if __name__ == "__main__":
    main()
