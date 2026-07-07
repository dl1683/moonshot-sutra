"""Minimal mechanism test: does codec + token KD transfer teacher knowledge?

Tests the core hypothesis at tiny scale (~300K trainable params, ~5 min):
- Condition A: byte CE only → student predicts next bytes
- Condition B: byte CE + token KD → student also predicts teacher's next tokens
- Condition C: byte CE + SHUFFLED token KD → control for artifacts

If B >> A on teacher-prediction accuracy on held-out data, the mechanism works.
If B ≈ A, the codec bridge doesn't enable knowledge transfer.

Designed per "old school science" principle: understand before scaling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from s0_architecture import RMSNorm
from semantic_codec import SemanticCodec, CodecConfig, CausalByteTransformer


class TinyReasoner(nn.Module):
    """Single attention layer + byte decoder for minimal testing."""
    def __init__(self, d_model: int = 128, n_heads: int = 4, patch_size: int = 4):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = RMSNorm(d_model)

        self.byte_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 256 * patch_size),
        )

    def forward(self, patch_states):
        B, N, D = patch_states.shape
        mask = torch.triu(torch.ones(N, N, device=patch_states.device), diagonal=1).bool()
        h = self.norm1(patch_states)
        h = patch_states + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = h + self.ffn(self.norm2(h))

        logits = self.byte_head(h[:, :-1]).reshape(B, N - 1, self.patch_size, 256)
        return h, logits


class TinyKDHead(nn.Module):
    """Minimal tied-embedding KD head."""
    def __init__(self, d_model: int, teacher_dim: int, teacher_emb: torch.Tensor):
        super().__init__()
        self.proj = nn.Linear(d_model, teacher_dim, bias=False)
        self.register_buffer("teacher_emb", teacher_emb)

    def forward(self, hidden):
        h = self.proj(hidden)
        return h @ self.teacher_emb.T


class MinimalCodecModel(nn.Module):
    """Frozen codec + tiny reasoner + optional KD head."""
    def __init__(self, codec_encoder: CausalByteTransformer, d_model: int = 128,
                 patch_size: int = 4, teacher_emb: torch.Tensor | None = None):
        super().__init__()
        self.codec_encoder = codec_encoder
        for p in self.codec_encoder.parameters():
            p.requires_grad_(False)

        codec_dim = codec_encoder.cfg.codec_dim
        self.patch_proj = nn.Sequential(
            nn.Linear(codec_dim, d_model, bias=False),
            RMSNorm(d_model),
        )
        self.reasoner = TinyReasoner(d_model, patch_size=patch_size)
        self.patch_size = patch_size

        self.kd_head = None
        if teacher_emb is not None:
            self.kd_head = TinyKDHead(d_model, teacher_emb.shape[1], teacher_emb)

    def forward(self, byte_ids):
        with torch.no_grad():
            hidden = self.codec_encoder(byte_ids)
        P = self.patch_size
        patch_states = hidden[:, P - 1::P, :]
        patch_states = self.patch_proj(patch_states)

        h, byte_logits = self.reasoner(patch_states)

        result = {"byte_logits": byte_logits, "hidden": h}
        if self.kd_head is not None:
            result["token_logits"] = self.kd_head(h)
        return result


def load_teacher_logits(sequences, teacher_model_name="Qwen/Qwen3-0.6B"):
    """Pre-compute teacher next-token logits for a set of byte sequences."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading teacher ({teacher_model_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name, torch_dtype=torch.float16
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(teacher_model_name)

    all_logits = []
    all_token_positions = []

    for seq_bytes in sequences:
        text = bytes(seq_bytes.cpu().numpy()).decode("utf-8", errors="replace")
        input_ids = tok.encode(text, add_special_tokens=False, return_tensors="pt").cuda()

        with torch.no_grad():
            out = model(input_ids)
            logits = out.logits[0].float().cpu()

        token_texts = [tok.decode([tid]) for tid in input_ids[0].cpu().tolist()]
        byte_positions = []
        pos = 0
        for tt in token_texts:
            tlen = len(tt.encode("utf-8"))
            byte_positions.append(pos + tlen - 1)
            pos += tlen

        shifted_logits = logits[:-1]
        shifted_positions = byte_positions[:-1]

        all_logits.append(shifted_logits)
        all_token_positions.append(shifted_positions)

    del model
    torch.cuda.empty_cache()
    return all_logits, all_token_positions


def compute_byte_ce(byte_logits, byte_ids, patch_size):
    B, N_minus_1, P, V = byte_logits.shape
    targets = byte_ids.reshape(B, -1, P)[:, 1 : N_minus_1 + 1]
    loss = F.cross_entropy(byte_logits.reshape(-1, V), targets.reshape(-1).long())
    return loss


def compute_token_kd(token_logits, teacher_logits_list, token_positions_list,
                     byte_ids, patch_size, k=64):
    """Token KD loss at clean token-boundary positions."""
    B = byte_ids.shape[0]
    device = token_logits.device
    total_loss = 0.0
    count = 0

    N = token_logits.shape[1]

    for b in range(B):
        if b >= len(teacher_logits_list):
            continue
        t_logits = teacher_logits_list[b].to(device)
        t_positions = token_positions_list[b]

        for tok_idx, byte_pos in enumerate(t_positions):
            patch_idx = byte_pos // patch_size
            if patch_idx >= N:
                continue

            student_l = token_logits[b, patch_idx].unsqueeze(0)
            teacher_l = t_logits[tok_idx].unsqueeze(0)

            _, top_idx = teacher_l.topk(k, dim=-1)
            s_top = student_l.gather(-1, top_idx)
            t_top = teacher_l.gather(-1, top_idx)

            s_log = F.log_softmax(s_top, dim=-1)
            t_prob = F.softmax(t_top, dim=-1)
            loss = F.kl_div(s_log, t_prob, reduction="batchmean")
            total_loss += loss
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)
    return total_loss / count


def evaluate_teacher_agreement(model, eval_bytes, teacher_logits_list,
                               token_positions_list, patch_size, k=10):
    """How well does the student predict teacher's top-k ranking?"""
    model.eval()
    device = next(model.parameters()).device

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for i, seq in enumerate(eval_bytes):
            if i >= len(teacher_logits_list):
                break
            byte_ids = seq.unsqueeze(0).to(device)
            out = model(byte_ids)

            if "token_logits" not in out or out["token_logits"] is None:
                continue

            s_logits = out["token_logits"][0]
            t_logits = teacher_logits_list[i].to(device)
            t_positions = token_positions_list[i]

            N = s_logits.shape[0]

            for tok_idx, byte_pos in enumerate(t_positions):
                patch_idx = byte_pos // patch_size
                if patch_idx >= N:
                    continue

                teacher_top1 = t_logits[tok_idx].argmax().item()
                student_ranking = s_logits[patch_idx].argsort(descending=True)

                if student_ranking[0].item() == teacher_top1:
                    correct_top1 += 1
                if teacher_top1 in student_ranking[:5].tolist():
                    correct_top5 += 1
                total += 1

    model.train()
    if total == 0:
        return {"top1_agreement": 0, "top5_agreement": 0, "total": 0}
    return {
        "top1_agreement": correct_top1 / total,
        "top5_agreement": correct_top5 / total,
        "total": total,
        "chance_top1": 1 / 151936,
    }


def run_condition(name, model, train_bytes, train_teacher_logits,
                  train_teacher_positions, eval_bytes, eval_teacher_logits,
                  eval_teacher_positions, use_kd=False, shuffle_kd=False,
                  steps=500, lr=1e-3, patch_size=4, alpha=1.0):
    """Run one experimental condition."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    print(f"\n{'='*60}")
    print(f"Condition: {name}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"KD: {use_kd}, Shuffled: {shuffle_kd}, Steps: {steps}")
    print(f"{'='*60}")

    teacher_logits_for_kd = train_teacher_logits
    if shuffle_kd and use_kd:
        teacher_logits_for_kd = [tl[torch.randperm(len(tl))] for tl in train_teacher_logits]

    batch_size = min(4, len(train_bytes))
    n_seqs = len(train_bytes)
    log = []

    for step in range(1, steps + 1):
        indices = torch.randint(0, n_seqs, (batch_size,))
        byte_ids = torch.stack([train_bytes[i] for i in indices]).to(device)

        out = model(byte_ids)
        byte_loss = compute_byte_ce(out["byte_logits"], byte_ids, patch_size)

        loss = byte_loss
        kd_loss_val = 0.0

        if use_kd and "token_logits" in out and out["token_logits"] is not None:
            batch_teacher_l = [teacher_logits_for_kd[i] for i in indices.tolist()]
            batch_teacher_p = [train_teacher_positions[i] for i in indices.tolist()]
            kd_loss = compute_token_kd(
                out["token_logits"], batch_teacher_l, batch_teacher_p,
                byte_ids, patch_size
            )
            loss = byte_loss + alpha * kd_loss
            kd_loss_val = kd_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            bpb = byte_loss.item() / math.log(2)
            entry = {"step": step, "byte_loss": round(byte_loss.item(), 4),
                     "bpb": round(bpb, 3), "kd_loss": round(kd_loss_val, 4)}
            log.append(entry)
            kd_str = f" kd={kd_loss_val:.4f}" if use_kd else ""
            print(f"  [{name}] step {step:>4d}: bpb={bpb:.3f}{kd_str}")

    agreement = evaluate_teacher_agreement(
        model, eval_bytes, eval_teacher_logits, eval_teacher_positions, patch_size
    )

    print(f"\n  [{name}] RESULTS:")
    print(f"    Final BPB: {log[-1]['bpb']:.3f}")
    print(f"    Teacher top-1 agreement: {agreement['top1_agreement']:.4f}")
    print(f"    Teacher top-5 agreement: {agreement['top5_agreement']:.4f}")
    print(f"    Positions evaluated: {agreement['total']}")
    print(f"    Chance: {agreement.get('chance_top1', 0):.6f}")

    return {"name": name, "log": log, "agreement": agreement, "final_bpb": log[-1]["bpb"]}


def main():
    parser = argparse.ArgumentParser(description="Minimal KD mechanism test")
    parser.add_argument("--codec-checkpoint", default="C:/sutra_fast/codec_phase1/codec_final.pt")
    parser.add_argument("--teacher-embeddings", default="C:/sutra_fast/teacher_embeddings.pt")
    parser.add_argument("--data-dir", default="C:/sutra_fast/data/shards_diverse")
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-eval", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--output", default="C:/sutra_fast/minimal_kd_test_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 4

    # Load codec encoder
    print("Loading codec encoder...")
    ckpt = torch.load(args.codec_checkpoint, map_location="cpu", weights_only=True)
    codec_cfg_dict = ckpt.get("config", {})
    codec_cfg = CodecConfig(
        codec_dim=codec_cfg_dict.get("codec_dim", 256),
        codec_layers=codec_cfg_dict.get("codec_layers", 4),
        window_size=codec_cfg_dict.get("window_size", 256),
    )
    codec_encoder = CausalByteTransformer(codec_cfg)
    codec_state = {k.replace("encoder.", ""): v
                   for k, v in ckpt["codec_state_dict"].items()
                   if k.startswith("encoder.")}
    codec_encoder.load_state_dict(codec_state)
    codec_encoder.eval()
    print(f"  Codec: {sum(p.numel() for p in codec_encoder.parameters()):,} params")

    # Load teacher embeddings
    print("Loading teacher embeddings...")
    teacher_data = torch.load(args.teacher_embeddings, map_location="cpu", weights_only=True)
    teacher_emb = teacher_data["embeddings"]
    print(f"  Teacher embeddings: {teacher_emb.shape}")

    # Load data sequences
    print("Loading data...")
    shards = sorted(Path(args.data_dir).glob("*.bin"))
    raw = bytearray()
    for s in shards[:2]:
        raw.extend(s.read_bytes())
        if len(raw) >= (args.n_train + args.n_eval) * args.seq_len * 2:
            break

    all_seqs = []
    for i in range(0, len(raw) - args.seq_len, args.seq_len):
        seq = torch.tensor(list(raw[i:i + args.seq_len]), dtype=torch.long)
        all_seqs.append(seq)
        if len(all_seqs) >= args.n_train + args.n_eval:
            break

    train_bytes = all_seqs[:args.n_train]
    eval_bytes = all_seqs[args.n_train:args.n_train + args.n_eval]
    print(f"  Train: {len(train_bytes)} seqs, Eval: {len(eval_bytes)} seqs")

    # Pre-compute teacher logits
    print("Pre-computing teacher logits...")
    train_teacher_logits, train_teacher_positions = load_teacher_logits(train_bytes)
    eval_teacher_logits, eval_teacher_positions = load_teacher_logits(eval_bytes)
    print(f"  Train teacher logits: {len(train_teacher_logits)} sequences")
    print(f"  Eval teacher logits: {len(eval_teacher_logits)} sequences")

    results = []

    # Condition A: byte CE only (WITH kd_head for evaluation, but no KD loss)
    print("\n" + "=" * 70)
    print("CONDITION A: Byte CE only")
    model_a = MinimalCodecModel(
        codec_encoder, d_model=args.d_model, patch_size=patch_size,
        teacher_emb=teacher_emb,
    ).to(device)
    result_a = run_condition(
        "A_byte_ce", model_a, train_bytes, train_teacher_logits,
        train_teacher_positions, eval_bytes, eval_teacher_logits,
        eval_teacher_positions, use_kd=False, steps=args.steps,
    )
    results.append(result_a)
    del model_a

    # Condition B: byte CE + token KD
    print("\n" + "=" * 70)
    print("CONDITION B: Byte CE + Token KD")
    model_b = MinimalCodecModel(
        codec_encoder, d_model=args.d_model, patch_size=patch_size,
        teacher_emb=teacher_emb,
    ).to(device)
    result_b = run_condition(
        "B_byte_ce_kd", model_b, train_bytes, train_teacher_logits,
        train_teacher_positions, eval_bytes, eval_teacher_logits,
        eval_teacher_positions, use_kd=True, steps=args.steps,
    )
    results.append(result_b)
    del model_b

    # Condition C: byte CE + SHUFFLED token KD
    print("\n" + "=" * 70)
    print("CONDITION C: Byte CE + Shuffled Token KD (control)")
    model_c = MinimalCodecModel(
        codec_encoder, d_model=args.d_model, patch_size=patch_size,
        teacher_emb=teacher_emb,
    ).to(device)
    result_c = run_condition(
        "C_shuffled_kd", model_c, train_bytes, train_teacher_logits,
        train_teacher_positions, eval_bytes, eval_teacher_logits,
        eval_teacher_positions, use_kd=True, shuffle_kd=True, steps=args.steps,
    )
    results.append(result_c)
    del model_c

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<25} {'BPB':>8} {'Top1 Agree':>12} {'Top5 Agree':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['final_bpb']:>8.3f} "
              f"{r['agreement']['top1_agreement']:>12.4f} "
              f"{r['agreement']['top5_agreement']:>12.4f}")

    print()
    a_top1 = results[0]["agreement"]["top1_agreement"]
    b_top1 = results[1]["agreement"]["top1_agreement"]
    c_top1 = results[2]["agreement"]["top1_agreement"]

    if b_top1 > a_top1 * 1.5 and b_top1 > c_top1 * 1.5:
        print("VERDICT: Token KD via codec WORKS. B >> A and B >> C.")
        print("The codec bridge enables knowledge transfer.")
    elif b_top1 > a_top1 * 1.2:
        print("VERDICT: Modest signal. B > A but not overwhelming.")
        print("Mechanism exists but may need more capacity or training.")
    else:
        print("VERDICT: Token KD via codec shows NO clear advantage.")
        print("The codec bridge does NOT enable effective knowledge transfer at this scale.")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
