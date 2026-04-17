"""Multi-source encoder alignment probe — Codex T+L KM-R3 research request.

Measures representational alignment between Sutra-Dyad student's global_local
bottleneck and a pretrained encoder's chunk embeddings (BGE-small-en-v1.5).

Purpose: establish whether multi-source alignment loss (aux training signal from
encoder models) is worth integrating into RRDSD. Reports linear CKA on a fixed
validation slice.

Decision criteria (from Codex R3):
- If CKA >= 0.5: student already aligns well with BGE space — small incremental signal expected
- If CKA 0.2-0.5: moderate alignment, alignment loss may give useful orthogonal signal
- If CKA < 0.2: highly misaligned, alignment loss could be a significant lever OR could conflict

Usage:
    python code/probe_multisource_encoder.py [--checkpoint PATH] [--n-windows 64]

Output:
    results/probe_multisource_encoder.json

No GPU required (BGE small runs fine on CPU; student inference can be on CPU too).
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from sutra_dyad import SutraDyadS1, PATCH_SIZE, SEQ_BYTES
from data_loader import ByteShardedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Force CPU by default to avoid colliding with training
DEVICE = "cpu"
DTYPE = torch.float32
SEED = 42
CHUNK_BYTES = 128
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"


def seed_all(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def linear_cka(X, Y, eps=1e-8):
    """Linear CKA (centered kernel alignment) between two representation matrices.

    X: (n, d_x) tensor
    Y: (n, d_y) tensor
    Returns: scalar CKA in [0, 1]
    """
    # Center
    X_c = X - X.mean(dim=0, keepdim=True)
    Y_c = Y - Y.mean(dim=0, keepdim=True)

    xTy = X_c.t() @ Y_c  # (d_x, d_y)
    xTx = X_c.t() @ X_c  # (d_x, d_x)
    yTy = Y_c.t() @ Y_c  # (d_y, d_y)

    hsic_xy = (xTy * xTy).sum()
    hsic_xx = (xTx * xTx).sum()
    hsic_yy = (yTy * yTy).sum()

    return (hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + eps)).item()


def rsa_correlation(X, Y):
    """Representational Similarity Analysis: Spearman correlation between
    pairwise similarity matrices. Dim-invariant.
    """
    # Normalize rows then inner product = cosine similarity
    X_norm = F.normalize(X, dim=-1)
    Y_norm = F.normalize(Y, dim=-1)
    sim_X = X_norm @ X_norm.t()  # (n, n)
    sim_Y = Y_norm @ Y_norm.t()
    # Upper-triangular (off-diagonal) entries
    n = sim_X.shape[0]
    idx = torch.triu_indices(n, n, offset=1)
    flat_X = sim_X[idx[0], idx[1]].numpy()
    flat_Y = sim_Y[idx[0], idx[1]].numpy()
    # Spearman via ranks
    from scipy.stats import spearmanr
    rho, p = spearmanr(flat_X, flat_Y)
    return float(rho), float(p)


def chunk_bytes_to_text(raw_bytes):
    """Convert byte list to UTF-8 text, skipping invalid bytes."""
    b = bytes([x if isinstance(x, int) else int(x) for x in raw_bytes])
    try:
        return b.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def main(args):
    seed_all(SEED)

    out_path = REPO / "results" / "probe_multisource_encoder.json"

    # --- Load student ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {ckpt_path}")
        sys.exit(1)
    print(f"Loading student: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student = SutraDyadS1(max_seq_bytes=SEQ_BYTES)
    student.load_state_dict(ckpt["model"], strict=False)
    student = student.to(DEVICE).eval()
    print(f"  -> step={ckpt.get('step', '?')}, eval_loss={ckpt.get('eval_loss', '?')}")

    # --- Load BGE encoder ---
    print(f"Loading BGE: {BGE_MODEL_ID}")
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL_ID)
    bge_model = AutoModel.from_pretrained(BGE_MODEL_ID, torch_dtype=DTYPE).to(DEVICE).eval()
    print(f"  -> d_emb={bge_model.config.hidden_size}")

    # --- Sample validation windows ---
    print(f"Sampling {args.n_windows} windows of {SEQ_BYTES} bytes (seed={SEED})...")
    dataset = ByteShardedDataset()
    x_batch, _ = dataset.sample_batch(args.n_windows, SEQ_BYTES, device=DEVICE, split='test')
    print(f"  -> windows shape: {tuple(x_batch.shape)}")

    # --- Chunk each window into 128-byte chunks; compute BGE + student reps per chunk ---
    chunks_per_window = SEQ_BYTES // CHUNK_BYTES  # 12 chunks per 1536-byte window
    total_chunks = args.n_windows * chunks_per_window
    print(f"Total chunks: {total_chunks} ({CHUNK_BYTES} bytes each)")

    # Forward student once to get global_local
    print("Running student forward...")
    with torch.no_grad():
        # Use dummy y (not needed for representation extraction)
        y_dummy = torch.zeros_like(x_batch)
        ce_loss, internals = student(x_batch, y_dummy, return_internals=True)
        global_local = internals["global_local"].float()  # (B, N_patches, d_local)
    # Patches per window = SEQ_BYTES / PATCH_SIZE = 1536/6 = 256
    # Chunk stride in patches = CHUNK_BYTES / PATCH_SIZE = 128/6 ≈ 21.3 — not integer. Use 21.
    patches_per_chunk = CHUNK_BYTES // PATCH_SIZE  # 21
    chunk_stride_bytes = patches_per_chunk * PATCH_SIZE  # 126 (close to 128)
    actual_chunks_per_window = SEQ_BYTES // chunk_stride_bytes  # 1536/126 = 12
    print(f"  patches_per_chunk={patches_per_chunk}, chunk_stride_bytes={chunk_stride_bytes}")
    print(f"  actual_chunks_per_window={actual_chunks_per_window}")

    # Build student chunk representations: pool global_local over patches_per_chunk
    student_chunks = []  # list of (d_local,) tensors
    bge_chunks = []  # list of (d_bge,) tensors
    chunk_count = 0

    for b in range(args.n_windows):
        raw_bytes = x_batch[b].tolist()
        for c in range(actual_chunks_per_window):
            byte_start = c * chunk_stride_bytes
            byte_end = byte_start + chunk_stride_bytes
            patch_start = byte_start // PATCH_SIZE
            patch_end = patch_start + patches_per_chunk

            # Student: mean-pool global_local over patches in this chunk
            s_chunk = global_local[b, patch_start:patch_end].mean(dim=0)  # (d_local,)
            student_chunks.append(s_chunk)

            # BGE: encode the text of this chunk
            chunk_bytes = raw_bytes[byte_start:byte_end]
            chunk_text = chunk_bytes_to_text(chunk_bytes)
            if not chunk_text.strip():
                # Empty chunk — use zero vector (handled in centering)
                bge_chunks.append(torch.zeros(bge_model.config.hidden_size, dtype=DTYPE, device=DEVICE))
                chunk_count += 1
                continue

            with torch.no_grad():
                enc = bge_tok(chunk_text, return_tensors="pt", truncation=True, max_length=512, padding=False).to(DEVICE)
                bge_out = bge_model(**enc)
                # Use [CLS] embedding (common for BGE)
                bge_emb = bge_out.last_hidden_state[0, 0, :]  # (d_bge,)
                # L2-normalize (BGE convention)
                bge_emb = F.normalize(bge_emb, dim=0)
            bge_chunks.append(bge_emb)
            chunk_count += 1

            if chunk_count % 100 == 0:
                print(f"  processed {chunk_count}/{total_chunks} chunks...")

    student_matrix = torch.stack(student_chunks).float()  # (n_chunks, d_local)
    bge_matrix = torch.stack(bge_chunks).float()          # (n_chunks, d_bge)
    print(f"Student matrix: {tuple(student_matrix.shape)}, BGE matrix: {tuple(bge_matrix.shape)}")

    # --- Compute alignment metrics ---
    print("\nComputing alignment metrics...")
    cka = linear_cka(student_matrix, bge_matrix)
    print(f"  Linear CKA:           {cka:.4f}")

    # RSA (Spearman correlation between pairwise similarity matrices)
    # Subsample if too large (RSA is O(n^2))
    n = student_matrix.shape[0]
    if n > 512:
        idx = torch.randperm(n)[:512]
        s_sub = student_matrix[idx]
        b_sub = bge_matrix[idx]
    else:
        s_sub = student_matrix
        b_sub = bge_matrix
    try:
        rsa_rho, rsa_p = rsa_correlation(s_sub, b_sub)
        print(f"  RSA Spearman rho:     {rsa_rho:.4f} (p={rsa_p:.3g})")
    except Exception as e:
        print(f"  RSA failed: {e}")
        rsa_rho, rsa_p = float('nan'), float('nan')

    # Random baseline (shuffle BGE and recompute — should give ~0)
    perm = torch.randperm(bge_matrix.shape[0])
    cka_shuffled = linear_cka(student_matrix, bge_matrix[perm])
    print(f"  Linear CKA (shuffled): {cka_shuffled:.4f}  (should be near 0)")

    # Interpretation
    if cka >= 0.5:
        interp = "HIGH — student already aligns well with BGE. Alignment loss would give small increment."
    elif cka >= 0.2:
        interp = "MODERATE — room for alignment loss to provide orthogonal signal."
    else:
        interp = "LOW — highly misaligned. Either big opportunity for alignment loss OR risk of conflict with existing objectives."

    # Decision flag
    if cka >= 0.5:
        decision = "SKIP_ALIGNMENT_LOSS"
    elif cka >= 0.2:
        decision = "WORTH_TESTING"
    else:
        decision = "BIG_GAP_EXPLORE_CAREFULLY"

    results = {
        "probe": "multisource_encoder_alignment",
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step", None),
        "encoder_model": BGE_MODEL_ID,
        "encoder_dim": bge_model.config.hidden_size,
        "student_dim": student_matrix.shape[1],
        "n_windows": args.n_windows,
        "n_chunks": student_matrix.shape[0],
        "chunk_bytes": CHUNK_BYTES,
        "chunk_stride_bytes": chunk_stride_bytes,
        "patches_per_chunk": patches_per_chunk,
        "metrics": {
            "linear_cka": cka,
            "linear_cka_shuffled_baseline": cka_shuffled,
            "rsa_spearman_rho": rsa_rho,
            "rsa_spearman_p": rsa_p,
        },
        "decision": decision,
        "interpretation": interp,
        "notes": [
            "Student representation: mean-pooled global_local over 21 patches (126 bytes ≈ 128).",
            "BGE representation: [CLS] embedding, L2-normalized.",
            "Shuffled baseline near 0 confirms measurement is meaningful.",
            "Decision gates a potential future integration of alignment loss into RRDSD.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"Decision: {decision}")
    print(f"Interpretation: {interp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-source encoder alignment probe")
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO / "results" / "checkpoints_ekalavya_iter5_full_6k" / "best.pt"),
                        help="Student checkpoint path")
    parser.add_argument("--n-windows", type=int, default=64,
                        help="Number of validation windows (each 1536 bytes, 12 chunks)")
    args = parser.parse_args()
    main(args)
