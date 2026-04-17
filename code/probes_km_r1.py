"""Probes for Codex T+L R1 session — Sutra KD mechanism decision.

Three offline probes:
  1. Kuhnian Survival — does byte-level covering preserve teacher information?
  2. Offline ZPD Utility — is mid-entropy x high-advantage where learning happens?
  3. Pythia Checkpoint Ladder — are intermediate Pythia checkpoints a free win?

Runs on GPU (teachers 4-bit quantized). Writes defensively to JSON after each probe.

Usage:
    python code/probes_km_r1.py
"""

import os
import sys
import math
import json
import time
import random
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

# Import from sutra_dyad — reuse covering + student model infrastructure
from sutra_dyad import (
    SutraDyadS1,
    _build_covering_tables,
    _covering_one_sequence,
    _get_teacher_targets_covering_batched,
)
from data_loader import ByteShardedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
SEED = 42
SEQ_BYTES = 1536

CKPT_BEST = REPO / "results" / "checkpoints_ekalavya_iter5_full_6k" / "best.pt"
CKPT_1000 = REPO / "results" / "checkpoints_ekalavya_iter5_full_6k" / "step_1000.pt"

RESULTS_DIR = REPO / "results"
LOG_PATH = RESULTS_DIR / "probe_km_r1.log"
OUT_KUHN = RESULTS_DIR / "probe_km_r1_kuhnian.json"
OUT_ZPD = RESULTS_DIR / "probe_km_r1_zpd.json"
OUT_LADDER = RESULTS_DIR / "probe_km_r1_pythia_ladder.json"

SMOL_ID = "HuggingFaceTB/SmolLM2-1.7B"
PYTHIA_ID = "EleutherAI/pythia-1.4b"

# ---- Logging ----
_log_f = None

def log(msg):
    global _log_f
    if _log_f is None:
        _log_f = open(LOG_PATH, "w", encoding="utf-8")
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_f.write(line + "\n")
    _log_f.flush()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---- Shared infrastructure ----

def load_student(ckpt_path, device=DEVICE):
    """Load SutraDyadS1 student checkpoint (weights-only for model)."""
    log(f"Loading student: {ckpt_path.name}")
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = SutraDyadS1(max_seq_bytes=SEQ_BYTES)
    model.load_state_dict(ck["model"], strict=False)
    model = model.to(device).eval()
    step = ck.get("step", "?")
    log(f"  -> step={step}, eval_loss={ck.get('eval_loss','?')}")
    return model


def load_teacher(teacher_id, device=DEVICE, revision=None):
    """Load a teacher with 4-bit quantization (matches iter5 setup)."""
    log(f"Loading teacher: {teacher_id}{' @' + revision if revision else ''} (4-bit)")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tok_kwargs = {}
    model_kwargs = dict(quantization_config=bnb_cfg, device_map=device)
    if revision is not None:
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(teacher_id, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(teacher_id, **model_kwargs)
    model.eval()
    vocab = model.config.vocab_size
    log(f"  -> vocab={vocab}")
    return model, tokenizer, vocab


def sample_validation_windows(n_windows, seq_len=SEQ_BYTES, seed=SEED):
    """Deterministically sample n_windows byte windows from test split."""
    dataset = ByteShardedDataset()
    # Reset RNG before sampling so everything is deterministic
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ByteShardedDataset.sample_batch uses the Python random module.
    # Collect windows via repeated sample_batch(batch_size=1).
    x_list, y_list = [], []
    # To avoid the 'test' split having only a fraction of the data, we iterate.
    while len(x_list) < n_windows:
        bs = min(32, n_windows - len(x_list))
        x, y = dataset.sample_batch(bs, seq_len, device='cpu', split='test')
        for i in range(x.shape[0]):
            x_list.append(x[i])
            y_list.append(y[i])
            if len(x_list) >= n_windows:
                break
    x_all = torch.stack(x_list[:n_windows]).long()
    y_all = torch.stack(y_list[:n_windows]).long()
    return x_all, y_all


# ---- Student per-position log-prob ----

@torch.no_grad()
def student_log_probs(model, byte_ids_batch, device=DEVICE, batch_size=4):
    """Compute student log P(byte_t | prefix) for each (b, t) in batch. (B, T, 256)"""
    log_p_chunks = []
    N = byte_ids_batch.shape[0]
    for i in range(0, N, batch_size):
        chunk = byte_ids_batch[i:i+batch_size].to(device)
        with torch.amp.autocast('cuda', dtype=DTYPE):
            logits = model(chunk)  # (B, T, 256)
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_p_chunks.append(log_p.cpu())
    return torch.cat(log_p_chunks, dim=0)  # (N, T, 256)


# ---- PROBE 1: Kuhnian Survival ----

def greedy_tokenization_prefix_logp(teacher_model, tokenizer, prefix_text_bytes,
                                    full_context_ids, teacher_device=DEVICE,
                                    context_attention_mask=None):
    """Compute log P_tok of a byte prefix under the teacher.

    We score the leftmost greedy tokenization of the byte prefix, given the
    tokenized context that precedes it.

    prefix_text_bytes: bytes object of length h (the actual future bytes).
    full_context_ids: (1, T_ctx) tokenizer ids for the prefix context (bytes before position).
    Returns:
        log_p_total: scalar log P_tok(prefix)
        n_tokens_covered: how many teacher tokens the prefix spans
    """
    # Decode as text and append to context. Work in text space: find the tokens
    # whose concatenation matches `prefix_text_bytes`.
    try:
        prefix_text = prefix_text_bytes.decode('utf-8', errors='replace')
    except Exception:
        prefix_text = prefix_text_bytes.decode('utf-8', errors='replace')

    # Tokenize prefix alone (standalone) to get candidate token IDs
    # Then compute, given context, the log-prob sequentially.
    # NOTE: This is an approximation for cross-tokenizer scoring — the
    # cleanest approach is to tokenize the joint string (context + prefix) and
    # identify which new tokens correspond to the prefix.
    return prefix_text  # actual logic below uses joint tokenization


@torch.no_grad()
def compute_prefix_tok_nll_batched(teacher_model, tokenizer, context_bytes_list,
                                    prefix_h_list, device=DEVICE,
                                    max_context_bytes=1024):
    """For each (context_bytes, prefix_bytes), compute -log P_tok(prefix | context).

    Strategy: tokenize joint (context + prefix), identify which new tokens cover
    the prefix bytes, sum log-probs over those tokens under the teacher.

    Args:
        context_bytes_list: list of byte-tuples (the prefix before the probe position)
        prefix_h_list:      list of byte-tuples (the h bytes starting at probe position)

    Returns:
        nll_tok_list: list[float] length = len(context_bytes_list)
        n_tokens_list: list[int] number of teacher tokens the prefix spans
    """
    # Prepare joint strings
    N = len(context_bytes_list)
    joint_texts = []
    context_texts = []
    for ctx_bytes, pref_bytes in zip(context_bytes_list, prefix_h_list):
        # Limit context to avoid OOM
        if len(ctx_bytes) > max_context_bytes:
            ctx_bytes = ctx_bytes[-max_context_bytes:]
        ctx_text = bytes(ctx_bytes).decode('utf-8', errors='replace')
        pref_text = bytes(pref_bytes).decode('utf-8', errors='replace')
        joint_texts.append(ctx_text + pref_text)
        context_texts.append(ctx_text)

    # Tokenize in small batches for memory
    nll_list = []
    ntok_list = []
    batch = 8
    for i in range(0, N, batch):
        j = min(i + batch, N)
        joint_batch = joint_texts[i:j]
        ctx_batch = context_texts[i:j]

        # Encode separately to find boundary
        enc_joint = tokenizer(joint_batch, padding=True, return_tensors='pt', truncation=False)
        enc_ctx = tokenizer(ctx_batch, padding=True, return_tensors='pt', truncation=False)

        input_ids = enc_joint['input_ids'].to(device)
        attn = enc_joint['attention_mask'].to(device)

        # Teacher forward to get log-probs over next-token distribution
        out = teacher_model(input_ids=input_ids, attention_mask=attn, use_cache=False)
        logits = out.logits  # (B, T, V)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        for b in range(j - i):
            joint_ids = enc_joint['input_ids'][b]
            joint_attn = enc_joint['attention_mask'][b]
            ctx_ids = enc_ctx['input_ids'][b]
            ctx_attn = enc_ctx['attention_mask'][b]

            # Real (non-pad) token counts
            n_joint = int(joint_attn.sum().item())
            n_ctx = int(ctx_attn.sum().item())

            # The "new" tokens added by appending the prefix are at positions
            # [n_ctx .. n_joint) in joint_ids. Their log-prob comes from
            # logits at position [n_ctx - 1 .. n_joint - 1).
            # However, tokenization of (ctx + prefix) might re-tokenize the
            # boundary. The safest approximation: find the longest common
            # prefix between ctx_ids and joint_ids (both non-pad), then the
            # remaining joint tokens are the ones covering the prefix.
            ctx_real = ctx_ids[:n_ctx].tolist()
            joint_real = joint_ids[:n_joint].tolist()
            k = 0
            while k < len(ctx_real) and k < len(joint_real) and ctx_real[k] == joint_real[k]:
                k += 1
            # Tokens [k, n_joint) are the prefix-covering tokens.
            # Sum log-prob: sum_{t=k}^{n_joint-1} log_p[t-1, joint_real[t]]
            # where log_p is at position t-1 (predicts token at position t).
            if k >= n_joint:
                # No new tokens — prefix was absorbed into context (unlikely)
                nll_list.append(0.0)
                ntok_list.append(0)
                continue
            total_lp = 0.0
            for t in range(k, n_joint):
                if t == 0:
                    # Corner case: no predecessor — approximate with uniform
                    total_lp += -math.log(logits.shape[-1])
                else:
                    total_lp += float(log_probs[b, t - 1, joint_real[t]].item())
            nll = -total_lp
            nll_list.append(nll)
            ntok_list.append(n_joint - k)

    return nll_list, ntok_list


@torch.no_grad()
def compute_prefix_cov_nll(covering_tables, probs_np, token_ids, raw_bytes,
                           pos_in_bytes, h, max_depth=8):
    """Compute -log P_cov(prefix_h) using covering decomposition.

    P_cov(prefix_h | context) = prod_k q_cov(b_k | prefix_{k-1})
    where q_cov is the byte-conditional derived from teacher probs.

    Because covering gives per-byte conditionals at each byte position, we can
    sum -log of the conditional for each byte in the prefix.

    Args:
        covering_tables: output of _build_covering_tables for the teacher
        probs_np: (T_tok, V) teacher probs under (softmax of temperature-scaled logits)
                  for this sequence. NOTE: MUST be T=1.0 (not KD temperature) for honest comparison.
        token_ids: list of teacher token IDs for the full sequence
        raw_bytes: list of bytes for the full sequence
        pos_in_bytes: byte position in raw_bytes
        h: prefix length
        max_depth: covering max depth

    Returns:
        (nll_cov, valid): nll_cov float (nan if not fully covered), valid bool
    """
    # Run the covering computation on this sequence and extract byte_probs at
    # positions [pos_in_bytes .. pos_in_bytes + h).
    byte_probs_np, byte_mask_np, byte_offsets, cov_ratio = _covering_one_sequence(
        0, probs_np, token_ids, raw_bytes,
        covering_tables["token_byte_seqs"], covering_tables["first_byte_matrix_np"],
        covering_tables, max_depth,
    )
    total_lp = 0.0
    all_covered = True
    for k in range(h):
        p = pos_in_bytes + k
        if p >= byte_probs_np.shape[0] or not byte_mask_np[p]:
            all_covered = False
            break
        b_val = raw_bytes[p]
        p_b = float(byte_probs_np[p, b_val])
        if p_b <= 0:
            all_covered = False
            break
        total_lp += math.log(p_b)
    if not all_covered:
        return math.nan, False
    return -total_lp, True


def probe_1_kuhnian(n_positions=2048, h_values=(1, 2, 4, 8), seed=SEED,
                    n_windows=256):
    """Probe 1: Kuhnian Survival.

    Sample n_windows windows of SEQ_BYTES bytes. Pick n_positions fixed positions
    total (distributed across windows). For each teacher + h, compute
    R_h = (NLL_student - NLL_cov) / (NLL_student - NLL_tok) and decide.
    """
    log("=" * 60)
    log("PROBE 1: KUHNIAN SURVIVAL")
    log("=" * 60)
    seed_all(seed)

    # Sample validation windows
    log(f"Sampling {n_windows} validation windows, {n_positions} probe positions...")
    x_windows, y_windows = sample_validation_windows(n_windows, SEQ_BYTES, seed)
    log(f"  windows shape: {x_windows.shape}")

    # Pick positions: distribute n_positions positions across windows.
    # Make sure there's room for max(h)=8 future bytes + at least 64 bytes context.
    min_ctx = 64
    max_h = max(h_values)
    valid_start = min_ctx
    valid_end = SEQ_BYTES - max_h - 1
    pos_per_window = n_positions // n_windows
    extra = n_positions - pos_per_window * n_windows

    rng = random.Random(seed)
    positions_per_window = []
    for w in range(n_windows):
        k = pos_per_window + (1 if w < extra else 0)
        # Sample k positions in [valid_start, valid_end)
        ps = rng.sample(range(valid_start, valid_end), min(k, valid_end - valid_start))
        positions_per_window.append(sorted(ps))

    total_positions = sum(len(ps) for ps in positions_per_window)
    log(f"  Total positions: {total_positions}")

    # Load student and compute per-position log-probs
    student = load_student(CKPT_BEST)
    log("Computing student log-probs on all windows...")
    x_windows_dev = x_windows.to(DEVICE)
    y_windows_dev = y_windows.to(DEVICE)
    student_logp = student_log_probs(student, x_windows_dev, batch_size=4)  # (N, T, 256) CPU
    # Free student after use
    del student
    torch.cuda.empty_cache(); gc.collect()

    # Student NLL per position for prefix of length h: sum of -log p_s(b_k | prefix_{k-1})
    # For h=1: just -log p_s(y[pos]) predicted from byte at pos in window (x).
    # Student forward uses x as input, predicts next byte. Student_logp[b, t] is
    # the prediction at position t (for y[t]). So for probe position p:
    #    p_s(b_k = y[p+k]) = log_p[b, p+k, y[p+k]]   (approximation using autoregressive log-probs from one forward pass)
    # This is exact because the student is autoregressive with causal attention.
    log("Computing student NLL for all (position, h) pairs...")
    student_nll_by_h = {h: [] for h in h_values}
    position_meta = []  # list of (win_idx, pos, ctx_bytes, prefix_bytes_by_h)
    for win_idx, ps in enumerate(positions_per_window):
        for pos in ps:
            ctx_bytes = tuple(x_windows[win_idx, :pos].tolist())
            # The true next bytes at pos are y_windows[win_idx, pos-1] to y[pos+h-2]?
            # Actually: student predicts at index t the byte at t+1 in x, which equals y[t].
            # So student_logp[b, t, v] = log P(x[t+1] = v | x[0..t]) = log P(y[t] = v | x[0..t]).
            # For prefix h starting at byte position pos (meaning bytes x[pos], x[pos+1], ...):
            # NO — the "probe position" means predicting bytes at/after pos.
            # Let's define: the h bytes being scored are x[pos], x[pos+1], ..., x[pos+h-1].
            # These are predicted given context x[0..pos-1], so:
            #   log P(x[pos] = ...) = student_logp[b, pos-1, x[pos]]  (predicts x[pos] from x[0..pos-1])
            # i.e., we use student_logp[win_idx, pos-1+k, x[pos+k]] for k in 0..h-1.
            prefix_bytes_per_h = {}
            for h in h_values:
                nll = 0.0
                for k in range(h):
                    t_pred = pos - 1 + k  # predicts x[pos+k]
                    target_byte = int(x_windows[win_idx, pos + k].item())
                    if t_pred < 0 or t_pred >= student_logp.shape[1]:
                        nll = float('nan')
                        break
                    lp = float(student_logp[win_idx, t_pred, target_byte].item())
                    nll -= lp
                student_nll_by_h[h].append(nll)
                prefix_bytes_per_h[h] = tuple(x_windows[win_idx, pos:pos+h].tolist())
            position_meta.append((win_idx, pos, ctx_bytes, prefix_bytes_per_h))

    log(f"  Student NLL computed for {len(position_meta)} positions")

    # Free student log-probs (large)
    del student_logp
    gc.collect()

    # Now iterate over teachers
    teachers_info = [
        ("teacher_smol", SMOL_ID),
        ("teacher_pythia", PYTHIA_ID),
    ]

    per_teacher_results = {}
    for teacher_key, teacher_id in teachers_info:
        log(f"\n--- Teacher: {teacher_id} ---")
        teacher, tokenizer, vocab = load_teacher(teacher_id)

        # Build covering tables for this teacher
        log(f"Building covering tables (vocab={vocab})...")
        cov_tables = _build_covering_tables(tokenizer, vocab, device='cpu')
        log(f"  {cov_tables['n_prefixes']} prefixes, max {cov_tables['max_token_bytes']} bytes/token")

        # For each window, we need the teacher probs (T=1.0 softmax of logits)
        # over the full sequence ONCE, then extract at probe positions.
        log("Running teacher forward on validation windows...")
        teacher_results_per_window = []  # list of (probs_np, token_ids, raw_bytes)
        for win_idx in range(n_windows):
            raw_bytes = x_windows[win_idx].tolist()
            text = bytes(raw_bytes).decode('utf-8', errors='replace')
            enc = tokenizer(text, return_tensors='pt', truncation=False)
            input_ids = enc['input_ids'].to(DEVICE)
            with torch.inference_mode():
                out = teacher(input_ids, use_cache=False)
                logits = out.logits[0]
            probs = F.softmax(logits.float(), dim=-1)  # T=1.0 (honest scoring)
            probs_np = probs.cpu().numpy()
            token_ids = input_ids[0].cpu().tolist()
            teacher_results_per_window.append((probs_np, token_ids, raw_bytes))
            if (win_idx + 1) % 32 == 0:
                log(f"  teacher forward: {win_idx+1}/{n_windows}")

        # Compute NLL_tok and NLL_cov for each position, each h
        log("Computing token-space and covering NLLs per position...")
        nll_cov_by_h = {h: [] for h in h_values}
        nll_tok_by_h = {h: [] for h in h_values}

        # Prepare all context/prefix pairs and batch teacher forward for NLL_tok
        # Group by h to allow efficient batching
        for h in h_values:
            contexts = []
            prefixes = []
            meta_idx = []
            for idx, (win_idx, pos, ctx_bytes, prefix_per_h) in enumerate(position_meta):
                contexts.append(ctx_bytes)
                prefixes.append(prefix_per_h[h])
                meta_idx.append(idx)
            log(f"  h={h}: computing NLL_tok over {len(contexts)} positions...")
            nll_tok_list, _ = compute_prefix_tok_nll_batched(
                teacher, tokenizer, contexts, prefixes, device=DEVICE,
                max_context_bytes=1024,
            )
            nll_tok_by_h[h] = nll_tok_list

        # Covering NLL — uses the per-window teacher probs we already computed
        log("  Computing NLL_cov...")
        for h in h_values:
            cov_list = []
            for idx, (win_idx, pos, ctx_bytes, prefix_per_h) in enumerate(position_meta):
                probs_np, token_ids, raw_bytes = teacher_results_per_window[win_idx]
                nll_cov, valid = compute_prefix_cov_nll(
                    cov_tables, probs_np, token_ids, raw_bytes,
                    pos_in_bytes=pos, h=h, max_depth=8,
                )
                cov_list.append(nll_cov if valid else float('nan'))
            nll_cov_by_h[h] = cov_list

        # Compute R_h for each h, filtering NaN/inf
        per_h_summary = {}
        for h in h_values:
            R_list = []
            nll_stud_list = student_nll_by_h[h]
            for idx in range(len(position_meta)):
                ns = nll_stud_list[idx]
                nc = nll_cov_by_h[h][idx]
                nt = nll_tok_by_h[h][idx]
                if not all(np.isfinite([ns, nc, nt])):
                    continue
                denom = ns - nt
                if abs(denom) < 1e-6:
                    continue
                R = (ns - nc) / denom
                R_list.append(R)
            R_arr = np.array(R_list, dtype=np.float64)
            if len(R_arr) == 0:
                per_h_summary[h] = {"n_valid": 0}
                continue
            per_h_summary[h] = {
                "n_valid": int(len(R_arr)),
                "R_h_mean": float(np.mean(R_arr)),
                "R_h_median": float(np.median(R_arr)),
                "R_h_p10": float(np.percentile(R_arr, 10)),
                "R_h_p90": float(np.percentile(R_arr, 90)),
                "nll_student_mean": float(np.nanmean([student_nll_by_h[h][i] for i in range(len(position_meta))])),
                "nll_cov_mean": float(np.nanmean(nll_cov_by_h[h])),
                "nll_tok_mean": float(np.nanmean(nll_tok_by_h[h])),
            }
            log(f"  h={h}: R_mean={per_h_summary[h]['R_h_mean']:.3f}, "
                f"median={per_h_summary[h]['R_h_median']:.3f}, n_valid={per_h_summary[h]['n_valid']}")
        per_teacher_results[teacher_key] = per_h_summary

        # Free teacher memory before loading next
        del teacher, tokenizer, cov_tables, teacher_results_per_window
        torch.cuda.empty_cache(); gc.collect()

    # Assemble per_h list of dicts (expected format)
    per_h_list = []
    for h in h_values:
        entry = {"h": h}
        for tk in per_teacher_results:
            entry[tk] = per_teacher_results[tk].get(h, {"n_valid": 0})
        per_h_list.append(entry)

    # Decision
    # Use mean R_h at h=8 across both teachers
    r_values_h8 = []
    for tk in per_teacher_results:
        if 8 in per_teacher_results[tk] and "R_h_mean" in per_teacher_results[tk][8]:
            r_values_h8.append(per_teacher_results[tk][8]["R_h_mean"])
    avg_r_h8 = float(np.mean(r_values_h8)) if r_values_h8 else float('nan')

    r_values_h1 = []
    for tk in per_teacher_results:
        if 1 in per_teacher_results[tk] and "R_h_mean" in per_teacher_results[tk][1]:
            r_values_h1.append(per_teacher_results[tk][1]["R_h_mean"])
    avg_r_h1 = float(np.mean(r_values_h1)) if r_values_h1 else float('nan')

    min_r_any_h = 1e9
    max_r_any_h = -1e9
    for h in h_values:
        for tk in per_teacher_results:
            if h in per_teacher_results[tk] and "R_h_mean" in per_teacher_results[tk][h]:
                rv = per_teacher_results[tk][h]["R_h_mean"]
                min_r_any_h = min(min_r_any_h, rv)
                max_r_any_h = max(max_r_any_h, rv)

    if min_r_any_h >= 0.8:
        decision = "WEAKENED"
        reasoning = f"R_h >= 0.8 for all (teacher, h<=8) pairs: min R={min_r_any_h:.3f}. Covering preserves most teacher information."
    elif max_r_any_h < 0.3:
        decision = "STRENGTHENED"
        reasoning = f"R_h < 0.3 for all (teacher, h<=8) pairs: max R={max_r_any_h:.3f}. Kuhnian incommensurability claim gains real support."
    else:
        decision = "UNCLEAR"
        reasoning = (
            f"Mixed results: min R={min_r_any_h:.3f}, max R={max_r_any_h:.3f}, "
            f"avg R_h=1 across teachers={avg_r_h1:.3f}, avg R_h=8={avg_r_h8:.3f}. "
            f"Need more data or h-dependent interpretation."
        )

    result = {
        "probe": "kuhnian_survival",
        "n_positions": total_positions,
        "seed": seed,
        "checkpoint": str(CKPT_BEST.name),
        "teachers": [SMOL_ID, PYTHIA_ID],
        "h_values": list(h_values),
        "per_h": per_h_list,
        "decision": decision,
        "decision_reasoning": reasoning,
        "notes": (
            "NLL_tok computed from joint-tokenization matching: tokenize (ctx + prefix), "
            "sum log-probs over tokens that appear in joint but not in ctx (longest-common-prefix split). "
            "When a boundary byte is absorbed into a context token by the teacher tokenizer, "
            "this may slightly over-attribute log-prob to the prefix — defensible but approximate. "
            "NLL_cov uses teacher probs at T=1.0 (honest scoring, not KD temperature 1.3). "
            "NLL_student uses one student forward pass (exact autoregressive)."
        ),
    }
    with open(OUT_KUHN, "w") as f:
        json.dump(result, f, indent=2)
    log(f"PROBE 1 COMPLETE: decision={decision}")
    log(f"  Wrote: {OUT_KUHN}")
    return result


# ---- PROBE 2: Offline ZPD Utility ----

def probe_2_zpd(n_windows=256, seed=SEED):
    """Probe 2: Offline ZPD Utility.

    For best.pt and step_1000.pt, compute per-position:
      H_s = student entropy
      JSD_m,t = JSD(p_teacher || stopgrad(p_student)) for each teacher
      A_m,t = log p_teacher(y_true) - log p_student(y_true)
      UG_t = iter5's uncertainty gate value
      grad_norm_t = ||d KD_loss / d z_student_t||

    Bucket by (H_s quantile, best-teacher-A quantile) -> 3x3 buckets.
    Report grad_norm, JSD, KD gradient share per bucket + top-15% utility overlap
    with q60-q95 entropy band.
    """
    log("=" * 60)
    log("PROBE 2: OFFLINE ZPD UTILITY")
    log("=" * 60)
    seed_all(seed)

    # Sample windows
    log(f"Sampling {n_windows} windows...")
    x_windows, y_windows = sample_validation_windows(n_windows, SEQ_BYTES, seed)

    # Precompute teachers and covering once (shared across checkpoints)
    log("Loading teachers (SmolLM2 + Pythia-1.4B)...")
    teachers_info = []
    for teacher_id in [SMOL_ID, PYTHIA_ID]:
        t_model, t_tok, t_vocab = load_teacher(teacher_id)
        log(f"  Building covering tables for {teacher_id}...")
        t_cov = _build_covering_tables(t_tok, t_vocab, device='cpu')
        teachers_info.append({
            "id": teacher_id, "model": t_model, "tokenizer": t_tok,
            "vocab": t_vocab, "covering": t_cov,
        })

    # Teacher byte probs per window: use _get_teacher_targets_covering_batched at T=1.0
    # (honest probs, not distillation temperature). Process in small batches.
    kd_temperature = 1.0  # honest: measure actual teacher distribution
    teacher_probs_per_t = []  # list per teacher -> (N_win, T, 256), mask same shape
    teacher_masks_per_t = []
    log("Computing teacher byte probs on all windows...")
    for ti, t_info in enumerate(teachers_info):
        log(f"  Teacher {ti}: {t_info['id']}")
        all_probs = []
        all_masks = []
        batch_raw = []
        batch_size = 4
        for win_idx in range(n_windows):
            batch_raw.append(x_windows[win_idx].tolist())
            if len(batch_raw) == batch_size or win_idx == n_windows - 1:
                results = _get_teacher_targets_covering_batched(
                    t_info["model"], t_info["tokenizer"], t_info["covering"],
                    batch_raw, DEVICE, temperature=kd_temperature,
                    extract_hidden=False, max_depth=8,
                )
                for r in results:
                    p = r["byte_probs"].cpu()  # (T, 256)
                    m = r["byte_mask"].cpu()   # (T,)
                    # Pad/trim to SEQ_BYTES
                    if p.shape[0] < SEQ_BYTES:
                        pad_p = torch.zeros(SEQ_BYTES - p.shape[0], 256, dtype=p.dtype)
                        pad_m = torch.zeros(SEQ_BYTES - m.shape[0], dtype=m.dtype)
                        p = torch.cat([p, pad_p], dim=0)
                        m = torch.cat([m, pad_m], dim=0)
                    elif p.shape[0] > SEQ_BYTES:
                        p = p[:SEQ_BYTES]
                        m = m[:SEQ_BYTES]
                    all_probs.append(p)
                    all_masks.append(m)
                batch_raw = []
        tp = torch.stack(all_probs)    # (N, T, 256)
        tm = torch.stack(all_masks)    # (N, T)
        teacher_probs_per_t.append(tp)
        teacher_masks_per_t.append(tm)
        log(f"    teacher probs shape: {tp.shape}, mask density: {tm.float().mean():.3f}")

    # Free teacher models now that we have probs cached
    for t_info in teachers_info:
        del t_info["model"]
    torch.cuda.empty_cache(); gc.collect()

    # Process both checkpoints
    per_checkpoint = {}
    for ckpt_name, ckpt_path in [("best.pt", CKPT_BEST), ("step_1000.pt", CKPT_1000)]:
        log(f"\n--- Checkpoint: {ckpt_name} ---")
        student = load_student(ckpt_path)

        log("Computing student log-probs on all windows...")
        student_logp = student_log_probs(student, x_windows.to(DEVICE), batch_size=4)  # (N, T, 256) CPU

        # H_s: student entropy
        s_probs = student_logp.exp()
        H_s = -(s_probs * student_logp).sum(dim=-1)  # (N, T)

        # For each teacher compute JSD(p_t || p_s), A = log p_t(y) - log p_s(y), UG gate
        # UG formula from iter5: gate = t_conf * (1 - s_match)^exp (exp=2.0 at max, renorm + clamp 1.5)
        # For offline analysis we use exp=2.0 (steady-state).
        # Combine "best-teacher" A at each position -> A_best = max_m A_m
        n_teachers = len(teachers_info)
        N, T = H_s.shape
        A_per_t = torch.zeros(n_teachers, N, T)
        JSD_per_t = torch.zeros(n_teachers, N, T)
        UG_per_t = torch.zeros(n_teachers, N, T)
        mask_per_t = torch.zeros(n_teachers, N, T, dtype=torch.bool)

        # Compute shifted targets: student_logp at position t-1 predicts y_true at position t in x space.
        # The teacher byte_probs at byte position p in raw_bytes represents p(byte at p | bytes 0..p-1).
        # This aligns with student_logp[t=p-1] -> prediction of x[p]. So we align by shifting:
        #   teacher_prob at t corresponds to student_logp at t-1
        # Equivalently: student_logp[b, t, v] is prediction of x[t+1].
        # Teacher byte_probs[b, t, v] is prediction of x[t+1] given x[0..t] — SAME alignment.
        # So no shift needed; they're both at position t predicting x[t+1] (= y[t] for training).
        # Actually: our covering produces teacher byte probs at byte position 'p' that represent
        # the conditional P(x[p+1] = v | x[0..p]) — i.e., prediction OF the byte at p+1.
        # Looking at _covering_one_sequence: byte_pos = token_start - 1, and it stores depth-0
        # (first byte) AND subsequent bytes inside the next token. That byte_pos is the position
        # *before* the predicted byte. So:
        #   byte_probs[pos] represents P(byte at pos+1 | bytes 0..pos)
        # while student_logp[pos] represents P(x[pos+1] | x[0..pos]). Same semantics.

        for ti in range(n_teachers):
            t_probs = teacher_probs_per_t[ti]  # (N, T, 256)
            t_mask = teacher_masks_per_t[ti]   # (N, T)
            # JSD
            m_mix = 0.5 * (t_probs + s_probs).clamp_min(1e-10)
            t_safe = t_probs.clamp_min(1e-10)
            s_safe = s_probs.clamp_min(1e-10)
            jsd = 0.5 * (
                (t_probs * (t_safe / m_mix).log()).sum(-1)
                + (s_probs * (s_safe / m_mix).log()).sum(-1)
            )
            JSD_per_t[ti] = jsd

            # Actual-label advantage A = log p_t(y_true) - log p_s(y_true)
            # y_true at position t is y_windows[b, t] == x_windows[b, t+1].
            # student_logp[b, t, v] predicts x[t+1] = y_windows[b, t].
            # teacher byte_probs[b, t, v] similarly predicts x[t+1]; so y index = y_windows[b, t].
            y_idx = y_windows.long().unsqueeze(-1)  # (N, T, 1)
            log_p_s_y = student_logp.gather(-1, y_idx).squeeze(-1)
            t_log_probs = t_safe.log()
            log_p_t_y = t_log_probs.gather(-1, y_idx).squeeze(-1)
            A = log_p_t_y - log_p_s_y
            A_per_t[ti] = A

            # UG gate
            t_conf = t_probs.max(dim=-1).values
            teacher_top = t_probs.argmax(dim=-1, keepdim=True)
            s_match = s_probs.gather(-1, teacher_top).squeeze(-1)
            ug_exp = 2.0
            ug_raw = t_conf * (1.0 - s_match).pow(ug_exp)
            # Renorm per-batch (match iter5)
            mask_f = t_mask.float()
            raw_mean = (ug_raw * mask_f).sum() / mask_f.sum().clamp_min(1e-10)
            ug = ug_raw / raw_mean.clamp_min(1e-10)
            ug = ug.clamp(max=1.5)
            UG_per_t[ti] = ug

            mask_per_t[ti] = t_mask

        # Aggregate: best teacher per position (by A)
        A_best, best_teacher_idx = A_per_t.max(dim=0)  # (N, T)
        JSD_best = JSD_per_t.gather(0, best_teacher_idx.unsqueeze(0)).squeeze(0)
        UG_best = UG_per_t.gather(0, best_teacher_idx.unsqueeze(0)).squeeze(0)
        any_mask = mask_per_t.any(dim=0)  # (N, T)

        # grad_norm per position: ||d KD / d z_student_t||
        # For KL(p_teacher || p_student) = sum_v p_t log(p_t/p_s):
        #   d/dz_s[v] = p_s[v] - p_t[v]   (this is for cross-entropy form;
        #   for KL(p || q) where q=softmax(z): dKL/dz = q - p)
        # So grad_norm = sqrt( sum_v (p_s[v] - p_t[v])^2 )
        grad_norm_per_t = []
        for ti in range(n_teachers):
            t_probs = teacher_probs_per_t[ti]
            gn = ((s_probs - t_probs).pow(2).sum(-1)).sqrt()  # (N, T)
            grad_norm_per_t.append(gn)
        # Use grad_norm for best teacher at each position
        grad_norm_all = torch.stack(grad_norm_per_t, dim=0)  # (Nt, N, T)
        grad_norm_best = grad_norm_all.gather(0, best_teacher_idx.unsqueeze(0)).squeeze(0)

        # Bucket by (H_s quantile, A_best quantile)
        # Only consider positions where any teacher provides signal
        flat_mask = any_mask.flatten().bool()
        flat_H = H_s.flatten()[flat_mask]
        flat_A = A_best.flatten()[flat_mask]
        flat_JSD = JSD_best.flatten()[flat_mask]
        flat_UG = UG_best.flatten()[flat_mask]
        flat_gn = grad_norm_best.flatten()[flat_mask]

        log(f"  Bucketing {flat_H.numel()} supervised positions...")

        # Quantiles
        q33_H = torch.quantile(flat_H, 0.33).item()
        q66_H = torch.quantile(flat_H, 0.66).item()
        q60_H = torch.quantile(flat_H, 0.60).item()
        q95_H = torch.quantile(flat_H, 0.95).item()
        q33_A = torch.quantile(flat_A, 0.33).item()
        q66_A = torch.quantile(flat_A, 0.66).item()

        def ent_bucket(h):
            if h < q33_H: return "low"
            if h < q66_H: return "mid"
            return "high"
        def adv_bucket(a):
            if a < q33_A: return "low"
            if a < q66_A: return "mid"
            return "high"

        buckets_data = {}
        flat_H_arr = flat_H.numpy()
        flat_A_arr = flat_A.numpy()
        flat_JSD_arr = flat_JSD.numpy()
        flat_UG_arr = flat_UG.numpy()
        flat_gn_arr = flat_gn.numpy()

        total_grad_sum = float(flat_gn_arr.sum())
        # Vectorized bucket assignment
        e_idx = np.where(flat_H_arr < q33_H, 0, np.where(flat_H_arr < q66_H, 1, 2))
        a_idx = np.where(flat_A_arr < q33_A, 0, np.where(flat_A_arr < q66_A, 1, 2))
        ent_labels = ["low", "mid", "high"]
        adv_labels = ["low", "mid", "high"]
        buckets_list = []
        for ei in range(3):
            for ai in range(3):
                sel = (e_idx == ei) & (a_idx == ai)
                if sel.sum() == 0:
                    buckets_list.append({
                        "entropy": ent_labels[ei], "advantage": adv_labels[ai],
                        "count": 0, "grad_norm_mean": 0.0, "jsd_mean": 0.0,
                        "ug_mean": 0.0, "A_mean": 0.0, "kd_gradient_share": 0.0,
                    })
                    continue
                count = int(sel.sum())
                gn_mean = float(flat_gn_arr[sel].mean())
                jsd_mean = float(flat_JSD_arr[sel].mean())
                ug_mean = float(flat_UG_arr[sel].mean())
                A_mean = float(flat_A_arr[sel].mean())
                grad_share = float(flat_gn_arr[sel].sum() / max(total_grad_sum, 1e-10))
                buckets_list.append({
                    "entropy": ent_labels[ei], "advantage": adv_labels[ai],
                    "count": count, "grad_norm_mean": gn_mean, "jsd_mean": jsd_mean,
                    "ug_mean": ug_mean, "A_mean": A_mean, "kd_gradient_share": grad_share,
                })
                log(f"    ent={ent_labels[ei]:<4} adv={adv_labels[ai]:<4}: "
                    f"n={count:>6}, grad={gn_mean:.3f}, share={grad_share:.3f}, "
                    f"JSD={jsd_mean:.3f}, UG={ug_mean:.3f}, A={A_mean:.3f}")

        # ZAR mask analysis: top-15% by utility U = max(0, A) * JSD
        U = np.maximum(flat_A_arr, 0) * flat_JSD_arr
        top15_cutoff = np.quantile(U, 0.85)
        top15_mask = U >= top15_cutoff
        in_band_mask = (flat_H_arr >= q60_H) & (flat_H_arr <= q95_H)
        top15_in_band = float((top15_mask & in_band_mask).sum() / max(top15_mask.sum(), 1))
        top15_outside_band = float((top15_mask & ~in_band_mask).sum() / max(top15_mask.sum(), 1))

        # UG overlap: UG > some threshold (e.g., > 1.0 means above mean)
        ug_active_mask = flat_UG_arr > 1.0
        ug_overlap = float((top15_mask & ug_active_mask).sum() / max(top15_mask.sum(), 1))

        zar_mask_analysis = {
            "top15_utility_in_q60_q95_entropy_pct": top15_in_band,
            "top15_utility_outside_band_pct": top15_outside_band,
            "current_ug_overlap_pct": ug_overlap,
            "q60_H": q60_H, "q95_H": q95_H,
        }

        per_checkpoint[ckpt_name] = {
            "buckets": buckets_list,
            "zar_mask_analysis": zar_mask_analysis,
            "entropy_quantiles": {"q33": q33_H, "q66": q66_H, "q60": q60_H, "q95": q95_H},
            "advantage_quantiles": {"q33": q33_A, "q66": q66_A},
            "n_supervised": int(flat_H.numel()),
        }

        del student, student_logp, s_probs, H_s, A_per_t, JSD_per_t, UG_per_t
        torch.cuda.empty_cache(); gc.collect()

    # Decision logic
    # Compare mid-entropy x high-advantage bucket grad_norm share to other buckets
    def mid_high_share(ckpt_entry):
        for b in ckpt_entry["buckets"]:
            if b["entropy"] == "mid" and b["advantage"] == "high":
                return b["kd_gradient_share"], b["grad_norm_mean"], b["count"]
        return 0.0, 0.0, 0

    mh_share_500, mh_gn_500, mh_count_500 = mid_high_share(per_checkpoint["best.pt"])
    mh_share_1000, mh_gn_1000, mh_count_1000 = mid_high_share(per_checkpoint["step_1000.pt"])

    # Average grad_norm in other buckets
    def uniform_test(ckpt_entry):
        gns = [b["grad_norm_mean"] for b in ckpt_entry["buckets"] if b["count"] > 0]
        if len(gns) < 2:
            return 1.0
        return float(np.std(gns) / max(np.mean(gns), 1e-10))

    cv_500 = uniform_test(per_checkpoint["best.pt"])
    cv_1000 = uniform_test(per_checkpoint["step_1000.pt"])

    top15_in_band_500 = per_checkpoint["best.pt"]["zar_mask_analysis"]["top15_utility_in_q60_q95_entropy_pct"]
    top15_in_band_1000 = per_checkpoint["step_1000.pt"]["zar_mask_analysis"]["top15_utility_in_q60_q95_entropy_pct"]

    if mh_share_500 > 0.2 and cv_500 > 0.3 and top15_in_band_500 >= 0.5:
        decision = "ZPD_VALIDATED"
        reasoning = (
            f"mid-entropy x high-advantage bucket has disproportionate grad share "
            f"({mh_share_500:.2f} at step 500), grad_norm is not uniform (CV={cv_500:.2f}), "
            f"and {top15_in_band_500*100:.0f}% of top-15% utility positions fall in q60-q95 entropy band."
        )
    elif cv_500 < 0.2:
        decision = "ZPD_REJECTED"
        reasoning = (
            f"grad_norm is approximately uniform across buckets (CV={cv_500:.2f}). "
            f"ZPD hypothesis (mid-entropy x high-advantage) not supported by gradient distribution."
        )
    elif top15_in_band_500 < 0.5:
        decision = "ZPD_NEEDS_REFORMULATION"
        reasoning = (
            f"top-15% utility positions only {top15_in_band_500*100:.0f}% in q60-q95 entropy band "
            f"(< 50% threshold). Masking rule needs reformulation — "
            f"utility and entropy are not aligned as hypothesized."
        )
    else:
        decision = "ZPD_NEEDS_REFORMULATION"
        reasoning = (
            f"Partial signal: mid-high share={mh_share_500:.2f}, CV={cv_500:.2f}, "
            f"top-15% in band={top15_in_band_500:.2f}. Not a clean ZPD validation."
        )

    result = {
        "probe": "zpd_utility",
        "checkpoints": ["best.pt", "step_1000.pt"],
        "n_windows": n_windows,
        "teachers": [SMOL_ID, PYTHIA_ID],
        "per_checkpoint": per_checkpoint,
        "decision": decision,
        "decision_reasoning": reasoning,
        "notes": (
            "grad_norm computed as L2 norm of (p_student - p_teacher) — equals magnitude of "
            "KL(p_teacher || softmax(z_student)) gradient on student logits. "
            "Utility U = max(0, A) * JSD per Codex's ZAR-gJSD spec. "
            "JSD(teacher || stopgrad(student)), A = log p_t(y_true) - log p_s(y_true). "
            "UG gate = t_conf * (1-s_match)^2 renormed + clamped at 1.5 (iter5 steady state). "
            "Best-teacher-per-position aggregation uses argmax advantage A."
        ),
    }
    with open(OUT_ZPD, "w") as f:
        json.dump(result, f, indent=2)
    log(f"PROBE 2 COMPLETE: decision={decision}")
    log(f"  Wrote: {OUT_ZPD}")
    return result


# ---- PROBE 3: Pythia Checkpoint Ladder ----

def compute_teacher_byte_probs_simple(teacher, tokenizer, covering_tables,
                                       x_windows, device=DEVICE, batch_size=4):
    """Helper: compute covering byte probs for all windows. Returns (N, T, 256), (N, T) mask."""
    n_wins = x_windows.shape[0]
    all_probs = []
    all_masks = []
    batch_raw = []
    for win_idx in range(n_wins):
        batch_raw.append(x_windows[win_idx].tolist())
        if len(batch_raw) == batch_size or win_idx == n_wins - 1:
            results = _get_teacher_targets_covering_batched(
                teacher, tokenizer, covering_tables, batch_raw, device,
                temperature=1.0, extract_hidden=False, max_depth=8,
            )
            for r in results:
                p = r["byte_probs"].cpu()
                m = r["byte_mask"].cpu()
                if p.shape[0] < SEQ_BYTES:
                    pad_p = torch.zeros(SEQ_BYTES - p.shape[0], 256, dtype=p.dtype)
                    pad_m = torch.zeros(SEQ_BYTES - m.shape[0], dtype=m.dtype)
                    p = torch.cat([p, pad_p], dim=0)
                    m = torch.cat([m, pad_m], dim=0)
                elif p.shape[0] > SEQ_BYTES:
                    p = p[:SEQ_BYTES]
                    m = m[:SEQ_BYTES]
                all_probs.append(p)
                all_masks.append(m)
            batch_raw = []
    return torch.stack(all_probs), torch.stack(all_masks)


def probe_3_pythia_ladder(n_windows=128, seed=SEED,
                           checkpoints=("step10000", "step50000", "step143000")):
    """Probe 3: Pythia checkpoint ladder. Offline oracle measurement only."""
    log("=" * 60)
    log("PROBE 3: PYTHIA CHECKPOINT LADDER")
    log("=" * 60)
    seed_all(seed)

    # Use fewer windows for speed. Aim: 1024 positions -> sample 128 windows, use all positions.
    log(f"Sampling {n_windows} windows...")
    x_windows, y_windows = sample_validation_windows(n_windows, SEQ_BYTES, seed)
    # We'll later filter by supervised mask -> >1024 positions easily

    # Student: best.pt
    student = load_student(CKPT_BEST)
    log("Computing student log-probs...")
    student_logp = student_log_probs(student, x_windows.to(DEVICE), batch_size=4)  # (N, T, 256) CPU
    s_probs = student_logp.exp()
    # Student next-byte log-prob at gold target
    y_idx = y_windows.long().unsqueeze(-1)
    log_p_s_y = student_logp.gather(-1, y_idx).squeeze(-1)  # (N, T)
    del student
    torch.cuda.empty_cache(); gc.collect()

    # First compute SmolLM reference probs (needed for overlap analysis)
    log("Computing SmolLM2 (reference) byte probs...")
    smol_model, smol_tok, smol_vocab = load_teacher(SMOL_ID)
    smol_cov = _build_covering_tables(smol_tok, smol_vocab, device='cpu')
    smol_probs, smol_mask = compute_teacher_byte_probs_simple(
        smol_model, smol_tok, smol_cov, x_windows, device=DEVICE, batch_size=4,
    )
    smol_log_p_y = torch.log(smol_probs.clamp_min(1e-10)).gather(-1, y_idx).squeeze(-1)
    smol_A = smol_log_p_y - log_p_s_y  # advantage of smol over student at gold byte

    del smol_model, smol_tok, smol_cov
    torch.cuda.empty_cache(); gc.collect()

    # For each Pythia checkpoint, load, compute byte probs, compute metrics, unload
    per_ckpt = []
    for ck_name in checkpoints:
        log(f"\n--- Pythia checkpoint: {ck_name} ---")
        try:
            py_model, py_tok, py_vocab = load_teacher(PYTHIA_ID, revision=ck_name)
        except Exception as e:
            log(f"  FAILED to load {ck_name}: {e}")
            per_ckpt.append({
                "step": int(ck_name.replace("step", "")),
                "error": str(e),
            })
            continue

        log("  Building covering...")
        py_cov = _build_covering_tables(py_tok, py_vocab, device='cpu')
        log("  Computing byte probs...")
        py_probs, py_mask = compute_teacher_byte_probs_simple(
            py_model, py_tok, py_cov, x_windows, device=DEVICE, batch_size=4,
        )
        # Both teacher and student aligned at position t (predicting x[t+1]).
        # teacher_log p_t(y_true) at position t = log(py_probs[b, t, y_windows[b, t]]).
        py_log_p_y = torch.log(py_probs.clamp_min(1e-10)).gather(-1, y_idx).squeeze(-1)

        # JSD between teacher_c and student
        m_mix = 0.5 * (py_probs + s_probs).clamp_min(1e-10)
        py_safe = py_probs.clamp_min(1e-10)
        s_safe = s_probs.clamp_min(1e-10)
        jsd = 0.5 * (
            (py_probs * (py_safe / m_mix).log()).sum(-1)
            + (s_probs * (s_safe / m_mix).log()).sum(-1)
        )

        # CE on actual next byte (measured in bits/byte)
        ce_bits = -py_log_p_y / math.log(2)
        mask_f = py_mask.float()
        mean_ce_bits = float((ce_bits * mask_f).sum() / mask_f.sum().clamp_min(1e-10))
        mean_jsd = float((jsd * mask_f).sum() / mask_f.sum().clamp_min(1e-10))

        # Oracle gain: optimal mixing log_q_opt = log((1-w)*p_s + w*p_t) — find best w per position, then mean
        # For simplicity: compute BPB reduction from pure teacher vs pure student on SUPERVISED positions.
        # mean BPB(student) = -log_p_s_y / log2 averaged; mean BPB(teacher) = ce_bits averaged.
        # oracle_gain_bpb = max(0, BPB_student - BPB_teacher) at optimal mixing.
        # Better: find best w in [0,1] in grid.
        ws = torch.linspace(0.0, 1.0, 11)
        mixed_nll_by_w = []
        for w in ws:
            mixed_p_y = (1 - w) * log_p_s_y.exp() + w * py_log_p_y.exp()
            mixed_nll = -torch.log(mixed_p_y.clamp_min(1e-20))
            mixed_nll_masked = (mixed_nll * mask_f).sum() / mask_f.sum().clamp_min(1e-10)
            mixed_nll_by_w.append(mixed_nll_masked.item())
        best_w = float(ws[int(np.argmin(mixed_nll_by_w))].item())
        best_mix_bpb = float(min(mixed_nll_by_w) / math.log(2))
        student_bpb_supervised = float((-log_p_s_y * mask_f).sum() / mask_f.sum().clamp_min(1e-10) / math.log(2))
        oracle_gain_bpb = student_bpb_supervised - best_mix_bpb

        # Overlap with SmolLM: of positions where this Pythia ckpt has A > 0 (beats student at gold),
        # how many ALSO have smol A > 0?
        # More precisely: top-100 Pythia advantage positions, what fraction have SmolLM advantage > 0?
        py_A = py_log_p_y - log_p_s_y  # (N, T)
        any_mask_both = (py_mask & smol_mask).flatten().bool()
        py_A_flat = py_A.flatten()[any_mask_both]
        smol_A_flat = smol_A.flatten()[any_mask_both]
        if py_A_flat.numel() >= 100:
            top_k = 100
            topk_py_idx = torch.topk(py_A_flat, k=top_k).indices
            smol_A_at_topk = smol_A_flat[topk_py_idx]
            smol_overlap = float((smol_A_at_topk > 0).float().mean())
        else:
            smol_overlap = float('nan')

        entry = {
            "step": int(ck_name.replace("step", "")),
            "oracle_gain_bpb": oracle_gain_bpb,
            "best_mix_weight": best_w,
            "ce_on_next_byte_bits_per_byte": mean_ce_bits,
            "jsd_to_student": mean_jsd,
            "smol_overlap_pct": smol_overlap,
            "n_supervised": int(any_mask_both.sum().item()),
        }
        per_ckpt.append(entry)
        log(f"  step={entry['step']}: oracle_gain_bpb={oracle_gain_bpb:.4f}, "
            f"ce_bpb={mean_ce_bits:.3f}, jsd={mean_jsd:.3f}, smol_overlap={smol_overlap:.2f}")

        del py_model, py_tok, py_cov, py_probs, py_mask
        torch.cuda.empty_cache(); gc.collect()

    # Decision
    valid_entries = [e for e in per_ckpt if "error" not in e]
    if len(valid_entries) < 2:
        decision = "USE_FINAL_ONLY"
        reasoning = "Insufficient valid checkpoints to compare."
    else:
        # Is oracle_gain monotone?
        gains = [(e["step"], e["oracle_gain_bpb"]) for e in valid_entries]
        gains.sort(key=lambda x: x[0])
        monotone_up = all(gains[i][1] >= gains[i-1][1] - 0.001 for i in range(1, len(gains)))
        overlaps = [e["smol_overlap_pct"] for e in valid_entries if not math.isnan(e["smol_overlap_pct"])]
        max_gain_step = max(gains, key=lambda x: x[1])
        final_gain = gains[-1][1]
        # Curriculum if earlier ckpt has comparable gain (within 0.005) AND lower overlap
        # (more diverse from SmolLM at earlier steps)
        early_ckpt = gains[0]
        final_ckpt = gains[-1]
        comparable = abs(early_ckpt[1] - final_ckpt[1]) < 0.005
        early_entry = next(e for e in valid_entries if e["step"] == early_ckpt[0])
        final_entry = next(e for e in valid_entries if e["step"] == final_ckpt[0])
        early_overlap = early_entry.get("smol_overlap_pct", float('nan'))
        final_overlap = final_entry.get("smol_overlap_pct", float('nan'))
        lower_overlap_early = (not math.isnan(early_overlap) and not math.isnan(final_overlap)
                                and early_overlap < final_overlap - 0.05)
        if max_gain_step[0] != gains[-1][0]:
            decision = f"USE_INTERMEDIATE_{max_gain_step[0]}"
            reasoning = (
                f"oracle_gain peaks at intermediate step {max_gain_step[0]} "
                f"({max_gain_step[1]:.4f} BPB), exceeding final ({final_gain:.4f})."
            )
        elif comparable and lower_overlap_early:
            decision = "USE_CURRICULUM"
            reasoning = (
                f"Early ckpt ({early_ckpt[0]}) has comparable oracle_gain "
                f"({early_ckpt[1]:.4f} vs final {final_ckpt[1]:.4f}) and lower SmolLM overlap "
                f"({early_overlap:.2f} vs {final_overlap:.2f}). Use as curriculum."
            )
        elif monotone_up:
            decision = "USE_FINAL_ONLY"
            reasoning = (
                f"oracle_gain monotone non-decreasing with training step "
                f"({[round(g[1],4) for g in gains]}). Final checkpoint is correct choice."
            )
        else:
            decision = "USE_FINAL_ONLY"
            reasoning = (
                f"No clear curriculum benefit: oracle_gains={[round(g[1],4) for g in gains]}, "
                f"overlaps={overlaps}."
            )

    result = {
        "probe": "pythia_checkpoint_ladder",
        "student": "best.pt",
        "reference_teacher": SMOL_ID,
        "checkpoints_tested": list(checkpoints),
        "n_windows": n_windows,
        "per_checkpoint": per_ckpt,
        "decision": decision,
        "decision_reasoning": reasoning,
        "notes": (
            "oracle_gain_bpb = student_bpb_on_supervised - min over w in linspace(0,1,11) of "
            "mean(-log((1-w)*p_s(y_true) + w*p_t(y_true))) / log(2). "
            "smol_overlap_pct = fraction of top-100 Pythia-advantage positions that also have "
            "SmolLM advantage > 0 over student. "
            "All metrics use supervised-only positions (teacher covering mask = True)."
        ),
    }
    with open(OUT_LADDER, "w") as f:
        json.dump(result, f, indent=2)
    log(f"PROBE 3 COMPLETE: decision={decision}")
    log(f"  Wrote: {OUT_LADDER}")
    return result


# ---- Main ----

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("KM R1 PROBES — Kuhnian Survival, ZPD Utility, Pythia Ladder")
    log("=" * 70)
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        log(f"  VRAM: free={free/1e9:.1f}G / total={total/1e9:.1f}G")

    t_start = time.time()
    errors = []

    try:
        r1 = probe_1_kuhnian(n_positions=2048, h_values=(1, 2, 4, 8), seed=SEED, n_windows=256)
    except Exception as e:
        import traceback
        log(f"PROBE 1 FAILED: {e}\n{traceback.format_exc()}")
        errors.append(("probe_1_kuhnian", str(e)))
    log(f"\nElapsed so far: {(time.time() - t_start)/60:.1f} min")

    try:
        r2 = probe_2_zpd(n_windows=256, seed=SEED)
    except Exception as e:
        import traceback
        log(f"PROBE 2 FAILED: {e}\n{traceback.format_exc()}")
        errors.append(("probe_2_zpd", str(e)))
    log(f"\nElapsed so far: {(time.time() - t_start)/60:.1f} min")

    # Budget check: 45 min total; if we're > 35, skip probe 3
    elapsed_min = (time.time() - t_start) / 60
    if elapsed_min > 35:
        log(f"SKIPPING PROBE 3 — elapsed {elapsed_min:.1f} min > 35 min budget")
        stub = {
            "probe": "pythia_checkpoint_ladder",
            "decision": "SKIPPED",
            "decision_reasoning": f"skipped due to time budget ({elapsed_min:.1f} min elapsed)",
            "checkpoints_tested": [],
            "per_checkpoint": [],
        }
        with open(OUT_LADDER, "w") as f:
            json.dump(stub, f, indent=2)
    else:
        try:
            r3 = probe_3_pythia_ladder(n_windows=128, seed=SEED,
                                        checkpoints=("step10000", "step50000", "step143000"))
        except Exception as e:
            import traceback
            log(f"PROBE 3 FAILED: {e}\n{traceback.format_exc()}")
            errors.append(("probe_3_pythia_ladder", str(e)))

    log(f"\n{'='*60}")
    log(f"ALL PROBES DONE in {(time.time() - t_start)/60:.1f} min")
    if errors:
        log("ERRORS:")
        for name, err in errors:
            log(f"  {name}: {err}")
    log(f"Files written:")
    for f in [OUT_KUHN, OUT_ZPD, OUT_LADDER]:
        if f.exists():
            log(f"  {f} ({f.stat().st_size} bytes)")

    if _log_f:
        _log_f.close()


if __name__ == "__main__":
    main()
