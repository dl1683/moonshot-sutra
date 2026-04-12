"""Multisignal init audit (64 windows) — zero-cost gradient conflict detection.

Loads student + 3 teachers, computes per-channel losses and gradient
norms/cosines on shared params (layers 12-23 + final norm + LM head).
No optimizer step — pure diagnostic.

Per Codex R7 design: multisignal_init_audit_64w
"""

import sys, os, json, torch, torch.nn.functional as F, math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = "cuda"
CKPT_PATH = REPO / "results" / "checkpoints_kd_197m_60k_gate" / "control_197m_60k_step60000.pt"
OUTPUT_PATH = REPO / "results" / "multisignal_audit_64w.json"
EVAL_CACHE = REPO / "results" / "eval_cache_16k.pt"

# Teachers
QWEN_NAME = "Qwen/Qwen3-0.6B-Base"
LFM_NAME = "LiquidAI/LFM2.5-1.2B-Base"
GEMMA_NAME = "google/embeddinggemma-300m"

N_WINDOWS = 64
BATCH_SIZE = 8
N_SPANS = 16
SEQ_LEN = 512
TEMPERATURE = 2.0
TOP_K = 64


def main():
    print("=== Multisignal Init Audit (64 windows) ===", flush=True)

    # ---- Import from dense_baseline ----
    from dense_baseline import (
        DenseTransformer, VOCAB_SIZE, TeacherAdapter,
        byte_span_pool, compute_cka, compute_state_kd_loss,
        compute_semantic_kd_loss, compute_cross_tok_logit_kd,
        compute_vocab_overlap,
    )
    from tokenizers import Tokenizer

    # ---- Load student ----
    print("Loading student...", flush=True)
    ckpt = torch.load(str(CKPT_PATH), weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = DenseTransformer(
        vocab_size=VOCAB_SIZE,
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 24),
        n_heads=cfg.get("n_heads", 12),
        ff_dim=cfg.get("ff_dim", 2048),
        exit_layers=cfg.get("exit_layers", [7, 15, 23]),
        norm_type=cfg.get("norm_type", "rmsnorm"),
        block_schedule=cfg.get("block_schedule", None),
        conv_kernel_size=cfg.get("conv_kernel_size", 64),
        d_attn=cfg.get("d_attn", None),
        d_conv=cfg.get("d_conv", None),
        n_q_heads=cfg.get("n_q_heads", None),
        n_kv_heads=cfg.get("n_kv_heads", None),
        head_dim=cfg.get("head_dim", 64),
    )
    init_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[init_key])
    model = model.to(DEVICE)
    model.train()  # Need gradients
    dim = cfg.get("dim", 768)
    n_layers = cfg.get("n_layers", 24)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Student: {n_params:,} params, dim={dim}, layers={n_layers}", flush=True)
    del ckpt

    # ---- Load tokenizer ----
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    student_tokenizer = Tokenizer.from_file(str(tok_path))

    # ---- Load eval windows ----
    cache = torch.load(str(EVAL_CACHE), weights_only=False)
    windows = cache["windows"][:N_WINDOWS]
    print(f"  {len(windows)} eval windows loaded", flush=True)

    # ---- Load teachers ----
    print("Loading teachers...", flush=True)
    t_qwen = TeacherAdapter(QWEN_NAME, device=DEVICE, dtype=torch.float16)
    t_lfm = TeacherAdapter(LFM_NAME, device=DEVICE, dtype=torch.float16)
    t_gemma = TeacherAdapter(GEMMA_NAME, device=DEVICE, dtype=torch.float32)  # float16 overflows

    # ---- Pre-compute vocab overlap for Qwen logit KD ----
    s_ids, t_ids, n_shared = compute_vocab_overlap(student_tokenizer, t_qwen.tokenizer, device=DEVICE)
    print(f"  Qwen vocab overlap: {n_shared}/{VOCAB_SIZE} ({100*n_shared/VOCAB_SIZE:.1f}%)", flush=True)

    # ---- Create projectors ----
    proj_lfm_state = torch.nn.Linear(dim, t_lfm.hidden_dim).to(DEVICE)  # 768 -> 2048
    proj_gemma_sem = torch.nn.Linear(dim, t_gemma.hidden_dim).to(DEVICE)  # 768 -> 768 (or 256)
    torch.nn.init.normal_(proj_lfm_state.weight, std=0.02)
    # Gemma projector: if dims match, init to identity; else small random
    if dim == t_gemma.hidden_dim:
        torch.nn.init.eye_(proj_gemma_sem.weight)
        torch.nn.init.zeros_(proj_gemma_sem.bias)
    else:
        torch.nn.init.normal_(proj_gemma_sem.weight, std=0.02)
        torch.nn.init.zeros_(proj_gemma_sem.bias)

    # ---- Identify shared params for gradient audit ----
    # Layers 12-23 + final norm + LM head (emb.weight is tied)
    shared_param_names = []
    shared_params = []
    for name, p in model.named_parameters():
        # layers.12 through layers.23
        is_deep_layer = False
        for li in range(12, 24):
            if f"layers.{li}." in name:
                is_deep_layer = True
                break
        is_final = name == "norm.weight" or name == "emb.weight"
        if is_deep_layer or is_final:
            shared_param_names.append(name)
            shared_params.append(p)

    n_shared_params = sum(p.numel() for p in shared_params)
    print(f"  Shared params for gradient audit: {len(shared_params)} tensors, {n_shared_params:,} params", flush=True)

    # ---- Run audit batches ----
    all_losses = {"ce": [], "q_logit": [], "lfm_state": [], "gemma_sem": []}
    all_grad_norms = {"ce": [], "q_logit": [], "lfm_state": [], "gemma_sem": []}
    all_grad_cosines = {
        "ce_vs_qlogit": [], "ce_vs_lstate": [], "ce_vs_gsem": [],
        "qlogit_vs_lstate": [], "qlogit_vs_gsem": [], "lstate_vs_gsem": [],
    }

    n_batches = N_WINDOWS // BATCH_SIZE
    print(f"\nRunning {n_batches} batches (BS={BATCH_SIZE})...", flush=True)

    for bi in range(n_batches):
        batch_windows = windows[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        # Build token batch
        x_batch = torch.stack([w[:SEQ_LEN].clone().detach() for w in batch_windows]).to(DEVICE)
        y_batch = torch.stack([w[1:SEQ_LEN+1].clone().detach() for w in batch_windows]).to(DEVICE)
        B = x_batch.shape[0]

        # Decode to text for teacher input
        texts = [student_tokenizer.decode(w[:SEQ_LEN].tolist()) for w in batch_windows]

        # Compute student byte offsets (same method as train_kd)
        stu_offsets = torch.zeros(B, SEQ_LEN, 2, device=DEVICE, dtype=torch.long)
        for b_idx, text in enumerate(texts):
            enc_s = student_tokenizer.encode(text)
            n_tok = min(len(enc_s.offsets), SEQ_LEN)
            for ti in range(n_tok):
                stu_offsets[b_idx, ti, 0] = enc_s.offsets[ti][0]
                stu_offsets[b_idx, ti, 1] = enc_s.offsets[ti][1]

        # ---- Student forward ----
        model.zero_grad()
        proj_lfm_state.zero_grad()
        proj_gemma_sem.zero_grad()

        out = model(x_batch, return_exits=False, return_hidden=True)
        s_logits = out["logits"]
        s_hidden = out["hidden"]  # pre-norm, (B, T, D)

        s_mask = torch.ones(B, SEQ_LEN, device=DEVICE)

        # ---- 1. CE loss ----
        ce_loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), y_batch.view(-1))
        ce_loss.backward(retain_graph=True)
        g_ce = torch.cat([p.grad.flatten() for p in shared_params if p.grad is not None])
        ce_grad_norm = g_ce.norm().item()
        all_losses["ce"].append(ce_loss.item())
        all_grad_norms["ce"].append(ce_grad_norm)

        # Save CE grads for cosine computation
        g_ce_saved = g_ce.clone()

        # ---- 2. Qwen logit KD loss ----
        model.zero_grad()
        proj_lfm_state.zero_grad()
        proj_gemma_sem.zero_grad()

        # Re-forward student (need clean graph)
        out = model(x_batch, return_exits=False, return_hidden=True)
        s_logits = out["logits"]
        s_hidden = out["hidden"]

        # Teacher forward
        t_qwen_out = t_qwen.forward(texts, max_length=SEQ_LEN)

        q_loss = compute_cross_tok_logit_kd(
            student_logits=s_logits,
            teacher_logits=t_qwen_out["logits"],
            student_offsets=stu_offsets,
            teacher_offsets=t_qwen_out.get("byte_offsets", None),
            teacher_mask=t_qwen_out["attention_mask"],
            shared_s_ids=s_ids,
            shared_t_ids=t_ids,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            taid_beta=1.0,  # Full teacher signal for audit (no ramp)
        )

        if q_loss.requires_grad:
            q_loss.backward(retain_graph=True)
            g_q = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=DEVICE) for p in shared_params])
            q_grad_norm = g_q.norm().item()
        else:
            g_q = torch.zeros_like(g_ce_saved)
            q_grad_norm = 0.0

        all_losses["q_logit"].append(q_loss.item())
        all_grad_norms["q_logit"].append(q_grad_norm)
        g_q_saved = g_q.clone()

        # ---- 3. LFM state CKA loss ----
        model.zero_grad()
        proj_lfm_state.zero_grad()
        proj_gemma_sem.zero_grad()

        out = model(x_batch, return_exits=False, return_hidden=True)
        s_hidden = out["hidden"]

        t_lfm_out = t_lfm.forward(texts, max_length=SEQ_LEN)
        t_lfm_hidden = t_lfm_out["hidden"]

        if t_lfm_hidden is not None:
            lfm_loss = compute_state_kd_loss(
                student_hidden=s_hidden,
                teacher_hidden=t_lfm_hidden.float(),
                student_offsets=None,
                teacher_offsets=t_lfm_out.get("byte_offsets", None),
                student_mask=s_mask,
                teacher_mask=t_lfm_out["attention_mask"],
                projector=proj_lfm_state,
                n_spans=N_SPANS,
            )
        else:
            lfm_loss = torch.tensor(0.0, device=DEVICE)

        if lfm_loss.requires_grad and lfm_loss.item() > 0:
            lfm_loss.backward(retain_graph=True)
            g_lfm = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=DEVICE) for p in shared_params])
            lfm_grad_norm = g_lfm.norm().item()
        else:
            g_lfm = torch.zeros_like(g_ce_saved)
            lfm_grad_norm = 0.0

        all_losses["lfm_state"].append(lfm_loss.item())
        all_grad_norms["lfm_state"].append(lfm_grad_norm)
        g_lfm_saved = g_lfm.clone()

        # ---- 4. Gemma semantic loss ----
        model.zero_grad()
        proj_lfm_state.zero_grad()
        proj_gemma_sem.zero_grad()

        out = model(x_batch, return_exits=False, return_hidden=True)
        s_hidden = out["hidden"]

        t_gemma_out = t_gemma.forward(texts, max_length=SEQ_LEN)
        t_gemma_hidden = t_gemma_out["hidden"]

        gemma_loss = compute_semantic_kd_loss(
            student_hidden=s_hidden,
            teacher_hidden=t_gemma_hidden.float(),
            student_mask=s_mask,
            teacher_mask=t_gemma_out["attention_mask"],
            projector=proj_gemma_sem,
        )

        if gemma_loss.requires_grad and not torch.isnan(gemma_loss) and not torch.isinf(gemma_loss):
            gemma_loss.backward(retain_graph=False)
            g_gem = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros(p.numel(), device=DEVICE) for p in shared_params])
            # Guard NaN in gradients
            if torch.isnan(g_gem).any():
                g_gem = torch.zeros_like(g_ce_saved)
                gem_grad_norm = 0.0
                gemma_loss = torch.tensor(float('nan'))
            else:
                gem_grad_norm = g_gem.norm().item()
        else:
            g_gem = torch.zeros_like(g_ce_saved)
            gem_grad_norm = 0.0

        all_losses["gemma_sem"].append(gemma_loss.item())
        all_grad_norms["gemma_sem"].append(gem_grad_norm)

        # ---- Pairwise cosines ----
        def cosine(a, b):
            na, nb = a.norm(), b.norm()
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return (a @ b / (na * nb)).item()

        all_grad_cosines["ce_vs_qlogit"].append(cosine(g_ce_saved, g_q_saved))
        all_grad_cosines["ce_vs_lstate"].append(cosine(g_ce_saved, g_lfm_saved))
        all_grad_cosines["ce_vs_gsem"].append(cosine(g_ce_saved, g_gem))
        all_grad_cosines["qlogit_vs_lstate"].append(cosine(g_q_saved, g_lfm_saved))
        all_grad_cosines["qlogit_vs_gsem"].append(cosine(g_q_saved, g_gem))
        all_grad_cosines["lstate_vs_gsem"].append(cosine(g_lfm_saved, g_gem))

        # Clean up
        del g_ce_saved, g_q_saved, g_lfm_saved, g_q, g_lfm, g_gem, g_ce
        del t_qwen_out, t_lfm_out, t_gemma_out
        torch.cuda.empty_cache()

        print(f"  Batch {bi+1}/{n_batches}: CE={ce_loss.item():.4f}, "
              f"Q_logit={q_loss.item():.4f}, LFM_state={lfm_loss.item():.4f}, "
              f"Gemma_sem={gemma_loss.item():.4f}", flush=True)

    # ---- Aggregate results ----
    import statistics

    def stats(vals):
        return {
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }

    # Gradient norm ratios (relative to CE)
    ce_norms = all_grad_norms["ce"]
    norm_ratios = {}
    for channel in ["q_logit", "lfm_state", "gemma_sem"]:
        ratios = []
        for i in range(len(ce_norms)):
            if ce_norms[i] > 1e-12:
                ratios.append(all_grad_norms[channel][i] / ce_norms[i])
            else:
                ratios.append(0.0)
        norm_ratios[channel] = stats(ratios)

    results = {
        "probe": "multisignal_init_audit_64w",
        "n_windows": N_WINDOWS,
        "batch_size": BATCH_SIZE,
        "n_batches": n_batches,
        "shared_params_audited": f"layers 12-23 + norm + emb ({n_shared_params:,} params)",
        "losses": {k: stats(v) for k, v in all_losses.items()},
        "grad_norms": {k: stats(v) for k, v in all_grad_norms.items()},
        "grad_norm_ratios_vs_ce": norm_ratios,
        "grad_cosines": {k: stats(v) for k, v in all_grad_cosines.items()},
        "conflict_analysis": {
            "threshold_median_cosine": -0.05,
            "threshold_ratio": 0.5,
        },
    }

    # ---- Conflict detection per Codex R7 ----
    for pair, vals in all_grad_cosines.items():
        med = statistics.median(vals)
        n_neg = sum(1 for v in vals if v < -0.10)
        results["conflict_analysis"][pair] = {
            "median_cosine": med,
            "n_below_neg010": n_neg,
            "pct_below_neg010": 100 * n_neg / len(vals),
            "CONFLICT": med < -0.05 or (n_neg / len(vals)) > 0.20,
        }

    for channel in ["q_logit", "lfm_state", "gemma_sem"]:
        ratio_mean = norm_ratios[channel]["mean"]
        results["conflict_analysis"][f"{channel}_ratio_mean"] = ratio_mean
        results["conflict_analysis"][f"{channel}_DOMINATES"] = ratio_mean > 0.5

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}", flush=True)

    # ---- Print summary ----
    print("\n=== AUDIT SUMMARY ===", flush=True)
    print(f"Loss magnitudes:", flush=True)
    for ch, vals in all_losses.items():
        print(f"  {ch}: mean={statistics.mean(vals):.6f}, med={statistics.median(vals):.6f}", flush=True)

    print(f"\nGradient norm ratios (vs CE):", flush=True)
    for ch, s in norm_ratios.items():
        print(f"  {ch}: mean={s['mean']:.4f}, med={s['median']:.4f}", flush=True)

    print(f"\nGradient cosines:", flush=True)
    for pair, vals in all_grad_cosines.items():
        med = statistics.median(vals)
        flag = " *** CONFLICT ***" if med < -0.05 else ""
        print(f"  {pair}: median={med:.4f}{flag}", flush=True)

    any_conflict = any(
        v.get("CONFLICT", False)
        for k, v in results["conflict_analysis"].items()
        if isinstance(v, dict)
    )
    print(f"\nOverall conflict detected: {any_conflict}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
