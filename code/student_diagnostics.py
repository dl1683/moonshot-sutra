"""Diagnostic probes for Sutra-24A-197M student internals.

Runs CPU-only analysis on the 60K checkpoint to understand:
1. Per-layer representation structure (SVD spectrum, effective rank)
2. Inter-layer CKA (how similar are representations across layers?)
3. Student vs teacher CKA at matched layers
4. Token-level entropy distribution (what does the student know/not know?)
5. Weight matrix analysis (where is information concentrated?)
6. Gradient sensitivity per layer

All results written to results/student_diagnostics.json
"""

import sys, os, json, math, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
DEVICE = "cpu"

STUDENT_CKPT = REPO / "results" / "checkpoints_kd_197m_60k_gate" / "control_197m_60k_step60000.pt"
OUTPUT = REPO / "results" / "student_diagnostics.json"

# Number of eval batches (CPU, so keep small but sufficient)
N_BATCHES = 8
SEQ_LEN = 512
BATCH_SIZE = 2


def load_student():
    """Load 60K student checkpoint on CPU."""
    from dense_baseline import DenseTransformer, VOCAB_SIZE

    ckpt = torch.load(str(STUDENT_CKPT), weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=cfg.get("dim", 768),
        n_layers=cfg.get("n_layers", 24),
        n_heads=cfg.get("n_heads", 12),
        ff_dim=cfg.get("ff_dim", 2304),
        exit_layers=cfg.get("exit_layers", [7, 15, 23]),
        norm_type=cfg.get("norm_type", "ss_rmsnorm"),
        block_schedule=cfg.get("block_schedule", None),
        n_q_heads=cfg.get("n_q_heads", None),
        n_kv_heads=cfg.get("n_kv_heads", None),
        head_dim=cfg.get("head_dim", 64),
    ).to(DEVICE).float()  # float32 on CPU

    init_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[init_key], strict=False)
    model.eval()
    step = ckpt.get("step", "?")
    del ckpt
    print(f"Student loaded (step {step}, {model.count_params():,} params)")
    return model, cfg


def get_eval_data():
    """Load a few batches of eval data. Returns list of (input, target) pairs."""
    from data_loader import ShardedDataset
    shard_dir = REPO / "data" / "shards_16k"
    ds = ShardedDataset(shard_dir=str(shard_dir))

    batches = []
    for _ in range(N_BATCHES):
        x, y = ds.sample_batch(BATCH_SIZE, SEQ_LEN, device='cpu', split='test')
        batches.append((x, y))  # x=(B, T), y=(B, T)
    return batches


def extract_all_hidden_states(model, x):
    """Run forward pass, extracting hidden states at every layer.

    Returns: list of (B, T, D) tensors, one per layer (0=embedding, 1-24=after each layer).
    """
    with torch.no_grad():
        B, T = x.shape
        h = model.emb(x) * math.sqrt(model.dim)
        states = [h.clone()]  # layer 0 = embedding

        cos, sin = model.rope_cos[:T], model.rope_sin[:T]

        for i, layer in enumerate(model.layers):
            h = layer(h, cos, sin)
            states.append(h.clone())

    return states  # 25 tensors: embedding + 24 layers


# ---- Probe 1: SVD Spectrum (effective rank per layer) ----
def svd_spectrum(states):
    """Compute singular value spectrum and effective rank at each layer.

    Effective rank (Roy & Vetterli 2007): exp(H(sigma/sum(sigma)))
    where H is Shannon entropy. Measures how many dimensions are "used."
    """
    results = {}
    for layer_idx, h in enumerate(states):
        # Flatten to (B*T, D)
        flat = h.reshape(-1, h.shape[-1]).float()
        # Subsample if too large for CPU SVD
        if flat.shape[0] > 2000:
            idx = torch.randperm(flat.shape[0])[:2000]
            flat = flat[idx]

        # Center
        flat = flat - flat.mean(dim=0, keepdim=True)

        # SVD
        try:
            U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
        except Exception as e:
            print(f"  SVD failed at layer {layer_idx}: {e}")
            results[layer_idx] = {"error": str(e)}
            continue

        S = S.float()
        # Normalized singular values
        S_norm = S / S.sum()
        S_norm = S_norm[S_norm > 1e-10]

        # Effective rank
        entropy = -(S_norm * torch.log(S_norm)).sum().item()
        eff_rank = math.exp(entropy)

        # Top-k concentration
        total_var = (S ** 2).sum().item()
        top10_var = (S[:10] ** 2).sum().item() / total_var if total_var > 0 else 0
        top50_var = (S[:50] ** 2).sum().item() / total_var if total_var > 0 else 0
        top100_var = (S[:100] ** 2).sum().item() / total_var if total_var > 0 else 0

        # Condition number
        cond = (S[0] / S[-1]).item() if S[-1] > 1e-10 else float("inf")

        results[layer_idx] = {
            "effective_rank": round(eff_rank, 2),
            "max_possible_rank": min(flat.shape),
            "top10_variance_fraction": round(top10_var, 4),
            "top50_variance_fraction": round(top50_var, 4),
            "top100_variance_fraction": round(top100_var, 4),
            "condition_number": round(cond, 2) if cond != float("inf") else "inf",
            "top5_singular_values": [round(s, 4) for s in S[:5].tolist()],
        }
        print(f"  Layer {layer_idx:2d}: eff_rank={eff_rank:.1f}/{min(flat.shape)}, "
              f"top10={top10_var:.3f}, top50={top50_var:.3f}, cond={cond:.1f}")

    return results


# ---- Probe 2: Inter-layer CKA ----
def linear_CKA(X, Y):
    """Linear CKA (Kornblith et al., 2019).
    X, Y: (n, d) centered matrices.
    """
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = (XtY ** 2).sum().item()
    hsic_xx = (XtX ** 2).sum().item()
    hsic_yy = (YtY ** 2).sum().item()

    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / math.sqrt(hsic_xx * hsic_yy)


def inter_layer_cka(states, sample_size=1500):
    """Compute CKA between all pairs of student layers."""
    n_layers = len(states)
    flats = []
    for h in states:
        flat = h.reshape(-1, h.shape[-1]).float()
        if flat.shape[0] > sample_size:
            idx = torch.randperm(flat.shape[0])[:sample_size]
            flat = flat[idx]
        flat = flat - flat.mean(dim=0, keepdim=True)
        flats.append(flat)

    cka_matrix = [[0.0] * n_layers for _ in range(n_layers)]
    for i in range(n_layers):
        for j in range(i, n_layers):
            cka = linear_CKA(flats[i], flats[j])
            cka_matrix[i][j] = round(cka, 4)
            cka_matrix[j][i] = round(cka, 4)
            if abs(i - j) <= 1 or i == 0 or j == 0 or i == n_layers - 1 or j == n_layers - 1:
                print(f"  CKA({i},{j}) = {cka:.4f}")

    return cka_matrix


# ---- Probe 3: Token entropy distribution ----
def entropy_analysis(model, batches):
    """Analyze per-token entropy of student's output distribution."""
    all_entropies = []
    log_vocab = math.log(16000)

    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            if isinstance(logits, dict):
                logits = logits['logits']

            log_probs = F.log_softmax(logits.float(), dim=-1)
            probs = log_probs.exp()
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
            norm_entropy = token_entropy / log_vocab
            all_entropies.append(norm_entropy.reshape(-1))

    all_ent = torch.cat(all_entropies)

    results = {
        "mean": round(all_ent.mean().item(), 4),
        "std": round(all_ent.std().item(), 4),
        "median": round(all_ent.median().item(), 4),
        "p10": round(all_ent.quantile(0.1).item(), 4),
        "p25": round(all_ent.quantile(0.25).item(), 4),
        "p75": round(all_ent.quantile(0.75).item(), 4),
        "p90": round(all_ent.quantile(0.9).item(), 4),
        "p99": round(all_ent.quantile(0.99).item(), 4),
        "fraction_below_0.3": round((all_ent < 0.3).float().mean().item(), 4),
        "fraction_below_0.5": round((all_ent < 0.5).float().mean().item(), 4),
        "fraction_above_0.8": round((all_ent > 0.8).float().mean().item(), 4),
        "n_tokens": all_ent.shape[0],
    }

    print(f"  Entropy: mean={results['mean']:.4f}, median={results['median']:.4f}, "
          f"std={results['std']:.4f}")
    print(f"  Low-ent (<0.3): {results['fraction_below_0.3']:.1%}, "
          f"High-ent (>0.8): {results['fraction_above_0.8']:.1%}")

    return results


# ---- Probe 4: Exit-layer analysis ----
def exit_analysis(model, batches):
    """Compare representations at exit layers (7, 15, 23)."""
    exit_states = {7: [], 15: [], 23: []}

    with torch.no_grad():
        for x, y in batches[:4]:
            out = model(x, return_exits=True, return_hidden=True)
            for eidx in [7, 15, 23]:
                if eidx in out.get('exit_hidden', {}):
                    exit_states[eidx].append(out['exit_hidden'][eidx])

    results = {}
    for eidx in [7, 15, 23]:
        if exit_states[eidx]:
            cat = torch.cat(exit_states[eidx], dim=0)
            flat = cat.reshape(-1, cat.shape[-1]).float()
            if flat.shape[0] > 1500:
                idx = torch.randperm(flat.shape[0])[:1500]
                flat = flat[idx]

            norms = flat.norm(dim=-1)
            results[f"exit_{eidx}_mean_norm"] = round(norms.mean().item(), 4)
            results[f"exit_{eidx}_std_norm"] = round(norms.std().item(), 4)

    # CKA between exits
    if all(len(exit_states[e]) > 0 for e in [7, 15, 23]):
        for a, b in [(7, 15), (15, 23), (7, 23)]:
            fa = torch.cat(exit_states[a], dim=0).reshape(-1, 768).float()
            fb = torch.cat(exit_states[b], dim=0).reshape(-1, 768).float()
            n = min(1500, fa.shape[0], fb.shape[0])
            idx = torch.randperm(min(fa.shape[0], fb.shape[0]))[:n]
            fa, fb = fa[idx], fb[idx]
            fa = fa - fa.mean(0, keepdim=True)
            fb = fb - fb.mean(0, keepdim=True)
            cka = linear_CKA(fa, fb)
            results[f"cka_exit_{a}_exit_{b}"] = round(cka, 4)
            print(f"  CKA(exit_{a}, exit_{b}) = {cka:.4f}")

    return results


# ---- Probe 5: Weight matrix analysis ----
def weight_analysis(model):
    """Analyze weight matrices for information distribution."""
    results = {"per_layer": {}, "summary": {}}

    total_attn_norm = 0.0
    total_ffn_norm = 0.0

    emb_w = model.emb.weight.data.float()
    results["embedding_norm"] = round(emb_w.norm().item(), 4)

    for i, layer in enumerate(model.layers):
        layer_info = {}
        attn_norm = 0.0
        ffn_norm = 0.0

        for name, param in layer.named_parameters():
            pnorm = param.data.float().norm().item()
            if any(k in name for k in ["attn", "wq", "wk", "wv", "wo"]):
                attn_norm += pnorm ** 2
            elif any(k in name for k in ["ff", "w1", "w2", "w3"]):
                ffn_norm += pnorm ** 2

        attn_norm = math.sqrt(attn_norm)
        ffn_norm = math.sqrt(ffn_norm)
        total_attn_norm += attn_norm ** 2
        total_ffn_norm += ffn_norm ** 2

        layer_info["attn_frobenius_norm"] = round(attn_norm, 4)
        layer_info["ffn_frobenius_norm"] = round(ffn_norm, 4)
        layer_info["attn_ffn_ratio"] = round(attn_norm / ffn_norm if ffn_norm > 0 else 0, 4)

        results["per_layer"][i] = layer_info

        if i % 6 == 0 or i == 23:
            print(f"  Layer {i:2d}: attn_norm={attn_norm:.2f}, ffn_norm={ffn_norm:.2f}, "
                  f"ratio={attn_norm/ffn_norm:.3f}")

    results["summary"]["total_attn_norm"] = round(math.sqrt(total_attn_norm), 4)
    results["summary"]["total_ffn_norm"] = round(math.sqrt(total_ffn_norm), 4)
    results["summary"]["attn_fraction"] = round(
        total_attn_norm / (total_attn_norm + total_ffn_norm), 4)

    return results


# ---- Probe 6: Gradient sensitivity per layer ----
def gradient_sensitivity(model, batches):
    """Which layers have the steepest loss landscape?

    Compute |dL/dh_i| at each layer — layers with high gradient norm
    are where the model is most "sensitive" and where grafted knowledge
    would have the most impact.
    """
    results = {}
    model.eval()
    batch, targets = batches[0]
    B, T = batch.shape

    # We need to hook into intermediate layers
    layer_outputs = {}

    def make_hook(idx):
        def hook(module, inp, out):
            out.retain_grad()
            layer_outputs[idx] = out
        return hook

    hooks = []
    for i in [0, 3, 7, 11, 15, 19, 23]:
        h = model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    # Forward
    logits = model(batch)
    if isinstance(logits, dict):
        logits = logits['logits']

    logits_shifted = logits[:, :-1, :]
    loss = F.cross_entropy(logits_shifted.reshape(-1, logits_shifted.shape[-1]),
                          targets[:, :logits_shifted.shape[1]].reshape(-1))
    loss.backward()

    for i, tensor in layer_outputs.items():
        if tensor.grad is not None:
            grad_norm = tensor.grad.norm().item()
            grad_per_dim = tensor.grad.norm(dim=-1).mean().item()
            results[f"layer_{i}_grad_norm"] = round(grad_norm, 4)
            results[f"layer_{i}_grad_per_token"] = round(grad_per_dim, 6)
            print(f"  Layer {i:2d}: |dL/dh| = {grad_norm:.4f}, per-token = {grad_per_dim:.6f}")

    for h in hooks:
        h.remove()

    return results


# ---- Probe 7: Student vs Teacher CKA ----
def student_teacher_cka(model, batches):
    """Compare student representations to Qwen3-1.7B teacher at matched layers.

    Uses global-average-pooled hidden states on shared text.
    Student [7, 15, 23] → Teacher [8, 16, 24].
    """
    print(f"\n  Loading teacher: Qwen/Qwen3-1.7B (CPU, slow)...")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        teacher_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-1.7B", torch_dtype=torch.float32,
            output_hidden_states=True
        ).to("cpu")
        teacher_model.eval()
    except Exception as e:
        print(f"  Failed to load teacher: {e}")
        return {"error": str(e)}

    from tokenizers import Tokenizer
    tok_path = REPO / "data" / "tokenizer_16k" / "tokenizer.json"
    student_tok = Tokenizer.from_file(str(tok_path))

    n_samples = 6
    student_hiddens = {7: [], 15: [], 23: []}
    teacher_hiddens = {8: [], 16: [], 24: []}

    from data_loader import ShardedDataset
    ds = ShardedDataset(shard_dir=str(REPO / "data" / "shards_16k"))

    for sample_idx in range(n_samples):
        x, _ = ds.sample_batch(1, SEQ_LEN, device='cpu', split='test')
        student_tokens = x  # (1, SEQ_LEN)
        decoded = student_tok.decode(student_tokens[0].tolist())

        teacher_tokens = teacher_tok(decoded, return_tensors="pt",
                                     max_length=256, truncation=True)

        with torch.no_grad():
            # Student
            s_states = extract_all_hidden_states(model, student_tokens)
            for eidx in [7, 15, 23]:
                student_hiddens[eidx].append(
                    s_states[eidx + 1].mean(dim=1).squeeze(0))

            # Teacher
            t_out = teacher_model(**teacher_tokens)
            t_hidden = t_out.hidden_states
            for tidx in [8, 16, 24]:
                if tidx < len(t_hidden):
                    teacher_hiddens[tidx].append(
                        t_hidden[tidx].float().mean(dim=1).squeeze(0))

        if (sample_idx + 1) % 2 == 0:
            print(f"    Sample {sample_idx+1}/{n_samples} done")

    results = {}
    pairs = [(7, 8), (15, 16), (23, 24)]

    for s_idx, t_idx in pairs:
        if student_hiddens[s_idx] and teacher_hiddens[t_idx]:
            S = torch.stack(student_hiddens[s_idx]).float()
            T_mat = torch.stack(teacher_hiddens[t_idx]).float()

            n = min(S.shape[0], T_mat.shape[0])
            S, T_mat = S[:n], T_mat[:n]
            S = S - S.mean(0, keepdim=True)
            T_mat = T_mat - T_mat.mean(0, keepdim=True)

            cka = linear_CKA(S, T_mat)
            results[f"cka_student{s_idx}_teacher{t_idx}"] = round(cka, 4)
            print(f"  CKA(student_L{s_idx}, teacher_L{t_idx}) = {cka:.4f}")

    del teacher_model
    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("STUDENT DIAGNOSTIC PROBES (CPU-only)")
    print("=" * 60)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(STUDENT_CKPT),
        "n_batches": N_BATCHES,
        "seq_len": SEQ_LEN,
    }

    print("\n[1/7] Loading student model...")
    model, cfg = load_student()
    results["config"] = cfg

    print("\n[2/7] Loading eval data...")
    batches = get_eval_data()

    print("\n[3/7] Extracting hidden states + SVD spectrum...")
    all_states = []
    for x, y in batches[:4]:
        states = extract_all_hidden_states(model, x)
        all_states.append(states)

    n_layers_total = len(all_states[0])
    merged_states = []
    for layer_idx in range(n_layers_total):
        merged_states.append(torch.cat([s[layer_idx] for s in all_states], dim=0))

    results["svd_spectrum"] = svd_spectrum(merged_states)

    print("\n[4/7] Inter-layer CKA...")
    key_layers = [0, 3, 7, 11, 15, 19, 23, 24]
    key_states = [merged_states[i] for i in key_layers if i < len(merged_states)]
    cka_matrix = inter_layer_cka(key_states)
    results["inter_layer_cka"] = {
        "layer_indices": key_layers[:len(key_states)],
        "cka_matrix": cka_matrix
    }

    print("\n[5/7] Token entropy + exit analysis...")
    results["entropy"] = entropy_analysis(model, batches)
    results["exit_analysis"] = exit_analysis(model, batches)

    print("\n[6/7] Weight analysis + gradient sensitivity...")
    results["weight_analysis"] = weight_analysis(model)
    try:
        results["gradient_sensitivity"] = gradient_sensitivity(model, batches)
    except Exception as e:
        print(f"  Gradient analysis failed: {e}")
        results["gradient_sensitivity"] = {"error": str(e)}

    print("\n[7/7] Student vs Teacher CKA (Qwen3-1.7B on CPU)...")
    try:
        results["student_teacher_cka"] = student_teacher_cka(model, batches)
    except Exception as e:
        print(f"  Student-teacher CKA failed: {e}")
        results["student_teacher_cka"] = {"error": str(e)}

    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    with open(str(OUTPUT), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS COMPLETE in {elapsed:.0f}s")
    print(f"Results: {OUTPUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
