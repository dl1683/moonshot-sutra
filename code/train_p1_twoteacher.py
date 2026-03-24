"""Sutra Ekalavya Protocol: Two-Queue Multi-Teacher Knowledge Distillation.

PURPOSE: Absorb external knowledge from diverse pretrained models into Sutra's
recurrent architecture via two independent teacher queues (guru parampara).

Parent: v0.6.0a step 20K (model + optimizer, preserved)
Architecture:
  Q1 (CKA+XTKD): Light intelligent models — teach representations AND output
    - LFM2-1.2B, Qwen3-0.6B-Base, Granite-4.0-h-350m
  Q2 (CKA-only): Heavy diverse models — teach representational geometry
    - Phi-2, Qwen3-4B, Gemma-1B, Pythia, BERT, MiniLM, bge, GPT-2, etc.

Zero permanent teachers. Only student always loaded. Two queue slots active.
Benchmark-based static step allocations (no runtime adaptation).

Loss = L_CE + alpha_q1_cka * L_CKA_Q1 + alpha_q1_xtkd * L_XTKD_Q1
     + alpha_q2_cka * L_CKA_Q2 + 0.20 * L_probe

Key design:
  - CKA every N micro-batches for stable Gram matrices
  - XTKD: byte-level cross-tokenizer logit KD from Q1 teachers
  - Auto VRAM management: GPU bf16 -> GPU NF4 -> CPU fallback
  - Steps 0-250: deep-biased rd burst, then D=10 default
  - 15K steps, full 7-task eval + generation at 15K

Usage:
  python code/train_p1_twoteacher.py --q1-queue code/q1_xtkd_queue.json --q2-queue code/q2_cka_queue.json
"""

import json, math, os, random, time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO = Path(__file__).parent.parent

# Architecture (same as v0.6.0a)
DIM = 768
FF_DIM = 1536
MAX_STEPS = 12
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 4
GRAD_ACCUM = 16          # Effective batch = 64
VOCAB_SIZE = 50257
PROBE_LOSS_COEF = 0.20
GRAD_CLIP_NORM = 0.5

# Distillation coefficients (two-queue Ekalavya Protocol)
ALPHA_Q1_CKA = 0.3       # CKA weight for Q1 (CKA+XTKD) teachers
ALPHA_Q1_XTKD = 0.3      # XTKD weight for Q1 teachers (byte-level logit KD)
ALPHA_Q2_CKA = 0.2       # CKA weight for Q2 (CKA-only) teachers
KD_TEMP = 2.0            # Temperature for XTKD byte-level distillation
CKA_EVERY_N = 4          # Compute CKA every N micro-batches (accumulate between)

# Training schedule
MAX_TRAIN_STEPS = 15000
EVAL_EVERY = 1000
SAVE_EVERY = 1000
ROLLING_SAVE = 500  # Reduced from 100 to avoid OneDrive sync I/O bottleneck
LOG_EVERY = 50

# LR: continuation from parent
CONTINUATION_LR = 3.2e-4
MIN_LR = 1e-4
WARMUP_STEPS = 0

# Depth schedule: rd burst then D=8 default
RD_BURST_END = 250      # Deep-biased rd burst for 250 steps
D_DEFAULT = 10          # Default depth after burst (D8 INVERTED at v060b 15K, D10 optimal at 96.7%)
D_REFRESH_PROB = 0.05   # 5% chance of D=12 refresh after burst

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v060a import create_v060a
from data_loader import ShardedDataset


# ---- Teacher Queue (Ekalavya Protocol: guru parampara) ----

class TeacherSlot:
    """Rotating teacher slot with benchmark-based static step allocations.

    Loads one teacher at a time from a queue. Each teacher gets a fixed number
    of training steps (set in the queue JSON based on model benchmark quality).
    Stronger models get more steps, weaker ones act as regularizers.

    Auto VRAM management: GPU bf16 -> GPU NF4 -> CPU bf16 fallback chain.
    """

    def __init__(self, queue, device, slot_name="slot"):
        self.queue = queue
        self.device = device
        self.slot_name = slot_name  # "Q1" or "Q2" for logging
        self.idx = 0
        self.model = None
        self.tokenizer = None
        self.steps_remaining = 0
        self.current_name = None
        self.cycle = 0

    def check_swap(self, step):
        """Check if current teacher's window is done. Swap if needed.
        Returns (model, tokenizer) or (None, None) if queue empty/all failed.
        """
        if not self.queue:
            return None, None

        if self.model is None or self.steps_remaining <= 0:
            self._swap_next()

        if self.model is not None:
            self.steps_remaining -= 1
        return self.model, self.tokenizer

    def _swap_next(self):
        self._unload()
        attempts = 0
        while attempts < len(self.queue):
            if self.idx >= len(self.queue):
                self.idx = 0
                self.cycle += 1
            entry = self.queue[self.idx]
            self.idx += 1
            attempts += 1

            self.steps_remaining = entry.get("steps", 2000)
            self._load(entry)
            if self.model is not None:
                self.current_name = entry["name"]
                return

        # All teachers failed — clear state, retry after 1000 steps
        print(f"[{self.slot_name}] WARNING: No teachers could be loaded!", flush=True)
        self.current_name = None
        self.steps_remaining = 1000

    def state_dict(self):
        """Save slot state for checkpoint resume."""
        return {
            "idx": self.idx,
            "cycle": self.cycle,
            "steps_remaining": self.steps_remaining,
            "current_name": self.current_name,
        }

    def load_state_dict(self, state):
        """Restore slot state from checkpoint. Model re-loaded on next check_swap."""
        self.idx = state.get("idx", 0)
        self.cycle = state.get("cycle", 0)
        self.steps_remaining = state.get("steps_remaining", 0)
        self.current_name = state.get("current_name", None)
        # Force reload on next check_swap
        self.model = None
        self.tokenizer = None

    @staticmethod
    def _free_vram_gb():
        """Get free GPU VRAM in GB."""
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free / 1e9
        return 0.0

    @staticmethod
    def _estimate_model_gb(name, trust):
        """Quick estimate of model size in GB (bf16). Downloads config only."""
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(name, trust_remote_code=trust)
            # Rough estimate: most models store total params in config
            if hasattr(cfg, 'num_parameters'):
                return cfg.num_parameters * 2 / 1e9  # bf16 = 2 bytes
            # Fallback: vocab * hidden + layers * (4 * hidden^2) * 2 bytes
            v = getattr(cfg, 'vocab_size', 50000)
            h = getattr(cfg, 'hidden_size', 768)
            n = getattr(cfg, 'num_hidden_layers', 12)
            params = v * h + n * 4 * h * h
            return params * 2 / 1e9
        except Exception:
            return 2.0  # conservative default

    def _load(self, entry):
        from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
        name = entry["name"]
        trust = entry.get("trust_remote_code", False)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=trust)
        except Exception as e:
            print(f"[{self.slot_name}] Tokenizer load failed for {name}: {e}", flush=True)
            self.model = None
            return

        # Auto VRAM check -> decide: GPU bf16, GPU NF4, or CPU
        free_vram = self._free_vram_gb()
        est_size = self._estimate_model_gb(name, trust)
        vram_margin = 2.0  # keep 2GB free for training activations

        load_device = "cpu"
        quant_mode = "none"
        if free_vram > est_size + vram_margin:
            load_device = str(self.device)
            quant_mode = "bf16"
        elif free_vram > est_size * 0.25 + vram_margin:
            load_device = str(self.device)
            quant_mode = "int4"
        else:
            load_device = "cpu"
            quant_mode = "cpu_bf16"

        print(f"\n[{self.slot_name}] Loading: {name} "
              f"({self.steps_remaining} steps, cycle={self.cycle})", flush=True)
        print(f"[{self.slot_name}] VRAM free={free_vram:.1f}GB, est={est_size:.1f}GB "
              f"-> {quant_mode} on {load_device}", flush=True)

        self.model = self._try_load_model(entry, name, trust, quant_mode, load_device)

        if self.model is None:
            print(f"[{self.slot_name}] Skipping {name} (load failed)", flush=True)
            return

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        n = sum(p.numel() for p in self.model.parameters())
        loc = "CPU" if self._on_cpu else f"GPU ({quant_mode})"
        print(f"[{self.slot_name}] Loaded: {n / 1e6:.0f}M params on {loc}", flush=True)

    def _try_load_model(self, entry, name, trust, quant_mode, load_device):
        """Load model with automatic fallback: GPU bf16 -> GPU NF4 -> CPU bf16.
        Returns loaded model or None. Handles OOM gracefully."""
        from transformers import AutoModelForCausalLM, AutoModel

        needs_logits = entry.get("xtkd", False)  # Q1 teachers need logit head

        def _load_with(kwargs, target_device=None):
            if needs_logits:
                m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
            else:
                # Try CausalLM first (has hidden_states), fallback to AutoModel
                try:
                    m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
                except Exception:
                    m = AutoModel.from_pretrained(name, **kwargs)
            if target_device:
                m = m.to(target_device)
            return m

        # NF4 = most aggressive GPU quantization (~4x compression)
        nf4_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": trust,
        }

        attempts = []
        if quant_mode == "bf16":
            attempts = [
                ("GPU bf16", {"torch_dtype": torch.bfloat16, "trust_remote_code": trust}, self.device),
                ("GPU NF4", nf4_kwargs, None),
                ("CPU bf16", {"torch_dtype": torch.bfloat16, "trust_remote_code": trust}, None),
            ]
        elif quant_mode == "int4":
            attempts = [
                ("GPU NF4", nf4_kwargs, None),
                ("CPU bf16", {"torch_dtype": torch.bfloat16, "trust_remote_code": trust}, None),
            ]
        else:  # cpu_bf16
            attempts = [
                ("CPU bf16", {"torch_dtype": torch.bfloat16, "trust_remote_code": trust}, None),
            ]

        for label, kwargs, target in attempts:
            try:
                model = _load_with(kwargs, target)
                self._on_cpu = (target is None and "device_map" not in kwargs)
                if label != f"{'GPU' if quant_mode == 'bf16' else quant_mode}":
                    print(f"[{self.slot_name}] Fallback -> {label}", flush=True)
                return model
            except Exception as e:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{self.slot_name}] {label} failed: {type(e).__name__}: {e}", flush=True)
                continue

        print(f"[{self.slot_name}] WARNING: All load attempts failed for {name}", flush=True)
        self._on_cpu = True
        return None

    def _unload(self):
        if self.model is not None:
            print(f"[{self.slot_name}] Unloading: {self.current_name}", flush=True)
            del self.model, self.tokenizer
            self.model = self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def get_slot_repr(model, tokenizer, texts):
    """Generic mean-pooled representations from any HuggingFace model (encoder or decoder).
    Auto-detects model device (GPU or CPU) and routes inputs accordingly.
    Always returns result on DEVICE (GPU) for CKA computation."""
    model_device = next(model.parameters()).device
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512,
                     return_tensors="pt").to(model_device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            hidden = out.hidden_states[-1]
        else:
            hidden = out.last_hidden_state
        attn_mask = enc["attention_mask"].unsqueeze(-1).float()
        r = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1)
    return r.float().to(DEVICE)  # always return on GPU for CKA


# ---- Distillation losses ----


def linear_cka(X, Y):
    """Linear CKA (Centered Kernel Alignment) between two sets of representations.

    Args:
        X: (N, D1) — student representations
        Y: (N, D2) — teacher representations
    Returns:
        scalar CKA similarity in [0, 1]
    """
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Gram matrices
    XtX = X @ X.T  # (N, N)
    YtY = Y @ Y.T  # (N, N)

    # HSIC
    hsic_xy = (XtX * YtY).sum()
    hsic_xx = (XtX * XtX).sum()
    hsic_yy = (YtY * YtY).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-8)
    return hsic_xy / denom


def compute_cka_loss(student_reprs, teacher_reprs):
    """CKA alignment loss between accumulated student and teacher representations.

    Args:
        student_reprs: (N, D_student) mean-pooled student hidden states
        teacher_reprs: (N, D_teacher) mean-pooled teacher hidden states
    Returns:
        scalar (1 - CKA) loss
    """
    cka_sim = linear_cka(student_reprs, teacher_reprs)
    return 1.0 - cka_sim


# ---- Cross-tokenizer logit KD (XTKD: byte-level alignment for Q1 teachers) ----

_byte_groups_cache = {}


def _build_byte_groups(tokenizer, device):
    """For each byte (0-255), precompute a tensor of token IDs whose decoded
    string starts with that byte. Cached per tokenizer instance."""
    key = id(tokenizer)
    if key in _byte_groups_cache:
        cached = _byte_groups_cache[key]
        if cached[0].device == device:
            return cached

    vocab_size = getattr(tokenizer, 'vocab_size', len(tokenizer))
    groups = [[] for _ in range(256)]
    for vid in range(min(vocab_size, 200000)):
        try:
            s = tokenizer.decode([vid])
            if s:
                groups[min(ord(s[0]), 255)].append(vid)
        except Exception:
            pass

    result = tuple(
        torch.tensor(g, dtype=torch.long, device=device) if g
        else torch.tensor([], dtype=torch.long, device=device)
        for g in groups
    )
    _byte_groups_cache[key] = result
    print(f"  Built byte groups: {sum(len(g) for g in result)} tokens mapped to "
          f"{sum(1 for g in result if len(g) > 0)}/256 active bytes")
    return result


def _byte_logsumexp(logits, byte_groups):
    """Project (N, V) token logits to (N, 256) byte logits via grouped logsumexp.
    Gradient-safe: uses torch.logsumexp on indexed views."""
    N = logits.shape[0]
    result = torch.full((N, 256), -1e20, device=logits.device, dtype=logits.dtype)
    for b, grp in enumerate(byte_groups):
        if len(grp) > 0:
            result[:, b] = torch.logsumexp(logits[:, grp], dim=-1)
    return result


def _char_ends(tokenizer, token_ids):
    """Character end positions for a sequence of token IDs."""
    ends = []
    pos = 0
    for tid in token_ids:
        try:
            pos += len(tokenizer.decode([tid]))
        except Exception:
            pos += 1
        ends.append(pos)
    return ends


def compute_cross_tok_kd_loss(s_logits, x_tokens, teacher_model, teacher_tokenizer,
                               gpt2_tokenizer, s_byte_groups, t_byte_groups,
                               temperature=2.0):
    """Cross-tokenizer KL at aligned character boundaries in byte space.

    For each text, finds positions where both GPT-2 and teacher tokenizers have
    a token ending at the same character. At those aligned boundaries, projects
    both models' next-token distributions to 256-byte space and computes KL.

    Args:
        s_logits: (B, T, V_student) -- HAS gradients (student predictions)
        x_tokens: (B, T) GPT-2 token IDs
        teacher_model: any CausalLM teacher (frozen, any tokenizer)
        teacher_tokenizer: teacher's tokenizer
    Returns:
        scalar KL loss (temperature-scaled, averaged over aligned positions)
    """
    device = s_logits.device
    teacher_device = next(teacher_model.parameters()).device
    aligned_s, aligned_t = [], []

    for b in range(s_logits.shape[0]):
        text = gpt2_tokenizer.decode(x_tokens[b].tolist(), skip_special_tokens=False)
        if len(text.strip()) < 10:
            continue

        # Student character boundaries
        s_ends = _char_ends(gpt2_tokenizer, x_tokens[b].tolist())

        # Teacher forward + character boundaries
        with torch.no_grad():
            t_enc = teacher_tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512).to(teacher_device)
            t_logits = teacher_model(t_enc["input_ids"]).logits[0].to(device)  # (T_t, V_t)
            t_ends = _char_ends(teacher_tokenizer, t_enc["input_ids"][0].tolist())

        # Build teacher end->position lookup (first token ending at each char pos)
        t_map = {}
        for i, e in enumerate(t_ends):
            if e not in t_map:
                t_map[e] = i

        # Match boundaries: both tokenizers end a token at the same character
        for s_pos, s_end in enumerate(s_ends):
            if s_end in t_map:
                t_pos = t_map[s_end]
                if s_pos < s_logits.shape[1] and t_pos < t_logits.shape[0]:
                    aligned_s.append(s_logits[b, s_pos])  # (V_s,) with grad
                    aligned_t.append(t_logits[t_pos])      # (V_t,) no grad

    if len(aligned_s) < 4:
        return torch.tensor(0.0, device=device)

    s_stack = torch.stack(aligned_s)  # (N_aligned, V_s) with grad
    t_stack = torch.stack(aligned_t)  # (N_aligned, V_t) no grad

    # Project to byte space via logsumexp
    s_byte = _byte_logsumexp(s_stack / temperature, s_byte_groups)  # (N, 256)
    t_byte = _byte_logsumexp(t_stack / temperature, t_byte_groups)  # (N, 256)

    # KL divergence in byte space (student learns teacher's byte-level preferences)
    return F.kl_div(
        F.log_softmax(s_byte, dim=-1),
        F.softmax(t_byte, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)


# ---- Depth sampling ----

def sample_depth_p1(step):
    """P1 depth schedule: rd burst then D=8 default with occasional D=12."""
    if step <= RD_BURST_END:
        # Deep-biased burst (same as P0)
        if step <= 100:
            return random.choice([10, 11, 12])
        else:
            weights = torch.arange(8, MAX_STEPS + 1, dtype=torch.float) ** 2
            idx = torch.multinomial(weights, 1).item()
            return idx + 8
    else:
        # Mostly D=8 with occasional D=12 refresh
        if random.random() < D_REFRESH_PROB:
            return 12
        return D_DEFAULT


# ---- Standard functions (from v0.6.0c) ----

def cosine_continuation_lr(step):
    progress = step / max(MAX_TRAIN_STEPS, 1)
    return MIN_LR + 0.5 * (CONTINUATION_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def autocast_ctx():
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(f"{path.stem}.tmp")
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def compute_base_losses(logits, aux, y):
    """Base CE + probe loss (same as v0.6.0c)."""
    B, T, V = logits.shape
    n_steps = aux["avg_steps"]

    L_final = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

    L_probe = torch.tensor(0.0, device=logits.device)
    if "probe_pred" in aux and "sampled_ce_hist" in aux:
        probe_pred = aux["probe_pred"]
        sampled_ce = aux["sampled_ce_hist"]
        with torch.no_grad():
            targets = torch.zeros_like(sampled_ce)
            for p in range(n_steps - 1):
                future_min = sampled_ce[:, :, p+1:].min(dim=2).values
                targets[:, :, p] = (sampled_ce[:, :, p] - future_min).clamp(min=0, max=4)
        L_probe = F.smooth_l1_loss(probe_pred, targets.detach())

    return L_final, L_probe


def evaluate(model, dataset, n_batches=20):
    model.eval()
    total_loss = 0
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = dataset.sample_batch(min(BATCH_SIZE, 4), SEQ_LEN, device=DEVICE, split="test")
                with autocast_ctx():
                    logits, _ = model(x, n_steps=MAX_STEPS)
                    Tc = min(logits.size(1), y.size(1))
                    loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                total_loss += loss.item()
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"bpt": (total_loss / n_batches) / math.log(2)}


def evaluate_per_depth(model, dataset, n_batches=10):
    model.eval()
    results = {}
    try:
        with torch.no_grad():
            cached = []
            for _ in range(n_batches):
                x, y = dataset.sample_batch(2, SEQ_LEN, device=DEVICE, split="test")
                cached.append((x, y))
            for d in range(1, MAX_STEPS + 1):
                total = 0
                for x, y in cached:
                    with autocast_ctx():
                        logits, _ = model(x, n_steps=d)
                        Tc = min(logits.size(1), y.size(1))
                        loss = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE), y[:, :Tc].reshape(-1))
                    total += loss.item()
                results[d] = round((total / n_batches) / math.log(2), 4)
    finally:
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def run_eval(model, dataset, step, ckpt_dir, metrics_history, best_bpt):
    print(f"\n{'='*50}")
    print(f"EVAL at step {step}")
    print(f"{'='*50}")

    metrics = evaluate(model, dataset)
    bpt = metrics["bpt"]
    is_best = bpt < best_bpt
    if is_best:
        best_bpt = bpt

    print(f"  BPT (D={MAX_STEPS}): {bpt:.4f} {'*BEST*' if is_best else ''}")

    per_depth = evaluate_per_depth(model, dataset, n_batches=10)
    d8_bpt = per_depth.get(8, 99)
    d12_bpt = per_depth.get(12, 99)
    print(f"  Per-depth: D=1:{per_depth.get(1,99):.2f} D=4:{per_depth.get(4,99):.2f} "
          f"D=8:{d8_bpt:.2f} D=12:{d12_bpt:.2f}")

    # late_pct
    d1_bpt = per_depth.get(1, 99)
    if d1_bpt > d12_bpt > 0:
        total_improv = d1_bpt - d12_bpt
        mid_bpt = per_depth.get(6, 99)
        late_improv = mid_bpt - d12_bpt
        late_pct = late_improv / total_improv * 100
        print(f"  late_pct: {late_pct:.1f}%")

    # Save checkpoint
    atomic_torch_save({
        "model": model.state_dict(),
        "step": step,
        "bpt": bpt,
        "best_bpt": best_bpt,
        "per_depth_bpt": per_depth,
    }, ckpt_dir / f"step_{step}.pt")
    print(f"  Checkpoint saved: step_{step}.pt")

    entry = {
        "step": step,
        "test_bpt": round(bpt, 4),
        "best_bpt": round(best_bpt, 4),
        "is_best": is_best,
        "per_depth_bpt": {str(k): v for k, v in per_depth.items()},
        "timestamp": datetime.now().isoformat(),
    }
    metrics_history.append(entry)
    return best_bpt, metrics_history


def save_rolling(model, opt, dataset, step, best_bpt, metrics_history, depth_counts, ckpt_dir,
                  q1_slot=None, q2_slot=None):
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "best_bpt": best_bpt,
        "metrics": metrics_history,
        "depth_counts": depth_counts,
        "dataset": dataset.state_dict() if hasattr(dataset, 'state_dict') else None,
        "torch_rng": torch.random.get_rng_state(),
        "random_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    if q1_slot is not None:
        state["q1_state"] = q1_slot.state_dict()
    if q2_slot is not None:
        state["q2_state"] = q2_slot.state_dict()
    atomic_torch_save(state, ckpt_dir / "rolling_latest.pt")


# ---- GPT-2 tokenizer for text decoding ----

_gpt2_tokenizer = None

def get_gpt2_tokenizer():
    global _gpt2_tokenizer
    if _gpt2_tokenizer is None:
        from transformers import AutoTokenizer
        _gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return _gpt2_tokenizer


def decode_tokens_to_texts(token_ids):
    """Decode GPT-2 token IDs to text strings for encoder teacher."""
    tok = get_gpt2_tokenizer()
    texts = []
    for seq in token_ids:
        text = tok.decode(seq.tolist(), skip_special_tokens=True)
        texts.append(text[:1024])  # truncate for encoder
    return texts


# ---- Main ----

def main(weight_file=None, run_name="p1", q1_queue=None, q2_queue=None):
    variant = "P1b (MiniPLM weighted)" if weight_file else "P1a (uniform)"
    n_q1 = len(q1_queue) if q1_queue else 0
    n_q2 = len(q2_queue) if q2_queue else 0
    print(f"SUTRA EKALAVYA PROTOCOL: TWO-QUEUE MULTI-TEACHER KD -- {variant}")
    print(f"  Parent: v0.6.0a step 20K (model + optimizer, preserved)")
    print(f"  Q1 (CKA+XTKD): {n_q1} teachers, alpha_cka={ALPHA_Q1_CKA}, alpha_xtkd={ALPHA_Q1_XTKD}")
    print(f"  Q2 (CKA-only): {n_q2} teachers, alpha_cka={ALPHA_Q2_CKA}")
    print(f"  Depth: rd burst 0-{RD_BURST_END}, then D={D_DEFAULT}")
    print(f"  LR: {CONTINUATION_LR:.1e} -> {MIN_LR:.1e} cosine")
    print(f"  Steps: {MAX_TRAIN_STEPS}, eval every {EVAL_EVERY}")
    if weight_file:
        print(f"  Weight file: {weight_file}")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"Training: bs={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}, seq={SEQ_LEN}")
    print(f"{'='*60}")

    # Load dataset (with optional MiniPLM weights for P1b)
    dataset = ShardedDataset(weight_file=weight_file)

    ckpt_dir = REPO / "results" / f"checkpoints_{run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_file = REPO / "results" / f"{run_name}_log.txt"
    metrics_file = REPO / "results" / f"{run_name}_metrics.json"

    # Create student model
    model = create_v060a(dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS,
                         window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # --- Resume or load from parent ---
    start_step = 0
    best_bpt = float("inf")
    metrics_history = []
    depth_counts = [0] * (MAX_STEPS + 1)
    resumed = False

    # Check for P1 resume
    resume_candidates = []
    rolling = ckpt_dir / "rolling_latest.pt"
    if rolling.exists():
        resume_candidates.append(rolling)
    for p in sorted(ckpt_dir.glob("step_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]), reverse=True):
        resume_candidates.append(p)

    resumed_ckpt = None
    for cand in resume_candidates:
        try:
            c = torch.load(cand, weights_only=False, map_location=DEVICE)
            if "optimizer" in c and (resumed_ckpt is None or c.get("step", 0) > resumed_ckpt.get("step", 0)):
                resumed_ckpt = c
                resumed_ckpt["_path"] = cand.name
        except Exception:
            continue

    if resumed_ckpt is not None:
        model.load_state_dict(resumed_ckpt["model"])
        start_step = resumed_ckpt["step"]
        best_bpt = resumed_ckpt.get("best_bpt", float("inf"))
        metrics_history = resumed_ckpt.get("metrics", [])
        depth_counts = resumed_ckpt.get("depth_counts", [0] * (MAX_STEPS + 1))
        if "torch_rng" in resumed_ckpt:
            rng = resumed_ckpt["torch_rng"]
            torch.random.set_rng_state(rng.cpu() if isinstance(rng, torch.Tensor) else rng)
        if "random_rng" in resumed_ckpt:
            random.setstate(resumed_ckpt["random_rng"])
        if "cuda_rng" in resumed_ckpt and resumed_ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            cuda_rng = resumed_ckpt["cuda_rng"]
            torch.cuda.set_rng_state(cuda_rng.cpu() if isinstance(cuda_rng, torch.Tensor) else cuda_rng)
        if "dataset" in resumed_ckpt and resumed_ckpt["dataset"] is not None and hasattr(dataset, 'load_state_dict'):
            dataset.load_state_dict(resumed_ckpt["dataset"])
        print(f"RESUMED P1 from step {start_step} ({resumed_ckpt['_path']})")
        resumed = True

    if not resumed:
        # Load from v0.6.0a step 20K
        source_path = REPO / "results" / "checkpoints_v060a" / "step_20000.pt"
        if not source_path.exists():
            print(f"ERROR: Parent checkpoint not found: {source_path}")
            return
        source_ckpt = torch.load(source_path, weights_only=False, map_location=DEVICE)
        model.load_state_dict(source_ckpt["model"])
        print(f"Loaded model from v0.6.0a step {source_ckpt['step']} (BPT={source_ckpt['best_bpt']:.4f})")

    print(f"Params: {model.count_params():,} ({model.count_params()/1e6:.1f}M)")

    # Optimizer (preserve moments from parent)
    opt = torch.optim.AdamW(model.parameters(), lr=CONTINUATION_LR, weight_decay=0.01, betas=(0.9, 0.95))

    if resumed and "optimizer" in resumed_ckpt:
        opt.load_state_dict(resumed_ckpt["optimizer"])
        print(f"  Restored optimizer state from {resumed_ckpt['_path']}")
    elif not resumed:
        opt.load_state_dict(source_ckpt["optimizer"])
        for pg in opt.param_groups:
            pg["lr"] = CONTINUATION_LR
        print(f"Loaded optimizer moments from parent (preserved, NOT reset)")
        del source_ckpt

    model.train()

    # ---- Initialize teacher queues (Ekalavya Protocol) ----
    # Mark Q1 teachers as needing logits for XTKD
    if q1_queue:
        for entry in q1_queue:
            entry["xtkd"] = True  # flag for _try_load_model to use CausalLM

    q1_slot = TeacherSlot(q1_queue or [], DEVICE, slot_name="Q1") if q1_queue else None
    q2_slot = TeacherSlot(q2_queue or [], DEVICE, slot_name="Q2") if q2_queue else None

    # Pre-build student byte groups for XTKD (GPT-2 tokenizer, constant)
    s_byte_groups = None
    gpt2_tok = get_gpt2_tokenizer()
    if q1_slot is not None:
        print("Precomputing student byte groups for XTKD...", flush=True)
        s_byte_groups = _build_byte_groups(gpt2_tok, DEVICE)

    # Restore queue state from checkpoint if resuming
    if resumed and resumed_ckpt is not None:
        if q1_slot and "q1_state" in resumed_ckpt:
            q1_slot.load_state_dict(resumed_ckpt["q1_state"])
            print(f"  Restored Q1 queue state (idx={q1_slot.idx}, cycle={q1_slot.cycle})")
        if q2_slot and "q2_state" in resumed_ckpt:
            q2_slot.load_state_dict(resumed_ckpt["q2_state"])
            print(f"  Restored Q2 queue state (idx={q2_slot.idx}, cycle={q2_slot.cycle})")
        del resumed_ckpt

    print(f"\n{'='*60}")
    print(f"Ekalavya Protocol ready: Q1={n_q1} teachers, Q2={n_q2} teachers")
    print(f"{'='*60}\n")

    # Step 0 eval
    if start_step == 0:
        best_bpt, metrics_history = run_eval(model, dataset, 0, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    # ---- Training loop ----
    step = start_step
    running_losses = {"L_CE": 0, "L_probe": 0, "L_CKA_Q1": 0, "L_XTKD": 0, "L_CKA_Q2": 0, "L_total": 0}
    running_depth = 0
    loss_count = 0
    start_time = time.time()

    # CKA accumulation buffers (larger effective batch for stable Gram matrices)
    cka_student_buf = []   # list of (B, D) tensors — DETACHED except last
    cka_text_buf = []      # list of text strings
    cka_x_buf = []         # list of (B, T) token tensors for XTKD

    # Cache for teacher byte groups (rebuilt on teacher swap)
    _t_byte_groups_cache = {}  # {teacher_name: byte_groups}

    while step < MAX_TRAIN_STEPS:
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split="train")

        # LR schedule
        lr = cosine_continuation_lr(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Depth schedule
        D = sample_depth_p1(step)
        depth_counts[D] += 1

        with autocast_ctx():
            # Student forward
            logits, aux = model(x, y=y, n_steps=D)
            Tc = min(logits.size(1), y.size(1))
            s_logits = logits[:, :Tc]

            # Base losses (CE + probe)
            L_CE, L_probe = compute_base_losses(s_logits, aux, y[:, :Tc])

            # ---- Two-Queue Distillation (Ekalavya Protocol) ----
            L_CKA_Q1 = torch.tensor(0.0, device=DEVICE)
            L_XTKD = torch.tensor(0.0, device=DEVICE)
            L_CKA_Q2 = torch.tensor(0.0, device=DEVICE)

            if "mu_hist" in aux and aux["mu_hist"] is not None:
                student_hidden = aux["mu_hist"][:, :Tc, -1, :]  # (B, T, D)
                s_repr_live = student_hidden.float().mean(dim=1)  # (B, D) with gradients
                texts = decode_tokens_to_texts(x)

                # FIX: Detach all prior micro-batch reps (their graphs were consumed
                # by earlier backward passes). Only keep current batch live.
                for i in range(len(cka_student_buf)):
                    if cka_student_buf[i].requires_grad:
                        cka_student_buf[i] = cka_student_buf[i].detach()
                cka_student_buf.append(s_repr_live)  # only this one has live graph
                cka_text_buf.extend(texts)
                cka_x_buf.append(x.detach())  # for XTKD (always detached, teacher uses it)

                # Compute distillation at end of CKA accumulation window
                if len(cka_student_buf) >= CKA_EVERY_N:
                    all_s = torch.cat(cka_student_buf, dim=0)  # (N*B, D)
                    all_texts = cka_text_buf[:]

                    # Q1: CKA + XTKD from light intelligent teacher
                    if q1_slot is not None:
                        q1_model, q1_tok = q1_slot.check_swap(step)
                        if q1_model is not None:
                            # CKA alignment
                            q1_repr = get_slot_repr(q1_model, q1_tok, all_texts)
                            L_CKA_Q1 = compute_cka_loss(all_s, q1_repr)

                            # XTKD: byte-level cross-tokenizer logit KD
                            if s_byte_groups is not None:
                                t_name = q1_slot.current_name or ""
                                if t_name not in _t_byte_groups_cache:
                                    _t_byte_groups_cache[t_name] = _build_byte_groups(q1_tok, DEVICE)
                                t_bg = _t_byte_groups_cache[t_name]
                                L_XTKD = compute_cross_tok_kd_loss(
                                    s_logits, x[:, :Tc], q1_model, q1_tok,
                                    gpt2_tok, s_byte_groups, t_bg,
                                    temperature=KD_TEMP)

                    # Q2: CKA-only from heavy diverse teacher
                    if q2_slot is not None:
                        q2_model, q2_tok = q2_slot.check_swap(step)
                        if q2_model is not None:
                            q2_repr = get_slot_repr(q2_model, q2_tok, all_texts)
                            L_CKA_Q2 = compute_cka_loss(all_s, q2_repr)

                    # Clear buffers
                    cka_student_buf.clear()
                    cka_text_buf.clear()
                    cka_x_buf.clear()

            # Combined loss (CKA compensated for 1-in-N frequency)
            q1_cka_w = ALPHA_Q1_CKA * CKA_EVERY_N if L_CKA_Q1.item() > 0 else 0.0
            q1_xtkd_w = ALPHA_Q1_XTKD * CKA_EVERY_N if L_XTKD.item() > 0 else 0.0
            q2_cka_w = ALPHA_Q2_CKA * CKA_EVERY_N if L_CKA_Q2.item() > 0 else 0.0
            L_total = (L_CE
                       + q1_cka_w * L_CKA_Q1
                       + q1_xtkd_w * L_XTKD
                       + q2_cka_w * L_CKA_Q2
                       + PROBE_LOSS_COEF * L_probe)
            L_total = L_total / GRAD_ACCUM

        if not torch.isfinite(L_total):
            print(f"WARNING: Non-finite loss at step {step} (D={D}), skipping", flush=True)
            opt.zero_grad()
            loss_count = 0
            running_losses = {k: 0 for k in running_losses}
            running_depth = 0
            cka_student_buf.clear()
            cka_text_buf.clear()
            cka_x_buf.clear()
            continue

        L_total.backward()
        running_losses["L_CE"] += L_CE.item()
        running_losses["L_probe"] += L_probe.item()
        running_losses["L_CKA_Q1"] += L_CKA_Q1.item()
        running_losses["L_XTKD"] += L_XTKD.item()
        running_losses["L_CKA_Q2"] += L_CKA_Q2.item()
        running_losses["L_total"] += (L_total * GRAD_ACCUM).item()
        running_depth += D
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in model.parameters()):
                print(f"WARNING: NaN/Inf grad at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                cka_student_buf.clear()
                cka_text_buf.clear()
                cka_x_buf.clear()
                continue

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avgs = {k: v / loss_count for k, v in running_losses.items()}
                avg_d = running_depth / loss_count
                elapsed = time.time() - start_time
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                q1_name = (q1_slot.current_name or "none").split("/")[-1] if q1_slot else "off"
                q2_name = (q2_slot.current_name or "none").split("/")[-1] if q2_slot else "off"
                msg = (f"Step {step:>5d}: L={avgs['L_total']:.4f} "
                       f"(CE={avgs['L_CE']:.3f} "
                       f"Q1cka={avgs['L_CKA_Q1']:.3f} XTKD={avgs['L_XTKD']:.3f} "
                       f"Q2cka={avgs['L_CKA_Q2']:.3f} "
                       f"prb={avgs['L_probe']:.3f}) "
                       f"D={avg_d:.1f} lr={lr:.2e} {tps:.0f}tok/s "
                       f"[Q1:{q1_name} Q2:{q2_name}]")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_losses = {k: 0 for k in running_losses}
                running_depth = 0
                loss_count = 0

            # Eval BEFORE rolling save so checkpoint captures post-eval metadata
            if step % EVAL_EVERY == 0:
                best_bpt, metrics_history = run_eval(
                    model, dataset, step, ckpt_dir, metrics_history, best_bpt)
                with open(metrics_file, "w") as mf:
                    json.dump(metrics_history, mf, indent=2)

            if step % ROLLING_SAVE == 0:
                save_rolling(model, opt, dataset, step, best_bpt, metrics_history,
                             depth_counts, ckpt_dir, q1_slot=q1_slot, q2_slot=q2_slot)

    # Final eval
    if step > 0 and step % EVAL_EVERY != 0:
        best_bpt, metrics_history = run_eval(
            model, dataset, step, ckpt_dir, metrics_history, best_bpt)
        with open(metrics_file, "w") as mf:
            json.dump(metrics_history, mf, indent=2)

    print(f"\n{'='*60}")
    print(f"{run_name.upper()} COMPLETE: {step} steps, best BPT={best_bpt:.4f}")
    print(f"\nDepth distribution:")
    total_samples = sum(depth_counts)
    for d in range(1, MAX_STEPS + 1):
        if depth_counts[d] > 0:
            pct = depth_counts[d] / max(total_samples, 1) * 100
            print(f"  D={d:2d}: {depth_counts[d]:5d} ({pct:5.1f}%)")
    print(f"\nRun full eval: python code/lm_eval_wrapper.py --checkpoint results/checkpoints_p1/step_{step}.pt --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa,sciq,lambada_openai")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sutra Ekalavya Protocol: Two-Queue Multi-Teacher KD")
    parser.add_argument("--weight-file", type=str, default=None,
                        help="MiniPLM shard weight file for P1b weighted sampling")
    parser.add_argument("--run-name", type=str, default="p1",
                        help="Run name for checkpoint/metrics dirs (default: p1)")
    parser.add_argument("--q1-queue", type=str, default=None,
                        help="JSON file for Q1 teachers (CKA+XTKD, light intelligent models)")
    parser.add_argument("--q2-queue", type=str, default=None,
                        help="JSON file for Q2 teachers (CKA-only, heavy diverse models)")
    args = parser.parse_args()

    def _load_queue(path_str, label):
        if path_str is None:
            return None
        p = Path(path_str)
        if p.exists():
            q = json.load(open(p))
            print(f"Loaded {label}: {len(q)} teachers from {p}")
            return q
        print(f"WARNING: {label} file not found: {p}")
        return None

    q1 = _load_queue(args.q1_queue, "Q1 (CKA+XTKD)")
    q2 = _load_queue(args.q2_queue, "Q2 (CKA-only)")

    main(weight_file=args.weight_file, run_name=args.run_name,
         q1_queue=q1, q2_queue=q2)
