"""Sutra v0.5.3 + Grokfast Production Training.

SAME architecture as v0.5.3 (no changes). Adds Grokfast gradient filter only.
Chrome-validated: +11% BPT improvement from Grokfast(alpha=0.95, lambda=2.0).

This is the safest deployment path per Codex recommendation:
"Ship v0.5.3 + Grokfast as the immediate recovery path."

Resumes from latest v0.5.3 checkpoint. Model architecture unchanged.

Usage:
    1. Stop the running v0.5.3 trainer
    2. python code/train_v053_grokfast.py
    (auto-resumes from latest v0.5.3 checkpoint with Grokfast enabled)
"""

import json, math, os, random, time
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

# --- Hyperparameters (SAME as v0.5.3 trainer) ---
DIM = 768
FF_DIM = 1536
MAX_STEPS_PER_POSITION = 8
WINDOW = 4
K_RETRIEVAL = 8
SEQ_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 8
LR = 8e-4
WARMUP_STEPS = 1000
MAX_STEPS = 100000
EVAL_EVERY = 5000
SAVE_EVERY = 5000
LOG_EVERY = 100
VOCAB_SIZE = 50257

# Grokfast config
# Chrome at dim=128: lambda=2.0 worked. At dim=768: lambda=2.0 DIVERGED (loss 4.26->4.63).
# Fix: start lambda=0 and ramp to 1.0 over 500 steps.
GROKFAST_ALPHA = 0.95
GROKFAST_LAMBDA_MAX = 1.0   # reduced from 2.0 (diverged at production scale)
GROKFAST_DELAY = 200         # steps after resume before enabling
GROKFAST_RAMP = 500          # steps to ramp lambda from 0 to max

import sys
sys.path.insert(0, str(REPO / "code"))
from launch_v053 import create_v053 as _create_model
from grokfast import GrokfastFilter


def load_tokens():
    full_path = REPO / "data" / "minipile_full_tokens.pt"
    subset_path = REPO / "data" / "minipile_tokens.pt"
    path = full_path if full_path.exists() else subset_path
    print(f"Loading tokens from {path.name}...")
    tokens = torch.load(path, weights_only=True)
    print(f"  {len(tokens):,} tokens ({len(tokens)/1e9:.3f}B)")
    n_test = max(1000, int(len(tokens) * 0.005))
    return tokens[:-n_test], tokens[-n_test:]


def sample_batch(tokens, batch_size, seq_len):
    max_start = len(tokens) - seq_len - 1
    idx = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(DEVICE), y.to(DEVICE)


def evaluate(model, test_tokens, n_batches=20):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sample_batch(test_tokens, min(BATCH_SIZE, 8), SEQ_LEN)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = model(x)
                Tc = min(logits.size(1), y.size(1))
                loss = F.cross_entropy(
                    logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                    y[:, :Tc].reshape(-1))
            total_loss += loss.item()
    model.train()
    return {"bpt": (total_loss / n_batches) / math.log(2),
            "loss": total_loss / n_batches}


def generate_sample(model, test_tokens, tokenizer, max_new=100):
    model.eval()
    prompt = test_tokens[:32]
    tokens = prompt.clone().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        for _ in range(max_new):
            ctx = tokens[:, -SEQ_LEN:] if tokens.size(1) > SEQ_LEN else tokens
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(ctx)
            next_logits = logits[:, -1, :].float() / 0.9
            topk_vals, topk_idx = next_logits.topk(40)
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(-1, topk_idx, topk_vals)
            next_token = torch.multinomial(F.softmax(filtered, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)
    model.train()
    prompt_text = tokenizer.decode(prompt.tolist())
    gen_text = tokenizer.decode(tokens[0, 32:].cpu().tolist())
    return {"prompt": prompt_text[:200], "output": gen_text[:400]}


def main():
    print(f"SUTRA v0.5.3 + GROKFAST TRAINING")
    print(f"  Grokfast(alpha={GROKFAST_ALPHA}, lambda_max={GROKFAST_LAMBDA_MAX}, ramp={GROKFAST_RAMP})")
    print(f"  Same v0.5.3 architecture, gradient filter only")
    print(f"Device: {DEVICE}, bf16: True")
    print(f"{'='*60}")

    train_tokens, test_tokens = load_tokens()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = _create_model(
        dim=DIM, ff_dim=FF_DIM, max_steps=MAX_STEPS_PER_POSITION,
        window=WINDOW, k_retrieval=K_RETRIEVAL).to(DEVICE)

    # MUST resume from v0.5.3 checkpoint
    ckpt_dir = REPO / "results" / "checkpoints_v053"
    log_file = REPO / "results" / "v053_grokfast_log.txt"
    metrics_file = REPO / "results" / "v053_grokfast_metrics.json"

    latest = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not latest:
        print("ERROR: No v0.5.3 checkpoint found. Run v0.5.3 trainer first.")
        return

    ckpt = torch.load(latest[-1], weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    start_step = ckpt["step"]
    best_bpt = ckpt.get("best_bpt", float("inf"))
    metrics_history = ckpt.get("metrics", [])
    print(f"RESUMED v0.5.3 from step {start_step}, best BPT={best_bpt:.4f}")
    print(f"Grokfast will activate at step {start_step + GROKFAST_DELAY}")

    params = model.count_params()
    print(f"Params: {params:,} ({params/1e6:.1f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    if "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # Grokfast filter (starts at lam=0, ramps to GROKFAST_LAMBDA_MAX)
    gf = GrokfastFilter(model, alpha=GROKFAST_ALPHA, lam=0.0)  # start at 0
    grokfast_start_step = start_step + GROKFAST_DELAY
    grokfast_full_step = grokfast_start_step + GROKFAST_RAMP

    model.train()
    step = start_step
    running_loss = 0
    loss_count = 0
    start = time.time()

    while step < MAX_STEPS:
        x, y = sample_batch(train_tokens, BATCH_SIZE, SEQ_LEN)

        lr = LR * min(1.0, (step + 1) / WARMUP_STEPS) * (
            0.5 * (1 + math.cos(math.pi * max(0, step - WARMUP_STEPS)
                                 / max(1, MAX_STEPS - WARMUP_STEPS))))
        for pg in opt.param_groups:
            pg["lr"] = lr

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = model(x)
            Tc = min(logits.size(1), y.size(1))
            ce = F.cross_entropy(logits[:, :Tc].reshape(-1, VOCAB_SIZE),
                                 y[:, :Tc].reshape(-1))
            compute_penalty = 0.01 * aux["compute_cost"] if step > MAX_STEPS * 0.6 else 0
            loss = (ce + compute_penalty) / GRAD_ACCUM

        loss.backward()
        running_loss += ce.item()
        loss_count += 1

        if loss_count % GRAD_ACCUM == 0:
            if torch.isnan(loss) or any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            ):
                print(f"WARNING: NaN at step {step}, skipping", flush=True)
                opt.zero_grad()
                loss_count = 0
                running_loss = 0
                continue

            # GROKFAST: apply after delay with lambda ramp
            if step >= grokfast_start_step:
                # Ramp lambda from 0 to max over GROKFAST_RAMP steps
                if step < grokfast_full_step:
                    ramp_frac = (step - grokfast_start_step) / GROKFAST_RAMP
                    gf.lam = GROKFAST_LAMBDA_MAX * ramp_frac
                else:
                    gf.lam = GROKFAST_LAMBDA_MAX
                gf.apply()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            step += 1

            if step % LOG_EVERY == 0:
                avg = running_loss / loss_count
                elapsed = time.time() - start
                tps = (step - start_step) * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / max(elapsed, 1)
                mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                if step >= grokfast_full_step:
                    gf_status = f"ON(l={GROKFAST_LAMBDA_MAX})"
                elif step >= grokfast_start_step:
                    gf_status = f"ramp(l={gf.lam:.2f})"
                else:
                    gf_status = f"in {grokfast_start_step - step}"
                msg = (f"Step {step:>6d}/{MAX_STEPS}: loss={avg:.4f} "
                       f"lr={lr:.2e} {tps:.0f}tok/s {mem:.1f}GB "
                       f"gf={gf_status}")
                print(msg, flush=True)
                with open(log_file, "a") as f:
                    f.write(msg + "\n")
                running_loss = 0
                loss_count = 0

            if step % EVAL_EVERY == 0:
                try:
                    metrics = evaluate(model, test_tokens)
                    bpt = metrics["bpt"]
                    is_best = bpt < best_bpt
                    if is_best:
                        best_bpt = bpt
                        torch.save(model.state_dict(),
                                   REPO / "results" / "v053_grokfast_best.pt")

                    gen = {"prompt": "(eval)", "output": "(eval)"}
                    try:
                        gen = generate_sample(model, test_tokens, tokenizer)
                        gen = {k: v.encode("ascii", errors="replace").decode("ascii")
                               for k, v in gen.items()}
                        print(f"\nGENERATION @ step {step}: {gen['output'][:200]}",
                              flush=True)
                    except Exception as e:
                        print(f"\nGeneration failed: {e}", flush=True)

                    entry = {
                        "step": step, "test_bpt": round(bpt, 4),
                        "best_bpt": round(best_bpt, 4), "is_best": is_best,
                        "lr": lr, "grokfast": step >= grokfast_start_step,
                        "generation": gen,
                        "timestamp": datetime.now().isoformat(),
                    }
                    metrics_history.append(entry)
                    json.dump(metrics_history, open(metrics_file, "w"), indent=2)

                    best_marker = " *BEST*" if is_best else ""
                    eval_msg = f"  EVAL Step {step}: BPT={bpt:.4f}{best_marker}"
                    print(eval_msg, flush=True)
                    with open(log_file, "a") as f:
                        f.write(eval_msg + "\n")
                except Exception as e:
                    print(f"EVAL FAILED at step {step}: {e}", flush=True)

            if step % SAVE_EVERY == 0:
                save_ckpt = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "grokfast": gf.state_dict(),
                    "step": step, "best_bpt": best_bpt,
                    "metrics": metrics_history,
                }
                torch.save(save_ckpt, ckpt_dir / f"step_{step}.pt")
                old = sorted(ckpt_dir.glob("step_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
                for o in old[:-3]:
                    o.unlink()

    elapsed_h = (time.time() - start) / 3600
    print(f"\nDone. {step} steps, {elapsed_h:.1f}h, best BPT={best_bpt:.4f}")


if __name__ == "__main__":
    main()
