"""Eklavya Experiment E1 / E1.5 — Embedding tomography vs standard KD.

E1 (original): 4 arms with KL-based tomography loss.
  Known flaw: avg(KL(P_t||Q)) = KL(avg(P_t)||Q), erasing teacher identity.

E1.5 (corrected adjudication): Teacher-indexed auxiliary heads.
  Each teacher gets its own linear head over the shared encoder. This
  genuinely breaks the algebraic identity: Q_t != Q_t' for different
  teachers. B4c absorber is support-matched (same heads, gold targets).
  32-doc raw-student hard negatives, 3+ seeds, paired t-test CI.

Arms (E1.5):
  B0:   Contrastive-only (no teacher)
  B2:   Single-teacher KD (identity probe only)
  B3:   Calibrated multi-teacher avg KD (softmax then average)
  E1.5: Teacher-indexed KL (multi-probe, multi-teacher, per-teacher heads)
  E1.5id: Teacher-indexed KL (identity probe only) — ablation
  B4c:  Matched absorber (same heads, same support, gold targets)

Usage:
  python code/experiment_e1.py --device cuda --steps 600 --out_dir outputs/E1
  python code/experiment_e1.py --mode e1.5 --device cuda --out_dir outputs/E1_5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def gpu_thermal_guard(max_temp: int = 85, check_interval: float = 5.0):
    """Block until GPU temperature drops below max_temp. No-op on CPU.
    Times out after 120s and resumes with a warning to prevent infinite hangs."""
    if not torch.cuda.is_available():
        return
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        temp = int(result.stdout.strip())
        if temp >= max_temp:
            print(f"  [thermal] GPU at {temp}°C (limit {max_temp}°C), cooling...")
            t0 = time.time()
            while temp >= max_temp - 3:
                if time.time() - t0 > 120:
                    print(f"  [thermal] timeout after 120s at {temp}°C, resuming anyway")
                    break
                time.sleep(check_interval)
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                temp = int(result.stdout.strip())
            else:
                print(f"  [thermal] GPU cooled to {temp}°C, resuming")
    except Exception:
        pass

from embed_tomography import (
    generate_probes,
    Probe,
    load_model as load_st_model,
)
from data_loader import load_hard_toy, load_msmarco_pairs


class ModernBERTEmbedder(nn.Module):
    """Wraps ModernBERT-base as a sentence embedder with mean pooling + projection."""

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", dim: int = 384,
                 proj_seed: int | None = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768
        if proj_seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(proj_seed)
            self.proj = nn.Linear(hidden, dim)
            torch.random.set_rng_state(rng_state)
        else:
            self.proj = nn.Linear(hidden, dim)
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        )
        encoded = {k: v.to(next(self.encoder.parameters()).device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)
        token_embs = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode(self, texts, convert_to_tensor=True, normalize_embeddings=True, **kwargs):
        """sentence-transformers compatible interface."""
        embs = self.forward(texts)
        if not convert_to_tensor:
            return embs.cpu().numpy()
        return embs


@torch.no_grad()
def extract_teacher_scores(teacher, query: str, documents: list[str], probes: list[Probe]) -> dict[str, list[float]]:
    """Get teacher similarity scores for each probe."""
    doc_embs = teacher.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    scores = {}
    for probe in probes:
        q_emb = teacher.encode([probe.text], convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ doc_embs.T).squeeze(0).cpu().tolist()
        scores[probe.probe_id] = sims
    return scores


def compute_tomography_loss(
    student: ModernBERTEmbedder,
    query_probes: list[Probe],
    documents: list[str],
    teacher_scores: dict[str, dict[str, list[float]]],
    tau: float = 0.05,
) -> torch.Tensor:
    """KL divergence between student and teacher ranking distributions across all probes and teachers."""
    doc_embs = student.forward(documents)
    loss = torch.tensor(0.0, device=doc_embs.device)
    n = 0

    for probe in query_probes:
        q_emb = student.forward([probe.text])
        student_sims = (q_emb @ doc_embs.T).squeeze(0)
        student_log_dist = F.log_softmax(student_sims / tau, dim=0)

        for tname, tsig in teacher_scores.items():
            if probe.probe_id in tsig:
                target_sims = torch.tensor(tsig[probe.probe_id], dtype=torch.float32, device=doc_embs.device)
                target_dist = F.softmax(target_sims / tau, dim=0)
                kl = F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)
                loss = loss + kl
                n += 1

    return loss / max(n, 1)


def compute_kd_loss(
    student: ModernBERTEmbedder,
    query: str,
    documents: list[str],
    teacher_scores: list[float],
    tau: float = 0.05,
) -> torch.Tensor:
    """Standard KD: match one teacher's ranking distribution on identity query only."""
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    student_sims = (q_emb @ doc_embs.T).squeeze(0)
    student_log_dist = F.log_softmax(student_sims / tau, dim=0)
    target = torch.tensor(teacher_scores, dtype=torch.float32, device=doc_embs.device)
    target_dist = F.softmax(target / tau, dim=0)
    return F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)


def compute_reverse_kl_tomography_loss(
    student: ModernBERTEmbedder,
    query_probes: list[Probe],
    documents: list[str],
    teacher_scores: dict[str, dict[str, list[float]]],
    tau: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Reverse KL tomography: KL(Q_student || P_teacher) per teacher per probe.

    Breaks the algebraic identity because the gradient involves log(P_t)
    nonlinearly: avg_t(log P_t(x)) != log(avg_t(P_t(x))) by Jensen's
    inequality. Unlike forward KL, there is no dead zone — gradient is
    well-defined everywhere softmax outputs are positive.
    """
    doc_embs = student.forward(documents)
    device = doc_embs.device
    loss = torch.tensor(0.0, device=device)
    n = 0
    n_agreements = 0
    n_disagreements = 0

    teacher_names = list(teacher_scores.keys())

    for probe in query_probes:
        q_emb = student.forward([probe.text])
        student_sims = (q_emb @ doc_embs.T).squeeze(0)
        student_log_dist = F.log_softmax(student_sims / tau, dim=0)
        student_dist = student_log_dist.exp()

        for tname in teacher_names:
            tsig = teacher_scores[tname]
            if probe.probe_id not in tsig:
                continue
            target_sims = torch.tensor(
                tsig[probe.probe_id], dtype=torch.float32, device=device,
            )
            target_log_dist = F.log_softmax(target_sims / tau, dim=0)

            kl = F.kl_div(target_log_dist.detach(), student_dist,
                          reduction="batchmean", log_target=False)
            loss = loss + kl
            n += 1

        if len(teacher_names) >= 2:
            probe_scores = []
            for tname in teacher_names:
                if probe.probe_id in teacher_scores.get(tname, {}):
                    probe_scores.append(teacher_scores[tname][probe.probe_id])
            if len(probe_scores) >= 2:
                s0, s1 = probe_scores[0], probe_scores[1]
                for i in range(len(documents)):
                    for j in range(i + 1, len(documents)):
                        d0 = s0[i] - s0[j]
                        d1 = s1[i] - s1[j]
                        if abs(d0) > 0.01 and abs(d1) > 0.01:
                            if (d0 > 0) == (d1 > 0):
                                n_agreements += 1
                            else:
                                n_disagreements += 1

    diag = {
        "n_terms": n,
        "n_teacher_agreements": n_agreements,
        "n_teacher_disagreements": n_disagreements,
        "disagreement_rate": n_disagreements / max(n_agreements + n_disagreements, 1),
    }
    return loss / max(n, 1), diag


def compute_b4c_pairwise_loss(
    student: ModernBERTEmbedder,
    query_probes: list[Probe],
    documents: list[str],
    gold_idx: int,
    margin: float = 0.05,
) -> torch.Tensor:
    """B4c absorber for identity-preserving loss: same probes, same pairwise
    structure, but targets are gold-label preferences (gold doc > every other)
    instead of teacher-specific rankings. If E1.5 = B4c, then the per-teacher
    signal is just augmentation."""
    doc_embs = student.forward(documents)
    device = doc_embs.device
    loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for probe in query_probes:
        q_emb = student.forward([probe.text])
        student_sims = (q_emb @ doc_embs.T).squeeze(0)

        for j in range(len(documents)):
            if j == gold_idx:
                continue
            s_diff = student_sims[gold_idx] - student_sims[j]
            loss = loss + F.relu(margin - s_diff)
            n_pairs += 1

    return loss / max(n_pairs, 1)


def filter_safe_probes(probes: list[Probe]) -> list[Probe]:
    """Exclude negation probes — gold label may not be valid for negated queries."""
    return [p for p in probes if p.probe_id != "negation"]


def make_teacher_heads(
    teacher_names: list[str], dim: int, device, proj_seed: int | None = None,
) -> nn.ModuleDict:
    """Create per-teacher auxiliary linear heads over the shared embedding space."""
    heads = {}
    for tname in teacher_names:
        key = tname.replace("/", "_").replace("-", "_").replace(".", "_")
        if proj_seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(proj_seed + hash(key) % (2**31))
            heads[key] = nn.Linear(dim, dim)
            torch.random.set_rng_state(rng_state)
        else:
            heads[key] = nn.Linear(dim, dim)
    return nn.ModuleDict(heads).to(device)


def _head_key(tname: str) -> str:
    return tname.replace("/", "_").replace("-", "_").replace(".", "_")


def compute_teacher_indexed_kl_loss(
    student: ModernBERTEmbedder,
    teacher_heads: nn.ModuleDict,
    query_probes: list[Probe],
    documents: list[str],
    teacher_scores: dict[str, dict[str, list[float]]],
    tau: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Teacher-indexed KL: each teacher's distribution is matched through its own head.

    Student embedding goes through shared encoder + proj, then through a
    teacher-specific linear head before computing similarity and KL divergence.
    This breaks the algebraic identity: Q_t != Q_t' for different teachers,
    so avg_t KL(P_t || Q_t) != KL(avg(P_t) || Q_shared).
    """
    doc_embs_base = student.forward(documents)
    device = doc_embs_base.device
    loss = torch.tensor(0.0, device=device)
    n = 0

    for probe in query_probes:
        q_emb_base = student.forward([probe.text])

        for tname, tsig in teacher_scores.items():
            if probe.probe_id not in tsig:
                continue
            head = teacher_heads[_head_key(tname)]
            q_emb = F.normalize(head(q_emb_base), p=2, dim=-1)
            doc_embs = F.normalize(head(doc_embs_base), p=2, dim=-1)

            student_sims = (q_emb @ doc_embs.T).squeeze(0)
            student_log_dist = F.log_softmax(student_sims / tau, dim=0)

            target_sims = torch.tensor(
                tsig[probe.probe_id], dtype=torch.float32, device=device,
            )
            target_dist = F.softmax(target_sims / tau, dim=0)
            kl = F.kl_div(
                student_log_dist, target_dist.detach(),
                reduction="batchmean", log_target=False,
            )
            loss = loss + kl
            n += 1

    return loss / max(n, 1), {"n_terms": n}


def compute_b4c_matched_kl_loss(
    student: ModernBERTEmbedder,
    teacher_heads: nn.ModuleDict,
    query_probes: list[Probe],
    documents: list[str],
    gold_idx: int,
    teacher_names: list[str],
    tau: float = 0.05,
) -> torch.Tensor:
    """B4c absorber matched to teacher-indexed loss: identical architecture,
    identical support (every teacher × every probe), but targets are gold
    preferences (one-hot on gold doc) instead of teacher distributions.
    If teacher-indexed E1.5 = matched B4c, teacher distributions add nothing."""
    doc_embs_base = student.forward(documents)
    device = doc_embs_base.device
    loss = torch.tensor(0.0, device=device)
    n = 0
    target = torch.zeros(len(documents), device=device)
    target[gold_idx] = 1.0

    for probe in query_probes:
        q_emb_base = student.forward([probe.text])
        for tname in teacher_names:
            head = teacher_heads[_head_key(tname)]
            q_emb = F.normalize(head(q_emb_base), p=2, dim=-1)
            doc_embs = F.normalize(head(doc_embs_base), p=2, dim=-1)

            student_sims = (q_emb @ doc_embs.T).squeeze(0)
            student_log_dist = F.log_softmax(student_sims / tau, dim=0)

            kl = F.kl_div(
                student_log_dist, target.detach(),
                reduction="batchmean", log_target=False,
            )
            loss = loss + kl
            n += 1

    return loss / max(n, 1)


def compute_contrastive_loss(
    student: ModernBERTEmbedder,
    query: str,
    documents: list[str],
    gold_idx: int,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss — no teacher, just (query, gold_doc) pairs."""
    doc_embs = student.forward(documents)
    q_emb = student.forward([query])
    sims = (q_emb @ doc_embs.T).squeeze(0) / tau
    target = torch.tensor(gold_idx, device=sims.device)
    return F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))


@torch.no_grad()
def evaluate(student: ModernBERTEmbedder, pairs: list[dict], mrr_cutoff: int = 10) -> dict:
    was_training = student.training
    student.eval()
    hits1 = hits5 = 0
    mrr_sum = 0.0
    mrr_cut_sum = 0.0
    per_query = []
    for pair in pairs:
        q_emb = student.forward([pair["query"]])
        d_embs = student.forward(pair["documents"])
        sims = (q_emb @ d_embs.T).squeeze(0)
        ranked = sims.argsort(descending=True).tolist()
        gold = pair["gold_idx"]
        if ranked[0] == gold:
            hits1 += 1
        if gold in ranked[:5]:
            hits5 += 1
        rank = ranked.index(gold) + 1
        rr = 1.0 / rank
        rr_cut = rr if rank <= mrr_cutoff else 0.0
        mrr_sum += rr
        mrr_cut_sum += rr_cut
        per_query.append({"id": pair["id"], "gold_rank": rank, "rr": rr, f"rr@{mrr_cutoff}": rr_cut})
    if was_training:
        student.train()
    n = len(pairs)
    return {
        "hit@1": hits1/n, "hit@5": hits5/n, "mrr": mrr_sum/n,
        f"mrr@{mrr_cutoff}": mrr_cut_sum/n, "n": n, "per_query": per_query,
    }


def run_arm(
    arm_name: str,
    student: ModernBERTEmbedder,
    train_pairs: list[dict],
    eval_pairs: list[dict],
    teacher_data: dict | None,
    steps: int,
    lr: float,
    tau: float,
    out_dir: str,
    arm_type: str = "tomography",
    teacher_heads: nn.ModuleDict | None = None,
    teacher_names: list[str] | None = None,
    frozen: bool = False,
    warmup_frac: float = 0.0,
    fade_step: int | None = None,
    eval_every: int = 200,
):
    gpu_thermal_guard(max_temp=85)
    fade_info = f" [FADE->contrastive@{fade_step}]" if fade_step else ""
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name} ({arm_type}){' [FROZEN]' if frozen else ''}{fade_info}")
    print(f"{'='*60}")

    if frozen:
        for p in student.encoder.parameters():
            p.requires_grad = False
        params = list(student.proj.parameters())
    else:
        params = list(student.parameters())
    if teacher_heads is not None:
        params += list(teacher_heads.parameters())
    optimizer = AdamW(params, lr=lr, weight_decay=0.01)

    import math
    warmup_steps = int(steps * warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Baseline
    base = evaluate(student, eval_pairs)
    print(f"  Baseline: Hit@1={base['hit@1']:.4f}  Hit@5={base['hit@5']:.4f}  MRR={base['mrr']:.4f}")

    arm_dir = os.path.join(out_dir, arm_name)
    Path(arm_dir).mkdir(parents=True, exist_ok=True)
    log_f = open(os.path.join(arm_dir, "log.jsonl"), "w")

    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    meta_entry = {
        "event": "arm_meta", "arm": arm_name, "type": arm_type,
        "frozen": frozen, "trainable_params": n_trainable,
        "fade_step": fade_step, "eval_every": eval_every, "steps": steps,
    }
    log_f.write(json.dumps(meta_entry) + "\n")
    log_f.flush()

    checkpoint_steps = set()
    if fade_step:
        checkpoint_steps.add(fade_step)
    checkpoint_steps.add(steps)

    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        idx = (step - 1) % len(train_pairs)
        pair = train_pairs[idx]

        optimizer.zero_grad()

        effective_type = arm_type
        if fade_step and step > fade_step:
            effective_type = "contrastive"

        if effective_type == "tomography":
            probes = generate_probes(pair["query"], seed=idx)
            loss = compute_tomography_loss(
                student, probes, pair["documents"],
                teacher_data[pair["id"]], tau=tau,
            )
        elif effective_type == "teacher_indexed":
            probes = filter_safe_probes(generate_probes(pair["query"], seed=idx))
            loss, _diag = compute_teacher_indexed_kl_loss(
                student, teacher_heads, probes, pair["documents"],
                teacher_data[pair["id"]], tau=tau,
            )
        elif effective_type == "teacher_indexed_id_only":
            probes = [Probe(probe_id="identity", text=pair["query"])]
            loss, _diag = compute_teacher_indexed_kl_loss(
                student, teacher_heads, probes, pair["documents"],
                teacher_data[pair["id"]], tau=tau,
            )
        elif effective_type == "b4c_matched":
            probes = filter_safe_probes(generate_probes(pair["query"], seed=idx))
            loss = compute_b4c_matched_kl_loss(
                student, teacher_heads, probes, pair["documents"],
                pair["gold_idx"], teacher_names, tau=tau,
            )
        elif effective_type == "kd_single":
            tid = list(teacher_data[pair["id"]].keys())[0]
            loss = compute_kd_loss(
                student, pair["query"], pair["documents"],
                teacher_data[pair["id"]][tid]["identity"], tau=tau,
            )
        elif effective_type == "kd_avg":
            scores_lists = [t["identity"] for t in teacher_data[pair["id"]].values()]
            device = next(student.parameters()).device
            calibrated = []
            for scores in scores_lists:
                t = torch.tensor(scores, dtype=torch.float32)
                calibrated.append(F.softmax(t / tau, dim=0))
            avg_dist = torch.stack(calibrated).mean(dim=0)
            avg_scores = (avg_dist.log() * tau).tolist()
            loss = compute_kd_loss(
                student, pair["query"], pair["documents"],
                avg_scores, tau=tau,
            )
        elif effective_type == "contrastive":
            loss = compute_contrastive_loss(
                student, pair["query"], pair["documents"],
                pair["gold_idx"], tau=tau,
            )
        else:
            raise ValueError(f"Unknown arm type: {effective_type}")

        if not loss.requires_grad:
            continue
        loss.backward()
        all_params = list(student.parameters())
        if teacher_heads is not None:
            all_params += list(teacher_heads.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 100 == 0:
            gpu_thermal_guard(max_temp=85)
        if step % 50 == 0:
            avg = running_loss / 50
            current_lr = scheduler.get_last_lr()[0]
            phase = "contrastive" if (fade_step and step > fade_step) else arm_type
            entry = {"step": step, "loss": round(avg, 6), "elapsed_s": round(time.time()-t0, 1),
                     "lr": current_lr, "phase": phase}
            if step % eval_every == 0:
                m = evaluate(student, eval_pairs)
                entry.update(m)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}  mrr@10={m.get('mrr@10', 0):.4f}  [{phase}]")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}  [{phase}]")
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

        if step in checkpoint_steps:
            ckpt_path = os.path.join(arm_dir, f"proj_step{step}.pt")
            torch.save(student.proj.state_dict(), ckpt_path)

    # Final eval
    final = evaluate(student, eval_pairs)
    result = {
        "arm": arm_name,
        "type": arm_type,
        "steps": steps,
        "baseline": base,
        "final": final,
        "gain_hit1": final["hit@1"] - base["hit@1"],
        "gain_mrr": final["mrr"] - base["mrr"],
    }
    print(f"\n  RESULT: Hit@1 {base['hit@1']:.4f} -> {final['hit@1']:.4f} ({result['gain_hit1']:+.4f})")
    print(f"          MRR   {base['mrr']:.4f} -> {final['mrr']:.4f} ({result['gain_mrr']:+.4f})")

    if frozen:
        for p in student.encoder.parameters():
            p.requires_grad = True

    with open(os.path.join(arm_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log_f.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Eklavya E1 — Embedding Tomography Experiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=150)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--out_dir", default="outputs/E1")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load data — MS MARCO for real retrieval difficulty
    print("Loading MS MARCO data...")
    all_pairs = load_msmarco_pairs(n=args.n_train + args.n_eval, seed=42)
    if len(all_pairs) < args.n_train + args.n_eval:
        print(f"Warning: only got {len(all_pairs)} pairs, falling back to hard toy data")
        all_pairs = load_hard_toy(n=args.n_train + args.n_eval, seed=42)
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]
    print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval pairs")

    # Extract teacher signatures for ALL probes on train data
    print("\nExtracting teacher signatures...")
    teachers = {}
    for tname in args.teachers:
        print(f"  Loading {tname}")
        teachers[tname] = load_st_model(tname, device=args.device)

    teacher_data = {}
    for pair in train_pairs:
        probes = generate_probes(pair["query"], seed=train_pairs.index(pair))
        td = {}
        for tname, tmodel in teachers.items():
            td[tname] = extract_teacher_scores(tmodel, pair["query"], pair["documents"], probes)
        teacher_data[pair["id"]] = td

    # Free teacher memory
    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted signatures for {len(teacher_data)} pairs")

    # Save experiment config
    config = {
        "student": args.student,
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "device": args.device,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    results = {}

    # ARM B0: Contrastive only (no teacher)
    student_b0 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B0_contrastive"] = run_arm(
        "B0_contrastive", student_b0, train_pairs, eval_pairs,
        teacher_data=None, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="contrastive",
    )
    del student_b0
    torch.cuda.empty_cache()

    # ARM B2: Single-teacher KD (best teacher, identity probe only)
    student_b2 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B2_kd_single"] = run_arm(
        "B2_kd_single", student_b2, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_single",
    )
    del student_b2
    torch.cuda.empty_cache()

    # ARM B3: Multi-teacher average KD (average scores, identity only)
    student_b3 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["B3_kd_avg"] = run_arm(
        "B3_kd_avg", student_b3, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_avg",
    )
    del student_b3
    torch.cuda.empty_cache()

    # ARM E1: Full tomography (multi-probe, multi-teacher)
    student_e1 = ModernBERTEmbedder(args.student, dim=384).to(args.device)
    results["E1_tomography"] = run_arm(
        "E1_tomography", student_e1, train_pairs, eval_pairs,
        teacher_data=teacher_data, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="tomography",
    )
    del student_e1
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT E1 SUMMARY")
    print("=" * 60)
    print(f"{'Arm':<20} {'Hit@1':>8} {'MRR':>8} {'Gain Hit@1':>12} {'Gain MRR':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['final']['hit@1']:>8.4f} {r['final']['mrr']:>8.4f} "
              f"{r['gain_hit1']:>+12.4f} {r['gain_mrr']:>+10.4f}")

    # Save summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Verdict
    tomo = results["E1_tomography"]
    best_baseline = max(results["B0_contrastive"]["final"]["mrr"],
                        results["B2_kd_single"]["final"]["mrr"],
                        results["B3_kd_avg"]["final"]["mrr"])
    margin = tomo["final"]["mrr"] - best_baseline
    print(f"\nTomography vs best baseline MRR margin: {margin:+.4f}")
    if margin > 0.01:
        print("VERDICT: Tomography shows signal. Proceed to E2.")
    elif margin > -0.01:
        print("VERDICT: Inconclusive. Need more data/training/harder eval.")
    else:
        print("VERDICT: Tomography absorbed. Investigate why.")


def main_e15():
    """E1.5 — Corrected text adjudication with teacher-indexed auxiliary heads.

    Fixes from Codex design gate:
    - Teacher-indexed heads break the algebraic identity genuinely
    - B4c matched on architecture + support (same heads, gold targets)
    - Negation probes excluded from label-sensitive arms
    - Eval mode in evaluate(), proper RNG seeding
    - Paired bootstrap CI with t-distribution for small seed counts
    - Calibrated B3 (average softmax distributions, not raw scores)
    """
    import random as stdlib_random
    import numpy as np
    from scipy import stats as sp_stats

    parser = argparse.ArgumentParser(description="Eklavya E1.5 — Corrected Adjudication")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_docs", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 271])
    parser.add_argument("--proj_seed", type=int, default=9999)
    parser.add_argument("--out_dir", default="outputs/E1_5")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"])
    parser.add_argument("--frozen", action="store_true",
                        help="Freeze encoder, train only projection + teacher heads")
    parser.add_argument("--warmup_frac", type=float, default=0.1,
                        help="Fraction of steps for linear LR warmup")
    args = parser.parse_args()

    from data_loader import load_msmarco_pairs, mine_hard_negatives

    ARMS = [
        ("B0_contrastive", "contrastive"),
        ("B2_kd_single", "kd_single"),
        ("B3_kd_avg_cal", "kd_avg"),
        ("E15_teacher_indexed", "teacher_indexed"),
        ("E15_teacher_idx_id", "teacher_indexed_id_only"),
        ("B4c_matched", "b4c_matched"),
    ]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_seed_results = {}

    for seed_idx, data_seed in enumerate(args.seeds):
        print(f"\n{'#'*60}")
        print(f"SEED {seed_idx + 1}/{len(args.seeds)}: data_seed={data_seed}")
        print(f"{'#'*60}")

        stdlib_random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(data_seed)

        seed_dir = os.path.join(args.out_dir, f"seed_{data_seed}")
        Path(seed_dir).mkdir(parents=True, exist_ok=True)

        all_done = all(
            os.path.exists(os.path.join(seed_dir, arm_name, "result.json"))
            for arm_name, _ in ARMS
        )
        if all_done:
            print(f"\n--- Seed {data_seed}: all arms complete, loading results ---")
            seed_results_loaded = {}
            for arm_name, _ in ARMS:
                seed_results_loaded[arm_name] = json.load(
                    open(os.path.join(seed_dir, arm_name, "result.json"))
                )
            all_seed_results[data_seed] = seed_results_loaded
            continue

        raw_pairs = load_msmarco_pairs(
            n=args.n_train + args.n_eval, n_docs=10, seed=data_seed,
        )
        if len(raw_pairs) < args.n_train + args.n_eval:
            print(f"Warning: only got {len(raw_pairs)} pairs")

        train_raw = raw_pairs[:args.n_train]
        eval_raw = raw_pairs[args.n_train : args.n_train + args.n_eval]

        gpu_thermal_guard(max_temp=85)
        print("\nMining hard negatives with raw student...")
        raw_student = ModernBERTEmbedder(
            args.student, dim=384, proj_seed=args.proj_seed,
        ).to(args.device)
        raw_student.eval()

        train_pairs = mine_hard_negatives(
            train_raw, raw_student, n_docs=args.n_docs,
        )
        eval_pairs = mine_hard_negatives(
            eval_raw, raw_student, n_docs=args.n_docs,
        )
        del raw_student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval, {args.n_docs} docs/query")

        print("\nExtracting teacher signatures...")
        teachers = {}
        for tname in args.teachers:
            print(f"  Loading {tname}")
            teachers[tname] = load_st_model(tname, device=args.device)

        teacher_data = {}
        for pair in train_pairs:
            probes = generate_probes(pair["query"], seed=train_pairs.index(pair))
            td = {}
            for tname, tmodel in teachers.items():
                td[tname] = extract_teacher_scores(
                    tmodel, pair["query"], pair["documents"], probes,
                )
            teacher_data[pair["id"]] = td

        del teachers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed_results = {}
        for arm_name, arm_type in ARMS:
            arm_result_path = os.path.join(seed_dir, arm_name, "result.json")
            if os.path.exists(arm_result_path):
                print(f"\n--- {arm_name}: result.json exists, resuming (skip) ---")
                seed_results[arm_name] = json.load(open(arm_result_path))
                continue

            torch.manual_seed(data_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(data_seed)

            student = ModernBERTEmbedder(
                args.student, dim=384, proj_seed=args.proj_seed,
            ).to(args.device)

            t_heads = None
            if arm_type in ("teacher_indexed", "teacher_indexed_id_only", "b4c_matched"):
                t_heads = make_teacher_heads(
                    args.teachers, student.dim, args.device,
                    proj_seed=args.proj_seed,
                )

            td = teacher_data if arm_type != "contrastive" else None
            result = run_arm(
                arm_name, student, train_pairs, eval_pairs,
                teacher_data=td, steps=args.steps, lr=args.lr, tau=args.tau,
                out_dir=seed_dir, arm_type=arm_type,
                teacher_heads=t_heads, teacher_names=args.teachers,
                frozen=args.frozen, warmup_frac=args.warmup_frac,
            )

            seed_results[arm_name] = result
            del student
            if t_heads is not None:
                del t_heads
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_seed_results[data_seed] = seed_results

        with open(os.path.join(seed_dir, "summary.json"), "w") as f:
            summary_clean = {}
            for k, v in seed_results.items():
                entry = {kk: vv for kk, vv in v.items() if kk != "per_query"}
                entry["baseline"] = {kk: vv for kk, vv in v["baseline"].items() if kk != "per_query"}
                entry["final"] = {kk: vv for kk, vv in v["final"].items() if kk != "per_query"}
                summary_clean[k] = entry
            json.dump(summary_clean, f, indent=2)
        with open(os.path.join(seed_dir, "per_query.json"), "w") as f:
            pq = {k: {"baseline": v["baseline"].get("per_query", []),
                       "final": v["final"].get("per_query", [])}
                  for k, v in seed_results.items()}
            json.dump(pq, f, indent=2)

    # --- Multi-seed summary ---
    print(f"\n{'='*60}")
    print("E1.5 MULTI-SEED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Arm':<25} {'Mean MRR':>10} {'Std':>8} {'Mean Gain':>10}")
    print("-" * 55)

    arm_stats = {}
    for arm_name, _ in ARMS:
        mrrs = [all_seed_results[s][arm_name]["final"]["mrr"] for s in args.seeds]
        gains = [all_seed_results[s][arm_name]["gain_mrr"] for s in args.seeds]
        mean_mrr = np.mean(mrrs)
        std_mrr = np.std(mrrs, ddof=1) if len(mrrs) > 1 else 0.0
        mean_gain = np.mean(gains)
        print(f"{arm_name:<25} {mean_mrr:>10.4f} {std_mrr:>8.4f} {mean_gain:>+10.4f}")
        arm_stats[arm_name] = {
            "mrrs": [float(m) for m in mrrs],
            "gains": [float(g) for g in gains],
            "mean_mrr": float(mean_mrr),
            "std_mrr": float(std_mrr),
            "mean_gain": float(mean_gain),
        }

    # Paired comparison: E15 teacher-indexed vs B4c matched absorber
    e15 = arm_stats["E15_teacher_indexed"]
    b4c = arm_stats["B4c_matched"]
    n_seeds = len(args.seeds)
    delta_mrrs = [e15["mrrs"][i] - b4c["mrrs"][i] for i in range(n_seeds)]
    mean_delta = np.mean(delta_mrrs)
    std_delta = np.std(delta_mrrs, ddof=1) if n_seeds > 1 else 0.0
    se_delta = std_delta / max(n_seeds ** 0.5, 1)
    t_crit = sp_stats.t.ppf(0.975, df=max(n_seeds - 1, 1))

    print(f"\nE1.5 vs B4c (paired): mean delta = {mean_delta:+.4f}, SE = {se_delta:.4f}")
    ci_lo = mean_delta - t_crit * se_delta
    ci_hi = mean_delta + t_crit * se_delta
    print(f"95% CI (t, df={n_seeds-1}): [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    threshold = 0.005
    if ci_lo > threshold:
        verdict = "Teacher-indexed tomography shows signal above B4c absorber."
    elif ci_hi < threshold:
        verdict = "Teacher-indexed tomography absorbed by B4c. Kill confirmed."
    else:
        verdict = "Inconclusive. Need more seeds or data."
    print(f"VERDICT: {verdict}")

    final_summary = {
        "arms": arm_stats,
        "seeds": args.seeds,
        "config": {k: v for k, v in vars(args).items() if k not in ("device",)},
        "paired_test": {
            "e15_vs_b4c_delta": [float(d) for d in delta_mrrs],
            "mean_delta": float(mean_delta),
            "se": float(se_delta),
            "ci_95": [float(ci_lo), float(ci_hi)],
            "t_crit": float(t_crit),
            "df": n_seeds - 1,
            "threshold": threshold,
            "verdict": verdict,
        },
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)


def paired_bootstrap_ci(diffs: list[float], n_boot: int = 10000, alpha: float = 0.05) -> dict:
    """Bootstrap CI for mean of paired differences."""
    arr = np.array(diffs)
    n = len(arr)
    rng = np.random.RandomState(42)
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": n, "significant": (lo > 0) or (hi < 0)}


def main_fade():
    """Teacher fade experiment — stability-plasticity discriminator.

    Tests whether early teacher scaffolding transmits real information that
    survives removal. Uses the same data pipeline as E1.5.

    Arms (2 new, 2 reused from E1.5):
    - B0_contrastive: reuse from E1.5 (pure contrastive baseline)
    - B2_kd_single: reuse from E1.5 (full KD reference)
    - B2_fade_200: KD for 200 steps, then contrastive for 400 (NEW)
    - B0_sched_match: contrastive with matched schedule control (NEW)
    """
    import random as stdlib_random
    import numpy as np

    parser = argparse.ArgumentParser(description="Eklavya Fade — Stability-Plasticity Test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fade_step", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_docs", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 271])
    parser.add_argument("--proj_seed", type=int, default=9999)
    parser.add_argument("--out_dir", default="outputs/E_fade")
    parser.add_argument("--e15_dir", default="outputs/E1_5_text")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teachers", nargs="+",
                        default=["sentence-transformers/all-MiniLM-L12-v2", "BAAI/bge-large-en-v1.5"])
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--manifest", default=None,
                        help="Path to frozen manifest JSON. If provided, load pairs from it.")
    args = parser.parse_args()

    from data_loader import load_msmarco_pairs, mine_hard_negatives
    import hashlib

    FADE_ARMS = [
        ("B3_fade_200", "kd_avg"),
        ("B2_fade_200", "kd_single"),
        ("B0_sched_match", "contrastive"),
    ]

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    manifest_data = None
    if args.manifest:
        print(f"Loading frozen manifest from {args.manifest}")
        manifest_data = json.load(open(args.manifest))
        stored_hash = manifest_data.get("_sha256", "")
        print(f"  Manifest SHA-256: {stored_hash[:16]}...")
        print(f"  Seeds in manifest: {list(manifest_data.get('seeds', {}).keys())}")

    all_seed_results = {}

    for seed_idx, data_seed in enumerate(args.seeds):
        print(f"\n{'#'*60}")
        print(f"FADE SEED {seed_idx + 1}/{len(args.seeds)}: data_seed={data_seed}")
        print(f"{'#'*60}")

        stdlib_random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(data_seed)

        seed_dir = os.path.join(args.out_dir, f"seed_{data_seed}")
        Path(seed_dir).mkdir(parents=True, exist_ok=True)

        all_done = all(
            os.path.exists(os.path.join(seed_dir, arm_name, "result.json"))
            for arm_name, _ in FADE_ARMS
        )
        if all_done:
            print(f"\n--- Seed {data_seed}: all fade arms complete, loading ---")
            seed_results = {}
            for arm_name, _ in FADE_ARMS:
                seed_results[arm_name] = json.load(
                    open(os.path.join(seed_dir, arm_name, "result.json"))
                )
            all_seed_results[data_seed] = seed_results
            continue

        if manifest_data and str(data_seed) in manifest_data.get("seeds", {}):
            print("Loading pairs from frozen manifest...")
            seed_manifest = manifest_data["seeds"][str(data_seed)]
            eval_pairs_from_manifest = []
            for qdata in seed_manifest["queries"]:
                eval_pairs_from_manifest.append({
                    "id": qdata["id"],
                    "query": qdata["query"],
                    "documents": qdata.get("documents", []),
                    "gold_idx": qdata["gold_idx"],
                })
            if eval_pairs_from_manifest and eval_pairs_from_manifest[0].get("documents"):
                print(f"  Loaded {len(eval_pairs_from_manifest)} eval pairs from manifest")
            else:
                print("  WARNING: Manifest lacks document texts. Falling back to mining.")
                manifest_data = None

        if not manifest_data or str(data_seed) not in manifest_data.get("seeds", {}):
            raw_pairs = load_msmarco_pairs(
                n=args.n_train + args.n_eval, n_docs=10, seed=data_seed,
            )
            if len(raw_pairs) < args.n_train + args.n_eval:
                print(f"Warning: only got {len(raw_pairs)} pairs")

            train_raw = raw_pairs[:args.n_train]
            eval_raw = raw_pairs[args.n_train : args.n_train + args.n_eval]

            gpu_thermal_guard(max_temp=85)
            print("\nMining hard negatives with raw student...")
            raw_student = ModernBERTEmbedder(
                args.student, dim=384, proj_seed=args.proj_seed,
            ).to(args.device)
            raw_student.eval()

            train_pairs = mine_hard_negatives(
                train_raw, raw_student, n_docs=args.n_docs,
            )
            eval_pairs = mine_hard_negatives(
                eval_raw, raw_student, n_docs=args.n_docs,
            )
            del raw_student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            raw_pairs = load_msmarco_pairs(
                n=args.n_train + args.n_eval, n_docs=10, seed=data_seed,
            )
            train_raw = raw_pairs[:args.n_train]
            raw_student = ModernBERTEmbedder(
                args.student, dim=384, proj_seed=args.proj_seed,
            ).to(args.device)
            raw_student.eval()
            train_pairs = mine_hard_negatives(
                train_raw, raw_student, n_docs=args.n_docs,
            )
            eval_pairs = eval_pairs_from_manifest
            del raw_student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval")

        print("\nExtracting teacher signatures for fade arms...")
        teachers = {}
        for tname in args.teachers:
            print(f"  Loading {tname}")
            teachers[tname] = load_st_model(tname, device=args.device)

        teacher_data = {}
        for pair in train_pairs:
            probes = generate_probes(pair["query"], seed=train_pairs.index(pair))
            td = {}
            for tname, tmodel in teachers.items():
                td[tname] = extract_teacher_scores(
                    tmodel, pair["query"], pair["documents"], probes,
                )
            teacher_data[pair["id"]] = td

        del teachers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        seed_results = {}
        for arm_name, arm_type in FADE_ARMS:
            arm_result_path = os.path.join(seed_dir, arm_name, "result.json")
            if os.path.exists(arm_result_path):
                print(f"\n--- {arm_name}: exists, skip ---")
                seed_results[arm_name] = json.load(open(arm_result_path))
                continue

            torch.manual_seed(data_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(data_seed)

            student = ModernBERTEmbedder(
                args.student, dim=384, proj_seed=args.proj_seed,
            ).to(args.device)

            fade = args.fade_step if "fade" in arm_name else None

            result = run_arm(
                arm_name, student, train_pairs, eval_pairs,
                teacher_data=teacher_data, steps=args.steps, lr=args.lr,
                tau=args.tau, out_dir=seed_dir, arm_type=arm_type,
                frozen=args.frozen, warmup_frac=args.warmup_frac,
                fade_step=fade, eval_every=100,
            )

            seed_results[arm_name] = result
            del student
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_seed_results[data_seed] = seed_results

    # Cross-seed comparison with E1.5 baselines
    print(f"\n{'='*70}")
    print(f"FADE EXPERIMENT RESULTS (vs E1.5 baselines)")
    print(f"{'='*70}")

    e15_arms = ["B0_contrastive", "B2_kd_single"]
    fade_arms_names = [a for a, _ in FADE_ARMS]
    all_arm_names = e15_arms + fade_arms_names

    print(f"\n{'Arm':<25} {'Mean MRR':>10} {'MRR@10':>10} {'SD':>8} {'Mean Gain':>10}")
    print("-" * 65)

    arm_pq = {}
    for arm_name in all_arm_names:
        mrrs = []
        mrr10s = []
        gains = []
        pqs = []
        for data_seed in args.seeds:
            if arm_name in e15_arms:
                fpath = os.path.join(args.e15_dir, f"seed_{data_seed}", arm_name, "result.json")
            else:
                fpath = os.path.join(args.out_dir, f"seed_{data_seed}", arm_name, "result.json")
            if os.path.exists(fpath):
                r = json.load(open(fpath))
                mrrs.append(r["final"]["mrr"])
                mrr10s.append(r["final"].get("mrr@10", r["final"]["mrr"]))
                gains.append(r.get("gain_mrr", r["final"]["mrr"] - r["baseline"]["mrr"]))
                pqs.append({q["id"]: q for q in r["final"]["per_query"]})

        if mrrs:
            mean_mrr = np.mean(mrrs)
            mean_mrr10 = np.mean(mrr10s)
            std_mrr = np.std(mrrs, ddof=1) if len(mrrs) > 1 else 0.0
            mean_gain = np.mean(gains)
            print(f"{arm_name:<25} {mean_mrr:>10.4f} {mean_mrr10:>10.4f} {std_mrr:>8.4f} {mean_gain:>+10.4f}")
            arm_pq[arm_name] = pqs

    print(f"\n--- Paired Bootstrap CI (fade vs B0, per-query RR differences) ---")
    b0_pqs = arm_pq.get("B0_contrastive", [])
    for fade_arm in fade_arms_names:
        fade_pqs = arm_pq.get(fade_arm, [])
        if not fade_pqs or not b0_pqs or len(fade_pqs) != len(b0_pqs):
            continue
        all_diffs = []
        for si in range(len(fade_pqs)):
            for qid in fade_pqs[si]:
                if qid in b0_pqs[si]:
                    all_diffs.append(fade_pqs[si][qid]["rr"] - b0_pqs[si][qid]["rr"])
        if all_diffs:
            ci = paired_bootstrap_ci(all_diffs)
            sig = "***" if ci["significant"] else "n.s."
            print(f"  {fade_arm} - B0: mean={ci['mean']:+.4f} "
                  f"95% CI [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] "
                  f"n={ci['n']} {sig}")

    # Per-query analysis using LEAVE-ONE-SEED-OUT subgroup definition
    # (avoids circular selection: define helped/hurt from OTHER seeds)
    print(f"\n{'='*70}")
    print("PER-QUERY FADE ANALYSIS (leave-one-seed-out subgroups)")
    print(f"{'='*70}")

    for fade_arm in ["B3_fade_200", "B2_fade_200"]:
        ref_arm = "B3_kd_avg_cal" if "B3" in fade_arm else "B2_kd_single"
        print(f"\n  --- {fade_arm} vs B0 (ref: {ref_arm}) ---")

        for data_seed in args.seeds:
            other_seeds = [s for s in args.seeds if s != data_seed]
            fade_path = os.path.join(args.out_dir, f"seed_{data_seed}", fade_arm, "result.json")
            b0_path = os.path.join(args.e15_dir, f"seed_{data_seed}", "B0_contrastive", "result.json")

            if not all(os.path.exists(p) for p in [fade_path, b0_path]):
                continue

            b0 = {q["id"]: q for q in json.load(open(b0_path))["final"]["per_query"]}
            fade_r = {q["id"]: q for q in json.load(open(fade_path))["final"]["per_query"]}

            helped_votes = {}
            hurt_votes = {}
            for qid in b0:
                h_count = 0
                r_count = 0
                for os_seed in other_seeds:
                    os_b0 = os.path.join(args.e15_dir, f"seed_{os_seed}", "B0_contrastive", "result.json")
                    os_ref = os.path.join(args.e15_dir, f"seed_{os_seed}", ref_arm, "result.json")
                    if not all(os.path.exists(p) for p in [os_b0, os_ref]):
                        continue
                    os_b0_pq = {q["id"]: q for q in json.load(open(os_b0))["final"]["per_query"]}
                    os_ref_pq = {q["id"]: q for q in json.load(open(os_ref))["final"]["per_query"]}
                    if os_ref_pq[qid]["rr"] - os_b0_pq[qid]["rr"] > 0.001:
                        h_count += 1
                    elif os_b0_pq[qid]["rr"] - os_ref_pq[qid]["rr"] > 0.001:
                        r_count += 1
                helped_votes[qid] = h_count
                hurt_votes[qid] = r_count

            helped = [q for q in b0 if helped_votes[q] >= 2]
            hurt = [q for q in b0 if hurt_votes[q] >= 2]

            if helped:
                fade_help_delta = np.mean([fade_r[q]["rr"] - b0[q]["rr"] for q in helped])
                print(f"  Seed {data_seed}: {len(helped)} cross-validated helped queries")
                print(f"    Fade delta vs B0: {fade_help_delta:+.4f}")

            if hurt:
                fade_hurt_delta = np.mean([fade_r[q]["rr"] - b0[q]["rr"] for q in hurt])
                print(f"    {len(hurt)} cross-validated hurt queries")
                print(f"    Fade delta vs B0: {fade_hurt_delta:+.4f}")

    # Manipulation check: B3_fade must beat B0 at step 200 (before switch)
    print(f"\n{'='*70}")
    print("MANIPULATION CHECK: B3_fade_200 vs B0_sched_match at step 200")
    print(f"{'='*70}")
    manip_pass = True
    for data_seed in args.seeds:
        b3_log = os.path.join(args.out_dir, f"seed_{data_seed}", "B3_fade_200", "log.jsonl")
        b0_log = os.path.join(args.out_dir, f"seed_{data_seed}", "B0_sched_match", "log.jsonl")
        b3_mrr200 = b0_mrr200 = None
        for logpath, label in [(b3_log, "B3"), (b0_log, "B0")]:
            if os.path.exists(logpath):
                for line in open(logpath):
                    entry = json.loads(line)
                    if entry.get("step") == args.fade_step and "mrr" in entry:
                        if label == "B3":
                            b3_mrr200 = entry.get("mrr@10", entry["mrr"])
                        else:
                            b0_mrr200 = entry.get("mrr@10", entry["mrr"])
        if b3_mrr200 is not None and b0_mrr200 is not None:
            delta = b3_mrr200 - b0_mrr200
            status = "PASS" if delta > 0 else "FAIL"
            if delta <= 0:
                manip_pass = False
            print(f"  Seed {data_seed}: B3={b3_mrr200:.4f}, B0={b0_mrr200:.4f}, delta={delta:+.4f} [{status}]")
        else:
            print(f"  Seed {data_seed}: missing step-200 eval data")
            manip_pass = False
    if not manip_pass:
        print("\n  *** MANIPULATION CHECK FAILED — fade pilot has not instantiated its premise ***")
        print("  *** Final result is NOT evidence against fading generally ***")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"seeds": args.seeds, "fade_step": args.fade_step,
                   "frozen": args.frozen, "manipulation_check": manip_pass,
                   "results": {
            str(s): {a: r for a, r in sr.items()} for s, sr in all_seed_results.items()
        }}, f, indent=2, default=str)


def main_ship():
    """Ship mode — train a single standard-KD model at scale for deployment.

    Runs only B2-style single-teacher KD + contrastive loss on more data.
    Saves the model in a format compatible with export_model.py.
    """
    parser = argparse.ArgumentParser(description="Ship — Standard KD at scale")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=500)
    parser.add_argument("--n_docs", type=int, default=10)
    parser.add_argument("--kd_weight", type=float, default=0.5,
                        help="Weight for KD loss; contrastive gets 1 - kd_weight")
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proj_seed", type=int, default=9999)
    parser.add_argument("--out_dir", default="outputs/ship_v0")
    parser.add_argument("--student", default="answerdotai/ModernBERT-base")
    parser.add_argument("--teacher", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=1000)
    args = parser.parse_args()

    import random as stdlib_random
    import numpy as np

    stdlib_random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SHIP MODE — Standard KD at scale")
    print(f"Student: {args.student}")
    print(f"Teacher: {args.teacher}")
    print(f"Data: {args.n_train} train, {args.n_eval} eval, {args.n_docs} docs/query")
    print(f"Steps: {args.steps}, LR: {args.lr}, KD weight: {args.kd_weight}")
    print("=" * 60)

    print("\nLoading MS MARCO data...")
    all_pairs = load_msmarco_pairs(
        n=args.n_train + args.n_eval, n_docs=args.n_docs, seed=args.seed,
    )
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:args.n_train + args.n_eval]
    print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval")

    print(f"\nExtracting teacher scores ({args.teacher})...")
    teacher = load_st_model(args.teacher, device=args.device)
    teacher_data = {}
    for i, pair in enumerate(train_pairs):
        identity_probe = Probe(probe_id="identity", text=pair["query"])
        scores = extract_teacher_scores(
            teacher, pair["query"], pair["documents"], [identity_probe],
        )
        teacher_data[pair["id"]] = {args.teacher: scores}
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(train_pairs)} pairs extracted")
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted {len(teacher_data)} training pairs")

    print("\nInitializing student...")
    student = ModernBERTEmbedder(
        args.student, dim=384, proj_seed=args.proj_seed,
    ).to(args.device)

    base = evaluate(student, eval_pairs)
    print(f"Baseline: Hit@1={base['hit@1']:.4f}  MRR={base['mrr']:.4f}")

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    import math
    warmup_steps = int(args.steps * args.warmup_frac)
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(args.steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(progress * math.pi))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "w")
    t0 = time.time()
    running_loss = 0.0
    best_mrr = 0.0

    config = {k: v for k, v in vars(args).items() if k != "device"}
    config["baseline"] = {k: v for k, v in base.items() if k != "per_query"}
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    gpu_thermal_guard(max_temp=85)
    print(f"\nTraining ({args.steps} steps, sampling from {len(train_pairs)} pairs)...")
    for step in range(1, args.steps + 1):
        idx = stdlib_random.randint(0, len(train_pairs) - 1)
        pair = train_pairs[idx]

        optimizer.zero_grad()

        kd_loss = compute_kd_loss(
            student, pair["query"], pair["documents"],
            teacher_data[pair["id"]][args.teacher]["identity"],
            tau=args.tau,
        )
        contrastive_loss = compute_contrastive_loss(
            student, pair["query"], pair["documents"],
            pair["gold_idx"], tau=args.tau,
        )
        loss = args.kd_weight * kd_loss + (1 - args.kd_weight) * contrastive_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 200 == 0:
            gpu_thermal_guard(max_temp=85)

        if step % 50 == 0:
            avg = running_loss / 50
            entry = {"step": step, "loss": round(avg, 6),
                     "elapsed_s": round(time.time() - t0, 1),
                     "lr": scheduler.get_last_lr()[0]}

            if step % args.eval_every == 0:
                m = evaluate(student, eval_pairs)
                entry.update({k: v for k, v in m.items() if k != "per_query"})
                tag = ""
                if m["mrr"] > best_mrr:
                    best_mrr = m["mrr"]
                    tag = " *BEST*"
                    ckpt_dir = os.path.join(args.out_dir, "best")
                    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                    student.encoder.save_pretrained(os.path.join(ckpt_dir, "encoder"))
                    student.tokenizer.save_pretrained(os.path.join(ckpt_dir, "encoder"))
                    torch.save(
                        {"weight": student.proj.weight.data.cpu(), "bias": student.proj.bias.data.cpu()},
                        os.path.join(ckpt_dir, "proj.pt"),
                    )
                    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                        json.dump({"step": step, "mrr": m["mrr"], "hit1": m["hit@1"]}, f)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}{tag}")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}")

            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

        if step % args.save_every == 0:
            gpu_thermal_guard(max_temp=85)
            ckpt_dir = os.path.join(args.out_dir, f"checkpoint-{step}")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            student.encoder.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            student.tokenizer.save_pretrained(os.path.join(ckpt_dir, "encoder"))
            torch.save(
                {"weight": student.proj.weight.data.cpu(), "bias": student.proj.bias.data.cpu()},
                os.path.join(ckpt_dir, "proj.pt"),
            )

    final = evaluate(student, eval_pairs)
    print(f"\nFINAL: Hit@1={final['hit@1']:.4f}  MRR={final['mrr']:.4f}")
    print(f"  Gain: Hit@1 {final['hit@1'] - base['hit@1']:+.4f}  MRR {final['mrr'] - base['mrr']:+.4f}")
    print(f"  Best MRR seen: {best_mrr:.4f}")

    final_dir = os.path.join(args.out_dir, "final")
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), os.path.join(final_dir, "model.pt"))
    student.encoder.save_pretrained(os.path.join(final_dir, "encoder"))
    student.tokenizer.save_pretrained(os.path.join(final_dir, "encoder"))
    torch.save(
        {"weight": student.proj.weight.data.cpu(), "bias": student.proj.bias.data.cpu()},
        os.path.join(final_dir, "proj.pt"),
    )
    with open(os.path.join(final_dir, "config.json"), "w") as f:
        json.dump({
            "student": args.student, "teacher": args.teacher, "dim": 384,
            "proj_seed": args.proj_seed, "steps": args.steps, "lr": args.lr,
        }, f, indent=2)

    result = {
        "baseline": {k: v for k, v in base.items() if k != "per_query"},
        "final": {k: v for k, v in final.items() if k != "per_query"},
        "best_mrr": best_mrr,
        "gain_mrr": final["mrr"] - base["mrr"],
        "gain_hit1": final["hit@1"] - base["hit@1"],
        "config": config,
    }
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log_f.close()
    print(f"\nModel saved to {final_dir}")
    print(f"To export: python code/export_model.py --checkpoint {final_dir}")


if __name__ == "__main__":
    import sys
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1]
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        if mode == "e1.5":
            main_e15()
        elif mode == "fade":
            main_fade()
        elif mode == "ship":
            main_ship()
        else:
            main()
    else:
        main()
