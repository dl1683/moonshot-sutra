"""R1: Frozen Readout Experiment — is the LM decoder the bottleneck?

Designed by Codex R57. Tests whether trained students already contain
the answer in their hidden state by training small readout heads over
frozen representations.

Three readout types:
  1. LinearWordReadout: linear projection from answer_state to word logits
  2. BilinearEnergyReadout: bilinear scoring h_answer vs candidate embeddings
  3. CalibratedLMReadout: temperature + bias correction on existing LM scores

Success gates:
  1. Frozen Linear/Bilinear readout >= 95% MCQ
  2. >= 95% heldout CF direction accuracy
  3. <= 1000 labeled examples to exceed 90% MCQ
  4. Shuffled-label readout stays near chance
  5. CalibratedLMReadout does NOT close most of the gap
"""

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_opgeom_og1b import (
    ALL_WORDS, ANSWER_POS, ATTR_POOLS, ATTRS, BYTE_VOCAB, DEVICE,
    NAMES, PATCH_SIZE, VOCAB_SIZE, WORD2ID, WORD_BYTES,
    generate_binding_example, make_same_attr_candidates,
    tokens_to_bytes_seq, train_teacher, ToyTeacher,
    apply_counterfactual_transform_v2,
    N_STEPS, LR_PEAK, LR_MIN, LR_WARMUP_STEPS, GRAD_CLIP,
    WARMUP_STEPS, EVAL_EXAMPLES, compute_ce_loss, lr_at_step,
)
from toy_opgeom_probes import ToyByteStudentProbed, extract_context_states

RESULTS_DIR = Path("C:/sutra_fast/r1_readout")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

READOUT_STEPS = 2000
READOUT_LR = 1e-3
READOUT_WD = 1e-3


# --- Readout Heads ---

class LinearWordReadout(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, h_answer, candidate_ids):
        logits = self.proj(h_answer)
        return logits.gather(1, candidate_ids)


class BilinearEnergyReadout(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.h_proj = nn.Linear(d_model, d_model, bias=False)
        self.word_emb = nn.Embedding(vocab_size, d_model)
        self.bias = nn.Embedding(vocab_size, 1)

    def forward(self, h_answer, candidate_ids):
        h = self.h_proj(h_answer)
        c = self.word_emb(candidate_ids)
        b = self.bias(candidate_ids).squeeze(-1)
        return (h.unsqueeze(1) * c).sum(-1) / math.sqrt(h.shape[-1]) + b


class CalibratedLMReadout(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Embedding(vocab_size, 1)

    def forward(self, lm_scores, candidate_ids):
        b = self.bias(candidate_ids).squeeze(-1)
        return lm_scores / self.temperature.clamp_min(0.05) + b


# --- Data Collection ---

MAX_CANDIDATES = 8  # max pool size (COLORS/ROOMS=8, ACTIONS=4)
PAD_WORD_ID = 0  # padding token for short candidate lists


def collect_answer_state_dataset(student, n_examples, seed, split="train"):
    """Collect (answer_state, candidates, gold_idx, lm_scores) tuples.

    Pads candidate lists to MAX_CANDIDATES with PAD_WORD_ID and -inf scores.
    """
    rng = random.Random(seed)
    student.eval()

    h_states = []
    all_candidates = []
    all_gold_idx = []
    all_lm_scores = []
    all_n_cands = []
    all_meta = []

    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)

        states = extract_context_states(student, tokens + [correct])
        h_answer = states["answer_state"]

        with torch.no_grad():
            lm_scores = student.score_candidates_batch(tokens + [correct], candidates)

        candidate_ids = [WORD2ID[c] for c in candidates]
        n_cands = len(candidate_ids)

        # Pad to MAX_CANDIDATES
        while len(candidate_ids) < MAX_CANDIDATES:
            candidate_ids.append(PAD_WORD_ID)
        lm_padded = torch.full((MAX_CANDIDATES,), -1e9, device=DEVICE)
        lm_padded[:n_cands] = lm_scores

        h_states.append(h_answer)
        all_candidates.append(torch.tensor(candidate_ids, device=DEVICE))
        all_gold_idx.append(gold_idx)
        all_lm_scores.append(lm_padded)
        all_n_cands.append(n_cands)
        all_meta.append({"correct": correct, "query_attr": meta["query_attr"],
                         "candidates": candidates, "n_cands": n_cands})

    return {
        "h_answer": torch.stack(h_states),
        "candidate_ids": torch.stack(all_candidates),
        "gold_idx": torch.tensor(all_gold_idx, dtype=torch.long, device=DEVICE),
        "lm_scores": torch.stack(all_lm_scores),
        "n_cands": torch.tensor(all_n_cands, dtype=torch.long, device=DEVICE),
        "meta": all_meta,
    }


def collect_cf_direction_dataset(student, n_examples, seed):
    """Generator yielding (h_answer, candidate_ids, gold_idx, candidates) for CF direction eval."""
    rng = random.Random(seed)
    student.eval()

    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        cf_result = apply_counterfactual_transform_v2(
            tokens, meta, rng, "change_query_slot")
        if cf_result.is_noop:
            continue

        candidates, gold_idx = make_same_attr_candidates(
            cf_result.correct, cf_result.query_attr, rng)

        states = extract_context_states(student, cf_result.tokens + [cf_result.correct])
        h_answer = states["answer_state"]

        candidate_ids = torch.tensor([WORD2ID[c] for c in candidates], device=DEVICE)

        yield h_answer, candidate_ids, gold_idx, candidates


# --- Training ---

def train_frozen_readout(student, readout, train_data, val_data,
                         steps=READOUT_STEPS, lr=READOUT_LR,
                         weight_decay=READOUT_WD, readout_type="linear"):
    """Train readout head with frozen student. Returns metrics dict."""
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = train_data["h_answer"].shape[0]
    batch_size = min(256, n_train)
    best_val_acc = 0.0
    best_state = None

    for step in range(steps):
        readout.train()
        idx = torch.randint(0, n_train, (batch_size,))

        h = train_data["h_answer"][idx]
        cands = train_data["candidate_ids"][idx]
        gold = train_data["gold_idx"][idx]
        n_c = train_data["n_cands"][idx]

        if readout_type == "calibrated":
            lm_s = train_data["lm_scores"][idx]
            scores = readout(lm_s, cands)
        else:
            scores = readout(h, cands)

        # Mask padded candidates
        mask = torch.arange(MAX_CANDIDATES, device=DEVICE).unsqueeze(0) >= n_c.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)

        loss = F.cross_entropy(scores, gold)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 200 == 0:
            val_acc = evaluate_readout_mcq(student, readout, val_data, readout_type)["mcq"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(readout.state_dict())

    if best_state is not None:
        readout.load_state_dict(best_state)
    final_metrics = evaluate_readout_mcq(student, readout, val_data, readout_type)
    final_metrics["best_val_acc"] = best_val_acc
    return final_metrics


# --- Evaluation ---

def evaluate_readout_mcq(student, readout, data, readout_type="linear"):
    """Evaluate readout on MCQ accuracy."""
    readout.eval()
    with torch.no_grad():
        h = data["h_answer"]
        cands = data["candidate_ids"]
        gold = data["gold_idx"]
        n_c = data["n_cands"]

        if readout_type == "calibrated":
            scores = readout(data["lm_scores"], cands)
        else:
            scores = readout(h, cands)

        # Mask padded candidates
        mask = torch.arange(MAX_CANDIDATES, device=DEVICE).unsqueeze(0) >= n_c.unsqueeze(1)
        scores = scores.masked_fill(mask, -1e9)

        preds = scores.argmax(dim=-1)
        mcq = (preds == gold).float().mean().item()

    return {"mcq": mcq}


def evaluate_cf_direction(student, readout, readout_type, n_examples=500, seed=777777):
    """Evaluate on heldout counterfactual direction (unseen transforms)."""
    readout.eval()
    correct = 0
    total = 0

    for h_answer, candidate_ids, gold_idx, candidates in collect_cf_direction_dataset(
            student, n_examples, seed):
        with torch.no_grad():
            n_cands = len(candidates)
            # Pad to MAX_CANDIDATES
            padded_ids = torch.full((MAX_CANDIDATES,), PAD_WORD_ID, device=DEVICE,
                                    dtype=candidate_ids.dtype)
            padded_ids[:n_cands] = candidate_ids

            cands = padded_ids.unsqueeze(0)
            h = h_answer.unsqueeze(0)

            if readout_type == "calibrated":
                lm_scores_raw = student.score_candidates_batch(
                    [], candidates)
                lm_padded = torch.full((MAX_CANDIDATES,), -1e9, device=DEVICE)
                lm_padded[:n_cands] = lm_scores_raw
                scores = readout(lm_padded.unsqueeze(0), cands)
            else:
                scores = readout(h, cands)

            # Mask padded
            scores[0, n_cands:] = -1e9
            pred = scores.argmax(dim=-1).item()
            if pred == gold_idx:
                correct += 1
            total += 1

    return correct / max(total, 1)


# --- Sample Efficiency ---

def sample_efficiency_curve(student, d_model, readout_type, full_data, val_data,
                            sizes=[100, 250, 500, 1000, 2000, 4000]):
    """Train fresh readout with increasing data sizes, report MCQ."""
    results = []
    for n in sizes:
        if n > full_data["h_answer"].shape[0]:
            break
        subset = {k: v[:n] if isinstance(v, torch.Tensor) else v[:n]
                  for k, v in full_data.items()}
        readout = LinearWordReadout(d_model, VOCAB_SIZE).to(DEVICE)
        metrics = train_frozen_readout(student, readout, subset, val_data,
                                       readout_type=readout_type)
        results.append({"n": n, "mcq": metrics["mcq"]})
    return results


# --- Shuffled Label Control ---

def train_shuffled_label_control(student, readout, train_data, val_data, readout_type):
    """Train with shuffled labels — should stay near chance."""
    shuffled_data = {k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in train_data.items()}
    perm = torch.randperm(shuffled_data["gold_idx"].shape[0])
    shuffled_data["gold_idx"] = shuffled_data["gold_idx"][perm]

    metrics = train_frozen_readout(student, readout, shuffled_data, val_data,
                                   readout_type=readout_type)
    return metrics["mcq"]


# --- Main Experiment ---

TARGETS = [
    ("A_ce", 0),
    ("A_ce", 1),
    ("B_rank", 0),
    ("B_rank", 1),
    ("B_rank", 2),
    ("D_aug_cf", 3),
    ("A_cf_ce", 0),
]


def train_student_for_readout(teacher, variant, seed, data_seed=424242):
    """Train a student (same as probes) and return it frozen."""
    from toy_opgeom_og1b import (
        compute_ranking_loss_v2, compute_invariance_loss_v2,
        make_relational_candidates, LAMBDA_RANK, LAMBDA_INV,
        LAMBDA_CF_AUG, LAMBDA_CF_REL, ANSWER_CE_WEIGHT,
        apply_preserving_transform, T_PRESERVE, T_CF,
    )

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(data_seed)

    student = ToyByteStudentProbed().to(DEVICE)
    opt = torch.optim.AdamW(student.parameters(), lr=LR_PEAK, weight_decay=1e-2)

    for step in range(1, N_STEPS + 1):
        lr = lr_at_step(step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        tokens, correct, distractors, meta = generate_binding_example(rng)
        L_ce = compute_ce_loss(student, tokens + [correct])

        loss = L_ce

        if variant in ("B_rank", "C_inv_fixed", "D_aug_cf", "D_rel_full", "F_rand_inv"):
            candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)
            L_rank, _ = compute_ranking_loss_v2(student, tokens + [correct], candidates, gold_idx)
            loss = loss + LAMBDA_RANK * L_rank

        if variant in ("A_cf_ce", "D_aug_cf"):
            cf_result = apply_counterfactual_transform_v2(
                tokens, meta, rng, rng.choice(T_CF))
            L_cf_ce = compute_ce_loss(student, cf_result.tokens + [cf_result.correct])
            loss = loss + LAMBDA_CF_AUG * L_cf_ce

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        opt.step()

    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


def run_single_target(teacher, variant, seed, data_seed=424242):
    """Run full readout experiment on one variant-seed."""
    print(f"\n{'='*60}")
    print(f"  R1 Readout: [{variant}] seed={seed}")
    print(f"{'='*60}")

    student = train_student_for_readout(teacher, variant, seed, data_seed)

    # Collect data
    print("  Collecting train data (4000 examples)...")
    train_data = collect_answer_state_dataset(student, 4000, seed=data_seed + 1)
    print("  Collecting val data (1000 examples)...")
    val_data = collect_answer_state_dataset(student, 1000, seed=data_seed + 2)

    d_model = student.d_model
    results = {"variant": variant, "seed": seed}

    # 1. LinearWordReadout
    print("  Training LinearWordReadout...")
    linear_head = LinearWordReadout(d_model, VOCAB_SIZE).to(DEVICE)
    linear_metrics = train_frozen_readout(student, linear_head, train_data, val_data,
                                          readout_type="linear")
    results["linear_mcq"] = linear_metrics["mcq"]
    print(f"    Linear MCQ: {linear_metrics['mcq']*100:.1f}%")

    # 2. BilinearEnergyReadout
    print("  Training BilinearEnergyReadout...")
    bilinear_head = BilinearEnergyReadout(d_model, VOCAB_SIZE).to(DEVICE)
    bilinear_metrics = train_frozen_readout(student, bilinear_head, train_data, val_data,
                                            readout_type="bilinear")
    results["bilinear_mcq"] = bilinear_metrics["mcq"]
    print(f"    Bilinear MCQ: {bilinear_metrics['mcq']*100:.1f}%")

    # 3. CalibratedLMReadout
    print("  Training CalibratedLMReadout...")
    calibrated_head = CalibratedLMReadout(VOCAB_SIZE).to(DEVICE)
    calibrated_metrics = train_frozen_readout(student, calibrated_head, train_data, val_data,
                                              readout_type="calibrated")
    results["calibrated_mcq"] = calibrated_metrics["mcq"]
    print(f"    Calibrated MCQ: {calibrated_metrics['mcq']*100:.1f}%")

    # 4. LM baseline (no readout, just existing scores — already masked via -1e9 padding)
    with torch.no_grad():
        lm_preds = val_data["lm_scores"].argmax(dim=-1)
        lm_mcq = (lm_preds == val_data["gold_idx"]).float().mean().item()
    results["lm_baseline_mcq"] = lm_mcq
    print(f"    LM baseline MCQ: {lm_mcq*100:.1f}%")

    # 5. CF direction eval (linear head)
    print("  Evaluating CF direction (linear)...")
    cf_acc = evaluate_cf_direction(student, linear_head, "linear")
    results["linear_cf_direction"] = cf_acc
    print(f"    Linear CF direction: {cf_acc*100:.1f}%")

    # 6. CF direction eval (bilinear head)
    print("  Evaluating CF direction (bilinear)...")
    cf_acc_bi = evaluate_cf_direction(student, bilinear_head, "bilinear")
    results["bilinear_cf_direction"] = cf_acc_bi
    print(f"    Bilinear CF direction: {cf_acc_bi*100:.1f}%")

    # 7. Sample efficiency (linear)
    print("  Sample efficiency curve (linear)...")
    eff_curve = sample_efficiency_curve(student, d_model, "linear",
                                        train_data, val_data,
                                        sizes=[100, 250, 500, 1000, 2000, 4000])
    results["sample_efficiency_linear"] = eff_curve
    for pt in eff_curve:
        print(f"    n={pt['n']}: {pt['mcq']*100:.1f}%")

    # 8. Shuffled label control (linear)
    print("  Shuffled label control (linear)...")
    shuffled_mcq = train_shuffled_label_control(
        student, LinearWordReadout(d_model, VOCAB_SIZE).to(DEVICE),
        train_data, val_data, "linear")
    results["shuffled_label_mcq"] = shuffled_mcq
    print(f"    Shuffled label MCQ: {shuffled_mcq*100:.1f}% (chance ~12.5%)")

    # Summary
    print(f"\n  SUMMARY [{variant}] seed={seed}:")
    print(f"    LM baseline:    {lm_mcq*100:.1f}%")
    print(f"    Linear readout: {linear_metrics['mcq']*100:.1f}%")
    print(f"    Bilinear:       {bilinear_metrics['mcq']*100:.1f}%")
    print(f"    Calibrated LM:  {calibrated_metrics['mcq']*100:.1f}%")
    print(f"    Shuffled ctrl:  {shuffled_mcq*100:.1f}%")
    print(f"    Linear CF dir:  {cf_acc*100:.1f}%")
    print(f"    Bilinear CF:    {cf_acc_bi*100:.1f}%")

    # Gate checks
    gates = {
        "gate1_linear_95": linear_metrics["mcq"] >= 0.95,
        "gate1_bilinear_95": bilinear_metrics["mcq"] >= 0.95,
        "gate2_cf_direction_95": cf_acc >= 0.95,
        "gate3_1000_samples_90": any(p["mcq"] >= 0.90 for p in eff_curve if p["n"] <= 1000),
        "gate4_shuffled_chance": shuffled_mcq < 0.25,
        "gate5_calibrated_no_close": calibrated_metrics["mcq"] < linear_metrics["mcq"] - 0.10,
    }
    results["gates"] = gates
    print(f"\n  Gates: {sum(gates.values())}/{len(gates)} passed")
    for g, v in gates.items():
        print(f"    {'PASS' if v else 'FAIL'}: {g}")

    return results


def main():
    output_file = RESULTS_DIR / "r1_results.json"
    print("R1: Frozen Readout Experiment")
    print(f"Device: {DEVICE}")
    print(f"Targets: {len(TARGETS)} variant-seed combinations")
    print(f"Output: {output_file}")

    print("\n=== Training Teacher ===")
    teacher = ToyTeacher().to(DEVICE)
    train_teacher(teacher)

    all_results = []
    for variant, seed in TARGETS:
        result = run_single_target(teacher, variant, seed)
        all_results.append(result)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    print("\n" + "="*60)
    print("  FINAL R1 SUMMARY")
    print("="*60)
    print(f"{'Variant':<12} {'Seed':<5} {'LM%':<7} {'Linear%':<9} {'Bilinear%':<11} {'Calib%':<8} {'CF_dir%':<8}")
    print("-"*60)
    for r in all_results:
        print(f"{r['variant']:<12} {r['seed']:<5} "
              f"{r['lm_baseline_mcq']*100:<7.1f} "
              f"{r['linear_mcq']*100:<9.1f} "
              f"{r['bilinear_mcq']*100:<11.1f} "
              f"{r['calibrated_mcq']*100:<8.1f} "
              f"{r.get('linear_cf_direction', 0)*100:<8.1f}")

    all_gates_pass = all(
        all(r["gates"].values()) for r in all_results
    )
    print(f"\nALL GATES PASS: {all_gates_pass}")
    if all_gates_pass:
        print("CONCLUSION: LM decoder is the bottleneck. Representation is sufficient.")
        print("ACTION: Retire OG ranking losses. Move to energy head on real Sutra.")
    else:
        failed = [(r["variant"], r["seed"], [g for g, v in r["gates"].items() if not v])
                  for r in all_results if not all(r["gates"].values())]
        print("PARTIAL FAILURES:")
        for v, s, gs in failed:
            print(f"  [{v}] seed={s}: {gs}")


if __name__ == "__main__":
    main()
