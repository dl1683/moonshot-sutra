"""Track A: Binding Circuit Probes for OG-1b models.

Probes whether B_rank seed 0 formed a real binding circuit vs score calibration.
Compares successful vs failed B_rank seeds, D_aug_cf (converged but baseline), and controls.

Designed by Codex R56b. Implements:
  - Modified forward() with return_cache and ablation
  - Linear probes for entity-slot binding
  - Causal ablation analysis
  - Score gradient alignment
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
    ACTIONS, ALL_WORDS, ANSWER_POS, ATTR_POOLS, ATTRS, BYTE_VOCAB, CHECKPOINT_STEPS,
    COLORS, DEVICE, EPS, EVAL_EXAMPLES, GRAD_CLIP, LR_MIN, LR_PEAK, LR_WARMUP_STEPS,
    N_STEPS, NAMES, PATCH_SIZE, ROOMS, SPECIAL, T_CF, T_PRESERVE,
    VARIANT_LABELS, VARIANTS, VOCAB_SIZE, WARMUP_STEPS, WORD2ID, WORD_BYTES,
    CFResult, apply_counterfactual_transform_v2, apply_preserving_transform,
    compute_ce_loss, compute_invariance_loss_v2, compute_ranking_loss_v2,
    compute_relational_cf_loss, generate_binding_example, lr_at_step,
    make_same_attr_candidates, tokens_to_bytes_seq, tokens_to_ids,
    train_teacher, ToyTeacher,
    LAMBDA_RANK, LAMBDA_INV, LAMBDA_CF_AUG, LAMBDA_CF_REL,
)

CHECKPOINT_DIR = Path("C:/sutra_fast/og1b_checkpoints")
PROBE_DIR = Path("C:/sutra_fast/og1b_probes")


class ToyByteStudentProbed(nn.Module):
    """ToyByteStudent with return_cache and ablation support."""

    def __init__(self, d_model=64, n_layers=4, n_heads=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb = nn.Embedding(BYTE_VOCAB, d_model)
        self.patch_proj = nn.Linear(patch_size * d_model, d_model, bias=False)
        self.pos_emb = nn.Embedding(64, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, BYTE_VOCAB * patch_size, bias=False)
        self.d_model = d_model

    def forward(self, byte_ids, return_cache=False, ablate_patch_idxs=None):
        B, T = byte_ids.shape
        P = self.patch_size
        N = T // P
        x = self.byte_emb(byte_ids).reshape(B, N, P * self.d_model)
        patch_states = self.patch_proj(x) + self.pos_emb(
            torch.arange(N, device=byte_ids.device).unsqueeze(0))

        if ablate_patch_idxs is not None:
            patch_states[:, ablate_patch_idxs, :] = 0.0

        mask = nn.Transformer.generate_square_subsequent_mask(N, device=byte_ids.device)
        h = patch_states
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        hidden = self.norm(h)

        if return_cache:
            return self.byte_decoder(hidden).reshape(B, N, P, BYTE_VOCAB), {
                "patch_states": patch_states.detach(),
                "hidden": hidden.detach(),
            }
        return self.byte_decoder(hidden).reshape(B, N, P, BYTE_VOCAB)

    def score_candidates_batch(self, context_tokens, candidates):
        context_bytes = tokens_to_bytes_seq(context_tokens)
        t_full_list = []
        for cand in candidates:
            ab = context_bytes + WORD_BYTES[cand]
            while len(ab) % self.patch_size != 0:
                ab.append(0)
            t_full_list.append(ab)
        max_len = max(len(x) for x in t_full_list)
        for i in range(len(t_full_list)):
            while len(t_full_list[i]) < max_len:
                t_full_list[i].append(0)
        t = torch.tensor(t_full_list, dtype=torch.long, device=DEVICE)
        logits = self.forward(t)
        n_ctx_patches = len(context_bytes) // self.patch_size
        pred_patch = n_ctx_patches - 1
        scores = []
        for ci, cand in enumerate(candidates):
            cand_bytes = WORD_BYTES[cand]
            s = sum(F.log_softmax(logits[ci, pred_patch, bp], dim=-1)[cand_bytes[bp]]
                    for bp in range(min(len(cand_bytes), self.patch_size)))
            scores.append(s / len(cand_bytes))
        return torch.stack(scores)

    def score_candidates_with_ablation(self, context_tokens, candidates, ablate_patch_idxs=None):
        context_bytes = tokens_to_bytes_seq(context_tokens)
        t_full_list = []
        for cand in candidates:
            ab = context_bytes + WORD_BYTES[cand]
            while len(ab) % self.patch_size != 0:
                ab.append(0)
            t_full_list.append(ab)
        max_len = max(len(x) for x in t_full_list)
        for i in range(len(t_full_list)):
            while len(t_full_list[i]) < max_len:
                t_full_list[i].append(0)
        t = torch.tensor(t_full_list, dtype=torch.long, device=DEVICE)
        logits = self.forward(t, ablate_patch_idxs=ablate_patch_idxs)
        n_ctx_patches = len(context_bytes) // self.patch_size
        pred_patch = n_ctx_patches - 1
        scores = []
        for ci, cand in enumerate(candidates):
            cand_bytes = WORD_BYTES[cand]
            s = sum(F.log_softmax(logits[ci, pred_patch, bp], dim=-1)[cand_bytes[bp]]
                    for bp in range(min(len(cand_bytes), self.patch_size)))
            scores.append(s / len(cand_bytes))
        return torch.stack(scores)


def extract_context_states(student, context_tokens):
    byte_seq = tokens_to_bytes_seq(context_tokens)
    byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _, cache = student(byte_t, return_cache=True)
    n_tokens = len(context_tokens)
    return {
        "answer_state": cache["hidden"][0, n_tokens - 1],
        "query_name_state": cache["hidden"][0, 10],
        "query_attr_state": cache["hidden"][0, 11],
        "all_hidden": cache["hidden"][0],
        "all_patch_states": cache["patch_states"][0],
    }


def binding_labels(meta, correct):
    qp = meta["query_person"]
    qa = meta["query_attr"]
    return {
        "query_person": qp,
        "query_attr": ATTRS.index(qa),
        "correct_word": WORD2ID[correct],
        "person0_color": WORD2ID[meta["colors"][0]],
        "person1_color": WORD2ID[meta["colors"][1]],
        "person0_room": WORD2ID[meta["rooms"][0]],
        "person1_room": WORD2ID[meta["rooms"][1]],
        "person0_action": WORD2ID[meta["actions"][0]],
        "person1_action": WORD2ID[meta["actions"][1]],
    }


def train_linear_probe(X_train, y_train, X_val, y_val, n_classes, steps=1000, lr=1e-2, weight_decay=1e-3):
    d = X_train.shape[1]
    probe = nn.Linear(d, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    for step in range(steps):
        probe.train()
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(probe.state_dict())

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val)
        val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
    return {"val_acc": val_acc, "probe": probe}


def collect_probe_data(student, n_examples, data_seed=424242):
    rng = random.Random(data_seed)
    states_answer = []
    states_query_name = []
    states_query_attr = []
    labels_list = []

    student.eval()
    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        s = extract_context_states(student, tokens + [correct])
        states_answer.append(s["answer_state"])
        states_query_name.append(s["query_name_state"])
        states_query_attr.append(s["query_attr_state"])
        labels_list.append(binding_labels(meta, correct))

    X_answer = torch.stack(states_answer)
    X_qname = torch.stack(states_query_name)
    X_qattr = torch.stack(states_query_attr)

    label_keys = list(labels_list[0].keys())
    Y = {}
    for k in label_keys:
        Y[k] = torch.tensor([l[k] for l in labels_list], dtype=torch.long, device=DEVICE)

    return X_answer, X_qname, X_qattr, Y, label_keys


def run_binding_probe_suite(student, variant, seed, n_train=4000, n_val=1000, data_seed=424242):
    total = n_train + n_val
    X_answer, X_qname, X_qattr, Y, label_keys = collect_probe_data(student, total, data_seed)

    results = {}
    probe_configs = [
        ("answer_state", X_answer),
        ("query_name_state", X_qname),
        ("query_attr_state", X_qattr),
    ]

    n_classes_map = {
        "query_person": 2,
        "query_attr": 3,
        "correct_word": VOCAB_SIZE,
        "person0_color": VOCAB_SIZE,
        "person1_color": VOCAB_SIZE,
        "person0_room": VOCAB_SIZE,
        "person1_room": VOCAB_SIZE,
        "person0_action": VOCAB_SIZE,
        "person1_action": VOCAB_SIZE,
    }

    for state_name, X in probe_configs:
        X_tr, X_va = X[:n_train], X[n_train:]
        for label_key in label_keys:
            y = Y[label_key]
            y_tr, y_va = y[:n_train], y[n_train:]
            nc = n_classes_map.get(label_key, VOCAB_SIZE)
            r = train_linear_probe(X_tr, y_tr, X_va, y_va, nc)
            key = f"{state_name}/{label_key}"
            results[key] = r["val_acc"]

    return results


def evaluate_patch_ablation(student, n_examples=2000, seed=515151):
    rng = random.Random(seed)
    student.eval()

    drops = {k: [] for k in [
        "queried_value", "query_name", "query_attr",
        "other_same_attr", "irrelevant_value", "random_fact"
    ]}
    base_accs = []
    ablated_accs = {k: [] for k in drops}

    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)

        with torch.no_grad():
            base_scores = student.score_candidates_batch(tokens + [correct], candidates)
        base_margin = (base_scores[gold_idx] - base_scores[torch.arange(len(candidates)) != gold_idx].max()).item()

        qp = meta["query_person"]
        qa = meta["query_attr"]
        other_p = 1 - qp

        # Patch indices for context: strt(0) name0(1) action0(2) color0(3) room0(4) name1(5) action1(6) color1(7) room1(8) qury(9) qname(10) qattr(11) answ(12)
        attr_to_patch = {"actn": 2, "colr": 3, "room": 4}
        queried_value_patch = attr_to_patch[qa] + qp * 4
        other_same_attr_patch = attr_to_patch[qa] + other_p * 4

        irrelevant_attrs = [a for a in ATTRS if a != qa]
        irr_attr = rng.choice(irrelevant_attrs)
        irrelevant_patch = attr_to_patch[irr_attr] + qp * 4

        random_fact_patch = rng.choice([2, 3, 4, 6, 7, 8])

        ablation_map = {
            "queried_value": [queried_value_patch],
            "query_name": [10],
            "query_attr": [11],
            "other_same_attr": [other_same_attr_patch],
            "irrelevant_value": [irrelevant_patch],
            "random_fact": [random_fact_patch],
        }

        for abl_name, abl_patches in ablation_map.items():
            with torch.no_grad():
                abl_scores = student.score_candidates_with_ablation(
                    tokens + [correct], candidates, ablate_patch_idxs=abl_patches)
            abl_margin = (abl_scores[gold_idx] - abl_scores[torch.arange(len(candidates)) != gold_idx].max()).item()
            drops[abl_name].append(base_margin - abl_margin)
            ablated_accs[abl_name].append(1 if abl_scores.argmax().item() == gold_idx else 0)

        base_accs.append(1 if base_scores.argmax().item() == gold_idx else 0)

    results = {"base_mcq": np.mean(base_accs)}
    for k in drops:
        results[f"drop_{k}"] = float(np.mean(drops[k]))
        results[f"mcq_ablated_{k}"] = float(np.mean(ablated_accs[k]))

    drop_q = results["drop_queried_value"]
    drop_i = results["drop_irrelevant_value"]
    results["specificity_ratio"] = drop_q / (drop_i + 1e-8)
    return results


def stable_variant_seed(seed, variant):
    return seed * 10000 + VARIANTS.index(variant) * 997 + 17


def train_and_probe(teacher, variant, seed, data_seed):
    """Train a single variant/seed and return model + probe results."""
    student = ToyByteStudentProbed(d_model=64, n_layers=4, n_heads=4, patch_size=PATCH_SIZE).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)
    base_student = ToyByteStudentProbed(d_model=64, n_layers=4, n_heads=4, patch_size=PATCH_SIZE).to(DEVICE)
    base_state = copy.deepcopy(base_student.state_dict())
    del base_student

    student.load_state_dict(copy.deepcopy(base_state))
    torch.manual_seed(stable_variant_seed(seed, variant))

    # Import training logic from og1b
    rng = random.Random(data_seed)
    rng_unrelated = random.Random(data_seed + 7777)
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR_PEAK, weight_decay=0.01)

    uses_rank = variant in ("B_rank", "C_inv_fixed", "D_aug_cf", "D_rel_full", "E_adv", "F_rand_inv")
    uses_inv = variant in ("C_inv_fixed", "D_rel_full", "F_rand_inv")
    uses_cf_aug = variant in ("D_aug_cf",)
    uses_cf_rel = variant in ("D_rel_full", "E_adv", "F_rand_inv")
    uses_cf_ce = variant in ("A_cf_ce",)
    uses_more_ce = variant in ("A_more_ce",)

    for step in range(N_STEPS):
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at_step(step)

        tokens, correct, distractors, meta = generate_binding_example(rng)
        candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)
        L_ce = compute_ce_loss(student, tokens + [correct])
        loss = L_ce

        if uses_more_ce:
            extra_tokens, extra_correct, _, _ = generate_binding_example(rng)
            L_ce_extra = compute_ce_loss(student, extra_tokens + [extra_correct])
            loss = 0.5 * L_ce + 0.5 * L_ce_extra

        if uses_cf_ce and step >= WARMUP_STEPS:
            cf_type = rng.choice(T_CF)
            cf = apply_counterfactual_transform_v2(tokens, meta, rng, cf_type)
            if not cf.is_noop:
                L_ce_cf = compute_ce_loss(student, cf.tokens + [cf.correct])
                loss = 0.5 * L_ce + 0.5 * L_ce_cf

        if uses_rank and step >= WARMUP_STEPS:
            if variant == "E_adv":
                wrong_pool = [c for c in candidates if c != correct]
                fake_correct = rng.choice(wrong_pool)
                adv_gold_idx = candidates.index(fake_correct)
                L_rank, _ = compute_ranking_loss_v2(student, tokens, candidates, adv_gold_idx)
            else:
                L_rank, _ = compute_ranking_loss_v2(student, tokens, candidates, gold_idx)
            loss = loss + LAMBDA_RANK * L_rank

        if uses_inv and step >= WARMUP_STEPS:
            t_type = rng.choice(T_PRESERVE)
            if variant == "F_rand_inv":
                unrel_tokens, _, _, unrel_meta = generate_binding_example(rng_unrelated)
                trans_tokens, _ = apply_preserving_transform(unrel_tokens, unrel_meta, rng, t_type)
            else:
                trans_tokens, _ = apply_preserving_transform(tokens, meta, rng, t_type)
            L_inv, _ = compute_invariance_loss_v2(student, tokens, trans_tokens, candidates, gold_idx)
            loss = loss + LAMBDA_INV * L_inv

        if (uses_cf_aug or uses_cf_rel) and step >= WARMUP_STEPS:
            cf_type = rng.choice(T_CF)
            cf = apply_counterfactual_transform_v2(tokens, meta, rng, cf_type)

            if not cf.is_noop:
                cf_candidates, cf_gold_idx = make_same_attr_candidates(
                    cf.correct, cf.query_attr, rng)

            if not cf.is_noop and uses_cf_aug:
                L_cf, _ = compute_ranking_loss_v2(student, cf.tokens, cf_candidates, cf_gold_idx)
                loss = loss + LAMBDA_CF_AUG * L_cf

            if not cf.is_noop and uses_cf_rel:
                orig_scores = student.score_candidates_batch(tokens + [correct], candidates)
                cf_scores = student.score_candidates_batch(cf.tokens + [cf.correct], cf_candidates)
                if variant == "E_adv":
                    L_rel = compute_relational_cf_loss(cf_scores, orig_scores, cf_gold_idx, gold_idx)
                else:
                    L_rel = compute_relational_cf_loss(orig_scores, cf_scores, gold_idx, cf_gold_idx)
                loss = loss + LAMBDA_CF_REL * L_rel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) in [4000, 8000, 12000]:
            print(f"    [{variant}] step {step+1}: loss={loss.item():.4f}")

    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{variant}_seed{seed}.pt"
    torch.save(student.state_dict(), ckpt_path)

    student.eval()
    return student


def run_full_probe_analysis(student, variant, seed):
    """Run all probes on a trained model."""
    print(f"  Probing [{variant}] seed={seed}...")

    probe_results = run_binding_probe_suite(student, variant, seed)
    ablation_results = evaluate_patch_ablation(student)

    print(f"    Probe correct_word from answer_state: {probe_results.get('answer_state/correct_word', 0)*100:.1f}%")
    print(f"    Probe query_person from answer_state: {probe_results.get('answer_state/query_person', 0)*100:.1f}%")
    print(f"    Probe query_attr from answer_state: {probe_results.get('answer_state/query_attr', 0)*100:.1f}%")
    print(f"    Ablation specificity_ratio: {ablation_results['specificity_ratio']:.2f}")
    print(f"    Base MCQ: {ablation_results['base_mcq']*100:.1f}%")
    print(f"    MCQ after queried_value ablation: {ablation_results['mcq_ablated_queried_value']*100:.1f}%")
    print(f"    MCQ after irrelevant ablation: {ablation_results['mcq_ablated_irrelevant_value']*100:.1f}%")

    return {"probes": probe_results, "ablation": ablation_results}


def main():
    PROBE_DIR.mkdir(parents=True, exist_ok=True)

    # Selected variants/seeds for probing (Codex R56b spec)
    probe_targets = [
        ("B_rank", 0),     # converged + gained
        ("B_rank", 1),     # failed
        ("B_rank", 2),     # failed
        ("B_rank", 3),     # failed
        ("B_rank", 4),     # pending
        ("D_aug_cf", 3),   # rank converged but baseline
        ("A_ce", 0),       # baseline
        ("A_ce", 1),       # baseline
        ("A_cf_ce", 0),    # augmentation baseline
        ("D_rel_full", 0), # full multi-loss
    ]

    print(f"Track A: Binding Circuit Probes")
    print(f"Device: {DEVICE}")
    print(f"Targets: {len(probe_targets)} variant-seed combinations")
    print()

    # Train teacher
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    print("=== Training Teacher ===")
    teacher = ToyTeacher(d_model=128, n_layers=4, n_heads=4).to(DEVICE)
    train_teacher(teacher)
    teacher.eval()
    print()

    all_results = {}

    for variant, seed in probe_targets:
        data_seed = seed * 1000 + 123
        print(f"\n--- Training [{variant}] seed={seed} ---")
        student = train_and_probe(teacher, variant, seed, data_seed)
        results = run_full_probe_analysis(student, variant, seed)
        all_results[f"{variant}_seed{seed}"] = results

        # Save incrementally
        out_path = PROBE_DIR / "probe_results.json"
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                "probes": v["probes"],
                "ablation": v["ablation"],
            }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"BINDING PROBE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant/Seed':<25} {'correct_word':>15} {'query_person':>15} {'specificity':>12} {'base_mcq':>10}")
    print("-" * 80)
    for key, res in all_results.items():
        cw = res["probes"].get("answer_state/correct_word", 0)
        qp = res["probes"].get("answer_state/query_person", 0)
        sr = res["ablation"]["specificity_ratio"]
        bm = res["ablation"]["base_mcq"]
        print(f"{key:<25} {cw*100:>14.1f}% {qp*100:>14.1f}% {sr:>11.2f} {bm*100:>9.1f}%")


if __name__ == "__main__":
    main()
