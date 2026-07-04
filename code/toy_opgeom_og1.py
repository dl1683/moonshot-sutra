# ARCHIVAL: reproduces OG-1 as run. Known issues are documented in
# research/OG1_RESULTS_ANALYSIS.md and research/OG1B_IMPLEMENTATION_SPEC.md.
"""OG-1: Operational Geometry toy experiment.

Tests whether ranking + invariance + counterfactual losses beat CE-only
when trained on public behavioral structure (rankings, transformations)
instead of private teacher coordinates (hidden states, byte marginals).

Variants:
  A: CE only (baseline)
  B: CE + listwise ranking
  C: CE + ranking + preserving invariance
  D: CE + ranking + preserving invariance + counterfactual loss
  E: D but shuffled teacher rankings (control)
  F: D but random "preserving" pairs from unrelated contexts (control)

Success gates (Codex-specified):
  D beats A by >= 8pp MCQ accuracy
  D beats B by >= 3pp on held-out transformed contexts
  D beats E and F by >= 6pp
  D does not worsen BPB by more than 5%
"""

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NAMES = ["alic", "bobx", "carl", "dana", "evan", "faye", "glen", "hope"]
COLORS = ["redx", "blue", "gren", "gold", "pink", "grey", "teal", "plum"]
ROOMS = ["rm_1", "rm_2", "rm_3", "rm_4", "rm_5", "rm_6", "rm_7", "rm_8"]
ACTIONS = ["pick", "grab", "take", "hold"]
SPECIAL = ["strt", "endx", "qury", "answ", "padx", "colr", "room", "actn"]

ALL_WORDS = NAMES + COLORS + ROOMS + ACTIONS + SPECIAL
WORD2ID = {w: i for i, w in enumerate(ALL_WORDS)}
VOCAB_SIZE = len(ALL_WORDS)
PAD_ID = WORD2ID["padx"]
BYTE_VOCAB = 256
PATCH_SIZE = 4

ANSWER_POS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 1.0


def word_to_bytes(word: str) -> list[int]:
    bs = list(word.encode("ascii"))
    while len(bs) < PATCH_SIZE:
        bs.append(0)
    return bs[:PATCH_SIZE]


WORD_BYTES = {w: word_to_bytes(w) for w in ALL_WORDS}


# --- Data generation with transformations ---

def generate_binding_example(rng: random.Random):
    names = rng.sample(NAMES, 2)
    colors = rng.sample(COLORS, 2)
    rooms = rng.sample(ROOMS, 2)
    actions = [rng.choice(ACTIONS) for _ in range(2)]

    tokens = ["strt"]
    for i in range(2):
        tokens.extend([names[i], actions[i], colors[i], rooms[i]])
    query_attr = rng.choice(["colr", "room", "actn"])
    query_person = rng.choice([0, 1])
    tokens.extend(["qury", names[query_person], query_attr, "answ"])

    if query_attr == "colr":
        correct = colors[query_person]
        wrongs = [c for c in COLORS if c != correct]
    elif query_attr == "room":
        correct = rooms[query_person]
        wrongs = [r for r in ROOMS if r != correct]
    else:
        correct = actions[query_person]
        wrongs = [a for a in ACTIONS if a != correct]

    distractors = rng.sample(wrongs, min(3, len(wrongs)))
    while len(distractors) < 3:
        distractors.append(rng.choice(wrongs))

    meta = {
        "names": names,
        "colors": colors,
        "rooms": rooms,
        "actions": actions,
        "query_attr": query_attr,
        "query_person": query_person,
    }
    return tokens, correct, distractors, meta


def apply_preserving_transform(tokens, meta, rng, transform_type="swap"):
    names = meta["names"]
    colors = meta["colors"]
    rooms = meta["rooms"]
    actions = meta["actions"]
    qp = meta["query_person"]
    qa = meta["query_attr"]

    if transform_type == "swap":
        new_tokens = ["strt"]
        new_tokens.extend([names[1], actions[1], colors[1], rooms[1]])
        new_tokens.extend([names[0], actions[0], colors[0], rooms[0]])
        new_tokens.extend(["qury", names[qp], qa, "answ"])
        return new_tokens

    elif transform_type == "change_irrelevant":
        other = 1 - qp
        new_colors = list(colors)
        new_rooms = list(rooms)
        new_actions = list(actions)

        if qa == "colr":
            available = [c for c in COLORS if c not in colors]
            if available:
                new_colors[other] = rng.choice(available)
        elif qa == "room":
            available = [r for r in ROOMS if r not in rooms]
            if available:
                new_rooms[other] = rng.choice(available)
        else:
            new_actions[other] = rng.choice(ACTIONS)

        new_tokens = ["strt"]
        for i in range(2):
            new_tokens.extend([names[i], new_actions[i], new_colors[i], new_rooms[i]])
        new_tokens.extend(["qury", names[qp], qa, "answ"])
        return new_tokens

    elif transform_type == "rename_other":
        other = 1 - qp
        available = [n for n in NAMES if n not in names]
        if not available:
            return list(tokens)
        new_names = list(names)
        new_names[other] = rng.choice(available)
        new_tokens = ["strt"]
        for i in range(2):
            new_tokens.extend([new_names[i], actions[i], colors[i], rooms[i]])
        new_tokens.extend(["qury", new_names[qp], qa, "answ"])
        return new_tokens

    return list(tokens)


def apply_counterfactual_transform(tokens, meta, rng):
    names = meta["names"]
    colors = meta["colors"]
    rooms = meta["rooms"]
    actions = meta["actions"]
    qp = meta["query_person"]
    qa = meta["query_attr"]

    other = 1 - qp
    new_tokens = ["strt"]
    for i in range(2):
        new_tokens.extend([names[i], actions[i], colors[i], rooms[i]])
    new_tokens.extend(["qury", names[other], qa, "answ"])

    if qa == "colr":
        cf_correct = colors[other]
        cf_wrongs = [c for c in COLORS if c != cf_correct]
    elif qa == "room":
        cf_correct = rooms[other]
        cf_wrongs = [r for r in ROOMS if r != cf_correct]
    else:
        cf_correct = actions[other]
        cf_wrongs = [a for a in ACTIONS if a != cf_correct]

    cf_distractors = rng.sample(cf_wrongs, min(3, len(cf_wrongs)))
    while len(cf_distractors) < 3:
        cf_distractors.append(rng.choice(cf_wrongs))

    return new_tokens, cf_correct, cf_distractors


# --- Model definitions (same as V2) ---

def tokens_to_ids(tokens):
    return [WORD2ID[t] for t in tokens]


def tokens_to_bytes_seq(tokens):
    bs = []
    for t in tokens:
        bs.extend(WORD_BYTES[t])
    return bs


class ToyTeacher(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(64, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.0, batch_first=True,
                                       norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.d_model = d_model

    def forward(self, token_ids, return_hidden=False):
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device).unsqueeze(0)
        x = self.emb(token_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=token_ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        h = self.norm(x)
        logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def score_continuation(self, context_ids, answer_id):
        ids = context_ids + [answer_id]
        t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.forward(t)
        return F.log_softmax(logits[0, -2], dim=-1)[answer_id].item()


class ToyByteStudent(nn.Module):
    def __init__(self, d_model=64, n_layers=4, n_heads=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.byte_emb = nn.Embedding(BYTE_VOCAB, d_model)
        self.patch_proj = nn.Linear(patch_size * d_model, d_model, bias=False)
        self.pos_emb = nn.Embedding(64, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                       dropout=0.0, batch_first=True,
                                       norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, BYTE_VOCAB * patch_size, bias=False)
        self.d_model = d_model

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        P = self.patch_size
        N = T // P
        x = self.byte_emb(byte_ids)
        x = x.reshape(B, N, P * self.d_model)
        raw_patches = self.patch_proj(x)
        pos = torch.arange(N, device=byte_ids.device).unsqueeze(0)
        patch_states = raw_patches + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=byte_ids.device)
        h = patch_states
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        hidden = self.norm(h)
        logits = self.byte_decoder(hidden).reshape(B, N, P, BYTE_VOCAB)
        return logits

    def score_candidate(self, context_tokens, candidate_word):
        context_bytes = tokens_to_bytes_seq(context_tokens)
        cand_bytes = WORD_BYTES[candidate_word]
        all_bytes = context_bytes + cand_bytes
        while len(all_bytes) % self.patch_size != 0:
            all_bytes.append(0)
        t = torch.tensor([all_bytes], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.forward(t)
        n_ctx_patches = len(context_bytes) // self.patch_size
        pred_patch = n_ctx_patches - 1
        if pred_patch < 0 or pred_patch >= logits.shape[1]:
            return torch.tensor(-999.0, device=DEVICE)
        score = 0.0
        for bp in range(min(len(cand_bytes), self.patch_size)):
            score += F.log_softmax(logits[0, pred_patch, bp], dim=-1)[cand_bytes[bp]]
        return score / len(cand_bytes)

    def score_candidates_batch(self, context_tokens, candidates):
        scores = []
        context_bytes = tokens_to_bytes_seq(context_tokens)
        t_full_list = []
        for cand in candidates:
            cand_bytes = WORD_BYTES[cand]
            ab = context_bytes + cand_bytes
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

        for ci, cand in enumerate(candidates):
            cand_bytes = WORD_BYTES[cand]
            s = 0.0
            for bp in range(min(len(cand_bytes), self.patch_size)):
                s = s + F.log_softmax(logits[ci, pred_patch, bp], dim=-1)[cand_bytes[bp]]
            scores.append(s / len(cand_bytes))

        return torch.stack(scores)


# --- Training ---

def train_teacher(teacher, n_epochs=500, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    rng = random.Random(42)
    ANSWER_WEIGHT = 5.0
    batches_per_epoch = 20

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches = 0
        for _ in range(batches_per_epoch):
            batch_ids = []
            for _ in range(batch_size):
                tokens, correct, _, _ = generate_binding_example(rng)
                tokens_full = tokens + [correct]
                batch_ids.append(tokens_to_ids(tokens_full))
            t = torch.tensor(batch_ids, dtype=torch.long, device=DEVICE)
            logits = teacher(t)
            targets = t[:, 1:]
            B, S = targets.shape
            loss_per_pos = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                targets.reshape(-1), reduction="none").reshape(B, S)
            weights = torch.ones(S, device=DEVICE)
            weights[ANSWER_POS] = ANSWER_WEIGHT
            loss = (loss_per_pos * weights.unsqueeze(0)).sum() / (B * weights.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 100 == 0:
            print(f"  Teacher epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


def compute_ce_loss(student, tokens_full):
    byte_seq = tokens_to_bytes_seq(tokens_full)
    byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)
    logits = student(byte_t)
    N = logits.shape[1]
    P = student.patch_size
    target_bytes = byte_t.reshape(1, N, P)
    targets_shifted = target_bytes[:, 1:]
    preds = logits[:, :-1]
    ANSWER_CE_WEIGHT = 5.0
    ce_per_byte = F.cross_entropy(preds.reshape(-1, BYTE_VOCAB),
                                   targets_shifted.reshape(-1),
                                   reduction="none").reshape(N - 1, P)
    ce_weights = torch.ones(N - 1, device=DEVICE)
    ce_weights[min(ANSWER_POS, N - 2)] = ANSWER_CE_WEIGHT
    L_ce = (ce_per_byte * ce_weights.unsqueeze(1)).sum() / (ce_weights.sum() * P)
    return L_ce


def compute_ranking_loss(student, context_tokens, correct, distractors, tau=TAU):
    candidates = [correct] + distractors
    scores = student.score_candidates_batch(context_tokens, candidates)
    gold_idx = torch.tensor([0], dtype=torch.long, device=DEVICE)
    L_rank = F.cross_entropy(scores.unsqueeze(0) / tau, gold_idx)
    return L_rank


def compute_invariance_loss(student, orig_tokens, transformed_tokens,
                            correct, distractors, tau=TAU):
    candidates = [correct] + distractors
    scores_orig = student.score_candidates_batch(orig_tokens, candidates)
    scores_trans = student.score_candidates_batch(transformed_tokens, candidates)

    p = F.log_softmax(scores_orig / tau, dim=-1)
    q = F.softmax(scores_trans.detach() / tau, dim=-1)
    kl1 = F.kl_div(p, q, reduction="batchmean")

    p2 = F.log_softmax(scores_trans / tau, dim=-1)
    q2 = F.softmax(scores_orig.detach() / tau, dim=-1)
    kl2 = F.kl_div(p2, q2, reduction="batchmean")

    return (kl1 + kl2) / 2


def compute_counterfactual_loss(student, cf_tokens, cf_correct, cf_distractors, tau=TAU):
    cf_candidates = [cf_correct] + cf_distractors
    scores = student.score_candidates_batch(cf_tokens, cf_candidates)
    gold_idx = torch.tensor([0], dtype=torch.long, device=DEVICE)
    L_cf = F.cross_entropy(scores.unsqueeze(0) / tau, gold_idx)
    return L_cf


def train_student(student, teacher, variant="A", n_steps=8000, lr=1e-3,
                  data_seed=123, warmup_steps=500):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    rng = random.Random(data_seed)
    rng_unrelated = random.Random(data_seed + 9999)

    lambda_rank = 0.35
    lambda_inv = 0.10
    lambda_cf = 0.25

    for step in range(n_steps):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        tokens_full = tokens + [correct]

        L_ce = compute_ce_loss(student, tokens_full)

        loss = L_ce
        L_rank_val = 0.0
        L_inv_val = 0.0
        L_cf_val = 0.0

        if variant in ("B", "C", "D", "E", "F") and step >= warmup_steps:
            if variant == "E":
                shuffled_distractors = list(distractors)
                rng.shuffle(shuffled_distractors)
                fake_correct = shuffled_distractors[0]
                fake_distractors = [correct] + shuffled_distractors[1:]
                L_rank = compute_ranking_loss(student, tokens, fake_correct,
                                              fake_distractors)
            else:
                L_rank = compute_ranking_loss(student, tokens, correct, distractors)
            L_rank_val = L_rank.item()
            loss = loss + lambda_rank * L_rank

        if variant in ("C", "D", "E", "F") and step >= warmup_steps:
            t_type = rng.choice(["swap", "change_irrelevant", "rename_other"])

            if variant == "F":
                unrelated_tokens, _, _, unrelated_meta = generate_binding_example(rng_unrelated)
                trans_tokens = apply_preserving_transform(
                    unrelated_tokens, unrelated_meta, rng, t_type)
            else:
                trans_tokens = apply_preserving_transform(tokens, meta, rng, t_type)

            L_inv = compute_invariance_loss(student, tokens, trans_tokens,
                                            correct, distractors)
            L_inv_val = L_inv.item()
            loss = loss + lambda_inv * L_inv

        if variant in ("D", "E", "F") and step >= warmup_steps:
            cf_tokens, cf_correct, cf_distractors = apply_counterfactual_transform(
                tokens, meta, rng)

            if variant == "E":
                cf_distractors_shuffled = list(cf_distractors)
                rng.shuffle(cf_distractors_shuffled)
                fake_cf_correct = cf_distractors_shuffled[0]
                fake_cf_dis = [cf_correct] + cf_distractors_shuffled[1:]
                L_cf = compute_counterfactual_loss(student, cf_tokens,
                                                   fake_cf_correct, fake_cf_dis)
            else:
                L_cf = compute_counterfactual_loss(student, cf_tokens,
                                                   cf_correct, cf_distractors)
            L_cf_val = L_cf.item()
            loss = loss + lambda_cf * L_cf

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 2000 == 0:
            print(f"    [{variant}] step {step+1}: CE={L_ce.item():.4f} "
                  f"rank={L_rank_val:.4f} inv={L_inv_val:.4f} cf={L_cf_val:.4f}")


# --- Evaluation ---

def evaluate_mcq(student, teacher, n_examples=500, seed=999):
    rng = random.Random(seed)
    correct_count = 0
    teacher_agree = 0
    total_bpb = 0.0
    n_bpb = 0

    for _ in range(n_examples):
        tokens, correct, distractors, _ = generate_binding_example(rng)
        choices = [correct] + distractors
        rng.shuffle(choices)

        student_scores = []
        for choice in choices:
            with torch.no_grad():
                score = student.score_candidate(tokens, choice)
            student_scores.append(score.item())

        student_pick = choices[np.argmax(student_scores)]
        if student_pick == correct:
            correct_count += 1

        context_ids = tokens_to_ids(tokens)
        teacher_scores = [teacher.score_continuation(context_ids, WORD2ID[c]) for c in choices]
        teacher_pick = choices[np.argmax(teacher_scores)]
        if student_pick == teacher_pick:
            teacher_agree += 1

        tokens_full = tokens + [correct]
        byte_seq = tokens_to_bytes_seq(tokens_full)
        byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)
        P = student.patch_size
        with torch.no_grad():
            logits = student(byte_t)
        N = logits.shape[1]
        byte_targets = torch.tensor(byte_seq, dtype=torch.long, device=DEVICE).reshape(N, P)
        for i in range(1, N):
            for bp in range(P):
                lp = F.log_softmax(logits[0, i - 1, bp], dim=-1)[byte_targets[i, bp]].item()
                total_bpb -= lp
                n_bpb += 1

    accuracy = correct_count / n_examples
    teacher_agreement = teacher_agree / n_examples
    bpb = (total_bpb / n_bpb) / math.log(2) if n_bpb > 0 else float("inf")

    return {"mcq_accuracy": accuracy, "teacher_agreement": teacher_agreement, "bpb": bpb}


def evaluate_transformed_mcq(student, n_examples=500, seed=888):
    rng = random.Random(seed)
    correct_clean = 0
    correct_swapped = 0
    correct_irrelevant = 0
    correct_renamed = 0

    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        choices = [correct] + distractors

        def score_all(ctx_tokens):
            scores = []
            for c in choices:
                with torch.no_grad():
                    s = student.score_candidate(ctx_tokens, c)
                scores.append(s.item())
            return choices[np.argmax(scores)] == correct

        correct_clean += int(score_all(tokens))

        swapped = apply_preserving_transform(tokens, meta, rng, "swap")
        correct_swapped += int(score_all(swapped))

        irrelevant = apply_preserving_transform(tokens, meta, rng, "change_irrelevant")
        correct_irrelevant += int(score_all(irrelevant))

        renamed = apply_preserving_transform(tokens, meta, rng, "rename_other")
        correct_renamed += int(score_all(renamed))

    return {
        "clean": correct_clean / n_examples,
        "swapped": correct_swapped / n_examples,
        "irrelevant_changed": correct_irrelevant / n_examples,
        "renamed": correct_renamed / n_examples,
        "avg_transformed": (correct_swapped + correct_irrelevant + correct_renamed) / (3 * n_examples),
    }


def main():
    print(f"OG-1: Operational Geometry Toy Experiment")
    print(f"Device: {DEVICE}")
    print(f"Variants: A(CE) B(+rank) C(+inv) D(+cf) E(shuffled) F(random-preserve)")
    print(f"8K steps, 3 seeds, warmup 500 steps")
    print()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("=== Training Teacher (500 epochs, batch=32, 5x answer weight) ===")
    teacher = ToyTeacher(d_model=128, n_layers=4, n_heads=4).to(DEVICE)
    train_teacher(teacher)
    teacher.eval()

    rng = random.Random(999)
    correct_count = 0
    for _ in range(500):
        tokens, correct, distractors, _ = generate_binding_example(rng)
        choices = [correct] + distractors
        rng.shuffle(choices)
        context_ids = tokens_to_ids(tokens)
        scores = [teacher.score_continuation(context_ids, WORD2ID[c]) for c in choices]
        if choices[np.argmax(scores)] == correct:
            correct_count += 1
    teacher_mcq = correct_count / 500
    print(f"\nTeacher MCQ accuracy: {teacher_mcq * 100:.1f}%")
    if teacher_mcq < 0.95:
        print(f"WARNING: Teacher MCQ ({teacher_mcq*100:.1f}%) < 95%")
    else:
        print("PASS: Teacher MCQ >= 95%")

    SEEDS = [0, 1, 2]
    variants = ["A", "B", "C", "D", "E", "F"]
    labels = {
        "A": "CE only (baseline)",
        "B": "CE + ranking",
        "C": "CE + ranking + invariance",
        "D": "CE + rank + inv + counterfactual",
        "E": "D shuffled rankings (control)",
        "F": "D random preserve (control)",
    }

    all_results = {v: [] for v in variants}
    all_transform_results = {v: [] for v in variants}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 70}")
        print(f"SEED {seed_idx + 1}/{len(SEEDS)} (seed={seed})")
        print(f"{'=' * 70}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        base_student = ToyByteStudent(d_model=64, n_layers=4, n_heads=4,
                                       patch_size=PATCH_SIZE).to(DEVICE)
        base_state = copy.deepcopy(base_student.state_dict())
        data_seed = seed * 1000 + 123

        for v in variants:
            print(f"\n  --- [{v}] {labels[v]} (seed={seed}) ---")
            student = ToyByteStudent(d_model=64, n_layers=4, n_heads=4,
                                      patch_size=PATCH_SIZE).to(DEVICE)
            student.load_state_dict(copy.deepcopy(base_state))
            torch.manual_seed(seed + hash(v) % 10000)
            train_student(student, teacher, variant=v, data_seed=data_seed)
            student.eval()

            r = evaluate_mcq(student, teacher, n_examples=500, seed=999)
            all_results[v].append(r)
            print(f"  [{v}] MCQ={r['mcq_accuracy']*100:.1f}%  "
                  f"T-Agree={r['teacher_agreement']*100:.1f}%  "
                  f"BPB={r['bpb']:.3f}")

            tr = evaluate_transformed_mcq(student, n_examples=500, seed=888)
            all_transform_results[v].append(tr)
            print(f"  [{v}] Transform: clean={tr['clean']*100:.1f}% "
                  f"swap={tr['swapped']*100:.1f}% "
                  f"irrel={tr['irrelevant_changed']*100:.1f}% "
                  f"rename={tr['renamed']*100:.1f}% "
                  f"avg_t={tr['avg_transformed']*100:.1f}%")

    # --- Results summary ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean +/- std over 3 seeds)")
    print("=" * 70)
    print(f"{'Variant':<40} {'MCQ':>10} {'Avg Trans':>10} {'BPB':>10}")
    print("-" * 72)
    print(f"{'Teacher':<40} {teacher_mcq*100:>8.1f}%")

    means = {}
    for v in variants:
        mcqs = [r["mcq_accuracy"] for r in all_results[v]]
        bpbs = [r["bpb"] for r in all_results[v]]
        avg_ts = [r["avg_transformed"] for r in all_transform_results[v]]
        m_mcq, s_mcq = np.mean(mcqs), np.std(mcqs)
        m_bpb, s_bpb = np.mean(bpbs), np.std(bpbs)
        m_at, s_at = np.mean(avg_ts), np.std(avg_ts)
        means[v] = {"mcq": m_mcq, "s_mcq": s_mcq, "bpb": m_bpb, "s_bpb": s_bpb,
                     "avg_trans": m_at, "s_at": s_at}
        print(f"[{v}] {labels[v]:<37} "
              f"{m_mcq*100:>5.1f}+/-{s_mcq*100:>4.1f}% "
              f"{m_at*100:>5.1f}+/-{s_at*100:>4.1f}% "
              f"{m_bpb:>5.3f}+/-{s_bpb:.3f}")

    # --- Gate checks ---
    print("\n--- OG-1 Success Gates ---")

    d_mcq = means["D"]["mcq"]
    a_mcq = means["A"]["mcq"]
    gap_da = (d_mcq - a_mcq) * 100
    print(f"\n1. D vs A (gate: +8pp): D={d_mcq*100:.1f}% A={a_mcq*100:.1f}% gap={gap_da:+.1f}pp")
    print(f"   {'PASS' if gap_da >= 8 else 'FAIL'}")

    b_at = means["B"]["avg_trans"]
    d_at = means["D"]["avg_trans"]
    gap_db_t = (d_at - b_at) * 100
    print(f"\n2. D vs B on transforms (gate: +3pp): D={d_at*100:.1f}% B={b_at*100:.1f}% gap={gap_db_t:+.1f}pp")
    print(f"   {'PASS' if gap_db_t >= 3 else 'FAIL'}")

    e_mcq = means["E"]["mcq"]
    f_mcq = means["F"]["mcq"]
    gap_de = (d_mcq - e_mcq) * 100
    gap_df = (d_mcq - f_mcq) * 100
    print(f"\n3. D vs E (gate: +6pp): D={d_mcq*100:.1f}% E={e_mcq*100:.1f}% gap={gap_de:+.1f}pp")
    print(f"   {'PASS' if gap_de >= 6 else 'FAIL'}")
    print(f"   D vs F (gate: +6pp): D={d_mcq*100:.1f}% F={f_mcq*100:.1f}% gap={gap_df:+.1f}pp")
    print(f"   {'PASS' if gap_df >= 6 else 'FAIL'}")

    a_bpb = means["A"]["bpb"]
    d_bpb = means["D"]["bpb"]
    bpb_deg = ((d_bpb - a_bpb) / a_bpb) * 100 if a_bpb > 0 else 0
    print(f"\n4. BPB (gate: <= +5%): D={d_bpb:.3f} A={a_bpb:.3f} ({bpb_deg:+.1f}%)")
    print(f"   {'PASS' if bpb_deg <= 5 else 'FAIL'}")

    all_pass = (gap_da >= 8 and gap_db_t >= 3 and gap_de >= 6
                and gap_df >= 6 and bpb_deg <= 5)
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL GATES PASS — Operational Geometry VALIDATED' if all_pass else 'SOME GATES FAILED'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
