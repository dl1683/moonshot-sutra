"""Toy experiment V2: Representation alignment vs byte-marginal KL.

V2 fixes (per Codex analysis of V1 failure):
  - Teacher trained 2000 epochs with 5x answer-position weight (V1: 35.6% MCQ)
  - Linear probe validates teacher hidden states contain binding info (>=70%)
  - Abort if teacher MCQ < 80%
  - Contextual alignment only at answer position with weight ramp and grad clip
  - Variant E: shuffled-hidden control (random position alignment)
  - 3 seeds per variant for statistical confidence

Design:
  - Synthetic binding task: "alice picked red rm_1 . bob picked blue rm_2 . query alice color answer"
  - Each word padded to 4 bytes (= 1 patch), giving clean token/patch alignment
  - Token-level teacher (width 128, 4 layers) trained to next-token accuracy
  - Byte-patch student (width 64, 4 layers, P=4) distilled via 5 methods:
    A: CE only (baseline)
    B: CE + byte-marginal KL
    C: CE + lexical embedding alignment
    D: CE + ctx hidden align at answer position + lex
    E: CE + shuffled hidden at answer position + lex (control)

Success gate:
  D beats B by >= 5pp on MCQ accuracy (mean over 3 seeds)
  D beats E (shuffled control) by >= 2pp
  D does not worsen BPB by > 3%
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


def word_to_bytes(word: str) -> list[int]:
    bs = list(word.encode("ascii"))
    while len(bs) < PATCH_SIZE:
        bs.append(0)
    return bs[:PATCH_SIZE]


WORD_BYTES = {w: word_to_bytes(w) for w in ALL_WORDS}


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

    return tokens, correct, distractors


def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [WORD2ID[t] for t in tokens]


def tokens_to_bytes(tokens: list[str]) -> list[int]:
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

    def forward(self, byte_ids, return_hidden=False):
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

        if return_hidden:
            return logits, hidden, raw_patches
        return logits

    def score_byte_continuation(self, context_bytes, answer_bytes):
        all_bytes = context_bytes + answer_bytes
        while len(all_bytes) % self.patch_size != 0:
            all_bytes.append(0)

        t = torch.tensor([all_bytes], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = self.forward(t)

        n_ctx_patches = len(context_bytes) // self.patch_size
        pred_patch = n_ctx_patches - 1
        if pred_patch < 0 or pred_patch >= logits.shape[1]:
            return -999.0

        score = 0.0
        for bp in range(min(len(answer_bytes), self.patch_size)):
            byte_val = answer_bytes[bp]
            score += F.log_softmax(logits[0, pred_patch, bp], dim=-1)[byte_val].item()
        return score


def word_logits_to_byte_marginals(word_logits, n_positions=4):
    probs = F.softmax(word_logits, dim=-1).detach().cpu().numpy()
    marginals = []
    for pos in range(n_positions):
        q = np.zeros(BYTE_VOCAB, dtype=np.float64)
        for wid in range(VOCAB_SIZE):
            bs = WORD_BYTES[ALL_WORDS[wid]]
            if pos < len(bs):
                q[bs[pos]] += probs[wid]
        total = q.sum()
        if total > 0:
            uncovered = max(0.0, 1.0 - total)
            q += uncovered / BYTE_VOCAB
            q /= q.sum()
        else:
            q = np.ones(BYTE_VOCAB) / BYTE_VOCAB
        marginals.append(torch.from_numpy(q.astype(np.float32)))
    return marginals


def byte_kl_loss(student_logits_patch, teacher_marginals, T=2.0):
    total = torch.tensor(0.0, device=student_logits_patch.device)
    for bp in range(min(student_logits_patch.shape[0], len(teacher_marginals))):
        log_p = F.log_softmax(student_logits_patch[bp] / T, dim=-1)
        q = teacher_marginals[bp].to(log_p.device)
        kl = F.kl_div(log_p, q, reduction="sum")
        total = total + (T * T) * kl
    return total / len(teacher_marginals)


def cosine_align_loss(student_state, teacher_state, proj):
    s = F.normalize(proj(student_state), dim=-1)
    t = F.normalize(teacher_state.detach(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


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
                tokens, correct, _ = generate_binding_example(rng)
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

        if (epoch + 1) % 50 == 0:
            print(f"  Teacher epoch {epoch+1}: loss={total_loss/n_batches:.4f}")


def probe_teacher_hidden(teacher, n_train=2000, n_test=500):
    rng_train = random.Random(77)
    train_hiddens = []
    train_targets = []
    for _ in range(n_train):
        tokens, correct, _ = generate_binding_example(rng_train)
        tokens_full = tokens + [correct]
        ids = tokens_to_ids(tokens_full)
        t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            _, hidden = teacher(t, return_hidden=True)
        train_hiddens.append(hidden[0, ANSWER_POS].clone())
        train_targets.append(WORD2ID[correct])

    H = torch.stack(train_hiddens)
    Y = torch.tensor(train_targets, dtype=torch.long, device=DEVICE)

    probe = nn.Linear(teacher.d_model, VOCAB_SIZE, bias=True).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for epoch in range(300):
        perm = torch.randperm(len(H))
        for start in range(0, len(H), 64):
            idx = perm[start:start + 64]
            logits = probe(H[idx])
            loss = F.cross_entropy(logits, Y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rng_test = random.Random(88)
    correct_count = 0
    for _ in range(n_test):
        tokens, correct, _ = generate_binding_example(rng_test)
        tokens_full = tokens + [correct]
        ids = tokens_to_ids(tokens_full)
        t = torch.tensor([ids], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            _, hidden = teacher(t, return_hidden=True)
        h = hidden[0, ANSWER_POS]
        with torch.no_grad():
            logits = probe(h.unsqueeze(0))
        pred = logits.argmax(dim=-1).item()
        if pred == WORD2ID[correct]:
            correct_count += 1

    return correct_count / n_test


def train_student(student, teacher, variant="A", n_steps=10000, lr=1e-3,
                  lambda_kl=0.10, lambda_lex=0.02, lambda_ctx=0.10,
                  ctx_ramp_steps=500, data_seed=123):
    ANSWER_CE_WEIGHT = 5.0
    lex_proj = nn.Linear(student.d_model, teacher.d_model, bias=False).to(DEVICE)
    ctx_proj = nn.Linear(student.d_model, teacher.d_model, bias=False).to(DEVICE)

    all_params = list(student.parameters())
    if variant in ("C", "D", "E"):
        all_params += list(lex_proj.parameters())
    if variant in ("D", "E"):
        all_params += list(ctx_proj.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)
    rng = random.Random(data_seed)

    for step in range(n_steps):
        tokens, correct, _ = generate_binding_example(rng)
        tokens_full = tokens + [correct]

        token_ids = tokens_to_ids(tokens_full)
        byte_seq = tokens_to_bytes(tokens_full)
        byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)

        student_logits, student_hidden, student_raw_patches = student(byte_t, return_hidden=True)

        N = student_logits.shape[1]
        P = student.patch_size
        target_bytes = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE).reshape(1, N, P)
        targets_shifted = target_bytes[:, 1:]
        preds = student_logits[:, :-1]
        ce_per_byte = F.cross_entropy(preds.reshape(-1, BYTE_VOCAB),
                                       targets_shifted.reshape(-1),
                                       reduction="none").reshape(N - 1, P)
        ce_weights = torch.ones(N - 1, device=DEVICE)
        ce_weights[ANSWER_POS] = ANSWER_CE_WEIGHT
        L_ce = (ce_per_byte * ce_weights.unsqueeze(1)).sum() / (ce_weights.sum() * P)

        L_kl = torch.tensor(0.0, device=DEVICE)
        L_lex = torch.tensor(0.0, device=DEVICE)
        L_ctx = torch.tensor(0.0, device=DEVICE)

        if variant in ("B", "D", "E"):
            teacher_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                t_logits = teacher(teacher_ids)
            kl_losses = []
            for i in range(min(N - 1, t_logits.shape[1] - 1)):
                marginals = word_logits_to_byte_marginals(t_logits[0, i])
                kl = byte_kl_loss(student_logits[0, i], marginals)
                kl_losses.append(kl)
            if kl_losses:
                L_kl = torch.stack(kl_losses).mean()

        if variant in ("C", "D", "E"):
            teacher_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                t_embs = teacher.emb(teacher_ids)
            n_align = min(student_raw_patches.shape[1], t_embs.shape[1])
            if n_align > 0:
                L_lex = cosine_align_loss(
                    student_raw_patches[0, :n_align],
                    t_embs[0, :n_align],
                    lex_proj)

        if variant in ("D", "E"):
            teacher_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                _, t_hidden = teacher(teacher_ids, return_hidden=True)

            s_h = student_hidden[0, ANSWER_POS:ANSWER_POS + 1]
            if variant == "D":
                t_h = t_hidden[0, ANSWER_POS:ANSWER_POS + 1]
            else:
                rand_pos = rng.randint(0, t_hidden.shape[1] - 2)
                if rand_pos >= ANSWER_POS:
                    rand_pos += 1
                t_h = t_hidden[0, rand_pos:rand_pos + 1]

            ramp = min(1.0, step / max(ctx_ramp_steps, 1))
            L_ctx = cosine_align_loss(s_h, t_h, ctx_proj) * ramp

        if variant == "A":
            loss = L_ce
        elif variant == "B":
            loss = L_ce + lambda_kl * L_kl
        elif variant == "C":
            loss = L_ce + lambda_lex * L_lex
        elif variant in ("D", "E"):
            loss = L_ce + lambda_lex * L_lex + lambda_ctx * L_ctx

        optimizer.zero_grad()
        loss.backward()

        if variant in ("D", "E"):
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        optimizer.step()

        if (step + 1) % 2000 == 0:
            ctx_val = L_ctx.item() if isinstance(L_ctx, torch.Tensor) else L_ctx
            print(f"    [{variant}] step {step+1}: CE={L_ce.item():.4f} "
                  f"KL={L_kl.item():.4f} lex={L_lex.item():.4f} ctx={ctx_val:.4f}")


def evaluate_mcq(student, teacher, n_examples=500, seed=999):
    rng = random.Random(seed)
    correct_count = 0
    teacher_agree = 0
    total_bpb = 0.0
    n_bpb = 0

    for _ in range(n_examples):
        tokens, correct, distractors = generate_binding_example(rng)
        choices = [correct] + distractors
        rng.shuffle(choices)

        context_bytes = tokens_to_bytes(tokens)

        student_scores = []
        for choice in choices:
            score = student.score_byte_continuation(context_bytes, WORD_BYTES[choice])
            student_scores.append(score)

        student_pick = choices[np.argmax(student_scores)]
        if student_pick == correct:
            correct_count += 1

        context_ids = tokens_to_ids(tokens)
        teacher_scores = [teacher.score_continuation(context_ids, WORD2ID[c]) for c in choices]
        teacher_pick = choices[np.argmax(teacher_scores)]
        if student_pick == teacher_pick:
            teacher_agree += 1

        tokens_full = tokens + [correct]
        byte_seq = tokens_to_bytes(tokens_full)
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

    return {
        "mcq_accuracy": accuracy,
        "teacher_agreement": teacher_agreement,
        "bpb": bpb,
    }


def main():
    print(f"Vocab: {VOCAB_SIZE} words, each padded to {PATCH_SIZE} bytes")
    print(f"Byte-patch student: P={PATCH_SIZE}, 1 word = 1 patch (clean alignment)")
    print(f"Student predicts ALL {PATCH_SIZE} bytes per patch, scores ALL bytes for MCQ")
    print(f"V2b: 500 teacher epochs (batch=32), 10K student steps, 5x answer weight, 3 seeds")
    print()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print(f"=== Training Teacher (500 epochs, batch=32, 5x answer weight, device={DEVICE}) ===")
    teacher = ToyTeacher(d_model=128, n_layers=4, n_heads=4).to(DEVICE)
    train_teacher(teacher)
    teacher.eval()

    rng = random.Random(999)
    correct_count = 0
    for _ in range(500):
        tokens, correct, distractors = generate_binding_example(rng)
        choices = [correct] + distractors
        rng.shuffle(choices)
        context_ids = tokens_to_ids(tokens)
        scores = [teacher.score_continuation(context_ids, WORD2ID[c]) for c in choices]
        if choices[np.argmax(scores)] == correct:
            correct_count += 1
    teacher_mcq = correct_count / 500
    print(f"\nTeacher MCQ accuracy: {teacher_mcq * 100:.1f}%")

    if teacher_mcq < 0.80:
        print(f"ABORT: Teacher MCQ ({teacher_mcq * 100:.1f}%) < 80%. "
              f"Teacher too weak to provide useful alignment signal.")
        return

    print("PASS: Teacher MCQ >= 80%")

    print("\n=== Probing teacher hidden states at answer position ===")
    probe_acc = probe_teacher_hidden(teacher)
    print(f"Probe accuracy: {probe_acc * 100:.1f}%")

    skip_ctx = False
    if probe_acc < 0.70:
        print(f"WARNING: Probe ({probe_acc * 100:.1f}%) < 70%. "
              f"Teacher hiddens may lack binding info. Skipping D/E.")
        skip_ctx = True
    else:
        print("PASS: Teacher hidden states encode binding info")

    SEEDS = [0, 1, 2]
    variants = ["A", "B", "C"]
    if not skip_ctx:
        variants.extend(["D", "E"])

    labels = {
        "A": "CE only (baseline)",
        "B": "CE + byte-marginal KL",
        "C": "CE + lexical embed align",
        "D": "CE + ctx align@answer + lex",
        "E": "CE + shuffled hidden (ctrl)",
    }

    all_results = {v: [] for v in variants}

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
            print(f"  [{v}] MCQ={r['mcq_accuracy'] * 100:.1f}%  "
                  f"T-Agree={r['teacher_agreement'] * 100:.1f}%  "
                  f"BPB={r['bpb']:.3f}")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean +/- std over 3 seeds)")
    print("=" * 70)
    print(f"{'Variant':<35} {'MCQ Acc':>14} {'T-Agree':>14} {'BPB':>12}")
    print("-" * 76)
    print(f"{'Teacher':<35} {teacher_mcq * 100:>8.1f}%")

    means = {}
    for v in variants:
        mcqs = [r["mcq_accuracy"] for r in all_results[v]]
        agrees = [r["teacher_agreement"] for r in all_results[v]]
        bpbs = [r["bpb"] for r in all_results[v]]
        m_mcq, s_mcq = np.mean(mcqs), np.std(mcqs)
        m_agr, s_agr = np.mean(agrees), np.std(agrees)
        m_bpb, s_bpb = np.mean(bpbs), np.std(bpbs)
        means[v] = {"mcq": m_mcq, "mcq_std": s_mcq, "bpb": m_bpb, "bpb_std": s_bpb}
        print(f"[{v}] {labels[v]:<32} "
              f"{m_mcq * 100:>5.1f}+/-{s_mcq * 100:>4.1f}% "
              f"{m_agr * 100:>5.1f}+/-{s_agr * 100:>4.1f}% "
              f"{m_bpb:>5.3f}+/-{s_bpb:.3f}")

    print("\n--- Gate Checks ---")

    if "D" in variants and "B" in variants:
        d_mcq = means["D"]["mcq"]
        b_mcq = means["B"]["mcq"]
        gap = (d_mcq - b_mcq) * 100
        print(f"D ({d_mcq * 100:.1f}%) vs B ({b_mcq * 100:.1f}%): gap = {gap:+.1f}pp")
        if gap >= 5.0:
            print("  PASS: D beats B by >= 5pp. Representation alignment validated.")
        elif gap > 0:
            print(f"  PARTIAL: D beats B but only by {gap:.1f}pp (need >= 5pp).")
        else:
            print("  FAIL: D does not beat B.")

    if "E" in variants and "D" in variants:
        e_mcq = means["E"]["mcq"]
        d_mcq = means["D"]["mcq"]
        gap = (d_mcq - e_mcq) * 100
        print(f"D ({d_mcq * 100:.1f}%) vs E ({e_mcq * 100:.1f}%): gap = {gap:+.1f}pp")
        if gap > 2.0:
            print("  PASS: D > shuffled control -> alignment signal is real.")
        else:
            print("  FAIL: D ~ E -> alignment signal may be noise.")

    if "D" in variants:
        a_bpb = means["A"]["bpb"]
        d_bpb = means["D"]["bpb"]
        bpb_deg = ((d_bpb - a_bpb) / a_bpb) * 100 if a_bpb > 0 else 0
        print(f"BPB: D={d_bpb:.3f} vs A={a_bpb:.3f} ({bpb_deg:+.1f}%)")
        if bpb_deg <= 3.0:
            print("  PASS: BPB degradation <= 3%.")
        else:
            print(f"  WARN: BPB degradation {bpb_deg:.1f}% > 3%.")

    a_mcq = means["A"]["mcq"]
    b_mcq = means["B"]["mcq"]
    print(f"\nCE-only baseline: {a_mcq * 100:.1f}%")
    if b_mcq <= a_mcq + 0.01:
        print("NOTE: Byte-marginal KL adds NO value over CE-only.")
    else:
        print(f"NOTE: Byte-marginal KL adds {(b_mcq - a_mcq) * 100:.1f}pp over CE-only.")

    if skip_ctx:
        print("\nNOTE: D/E skipped because teacher probe < 70%. "
              "Teacher hidden states lack binding information.")


if __name__ == "__main__":
    main()
