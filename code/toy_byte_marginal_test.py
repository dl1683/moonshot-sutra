"""Toy experiment: Does byte-marginal KD transfer ranking ability?

Tests the core hypothesis: matching byte marginals might improve byte
prediction while failing to transfer discriminative (ranking) ability.

Setup:
- Tiny "teacher" MLP that scores 4 choices given a context
- Vocabulary of 200 tokens, each mapping to 1-4 bytes
- Three student training modes:
  1. FULL: student matches teacher's full token distribution (upper bound)
  2. BYTE: student matches byte marginals derived from teacher tokens
  3. RANK: student matches teacher's ranking via contrastive loss
  4. CE-ONLY: student trained on ground truth labels only (no teacher)

Measures:
- Ranking accuracy (does student agree with teacher ranking?)
- Ground truth accuracy (does student pick the correct answer?)
- Byte prediction quality (BPB on the choices)
"""

import math
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_token_byte_table(vocab_size: int = 200, max_bytes: int = 4, seed: int = 42):
    rng = random.Random(seed)
    table = {}
    for tok_id in range(vocab_size):
        n_bytes = rng.choices([1, 2, 3, 4], weights=[0.1, 0.4, 0.3, 0.2])[0]
        bs = bytes([rng.randint(32, 127) for _ in range(n_bytes)])
        table[tok_id] = bs
    return table


def tokens_to_bytes(token_ids, table):
    result = []
    for t in token_ids:
        result.extend(table[t])
    return result


def byte_marginals_from_token_probs(token_probs, table, n_positions=4, top_k=16):
    marginals = []
    for pos in range(n_positions):
        q = np.zeros(256, dtype=np.float64)
        for tok_id, p in enumerate(token_probs):
            bs = table[tok_id]
            if pos < len(bs):
                q[bs[pos]] += p
        coverage = q.sum()
        if coverage > 0:
            uncovered = max(0.0, 1.0 - coverage)
            q += uncovered / 256.0
            q /= q.sum()
        else:
            q = np.ones(256) / 256.0
        marginals.append(q)
    return marginals


class ToyTeacher(nn.Module):
    def __init__(self, context_dim, vocab_size, hidden=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.scorer = nn.Linear(hidden, vocab_size)

    def forward(self, context):
        h = self.encoder(context)
        return self.scorer(h)

    def score_choice(self, context, choice_token_ids):
        logits = self.forward(context)
        log_probs = F.log_softmax(logits, dim=-1)
        score = sum(log_probs[0, tid].item() for tid in choice_token_ids)
        return score / len(choice_token_ids)


class ToyByteStudent(nn.Module):
    def __init__(self, context_dim, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.byte_head = nn.Linear(hidden, 256)

    def forward(self, context):
        h = self.encoder(context)
        return self.byte_head(h)

    def score_bytes(self, context, byte_sequence):
        logits = self.forward(context)
        log_probs = F.log_softmax(logits, dim=-1)
        score = sum(log_probs[0, b].item() for b in byte_sequence)
        return score / len(byte_sequence)


class ToyRankStudent(nn.Module):
    def __init__(self, context_dim, max_bytes=20, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim + max_bytes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.max_bytes = max_bytes

    def forward(self, context, choice_bytes):
        padded = choice_bytes + [0] * (self.max_bytes - len(choice_bytes))
        padded = padded[:self.max_bytes]
        x = torch.cat([context, torch.tensor([padded], dtype=torch.float32)], dim=-1)
        return self.encoder(x)


def generate_dataset(teacher, table, n_examples=2000, context_dim=32,
                     n_choices=4, choice_len=5, seed=0):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    dataset = []

    for _ in range(n_examples):
        ctx = torch.randn(1, context_dim)

        with torch.no_grad():
            teacher_logits = teacher(ctx)
            teacher_probs = F.softmax(teacher_logits, dim=-1).squeeze().numpy()

        choices_tokens = []
        choices_bytes = []
        for _ in range(n_choices):
            tokens = [rng.randint(0, teacher.vocab_size - 1) for _ in range(choice_len)]
            choices_tokens.append(tokens)
            choices_bytes.append(tokens_to_bytes(tokens, table))

        correct_idx = rng.randint(0, n_choices - 1)

        teacher_scores = []
        for tokens in choices_tokens:
            s = teacher.score_choice(ctx, tokens)
            teacher_scores.append(s)

        teacher_rank = sorted(range(n_choices), key=lambda i: -teacher_scores[i])
        teacher_best = teacher_rank[0]

        marginals = byte_marginals_from_token_probs(teacher_probs, table)

        dataset.append({
            'context': ctx,
            'choices_tokens': choices_tokens,
            'choices_bytes': choices_bytes,
            'correct_idx': correct_idx,
            'teacher_scores': teacher_scores,
            'teacher_best': teacher_best,
            'teacher_probs': teacher_probs,
            'marginals': marginals,
        })

    return dataset


def train_byte_student_kl(student, dataset, table, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for ex in dataset:
            ctx = ex['context']
            marginals = ex['marginals']

            student_logits = student(ctx)
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            loss = 0
            for pos_idx, q in enumerate(marginals):
                q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
                kl = F.kl_div(student_log_probs, q_tensor, reduction='batchmean')
                loss += kl

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataset))

    return losses


def train_byte_student_ce(student, dataset, table, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for ex in dataset:
            ctx = ex['context']
            correct_bytes = ex['choices_bytes'][ex['correct_idx']]

            student_logits = student(ctx)
            loss = 0
            for b in correct_bytes:
                target = torch.tensor([b], dtype=torch.long)
                loss += F.cross_entropy(student_logits, target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataset))

    return losses


def train_rank_student(student, dataset, table, epochs=50, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for ex in dataset:
            ctx = ex['context']
            teacher_scores = ex['teacher_scores']

            student_scores = []
            for choice_bytes in ex['choices_bytes']:
                s = student(ctx, choice_bytes)
                student_scores.append(s)

            student_scores_t = torch.cat(student_scores).squeeze()
            teacher_scores_t = torch.tensor(teacher_scores, dtype=torch.float32)

            teacher_target = F.softmax(teacher_scores_t * 5.0, dim=-1)
            student_log_probs = F.log_softmax(student_scores_t, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_target, reduction='batchmean')

            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss / len(dataset))

    return losses


def evaluate(model, dataset, table, mode='byte'):
    teacher_rank_agree = 0
    correct_agree = 0
    total = 0

    for ex in dataset:
        ctx = ex['context']
        teacher_best = ex['teacher_best']
        correct_idx = ex['correct_idx']

        if mode == 'byte':
            scores = []
            for choice_bytes in ex['choices_bytes']:
                s = model.score_bytes(ctx, choice_bytes)
                scores.append(s)
            pred = max(range(len(scores)), key=lambda i: scores[i])
        elif mode == 'rank':
            scores = []
            with torch.no_grad():
                for choice_bytes in ex['choices_bytes']:
                    s = model(ctx, choice_bytes).item()
                    scores.append(s)
            pred = max(range(len(scores)), key=lambda i: scores[i])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if pred == teacher_best:
            teacher_rank_agree += 1
        if pred == correct_idx:
            correct_agree += 1
        total += 1

    return {
        'teacher_rank_accuracy': teacher_rank_agree / total,
        'ground_truth_accuracy': correct_agree / total,
    }


def compute_mutual_information(teacher_probs, table, n_positions=4):
    h_teacher = -sum(p * math.log2(p + 1e-30) for p in teacher_probs)

    mi_per_pos = []
    for pos in range(n_positions):
        byte_groups = defaultdict(list)
        for tok_id, p in enumerate(teacher_probs):
            bs = table[tok_id]
            if pos < len(bs):
                byte_groups[bs[pos]].append((tok_id, p))

        h_t_given_b = 0
        for byte_val, entries in byte_groups.items():
            p_b = sum(p for _, p in entries)
            if p_b > 0:
                h_conditional = -sum(
                    (p / p_b) * math.log2(p / p_b + 1e-30)
                    for _, p in entries
                )
                h_t_given_b += p_b * h_conditional

        mi = h_teacher - h_t_given_b
        mi_per_pos.append(mi)

    return h_teacher, mi_per_pos


def main():
    print("=" * 60)
    print("TOY EXPERIMENT: Byte Marginal vs Ranking KD")
    print("=" * 60)

    context_dim = 32
    vocab_size = 200
    n_train = 1500
    n_test = 500

    table = build_token_byte_table(vocab_size)
    teacher = ToyTeacher(context_dim, vocab_size, hidden=128)

    print("\nTraining teacher on synthetic classification task...")
    torch.manual_seed(0)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    for epoch in range(100):
        loss = 0
        for _ in range(200):
            ctx = torch.randn(1, context_dim)
            logits = teacher(ctx)
            target = torch.randint(0, vocab_size, (1,))
            loss += F.cross_entropy(logits, target)
        teacher_opt.zero_grad()
        loss.backward()
        teacher_opt.step()
    print("Teacher trained.")

    print("\nGenerating dataset...")
    full_data = generate_dataset(teacher, table, n_examples=n_train + n_test,
                                  context_dim=context_dim)
    train_data = full_data[:n_train]
    test_data = full_data[n_train:]

    print(f"\n--- Information Theory Analysis ---")
    sample_probs = train_data[0]['teacher_probs']
    h_t, mi_positions = compute_mutual_information(sample_probs, table)
    print(f"Teacher entropy H(T): {h_t:.2f} bits")
    for pos, mi in enumerate(mi_positions):
        pct = 100 * mi / h_t if h_t > 0 else 0
        print(f"  I(T; B_{pos}): {mi:.2f} bits ({pct:.1f}% of H(T))")

    mi_total_avg = []
    for ex in train_data[:100]:
        h, mi_list = compute_mutual_information(ex['teacher_probs'], table)
        mi_total_avg.append((h, mi_list))
    avg_h = np.mean([h for h, _ in mi_total_avg])
    avg_mi = [np.mean([mi[i] for _, mi in mi_total_avg]) for i in range(4)]
    print(f"\nAverage over 100 examples:")
    print(f"  H(T): {avg_h:.2f} bits")
    for pos, mi in enumerate(avg_mi):
        pct = 100 * mi / avg_h if avg_h > 0 else 0
        print(f"  I(T; B_{pos}): {mi:.2f} bits ({pct:.1f}%)")
    total_mi = sum(avg_mi)
    total_pct = 100 * total_mi / avg_h if avg_h > 0 else 0
    print(f"  Total I(T; B_0..B_3): {total_mi:.2f} bits ({total_pct:.1f}%)")
    info_gap = avg_h - total_mi
    print(f"  INFORMATION GAP: {info_gap:.2f} bits ({100-total_pct:.1f}%) LOST in byte projection")

    print(f"\n--- Training Students (50 epochs each) ---")

    print("\n[1] Byte Student + Byte-Marginal KL (our current approach)...")
    byte_kl_student = ToyByteStudent(context_dim, hidden=64)
    byte_kl_losses = train_byte_student_kl(byte_kl_student, train_data, table, epochs=50)
    byte_kl_results = evaluate(byte_kl_student, test_data, table, mode='byte')

    print("[2] Byte Student + CE-only (no teacher)...")
    byte_ce_student = ToyByteStudent(context_dim, hidden=64)
    byte_ce_losses = train_byte_student_ce(byte_ce_student, train_data, table, epochs=50)
    byte_ce_results = evaluate(byte_ce_student, test_data, table, mode='byte')

    print("[3] Rank Student + Teacher Ranking (contrastive)...")
    rank_student = ToyRankStudent(context_dim, max_bytes=20, hidden=64)
    rank_losses = train_rank_student(rank_student, train_data, table, epochs=50)
    rank_results = evaluate(rank_student, test_data, table, mode='rank')

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<35} {'Teacher Rank Acc':>17} {'Ground Truth Acc':>17}")
    print("-" * 70)
    print(f"{'Random baseline':<35} {'25.0%':>17} {'25.0%':>17}")
    print(f"{'[1] Byte KL (our approach)':<35} {byte_kl_results['teacher_rank_accuracy']*100:>16.1f}% {byte_kl_results['ground_truth_accuracy']*100:>16.1f}%")
    print(f"{'[2] CE-only (no teacher)':<35} {byte_ce_results['teacher_rank_accuracy']*100:>16.1f}% {byte_ce_results['ground_truth_accuracy']*100:>16.1f}%")
    print(f"{'[3] Ranking (contrastive)':<35} {rank_results['teacher_rank_accuracy']*100:>16.1f}% {rank_results['ground_truth_accuracy']*100:>16.1f}%")

    print(f"\nFinal training losses:")
    print(f"  Byte KL: {byte_kl_losses[-1]:.4f}")
    print(f"  CE-only: {byte_ce_losses[-1]:.4f}")
    print(f"  Ranking: {rank_losses[-1]:.4f}")

    print(f"\n--- Key Finding ---")
    kl_rank = byte_kl_results['teacher_rank_accuracy']
    rank_rank = rank_results['teacher_rank_accuracy']
    if rank_rank > kl_rank + 0.05:
        print("CONFIRMED: Ranking loss transfers discriminative ability BETTER than byte KL.")
        print(f"  Gap: {(rank_rank - kl_rank)*100:.1f}pp")
    elif kl_rank > rank_rank + 0.05:
        print("SURPRISING: Byte KL transfers ranking better than direct ranking loss.")
        print(f"  Gap: {(kl_rank - rank_rank)*100:.1f}pp")
    else:
        print("INCONCLUSIVE: Both methods perform similarly on ranking.")

    print("\nDone.")


if __name__ == "__main__":
    main()
