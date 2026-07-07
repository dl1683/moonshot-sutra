"""Toy experiment v2: Structured KD with learnable discrimination.

Fixes the v1 problem (random tokens = no learnable signal) by creating
a synthetic language with GRAMMAR RULES that make some continuations
correct and others wrong.

Setup:
- Synthetic language: 3 word classes (subjects, verbs, objects)
- Grammar: Subject → Verb → Object (SVO order)
- Each "word" is a 2-4 byte sequence
- Teacher: word-level model that learns the grammar
- Student: byte-level model that must learn discrimination

Tests whether byte-marginal KD transfers discriminative ability
better or worse than ranking-based KD.
"""

import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SUBJECTS = [b'cat', b'dog', b'man', b'she', b'boy', b'elk', b'hen', b'fox',
            b'owl', b'ant', b'bee', b'cow', b'pig', b'rat', b'ape']
VERBS = [b'ate', b'saw', b'hit', b'ran', b'got', b'put', b'let', b'cut',
         b'bit', b'met', b'set', b'fed', b'led', b'won', b'dug']
OBJECTS = [b'pie', b'rug', b'box', b'hat', b'cup', b'bag', b'toy', b'map',
           b'pen', b'key', b'log', b'gem', b'jug', b'net', b'rod']

WORD_CLASSES = {'S': SUBJECTS, 'V': VERBS, 'O': OBJECTS}
ALL_WORDS = SUBJECTS + VERBS + OBJECTS
WORD_TO_ID = {w: i for i, w in enumerate(ALL_WORDS)}
VOCAB_SIZE = len(ALL_WORDS)

GRAMMAR_SEQUENCES = ['SVO', 'SVS', 'OVS']


def word_to_bytes(word):
    return list(word)


def sentence_to_bytes(words):
    result = []
    for i, w in enumerate(words):
        if i > 0:
            result.append(ord(' '))
        result.extend(w)
    return result


def generate_grammatical_sentence(rng, pattern='SVO'):
    words = []
    for c in pattern:
        words.append(rng.choice(WORD_CLASSES[c]))
    return words


def generate_ungrammatical_sentence(rng, correct_pattern='SVO'):
    wrong_patterns = ['VOS', 'OOO', 'SSS', 'VVV', 'OSV', 'VSO']
    pattern = rng.choice(wrong_patterns)
    return generate_grammatical_sentence(rng, pattern)


class WordTeacher(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, VOCAB_SIZE)

    def forward(self, word_ids):
        emb = self.embed(word_ids)
        out, _ = self.rnn(emb)
        return self.head(out)

    def score_sentence(self, word_ids):
        if len(word_ids) < 2:
            return 0.0
        ids = torch.tensor([word_ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.forward(ids)
        log_probs = F.log_softmax(logits, dim=-1)
        score = 0.0
        for i in range(1, len(word_ids)):
            score += log_probs[0, i - 1, word_ids[i]].item()
        return score / (len(word_ids) - 1)

    def get_next_word_probs(self, word_ids):
        ids = torch.tensor([word_ids], dtype=torch.long)
        with torch.no_grad():
            logits = self.forward(ids)
        return F.softmax(logits[0, -1], dim=-1).numpy()


def word_probs_to_byte_marginals(word_probs, n_byte_positions=4, top_k=16):
    marginals = []
    for pos in range(n_byte_positions):
        q = np.zeros(256, dtype=np.float64)
        for wid, p in enumerate(word_probs):
            word = ALL_WORDS[wid]
            bs = word_to_bytes(word)
            if pos < len(bs):
                q[bs[pos]] += p
        total = q.sum()
        if total > 0:
            uncovered = max(0.0, 1.0 - total)
            q += uncovered / 256.0
            q /= q.sum()
        else:
            q = np.ones(256) / 256.0
        marginals.append(q)
    return marginals


def compute_mi(word_probs, n_positions=4):
    word_probs = np.array(word_probs, dtype=np.float64)
    word_probs = word_probs / word_probs.sum()
    h_t = -np.sum(word_probs * np.log2(word_probs + 1e-30))

    mi_per_pos = []
    for pos in range(n_positions):
        byte_groups = {}
        for wid, p in enumerate(word_probs):
            bs = word_to_bytes(ALL_WORDS[wid])
            if pos < len(bs):
                b = bs[pos]
                if b not in byte_groups:
                    byte_groups[b] = []
                byte_groups[b].append((wid, p))

        h_t_given_b = 0
        for b, entries in byte_groups.items():
            p_b = sum(p for _, p in entries)
            if p_b > 0:
                h_cond = -sum(
                    (p / p_b) * math.log2(p / p_b + 1e-30)
                    for _, p in entries
                )
                h_t_given_b += p_b * h_cond
        mi_per_pos.append(h_t - h_t_given_b)

    return h_t, mi_per_pos


class ByteStudent(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(256, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 256)

    def forward(self, byte_ids):
        emb = self.embed(byte_ids)
        out, _ = self.rnn(emb)
        return self.head(out)

    def score_bytes(self, byte_seq):
        if len(byte_seq) < 2:
            return 0.0
        ids = torch.tensor([byte_seq], dtype=torch.long)
        with torch.no_grad():
            logits = self.forward(ids)
        log_probs = F.log_softmax(logits, dim=-1)
        score = 0.0
        for i in range(1, len(byte_seq)):
            score += log_probs[0, i - 1, byte_seq[i]].item()
        return score / (len(byte_seq) - 1)


class RankStudent(nn.Module):
    def __init__(self, max_bytes=20, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(256, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.scorer = nn.Linear(hidden, 1)
        self.max_bytes = max_bytes

    def forward(self, byte_seq):
        padded = byte_seq + [0] * (self.max_bytes - len(byte_seq))
        padded = padded[:self.max_bytes]
        ids = torch.tensor([padded], dtype=torch.long)
        emb = self.embed(ids)
        out, h = self.rnn(emb)
        return self.scorer(h.squeeze(0))


def train_teacher(teacher, n_epochs=200, lr=1e-3):
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    rng = random.Random(42)

    for epoch in range(n_epochs):
        total_loss = 0
        for _ in range(100):
            pattern = rng.choice(GRAMMAR_SEQUENCES)
            words = generate_grammatical_sentence(rng, pattern)
            word_ids = [WORD_TO_ID[w] for w in words]
            ids = torch.tensor([word_ids], dtype=torch.long)

            logits = teacher(ids)
            target = ids[:, 1:]
            logits_shifted = logits[:, :-1].reshape(-1, VOCAB_SIZE)
            loss = F.cross_entropy(logits_shifted, target.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    return total_loss / 100


def generate_discrimination_data(teacher, n_examples=2000, seed=0):
    rng = random.Random(seed)
    data = []

    for _ in range(n_examples):
        pattern = rng.choice(GRAMMAR_SEQUENCES)
        context_words = generate_grammatical_sentence(rng, pattern[:2])
        context_ids = [WORD_TO_ID[w] for w in context_words]
        context_bytes = sentence_to_bytes(context_words)

        correct_class = pattern[2]
        correct_word = rng.choice(WORD_CLASSES[correct_class])
        correct_bytes = [ord(' ')] + list(correct_word)

        wrong_choices = []
        for _ in range(3):
            wrong_class = rng.choice([c for c in ['S', 'V', 'O'] if c != correct_class])
            wrong_word = rng.choice(WORD_CLASSES[wrong_class])
            wrong_bytes = [ord(' ')] + list(wrong_word)
            wrong_choices.append(wrong_bytes)

        teacher_probs = teacher.get_next_word_probs(context_ids)
        marginals = word_probs_to_byte_marginals(teacher_probs)

        all_choices = [correct_bytes] + wrong_choices
        teacher_scores = []
        for choice in all_choices:
            full_words = context_words + [bytes(choice[1:])]
            full_ids = [WORD_TO_ID.get(w, 0) for w in full_words]
            s = teacher.score_sentence(full_ids)
            teacher_scores.append(s)

        data.append({
            'context_bytes': context_bytes,
            'choices': all_choices,
            'correct_idx': 0,
            'teacher_scores': teacher_scores,
            'marginals': marginals,
            'teacher_probs': teacher_probs,
        })

    return data


def train_byte_kl_student(student, data, epochs=100, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for ex in data:
            ctx = ex['context_bytes']
            marginals = ex['marginals']

            ids = torch.tensor([ctx], dtype=torch.long)
            logits = student(ids)

            loss = 0
            for pos, q in enumerate(marginals):
                if pos >= logits.shape[1]:
                    break
                student_log_probs = F.log_softmax(logits[0, -1], dim=-1).unsqueeze(0)
                q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
                loss += F.kl_div(student_log_probs, q_tensor, reduction='batchmean')

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    return total_loss / len(data)


def train_byte_ce_student(student, data, epochs=100, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for ex in data:
            correct_bytes = ex['context_bytes'] + ex['choices'][0]
            ids = torch.tensor([correct_bytes], dtype=torch.long)
            logits = student(ids)

            target = ids[:, 1:]
            pred = logits[:, :-1].reshape(-1, 256)
            loss = F.cross_entropy(pred, target.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    return total_loss / len(data)


def train_rank_student(student, data, epochs=100, lr=1e-3):
    opt = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for ex in data:
            scores = []
            for choice in ex['choices']:
                full_bytes = ex['context_bytes'] + choice
                s = student(full_bytes)
                scores.append(s)

            scores_t = torch.cat(scores).squeeze()
            target = torch.tensor([0], dtype=torch.long)
            loss = F.cross_entropy(scores_t.unsqueeze(0), target)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    return total_loss / len(data)


def evaluate_discrimination(model, data, mode='byte'):
    correct = 0
    teacher_agree = 0
    total = len(data)
    margins = []

    for ex in data:
        scores = []
        for choice in ex['choices']:
            full_bytes = ex['context_bytes'] + choice
            if mode == 'byte':
                s = model.score_bytes(full_bytes)
            elif mode == 'rank':
                with torch.no_grad():
                    s = model(full_bytes).item()
            scores.append(s)

        pred = max(range(len(scores)), key=lambda i: scores[i])
        teacher_best = max(range(len(ex['teacher_scores'])),
                          key=lambda i: ex['teacher_scores'][i])

        if pred == 0:
            correct += 1
        if pred == teacher_best:
            teacher_agree += 1

        margin = scores[0] - max(scores[1:])
        margins.append(margin)

    return {
        'ground_truth_acc': correct / total,
        'teacher_agree_acc': teacher_agree / total,
        'mean_margin': np.mean(margins),
        'positive_margin_frac': np.mean([1 if m > 0 else 0 for m in margins]),
    }


def main():
    print("=" * 60)
    print("TOY v2: Structured Language KD")
    print("=" * 60)

    print(f"\nVocab: {len(SUBJECTS)} subjects, {len(VERBS)} verbs, {len(OBJECTS)} objects")
    print(f"Grammar: {GRAMMAR_SEQUENCES}")

    print("\nTraining word-level teacher on grammar...")
    teacher = WordTeacher(hidden=64)
    final_loss = train_teacher(teacher, n_epochs=300)
    print(f"Teacher final loss: {final_loss:.4f}")

    rng = random.Random(0)
    test_patterns = ['SVO', 'SVS', 'OVS']
    print("\nTeacher grammar discrimination:")
    for pattern in test_patterns:
        words = generate_grammatical_sentence(rng, pattern)
        word_ids = [WORD_TO_ID[w] for w in words]
        s = teacher.score_sentence(word_ids)
        wrong = generate_ungrammatical_sentence(rng, pattern)
        wrong_ids = [WORD_TO_ID[w] for w in wrong]
        sw = teacher.score_sentence(wrong_ids)
        print(f"  {pattern} {[w.decode() for w in words]}: {s:.3f} | "
              f"wrong {[w.decode() for w in wrong]}: {sw:.3f} | "
              f"margin: {s - sw:+.3f}")

    print("\n--- Information Theory: Byte Marginal Loss ---")
    mi_samples = []
    for _ in range(50):
        ctx_words = generate_grammatical_sentence(rng, 'SV')
        ctx_ids = [WORD_TO_ID[w] for w in ctx_words]
        probs = teacher.get_next_word_probs(ctx_ids)
        h_t, mi_list = compute_mi(probs, n_positions=4)
        mi_samples.append((h_t, mi_list))

    avg_h = np.mean([h for h, _ in mi_samples])
    avg_mi = [np.mean([mi[i] for _, mi in mi_samples]) for i in range(4)]
    print(f"H(T): {avg_h:.2f} bits")
    for pos, mi in enumerate(avg_mi):
        pct = 100 * mi / avg_h if avg_h > 0 else 0
        print(f"  I(T; B_{pos}): {mi:.2f} bits ({pct:.1f}%)")

    total_unique_mi = avg_mi[0]
    print(f"\nByte-0 marginal preserves {100*avg_mi[0]/avg_h:.1f}% of teacher info")
    info_lost = avg_h - avg_mi[0]
    print(f"Information LOST: {info_lost:.2f} bits ({100*info_lost/avg_h:.1f}%)")
    print("(This lost information includes which specific WORD the teacher predicts)")

    print("\nGenerating discrimination data...")
    all_data = generate_discrimination_data(teacher, n_examples=2000)
    train_data = all_data[:1500]
    test_data = all_data[1500:]

    teacher_on_test = evaluate_discrimination(teacher, test_data, mode='byte')
    print(f"\nTeacher (word-level) on byte scoring:")
    print(f"  Ground truth acc: {teacher_on_test['ground_truth_acc']*100:.1f}%")

    print("\n--- Training Students ---")

    print("\n[1] Byte Student + KL on byte marginals...")
    byte_kl = ByteStudent(hidden=32)
    train_byte_kl_student(byte_kl, train_data, epochs=100)
    kl_results = evaluate_discrimination(byte_kl, test_data, mode='byte')

    print("[2] Byte Student + CE on correct bytes...")
    byte_ce = ByteStudent(hidden=32)
    train_byte_ce_student(byte_ce, train_data, epochs=100)
    ce_results = evaluate_discrimination(byte_ce, test_data, mode='byte')

    print("[3] Rank Student + contrastive ranking...")
    rank_s = RankStudent(max_bytes=20, hidden=32)
    train_rank_student(rank_s, train_data, epochs=100)
    rank_results = evaluate_discrimination(rank_s, test_data, mode='rank')

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<35} {'GT Acc':>10} {'Teacher Agree':>15} {'Mean Margin':>13}")
    print("-" * 75)
    print(f"{'Random':<35} {'25.0%':>10} {'25.0%':>15} {'0.000':>13}")
    print(f"{'[1] Byte KL (our approach)':<35} {kl_results['ground_truth_acc']*100:>9.1f}% "
          f"{kl_results['teacher_agree_acc']*100:>14.1f}% "
          f"{kl_results['mean_margin']:>13.4f}")
    print(f"{'[2] CE-only (no teacher)':<35} {ce_results['ground_truth_acc']*100:>9.1f}% "
          f"{ce_results['teacher_agree_acc']*100:>14.1f}% "
          f"{ce_results['mean_margin']:>13.4f}")
    print(f"{'[3] Ranking (contrastive)':<35} {rank_results['ground_truth_acc']*100:>9.1f}% "
          f"{rank_results['teacher_agree_acc']*100:>14.1f}% "
          f"{rank_results['mean_margin']:>13.4f}")

    print(f"\n--- Analysis ---")
    if rank_results['ground_truth_acc'] > kl_results['ground_truth_acc'] + 0.03:
        print(f"FINDING: Ranking loss ({rank_results['ground_truth_acc']*100:.1f}%) "
              f"beats byte KL ({kl_results['ground_truth_acc']*100:.1f}%) "
              f"by {(rank_results['ground_truth_acc']-kl_results['ground_truth_acc'])*100:.1f}pp")
        print("=> Byte-marginal KL DOES lose discriminative signal.")
    elif kl_results['ground_truth_acc'] > rank_results['ground_truth_acc'] + 0.03:
        print(f"SURPRISE: Byte KL ({kl_results['ground_truth_acc']*100:.1f}%) "
              f"beats ranking ({rank_results['ground_truth_acc']*100:.1f}%)")
    else:
        print("Methods perform similarly. Byte marginal bottleneck not confirmed in this setting.")

    if ce_results['ground_truth_acc'] >= kl_results['ground_truth_acc'] - 0.01:
        print("NOTE: CE-only matches or beats byte KL — teacher signal via marginals adds NO value.")
    else:
        print(f"NOTE: Byte KL ({kl_results['ground_truth_acc']*100:.1f}%) > CE-only "
              f"({ce_results['ground_truth_acc']*100:.1f}%) — marginals add some value.")


if __name__ == "__main__":
    main()
