"""OG-1b: Operational Geometry toy experiment — augmentation vs relational geometry.

Primary question: Does relational OG beat matched counterfactual augmentation?
Key comparison: D_rel_full vs A_cf_ce on held-out composite CF direction accuracy.

Variants:
  A_ce:        CE only (baseline)
  A_more_ce:   CE on orig + extra unrelated example (matched data control)
  A_cf_ce:     CE on orig + counterfactual example (matched CF augmentation)
  B_rank:      CE + ranking
  C_inv_fixed: CE + ranking + fixed invariance (score-vector matching)
  D_aug_cf:    CE + ranking + CF augmentation (OG-1 style, cleaned up)
  D_rel_full:  CE + ranking + fixed invariance + relational CF (MAIN OG-1b)
  E_adv:       CE + adversarial wrong ranking + wrong relational CF (control)
  F_rand_inv:  CE + ranking + random-context invariance + relational CF (control)

Success gates (Codex-specified):
  Debug: Teacher MCQ>=95%, A_ce>=50%, E_adv<=35%, L_inv active, BPB<=+5%
  Primary: D_rel_full beats A_cf_ce by >=+2pp on heldout_cf_direction_accuracy
  Secondary: D_rel_full beats D_aug_cf by >=+2pp on heldout composite CF
"""

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Constants ---

NAMES = ["alic", "bobx", "carl", "dana", "evan", "faye", "glen", "hope"]
COLORS = ["redx", "blue", "gren", "gold", "pink", "grey", "teal", "plum"]
ROOMS = ["rm_1", "rm_2", "rm_3", "rm_4", "rm_5", "rm_6", "rm_7", "rm_8"]
ACTIONS = ["pick", "grab", "take", "hold"]
SPECIAL = ["strt", "endx", "qury", "answ", "padx", "colr", "room", "actn"]
ATTRS = ["colr", "room", "actn"]
ATTR_POOLS = {"colr": COLORS, "room": ROOMS, "actn": ACTIONS}

ALL_WORDS = NAMES + COLORS + ROOMS + ACTIONS + SPECIAL
WORD2ID = {w: i for i, w in enumerate(ALL_WORDS)}
VOCAB_SIZE = len(ALL_WORDS)
BYTE_VOCAB = 256
PATCH_SIZE = 4
ANSWER_POS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (Codex-specified defaults = config P0)
TAU_RANK = 0.5
LAMBDA_RANK = 0.25
LAMBDA_INV = 0.20
LAMBDA_CF_AUG = 0.25
LAMBDA_CF_REL = 0.35
CF_MARGIN = 0.25
LR_PEAK = 1e-3
LR_MIN = 1e-4
LR_WARMUP_STEPS = 500
N_STEPS = 12_000
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
ANSWER_CE_WEIGHT = 5.0
EVAL_EXAMPLES = 2_000
CHECKPOINT_STEPS = [4_000, 6_000, 8_000, 12_000]
SEEDS = [0, 1, 2, 3, 4]
EPS = 1e-6

T_PRESERVE = ["swap", "change_irrelevant", "rename_other"]
T_CF = ["query_other_entity", "change_query_slot"]

WORD_BYTES = {}
for w in ALL_WORDS:
    bs = list(w.encode("ascii"))
    while len(bs) < PATCH_SIZE:
        bs.append(0)
    WORD_BYTES[w] = bs[:PATCH_SIZE]


# --- Data structures ---

@dataclass
class CFResult:
    tokens: list
    correct: str
    distractors: list
    query_attr: str
    query_person: int
    transform_type: str
    is_noop: bool


# --- Data generation ---

def generate_binding_example(rng):
    names = rng.sample(NAMES, 2)
    colors = rng.sample(COLORS, 2)
    rooms = rng.sample(ROOMS, 2)
    actions = [rng.choice(ACTIONS) for _ in range(2)]

    tokens = ["strt"]
    for i in range(2):
        tokens.extend([names[i], actions[i], colors[i], rooms[i]])
    query_attr = rng.choice(ATTRS)
    query_person = rng.choice([0, 1])
    tokens.extend(["qury", names[query_person], query_attr, "answ"])

    pool = ATTR_POOLS[query_attr]
    correct = {"colr": colors, "room": rooms, "actn": actions}[query_attr][query_person]
    wrongs = [v for v in pool if v != correct]
    distractors = rng.sample(wrongs, min(3, len(wrongs)))
    while len(distractors) < 3:
        distractors.append(rng.choice(wrongs))

    meta = {
        "names": names, "colors": colors, "rooms": rooms,
        "actions": actions, "query_attr": query_attr,
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
        new_names = [names[1], names[0]]
        new_colors = [colors[1], colors[0]]
        new_rooms = [rooms[1], rooms[0]]
        new_actions = [actions[1], actions[0]]
        new_qp = 1 - qp
        new_tokens = ["strt"]
        new_tokens.extend([new_names[0], new_actions[0], new_colors[0], new_rooms[0]])
        new_tokens.extend([new_names[1], new_actions[1], new_colors[1], new_rooms[1]])
        new_tokens.extend(["qury", new_names[new_qp], qa, "answ"])
        new_meta = {
            "names": new_names, "colors": new_colors, "rooms": new_rooms,
            "actions": new_actions, "query_attr": qa, "query_person": new_qp,
        }
        return new_tokens, new_meta
    elif transform_type == "change_irrelevant":
        other = 1 - qp
        new_colors, new_rooms, new_actions = list(colors), list(rooms), list(actions)
        if qa == "colr":
            avail = [c for c in COLORS if c not in colors]
            if avail:
                new_colors[other] = rng.choice(avail)
        elif qa == "room":
            avail = [r for r in ROOMS if r not in rooms]
            if avail:
                new_rooms[other] = rng.choice(avail)
        else:
            new_actions[other] = rng.choice(ACTIONS)
        new_tokens = ["strt"]
        for i in range(2):
            new_tokens.extend([names[i], new_actions[i], new_colors[i], new_rooms[i]])
        new_tokens.extend(["qury", names[qp], qa, "answ"])
        new_meta = {
            "names": list(names), "colors": new_colors, "rooms": new_rooms,
            "actions": new_actions, "query_attr": qa, "query_person": qp,
        }
        return new_tokens, new_meta
    elif transform_type == "rename_other":
        other = 1 - qp
        avail = [n for n in NAMES if n not in names]
        if not avail:
            new_meta = dict(meta)
            return list(tokens), new_meta
        new_names = list(names)
        new_names[other] = rng.choice(avail)
        new_tokens = ["strt"]
        for i in range(2):
            new_tokens.extend([new_names[i], actions[i], colors[i], rooms[i]])
        new_tokens.extend(["qury", new_names[qp], qa, "answ"])
        new_meta = {
            "names": new_names, "colors": list(colors), "rooms": list(rooms),
            "actions": list(actions), "query_attr": qa, "query_person": qp,
        }
        return new_tokens, new_meta
    new_meta = dict(meta)
    return list(tokens), new_meta


def apply_counterfactual_transform_v2(tokens, meta, rng, cf_type, reject_noop=True):
    names = meta["names"]
    colors = meta["colors"]
    rooms = meta["rooms"]
    actions = meta["actions"]
    qp = meta["query_person"]
    qa = meta["query_attr"]
    orig_correct = {"colr": colors, "room": rooms, "actn": actions}[qa][qp]

    for attempt in range(10 if reject_noop else 1):
        if cf_type == "query_other_entity":
            other = 1 - qp
            new_tokens = ["strt"]
            for i in range(2):
                new_tokens.extend([names[i], actions[i], colors[i], rooms[i]])
            new_tokens.extend(["qury", names[other], qa, "answ"])
            cf_correct = {"colr": colors, "room": rooms, "actn": actions}[qa][other]
            cf_qa = qa
            cf_qp = other
        elif cf_type == "change_query_slot":
            other_attrs = [a for a in ATTRS if a != qa]
            cf_qa = rng.choice(other_attrs)
            new_tokens = ["strt"]
            for i in range(2):
                new_tokens.extend([names[i], actions[i], colors[i], rooms[i]])
            new_tokens.extend(["qury", names[qp], cf_qa, "answ"])
            cf_correct = {"colr": colors, "room": rooms, "actn": actions}[cf_qa][qp]
            cf_qp = qp
        else:
            raise ValueError(f"Unknown CF type: {cf_type}")

        is_noop = (cf_correct == orig_correct)
        if not is_noop or not reject_noop:
            break

    pool = ATTR_POOLS[cf_qa]
    cf_wrongs = [v for v in pool if v != cf_correct]
    cf_distractors = rng.sample(cf_wrongs, min(3, len(cf_wrongs)))
    while len(cf_distractors) < 3:
        cf_distractors.append(rng.choice(cf_wrongs))

    return CFResult(
        tokens=new_tokens, correct=cf_correct, distractors=cf_distractors,
        query_attr=cf_qa, query_person=cf_qp,
        transform_type=cf_type, is_noop=is_noop,
    )


def apply_composite_cf(tokens, meta, rng, composite_type):
    if composite_type == "query_other_then_change_slot":
        cf1 = apply_counterfactual_transform_v2(tokens, meta, rng, "query_other_entity", reject_noop=False)
        meta2 = dict(meta)
        meta2["query_person"] = cf1.query_person
        meta2["query_attr"] = cf1.query_attr
        cf2 = apply_counterfactual_transform_v2(cf1.tokens, meta2, rng, "change_query_slot", reject_noop=False)
        return cf2
    elif composite_type == "change_slot_then_query_other":
        cf1 = apply_counterfactual_transform_v2(tokens, meta, rng, "change_query_slot", reject_noop=False)
        meta2 = dict(meta)
        meta2["query_attr"] = cf1.query_attr
        cf2 = apply_counterfactual_transform_v2(cf1.tokens, meta2, rng, "query_other_entity", reject_noop=False)
        return cf2
    elif composite_type == "preserve_then_query_other":
        t_type = rng.choice(T_PRESERVE)
        pres_tokens, pres_meta = apply_preserving_transform(tokens, meta, rng, t_type)
        cf = apply_counterfactual_transform_v2(pres_tokens, pres_meta, rng, "query_other_entity", reject_noop=False)
        return cf
    elif composite_type == "preserve_then_change_slot":
        t_type = rng.choice(T_PRESERVE)
        pres_tokens, pres_meta = apply_preserving_transform(tokens, meta, rng, t_type)
        cf = apply_counterfactual_transform_v2(pres_tokens, pres_meta, rng, "change_query_slot", reject_noop=False)
        return cf
    raise ValueError(f"Unknown composite CF: {composite_type}")


# --- Candidate builders ---

def make_same_attr_candidates(correct, query_attr, rng, shuffle=True):
    pool = list(ATTR_POOLS[query_attr])
    assert correct in pool
    if shuffle:
        rng.shuffle(pool)
    gold_idx = pool.index(correct)
    return pool, gold_idx


def make_relational_candidates(orig_correct, cf_correct, orig_attr, cf_attr, rng):
    candidates = list(ATTR_POOLS[orig_attr])
    if cf_attr != orig_attr:
        for v in ATTR_POOLS[cf_attr]:
            if v not in candidates:
                candidates.append(v)
    assert orig_correct in candidates
    assert cf_correct in candidates
    rng.shuffle(candidates)
    return candidates, candidates.index(orig_correct), candidates.index(cf_correct)


# --- Models (same as OG-1) ---

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
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.d_model = d_model

    def forward(self, token_ids):
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device).unsqueeze(0)
        x = self.emb(token_ids) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=token_ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.norm(x))

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
                                       dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.byte_decoder = nn.Linear(d_model, BYTE_VOCAB * patch_size, bias=False)
        self.d_model = d_model

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        P = self.patch_size
        N = T // P
        x = self.byte_emb(byte_ids).reshape(B, N, P * self.d_model)
        patch_states = self.patch_proj(x) + self.pos_emb(
            torch.arange(N, device=byte_ids.device).unsqueeze(0))
        mask = nn.Transformer.generate_square_subsequent_mask(N, device=byte_ids.device)
        h = patch_states
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=True)
        hidden = self.norm(h)
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


# --- Utility functions ---

def center(scores):
    return scores - scores.mean()


def zcenter(scores):
    c = center(scores)
    return c / (c.std(unbiased=False) + EPS)


def gold_margins(scores, gold_idx):
    gold = scores[gold_idx]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[gold_idx] = False
    return gold - scores[mask]


def entropy_from_scores(scores, tau):
    p = F.softmax(scores / tau, dim=-1)
    return -(p * (p + EPS).log()).sum().item()


def lr_at_step(step):
    if step < LR_WARMUP_STEPS:
        return LR_PEAK * (step + 1) / LR_WARMUP_STEPS
    progress = (step - LR_WARMUP_STEPS) / max(1, N_STEPS - LR_WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LR_PEAK - LR_MIN) * cosine


# --- Loss functions ---

def compute_ce_loss(student, tokens_full):
    byte_seq = tokens_to_bytes_seq(tokens_full)
    byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)
    logits = student(byte_t)
    N = logits.shape[1]
    P = student.patch_size
    target_bytes = byte_t.reshape(1, N, P)
    targets_shifted = target_bytes[:, 1:]
    preds = logits[:, :-1]
    ce_per_byte = F.cross_entropy(preds.reshape(-1, BYTE_VOCAB),
                                   targets_shifted.reshape(-1),
                                   reduction="none").reshape(N - 1, P)
    ce_weights = torch.ones(N - 1, device=DEVICE)
    ce_weights[min(ANSWER_POS, N - 2)] = ANSWER_CE_WEIGHT
    return (ce_per_byte * ce_weights.unsqueeze(1)).sum() / (ce_weights.sum() * P)


def compute_ranking_loss_v2(student, context_tokens, candidates, gold_idx, tau=TAU_RANK):
    scores = student.score_candidates_batch(context_tokens, candidates)
    target = torch.tensor([gold_idx], dtype=torch.long, device=DEVICE)
    return F.cross_entropy((scores / tau).unsqueeze(0), target), scores


def compute_invariance_loss_v2(student, orig_tokens, trans_tokens,
                                candidates, gold_idx, tau=TAU_RANK):
    L_pres_rank, scores_trans = compute_ranking_loss_v2(
        student, trans_tokens, candidates, gold_idx, tau)
    scores_orig = student.score_candidates_batch(orig_tokens, candidates)
    L_margin = F.smooth_l1_loss(gold_margins(scores_orig, gold_idx),
                                 gold_margins(scores_trans, gold_idx))
    L_vec = F.smooth_l1_loss(zcenter(scores_orig), zcenter(scores_trans))
    L_inv = 0.50 * L_pres_rank + 0.25 * L_margin + 0.25 * L_vec
    return L_inv, {
        "inv_pres_rank": L_pres_rank.item(),
        "inv_margin": L_margin.item(),
        "inv_vec": L_vec.item(),
    }


def compute_relational_cf_loss(student, orig_tokens, cf_tokens, cf_candidates,
                                cf_gold_idx, rel_candidates, orig_idx, cf_idx,
                                tau=TAU_RANK, margin=CF_MARGIN):
    L_cf_rank, _ = compute_ranking_loss_v2(student, cf_tokens, cf_candidates, cf_gold_idx, tau)
    s_orig = student.score_candidates_batch(orig_tokens, rel_candidates)
    s_cf = student.score_candidates_batch(cf_tokens, rel_candidates)
    orig_pref = s_orig[orig_idx] - s_orig[cf_idx]
    cf_pref = s_cf[cf_idx] - s_cf[orig_idx]
    L_reversal = (F.softplus(margin - orig_pref) + F.softplus(margin - cf_pref)) / 2
    delta_cf_correct = s_cf[cf_idx] - s_orig[cf_idx]
    delta_orig_correct = s_cf[orig_idx] - s_orig[orig_idx]
    L_delta = F.softplus(margin - (delta_cf_correct - delta_orig_correct))
    L_cf_rel = 0.50 * L_cf_rank + 0.25 * L_reversal + 0.25 * L_delta
    return L_cf_rel, {
        "cf_rank": L_cf_rank.item(),
        "cf_reversal": L_reversal.item(),
        "cf_delta": L_delta.item(),
        "orig_pref": orig_pref.item(),
        "cf_pref": cf_pref.item(),
    }


# --- Training ---

def train_teacher(teacher, n_epochs=500, lr=1e-3, batch_size=32):
    optimizer = torch.optim.Adam(teacher.parameters(), lr=lr)
    rng = random.Random(42)
    batches_per_epoch = 20
    for epoch in range(n_epochs):
        total_loss = 0.0
        for _ in range(batches_per_epoch):
            batch_ids = []
            for _ in range(batch_size):
                tokens, correct, _, _ = generate_binding_example(rng)
                batch_ids.append(tokens_to_ids(tokens + [correct]))
            t = torch.tensor(batch_ids, dtype=torch.long, device=DEVICE)
            logits = teacher(t)
            targets = t[:, 1:]
            B, S = targets.shape
            loss_per_pos = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                targets.reshape(-1), reduction="none").reshape(B, S)
            weights = torch.ones(S, device=DEVICE)
            weights[ANSWER_POS] = ANSWER_CE_WEIGHT
            loss = (loss_per_pos * weights.unsqueeze(0)).sum() / (B * weights.sum())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f"  Teacher epoch {epoch+1}: loss={total_loss/batches_per_epoch:.4f}")


VARIANTS = [
    "A_ce", "A_more_ce", "A_cf_ce", "B_rank", "C_inv_fixed",
    "D_aug_cf", "D_rel_full", "E_adv", "F_rand_inv",
]

VARIANT_LABELS = {
    "A_ce": "CE only (baseline)",
    "A_more_ce": "CE + extra data (control)",
    "A_cf_ce": "CE + CF augmentation (key baseline)",
    "B_rank": "CE + ranking",
    "C_inv_fixed": "CE + ranking + fixed invariance",
    "D_aug_cf": "CE + ranking + CF rank (OG-1 style)",
    "D_rel_full": "CE + ranking + invariance + relational CF (MAIN)",
    "E_adv": "adversarial labels (control)",
    "F_rand_inv": "random-context invariance (control)",
}


def train_student(student, variant, data_seed):
    optimizer = torch.optim.AdamW(student.parameters(), lr=LR_PEAK, weight_decay=0.01)
    rng = random.Random(data_seed)
    rng_unrelated = random.Random(data_seed + 9999)

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
        log_rank = log_inv = log_cf = 0.0

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
            log_rank = L_rank.item()
            loss = loss + LAMBDA_RANK * L_rank

        if uses_inv and step >= WARMUP_STEPS:
            t_type = rng.choice(T_PRESERVE)
            if variant == "F_rand_inv":
                unrel_tokens, _, _, unrel_meta = generate_binding_example(rng_unrelated)
                trans_tokens, _ = apply_preserving_transform(unrel_tokens, unrel_meta, rng, t_type)
            else:
                trans_tokens, _ = apply_preserving_transform(tokens, meta, rng, t_type)
            L_inv, _ = compute_invariance_loss_v2(
                student, tokens, trans_tokens, candidates, gold_idx)
            log_inv = L_inv.item()
            loss = loss + LAMBDA_INV * L_inv

        if (uses_cf_aug or uses_cf_rel) and step >= WARMUP_STEPS:
            cf_type = rng.choice(T_CF)
            cf = apply_counterfactual_transform_v2(tokens, meta, rng, cf_type)

            if not cf.is_noop:
                cf_candidates, cf_gold_idx = make_same_attr_candidates(
                    cf.correct, cf.query_attr, rng)

            if not cf.is_noop and uses_cf_aug:
                L_cf, _ = compute_ranking_loss_v2(
                    student, cf.tokens, cf_candidates, cf_gold_idx)
                log_cf = L_cf.item()
                loss = loss + LAMBDA_CF_AUG * L_cf

            if not cf.is_noop and uses_cf_rel:
                rel_candidates, orig_ri, cf_ri = make_relational_candidates(
                    correct, cf.correct, meta["query_attr"], cf.query_attr, rng)
                if variant == "E_adv":
                    wrong_cf_pool = [c for c in cf_candidates if c != cf.correct]
                    fake_cf_correct = rng.choice(wrong_cf_pool)
                    adv_cf_gold = cf_candidates.index(fake_cf_correct)
                    L_cf_rel, _ = compute_relational_cf_loss(
                        student, tokens, cf.tokens, cf_candidates, adv_cf_gold,
                        rel_candidates, cf_ri, orig_ri)
                else:
                    L_cf_rel, _ = compute_relational_cf_loss(
                        student, tokens, cf.tokens, cf_candidates, cf_gold_idx,
                        rel_candidates, orig_ri, cf_ri)
                log_cf = L_cf_rel.item()
                loss = loss + LAMBDA_CF_REL * L_cf_rel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), GRAD_CLIP)
        optimizer.step()

        if (step + 1) in CHECKPOINT_STEPS:
            print(f"    [{variant}] step {step+1}: CE={L_ce.item():.4f} "
                  f"rank={log_rank:.4f} inv={log_inv:.4f} cf={log_cf:.4f}")


# --- Evaluation ---

def evaluate_mcq(student, teacher, n_examples=EVAL_EXAMPLES, seed=999):
    rng = random.Random(seed)
    correct_count = 0
    teacher_agree = 0
    total_bpb = 0.0
    n_bpb = 0

    for _ in range(n_examples):
        tokens, correct, distractors, meta = generate_binding_example(rng)
        candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)
        with torch.no_grad():
            scores = student.score_candidates_batch(tokens, candidates)
        student_pick = candidates[scores.argmax().item()]
        if student_pick == correct:
            correct_count += 1

        context_ids = tokens_to_ids(tokens)
        teacher_scores = [teacher.score_continuation(context_ids, WORD2ID[c]) for c in candidates]
        if student_pick == candidates[np.argmax(teacher_scores)]:
            teacher_agree += 1

        byte_seq = tokens_to_bytes_seq(tokens + [correct])
        byte_t = torch.tensor([byte_seq], dtype=torch.long, device=DEVICE)
        P = student.patch_size
        with torch.no_grad():
            logits = student(byte_t)
        N = logits.shape[1]
        byte_targets = torch.tensor(byte_seq, dtype=torch.long, device=DEVICE).reshape(N, P)
        for i in range(1, N):
            for bp in range(P):
                total_bpb -= F.log_softmax(logits[0, i-1, bp], dim=-1)[byte_targets[i, bp]].item()
                n_bpb += 1

    return {
        "mcq": correct_count / n_examples,
        "t_agree": teacher_agree / n_examples,
        "bpb": (total_bpb / n_bpb) / math.log(2) if n_bpb > 0 else float("inf"),
    }


def evaluate_transformed_mcq(student, n_examples=EVAL_EXAMPLES, seed=888):
    rng = random.Random(seed)
    results = {k: 0 for k in ["clean", "swap", "irrel", "rename"]}

    for _ in range(n_examples):
        tokens, correct, _, meta = generate_binding_example(rng)
        candidates, _ = make_same_attr_candidates(correct, meta["query_attr"], rng)

        def score_ctx(ctx):
            with torch.no_grad():
                s = student.score_candidates_batch(ctx, candidates)
            return candidates[s.argmax().item()] == correct

        results["clean"] += int(score_ctx(tokens))
        results["swap"] += int(score_ctx(apply_preserving_transform(tokens, meta, rng, "swap")[0]))
        results["irrel"] += int(score_ctx(apply_preserving_transform(tokens, meta, rng, "change_irrelevant")[0]))
        results["rename"] += int(score_ctx(apply_preserving_transform(tokens, meta, rng, "rename_other")[0]))

    out = {k: v / n_examples for k, v in results.items()}
    out["avg_trans"] = sum(out[k] for k in ["swap", "irrel", "rename"]) / 3
    return out


def evaluate_preserving_geometry(student, n_examples=EVAL_EXAMPLES, seed=777):
    rng = random.Random(seed)
    agree = 0
    gold_agree = 0
    margin_deltas = []
    vec_l2s = []
    total = 0

    for _ in range(n_examples):
        tokens, correct, _, meta = generate_binding_example(rng)
        candidates, gold_idx = make_same_attr_candidates(correct, meta["query_attr"], rng)
        t_type = rng.choice(T_PRESERVE)
        trans, _ = apply_preserving_transform(tokens, meta, rng, t_type)

        with torch.no_grad():
            s_orig = student.score_candidates_batch(tokens, candidates)
            s_trans = student.score_candidates_batch(trans, candidates)

        orig_pick = s_orig.argmax().item()
        trans_pick = s_trans.argmax().item()
        agree += int(orig_pick == trans_pick)
        gold_agree += int(orig_pick == gold_idx and trans_pick == gold_idx)

        m_orig = gold_margins(s_orig, gold_idx)
        m_trans = gold_margins(s_trans, gold_idx)
        margin_deltas.append((m_orig - m_trans).abs().mean().item())
        vec_l2s.append((zcenter(s_orig) - zcenter(s_trans)).pow(2).sum().sqrt().item())
        total += 1

    return {
        "preserve_agree": agree / total,
        "preserve_gold_agree": gold_agree / total,
        "preserve_margin_delta": np.mean(margin_deltas),
        "preserve_vec_l2": np.mean(vec_l2s),
    }


def evaluate_cf_direction(student, n_examples=EVAL_EXAMPLES, seed=666, heldout=False):
    rng = random.Random(seed)
    cf_acc = 0
    dir_acc = 0
    delta_acc = 0
    reversal_margins = []
    delta_margins = []
    noop_count = 0
    total = 0

    cf_types = T_CF if not heldout else [
        "query_other_then_change_slot", "change_slot_then_query_other",
        "preserve_then_query_other", "preserve_then_change_slot",
    ]

    for _ in range(n_examples):
        tokens, correct, _, meta = generate_binding_example(rng)
        cf_type = rng.choice(cf_types)

        if heldout:
            cf = apply_composite_cf(tokens, meta, rng, cf_type)
        else:
            cf = apply_counterfactual_transform_v2(tokens, meta, rng, cf_type, reject_noop=False)

        if cf.is_noop:
            noop_count += 1
            continue

        cf_candidates, cf_gold = make_same_attr_candidates(cf.correct, cf.query_attr, rng)
        rel_candidates, orig_ri, cf_ri = make_relational_candidates(
            correct, cf.correct, meta["query_attr"], cf.query_attr, rng)

        with torch.no_grad():
            s_cf = student.score_candidates_batch(cf.tokens, cf_candidates)
            s_orig_rel = student.score_candidates_batch(tokens, rel_candidates)
            s_cf_rel = student.score_candidates_batch(cf.tokens, rel_candidates)

        cf_acc += int(cf_candidates[s_cf.argmax().item()] == cf.correct)
        orig_pref = (s_orig_rel[orig_ri] - s_orig_rel[cf_ri]).item()
        cf_pref = (s_cf_rel[cf_ri] - s_cf_rel[orig_ri]).item()
        dir_acc += int(orig_pref > 0 and cf_pref > 0)
        reversal_margins.append(min(orig_pref, cf_pref))
        delta = (s_cf_rel[cf_ri] - s_orig_rel[cf_ri]) - (s_cf_rel[orig_ri] - s_orig_rel[orig_ri])
        delta_acc += int(delta.item() > 0)
        delta_margins.append(delta.item())
        total += 1

    if total == 0:
        return {"cf_acc": 0, "cf_dir_acc": 0, "cf_delta_acc": 0,
                "cf_rev_margin": 0, "cf_delta_margin": 0,
                "cf_noop_rate": 0, "n_valid": 0}
    return {
        "cf_acc": cf_acc / total,
        "cf_dir_acc": dir_acc / total,
        "cf_delta_acc": delta_acc / total,
        "cf_rev_margin": np.mean(reversal_margins),
        "cf_delta_margin": np.mean(delta_margins),
        "cf_noop_rate": noop_count / (noop_count + total),
        "n_valid": total,
    }


# --- Main ---

def main():
    print(f"OG-1b: Operational Geometry — Augmentation vs Relational Geometry")
    print(f"Device: {DEVICE}")
    print(f"Variants: {', '.join(VARIANTS)}")
    print(f"{N_STEPS} steps, {len(SEEDS)} seeds, cosine LR {LR_PEAK}->{LR_MIN}")
    print(f"Checkpoints at: {CHECKPOINT_STEPS}")
    print()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("=== Training Teacher ===")
    teacher = ToyTeacher(d_model=128, n_layers=4, n_heads=4).to(DEVICE)
    train_teacher(teacher)
    teacher.eval()

    rng = random.Random(999)
    t_correct = 0
    for _ in range(500):
        tokens, correct, _, meta = generate_binding_example(rng)
        candidates, _ = make_same_attr_candidates(correct, meta["query_attr"], rng)
        scores = [teacher.score_continuation(tokens_to_ids(tokens), WORD2ID[c])
                  for c in candidates]
        if candidates[np.argmax(scores)] == correct:
            t_correct += 1
    teacher_mcq = t_correct / 500
    print(f"\nTeacher MCQ: {teacher_mcq*100:.1f}%")
    assert teacher_mcq >= 0.95, f"Teacher MCQ {teacher_mcq*100:.1f}% < 95%"

    all_results = {v: [] for v in VARIANTS}
    all_trans = {v: [] for v in VARIANTS}
    all_pres = {v: [] for v in VARIANTS}
    all_cf = {v: [] for v in VARIANTS}
    all_hcf = {v: [] for v in VARIANTS}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"SEED {seed_idx+1}/{len(SEEDS)} (seed={seed})")
        print(f"{'='*70}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        base_student = ToyByteStudent(d_model=64, n_layers=4, n_heads=4, patch_size=PATCH_SIZE).to(DEVICE)
        base_state = copy.deepcopy(base_student.state_dict())
        data_seed = seed * 1000 + 123

        for v in VARIANTS:
            print(f"\n  --- [{v}] {VARIANT_LABELS[v]} (seed={seed}) ---")
            student = ToyByteStudent(d_model=64, n_layers=4, n_heads=4, patch_size=PATCH_SIZE).to(DEVICE)
            student.load_state_dict(copy.deepcopy(base_state))
            torch.manual_seed(seed + abs(hash(v)) % 10000)
            train_student(student, v, data_seed)
            student.eval()

            r = evaluate_mcq(student, teacher)
            all_results[v].append(r)
            print(f"  [{v}] MCQ={r['mcq']*100:.1f}% T-Agree={r['t_agree']*100:.1f}% BPB={r['bpb']:.3f}")

            tr = evaluate_transformed_mcq(student)
            all_trans[v].append(tr)
            print(f"  [{v}] Trans: clean={tr['clean']*100:.1f}% swap={tr['swap']*100:.1f}% "
                  f"irrel={tr['irrel']*100:.1f}% rename={tr['rename']*100:.1f}% avg={tr['avg_trans']*100:.1f}%")

            pr = evaluate_preserving_geometry(student)
            all_pres[v].append(pr)
            print(f"  [{v}] Pres: agree={pr['preserve_agree']*100:.1f}% "
                  f"gold_agree={pr['preserve_gold_agree']*100:.1f}% "
                  f"margin_d={pr['preserve_margin_delta']:.4f} vec_l2={pr['preserve_vec_l2']:.4f}")

            cf_r = evaluate_cf_direction(student)
            all_cf[v].append(cf_r)
            print(f"  [{v}] CF: acc={cf_r['cf_acc']*100:.1f}% dir={cf_r['cf_dir_acc']*100:.1f}% "
                  f"delta={cf_r['cf_delta_acc']*100:.1f}%")

            hcf = evaluate_cf_direction(student, heldout=True)
            all_hcf[v].append(hcf)
            print(f"  [{v}] HeldoutCF: acc={hcf['cf_acc']*100:.1f}% dir={hcf['cf_dir_acc']*100:.1f}% "
                  f"delta={hcf['cf_delta_acc']*100:.1f}%")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY (mean +/- std over {len(SEEDS)} seeds)")
    print(f"{'='*70}")
    print(f"{'Variant':<45} {'MCQ':>10} {'AvgTrans':>10} {'BPB':>10} {'HeldCFDir':>10}")
    print("-" * 87)

    means = {}
    for v in VARIANTS:
        mcqs = [r["mcq"] for r in all_results[v]]
        ats = [r["avg_trans"] for r in all_trans[v]]
        bpbs = [r["bpb"] for r in all_results[v]]
        hcf_dirs = [r["cf_dir_acc"] for r in all_hcf[v]]
        m = {
            "mcq": np.mean(mcqs), "s_mcq": np.std(mcqs),
            "at": np.mean(ats), "s_at": np.std(ats),
            "bpb": np.mean(bpbs), "s_bpb": np.std(bpbs),
            "hcf_dir": np.mean(hcf_dirs), "s_hcf": np.std(hcf_dirs),
        }
        means[v] = m
        print(f"[{v}] {VARIANT_LABELS[v]:<42} "
              f"{m['mcq']*100:>5.1f}+/-{m['s_mcq']*100:>4.1f}% "
              f"{m['at']*100:>5.1f}+/-{m['s_at']*100:>4.1f}% "
              f"{m['bpb']:>5.3f}+/-{m['s_bpb']:.3f} "
              f"{m['hcf_dir']*100:>5.1f}+/-{m['s_hcf']*100:>4.1f}%")

    # --- Gates ---
    print(f"\n--- OG-1b Success Gates ---")

    # Debug gates
    a_mcq = means["A_ce"]["mcq"]
    e_mcq = means["E_adv"]["mcq"]
    d_bpb = means["D_rel_full"]["bpb"]
    a_bpb = means["A_ce"]["bpb"]
    bpb_deg = ((d_bpb - a_bpb) / a_bpb) * 100

    print(f"\nDebug 1: Teacher MCQ={teacher_mcq*100:.1f}% (>=95%): {'PASS' if teacher_mcq >= 0.95 else 'FAIL'}")
    print(f"Debug 2: A_ce MCQ={a_mcq*100:.1f}% (>=50%): {'PASS' if a_mcq >= 0.50 else 'FAIL'}")
    print(f"Debug 3: E_adv MCQ={e_mcq*100:.1f}% (<=35%): {'PASS' if e_mcq <= 0.35 else 'FAIL'}")
    print(f"Debug 6: BPB degradation={bpb_deg:+.1f}% (<=5%): {'PASS' if bpb_deg <= 5 else 'FAIL'}")

    # Scientific gates
    d_hcf = means["D_rel_full"]["hcf_dir"]
    acf_hcf = means["A_cf_ce"]["hcf_dir"]
    daug_hcf = means["D_aug_cf"]["hcf_dir"]
    d_at = means["D_rel_full"]["at"]
    acf_at = means["A_cf_ce"]["at"]

    gap_primary = (d_hcf - acf_hcf) * 100
    gap_daug = (d_hcf - daug_hcf) * 100
    gap_at = (d_at - acf_at) * 100

    print(f"\nPRIMARY: D_rel vs A_cf_ce heldout_cf_dir: {d_hcf*100:.1f}% vs {acf_hcf*100:.1f}% "
          f"gap={gap_primary:+.1f}pp (>=+2pp): {'PASS' if gap_primary >= 2 else 'FAIL'}")
    print(f"Sec 1: D_rel vs D_aug heldout_cf_dir: {d_hcf*100:.1f}% vs {daug_hcf*100:.1f}% "
          f"gap={gap_daug:+.1f}pp (>=+2pp): {'PASS' if gap_daug >= 2 else 'FAIL'}")
    print(f"Sec 2: D_rel vs A_cf_ce avg_trans: {d_at*100:.1f}% vs {acf_at*100:.1f}% "
          f"gap={gap_at:+.1f}pp (>=+2pp): {'PASS' if gap_at >= 2 else 'FAIL'}")

    # Preserving geometry comparison
    d_pres = np.mean([r["preserve_agree"] for r in all_pres["D_rel_full"]])
    acf_pres = np.mean([r["preserve_agree"] for r in all_pres["A_cf_ce"]])
    gap_pres = (d_pres - acf_pres) * 100
    print(f"Sec 3: D_rel vs A_cf_ce preserve_agree: {d_pres*100:.1f}% vs {acf_pres*100:.1f}% "
          f"gap={gap_pres:+.1f}pp (>=+2pp): {'PASS' if gap_pres >= 2 else 'FAIL'}")

    # CF augmentation check
    acf_mcq = means["A_cf_ce"]["mcq"]
    daug_mcq = means["D_aug_cf"]["mcq"]
    gap_cf_a = (max(acf_mcq, daug_mcq) - a_mcq) * 100
    print(f"Sec 4: CF augmentation signal: best_CF={max(acf_mcq,daug_mcq)*100:.1f}% vs A={a_mcq*100:.1f}% "
          f"gap={gap_cf_a:+.1f}pp (>=+3pp): {'PASS' if gap_cf_a >= 3 else 'FAIL'}")

    b_at = means["B_rank"]["at"]
    gap_db = (d_at - b_at) * 100
    print(f"Sec 5: D_rel vs B_rank avg_trans: gap={gap_db:+.1f}pp (>=+5pp): "
          f"{'PASS' if gap_db >= 5 else 'FAIL'}")

    # Debug 5: F_rand_inv vs D_rel_full on preserving geometry
    f_pres = np.mean([r["preserve_agree"] for r in all_pres["F_rand_inv"]])
    f_margin = np.mean([r["preserve_margin_delta"] for r in all_pres["F_rand_inv"]])
    d_margin = np.mean([r["preserve_margin_delta"] for r in all_pres["D_rel_full"]])
    f_worse = (f_pres < d_pres) or (f_margin > d_margin)
    print(f"\nDebug 5: F_rand_inv vs D_rel_full preserving: "
          f"F_agree={f_pres*100:.1f}% D_agree={d_pres*100:.1f}% "
          f"F_margin_d={f_margin:.4f} D_margin_d={d_margin:.4f} "
          f"F_worse_on_any={'PASS' if f_worse else 'FAIL'}")

    all_pass = (gap_primary >= 2 and gap_daug >= 2 and gap_at >= 2
                and gap_cf_a >= 3 and gap_db >= 5 and bpb_deg <= 5
                and teacher_mcq >= 0.95 and a_mcq >= 0.50
                and e_mcq <= 0.35 and f_worse)
    print(f"\n{'='*70}")
    print(f"OVERALL: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
