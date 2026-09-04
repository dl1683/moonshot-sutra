"""Eklavya Embedding Tomography — full experiment pipeline.

Three training arms for comparison:
  A. Ranking KD (identity probe only): student matches teacher distributions
  B. Probe tomography: student matches teacher distributions under probes
  C. Contrastive baseline: standard InfoNCE with same data (control)

Soul test: retained gain after teacher removal (no teacher at inference).

Key Sangam transfers:
  - Keep teacher axis (per-teacher KL, never average distributions)
  - Ordinal/relational loss (match ranking structure, not coordinates)
  - Temperature-scaled softmax for ranking distributions

Usage:
  # Run end-to-end experiment (smoke test on toy data)
  python -m code.embed_tomography experiment --device cpu --steps 100

  # Run with GPU and more data
  python -m code.embed_tomography experiment --device cuda --steps 500 --n_pairs 200

  # Extract teacher signatures only
  python -m code.embed_tomography extract \
    --teachers sentence-transformers/all-MiniLM-L12-v2 BAAI/bge-large-en-v1.5 \
    --data data/pairs.jsonl --out data/embed_signatures.jsonl

  # Train student from pre-extracted signatures
  python -m code.embed_tomography train \
    --signatures data/embed_signatures.jsonl \
    --student sentence-transformers/all-MiniLM-L6-v2 \
    --out_dir outputs/eklavya_student --steps 500

  # Evaluate retained gain
  python -m code.embed_tomography eval \
    --model outputs/eklavya_student \
    --data data/toy_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Probes — perturbations of a query to reveal teacher behavior
# ---------------------------------------------------------------------------

PARAPHRASE_TEMPLATES = [
    "Rephrase: {q}",
    "In other words: {q}",
    "Say this differently: {q}",
]

NEGATION_TEMPLATES = [
    "Find the opposite of: {q}",
    "NOT {q}",
    "Everything except: {q}",
]

NOISE_CHARS = list("aeiou ")


@dataclass(frozen=True)
class Probe:
    probe_id: str
    text: str


def generate_probes(query: str, seed: int = 0) -> list[Probe]:
    rng = random.Random(seed)
    probes = [Probe("identity", query)]

    probes.append(Probe("paraphrase", rng.choice(PARAPHRASE_TEMPLATES).format(q=query)))

    probes.append(Probe("negation", rng.choice(NEGATION_TEMPLATES).format(q=query)))

    # Typo noise — insert a random char at a random position
    if len(query) > 5:
        pos = rng.randint(1, len(query) - 1)
        noisy = query[:pos] + rng.choice(NOISE_CHARS) + query[pos:]
        probes.append(Probe("typo", noisy))

    # Length expand — repeat with filler
    probes.append(Probe("verbose", f"I am looking for information about: {query}. Please find relevant results."))

    # Length compress — first N words
    words = query.split()
    if len(words) > 3:
        probes.append(Probe("terse", " ".join(words[:max(3, len(words) // 2)])))

    return probes


# ---------------------------------------------------------------------------
# Signature extraction — compute teacher similarities under probes
# ---------------------------------------------------------------------------

@dataclass
class EmbedSignature:
    pair_id: str
    query: str
    documents: list[str]
    probes: list[dict]  # [{probe_id, text}]
    teacher_sigs: dict[str, dict[str, list[float]]]  # teacher -> probe_id -> [sim_scores]
    gold_idx: int = 0


def load_model(model_name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model


@torch.no_grad()
def compute_teacher_signature(
    model,
    model_name: str,
    query: str,
    documents: list[str],
    probes: list[Probe],
) -> dict[str, list[float]]:
    sig = {}
    doc_embs = model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    # Batch all probes in one encode call
    probe_texts = [p.text for p in probes]
    all_q_embs = model.encode(probe_texts, convert_to_tensor=True, normalize_embeddings=True)
    all_sims = (all_q_embs @ doc_embs.T).cpu().tolist()
    for i, probe in enumerate(probes):
        sig[probe.probe_id] = all_sims[i]
    return sig


def extract_signatures(
    teacher_names: list[str],
    pairs: list[dict],
    device: str = "cpu",
    seed: int = 42,
) -> list[EmbedSignature]:
    signatures = []
    teachers = {}
    for name in teacher_names:
        print(f"Loading teacher: {name}")
        teachers[name] = load_model(name, device)

    for i, pair in enumerate(pairs):
        query = pair["query"]
        documents = pair["documents"]
        gold_idx = pair.get("gold_idx", 0)

        probes = generate_probes(query, seed=seed + i)
        teacher_sigs = {}
        for tname, tmodel in teachers.items():
            teacher_sigs[tname] = compute_teacher_signature(
                tmodel, tname, query, documents, probes
            )

        signatures.append(EmbedSignature(
            pair_id=pair.get("id", f"pair_{i}"),
            query=query,
            documents=documents,
            probes=[{"probe_id": p.probe_id, "text": p.text} for p in probes],
            teacher_sigs=teacher_sigs,
            gold_idx=gold_idx,
        ))
        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(pairs)} signatures")

    # Unload teachers
    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return signatures


def save_signatures(sigs: list[EmbedSignature], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sig in sigs:
            f.write(json.dumps(asdict(sig), ensure_ascii=False) + "\n")
    print(f"Saved {len(sigs)} signatures to {path}")


def load_signatures(path: str) -> list[EmbedSignature]:
    sigs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sigs.append(EmbedSignature(**d))
    return sigs


# ---------------------------------------------------------------------------
# Training — student matches teacher response surfaces
# ---------------------------------------------------------------------------

def sims_to_distribution(sims: list[float], tau: float = 0.05) -> torch.Tensor:
    t = torch.tensor(sims, dtype=torch.float32) / tau
    return F.softmax(t, dim=0)


# ---------------------------------------------------------------------------
# Student wrapper — gradient-enabled encoding for training
# ---------------------------------------------------------------------------

class StudentWrapper:
    """Wraps SentenceTransformer for gradient-enabled training."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.model_name = model_name

    def encode_with_grad(self, texts: list[str]) -> torch.Tensor:
        features = self.model.tokenize(texts)
        features = {k: v.to(self.device) for k, v in features.items()}
        out = self.model(features)
        return F.normalize(out["sentence_embedding"], p=2, dim=1)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        return self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True,
        )

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ---------------------------------------------------------------------------
# Training — ranking KD from teacher distributions
# ---------------------------------------------------------------------------

def train_ranking_kd(
    student: StudentWrapper,
    signatures: list[EmbedSignature],
    steps: int = 500,
    lr: float = 2e-5,
    tau: float = 0.05,
    batch_size: int = 8,
    use_probes: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Train student to match teacher ranking distributions.

    use_probes=False → identity probe only (standard ranking KD).
    use_probes=True  → all probes (tomography).
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = random.Random(seed)

    log = []
    student.train()

    for step in range(steps):
        batch = [signatures[i] for i in rng.sample(range(len(signatures)),
                 min(batch_size, len(signatures)))]

        total_loss = torch.tensor(0.0, device=student.device, requires_grad=True)
        n_terms = 0

        for sig in batch:
            if use_probes:
                probe_texts = [p["text"] for p in sig.probes]
                probe_ids = [p["probe_id"] for p in sig.probes]
            else:
                probe_texts = [sig.query]
                probe_ids = ["identity"]

            all_texts = probe_texts + sig.documents
            embeddings = student.encode_with_grad(all_texts)

            probe_embs = embeddings[: len(probe_texts)]
            doc_embs = embeddings[len(probe_texts) :]

            for teacher_name, teacher_sig in sig.teacher_sigs.items():
                for i, pid in enumerate(probe_ids):
                    if pid not in teacher_sig:
                        continue
                    teacher_dist = sims_to_distribution(teacher_sig[pid], tau)
                    student_sims = probe_embs[i] @ doc_embs.T
                    student_log_dist = F.log_softmax(student_sims / tau, dim=0)

                    kl = F.kl_div(
                        student_log_dist,
                        teacher_dist.to(student.device),
                        reduction="batchmean",
                        log_target=False,
                    )
                    total_loss = total_loss + kl
                    n_terms += 1

        if n_terms > 0:
            loss = total_loss / n_terms
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 50 == 0 or step == 0:
                entry = {"step": step + 1, "loss": loss.item(), "n_terms": n_terms}
                log.append(entry)
                print(f"  Step {step + 1}/{steps}: loss={loss.item():.4f}")

    student.eval()
    return log


def train_contrastive(
    student: StudentWrapper,
    pairs: list[dict],
    steps: int = 500,
    lr: float = 2e-5,
    tau: float = 0.05,
    batch_size: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Standard InfoNCE contrastive baseline — same data, no teacher signal."""
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = random.Random(seed)

    log = []
    student.train()

    for step in range(steps):
        batch = [pairs[i] for i in rng.sample(range(len(pairs)),
                 min(batch_size, len(pairs)))]

        queries = [p["query"] for p in batch]
        positives = [p["documents"][p.get("gold_idx", 0)] for p in batch]

        all_texts = queries + positives
        embeddings = student.encode_with_grad(all_texts)

        q_embs = embeddings[: len(queries)]
        p_embs = embeddings[len(queries) :]

        sims = q_embs @ p_embs.T / tau
        labels = torch.arange(len(queries), device=student.device)
        loss = F.cross_entropy(sims, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            entry = {"step": step + 1, "loss": loss.item()}
            log.append(entry)
            print(f"  Step {step + 1}/{steps}: loss={loss.item():.4f}")

    student.eval()
    return log


def train_augmented_contrastive(
    student: StudentWrapper,
    pairs: list[dict],
    steps: int = 500,
    lr: float = 2e-5,
    tau: float = 0.05,
    batch_size: int = 8,
    seed: int = 42,
) -> list[dict]:
    """B4 hostile baseline: contrastive with probe-augmented queries.

    Same interventions as Eklavya tomography, but used as data augmentation
    for standard contrastive training. No teacher signal matching — just
    InfoNCE with augmented queries. This IS the absorption test: if B4
    matches tomography, the probes are just data augmentation.
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)
    rng = random.Random(seed)
    log = []
    student.train()

    for step in range(steps):
        batch = [pairs[i] for i in rng.sample(range(len(pairs)),
                 min(batch_size, len(pairs)))]

        queries = []
        positives = []
        for p in batch:
            probes = generate_probes(p["query"], seed=seed + hash(p.get("id", step)) % 10000)
            probe = rng.choice(probes)
            queries.append(probe.text)
            positives.append(p["documents"][p.get("gold_idx", 0)])

        all_texts = queries + positives
        embeddings = student.encode_with_grad(all_texts)

        q_embs = embeddings[: len(queries)]
        p_embs = embeddings[len(queries) :]

        sims = q_embs @ p_embs.T / tau
        labels = torch.arange(len(queries), device=student.device)
        loss = F.cross_entropy(sims, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 50 == 0 or step == 0:
            entry = {"step": step + 1, "loss": loss.item()}
            log.append(entry)
            print(f"  Step {step + 1}/{steps}: loss={loss.item():.4f}")

    student.eval()
    return log


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_retrieval(model, pairs: list[dict], k: int = 5) -> dict:
    hits = 0
    mrr_sum = 0.0
    for pair in pairs:
        query = pair["query"]
        documents = pair["documents"]
        gold_idx = pair.get("gold_idx", 0)

        q_emb = model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
        d_embs = model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ d_embs.T).squeeze(0)
        ranked = sims.argsort(descending=True).tolist()

        if gold_idx in ranked[:k]:
            hits += 1
        rank = ranked.index(gold_idx) + 1
        mrr_sum += 1.0 / rank

    n = len(pairs)
    return {"hit_at_k": hits / n if n else 0, "mrr": mrr_sum / n if n else 0, "n": n, "k": k}


# ---------------------------------------------------------------------------
# Response Jets — behavioral fingerprints under interventions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_response_jet(
    model,
    query: str,
    documents: list[str],
    probes: list[Probe],
) -> dict[str, list[float]]:
    """Compute response jet: base ranking + first-order response to each intervention.

    Returns dict: probe_id -> similarity vector. The "identity" probe gives J0
    (base ranking). For intervention probes, the JET is Jg = rank(g(q)) - J0.
    We store raw sims; the caller computes jets.
    """
    doc_embs = model.encode(documents, convert_to_tensor=True, normalize_embeddings=True)
    jet = {}
    for probe in probes:
        q_emb = model.encode([probe.text], convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ doc_embs.T).squeeze(0).cpu().tolist()
        jet[probe.probe_id] = sims
    return jet


def compute_first_order_jets(raw_jet: dict[str, list[float]]) -> dict[str, list[float]]:
    """Compute first-order response jets: Jg = sims(g(q)) - sims(identity(q))."""
    j0 = raw_jet.get("identity")
    if j0 is None:
        return {}
    j0_t = torch.tensor(j0)
    jets = {}
    for pid, sims in raw_jet.items():
        if pid == "identity":
            continue
        jets[pid] = (torch.tensor(sims) - j0_t).tolist()
    return jets


def ranking_agreement(sims_a: list[float], sims_b: list[float]) -> float:
    """Kendall-tau-style agreement: fraction of pairs ranked the same way."""
    n = len(sims_a)
    if n < 2:
        return 1.0
    concordant = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = sims_a[i] - sims_a[j]
            diff_b = sims_b[i] - sims_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            total += 1
    return concordant / total if total > 0 else 1.0


def jet_divergence(jet_a: dict[str, list[float]], jet_b: dict[str, list[float]]) -> float:
    """Mean L2 distance between two response jets across shared probes."""
    shared = set(jet_a.keys()) & set(jet_b.keys())
    if not shared:
        return 0.0
    total = 0.0
    for pid in shared:
        a = torch.tensor(jet_a[pid])
        b = torch.tensor(jet_b[pid])
        total += (a - b).pow(2).mean().item()
    return total / len(shared)


# ---------------------------------------------------------------------------
# Diagnostic — CPU-only test for whether Eklavya has signal
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_diagnostic(
    student_name: str,
    teacher_names: list[str],
    pairs: list[dict],
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """Run the cheapest falsification test.

    Questions answered:
    1. Do teachers show different response patterns under interventions?
    2. Is there a gap between student and teacher response surfaces?
    3. Which probes create the most teacher disagreement?
    4. Does conditional support exist (some pairs helped by one teacher, others by another)?

    All inference, no training. CPU-feasible with small models.
    """
    print("=" * 60)
    print("EKLAVYA EMBEDDING DIAGNOSTIC")
    print("=" * 60)

    # Load models
    print(f"\nLoading student: {student_name}")
    student = load_model(student_name, device)
    teachers = {}
    for name in teacher_names:
        print(f"Loading teacher: {name}")
        teachers[name] = load_model(name, device)

    all_models = {"student": student, **teachers}

    # Compute response jets for all models
    print(f"\nComputing response jets for {len(pairs)} pairs...")
    model_jets = {name: [] for name in all_models}
    model_first_jets = {name: [] for name in all_models}

    for i, pair in enumerate(pairs):
        query = pair["query"]
        documents = pair["documents"]
        probes = generate_probes(query, seed=seed + i)

        for mname, model in all_models.items():
            raw_jet = compute_response_jet(model, query, documents, probes)
            model_jets[mname].append(raw_jet)
            model_first_jets[mname].append(compute_first_order_jets(raw_jet))

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(pairs)} pairs processed")

    # 1. Teacher diversity: pairwise disagreement on base rankings
    print("\n--- 1. TEACHER DIVERSITY (base rankings) ---")
    tnames = list(teachers.keys())
    base_agreements = {}
    for i in range(len(tnames)):
        for j in range(i + 1, len(tnames)):
            pair_key = f"{tnames[i].split('/')[-1]} vs {tnames[j].split('/')[-1]}"
            agreements = []
            for k in range(len(pairs)):
                j0_a = model_jets[tnames[i]][k].get("identity", [])
                j0_b = model_jets[tnames[j]][k].get("identity", [])
                if j0_a and j0_b:
                    agreements.append(ranking_agreement(j0_a, j0_b))
            base_agreements[pair_key] = sum(agreements) / len(agreements) if agreements else 0
            print(f"  {pair_key}: {base_agreements[pair_key]:.3f} ranking agreement")

    # 2. Teacher diversity: pairwise disagreement on RESPONSE JETS
    print("\n--- 2. TEACHER JET DIVERSITY (response to interventions) ---")
    jet_divergences = {}
    for i in range(len(tnames)):
        for j in range(i + 1, len(tnames)):
            pair_key = f"{tnames[i].split('/')[-1]} vs {tnames[j].split('/')[-1]}"
            divs = []
            for k in range(len(pairs)):
                d = jet_divergence(model_first_jets[tnames[i]][k], model_first_jets[tnames[j]][k])
                divs.append(d)
            jet_divergences[pair_key] = sum(divs) / len(divs) if divs else 0
            print(f"  {pair_key}: {jet_divergences[pair_key]:.6f} mean jet L2 divergence")

    # 3. Student gap: how different is student from each teacher?
    print("\n--- 3. STUDENT GAP (student vs each teacher) ---")
    student_gaps = {}
    student_jet_gaps = {}
    for tname in tnames:
        short = tname.split("/")[-1]
        base_agr = []
        jet_div = []
        for k in range(len(pairs)):
            j0_s = model_jets["student"][k].get("identity", [])
            j0_t = model_jets[tname][k].get("identity", [])
            if j0_s and j0_t:
                base_agr.append(ranking_agreement(j0_s, j0_t))
            d = jet_divergence(model_first_jets["student"][k], model_first_jets[tname][k])
            jet_div.append(d)
        student_gaps[short] = sum(base_agr) / len(base_agr) if base_agr else 0
        student_jet_gaps[short] = sum(jet_div) / len(jet_div) if jet_div else 0
        print(f"  student vs {short}: base agreement {student_gaps[short]:.3f}, jet div {student_jet_gaps[short]:.6f}")

    # 4. Probe informativeness: which probes create the most teacher disagreement?
    print("\n--- 4. PROBE INFORMATIVENESS ---")
    probe_ids = set()
    for jets in model_first_jets[tnames[0]]:
        probe_ids.update(jets.keys())

    for pid in sorted(probe_ids):
        mean_teacher_spread = []
        for k in range(len(pairs)):
            teacher_jets_for_probe = []
            for tname in tnames:
                jet_k = model_first_jets[tname][k]
                if pid in jet_k:
                    teacher_jets_for_probe.append(torch.tensor(jet_k[pid]))
            if len(teacher_jets_for_probe) >= 2:
                stacked = torch.stack(teacher_jets_for_probe)
                spread = stacked.std(dim=0).mean().item()
                mean_teacher_spread.append(spread)
        avg_spread = sum(mean_teacher_spread) / len(mean_teacher_spread) if mean_teacher_spread else 0
        print(f"  {pid:15s}: teacher spread = {avg_spread:.6f}")

    # 5. Conditional support: per-pair, which teacher's ranking is best?
    print("\n--- 5. CONDITIONAL SUPPORT ---")
    best_teacher_counts = {tname.split("/")[-1]: 0 for tname in tnames}
    best_teacher_counts["student"] = 0
    ties = 0

    for k, pair in enumerate(pairs):
        gold_idx = pair.get("gold_idx", 0)
        best_rank = len(pair["documents"]) + 1
        best_model = None
        for mname in list(tnames) + ["student"]:
            sims = model_jets[mname][k].get("identity", [])
            if not sims:
                continue
            ranked = sorted(range(len(sims)), key=lambda x: sims[x], reverse=True)
            rank = ranked.index(gold_idx) + 1
            if rank < best_rank:
                best_rank = rank
                best_model = mname.split("/")[-1] if "/" in mname else mname
            elif rank == best_rank and best_model:
                best_model = "tie"
        if best_model == "tie":
            ties += 1
        elif best_model:
            best_teacher_counts[best_model] = best_teacher_counts.get(best_model, 0) + 1

    print(f"  Best-ranking model per pair (out of {len(pairs)}):")
    for mname, count in sorted(best_teacher_counts.items(), key=lambda x: -x[1]):
        print(f"    {mname:30s}: {count:3d} ({100*count/len(pairs):.1f}%)")
    print(f"    {'ties':30s}: {ties:3d} ({100*ties/len(pairs):.1f}%)")

    # 6. Verdict
    print("\n" + "=" * 60)
    print("DIAGNOSTIC VERDICT")
    print("=" * 60)

    mean_teacher_diversity = sum(jet_divergences.values()) / len(jet_divergences) if jet_divergences else 0
    mean_student_gap = sum(student_jet_gaps.values()) / len(student_jet_gaps) if student_jet_gaps else 0

    if mean_teacher_diversity < 1e-6:
        print("DEAD: Teachers show no diversity under interventions.")
        print("       Probes don't reveal teacher-specific knowledge.")
    elif mean_student_gap < mean_teacher_diversity * 0.1:
        print("WEAK: Student already matches teacher response surfaces.")
        print("      Eklavya has little to teach.")
    else:
        print("ALIVE: Teachers show diverse response jets AND student has a gap.")
        print(f"       Teacher jet diversity: {mean_teacher_diversity:.6f}")
        print(f"       Student gap:           {mean_student_gap:.6f}")
        print(f"       Gap/diversity ratio:   {mean_student_gap/mean_teacher_diversity:.2f}")

    # Cleanup
    del student, teachers, all_models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "teacher_base_agreements": base_agreements,
        "teacher_jet_divergences": jet_divergences,
        "student_base_gaps": student_gaps,
        "student_jet_gaps": student_jet_gaps,
        "best_teacher_counts": best_teacher_counts,
        "ties": ties,
        "mean_teacher_diversity": mean_teacher_diversity,
        "mean_student_gap": mean_student_gap,
    }
    return results


def eval_student(student: StudentWrapper, pairs: list[dict], k: int = 5) -> dict:
    student.eval()
    return eval_retrieval(student.model, pairs, k=k)


# ---------------------------------------------------------------------------
# Q3 Cache-Only Routing Audit — cheapest conditional-support falsification
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_routing_audit(
    student_name: str,
    teacher_names: list[str],
    n_queries: int = 500,
    device: str = "cpu",
    seed: int = 42,
    n_folds: int = 5,
) -> dict:
    """Test whether conditional teacher support is learnable from student features.

    Pure inference + lightweight sklearn. No embedding training.
    Precommit: >=25% oracle-uplift capture, positive bootstrap CI.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold

    print("=" * 60)
    print("Q3: CACHE-ONLY ROUTING AUDIT")
    print("=" * 60)

    pairs = load_msmarco_pairs(n=n_queries, seed=seed)
    n_actual = len(pairs)
    print(f"Loaded {n_actual} MSMARCO pairs")

    print(f"\nLoading student: {student_name}")
    student = load_model(student_name, device)
    teachers = {}
    for name in teacher_names:
        print(f"Loading teacher: {name}")
        teachers[name] = load_model(name, device)

    all_models = {"student": student, **teachers}
    model_names = list(all_models.keys())

    # Cache margins: m = sim(q, d+) - max(sim(q, d-)) over all negatives
    print(f"\nCaching margins for {n_actual} queries...")
    margins = {mn: [] for mn in model_names}

    for i, pair in enumerate(pairs):
        q = pair["query"]
        docs = pair["documents"]
        gold = pair["gold_idx"]

        for mn, model in all_models.items():
            q_emb = model.encode([q], convert_to_tensor=True, normalize_embeddings=True)
            d_embs = model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
            sims = (q_emb @ d_embs.T).squeeze(0)
            if sims.dim() == 0:
                margins[mn].append(0.0)
            else:
                neg_mask = torch.ones(len(docs), dtype=torch.bool)
                neg_mask[gold] = False
                hardest_neg_sim = sims[neg_mask].max().item()
                m = sims[gold].item() - hardest_neg_sim
                margins[mn].append(m)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_actual}")

    margins_np = {mn: np.array(v) for mn, v in margins.items()}

    # Per-query utility: u_{i,t} = teacher_margin - student_margin
    print("\n--- Per-query utility ---")
    utilities = {}
    for tname in teacher_names:
        short = tname.split("/")[-1]
        u = margins_np[tname] - margins_np["student"]
        utilities[short] = u
        pos = (u > 0).sum()
        neg_count = (u < 0).sum()
        print(f"  {short}: helps {pos}/{n_actual} ({100*pos/n_actual:.1f}%), hurts {neg_count}/{n_actual} ({100*neg_count/n_actual:.1f}%), mean utility {u.mean():.4f}")

    # Oracle routing: pick the model with best margin per query
    all_margins = np.stack([margins_np["student"]] + [margins_np[tn] for tn in teacher_names], axis=1)
    oracle_labels = np.argmax(all_margins, axis=1)
    oracle_margins = np.max(all_margins, axis=1)

    # Class distribution
    label_names = ["student"] + [tn.split("/")[-1] for tn in teacher_names]
    print("\n--- Oracle routing distribution ---")
    for c, ln in enumerate(label_names):
        count = (oracle_labels == c).sum()
        print(f"  {ln}: {count} ({100*count/n_actual:.1f}%)")

    # Constant policies
    policy_values = {}
    policy_values["always_student"] = margins_np["student"].mean()
    for tname in teacher_names:
        short = tname.split("/")[-1]
        policy_values[f"always_{short}"] = margins_np[tname].mean()
    best_constant_name = max(policy_values, key=policy_values.get)
    best_constant_val = policy_values[best_constant_name]
    oracle_val = oracle_margins.mean()

    print(f"\n--- Policy values (mean margin) ---")
    for pn, pv in sorted(policy_values.items(), key=lambda x: -x[1]):
        print(f"  {pn}: {pv:.4f}")
    print(f"  oracle: {oracle_val:.4f}")

    # Extract student features (no teacher info at deployment)
    print("\n--- Extracting student features ---")
    features = []
    s_margins = margins_np["student"]

    for i, pair in enumerate(pairs):
        q = pair["query"]
        docs = pair["documents"]
        gold = pair["gold_idx"]

        q_emb = student.encode([q], convert_to_tensor=True, normalize_embeddings=True)
        d_embs = student.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ d_embs.T).squeeze(0)

        if sims.dim() == 0:
            sims = sims.unsqueeze(0)
        sims_np = sims.cpu().numpy()

        sorted_sims = np.sort(sims_np)[::-1]
        top1_margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else sorted_sims[0]

        sm = torch.softmax(sims / 0.05, dim=0).cpu().numpy()
        entropy = -np.sum(sm * np.log(sm + 1e-10))

        q_words = len(q.split())
        q_chars = len(q)

        # Stability: paraphrase margin delta
        pq = f"In other words: {q}"
        pq_emb = student.encode([pq], convert_to_tensor=True, normalize_embeddings=True)
        p_sims = (pq_emb @ d_embs.T).squeeze(0)
        if p_sims.dim() == 0:
            p_sims = p_sims.unsqueeze(0)
        p_neg_mask = torch.ones(len(docs), dtype=torch.bool)
        p_neg_mask[gold] = False
        p_hardest = p_sims[p_neg_mask].max().item()
        p_margin = p_sims[gold].item() - p_hardest
        stability = abs(s_margins[i] - p_margin)

        features.append([
            s_margins[i],
            top1_margin,
            entropy,
            float(q_words),
            float(q_chars),
            stability,
            sorted_sims[0],
        ])

    X = np.array(features)
    y = oracle_labels

    # Train router with out-of-fold evaluation
    print(f"\n--- Training router ({n_folds}-fold CV) ---")

    # Skip if all labels are the same class
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"  SKIP: Only one class in oracle labels ({label_names[unique_labels[0]]})")
        print("  Conditional support not present — one model dominates all queries.")
        results = {
            "n_queries": n_actual,
            "oracle_val": float(oracle_val),
            "best_constant": best_constant_name,
            "best_constant_val": float(best_constant_val),
            "verdict": "NO_DIVERSITY",
            "note": f"One model ({label_names[unique_labels[0]]}) is best for all queries",
        }
        del student, teachers
        return results

    oof_preds_lr = np.full(n_actual, -1, dtype=int)
    oof_preds_tree = np.full(n_actual, -1, dtype=int)
    oof_preds_margin = np.full(n_actual, -1, dtype=int)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te = X[test_idx]

        lr = LogisticRegression(max_iter=500, random_state=seed)
        lr.fit(X_tr, y_tr)
        oof_preds_lr[test_idx] = lr.predict(X_te)

        tree = DecisionTreeClassifier(max_depth=3, random_state=seed)
        tree.fit(X_tr, y_tr)
        oof_preds_tree[test_idx] = tree.predict(X_te)

        # Margin-threshold baseline: route to best global teacher when student margin < threshold
        # Find threshold on train
        best_global_teacher_idx = 0
        best_global_teacher_mean = -999
        for c in range(1, len(label_names)):
            col_mean = all_margins[train_idx, c].mean()
            if col_mean > best_global_teacher_mean:
                best_global_teacher_mean = col_mean
                best_global_teacher_idx = c

        thresholds = np.percentile(X_tr[:, 0], [25, 50, 75])
        best_thresh = thresholds[1]
        best_thresh_val = -999
        for th in thresholds:
            preds = np.where(X_tr[:, 0] < th, best_global_teacher_idx, 0)
            val = np.mean([all_margins[train_idx[j], preds[j]] for j in range(len(train_idx))])
            if val > best_thresh_val:
                best_thresh_val = val
                best_thresh = th
        oof_preds_margin[test_idx] = np.where(X_te[:, 0] < best_thresh, best_global_teacher_idx, 0)

    # Compute policy values from OOF predictions
    def policy_value(preds):
        return np.mean([all_margins[i, preds[i]] for i in range(n_actual)])

    router_lr_val = policy_value(oof_preds_lr)
    router_tree_val = policy_value(oof_preds_tree)
    router_margin_val = policy_value(oof_preds_margin)

    # Shuffled baseline: shuffle oracle labels within difficulty deciles
    rng = np.random.RandomState(seed)
    decile_edges = np.percentile(s_margins, np.arange(0, 101, 10))
    shuffled_labels = oracle_labels.copy()
    for d in range(10):
        lo = decile_edges[d]
        hi = decile_edges[min(d + 1, 10)]
        mask = (s_margins >= lo) & (s_margins < hi) if d < 9 else (s_margins >= lo)
        idx = np.where(mask)[0]
        shuffled_labels[idx] = rng.permutation(shuffled_labels[idx])
    shuffled_val = policy_value(shuffled_labels)

    # Bootstrap confidence interval for LR router
    n_bootstrap = 1000
    bootstrap_captures = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_actual, n_actual, replace=True)
        b_router = np.mean([all_margins[i, oof_preds_lr[i]] for i in boot_idx])
        b_const = np.mean([all_margins[i, 0 if best_constant_name == "always_student"
                           else int(best_constant_name.split("_")[-1] == label_names[-1]) + 1
                           if len(label_names) > 2 else 1]
                          for i in boot_idx])
        b_const = best_constant_val  # use global best constant
        b_oracle = np.mean([oracle_margins[i] for i in boot_idx])
        denom = b_oracle - b_const
        if denom > 1e-8:
            bootstrap_captures.append((b_router - b_const) / denom)

    if bootstrap_captures:
        ci_lo = np.percentile(bootstrap_captures, 2.5)
        ci_hi = np.percentile(bootstrap_captures, 97.5)
        median_capture = np.median(bootstrap_captures)
    else:
        ci_lo = ci_hi = median_capture = 0.0

    # Oracle-uplift capture
    denom = oracle_val - best_constant_val
    if denom > 1e-8:
        capture_lr = (router_lr_val - best_constant_val) / denom
        capture_tree = (router_tree_val - best_constant_val) / denom
        capture_margin = (router_margin_val - best_constant_val) / denom
    else:
        capture_lr = capture_tree = capture_margin = 0.0

    print(f"\n--- Router results ---")
    print(f"  Best constant ({best_constant_name}): {best_constant_val:.4f}")
    print(f"  Margin-threshold router:              {router_margin_val:.4f}  (capture: {capture_margin:.1%})")
    print(f"  Logistic regression router:            {router_lr_val:.4f}  (capture: {capture_lr:.1%})")
    print(f"  Decision tree router:                  {router_tree_val:.4f}  (capture: {capture_tree:.1%})")
    print(f"  Shuffled within deciles:               {shuffled_val:.4f}")
    print(f"  Oracle per-query:                      {oracle_val:.4f}")
    print(f"  LR bootstrap 95% CI of capture:        [{ci_lo:.1%}, {ci_hi:.1%}]")

    # Verdict
    print("\n" + "=" * 60)
    print("ROUTING AUDIT VERDICT")
    print("=" * 60)

    pass_threshold = 0.25
    beats_margin = capture_lr > capture_margin
    positive_ci = ci_lo > 0
    sufficient_capture = capture_lr >= pass_threshold

    if sufficient_capture and positive_ci and beats_margin:
        verdict = "PASS"
        print(f"PASS: Router captures {capture_lr:.1%} of oracle uplift (>={pass_threshold:.0%}),")
        print(f"      beats margin-only baseline, positive bootstrap CI.")
        print("      Conditional support IS learnable from student features.")
    elif capture_lr > 0 and positive_ci:
        verdict = "WEAK_PASS"
        print(f"WEAK: Router captures {capture_lr:.1%} of oracle uplift.")
        if not beats_margin:
            print("      Does NOT beat margin-only. May just be 'this query is hard.'")
        if not sufficient_capture:
            print(f"      Below {pass_threshold:.0%} threshold.")
    else:
        verdict = "FAIL"
        print(f"FAIL: Router captures {capture_lr:.1%} of oracle uplift.")
        print("      Conditional support not learnable from student features.")
        if not positive_ci:
            print("      Bootstrap CI includes zero — no reliable signal.")

    del student, teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "n_queries": n_actual,
        "label_names": label_names,
        "oracle_distribution": {ln: int((oracle_labels == c).sum()) for c, ln in enumerate(label_names)},
        "policy_values": {
            **{k: float(v) for k, v in policy_values.items()},
            "margin_router": float(router_margin_val),
            "lr_router": float(router_lr_val),
            "tree_router": float(router_tree_val),
            "shuffled": float(shuffled_val),
            "oracle": float(oracle_val),
        },
        "oracle_uplift_capture": {
            "margin": float(capture_margin),
            "logistic_regression": float(capture_lr),
            "decision_tree": float(capture_tree),
        },
        "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
        "beats_margin": bool(beats_margin),
        "positive_ci": bool(positive_ci),
        "verdict": verdict,
    }
    return results


# ---------------------------------------------------------------------------
# V2 Experiment: Codex R2 spec — frozen encoder, B4c absorber
# ---------------------------------------------------------------------------

class FrozenStudentHead(torch.nn.Module):
    """Frozen encoder + trainable residual projection head (identical across arms)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.base = SentenceTransformer(model_name, device=device)
        for p in self.base.parameters():
            p.requires_grad = False
        dim = self.base.get_sentence_embedding_dimension()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        ).to(device)
        self.device = device
        self._model_name = model_name
        self._dim = dim

    def encode_with_grad(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            feats = self.base.tokenize(texts)
            feats = {k: v.to(self.device) for k, v in feats.items()}
            base_emb = self.base(feats)["sentence_embedding"]
        out = base_emb + self.head(base_emb)
        return F.normalize(out, p=2, dim=1)

    @torch.no_grad()
    def encode(self, texts: list[str], **kwargs) -> torch.Tensor:
        feats = self.base.tokenize(texts)
        feats = {k: v.to(self.device) for k, v in feats.items()}
        base_emb = self.base(feats)["sentence_embedding"]
        out = base_emb + self.head(base_emb)
        return F.normalize(out, p=2, dim=1)

    def trainable_parameters(self):
        return self.head.parameters()

    def train(self):
        self.head.train()
        return self

    def eval(self):
        self.head.eval()
        return self

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), os.path.join(path, "head.pt"))
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump({"encoder": self._model_name, "dim": self._dim}, f)

    def reset_head(self, seed: int = 42):
        torch.manual_seed(seed)
        for m in self.head:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


INTERVENTIONS = [
    lambda q: f"Rephrase: {q}",
    lambda q: " ".join(q.split()[:max(3, len(q.split()) // 2)]),
]


@torch.no_grad()
def cache_teacher_deltas(
    teacher_model,
    pairs: list[dict],
    interventions: list,
) -> list[dict]:
    """Compute teacher margin deltas under each intervention."""
    results = []
    for i, pair in enumerate(pairs):
        q = pair["query"]
        docs = pair["documents"]
        gold = pair["gold_idx"]

        d_embs = teacher_model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)

        q_emb = teacher_model.encode([q], convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ d_embs.T).squeeze(0)
        neg_mask = torch.ones(len(docs), dtype=torch.bool)
        neg_mask[gold] = False
        m_base = sims[gold].item() - sims[neg_mask].max().item()

        deltas = []
        for intv_fn in interventions:
            gq = intv_fn(q)
            gq_emb = teacher_model.encode([gq], convert_to_tensor=True, normalize_embeddings=True)
            g_sims = (gq_emb @ d_embs.T).squeeze(0)
            m_intv = g_sims[gold].item() - g_sims[neg_mask].max().item()
            delta = m_intv - m_base
            deltas.append({"text": gq, "m_intv": m_intv, "delta": delta})

        results.append({"m_base": m_base, "deltas": deltas})
        if (i + 1) % 100 == 0:
            print(f"  Teacher deltas: {i + 1}/{len(pairs)}")
    return results


def train_v2_arm(
    student: FrozenStudentHead,
    pairs: list[dict],
    teacher_deltas: list[dict],
    arm: str,
    steps: int = 500,
    lr: float = 1e-3,
    tau: float = 0.05,
    batch_size: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Train one arm. arm in {aug_contrastive, kd, b4c, eklavya}."""
    optimizer = torch.optim.AdamW(student.trainable_parameters(), lr=lr)
    rng = random.Random(seed)
    log = []
    student.train()

    for step in range(steps):
        idxs = rng.sample(range(len(pairs)), min(batch_size, len(pairs)))
        total_loss = torch.tensor(0.0, device=student.device, requires_grad=True)
        n_terms = 0

        for idx in idxs:
            pair = pairs[idx]
            q = pair["query"]
            docs = pair["documents"]
            gold = pair["gold_idx"]
            td = teacher_deltas[idx]

            neg_mask = torch.ones(len(docs), dtype=torch.bool)
            neg_mask[gold] = False

            if arm == "aug_contrastive":
                intv_fn = rng.choice(INTERVENTIONS)
                aug_q = intv_fn(q)
                all_texts = [aug_q] + docs
                embs = student.encode_with_grad(all_texts)
                sims = embs[0] @ embs[1:].T / tau
                target = torch.tensor(gold, device=student.device)
                total_loss = total_loss + F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))
                n_terms += 1

            elif arm == "kd":
                intv_texts = [d["text"] for d in td["deltas"]]
                all_texts = [q] + intv_texts + docs
                n_q = 1 + len(intv_texts)
                embs = student.encode_with_grad(all_texts)
                q_embs, d_embs = embs[:n_q], embs[n_q:]

                for qi in range(n_q):
                    s_sims = q_embs[qi] @ d_embs.T
                    s_margin = s_sims[gold] - s_sims[neg_mask].max()
                    t_margin = td["m_base"] if qi == 0 else td["deltas"][qi - 1]["m_intv"]
                    total_loss = total_loss + (s_margin - t_margin) ** 2
                    n_terms += 1

            elif arm in ("b4c", "eklavya"):
                intv_texts = [d["text"] for d in td["deltas"]]
                all_texts = [q] + intv_texts + docs
                n_q = 1 + len(intv_texts)
                embs = student.encode_with_grad(all_texts)
                q_embs, d_embs = embs[:n_q], embs[n_q:]

                support_w = 1.0 + sum(abs(d["delta"]) for d in td["deltas"])

                # Contrastive on base query
                base_sims = q_embs[0] @ d_embs.T / tau
                target = torch.tensor(gold, device=student.device)
                total_loss = total_loss + F.cross_entropy(
                    base_sims.unsqueeze(0), target.unsqueeze(0)) * support_w
                n_terms += 1

                s_base_all = q_embs[0] @ d_embs.T
                s_base_m = s_base_all[gold] - s_base_all[neg_mask].max()

                for j, d in enumerate(td["deltas"]):
                    # Consistency
                    cos = F.cosine_similarity(q_embs[0].unsqueeze(0), q_embs[j + 1].unsqueeze(0))
                    total_loss = total_loss + (1.0 - cos.squeeze()) * 0.1
                    n_terms += 1

                    s_intv_all = q_embs[j + 1] @ d_embs.T
                    s_intv_m = s_intv_all[gold] - s_intv_all[neg_mask].max()
                    s_delta = s_intv_m - s_base_m

                    if arm == "eklavya" and abs(d["delta"]) > 0.01:
                        sign = 1.0 if d["delta"] > 0 else -1.0
                        total_loss = total_loss + F.relu(0.01 - sign * s_delta) * support_w
                        n_terms += 1

        if n_terms > 0:
            loss = total_loss / n_terms
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.trainable_parameters(), 1.0)
            optimizer.step()

            if (step + 1) % 50 == 0 or step == 0:
                log.append({"step": step + 1, "loss": loss.item()})
                print(f"    Step {step + 1}/{steps}: loss={loss.item():.6f}")

    student.eval()
    return log


@torch.no_grad()
def ndcg_at_k(model, pairs: list[dict], k: int = 10) -> float:
    """Compute nDCG@k. Binary relevance: gold doc = 1, others = 0."""
    ndcgs = []
    for pair in pairs:
        docs = pair["documents"]
        gold = pair["gold_idx"]

        q_emb = model.encode([pair["query"]], convert_to_tensor=True, normalize_embeddings=True)
        d_embs = model.encode(docs, convert_to_tensor=True, normalize_embeddings=True)
        sims = (q_emb @ d_embs.T).squeeze(0)
        ranked = sims.argsort(descending=True).tolist()

        dcg = 0.0
        for i, doc_idx in enumerate(ranked[:k]):
            rel = 1.0 if doc_idx == gold else 0.0
            dcg += rel / math.log2(i + 2)
        idcg = 1.0 / math.log2(2)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def run_experiment_v2(config: dict) -> dict:
    """Codex R2 prescribed experiment: frozen encoder, B4c absorber."""
    device = config.get("device", "cpu")
    student_name = config.get("student", "sentence-transformers/all-MiniLM-L6-v2")
    teacher_name = config.get("teacher", "sentence-transformers/all-MiniLM-L12-v2")
    steps = config.get("steps", 500)
    lr = config.get("lr", 1e-3)
    tau = config.get("tau", 0.05)
    seed = config.get("seed", 42)
    n_pairs = config.get("n_pairs", 500)
    n_docs = config.get("n_docs", 10)
    out_dir = config.get("out_dir", "outputs/eklavya_v2")
    batch_size = config.get("batch_size", 8)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EKLAVYA V2 EXPERIMENT (Codex R2 spec)")
    print("=" * 60)
    print(f"Student: {student_name} (FROZEN + head)")
    print(f"Teacher: {teacher_name}")
    print(f"Steps: {steps}, LR: {lr}, Seed: {seed}")
    print(f"Pairs: {n_pairs} ({n_docs} docs each)")

    all_pairs = load_msmarco_pairs(n=n_pairs, n_docs=n_docs, seed=seed)
    n_total = len(all_pairs)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]
    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test")

    # Cache teacher deltas on training data
    print(f"\n--- Caching teacher deltas ({teacher_name}) ---")
    teacher = load_model(teacher_name, device)
    teacher_deltas = cache_teacher_deltas(teacher, train_pairs, INTERVENTIONS)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Baseline: frozen student, no head training
    print("\n--- Baseline (frozen, no training) ---")
    base_student = FrozenStudentHead(student_name, device)
    base_ndcg = ndcg_at_k(base_student, test_pairs)
    base_mrr = eval_retrieval(base_student, test_pairs)["mrr"]
    print(f"  nDCG@10={base_ndcg:.4f}  MRR={base_mrr:.4f}")
    del base_student

    # Overfit check: can the head fit a tiny subset?
    print("\n--- Overfit positive control (10 pairs, 200 steps) ---")
    overfit_student = FrozenStudentHead(student_name, device)
    tiny_deltas = cache_teacher_deltas(
        load_model(teacher_name, device), train_pairs[:10], INTERVENTIONS)
    train_v2_arm(overfit_student, train_pairs[:10], tiny_deltas, "eklavya",
                 steps=200, lr=1e-3, batch_size=10, seed=seed)
    overfit_ndcg = ndcg_at_k(overfit_student, train_pairs[:10])
    print(f"  Overfit nDCG@10={overfit_ndcg:.4f} (should be ~1.0 if head has capacity)")
    if overfit_ndcg < 0.8:
        print("  WARNING: Head cannot overfit tiny set — capacity may be insufficient")
    del overfit_student

    arms = ["aug_contrastive", "kd", "b4c", "eklavya"]
    results = {"config": config, "baseline_ndcg": base_ndcg, "baseline_mrr": base_mrr}

    for arm in arms:
        print(f"\n--- Arm: {arm} ---")
        student = FrozenStudentHead(student_name, device)
        student.reset_head(seed=seed)
        arm_log = train_v2_arm(
            student, train_pairs, teacher_deltas, arm,
            steps=steps, lr=lr, tau=tau, batch_size=batch_size, seed=seed,
        )
        arm_ndcg = ndcg_at_k(student, test_pairs)
        arm_mrr = eval_retrieval(student, test_pairs)["mrr"]
        student.save(os.path.join(out_dir, f"arm_{arm}"))
        print(f"  nDCG@10={arm_ndcg:.4f}  MRR={arm_mrr:.4f}")

        results[arm] = {
            "ndcg10": arm_ndcg, "mrr": arm_mrr,
            "gain_ndcg": arm_ndcg - base_ndcg,
            "log": arm_log,
        }

    # Critical comparison: Eklavya vs B4c
    ek = results["eklavya"]
    b4c = results["b4c"]
    delta = ek["ndcg10"] - b4c["ndcg10"]

    print("\n" + "=" * 60)
    print("V2 RESULTS")
    print("=" * 60)
    print(f"  Baseline nDCG@10:        {base_ndcg:.4f}")
    for arm in arms:
        r = results[arm]
        print(f"  {arm:25s}: {r['ndcg10']:.4f}  (gain: {r['gain_ndcg']:+.4f})")
    print()
    print(f"  Eklavya vs B4c delta:    {delta:+.4f}")
    print(f"  Threshold:               +0.005")

    if delta >= 0.005:
        verdict = "ALIVE"
        print(f"\n  >>> ALIVE: Eklavya beats B4c by {delta:+.4f} >= +0.005")
        print("      Response-delta matching adds value beyond equal-information baseline.")
    elif delta > 0:
        verdict = "WEAK"
        print(f"\n  >>> WEAK: Eklavya beats B4c by {delta:+.4f} < +0.005")
        print("      Positive but below precommit threshold. Need more seeds/data.")
    else:
        verdict = "DEAD"
        print(f"\n  >>> DEAD: Eklavya does NOT beat B4c ({delta:+.4f})")
        print("      Response-delta targets absorbed by equal-information baseline.")

    results["eklavya_vs_b4c_delta"] = delta
    results["verdict"] = verdict

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items()
                   if k != "config" or not callable(v)}, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# End-to-end experiment (v1, pre-Codex-R2)
# ---------------------------------------------------------------------------

def run_experiment(config: dict) -> dict:
    """Extract → train (3 arms) → eval → compare."""
    device = config.get("device", "cpu")
    student_name = config.get("student", "sentence-transformers/all-MiniLM-L6-v2")
    teacher_names = config.get("teachers", [
        "sentence-transformers/all-MiniLM-L12-v2",
        "BAAI/bge-small-en-v1.5",
    ])
    steps = config.get("steps", 300)
    lr = config.get("lr", 2e-5)
    tau = config.get("tau", 0.05)
    seed = config.get("seed", 42)
    n_pairs = config.get("n_pairs", 100)
    out_dir = config.get("out_dir", "outputs/eklavya_embed_exp")
    batch_size = config.get("batch_size", 8)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EKLAVYA EMBEDDING EXPERIMENT")
    print("=" * 60)
    print(f"Student: {student_name}")
    print(f"Teachers: {teacher_names}")
    print(f"Steps: {steps}, LR: {lr}, Tau: {tau}, Seed: {seed}")
    print(f"Pairs: {n_pairs}, Device: {device}")
    print()

    use_real_data = config.get("real_data", False)
    n_docs = config.get("n_docs", 10)
    if use_real_data:
        all_pairs = load_msmarco_pairs(n=n_pairs, n_docs=n_docs, seed=seed)
    else:
        all_pairs = generate_toy_pairs(n=n_pairs, n_docs=n_docs, seed=seed)
    split = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split]
    test_pairs = all_pairs[split:]
    print(f"Data: {len(train_pairs)} train, {len(test_pairs)} test"
          f" ({'MSMARCO' if use_real_data else 'toy'})")

    print("\n--- Extracting teacher signatures ---")
    sigs = extract_signatures(teacher_names, train_pairs, device=device, seed=seed)

    print("\n--- Baseline (pretrained, no fine-tuning) ---")
    base_model = load_model(student_name, device)
    base_metrics = eval_retrieval(base_model, test_pairs)
    print(f"  Hit@5={base_metrics['hit_at_k']:.4f}  MRR={base_metrics['mrr']:.4f}")
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Arm A: Ranking KD (identity probe only) ---")
    student_a = StudentWrapper(student_name, device)
    log_a = train_ranking_kd(
        student_a, sigs, steps=steps, lr=lr, tau=tau,
        batch_size=batch_size, use_probes=False, seed=seed,
    )
    metrics_a = eval_student(student_a, test_pairs)
    student_a.save(os.path.join(out_dir, "arm_a_ranking_kd"))
    print(f"  Hit@5={metrics_a['hit_at_k']:.4f}  MRR={metrics_a['mrr']:.4f}")

    print("\n--- Arm B: Probe Tomography (all probes) ---")
    student_b = StudentWrapper(student_name, device)
    log_b = train_ranking_kd(
        student_b, sigs, steps=steps, lr=lr, tau=tau,
        batch_size=batch_size, use_probes=True, seed=seed,
    )
    metrics_b = eval_student(student_b, test_pairs)
    student_b.save(os.path.join(out_dir, "arm_b_tomography"))
    print(f"  Hit@5={metrics_b['hit_at_k']:.4f}  MRR={metrics_b['mrr']:.4f}")

    print("\n--- Arm C: Contrastive Baseline (no teacher) ---")
    student_c = StudentWrapper(student_name, device)
    log_c = train_contrastive(
        student_c, train_pairs, steps=steps, lr=lr, tau=tau,
        batch_size=batch_size, seed=seed,
    )
    metrics_c = eval_student(student_c, test_pairs)
    student_c.save(os.path.join(out_dir, "arm_c_contrastive"))
    print(f"  Hit@5={metrics_c['hit_at_k']:.4f}  MRR={metrics_c['mrr']:.4f}")

    print("\n--- Arm D: Augmented Contrastive (B4 hostile baseline) ---")
    student_d = StudentWrapper(student_name, device)
    log_d = train_augmented_contrastive(
        student_d, train_pairs, steps=steps, lr=lr, tau=tau,
        batch_size=batch_size, seed=seed,
    )
    metrics_d = eval_student(student_d, test_pairs)
    student_d.save(os.path.join(out_dir, "arm_d_augmented_contrastive"))
    print(f"  Hit@5={metrics_d['hit_at_k']:.4f}  MRR={metrics_d['mrr']:.4f}")

    results = {
        "config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                   for k, v in config.items()},
        "baseline": base_metrics,
        "arm_a_ranking_kd": metrics_a,
        "arm_b_tomography": metrics_b,
        "arm_c_contrastive": metrics_c,
        "arm_d_augmented_contrastive": metrics_d,
        "retained_gain_a": metrics_a["mrr"] - base_metrics["mrr"],
        "retained_gain_b": metrics_b["mrr"] - base_metrics["mrr"],
        "retained_gain_c": metrics_c["mrr"] - base_metrics["mrr"],
        "retained_gain_d": metrics_d["mrr"] - base_metrics["mrr"],
        "control_adjusted_a_vs_c": metrics_a["mrr"] - metrics_c["mrr"],
        "control_adjusted_b_vs_c": metrics_b["mrr"] - metrics_c["mrr"],
        "control_adjusted_b_vs_d": metrics_b["mrr"] - metrics_d["mrr"],
        "training_log_a": log_a,
        "training_log_b": log_b,
        "training_log_c": log_c,
        "training_log_d": log_d,
    }

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Baseline MRR:              {base_metrics['mrr']:.4f}")
    print(f"  Arm A (Ranking KD) MRR:    {metrics_a['mrr']:.4f}  (gain: {results['retained_gain_a']:+.4f})")
    print(f"  Arm B (Tomography) MRR:    {metrics_b['mrr']:.4f}  (gain: {results['retained_gain_b']:+.4f})")
    print(f"  Arm C (Contrastive) MRR:   {metrics_c['mrr']:.4f}  (gain: {results['retained_gain_c']:+.4f})")
    print(f"  Arm D (Aug Contrastive):   {metrics_d['mrr']:.4f}  (gain: {results['retained_gain_d']:+.4f})")
    print()
    print(f"  B vs C (tomography over contrastive):    {results['control_adjusted_b_vs_c']:+.4f}")
    print(f"  B vs D (tomography over augmented):      {results['control_adjusted_b_vs_d']:+.4f}")
    print()

    if results["control_adjusted_b_vs_d"] > 0:
        print("  >>> Tomography beats B4 hostile baseline. Teacher response surfaces add value beyond augmentation.")
    elif results["control_adjusted_b_vs_c"] > 0:
        print("  >>> Tomography beats C but NOT D. Probes are data augmentation, not teacher signals. ABSORBED.")
    elif results["control_adjusted_a_vs_c"] > 0:
        print("  >>> Ranking KD beats contrastive but probes add nothing.")
    else:
        print("  >>> Contrastive baseline absorbs everything. Direction may be dead.")

    print(f"\n  Results saved to {results_path}")
    return results


# ---------------------------------------------------------------------------
# Data: toy generator and real data loader
# ---------------------------------------------------------------------------

def generate_toy_pairs(n: int = 100, n_docs: int = 8, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    topics = [
        ("machine learning", "deep learning algorithms and neural networks"),
        ("climate change", "global warming effects on weather patterns"),
        ("quantum computing", "quantum bits and superposition states"),
        ("ancient history", "civilizations of mesopotamia and egypt"),
        ("molecular biology", "DNA replication and protein synthesis"),
        ("space exploration", "mars rovers and satellite missions"),
        ("economic policy", "interest rates and inflation management"),
        ("music theory", "chord progressions and harmonic analysis"),
        ("culinary arts", "cooking techniques and flavor profiles"),
        ("cybersecurity", "network intrusion detection and prevention"),
    ]
    distractors = [
        "the weather forecast for tomorrow",
        "how to tie a shoelace properly",
        "the population of a small town",
        "instructions for assembling furniture",
        "a recipe for chocolate cake",
        "the rules of a card game",
        "a brief history of postal services",
        "tips for gardening in spring",
        "an overview of public transportation",
        "the basics of knitting patterns",
    ]

    pairs = []
    for i in range(n):
        topic_idx = i % len(topics)
        query_topic, gold_doc = topics[topic_idx]
        query = f"What is {query_topic}?"

        docs = [gold_doc]
        chosen_distractors = rng.sample(distractors, min(n_docs - 1, len(distractors)))
        docs.extend(chosen_distractors[:n_docs - 1])
        rng.shuffle(docs)
        gold_idx = docs.index(gold_doc)

        pairs.append({"id": f"toy_{i}", "query": query, "documents": docs, "gold_idx": gold_idx})

    return pairs


def load_msmarco_pairs(
    n: int = 500,
    n_docs: int = 10,
    seed: int = 42,
    cache_dir: str | None = None,
) -> list[dict]:
    """Load MSMARCO with BM25 hard negatives + random negatives.

    Each pair: 1 positive + 1 BM25 hard negative + (n_docs-2) random negatives
    from other queries' positives. This creates ranking distributions with real
    structure — hard negatives are topically similar, randoms are easy.
    """
    from datasets import load_dataset

    load_n = max(n * 3, 2000)
    print(f"Loading MSMARCO hard negatives (target={n}, n_docs={n_docs})...")
    ds = load_dataset(
        "sentence-transformers/msmarco-bm25",
        "triplet",
        split=f"train[:{load_n}]",
        cache_dir=cache_dir,
    )

    rng = random.Random(seed)
    raw = []
    seen_queries = set()

    for row in ds:
        query = row.get("anchor") or row.get("query") or row.get("sentence1", "")
        positive = row.get("positive") or row.get("pos") or row.get("sentence2", "")
        negative = row.get("negative") or row.get("neg") or row.get("sentence3", "")
        if not query or not positive or not negative:
            continue
        if query in seen_queries:
            continue
        seen_queries.add(query)
        raw.append({"query": query, "positive": positive, "hard_neg": negative})

    all_positives = [r["positive"] for r in raw]

    pairs = []
    for i, r in enumerate(raw[:n]):
        docs = [r["positive"], r["hard_neg"]]
        other_idxs = [j for j in range(len(all_positives)) if j != i]
        rand_neg_idxs = rng.sample(other_idxs, min(n_docs - 2, len(other_idxs)))
        docs.extend(all_positives[j] for j in rand_neg_idxs)
        rng.shuffle(docs)
        gold_idx = docs.index(r["positive"])

        pairs.append({
            "id": f"msmarco_{len(pairs)}",
            "query": r["query"],
            "documents": docs,
            "gold_idx": gold_idx,
        })

    print(f"  Loaded {len(pairs)} MSMARCO pairs ({n_docs} docs each)")
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_extract(args):
    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        print("No data file provided or found. Using toy data.")
        pairs = generate_toy_pairs(n=args.limit or 50)

    sigs = extract_signatures(
        teacher_names=args.teachers,
        pairs=pairs[:args.limit] if args.limit else pairs,
        device=args.device,
    )
    save_signatures(sigs, args.out)


def cmd_toy(args):
    pairs = generate_toy_pairs(n=args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Generated {len(pairs)} toy pairs to {args.out}")


def cmd_train(args):
    sigs = load_signatures(args.signatures)

    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        pairs = [{"query": s.query, "documents": s.documents, "gold_idx": s.gold_idx}
                 for s in sigs]

    student = StudentWrapper(args.student, args.device)
    log = train_ranking_kd(
        student, sigs, steps=args.steps, lr=args.lr, tau=args.tau,
        batch_size=args.batch_size, use_probes=args.probes, seed=args.seed,
    )
    student.save(args.out_dir)
    print(f"Student saved to {args.out_dir}")

    test_metrics = eval_student(student, pairs[-20:])
    print(f"Train-tail eval: {test_metrics}")


def cmd_eval(args):
    if os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        pairs = generate_toy_pairs(n=50)

    model = load_model(args.model, args.device)
    metrics = eval_retrieval(model, pairs, k=args.k)
    print(json.dumps(metrics, indent=2))


def cmd_diagnostic(args):
    if args.data and os.path.exists(args.data):
        with open(args.data, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f]
    else:
        pairs = generate_toy_pairs(n=args.n)

    results = run_diagnostic(
        student_name=args.student,
        teacher_names=args.teachers,
        pairs=pairs,
        device=args.device,
        seed=args.seed,
    )

    out_path = args.out or "outputs/diagnostic_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


def cmd_routing_audit(args):
    results = run_routing_audit(
        student_name=args.student,
        teacher_names=args.teachers,
        n_queries=args.n_queries,
        device=args.device,
        seed=args.seed,
        n_folds=args.folds,
    )
    out_path = args.out or "outputs/routing_audit.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


def cmd_experiment_v2(args):
    config = {
        "device": args.device,
        "student": args.student,
        "teacher": args.teacher,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "n_docs": args.n_docs,
        "out_dir": args.out_dir,
        "batch_size": args.batch_size,
    }
    run_experiment_v2(config)


def cmd_experiment(args):
    config = {
        "device": args.device,
        "student": args.student,
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "seed": args.seed,
        "n_pairs": args.n_pairs,
        "out_dir": args.out_dir,
        "batch_size": args.batch_size,
        "real_data": args.real_data,
        "n_docs": args.n_docs,
    }
    run_experiment(config)


def main():
    parser = argparse.ArgumentParser(description="Eklavya Embedding Tomography")
    sub = parser.add_subparsers(dest="cmd")

    p_extract = sub.add_parser("extract", help="Extract teacher embedding signatures")
    p_extract.add_argument("--teachers", nargs="+", required=True)
    p_extract.add_argument("--data", type=str, default=None)
    p_extract.add_argument("--out", type=str, default="data/embed_signatures.jsonl")
    p_extract.add_argument("--device", type=str, default="cpu")
    p_extract.add_argument("--limit", type=int, default=None)

    p_toy = sub.add_parser("toy", help="Generate toy query-document pairs")
    p_toy.add_argument("--n", type=int, default=100)
    p_toy.add_argument("--out", type=str, default="data/toy_pairs.jsonl")

    p_train = sub.add_parser("train", help="Train student from signatures")
    p_train.add_argument("--signatures", required=True)
    p_train.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p_train.add_argument("--data", default=None)
    p_train.add_argument("--out_dir", default="outputs/eklavya_student")
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--steps", type=int, default=500)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--tau", type=float, default=0.05)
    p_train.add_argument("--batch_size", type=int, default=8)
    p_train.add_argument("--probes", action="store_true", default=True)
    p_train.add_argument("--no_probes", dest="probes", action="store_false")
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Evaluate a model")
    p_eval.add_argument("--model", required=True)
    p_eval.add_argument("--data", default=None)
    p_eval.add_argument("--device", default="cpu")
    p_eval.add_argument("--k", type=int, default=5)

    p_diag = sub.add_parser("diagnostic", help="CPU-only signal diagnostic (no training)")
    p_diag.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p_diag.add_argument("--teachers", nargs="+", default=[
        "sentence-transformers/all-MiniLM-L12-v2",
        "BAAI/bge-small-en-v1.5",
    ])
    p_diag.add_argument("--data", default=None)
    p_diag.add_argument("--n", type=int, default=50)
    p_diag.add_argument("--device", default="cpu")
    p_diag.add_argument("--seed", type=int, default=42)
    p_diag.add_argument("--out", default=None)

    p_route = sub.add_parser("routing_audit", help="Q3: cache-only conditional support test")
    p_route.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p_route.add_argument("--teachers", nargs="+", default=[
        "sentence-transformers/all-MiniLM-L12-v2",
        "BAAI/bge-small-en-v1.5",
    ])
    p_route.add_argument("--n_queries", type=int, default=500)
    p_route.add_argument("--device", default="cpu")
    p_route.add_argument("--seed", type=int, default=42)
    p_route.add_argument("--folds", type=int, default=5)
    p_route.add_argument("--out", default=None)

    p_v2 = sub.add_parser("experiment_v2", help="Codex R2: frozen encoder + B4c absorber")
    p_v2.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p_v2.add_argument("--teacher", default="sentence-transformers/all-MiniLM-L12-v2")
    p_v2.add_argument("--device", default="cpu")
    p_v2.add_argument("--steps", type=int, default=500)
    p_v2.add_argument("--lr", type=float, default=1e-3)
    p_v2.add_argument("--tau", type=float, default=0.05)
    p_v2.add_argument("--seed", type=int, default=42)
    p_v2.add_argument("--n_pairs", type=int, default=500)
    p_v2.add_argument("--n_docs", type=int, default=10)
    p_v2.add_argument("--out_dir", default="outputs/eklavya_v2")
    p_v2.add_argument("--batch_size", type=int, default=8)

    p_exp = sub.add_parser("experiment", help="Run full 3-arm experiment (v1)")
    p_exp.add_argument("--student", default="sentence-transformers/all-MiniLM-L6-v2")
    p_exp.add_argument("--teachers", nargs="+", default=[
        "sentence-transformers/all-MiniLM-L12-v2",
        "BAAI/bge-small-en-v1.5",
    ])
    p_exp.add_argument("--device", default="cpu")
    p_exp.add_argument("--steps", type=int, default=300)
    p_exp.add_argument("--lr", type=float, default=2e-5)
    p_exp.add_argument("--tau", type=float, default=0.05)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--n_pairs", type=int, default=100)
    p_exp.add_argument("--out_dir", default="outputs/eklavya_embed_exp")
    p_exp.add_argument("--batch_size", type=int, default=8)
    p_exp.add_argument("--real_data", action="store_true",
                       help="Use MSMARCO hard negatives instead of toy data")
    p_exp.add_argument("--n_docs", type=int, default=10,
                       help="Documents per query (1 positive + N-1 negatives)")

    args = parser.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "toy":
        cmd_toy(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "diagnostic":
        cmd_diagnostic(args)
    elif args.cmd == "routing_audit":
        cmd_routing_audit(args)
    elif args.cmd == "experiment_v2":
        cmd_experiment_v2(args)
    elif args.cmd == "experiment":
        cmd_experiment(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
