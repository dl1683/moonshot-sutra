"""Data loading for Eklavya embedding experiments.

Supports:
  - Toy data (built-in, for smoke tests)
  - Hard-negative toy data (semantically similar distractors)
  - MS MARCO passage retrieval (real data)
  - NQ (Natural Questions) via BeIR
  - Custom JSONL files
  - E16 Boundary Inheritance: teacher-curated negative mining

Each loader returns list[dict] with keys: id, query, documents, gold_idx
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path


def load_hard_toy(n: int = 200, n_docs: int = 8, seed: int = 42) -> list[dict]:
    """Generate toy pairs with hard negatives — semantically related but wrong."""
    rng = random.Random(seed)

    topic_clusters = [
        {
            "queries": [
                "How do neural networks learn from data?",
                "What is backpropagation in deep learning?",
                "How does gradient descent optimize model weights?",
            ],
            "gold_docs": [
                "Neural networks learn by adjusting connection weights through exposure to labeled examples, using error signals to update parameters via backpropagation.",
                "Backpropagation computes gradients of a loss function with respect to each weight by applying the chain rule layer by layer from output to input.",
                "Gradient descent iteratively adjusts model weights in the direction that reduces the loss function, with the step size controlled by the learning rate.",
            ],
            "hard_negatives": [
                "Convolutional neural networks use spatial filters to detect local patterns in images, building hierarchical feature representations.",
                "Transformer models use self-attention mechanisms to process all positions in a sequence simultaneously rather than sequentially.",
                "Reinforcement learning agents maximize cumulative reward through trial-and-error interactions with an environment.",
                "Transfer learning allows models pre-trained on large datasets to be fine-tuned for specific downstream tasks with limited data.",
                "Batch normalization stabilizes training by normalizing layer inputs, allowing higher learning rates and faster convergence.",
            ],
        },
        {
            "queries": [
                "What causes global temperatures to rise?",
                "How do greenhouse gases affect Earth's climate?",
                "What is the relationship between CO2 and ocean acidification?",
            ],
            "gold_docs": [
                "Global temperatures rise primarily due to increased concentrations of greenhouse gases that trap outgoing infrared radiation in the atmosphere.",
                "Greenhouse gases like CO2 and methane absorb and re-emit thermal radiation, creating an insulating effect that warms the lower atmosphere and surface.",
                "Ocean acidification occurs when dissolved CO2 reacts with seawater to form carbonic acid, lowering pH and threatening marine calcifying organisms.",
            ],
            "hard_negatives": [
                "The ozone layer in the stratosphere absorbs most of the sun's ultraviolet radiation, protecting life on Earth from harmful UV-B and UV-C rays.",
                "El Nino events cause periodic warming of tropical Pacific waters, disrupting normal weather patterns across multiple continents.",
                "Volcanic eruptions can temporarily cool global temperatures by injecting sulfate aerosols into the stratosphere that reflect sunlight.",
                "The carbon cycle involves the exchange of carbon between the atmosphere, oceans, biosphere, and lithosphere over various timescales.",
                "Deforestation reduces the planet's capacity to absorb CO2 through photosynthesis, contributing to atmospheric carbon accumulation.",
            ],
        },
        {
            "queries": [
                "How does quantum entanglement work?",
                "What is superposition in quantum mechanics?",
                "How do quantum computers achieve speedup over classical ones?",
            ],
            "gold_docs": [
                "Quantum entanglement creates correlations between particles such that measuring one instantly determines the state of its partner regardless of distance.",
                "Superposition allows a quantum system to exist in multiple states simultaneously until measured, at which point it collapses to one definite state.",
                "Quantum computers exploit superposition and entanglement to explore many solution paths in parallel, achieving exponential speedup for certain problems.",
            ],
            "hard_negatives": [
                "The Heisenberg uncertainty principle states that certain pairs of physical properties cannot both be known to arbitrary precision simultaneously.",
                "Quantum decoherence occurs when a quantum system interacts with its environment, causing it to lose its quantum properties and behave classically.",
                "Shor's algorithm factors large integers in polynomial time on a quantum computer, threatening widely used public-key cryptography schemes.",
                "Quantum error correction encodes logical qubits across multiple physical qubits to protect quantum information from noise and decoherence.",
                "The double-slit experiment demonstrates wave-particle duality by showing that particles create interference patterns when not observed.",
            ],
        },
        {
            "queries": [
                "How did ancient Egyptians build the pyramids?",
                "What was the significance of the Rosetta Stone?",
                "How did Mesopotamian civilization develop writing?",
            ],
            "gold_docs": [
                "Ancient Egyptians built pyramids using limestone blocks quarried nearby, transported on sledges over wetted sand, and raised using internal ramps.",
                "The Rosetta Stone enabled decipherment of Egyptian hieroglyphics by providing the same text in three scripts: hieroglyphic, demotic, and Greek.",
                "Mesopotamian writing evolved from clay tokens for accounting into cuneiform, where wedge-shaped marks pressed into wet clay tablets represented words and sounds.",
            ],
            "hard_negatives": [
                "The Library of Alexandria was one of the largest repositories of knowledge in the ancient world before its gradual destruction over several centuries.",
                "Roman aqueducts transported water across long distances using gravity and precise engineering, supplying cities with fresh water for baths and fountains.",
                "The Silk Road connected East Asian and Mediterranean civilizations through a network of trade routes spanning thousands of miles.",
                "Ancient Greek city-states developed democratic governance systems that influenced political philosophy and government structures for millennia.",
                "The Code of Hammurabi is one of the oldest known written legal codes, establishing laws and punishments for ancient Babylonian society.",
            ],
        },
        {
            "queries": [
                "How does mRNA carry genetic instructions for protein synthesis?",
                "What role do ribosomes play in translating genetic code?",
                "How does CRISPR-Cas9 edit genes?",
            ],
            "gold_docs": [
                "mRNA is transcribed from DNA in the nucleus and carries the genetic code to ribosomes in the cytoplasm, where it directs amino acid assembly.",
                "Ribosomes read mRNA codons three nucleotides at a time, matching each codon with the corresponding tRNA carrying the specified amino acid.",
                "CRISPR-Cas9 uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it creates a double-strand break for precise gene editing.",
            ],
            "hard_negatives": [
                "Mitochondria generate ATP through oxidative phosphorylation, producing most of the chemical energy needed for cellular functions.",
                "Epigenetic modifications like DNA methylation can alter gene expression without changing the underlying DNA sequence.",
                "Stem cells have the unique ability to differentiate into specialized cell types, making them valuable for regenerative medicine research.",
                "Telomeres are protective caps at chromosome ends that shorten with each cell division, contributing to cellular aging.",
                "The human microbiome contains trillions of microorganisms that influence digestion, immunity, and even neurological function.",
            ],
        },
        {
            "queries": [
                "What algorithms power modern search engines?",
                "How do recommendation systems predict user preferences?",
                "What is the difference between SQL and NoSQL databases?",
            ],
            "gold_docs": [
                "Modern search engines use inverted indexes for retrieval, PageRank-like link analysis for authority, and neural re-rankers for semantic relevance.",
                "Recommendation systems use collaborative filtering to find users with similar preferences and content-based filtering to match item features to user profiles.",
                "SQL databases enforce rigid schemas with ACID transactions for consistency, while NoSQL databases offer flexible schemas and horizontal scaling for distributed data.",
            ],
            "hard_negatives": [
                "Hash tables provide O(1) average-case lookup by mapping keys to array indices through a hash function.",
                "TCP ensures reliable data delivery through sequence numbers, acknowledgments, and retransmission of lost packets.",
                "Load balancers distribute incoming network traffic across multiple servers to prevent any single server from becoming overwhelmed.",
                "Containerization packages applications with their dependencies into isolated units that run consistently across different computing environments.",
                "Version control systems like Git track changes to files over time, enabling collaboration and history management for software projects.",
            ],
        },
    ]

    pairs = []
    pair_id = 0
    for _ in range(n // len(topic_clusters) + 1):
        for cluster in topic_clusters:
            if pair_id >= n:
                break
            q_idx = pair_id % len(cluster["queries"])
            query = cluster["queries"][q_idx]
            gold = cluster["gold_docs"][q_idx]

            # Build doc set: gold + hard negatives from same cluster + some from other clusters
            docs = [gold]
            same_cluster_negs = [d for i, d in enumerate(cluster["gold_docs"]) if i != q_idx]
            docs.extend(same_cluster_negs[:2])
            docs.extend(rng.sample(cluster["hard_negatives"], min(3, len(cluster["hard_negatives"]))))

            other_clusters = [c for c in topic_clusters if c is not cluster]
            other_negs = []
            for oc in rng.sample(other_clusters, min(2, len(other_clusters))):
                other_negs.extend(rng.sample(oc["hard_negatives"], 1))
            docs.extend(other_negs[:n_docs - len(docs)])

            docs = docs[:n_docs]
            rng.shuffle(docs)
            gold_idx = docs.index(gold)

            pairs.append({"id": f"hard_{pair_id}", "query": query, "documents": docs, "gold_idx": gold_idx})
            pair_id += 1

    return pairs[:n]


def mine_hard_negatives(
    pairs: list[dict],
    student_model,
    n_docs: int = 32,
    batch_size: int = 64,
) -> list[dict]:
    """Re-mine hard negatives using the raw student's embeddings.

    Pools all passages across queries, encodes with the student,
    and for each query selects the top-n_docs hardest negatives
    (highest cosine similarity to query, excluding the gold doc).
    """
    import torch

    all_passages = []
    passage_to_idx = {}
    gold_passages = {}

    for pair in pairs:
        gold_doc = pair["documents"][pair["gold_idx"]]
        gold_passages[pair["id"]] = gold_doc
        for doc in pair["documents"]:
            if doc not in passage_to_idx:
                passage_to_idx[doc] = len(all_passages)
                all_passages.append(doc)

    print(f"  Mining hard negatives: {len(all_passages)} unique passages, {len(pairs)} queries")

    with torch.no_grad():
        passage_embs = []
        for i in range(0, len(all_passages), batch_size):
            batch = all_passages[i : i + batch_size]
            embs = student_model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            passage_embs.append(embs.cpu())
        passage_embs = torch.cat(passage_embs, dim=0)

    mined_pairs = []
    for pair in pairs:
        query = pair["query"]
        gold_doc = pair["documents"][pair["gold_idx"]]
        gold_idx_in_pool = passage_to_idx[gold_doc]

        with torch.no_grad():
            q_emb = student_model.encode([query], convert_to_tensor=True, normalize_embeddings=True).cpu()
        sims = (q_emb @ passage_embs.T).squeeze(0)

        sims[gold_idx_in_pool] = -1.0
        topk_indices = sims.argsort(descending=True)[: n_docs - 1].tolist()

        docs = [gold_doc] + [all_passages[idx] for idx in topk_indices]
        rng = random.Random(hash(pair["id"]))
        order = list(range(len(docs)))
        rng.shuffle(order)
        shuffled_docs = [docs[i] for i in order]
        new_gold_idx = order.index(0)

        mined_pairs.append({
            "id": pair["id"],
            "query": query,
            "documents": shuffled_docs,
            "gold_idx": new_gold_idx,
        })

    print(f"  Mined {n_docs} docs per query ({n_docs - 1} hard negatives + 1 gold)")
    return mined_pairs


# ---------------------------------------------------------------------------
# E16 Boundary Inheritance: teacher-curated negative mining
# ---------------------------------------------------------------------------

def stable_int(key: str, seed: int) -> int:
    """SHA-256 derived deterministic integer — never use Python hash()."""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:16], 16)


def build_passage_pool(pairs: list[dict]) -> dict:
    """Build a deduplicated passage pool with stable IDs from raw pairs.

    Returns dict with:
      passages: list[str]  — unique texts
      text_to_pid: dict[str, str]  — normalized text -> stable passage ID
      pid_to_idx: dict[str, int]  — passage ID -> index in passages list
      positive_pids: dict[str, set[str]]  — query_id -> set of positive passage IDs
    """
    passages = []
    text_to_pid = {}
    pid_to_idx = {}
    positive_pids = {}

    for pair in pairs:
        gold_doc = pair["documents"][pair["gold_idx"]]
        norm_gold = gold_doc.strip()
        qid = pair["id"]

        for doc in pair["documents"]:
            norm = doc.strip()
            if norm not in text_to_pid:
                pid = f"p_{hashlib.sha256(norm.encode()).hexdigest()[:12]}"
                text_to_pid[norm] = pid
                pid_to_idx[pid] = len(passages)
                passages.append(norm)

        if qid not in positive_pids:
            positive_pids[qid] = set()
        positive_pids[qid].add(text_to_pid[norm_gold])

        for i, doc in enumerate(pair["documents"]):
            if pair.get("selected", []):
                if pair["selected"][i]:
                    positive_pids[qid].add(text_to_pid[doc.strip()])

    print(f"  Passage pool: {len(passages)} unique, {len(positive_pids)} queries")
    return {
        "passages": passages,
        "text_to_pid": text_to_pid,
        "pid_to_idx": pid_to_idx,
        "positive_pids": positive_pids,
    }


def rank_embedding_model(
    pairs: list[dict],
    pool: dict,
    model,
    top_k: int = 128,
    batch_size: int = 64,
) -> dict:
    """Rank passage pool with an embedding model for each query.

    Returns dict[query_id -> list[(pid, score)]] sorted descending.
    """
    import torch

    passages = pool["passages"]
    pid_to_idx = pool["pid_to_idx"]
    idx_to_pid = {v: k for k, v in pid_to_idx.items()}

    with torch.no_grad():
        passage_embs = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            embs = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
            passage_embs.append(embs.cpu())
        passage_embs = torch.cat(passage_embs, dim=0)

    rankings = {}
    for pair in pairs:
        qid = pair["id"]
        with torch.no_grad():
            q_emb = model.encode(
                [pair["query"]], convert_to_tensor=True, normalize_embeddings=True,
            ).cpu()
        sims = (q_emb @ passage_embs.T).squeeze(0)
        topk_vals, topk_idxs = sims.topk(min(top_k, len(passages)))
        rankings[qid] = [
            (idx_to_pid[idx.item()], val.item())
            for idx, val in zip(topk_idxs, topk_vals)
        ]

    return rankings


def build_candidate_support(
    pairs: list[dict],
    pool: dict,
    raw_student_ranks: dict,
    top_k: int = 128,
) -> dict:
    """Build per-query candidate support: top-k(raw student) union top-k(BM25 from pairs).

    BM25 candidates come from the original MS MARCO pairs (already BM25-ranked).
    """
    text_to_pid = pool["text_to_pid"]
    positive_pids = pool["positive_pids"]
    support = {}

    for pair in pairs:
        qid = pair["id"]
        pos_pids = positive_pids.get(qid, set())

        student_pids = set()
        if qid in raw_student_ranks:
            for pid, _ in raw_student_ranks[qid][:top_k]:
                if pid not in pos_pids:
                    student_pids.add(pid)

        bm25_pids = set()
        for doc in pair["documents"]:
            norm = doc.strip()
            pid = text_to_pid.get(norm)
            if pid and pid not in pos_pids:
                bm25_pids.add(pid)

        support[qid] = student_pids | bm25_pids

    return support


def build_e16_manifests(
    pairs: list[dict],
    pool: dict,
    raw_student_ranks: dict,
    teacher_ranks: dict,
    candidate_support: dict,
    n_negatives: int = 31,
    teacher_slots: int = 16,
    min_turnover: float = 0.20,
) -> dict:
    """Build per-query E16 manifests with selective boundary inheritance.

    Returns dict with:
      arm_a2: list[dict] — student-mined pairs
      arm_a4: list[dict] — E16 selective pairs
      eligible_count: int
      ineligible_count: int
    """
    positive_pids = pool["positive_pids"]
    pid_to_idx = pool["pid_to_idx"]
    passages = pool["passages"]
    student_replay_slots = n_negatives - teacher_slots

    a2_pairs = []
    a4_pairs = []
    eligible_count = 0
    ineligible_count = 0

    for pair in pairs:
        qid = pair["id"]
        query = pair["query"]
        pos_pids = positive_pids.get(qid, set())
        pos_pid = sorted(pos_pids)[0]
        gold_text = passages[pid_to_idx[pos_pid]]

        cq = candidate_support.get(qid, set())

        student_ranked = [
            (pid, sc) for pid, sc in raw_student_ranks.get(qid, [])
            if pid in cq and pid not in pos_pids
        ]
        teacher_ranked = [
            (pid, sc) for pid, sc in teacher_ranks.get(qid, [])
            if pid in cq and pid not in pos_pids
        ]

        student_top31 = [pid for pid, _ in student_ranked[:n_negatives]]
        a2_neg_texts = [passages[pid_to_idx[pid]] for pid in student_top31]

        teacher_correct = any(
            pid in pos_pids
            for pid, _ in teacher_ranks.get(qid, [])[:1]
        )
        student_correct = any(
            pid in pos_pids
            for pid, _ in raw_student_ranks.get(qid, [])[:1]
        )

        student_top10 = set(pid for pid, _ in student_ranked[:10])
        teacher_top10 = set(pid for pid, _ in teacher_ranked[:10])
        if student_top10 and teacher_top10:
            jaccard = len(student_top10 & teacher_top10) / len(student_top10 | teacher_top10)
            turnover = 1.0 - jaccard
        else:
            turnover = 0.0

        eligible = teacher_correct and not student_correct and turnover >= min_turnover

        if eligible:
            eligible_count += 1
            replay_pids = [pid for pid, _ in student_ranked[:student_replay_slots]]
            replay_set = set(replay_pids)
            teacher_replacements = []
            for pid, _ in teacher_ranked:
                if pid not in replay_set and pid not in pos_pids:
                    all_teacher_scores = dict(teacher_ranks.get(qid, []))
                    if pos_pid in all_teacher_scores:
                        if all_teacher_scores.get(pid, 0) < all_teacher_scores[pos_pid]:
                            teacher_replacements.append(pid)
                    else:
                        teacher_replacements.append(pid)
                    if len(teacher_replacements) >= teacher_slots:
                        break

            fallback_count = 0
            if len(teacher_replacements) < teacher_slots and len(student_ranked) > student_replay_slots:
                extra = [pid for pid, _ in student_ranked[student_replay_slots:]
                         if pid not in replay_set and pid not in set(teacher_replacements) and pid not in pos_pids]
                fallback_count = min(len(extra), teacher_slots - len(teacher_replacements))
                teacher_replacements.extend(extra[:fallback_count])
            if fallback_count > 0:
                print(f"    WARNING: query {qid}: {fallback_count}/{teacher_slots} teacher slots filled by student fallback")

            e16_neg_pids = replay_pids + teacher_replacements
            if len(e16_neg_pids) < n_negatives:
                print(f"    WARNING: query {qid}: only {len(e16_neg_pids)} negatives (need {n_negatives})")
            e16_neg_texts = [passages[pid_to_idx[pid]] for pid in e16_neg_pids[:n_negatives]]
        else:
            ineligible_count += 1
            e16_neg_texts = a2_neg_texts

        rng_a2 = random.Random(stable_int(f"a2_{qid}", 42))
        docs_a2 = [gold_text] + a2_neg_texts[:n_negatives]
        order_a2 = list(range(len(docs_a2)))
        rng_a2.shuffle(order_a2)
        a2_pairs.append({
            "id": qid, "query": query,
            "documents": [docs_a2[i] for i in order_a2],
            "gold_idx": order_a2.index(0),
        })

        rng_a4 = random.Random(stable_int(f"a4_{qid}", 42))
        docs_a4 = [gold_text] + e16_neg_texts[:n_negatives]
        order_a4 = list(range(len(docs_a4)))
        rng_a4.shuffle(order_a4)
        a4_pairs.append({
            "id": qid, "query": query,
            "documents": [docs_a4[i] for i in order_a4],
            "gold_idx": order_a4.index(0),
            "e16_eligible": eligible,
        })

    print(f"  E16: {eligible_count} eligible, {ineligible_count} ineligible "
          f"({eligible_count/(eligible_count+ineligible_count)*100:.1f}% treated)")
    return {
        "arm_a2": a2_pairs,
        "arm_a4": a4_pairs,
        "eligible_count": eligible_count,
        "ineligible_count": ineligible_count,
    }


def build_hardness_shuffle(
    a4_pairs: list[dict],
    a2_pairs: list[dict],
    pool: dict,
    raw_student_ranks: dict,
    selector_seed: int = 16016,
) -> list[dict]:
    """Build Arm 5: hardness-matched shuffle of E16 negatives.

    For each eligible query, swaps the teacher-selected negatives with those from
    another eligible query (derangement), matching by hardness. Ineligible queries
    use exact A2 documents.
    """
    rng = random.Random(selector_seed)
    pid_to_idx = pool["pid_to_idx"]
    passages = pool["passages"]
    positive_pids = pool["positive_pids"]

    eligible_idxs = [i for i, p in enumerate(a4_pairs) if p.get("e16_eligible", False)]

    if len(eligible_idxs) < 2:
        print("  Shuffle: <2 eligible queries, using A2 for all")
        return [dict(p) for p in a2_pairs]

    deranged = list(eligible_idxs)
    for attempt in range(100):
        rng.shuffle(deranged)
        if all(deranged[i] != eligible_idxs[i] for i in range(len(eligible_idxs))):
            break
    else:
        for i in range(len(deranged)):
            if deranged[i] == eligible_idxs[i]:
                j = (i + 1) % len(deranged)
                deranged[i], deranged[j] = deranged[j], deranged[i]

    a5_pairs = []
    for i, p4 in enumerate(a4_pairs):
        if not p4.get("e16_eligible", False):
            a5_pairs.append({
                "id": p4["id"], "query": p4["query"],
                "documents": list(a2_pairs[i]["documents"]),
                "gold_idx": a2_pairs[i]["gold_idx"],
            })
            continue

        local_idx = eligible_idxs.index(i)
        donor_idx = deranged[local_idx]
        donor = a4_pairs[donor_idx]

        qid = p4["id"]
        pos_pids_q = positive_pids.get(qid, set())
        pos_pid = sorted(pos_pids_q)[0]
        gold_text = passages[pid_to_idx[pos_pid]]

        donor_docs = [d for d in donor["documents"]
                      if d != donor["documents"][donor["gold_idx"]]]

        clean_donor = [d for d in donor_docs if d.strip() not in
                       {passages[pid_to_idx[pid]] for pid in pos_pids_q
                        if pid in pid_to_idx}]

        student_ranked = [pid for pid, _ in raw_student_ranks.get(qid, [])
                          if pid not in pos_pids_q]
        student_neg_texts = [passages[pid_to_idx[pid]] for pid in student_ranked[:15]]

        teacher_replaced = clean_donor[:16]
        while len(teacher_replaced) < 16 and len(student_ranked) > 15:
            extras = [passages[pid_to_idx[pid]] for pid in student_ranked[15:]
                      if passages[pid_to_idx[pid]] not in set(student_neg_texts) | set(teacher_replaced)]
            teacher_replaced.extend(extras[:16 - len(teacher_replaced)])
            break

        all_negs = student_neg_texts + teacher_replaced
        all_negs = all_negs[:31]

        rng_a5 = random.Random(stable_int(f"a5_{qid}", selector_seed))
        docs = [gold_text] + all_negs
        order = list(range(len(docs)))
        rng_a5.shuffle(order)
        a5_pairs.append({
            "id": qid, "query": p4["query"],
            "documents": [docs[j] for j in order],
            "gold_idx": order.index(0),
        })

    print(f"  Shuffle: {len(eligible_idxs)} eligible queries deranged")
    return a5_pairs


def validate_e16_manifests(
    a2_pairs: list[dict],
    a4_pairs: list[dict],
    a5_pairs: list[dict],
    pool: dict,
    min_eligible_frac: float = 0.10,
    max_contamination: float = 0.02,
) -> dict:
    """Validate E16 manifest integrity before training."""
    positive_pids = pool["positive_pids"]
    text_to_pid = pool["text_to_pid"]

    eligible = sum(1 for p in a4_pairs if p.get("e16_eligible", False))
    eligible_frac = eligible / len(a4_pairs) if a4_pairs else 0

    contam_a4 = 0
    contam_a5 = 0
    for p4, p5 in zip(a4_pairs, a5_pairs):
        qid = p4["id"]
        pos_pids_q = positive_pids.get(qid, set())
        for doc in p4["documents"]:
            pid = text_to_pid.get(doc.strip())
            if pid and pid in pos_pids_q and doc != p4["documents"][p4["gold_idx"]]:
                contam_a4 += 1
        for doc in p5["documents"]:
            pid = text_to_pid.get(doc.strip())
            if pid and pid in pos_pids_q and doc != p5["documents"][p5["gold_idx"]]:
                contam_a5 += 1

    total_negs = sum(len(p["documents"]) - 1 for p in a4_pairs)
    contam_rate_a4 = contam_a4 / total_negs if total_negs else 0
    contam_rate_a5 = contam_a5 / total_negs if total_negs else 0

    a4_treated = sum(1 for p in a4_pairs if p.get("e16_eligible", False))
    a5_treated = sum(
        1 for i, p5 in enumerate(a5_pairs)
        if i < len(a2_pairs) and p5["documents"] != a2_pairs[i]["documents"]
    )

    checks = {
        "eligible_frac": eligible_frac,
        "eligible_frac_pass": eligible_frac >= min_eligible_frac,
        "contamination_a4": contam_rate_a4,
        "contamination_a5": contam_rate_a5,
        "contamination_pass": contam_rate_a4 <= max_contamination and contam_rate_a5 <= max_contamination,
        "treated_count_a4": a4_treated,
        "n_queries": len(a4_pairs),
        "all_pass": (eligible_frac >= min_eligible_frac
                     and contam_rate_a4 <= max_contamination
                     and contam_rate_a5 <= max_contamination),
    }

    print(f"  Validation: eligible={eligible_frac:.1%}, contam_a4={contam_rate_a4:.3%}, "
          f"contam_a5={contam_rate_a5:.3%}")
    for k, v in checks.items():
        if k.endswith("_pass"):
            status = "PASS" if v else "FAIL"
            print(f"    {k}: {status}")

    return checks


def load_msmarco_pairs(n: int = 500, n_docs: int = 8, seed: int = 42, cache_dir: str | None = None) -> list[dict]:
    """Load query-passage pairs from MS MARCO with BM25 hard negatives."""
    from datasets import load_dataset

    rng = random.Random(seed)

    ds = load_dataset(
        "microsoft/ms_marco", "v2.1",
        split=f"train[:{n * 3}]",
        cache_dir=cache_dir,
    )

    pairs = []
    for row in ds:
        if not row["passages"]["is_selected"] or not any(row["passages"]["is_selected"]):
            continue

        query = row["query"]
        passages = row["passages"]["passage_text"]
        selected = row["passages"]["is_selected"]

        gold_indices = [i for i, s in enumerate(selected) if s == 1]
        if not gold_indices:
            continue
        gold_idx_orig = gold_indices[0]
        gold_doc = passages[gold_idx_orig]

        negatives = [p for i, p in enumerate(passages) if i != gold_idx_orig and len(p.strip()) > 20]
        if len(negatives) < 2:
            continue

        docs = [gold_doc] + negatives[:n_docs - 1]
        rng.shuffle(docs)
        gold_idx = docs.index(gold_doc)

        pairs.append({
            "id": f"msmarco_{len(pairs)}",
            "query": query,
            "documents": docs,
            "gold_idx": gold_idx,
        })
        if len(pairs) >= n:
            break

    return pairs


def load_nq_pairs(n: int = 500, n_docs: int = 8, seed: int = 42, cache_dir: str | None = None) -> list[dict]:
    """Load Natural Questions pairs (via BeIR format)."""
    from datasets import load_dataset

    rng = random.Random(seed)

    ds = load_dataset(
        "BeIR/nq", "corpus",
        split=f"train[:{n * 5}]",
        cache_dir=cache_dir,
    )

    queries_ds = load_dataset(
        "BeIR/nq", "queries",
        split="train",
        cache_dir=cache_dir,
    )

    corpus = {row["_id"]: row["text"] for row in ds}
    queries = [(row["_id"], row["text"]) for row in queries_ds]

    # This is a simplified loader — for real experiments, use the qrels
    pairs = []
    for qid, qtext in queries[:n]:
        if len(qtext.strip()) < 10:
            continue
        sample_docs = rng.sample(list(corpus.values()), min(n_docs, len(corpus)))
        pairs.append({
            "id": f"nq_{qid}",
            "query": qtext,
            "documents": sample_docs,
            "gold_idx": 0,
        })
        if len(pairs) >= n:
            break

    return pairs


def load_pairs(source: str, **kwargs) -> list[dict]:
    """Unified loader. source can be a path or a dataset name."""
    if os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    loaders = {
        "toy": lambda **kw: load_hard_toy(**kw),
        "msmarco": lambda **kw: load_msmarco_pairs(**kw),
        "nq": lambda **kw: load_nq_pairs(**kw),
    }

    if source in loaders:
        return loaders[source](**kwargs)

    raise ValueError(f"Unknown data source: {source}. Use a file path or one of: {list(loaders.keys())}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", default="toy")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/pairs.jsonl")
    args = parser.parse_args()

    pairs = load_pairs(args.source, n=args.n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} pairs to {args.out}")
