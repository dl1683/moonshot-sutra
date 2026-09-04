"""Data loading for Eklavya embedding experiments.

Supports:
  - Toy data (built-in, for smoke tests)
  - Hard-negative toy data (semantically similar distractors)
  - MS MARCO passage retrieval (real data)
  - NQ (Natural Questions) via BeIR
  - Custom JSONL files

Each loader returns list[dict] with keys: id, query, documents, gold_idx
"""
from __future__ import annotations

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
