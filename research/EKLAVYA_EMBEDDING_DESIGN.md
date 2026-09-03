# Eklavya for Embeddings: Design Draft v2

**Status:** Design draft, integrating Sangam research and broadened Eklavya philosophy.
**Last updated:** 2026-09-03

## Philosophical Foundation

Eklavya is not probe-based tomography specifically. Eklavya is the art of
learning from existing models by watching what they do — stealing the useful
principles, not copying the outputs. Probes are one instrument. Response jets
are another. Any mechanism that reveals owned invariants from teacher behavior
is Eklavya.

Sangam's research independently arrived at the same insight: "Treat every
frozen neural network as an experimental instrument. Harvest the stable
functional responses it reveals under controlled probes and interventions; do
not treat its coordinates or answers as truth." (Sangam Failure Map, 2026-08-31)

## The Conditional Support Discovery (from Sangam)

Sangam's Flickr30k teacher-kernel atlas result is the strongest existing
evidence about what multi-teacher transfer actually looks like:

- Teacher identity attribution PASSES: the mechanism captured sample-specific
  8B relational structure, not just a spectrum or regularizer
- Student usefulness FAILS globally: the teacher atlas is -0.000883 below raw
  2B on the mean
- Per-identity diagnosis: helps 4,059 identities, ties 2,146, hurts 3,795
- The gains concentrate in the hardest raw-2B deciles; easy examples regress

**The conclusion: teacher knowledge is conditionally useful.** A global transfer
always fails because it helps some examples and hurts others. The real question
is not "what does the teacher know?" but "when does the teacher know something
the student doesn't?"

This shapes everything about Eklavya embeddings: the student must learn to
recognize WHEN teacher structure is useful and absorb only that.

## Core Translation

MCQ Eklavya: distills `p_T(candidate | probe(x))` — probability over candidates.
Embedding Eklavya: distills behavior under interventions — how rankings,
neighborhoods, and margins respond to controlled input changes.

The invariant is the response surface: what stays stable, what changes, how
margins shift. Not the embedding vector. Not the similarity score. The
*pattern of behavior* under controlled perturbation.

## Measurement Object: Functional Response Jets

Adapted from Sangam's response jet framework. For teacher T, query q, document
set D, and intervention g:

```
zeroth order:     J0(T,q,D)    = rank_T(sim(q, D))
first response:   Jg(T,q,D)   = rank_T(sim(g(q), D)) - J0
mixed interaction: Jgh(T,q,D) = rank_T(sim(g(h(q)), D)) - Jg - Jh - J0
stability:        S(T,q,D)    = agreement over meaning-preserving transforms
support:          U(T,q,D)    = calibrated confidence, instability markers
```

These operate on bounded rank percentiles and ordinal pair flips, not raw
cosine scores. This makes them comparable across teachers with different
score scales, dimensionalities, and training objectives.

Key: Jgh (mixed interaction) can reveal that combining paraphrase + domain
shift produces a ranking change that neither single intervention predicts.
This is where teacher-specific knowledge lives.

## Intervention Families (Text Embeddings)

```
interventions:
  meaning_preserving:
    paraphrase:     q -> semantically equivalent rephrasing
    typo_noise:     q -> surface corruption (typos, case changes)
    length_expand:  q -> verbose version with same meaning
    length_compress:q -> minimal version preserving intent

  meaning_changing:
    negation:       q -> negated intent ("find X" -> "find not X")
    entity_swap:    q -> replace key entity with plausible alternative
    domain_shift:   q -> same concept in different domain language
    specificity:    q -> broaden or narrow the query scope

  structural:
    instruction_prefix: q -> prepend retrieval instruction
    format_change:  q -> restructure (bullet to prose, etc.)
```

Expected behavior under good teacher:
- Meaning-preserving: rankings ~invariant (J_para ≈ 0)
- Meaning-changing: rankings shift (J_neg ≠ 0, J_entity ≠ 0)
- The magnitude and pattern of shifts is the informative object
- Teacher disagreement on Jg matters more than teacher agreement on J0

## Three Modes of Stealing (Not Just Probes)

### Mode 1: Response Surface Matching (Probe-Based)
Match the teacher's ranking response jets: how rankings change under
interventions. This is the direct Eklavya mechanism.

### Mode 2: Selective Knowledge Distillation
Use teacher signals as curriculum: let teachers identify hard examples,
informative negatives, and confidence-calibrated targets. The student
learns from the training data the teacher selects, not from the teacher's
outputs directly.

Teachers as curriculum selectors:
- Hard negative mining: teachers identify near-miss documents
- Difficulty labeling: teachers identify examples where the student is
  weakest (student-teacher gap as curriculum signal)
- Quality filtering: teachers identify training pairs that are too easy
  or too noisy

### Mode 3: Functional Concept Transfer
Adapted from Sangam's functional concept atlas. Decompose teacher behavior
into shared and private concepts:

- Shared features: response patterns that survive across independent
  teacher families (the real invariants)
- Factor-shared features: common within a model class (e.g., all BERT
  encoders agree) but different across classes
- Teacher-private features: useful dissent — one teacher sees something
  others miss
- Unsupported cells: explicitly missing, never zero

The student absorbs shared features as geometry and teacher-private
features as conditional capacity (only activated when the student
recognizes a context where that teacher was distinctively informative).

## Loss Architecture

### Primary: Ranking Response Surface KL
```
L_jet = Σ_T Σ_g w(T,g) * KL(
  softmax(rank_T(sim_T(g(q), D))/τ)
  ||
  softmax(rank_S(sim_S(g(q), D))/τ)
)
```
Where w(T,g) is a learned or scheduled weight that reflects the conditional
usefulness of teacher T under intervention g.

### Secondary: Invariance/Sensitivity Consistency
```
L_inv = Σ (|J_S(para) - J_T(para)|^2) + Σ (|J_S(neg) - J_T(neg)|^2)
```
The student should change its rankings in the same direction and magnitude
as the teachers when interventions are applied.

### Tertiary: Base Contrastive
Standard contrastive (InfoNCE) on (query, positive, negative) to maintain
general embedding quality independent of teacher signals.

### Quaternary: Conditional Support Regularization
Penalize teacher-dependent gains. If the student's quality on a subset
requires continued teacher access at inference, it is not owned:
```
L_conditional = max(0, quality_with_teacher - quality_without_teacher - ε)
```

## Multi-Teacher Strategy

**Do NOT average.** The Sangam evidence directly shows why averaging fails.

Instead:
1. Keep teacher identities separate (teacher axis preserved in training)
2. Discover where teachers agree vs disagree via response jets
3. Teacher agreement on Jg → high-confidence shared invariant → student
   must absorb this
4. Teacher disagreement → interesting region:
   - If one teacher is provably better (controlled eval), route to that
     teacher's signal
   - If disagreement is genuine ambiguity, present both views to the
     student as conditional capacity
   - If disagreement is noise (same teacher family echo), deduplicate
5. Conditional teacher weighting: per-query-type learned routing that
   determines which teacher is informative for which input class

## Retained Gain Protocol

```
base_score   = MTEB(student_before_any_teacher_exposure)
during_score = MTEB(student_during_teacher_training)
after_score  = MTEB(student_after_all_teacher_access_removed)
control_score= MTEB(matched_control: same data+compute, standard contrastive)

owned_gain = after_score - base_score
control_adjusted_gain = after_score - control_score
retention_ratio = owned_gain / max(during_score - base_score, ε)
```

Teacher removal = no teacher embedding calls at inference. Student must
produce its own embeddings from its own parameters.

**Slice the gain by:**
- Teacher family (which teacher contributed what)
- Intervention family (which probes were informative)
- Query type / difficulty decile (conditional support evidence)
- MTEB task family (does it generalize across task types)

## First Student Candidate

Start small to iterate fast. Embedding-only (encoder), not decoder:
- `all-MiniLM-L6-v2` (22M params, 384-dim) as the smallest test bed
- `BGE-small-en-v1.5` (33M params) as a slightly larger alternative
- Or train from scratch: 6-layer, 384-dim transformer encoder

Matryoshka support from the start (variable output dimensions: 384, 256,
128, 64) — this is standard practice in modern embedding models.

## First Teacher Candidates (RTX 5090 runnable)

Heterogeneous teachers maximize Eklavya value. Three families:

1. **Decoder-derived:** `Qwen3-Embedding-0.6B` (LLM-derived geometry,
   instruction-tuned, different inductive bias from encoders)
2. **Encoder-native:** `BGE-large-en-v1.5` (335M, BERT-like, trained on
   contrastive pairs, different geometry from decoder-derived)
3. **Contrastive-specialist:** `snowflake-arctic-embed-l-v2` (~335M,
   strong contrastive+distilled training, yet another approach)

These three teach different geometries. Eklavya should find invariants
that survive across all three (shared features) and distinctive
knowledge in each (teacher-private features).

## Kill Criteria

This direction is dead if:

1. **Absorption by augmentation:** Probe-based interventions produce ≤0
   control-adjusted gain vs standard contrastive training with the same
   augmented queries as training data
2. **Absorption by naive averaging:** A simple average of teacher
   similarity scores matches or beats the response surface approach
3. **No retained gain:** Student quality after teacher removal is ≤0
   (student didn't learn anything owned)
4. **No probe value:** Identity probe (no perturbation, standard KD)
   matches or beats the full intervention family
5. **No conditional support:** The teacher signal helps and hurts in
   equal measure with no learnable selection rule (the Sangam Flickr30k
   pattern without a fix)

## Absorption Ladder (Hostile Baselines)

```
absorption_baselines:
  B0_random_init:     student with random init, no training
  B1_contrastive_only: standard contrastive training, same data+compute
  B2_single_best_teacher: KD from the single strongest teacher only
  B3_naive_teacher_avg: simple average of all teacher similarities
  B4_augmented_contrastive: contrastive with same augmented queries (probes as
    data augmentation — this is the "strongest boring explanation")
  B5_conditional_single: best teacher selected per query type (oracle routing)
```

Eklavya must beat B4 (augmented contrastive) and B5 (oracle single teacher)
to prove the response surface approach adds value beyond data augmentation
and teacher selection.

## Narrative

"A 30M-parameter embedding model that matches models 10x its size — not by
copying their answers, but by watching how they think when you ask the same
question different ways."

## Strongest Boring Explanation

Standard multi-teacher contrastive distillation with augmented queries
(the interventions are just data augmentation for contrastive pairs). If
baseline B4 matches the response surface approach, the Eklavya structure
is cosmetic.

This IS the absorption test. B4 must be built first and beaten.

## From Sangam: Open Research Questions

1. **Can conditional support be learned?** Sangam showed teacher knowledge
   helps some examples and hurts others. Can a student learn to recognize
   which examples benefit from which teacher? If yes, this is the Eklavya
   breakthrough. If no, multi-teacher embedding distillation is fundamentally
   limited.

2. **Response jets vs coordinates:** Sangam showed that coordinate regression,
   global rotations, and fixed teacher triplets repeatedly fail. Do response
   jets (behavioral patterns under interventions) survive where coordinates
   don't?

3. **Shared/private decomposition:** Can we identify which response patterns
   are truly shared across model families vs teacher-private? The shared ones
   are the most trustworthy invariants; the private ones need conditional
   support evidence.
