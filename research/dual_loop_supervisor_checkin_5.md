# Dual-Loop Supervisor Check-in #5: Evidence-Native v0 Post-Mortem

Date: 2026-07-07

## Executive Summary

**Evidence-Native v0 prototype exists and FAILED ALL GATES.**

The 10M parameter evidence-conditioned judge scored 38.18% overall — worse than
no evidence (38.53%), worse than shuffled evidence (38.72%), and worse than BM25
overlap (41.41%). Teacher-mix didn't rescue it. The no-evidence-trained control
is actually a BETTER evidence user.

**Direction is on life support, not dead.** The v0 prototype tested a weak version
of the claim (10M frozen-codec mean-pool classifier) not the strong version (121M
judge with cross-attention evidence factorization). But burden of proof has shifted
decisively.

## The Numbers

### Evidence-Trained Judge (M_evidence)

| Condition | Overall | HellaSwag | PIQA |
|-----------|--------:|----------:|-----:|
| retrieved | 38.18% | 25.98% | 50.39% |
| none | 38.53% | 25.39% | 51.66% |
| shuffled | 38.72% | 25.39% | 52.05% |
| wrong_topic | 37.55% | 24.02% | 51.07% |
| gold | 38.04% | 25.29% | 50.78% |

### No-Evidence-Trained Control (M_none)

| Condition | Overall | HellaSwag | PIQA |
|-----------|--------:|----------:|-----:|
| retrieved | 40.28% | 27.44% | 53.12% |
| none | 38.38% | 26.95% | 49.80% |
| shuffled | 38.18% | 26.07% | 50.29% |
| wrong_topic | 38.67% | 26.27% | 51.07% |
| gold | 38.57% | 26.46% | 50.68% |

### Dumb Baselines

| Baseline | Overall |
|----------|--------:|
| BM25 overlap | 41.41% |
| shortest candidate | 39.89% |
| kNN train label | 39.21% |
| unigram frequency | 38.87% |
| majority label | 36.57% |

### Gate Results

| Gate | Required | Actual | Status |
|------|----------|--------|--------|
| evidence beats no-evidence | >= +5pp | -0.34pp | **FAIL** |
| evidence beats dumb baselines | >= +3pp | -3.22pp | **FAIL** |
| shuffled worse than retrieved | positive gap | -0.54pp | **FAIL** |

## The Most Damning Result

```
M_none(retrieved) - M_none(none) = +1.90pp  → evidence HELPS
M_evidence(retrieved) - M_evidence(none) = -0.34pp → evidence HURTS

M_none(retrieved) - M_none(shuffled) = +2.10pp → proper sensitivity
M_evidence(retrieved) - M_evidence(shuffled) = -0.54pp → REVERSED
```

Evidence-conditioned training made the model WORSE at using evidence. This directly
contradicts the steelman ("evidence training develops judgment geometry").

## Q-Loop B8 Analysis (56 total iterations)

Q-Loop B8 scored its B7 predictions:
- Retrieved lifts HellaSwag: **Mostly refuted** (tiny +0.68pp, below controls)
- Dumb baselines explain lift: **Confirmed harder than predicted** (BM25 wins by 3.12pp)
- Shuffled controls don't collapse: **Confirmed catastrophically** (shuffled > retrieved)
- First prototype won't prove byte-native: **Confirmed**

Q-Loop B8 set 10 precise conditions for Batch 9 survival, including:
- M_evidence(retrieved) >= M_none(retrieved) + 3pp (internalization gate)
- M_evidence(none) >= M_none(none) + 2pp (transfer gate)
- True gold evidence ceiling >= 35% HellaSwag
- 3 seeds with paired significance tests
- Clean corpus with explicit contamination conditions

Verdict: **"Evidence-Native Sutra is on life support."**

## Supervisor Assessment

### What the W-Loop built (credit)

The prototype is well-engineered: 1064 lines, complete pipeline with BM25 retrieval,
leakage audit, all 5 evidence conditions, 5 dumb baselines, teacher-mix variant,
and pre-committed gates. This is good science — the negative result is clean and
informative. The W-Loop did exactly what was asked.

### What slipped past Codex's skepticism

1. **The model is too simple to test the steelman.** Independent candidate scoring
   with mean pooling gives the model no mechanism to compare candidates against
   evidence jointly. The architecture was designed for fast falsification, not for
   testing whether evidence changes internal geometry.

2. **The corpus is circular.** Training examples (contexts + choices) are used as
   evidence documents. This means the model retrieves its own training data as
   "evidence." Future runs need external-only evidence conditions.

3. **Gold evidence is fake gold.** Label-conditioned BM25 retrieval is not true
   gold evidence (a passage sufficient to determine the correct answer). Real gold
   requires teacher-generated rationales or human-curated decisive passages.

4. **No geometry probes were run.** The claim is about internal reasoning geometry,
   but no representation analysis, counterfactual probing, or evidence-sensitivity
   measurement was performed.

### What the Q-Loop got right (credit)

Q-Loop B7-B8 predicted this failure pattern almost exactly:
- Dumb baselines explain more lift than expected ✓
- Shuffled controls don't fully collapse ✓ (they REVERSED)
- First prototype won't prove byte-native advantage ✓
- Small classifiers competitive ✓

The Q-Loop has been right about predictions for 4 consecutive batches.

### Narrative Gate

**Honest headline after 60 work + 56 question iterations:**

"The evidence-native judge opened the book, memorized the worksheet, and lost to a
keyword search. But the book was bad, the judge was tiny, and the real question —
does evidence training change the model's mind — hasn't been tested at a scale that
could answer it."

**Does it survive "that's obvious"?** No. The current result is "a small classifier
failed to use retrieved snippets." That's expected.

**Does it survive "that's trivial"?** No. BM25 overlap beating a neural judge is
exactly the trivial failure mode.

**Narrative status: DEAD for v0. ALIVE for the refined thesis, but untested.**

The refined thesis (from Devansh): Intelligence = reasoning geometry + factual knowledge.
Evidence training should change what the model learns INTERNALLY, not just what it can
access externally. This has NOT been tested because:
1. The architecture doesn't force evidence-dependent representations
2. No internalization probes were run
3. The scale (10M) may be too small for representation learning
4. The training data (1024 examples) is far too small for generalization

## Decision: Continue with Major Redesign

**Evidence-native v0 architecture: KILLED.**
**Evidence-native direction: ONE MORE SERIOUS SHOT.**

Rationale:
1. The v0 failure is architectural (mean-pool classifier), not theoretical
2. The closed-book control's positive evidence sensitivity (+1.9pp) proves evidence
   CAN help a byte-native model — the training procedure was the problem, not evidence itself
3. Q-Loop B8's 10 conditions provide a clear, pre-committed falsification framework
4. The refined thesis (geometry vs knowledge decomposition) is genuinely novel and untested

**W-Loop B7 must build v1 with:**
- Explicit evidence factorization (separate encoders, cross-attention)
- Internalization gate: M_evidence(none) >= M_none(none)
- True gold evidence ceiling (teacher-generated rationales)
- Clean external corpus (no training examples as evidence)
- 3 seeds minimum
- Geometry probes (representation analysis, counterfactual evidence)

**If v1 fails the internalization gate, evidence-native is demoted permanently.**

## Updated Confidence Table

| Claim | Confidence |
|-------|-----------|
| Brainseed v0 extraction is dead | **CONFIRMED** |
| Evidence-native v0 classifier is dead | **CONFIRMED** |
| Evidence can help byte-native model | MODERATE (M_none control shows +1.9pp) |
| Evidence training changes internal geometry | LOW (v0 says opposite) |
| Evidence-native v1 will pass internalization gate | LOW-MODERATE |
| Project will produce stop-scrolling result | LOW (honest) |
| Direction is worth one more serious shot | MODERATE |

## Batch 7/9 Launch Orders

**W-Loop B7 (iterations 61-70): Evidence-Native v1 — Architecture Redesign**

Build the evidence-factorized judge: separate context/evidence/candidate encoding,
cross-attention for evidence binding, explicit internalization controls. Test the
steelman directly.

**Q-Loop B9 (iterations 57-63): Already launched.**

Determining whether evidence-native should continue, be demoted, or killed. Will
set final gates for the direction.
