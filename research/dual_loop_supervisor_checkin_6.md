# Dual-Loop Supervisor Check-in #6: Evidence-Native v1 DEAD, Coordinate-Inheritance Formal Acceptance

Date: 2026-07-07

## Executive Summary

**Evidence-Native v1 failed ALL pre-committed kill gates. The direction is permanently dead.**

**Coordinate-inheritance is formally promoted to sole mainline.** This is not because it has
proven itself — it has not. It is the last direction standing after 70+ question-loop and 70+
work-loop iterations systematically killed everything else.

## Evidence-Native v1 Results (2 of 3 seeds completed)

The v1 prototype used cross-attention evidence factorization, clean external corpus
(decontaminated shard data), 4K training examples (2K HellaSwag + 2K PIQA), 3-seed design
(2 completed: 20260707, 42), and separate M_evidence / M_none training conditions.

Architecture: 9.2M total params, 4.7M trainable. Frozen codec + 1-layer 128-dim reasoner
with cross-attention evidence binding. This addressed every v0 diagnosis: separate evidence
encoding, clean corpus, expanded training data, proper controls.

### Seed 20260707

| Condition | M_evidence | M_none | Delta |
|-----------|-----------|--------|-------|
| retrieved | 39.79% | 38.72% | +1.07pp |
| none | 39.60% | 38.57% | +1.03pp |
| shuffled | 39.40% | 38.57% | +0.83pp |
| wrong_topic | 39.26% | 38.96% | +0.30pp |
| gold | 39.84% | 38.48% | +1.36pp |

### Seed 42

| Condition | M_evidence | M_none | Delta |
|-----------|-----------|--------|-------|
| retrieved | 38.72% | 38.28% | +0.44pp |
| none | 38.23% | 38.33% | -0.10pp |
| shuffled | 38.77% | 38.48% | +0.29pp |
| wrong_topic | 38.82% | 37.70% | +1.12pp |
| gold | 38.62% | 38.43% | +0.19pp |

### Dumb Baselines

| Baseline | Overall |
|----------|--------:|
| shortest_candidate | 39.31% |
| bm25_evidence_overlap | 39.31% |
| nearest_neighbor | 38.72% |
| unigram_frequency | 38.28% |
| majority_label | 34.91% |

### Pre-Committed Gate Results

| Gate | Required | Seed 20260707 | Seed 42 | Average | Status |
|------|----------|---------------|---------|---------|--------|
| INTERNALIZATION: M_evidence(none) >= M_none(none) + 2pp | +2.00pp | +1.03pp | -0.10pp | **+0.47pp** | **FAIL** |
| EVIDENCE_USE: M_evidence(ret) >= M_none(ret) + 3pp | +3.00pp | +1.07pp | +0.44pp | **+0.76pp** | **FAIL** |
| BASELINE: M_evidence(ret) >= best_dumb + 5pp | 44.31% | 39.79% | 38.72% | **39.26%** | **FAIL** |
| SENSITIVITY: M_evidence(ret) - M_evidence(shuf) >= 3pp | +3.00pp | +0.39pp | -0.05pp | **+0.17pp** | **FAIL** |

### The Verdict

The critical INTERNALIZATION gate — "does evidence training change internal reasoning
geometry?" — averaged +0.47pp across two seeds, with one seed actually going negative.
The pre-committed threshold was +2pp. This is noise, not signal.

Evidence sensitivity (+0.17pp average, one seed negative) is indistinguishable from zero.
The model cannot tell retrieved evidence from shuffled garbage.

The v1 fixes (cross-attention, clean corpus, 4K examples) improved absolute numbers slightly
over v0 but did NOT change the fundamental pattern: evidence-conditioned training does not
produce measurable internalized reasoning geometry changes.

**Per the pre-committed contract from supervisor check-in #5:**
> "If v1 fails the internalization gate, evidence-native is demoted permanently."

```
KILL_EVIDENCE_NATIVE_V1
KILL_EVIDENCE_NATIVE_DIRECTION_PERMANENTLY
```

## Q-Loop B10 Summary (Coordinate-Inheritance Attack)

Q-Loop B10 (7 iterations, 64-70) attacked coordinate-inheritance from 7 angles:
1. "That's just CBD/distillation" — valid; must prove geometry is load-bearing, not just init
2. "Codec destroys geometry" — real risk; gauge preflight is stage 1
3. "Controls are too easy to game" — must include pretrained + same-family controls
4. "Compression kills geometry" — 5x compression may destroy the signal
5. "Where's the moonshot?" — needs to prove transferable geometry, not just good init
6. "Byte-native is branding" — tokenized sibling must be compared
7. "Gauge dependence again" — same Brainseed failure mode possible

**Verdict: MAINLINE_BUT_NOT_BELIEVED**

Coordinate-inheritance is the strongest remaining path by elimination. The Q-Loop set a
rigorous 5-stage gate system:

- **Stage 0**: Artifact requirements (3 seeds, 6+ benchmarks, paired stats, compute accounting)
- **Stage 1**: Codec/gauge preflight (>=2 nats/token copied advantage, >=60% gap closure)
- **Stage 2**: Uncompressed byteified inheritance (>=35% HellaSwag, >=+8pp over Wide7)
- **Stage 3**: 121M compression gate (>=35% HellaSwag at 121M, >=50% lift retention)
- **Stage 4**: Byte-native story gate (robustness/cross-tokenizer advantages)
- **Stage 5**: Moonshot promotion gate (beat SmolLM2-135M targets)

## Supervisor Assessment

### What the W-Loop B7 built (credit)

The v1 codebase is substantially improved over v0: cross-attention evidence factorization,
decontaminated external corpus, proper M_evidence/M_none paired controls, expanded training
data, and all 5 evidence conditions. The Codex session managed the full pipeline autonomously
and the 2 completed seeds are clean and informative. Third seed was in progress when the
Codex session timed out; Python processes were stuck and killed after hours with no progress.

### What slipped past Codex's skepticism

1. **Scale excuse is dead.** The v0 excuse was "10M is too small." v1 used the same scale
   (9.2M) with better architecture. If the thesis required more params, it contradicts
   "Intelligence = Geometry, not Scale." The evidence-native direction has had its shot at
   every scale we can afford.

2. **The architectural fix wasn't enough.** Cross-attention vs concatenation did not change
   the fundamental pattern. The model still can't use evidence productively after being
   trained on it.

3. **The direction was already dead after v0.** The Q-Loop B9 pivot to coordinate-inheritance
   was correct. Evidence-native v1 was a kill-gated formality, not a genuine bet.

### Q-Loop B10 quality

The 5-stage gate system is the most rigorous kill framework the Q-Loop has produced. The
distinction between "last thing standing" vs "genuinely strongest" is honest and important.
The predictions for what the W-Loop will find are concrete and falsifiable.

### Narrative Gate

**Honest headline after 70+ work + 70+ question iterations:**

"Evidence training does not change how a small byte-native model reasons — not with
concatenation (v0), not with cross-attention (v1), not with clean corpora, not with more
data. The direction is conclusively dead."

**Does the coordinate-inheritance narrative survive "that's obvious?"**

Not yet. "We compressed a bigger model and it still works" is obvious. The narrative becomes
non-obvious ONLY if coordinate-disruption controls prove the SPECIFIC geometry is load-bearing.
That is what the W-Loop B8 must test.

**Does it survive "that's trivial?"**

Not yet. CBD already gets 42.65% HellaSwag at 138M. Coordinate-inheritance must show something
CBD doesn't: that the geometry transfers through bytes, survives compression, and degrades
predictably when disrupted.

## Decision: Pivot to Coordinate-Inheritance

### Evidence-native: PERMANENTLY DEAD

```
KILL_EVIDENCE_NATIVE_V0 (confirmed check-in #5)
KILL_EVIDENCE_NATIVE_V1 (this check-in)
KILL_EVIDENCE_NATIVE_DIRECTION
```

Evidence-native has been tested across:
- Two architectures (mean-pool v0, cross-attention v1)
- Two corpus designs (circular v0, decontaminated external v1)
- Two training scales (1K examples v0, 4K examples v1)
- Five evidence conditions each time
- Pre-committed gates, independently verified

The pattern is consistent: evidence training does not internalize reasoning geometry changes
in small byte-native models. The direction is falsified.

### Coordinate-Inheritance: FORMAL MAINLINE

The sole remaining direction. Inherits pretrained reasoning coordinates from teacher,
byteifies through codec, proves geometry is load-bearing via disruption controls.

**Only empirical signal so far:** +1.7 nats/token copied-vs-random advantage (Batch 5
chain-init probe). Weak but in the right direction.

**What must be proven (from Q-Loop B10 stage system):**
1. Codec preserves geometry with calibration (>=2 nats/token copied advantage)
2. Uncompressed byteified model beats Wide7 on real benchmarks
3. 121M compressed model retains >=50% of uncompressed lift
4. Disruption controls (random, shuffled, rotated, generic pretrained) lose the advantage
5. Byte-native advantage survives comparison with tokenized sibling

### Confidence Table

| Claim | Confidence |
|-------|-----------|
| Evidence-native is dead | **CONFIRMED** (two architectures, all gates fail) |
| Brainseed is dead | **CONFIRMED** (check-in #3) |
| Chain-init shows positive signal | MODERATE (+1.7 nats/token, single probe) |
| Coordinate-inheritance will pass Stage 1 | MODERATE (calibrated adapter should close gap) |
| Coordinate-inheritance will pass Stage 2 | LOW-MODERATE (uncompressed could work) |
| Coordinate-inheritance will pass Stage 3 | LOW (5x compression is severe) |
| Project will produce stop-scrolling result | LOW (honest) |

## Batch 8/11 Launch Orders

### W-Loop B8 (iterations 71-80): Coordinate-Inheritance Prototype

Build the first real coordinate-inheritance prototype:
1. Codec gauge preflight — copied vs random NLL at token-end and patch-boundary
2. Calibration adapter — small affine/RMSNorm/low-rank to close codec-to-true-embedding gap
3. Inherited Qwen layers through codec inputs — frozen core + byte adapter
4. Full disruption controls: random init, shuffled layers, rotated, generic pretrained
5. HellaSwag/PIQA/ARC evaluation at full (uncompressed) scale
6. If Stage 1 passes, proceed to Stage 2 uncompressed benchmark gate

### Q-Loop B11 (iterations 71-77): Attack Coordinate-Inheritance Deeper

Continue attacking coordinate-inheritance with fresh angles:
- How does the calibration adapter change the "just distillation" story?
- If calibration closes most of the gap, what's left for the inherited layers to do?
- How much of the benchmark lift comes from the adapter vs the inherited geometry?
- What does "transferable coordinate system" mean operationally vs "good initialization"?
- Can you separate "reasoning geometry" from "memorized training statistics"?

## Updated State

After 77+ Q-Loop iterations and 70+ W-Loop iterations:
- Brainseed: DEAD (confirmed check-in #3)
- Evidence-native v0: DEAD (confirmed check-in #5)
- Evidence-native v1: DEAD (this check-in)
- Evidence-native direction: PERMANENTLY DEAD
- Chain-init probe: weak positive (+1.7 nats/token)
- Coordinate-inheritance: MAINLINE (untested beyond probe)
- CBD competitor: 42.65% HellaSwag at 138M (the number to beat)
