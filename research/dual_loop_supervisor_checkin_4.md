# Dual-Loop Supervisor Check-in #4: The Pivot

Date: 2026-07-07

## Executive Summary

**Brainseed v0 is confirmed dead.** All downstream scorers (ridge, MLP, bilinear,
learned cosine) lose to codec-only. Zero-cost baselines don't rescue the chart.
Track A exhausted.

**Chain-init shows weak positive signal.** Copied Qwen layers beat random layers
by ~1.7 nats/token through codec inputs. Not benchmark-capable yet.

**Q-Loop recommends Evidence-Native Retrieval-Born Sutra as the moonshot mainline.**
W-Loop recommends byteified chain-init as the pragmatic next step.

**Supervisor decision: BOTH, with clear roles.**

## The Tension

The two loops disagree on direction:

| Loop | Recommendation | Rationale |
|------|---------------|-----------|
| Q-Loop B6 (42 iterations) | Evidence-Native Sutra | Serves Vision, novel narrative, achievable on 5090 |
| W-Loop B5 (50 iterations) | Byteified chain-init | Only path with positive signal, CBD validates it |

The Q-Loop attacks chain-init as "not a moonshot — just engineering." The W-Loop
pragmatically notes it's the only path with a positive data point.

## Supervisor Resolution

**Evidence-Native Retrieval-Born Sutra = THE MOONSHOT.**
**Chain-init = THE STRONG BASELINE / FALLBACK.**

Rationale:
1. Q-Loop has been right about predictions 3 consecutive times. Its direction
   judgment is highly credible.
2. The Vision says "Intelligence = Geometry, not Scale." Evidence-native directly
   embodies this: geometry of evidence-conditioned judgment, not compressed memory.
3. Chain-init without novelty is CBD-with-bytes. That's a paper, not a moonshot.
4. But chain-init IS useful as the number to beat. If evidence-native Sutra can't
   beat a chain-init baseline, it's not the moonshot either.

**The guiding question for the next 5 batches:**

```
Can a 121M byte-native evidence-conditioned judge produce a large,
control-resistant benchmark lift by using retrieved public evidence,
or do retrieval artifacts and dumb rerankers explain the gain?
```

## Track A Final Autopsy (Work Loop B5, iterations 41-45)

### Zero-Cost Baselines (iteration 41)

| Method | Patch target top-1 | Top-10 |
|--------|------------------:|-------:|
| Current patch state | 37.89% | 57.28% |
| Nearest preceding token-end | 18.05% | 21.62% |
| Local mean last 4 | 26.64% | 43.53% |
| Local max last 4 | 22.07% | 38.48% |
| Prev token-end + hidden linear | 38.94% | 54.29% |

No zero-cost method reaches 40% patch top-1. Phase 1.5 retraining was not
unnecessary plumbing — it IS the best available chart at consumed positions.
But "best available" is still below the toy cliff.

### Offset Slices (iteration 42)

| Offset | N | Top-1 | Top-10 |
|--------|----:|------:|-------:|
| First byte (hardest) | 976 | 20.08% | 34.63% |
| Offset 1 | 850 | 31.65% | 57.18% |
| Offset 2 | 790 | 48.86% | 91.52% |
| Offset 3+ | 1480 | 60.81% | 90.07% |
| Last byte | 970 | 83.20% | 95.67% |

Phase 1.5 DID learn mid-token prediction (offset 1-2 are well above chance).
But first-byte positions remain weak (20.08%). The aggregate 37.89% is carried
by near-token-end positions.

### All Scorers vs Codec-Only (iteration 43)

| Scorer | HellaSwag | PIQA | vs codec-only |
|--------|----------:|-----:|---------------|
| Codec-only | 28.71% | 51.66% | baseline |
| MLP PCA-256 | 24.51% | 50.10% | **-4.20pp / -1.56pp** |
| Bilinear rank 32 | 23.44% | 49.41% | **-5.27pp / -2.25pp** |
| Bilinear rank 64 | 25.20% | 50.88% | **-3.51pp / -0.78pp** |
| Learned weighted cosine | 25.98% | 48.44% | **-2.73pp / -3.22pp** |

Every learned scorer HURTS compared to raw codec cosine similarity. The
extracted teacher-margin signal is not just weak — it's actively misleading.
This is the definitive death of Brainseed extraction.

### Track A Verdict

```
BRAINSEED_DEAD_AS_BIRTH_ARTIFACT
BRAINSEED_COMPONENT_ONLY_AS_CODEC_DIAGNOSTIC
```

## Track B Chain-Init Probe (Work Loop B5, iterations 46-50)

### Compatibility Signal

4 copied Qwen layers vs random layers, fed by codec-derived inputs:

| Readout | Chain-init NLL | Random NLL | Delta | Teacher-embed NLL |
|---------|---------------:|-----------:|------:|-----------------:|
| Token-end | 15.52 | 17.17 | -1.65 | 11.94 |
| Patch-boundary | 16.62 | 18.34 | -1.72 | 12.52 |

Copied layers ARE more compatible with codec states than random layers. But
absolute NLL is enormous (PPL in millions). True teacher embeddings are 3-4
nats/token better — showing codec quality is still the bottleneck.

### Track B Verdict

```
PASS_CHAIN_INIT_COMPATIBILITY_PROBE_WEAK_SIGNAL
PROMOTE_AS_BASELINE_NOT_MAINLINE
```

## Q-Loop B6 Direction (iterations 36-42)

The Q-Loop's 42-iteration synthesis recommends:

**Evidence-Native Retrieval-Born Sutra:**
- 121M byte-native core trained as evidence-conditioned judge
- Codec as semantic addressability infrastructure
- External corpus as knowledge store
- Teachers as lesson/evidence generators, not coordinate donors
- Success: beats same-retriever dumb baselines by enough to prove learned judgment

**Reclassification:**
- Brainseed → negative-result science
- Codec → addressability infrastructure
- Chain-init → strong baseline/fallback
- Large teachers → data generators

**First gate:**
- +8pp HellaSwag over closed-book baseline
- >=35% HellaSwag as "this may be real" threshold
- >=42.65% to challenge CBD
- Controls: shuffled evidence, wrong-topic evidence, nearest-neighbor labels,
  same-retriever dumb baselines, corpus leakage audit

## Narrative Gate

**Honest headline after 50 work + 42 question iterations:**

"The brain scan failed. The inheritance twitched. But the real insight was:
tiny models don't need to memorize the world — they need to learn how to judge
what they find in it."

**Does it survive "that's obvious"?** RAG is obvious. Evidence-native training
from scratch with a byte-native judge is less obvious. The novelty gate is:
does the learned judgment geometry beat dumb retrieval baselines?

**Does it survive "that's trivial"?** Not yet — no evidence-native prototype
exists. It will be trivial if the retriever does all the work. It will be
non-trivial if the 121M judge produces meaningful lift over same-retriever
baselines.

**Narrative status: ALIVE but untested in new direction.**

## Batch 6/7 Launch Orders

**Work Loop B6 (iterations 51-60): Build Evidence-Native Sutra Prototype**

Priority: build the first evidence-conditioned judgment prototype and test it.
Chain-init as parallel baseline.

**Question Loop B7 (iterations 43-49): Attack Evidence-Native Sutra**

Priority: find every way the evidence-native direction could fail or be trivial.
Predict what the prototype will find. Set the controls.

## Updated Confidence Table

| Claim | Confidence |
|-------|-----------|
| Brainseed v0 extraction is dead | **CONFIRMED** |
| Codec is useful infrastructure | MODERATE-HIGH |
| Chain-init compatibility exists | MODERATE (weak signal) |
| Evidence-native is the right moonshot | MODERATE (untested) |
| Evidence-native will beat dumb baselines | LOW-MODERATE (unknown) |
| Evidence-native will challenge CBD | LOW (aspirational) |
| Project will produce stop-scrolling result | LOW (honest) |
