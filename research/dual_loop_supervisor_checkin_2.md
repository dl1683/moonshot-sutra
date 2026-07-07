# Dual-Loop Supervisor Check-in #2: After Batch 3

Date: 2026-07-07

## Question Loop Batch 3 (Iterations 15-21): COMPLETE

### Assessment: Exceptionally thorough adversarial attack

The Q-Loop attacked Brainseed from 7 concrete angles using real numbers (70.3%
retrieval, 20% boundary overlap, CBD 42.65%). Each iteration killed something.
The surviving Brainseed is much sharper and more honest than Batch 2's version.

### Key Threats (ranked by kill probability)

1. **Boundary mismatch (HIGHEST THREAT).** Only 20% of patch boundaries overlap
   with token-end positions where the codec was supervised. The codec may be
   best exactly where Sutra doesn't read it. If patch-boundary retrieval is near
   control levels, Brainseed dies at the current codec.

2. **Codec degradation cliff.** 61% average retrieval compounds: 4 critical
   anchors at 0.61^4 = 14% joint accuracy. If the transplant collapses below
   80-90% chart quality in the toy, the real codec isn't good enough.

3. **Just-distill baseline.** Teacher-margin regression over codec features IS
   supervised learning. If a boring MLP/bilinear scorer matches Brainseed
   under matched budget, the geometry story is unnecessary.

4. **Chain-init owns the born-knowing story.** CBD 42.65% HellaSwag at 138M
   already proves born-knowing. Brainseed is solving a harder problem (cross-
   architecture extraction) for potentially a worse result.

5. **Measurement contamination.** HellaSwag-train extraction → HellaSwag-val
   scoring can look like birth but actually be benchmark-format distillation.
   Must test cross-domain transfer.

6. **Scale ceiling.** 121M dense byte model may cap all birth methods at ~26-32%
   regardless of initialization. Chain-init may be the only thing that jumps
   to 40s.

7. **Narrative fragility.** A modest, expensive, hand-tuned lift is not the
   moonshot. Must be automatic, cheap, compact, and visible.

### What the Q-Loop Got Right

- Demanding a toy degradation curve BEFORE interpreting real results
- Splitting token-end vs patch-boundary as separate gates
- Requiring just-distill baselines, not just shuffled/random controls
- Cross-domain generalization requirement
- Honest confidence levels (very low for standalone 121M beating Arjun)

### Supervisor Corrections for Q-Loop

None — this batch was correctly calibrated. It attacks hard but never drifts
toward "play it safe" or "give up on the moonshot." It says "the moonshot
requires THIS, not THAT" which is exactly the right framing.

### Key Requirement Injected Into Work Loop

**BEFORE interpreting any Tier 3.0 result, the Work Loop must run a toy
degradation audit:**
- Corrupt Tier 2.5 chart accuracy to 90%, 80%, 70%, 60%, 50%, 40%, 25%
- Measure whether transplant degrades smoothly or falls off a cliff
- This tells us whether 61% real chart is above or below the cliff

**The Work Loop must also add "just-distill" baselines:**
- Same codec features, same teacher margins, but MLP/bilinear instead of
  extracted relational basis
- If they match Brainseed, the geometry is unnecessary

### Confidence Assessment (after 21 iterations)

| Claim | Confidence |
|-------|-----------|
| Toy gauntlet validates the principle | HIGH |
| Token-end chart quality passes some audit | MODERATE |
| Patch-boundary chart quality without Phase 1.5 | LOW-MODERATE |
| Frozen Brainseed beats controls AND distillation | LOW |
| Standalone 121M beats Arjun without chain-init | VERY LOW |

### Narrative Gate

**Honest headline from what we've proved so far:**
"Scientists built a brain scanner that works perfectly on mannequins. They're
about to try it on a real patient — and the glasses may be cracked."

**Does it survive "that's obvious"?** Not yet — no real evidence beyond toy.
**Does it survive "that's trivial"?** The toy controls are comprehensive, so
the principle is real. But toy→real is the whole remaining question.

**Narrative is alive but on life support.** It dies with the next batch if
the real chart fails or if boring distillation matches.

## Work Loop Batch 3 (Iterations 21-30): COMPLETE

### Assessment: Clean execution, honest failure, correct routing

The Work Loop built `code/tier3_brainseed_chart_probe.py` to spec, ran Gate A on
real data (16-seq + 64-seq confirmation), and correctly applied pre-committed gates.

### Gate A Results (64-sequence, N=4096 patch anchors)

| Anchor | N | Real top-1 | Real top-10 | Best control | Gap | Gate |
|--------|----:|----------:|------------:|-------------:|----:|------|
| Token end | 3992 | 86.57% | 97.82% | 4.31% | +82.26pp | PASS |
| Patch boundary | 4096 | 23.71% | 35.30% | 1.25% | +22.46pp | FAIL |
| Rare patch | 1893 | 18.28% | 40.52% | 0.11% | +18.17pp | PASS (gap) |

All 5 controls near floor (0-1.25%). The codec is real — massively above controls.
But it's real at the WRONG positions. It was trained on token-end boundaries and
doesn't generalize to 4-byte patch boundaries where Sutra actually reads.

### What the Work Loop Got Right

- Followed R65 spec exactly. No goalpost moving.
- Ran two sample sizes for stability check (16→64 sequences).
- Applied pre-committed gate thresholds honestly.
- Correctly voided Gate B scorer (iterations 26-30 not run).
- Routed to Phase 1.5 without hesitation.
- Narrative gate honest: "the bridge doesn't reach the door."

### What the Work Loop Missed

1. **No toy degradation curve.** The Q-Loop demanded: corrupt Tier 2.5 chart
   accuracy to 90/80/70/60/50/40/25% and see if transplant degrades smoothly or
   falls off a cliff. This tells us whether 23.71% real patch-boundary chart is
   above or below the cliff. **This MUST happen in Batch 4.**

2. **No just-distill baselines.** Q-Loop demanded: same codec features, same
   teacher margins, MLP/bilinear scorer instead of extracted relational basis.
   If they match Brainseed, geometry is unnecessary. **This is deferred until
   Gate A passes, but should be designed now.**

3. **No Phase 1.5 design.** The Work Loop identified the need but didn't spec
   the dense supervision approach. **Batch 4 must implement Phase 1.5.**

### Supervisor Corrections for Work Loop

- Must run the toy degradation curve BEFORE Phase 1.5 implementation. If the
  cliff is at 60% and Phase 1.5 only gets us to 40%, it's dead on arrival.
  The degradation curve tells us what Phase 1.5 needs to achieve.
- Phase 1.5 must supervise at EVERY 4-byte patch-boundary position, not just
  where tokens happen to end. The anchor is: for byte position 4k, supervise
  against the teacher token embedding that CONTAINS byte 4k.

### Q-Loop Prediction Accuracy

The Q-Loop's #1 threat (boundary mismatch, iteration 15) was validated EXACTLY.
Token-end excellent, patch-boundary insufficient. This is the strongest signal
yet that the Q-Loop is correctly calibrated. Confidence in Q-Loop direction: HIGH.

### Confidence Assessment (after 30 work + 21 question iterations)

| Claim | Confidence | Change |
|-------|-----------|--------|
| Toy gauntlet validates the principle | HIGH | unchanged |
| Token-end chart quality is real | **VERY HIGH** | ↑ confirmed |
| Patch-boundary chart quality | **FAIL** | was LOW-MODERATE |
| Phase 1.5 can fix the gap | MODERATE | new claim |
| Frozen Brainseed beats controls + distill | LOW | unchanged |
| Standalone 121M beats Arjun | VERY LOW | unchanged |

### Narrative Gate

**Honest headline from what we've proved so far:**
"Scientists proved they can read a bigger brain's thoughts through a translator —
but the translator only speaks at pauses, and the small brain needs every word."

**Does it survive "that's obvious"?** Not yet — reading at token-end positions is
exactly where the codec was trained. The non-obvious result would be reading at
arbitrary positions.

**Does it survive "that's trivial"?** The 82pp gap over controls is NOT trivial.
The codec genuinely learned something. But it learned the WRONG something for
Sutra's consumption pattern.

**Narrative status: alive but critical.** Phase 1.5 is the surgery that determines
whether the patient walks or stays in the ICU.

### Batch 4 Priorities (both loops)

**Work Loop B4 (iterations 31-40):**
1. Toy degradation curve (BEFORE Phase 1.5 — tells us the target)
2. Phase 1.5 dense patch-boundary codec supervision design + implementation
3. Phase 1.5 training (on same data, same objective, new anchor positions)
4. Re-run Gate A with Phase 1.5 codec
5. If Gate A passes: proceed to frozen scorer (Gate B)

**Question Loop B4 (iterations 22-28):**
1. Attack Phase 1.5 — is dense supervision even the right fix?
2. Attack the capacity gap — if the codec needs to work at 4x more positions,
   does 256-dim / 4-layer have enough capacity?
3. Attack the training signal — teacher token at byte position 4k may be
   ambiguous (mid-token supervision is a different problem than token-end)
4. Attack the narrative — even if Phase 1.5 works, "we retrained the codec
   with more supervision" is not a moonshot story
5. Attack alternatives — what about interpolation, attention pooling, or
   learned boundary prediction instead of fixed 4-byte patches?
