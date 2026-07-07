# Dual-Loop Supervisor Check-in #3: After Batch 4

Date: 2026-07-07

## Executive Summary

**Brainseed v0 is dead.** Phase 1.5 repaired the chart (Gate A passes formally)
but the extracted Brainseed scorer performs WORSE than raw codec features on both
HellaSwag and PIQA. The relational basis extraction adds noise, not signal.

The Q-Loop predicted this scenario exactly (iteration 28): "Phase 1.5 succeeds
technically while proving Brainseed is only a soft-tokenizer repair."

## Question Loop Batch 4 (Iterations 22-28): COMPLETE

### Assessment: Prophetic attack battery

Every major Q-Loop prediction was validated:

| Q-Loop Prediction | W-Loop Result | Verdict |
|------------------|---------------|---------|
| Mid-token causal mismatch (#22) | Non-overlap positions still weak | CONFIRMED |
| Degradation cliff above 30% (#23) | Cliff at ~50% chart top-1 | CONFIRMED |
| Interference degrades token-end (#24) | Token-end dropped 8.19pp | CONFIRMED |
| Phase 1.5 = plumbing not moonshot (#27) | Gate B failed, no birth effect | CONFIRMED |
| Chain-distill may be superior (#28) | Brainseed < codec-only | CONFIRMED |

The Q-Loop's most prescient contribution: iteration 23's calculation that the
non-overlap mid-token accuracy was ~8%, and that Phase 1.5 needed to raise it
from 8% to 40-55% to cross the cliff. The W-Loop got aggregate patch top-1 to
37.89% — likely meaning non-overlap positions are around 20-25%. Below the cliff.

### Key Unconsumed Q-Loop Outputs

Several Q-Loop attacks generated actionable diagnostics the W-Loop did NOT run:

1. **Zero-cost baselines** (iterations 25-26): interpolation, nearest-token-end
   state, local pooling. These would tell us whether the chart quality problem
   is missing supervision or representational smoothness. NOT YET TESTED.

2. **Offset-sliced diagnostics** (iteration 22): Early/mid/late-token position
   breakdown. Would reveal whether Phase 1.5 improvement is concentrated at
   near-token-end positions (easy) or genuinely raises mid-token quality (hard).
   NOT YET TESTED.

3. **Two-headed codec** (iteration 24): Separate alignment heads for token-end
   and patch-boundary might avoid the interference tradeoff. NOT YET TESTED.

### Q-Loop Fork Prediction (iteration 28)

```
If scorer loses to just-distill → use just-distill, demote Brainseed geometry.
If scorer wins modestly but below chain-init → keep as component, start chain-init mainline.
If scorer wins visibly + insertion accelerates → compare Brainseed + chain-init vs chain-init alone.
```

We are in the WORST branch: scorer loses to CODEC-ONLY, not even to just-distill.
Brainseed geometry should be demoted from mainline.

## Work Loop Batch 4 (Iterations 31-40): COMPLETE

### Assessment: Clean execution, honest failure

The Work Loop followed the mandate exactly:

1. **Toy degradation curve (iteration 31)**: Measured. Cliff at ~50% chart top-1.
   Below that, transplant accuracy approaches chance. Pre-committed verdict honored.

2. **Phase 1.5 implementation (iterations 32-34)**: Clean, minimal changes to
   `codec_phase1_train.py`. 75/25 patch/token anchor mix. Warm-start from Phase 1.

3. **Phase 1.5 training (iteration 35)**: 5000 steps, 1532.8s (~25.5 min GPU).
   Single run, no hyperparameter search. Matches the "cheap and automatic" constraint.

4. **Gate A re-run (iteration 36)**: PASS formally. Patch-boundary improved
   substantially. Token-end degraded as predicted. All controls near floor.

5. **Gate B scorer (iterations 38-40)**: FAIL. Brainseed ridge over codec features +
   teacher margins performs WORSE than codec-only cosine scoring.

### The Damning Numbers

| Metric | Brainseed v0 (rank 32) | Codec-only | Delta |
|--------|----------------------:|----------:|---------:|
| HellaSwag | 26.27% | 27.25% | **-0.98pp** |
| PIQA | 47.36% | 48.83% | **-1.46pp** |

The extracted relational basis and energy functional are not capturing useful
information beyond what the codec already provides. In fact, the ridge regression
on teacher margins is fitting noise that hurts generalization.

### What the Work Loop Got Right

- Ran toy degradation curve FIRST (as mandated)
- Honest about interference (-8pp token-end)
- Honest about Gate A being formal, not operational (37.89% < 50% cliff)
- Ran Gate B despite knowing the chart was below the cliff
- Correctly voided Gate C
- Did NOT overclaim

### What the Work Loop Missed

1. **Zero-cost baselines**: No interpolation, pooling, or adaptive readout tests.
   The Q-Loop demanded these and they would have provided important diagnostics.

2. **Just-distill comparison**: The Gate B failure was against codec-only, not
   against the just-distill suite. We don't know if MLP/bilinear on the same
   features would have worked. But codec-only being better than Brainseed ridge
   suggests the extraction is adding noise, not signal.

3. **Offset-sliced analysis**: No breakdown by token offset position. We don't
   know whether Phase 1.5 improvement is real mid-token quality or just more
   near-token-end supervision.

## Supervisor's Honest Assessment

### The Brainseed Story Arc

```
Batch 1-2: Toy gauntlet validates principle. Chart-aware: 100%. Raw SVD: 25%.
           → "The principle works in controlled settings."

Batch 3:   Real codec Gate A fails. Token-end excellent, patch-boundary weak.
           → "The principle might work if we fix the chart."

Batch 4:   Phase 1.5 fixes Gate A formally. Gate B fails. Scorer < codec-only.
           → "The chart is repaired but the extraction adds noise, not signal."
```

The trajectory is: strong toy → weak real → dead extraction. Each batch has been
honest and well-executed. The conclusion is also honest: **the gauge-invariant
relational geometry extraction, as currently implemented, does not produce useful
born-knowing artifacts from a real teacher.**

### Why Gate B Failed (Supervisor's Analysis)

Three possible explanations:

1. **Chart quality too low**: 37.89% patch top-1 is below the 50% cliff. The
   extracted basis is operating on a noisy chart and the ridge can't separate
   signal from noise. Fix: get chart above 50%.

2. **Extraction method wrong**: Ridge regression on [context, candidate, product,
   diff] features is too simple to capture the relational geometry. The toy used
   exact function transplant and chart-procrustes; the real scorer used a linear
   model. Fix: try nonlinear extraction (MLP, bilinear).

3. **The geometry doesn't transfer**: The toy's success depends on exact gauge
   charts in low-dimensional linear/nonlinear settings. Real LLM embedding spaces
   may not have transferable relational structure at the codec's resolution.
   Fix: there is no fix — pivot.

Explanation #1 is testable (improve chart, retry). Explanation #2 is testable
(try better extractors). Explanation #3 is the fatal one.

The honest supervisor position: we should run ONE more batch to test #1 and #2
before declaring #3. But if the next batch also shows zero lift, Brainseed is
confirmed dead and we pivot to chain-init.

### Confidence Assessment (after 40 work + 28 question iterations)

| Claim | Confidence | Change |
|-------|-----------|--------|
| Toy gauntlet validates principle | HIGH | unchanged |
| Token-end chart is real | HIGH | slightly down (78% after Phase 1.5) |
| Phase 1.5 repair is feasible | MODERATE | confirmed |
| Phase 1.5 chart is operationally sufficient | LOW | below toy cliff |
| Brainseed extraction adds useful signal | **VERY LOW** | Gate B failed |
| Born-knowing via Brainseed | **NEAR ZERO** | no positive result |
| Chain-init is the stronger path | MODERATE-HIGH | by elimination |

### Narrative Gate

**Honest headline from what we've proved:**
"Scientists built a brain scanner and a translator. Both work. But when they
printed the brain scan and gave it to the newborn, it was worse than just
using the translator alone."

**Does it survive "that's obvious"?** No. A negative extraction result is not
interesting unless it reveals WHY.

**Does it survive "that's trivial"?** The experimental rigor is not trivial.
The conclusion is: "this particular extraction doesn't work." That's a negative
result, not a moonshot.

**Narrative status: DEAD for Brainseed v0 as mainline.**

The narrative is alive ONLY if:
1. We diagnose WHY Gate B failed and it reveals something deeper about transfer
2. OR we pivot to something that actually produces a stop-scrolling result

### Decision: Batch 5 Direction

Given Gate B failure, the dual-loop faces a fork:

**Option A: One more Brainseed diagnostic batch.**
- Run zero-cost baselines (interpolation, pooling, adaptive readout)
- Run offset-sliced analysis to understand Phase 1.5 quality
- Try MLP/bilinear scorer instead of ridge
- If chart above 50% needed: try more steps, larger codec, or curriculum
- Timeline: 1 more batch (10 W-Loop + 7 Q-Loop iterations)
- Risk: more sunk cost into a dead path

**Option B: Pivot to chain-init / byteify exploration.**
- CBD already gets 42.65% HellaSwag at 138M
- Chain-init preserves coordinate continuity directly
- The codec can still be useful as a byte-to-token interface, not as
  a Brainseed chart
- Timeline: fresh direction, restart the W-Loop
- Risk: abandoning a path that might have worked with better extraction

**Option C: Hybrid — Brainseed diagnostics + chain-init prototype in parallel.**
- Q-Loop attacks both paths
- W-Loop split: 5 iterations on Brainseed diagnostics, 5 on chain-init prototype
- Decision at check-in #4 based on which shows signal
- Timeline: 1 batch, both paths explored
- Risk: divided attention

**Supervisor decision: Option C.** Brainseed gets exactly ONE more diagnostic
pass (zero-cost baselines + MLP scorer + offset slices). If still no signal,
it's dead and we go full chain-init. Meanwhile, the chain-init prototype starts
in parallel so we don't lose a batch if Brainseed fails again.

The Q-Loop should attack BOTH paths — Brainseed diagnostics AND chain-init risks.
