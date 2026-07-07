# Tier 3 Decision: Brainseed Real-Model Falsifier

Date: 2026-07-07
Role: senior architect review of dual-loop Batch 2 and toy transplant gauntlet

## Executive Decision

Proceed to Tier 3, but only to a minimal real-model falsifier.

Do not launch full Phase 2/2.5 training yet. The gauntlet is sufficient to justify
loading the real teacher and real codec to test chart quality and a frozen
Brainseed scorer. It is not sufficient to claim that real Sutra will inherit
semantic capability, and it is not sufficient to spend GPU on a 10K-20K training
run.

The next experiment must answer one narrow question:

```text
Does the real byte-to-token codec provide a good enough chart for a compact
teacher-free relational scorer to beat codec-only, shuffled, random, rotated,
length/frequency, and retrieval-lite controls on held-out semantic choices?
```

If yes, proceed to a tiny learning-acceleration run. If no, repair the codec
dense-boundary mismatch or kill Brainseed as the current Tier 3 mechanism.

## Decision 1: Is The Gauntlet Enough For Tier 3?

Yes, for Tier 3.0 only.

What the gauntlet proves cleanly:

- Raw per-layer SVD is not a serious transplant mechanism. The Tier 1 non-
  orthogonal gauge changes raw SVD by 0.7687 relative drift and raises MSE from
  0.0409 to 4.1939, while exact function and chart/Procrustes transplants stay
  at numerical zero.
- A chart-aware operator can preserve a synthetic binding relation where raw SVD
  and fake controls fail.
- A byte-patch codec can serve as a cross-architecture chart in a controlled
  synthetic setting, if the chart points to the positions the scorer consumes.

What it does not prove:

- It does not prove the existing Phase 1 codec is semantic. The repo evidence
  still says token-identity retrieval, not semantic addressability.
- It does not prove real Qwen hidden/margin geometry has a compact low-rank
  basis.
- It does not prove the real codec is good at 4-byte patch boundaries. This is
  the central known mismatch: Phase 1 supervised token ends, while Phase 2
  consumes patch ends; only about 20% overlap was measured in DEEP_RETHINK.
- It does not prove a real Sutra checkpoint will improve. Tier 2/2.5 are analytic
  scaffolds, not trained-transformer evidence.

Therefore: proceed to a real-model chart audit and frozen scorer. Do not proceed
directly to a training run.

## Minimal Tier 3 Experiment

Name: Tier 3.0 Brainseed Chart Probe

New file:

```text
code/tier3_brainseed_chart_probe.py
```

Inputs:

```text
Teacher: Qwen/Qwen3-0.6B
Teacher embeddings: C:/sutra_fast/teacher_embeddings.pt if present
Codec: C:/sutra_fast/codec_phase1/codec_final.pt
Codec implementation: code/semantic_codec.py
Sutra config for later insertion only: s0_wide7 from code/s0_configs.py
Extraction data: HellaSwag train slice, PIQA train slice, no eval labels used
Evaluation data: held-out HellaSwag validation slice and PIQA validation slice
```

Do not load or train a full Sutra checkpoint in Tier 3.0 unless the chart and
frozen-scorer gates pass.

Procedure:

1. Load the trained Phase 1 codec and Qwen3-0.6B tokenizer/embedding table.
2. Build two chart-quality caches:
   - token-end anchors, matching Phase 1 supervision;
   - 4-byte patch-boundary anchors, matching Sutra consumption.
3. Measure retrieval quality against teacher token embeddings:
   - top-1, top-5, top-10 in-batch retrieval;
   - token-frequency slices;
   - rare-token slice;
   - patch-boundary vs token-end degradation.
4. Run controls:
   - per-occurrence random target;
   - fixed shuffled target;
   - random codec;
   - rotated codec chart;
   - token-frequency lookup baseline.
5. Build a tiny frozen Brainseed v0 scorer from teacher margins:
   - extraction examples: 512 HellaSwag-train + 512 PIQA-train;
   - candidates: original multiple-choice endings;
   - no gold labels used during extraction;
   - teacher margins come from Qwen3-0.6B candidate log-likelihoods;
   - basis rank: 32 and 64 only;
   - energy is closed-form ridge/logistic regression over codec-chart features,
     not end-to-end Sutra training.
6. Evaluate zero-step held-out scoring:
   - 1024 HellaSwag validation examples;
   - 1024 PIQA validation examples;
   - compare real Brainseed against all fake/control seeds.
7. Write artifact and report:

```text
C:/sutra_fast/brainseed_v0/
  codec_manifest.json
  chart_metrics.json
  basis_B.fp16
  energy_E.fp16
  controls_manifest.json
  extraction_report.md
```

Repo report:

```text
research/tier3_brainseed_probe_report.md
```

Compute estimate:

- Chart-only retrieval audit: CPU possible, GPU preferred; expected <30 minutes
  if embeddings and codec checkpoint are local.
- Teacher margin extraction with Qwen3-0.6B: GPU preferred; expected 30-90
  minutes on the 24 GB 5090 for about 4K candidate forward passes. VRAM should
  stay well under 12 GB with fp16/bf16 batching.
- CPU fallback for teacher forwards is allowed only as a last resort and should
  be treated as an overnight job, not the default.
- No 121M Sutra training in Tier 3.0.

## Precommitted Gates

### Gate A: Real Codec Chart Quality

Confirm chart quality only if all hold:

- Token-end retrieval top-1 >= 50% on held-out anchors.
- Token-end real-vs-fixed-shuffled gap >= 25pp.
- Per-occurrence random target is within 2x chance.
- Patch-boundary retrieval top-1 >= 30% or top-10 >= 65%.
- Patch-boundary real-vs-best-control gap >= 15pp.
- Rare-token patch-boundary top-1 >= 15% and beats best control by >= 8pp.

Kill proceed-to-transplant if:

- Patch-boundary top-1 < 15%, or
- patch-boundary real-vs-best-control gap < 8pp, or
- fixed/per-occurrence controls explain most of the signal.

If token-end passes but patch-boundary fails, do not kill the whole idea. Route to
Phase 1.5 dense patch-boundary supervision and rerun Gate A once.

### Gate B: Frozen Brainseed Zero-Step Scorer

Confirm Brainseed v0 only if all hold:

- HellaSwag validation slice: real Brainseed beats codec-only by >= 5pp.
- HellaSwag validation slice: real Brainseed beats the best fake seed/control by
  >= 5pp.
- PIQA validation slice: real Brainseed beats codec-only and best fake control by
  >= 3pp.
- Aggregate HellaSwag+PIQA paired bootstrap 95% lower bound is > +2pp over the
  best non-retrieval fake control.
- The artifact is <= 25 MB excluding the frozen codec checkpoint.
- Evaluation uses no teacher calls.

Kill Brainseed v0 if:

- Aggregate lift over best fake/control is < 3pp, or
- codec-only is within 1pp of Brainseed, or
- length/frequency/retrieval-lite matches Brainseed within 1pp, or
- the artifact exceeds 50 MB without a clear accuracy win.

### Gate C: Learning Acceleration

Run only after Gates A and B pass.

Confirm acceleration only if a Brainseed-initialized/frozen-hook Sutra variant:

- reaches the matched baseline's 50K-step HellaSwag/PIQA aggregate in <= 5K
  steps, or
- beats matched random/shuffled-seed initialization by >= 3pp aggregate after
  <= 5K steps, with WikiText BPB no worse by >5%.

Kill the training line if:

- the gain is matched by shuffled Brainseed, random chart, extra-data control, or
  retrieval-lite control, or
- no >= 3pp aggregate lift appears by 5K steps.

Do not extend to 10K/20K steps without passing Gate C.

## Decision 4: More Loops Or Implementation?

Proceed directly to implementation.

No more Question Loop before Tier 3.0. The question loop has already converged
on the right falsifier: Brainseed as a compact birth artifact, not GISBE as a
belief system.

No broad Work Loop either. The next work should be a normal implementation pass:

```text
1. implement code/tier3_brainseed_chart_probe.py
2. add CPU unit tests with synthetic fixtures
3. run chart-only mode locally
4. run teacher-margin mode only after the chart-only path works
5. write research/tier3_brainseed_probe_report.md
```

If Tier 3.0 fails, then run a short postmortem loop. Not before.

## Structural Issues

### Gauntlet Code

1. Tier 2/2.5 are analytic scaffolds, not nonlinear transformer evidence. The
   positive methods receive the true synthetic chart relationship by construction.
   This is fine for a harness, but it must not be narrated as real model proof.

2. Tier 2 controls that call `student_slot_memory(..., shuffled_values=True)` or
   `frequency_matched=True` rebuild the corrupted memory independently for each
   candidate score. A real fake seed should be fixed per example or fixed per run.
   This likely makes those controls noisier and possibly weaker than they should
   be. Before using the gauntlet as a regression standard, refactor MCQ scoring so
   each example builds one memory and scores all choices against that same memory.

3. `jacobian_sketch` is not really a Jacobian sketch in the implementation. It is
   an analytic sum of fact-local outer-product slots mapped through the chart. The
   label should be renamed or the real Jacobian version should be implemented.

4. Tier 1 `chart_procrustes_transplant` uses a known canonical reference teacher.
   That is valid for the known-gauge proof, but the real Tier 3 problem lacks that
   canonical reference. Tier 3 must discover the chart quality from held-out pairs
   and controls, not assume it.

5. Tier 2.5's `BytePatchCodec` is a lookup table with small Gaussian noise. The
   real codec is a causal byte transformer with boundary mismatch, tokenization
   ambiguity, rare-token failures, and distribution shift. Treat 2.5 as a minimum
   plausibility test only.

### Dual-Loop Analysis

1. The loops now correctly demote the codec from "semantic proof" to "gauge
   chart." Preserve that discipline. Do not call Brainseed semantic unless the
   held-out margin and transformation gates pass.

2. The loops still risk narrative inflation. A +1pp or +2pp HellaSwag result is
   not a moonshot. It may justify more engineering, but it is not the Vision.

3. The alternative gate is real. If retrieval-lite or byteified chain-init-lite
   beats Brainseed on compactness-adjusted birth behavior, Brainseed should
   concede. The project objective is Born-Knowing Sutra, not proving GISBE.

4. The next report must include exact negatives. If patch-boundary chart quality
   is bad, say "the real codec is not a sufficient chart yet." If frozen Brainseed
   is matched by controls, say "the relational basis did not add semantic signal."

## Exact Next Step

Create:

```text
code/tier3_brainseed_chart_probe.py
```

with CLI:

```powershell
python code/tier3_brainseed_chart_probe.py `
  --codec-checkpoint C:/sutra_fast/codec_phase1/codec_final.pt `
  --teacher Qwen/Qwen3-0.6B `
  --teacher-embeddings C:/sutra_fast/teacher_embeddings.pt `
  --extract-hellaswag 512 `
  --extract-piqa 512 `
  --eval-hellaswag 1024 `
  --eval-piqa 1024 `
  --ranks 32 64 `
  --output-dir C:/sutra_fast/brainseed_v0
```

The script should support `--chart-only` so the first run can avoid teacher
candidate forward passes. If chart-only fails, stop immediately and implement
dense patch-boundary codec supervision before any Brainseed extraction.

Final verdict:

```text
CONFIRM_TIER3_0_CHART_FALSIFIER
VOID_FULL_TIER3_TRAINING_RUN
VOID_REAL_SUTRA_CLAIM_UNTIL_GATES_PASS
```
