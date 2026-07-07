# Project Status

**Last updated:** 2026-07-07
**Current loop state:** W-Loop B6 completed, Q-Loop B8 completed, supervisor check-in 5 written
**Live moonshot candidate:** Evidence-Native Retrieval-Born Sutra, on life support after v0 failure

This file is the current source of truth. Older research files remain as
provenance, but this page governs the live interpretation of the project.

## Executive Summary

Brainseed v0 is confirmed dead as a birth artifact after 50 work-loop iterations
and 42 question-loop iterations. All tested downstream scorers lose to codec-only
scoring. The codec remains useful infrastructure, but it is not the moonshot.

Evidence-Native Retrieval-Born Sutra became the replacement moonshot candidate,
but the first v0 prototype failed all precommitted gates. Retrieved evidence did
not beat no-evidence, shuffled evidence, or the best same-retriever dumb
baseline. The broader direction is not dead because v0 was a small frozen-codec
pooled classifier, not the full 121M judge claim. The burden of proof has shifted
hard: the next run must show that evidence training changes judgment geometry,
not merely that retrieval adds text to the prompt.

The refined thesis remains:

**Intelligence = reasoning geometry, which should be cheap and transferable, plus
factual knowledge, which should be retrieved from evidence.**

## Alive, Dead, In Progress

| Track | Status | Current role | Notes |
|-------|--------|--------------|-------|
| Evidence-Native Retrieval-Born Sutra | Alive but on life support | Moonshot candidate under strict survival gates | v0 failed all gates; next version must prove learned judgment over controls. |
| Chain-init | Alive as baseline | Strong fallback and benchmark to beat | Weak compatibility signal observed; not treated as the moonshot. |
| Codec | Alive as infrastructure | Byte-to-token addressability, diagnostics, possible bridge | Useful, but not the final claim. |
| Brainseed v0 | Dead | Negative-result science and diagnostic history | All learned scorers tested worse than codec-only. |
| Byte-marginal KD / old E1-Option-C direction | Dead as mainline | Historical baseline | Improved byte prediction without meaningful downstream judgment transfer. |
| S0 training stack | Infrastructure | Baseline and reusable code | Do not delete; still useful for baselines and evidence-native substrate work. |
| E1/E2 training stack | Infrastructure/historical | Useful controls and lessons | Not the current research claim. |

## Current Dual-Loop State

| Loop | Batch | Status | Artifact |
|------|-------|--------|----------|
| Work Loop | B6 | Completed | `research/work_loop_batch6.md` - Evidence-Native v0 prototype run; failed gates. |
| Question Loop | B8 | Completed | `research/question_loop_batch8.md` - adversarial survival conditions and kill criteria. |
| Supervisor | Check-in 5 | Completed | `research/dual_loop_supervisor_checkin_5.md` - Evidence-Native v0 post-mortem. |

Do not edit loop batch or supervisor files during concurrent runs. In this pass,
no `research/work_loop_batch*.md`, `research/question_loop_batch*.md`, or
`research/dual_loop_supervisor_checkin_*.md` files were modified.

## Key Findings

1. Brainseed v0 extraction is dead.
   - Ridge, MLP, bilinear, and learned-cosine scorers all lose to codec-only.
   - Zero-cost chart rescues do not recover the mainline.
   - Status: `BRAINSEED_DEAD_AS_BIRTH_ARTIFACT`.

2. Chain-init has weak positive signal.
   - Copied inherited-coordinate layers are more compatible with codec-derived
     inputs than random layers in the probe.
   - The signal is not benchmark-capable yet.
   - Status: baseline/fallback, not the moonshot.

3. Evidence-native v0 failed.
   - The learned judge did not benefit from retrieved evidence under the
     precommitted gates.
   - Shuffled and no-evidence conditions were not cleanly beaten.
   - A same-retriever dumb baseline beat the learned judge.
   - Status: v0 dead; broader direction on life support.

4. The codec is infrastructure.
   - It helps expose byte-to-token addressability and failure modes.
   - It should not be described as the breakthrough by itself.

## Artifact Index

### Pivot and Supervisory Documents

- `research/dual_loop_supervisor_checkin_5.md` - Evidence-Native v0 post-mortem; direction on life support.
- `research/dual_loop_supervisor_checkin_4.md` - formal pivot: Brainseed dead, evidence-native mainline candidate, chain-init baseline.
- `research/dual_loop_supervisor_checkin_3.md` - Brainseed near-death and one-final-batch decision.
- `research/dual_loop_supervisor_checkin_2.md` - earlier Brainseed pressure test.
- `research/dual_loop_supervisor_checkin_1.md` - original Brainseed direction.

### Loop Batches

- Work loop: `research/work_loop_batch1.md` through `research/work_loop_batch6.md`.
- Question loop: root `question_loop_batch1.md`, then `research/question_loop_batch2.md` through `research/question_loop_batch8.md`.

### Current Status and History

- `README.md` - public-facing current state.
- `research/STATUS.md` - this source-of-truth status file.
- `research/DEEP_RETHINK.md` - full historical research log; current pivot addendum updated on 2026-07-07.
- `research/INDEPENDENT_ANALYSIS.md` - still relevant as a pre-pivot first-principles analysis of the byte-KD failure, but superseded for mainline direction by supervisor check-ins 4 and 5.
- `experiments/EXPERIMENTS.md` - human experiment ledger index.
- `experiments/ledger.jsonl` - append-only machine-readable experiment ledger.

## Current Milestones

1. Evidence-native survival gate.
   - Replace v0's weak pooled classifier test with a stronger judge architecture.
   - Use external-only or explicitly partitioned evidence conditions.
   - Compare evidence-trained and no-evidence-trained controls directly.
   - Run geometry probes, not just benchmark scores.

2. Chain-init baseline gate.
   - Turn weak compatibility into a realistic baseline, or demote it.
   - Evidence-native must beat this path eventually, not just closed-book baselines.

3. Kill/defer gate.
   - If stronger evidence training still fails to beat controls, demote
     evidence-native to application-layer retrieval/reranking.
   - Mainline then returns to chain-init, inherited coordinates, or a larger
     Sutra-family anchor.

4. Repo hygiene gate.
   - Keep active loop artifacts isolated.
   - Keep temp outputs ignored.
   - Keep experiment results summarized in `experiments/` rather than scattered across scratch directories.

## Wording Rules For Fresh Readers

Use:

- "Brainseed v0 is dead as a mainline birth artifact."
- "The codec is infrastructure, not the moonshot."
- "Chain-init is the strong baseline/fallback, not the moonshot."
- "Evidence-native v0 failed; the broader direction is on life support."

Avoid:

- "Brainseed is promising" without historical context.
- "The codec proves semantic intelligence."
- "Chain-init is the current moonshot."
- "Evidence-native works" before controlled prototype results exist.
- "Evidence-native is dead" without specifying v0 versus the stronger 121M claim.