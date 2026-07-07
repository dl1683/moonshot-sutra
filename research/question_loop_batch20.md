# Q-Loop B20: CTI Direction — SANDBOX PARTIAL

**Date: 2026-07-07**
**Status: ANALYSIS COMPLETE, FILE WRITE FAILED (Windows sandbox apply_patch bug)**
**Codex session: 019f3cd8-2b64-7d63-b536-7cdcdc06bc99**

Codex completed full B20 analysis (web research on scaling laws, read B14 terminal results,
read B19 pivot playbook, composed all 7 iterations) but apply_patch failed on Windows.
The analysis content is preserved in Codex's session transcript.

## Binding State From B20 (extracted from Codex output)

Key conclusions Codex reached before the write failure:

1. **CTI is NOT just scaling laws** — the wedge is: functional distortion (not just loss) +
   intervention taxonomy + held-out prediction + one-GPU decision value

2. **Distortion definition is the hardest part** — if D is just validation loss, CTI = scaling
   laws. Must precommit a basket: held-out accuracy, NLL/BPB, calibration error, train/held gap,
   MCQ margin error

3. **CTI novelty claim must be**: "functional distortion obeys a precommitted compute law that
   distinguishes proxy improvement from real generalization, and geometry-changing interventions
   alter the law's parameters predictably"

4. **First artifact**: research/CTI_PRECOMMIT_SPEC.md (distortion def, compute accounting,
   model/task grid, baseline forecasters, prediction protocol, kill tokens)

5. **Kill tokens**: INVALID_CTI, NO_PREDICTIVE_LAW, PROXY_ONLY_LAW, PASS_CTI_LAW_0, STRONG_CTI

6. **Pass bar**: CTI predicts held-out distortion at unseen compute budgets better than all
   baseline forecasters on >=2/3 task families, and correctly classifies at least one
   intervention as "constant shift" vs "exponent shift" before full results observed

7. **Normal-person headline**: "A laptop predicted which AI training ideas were worth the
   electricity before they finished training"

8. **Renormalization = theory lane**, not main pivot. CTI absorbs more Eklavya arc data.

9. **The repo's weakness**: "Generating frameworks faster than results." The pivot must be
   visually boring: terminal verdict first, negative ledger second, salvage map third, new
   precommit fourth, only then experiments.

## Forward References

B19 I131-133 contain the full CTI/renormalization deep-dives and adversarial review.
The CTI_PRECOMMIT_SPEC.md is the first mandatory artifact before any CTI experiments.
