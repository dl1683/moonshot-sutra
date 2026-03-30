# T+L Round 3: Ekalavya Protocol — Convergence Push

This is Round 3. Round 2 output is in `results/tl_ekalavya_r2_output.md`.
Round 1 output is in `results/tl_ekalavya_r1_output.md`.

---

## NEW DATA SINCE ROUND 2

### A. CKA Compatibility Probe (REPLACES broken truncated-cosine)

The truncated-dim cosine compatibility metric from R1 was broken (near-zero values). We ran Linear CKA (Gram matrix similarity, dimension-agnostic) on 20 validation windows:

| Teacher | Mean CKA | Span 0 (hardest) | Span 15 (easiest) | Trend |
|---------|----------|-------------------|--------------------| ------|
| LFM-1.2B | **0.865** | 0.841 | **0.919** | INCREASES with context |
| Q0.6B | 0.736 | 0.842 | 0.772 | slight decrease |
| Q1.7B | 0.674 | 0.849 | 0.671 | DECREASES sharply |

**Key findings:**
1. **LFM is the most compatible teacher** — highest CKA at every span except span 0 (where all are ~0.84-0.85)
2. LFM representations CONVERGE toward student as context grows (SSM-hybrid develops increasingly aligned reps)
3. Q1.7B representations DIVERGE from student with context (larger model → specialized reps student can't match)
4. At span 0 (hardest, least context), all teachers have similar CKA — routing by gap/ZPD is valid there
5. **This strongly validates R2's decision to promote LFM to co-anchor**
6. **This strongly validates R2's decision to gate Q1.7B to early spans only**
7. CKA can be used as the compatibility term in the routing formula (replacing broken cosine)

Full data: `results/cka_compatibility_probe.json`

### B. Routing Formula Analysis (quantitative)

With the committee map data + CKA, we tested multiple routing formulas:

**Finding: Any formula based on gap × conf always gives Q1.7B 100% of spans.**
Q1.7B has both highest gap AND highest confidence at every span position. Removing the R1 prior (pi=0.5) doesn't fix routing — it just flips domination from Q0.6B to Q1.7B.

**ZPD (Zone of Proximal Development) routing works:**
- Score = CKA / (|gap - k × student_entropy| + ε)
- At k=1.0: LFM 8/16, Q0.6B 5/16, Q1.7B 3/16 — balanced
- At k=0.8-0.9: LFM 13/16 — LFM's gap/entropy ratio (0.843) is closest to 1.0

**Soft routing with CKA:**
- Score = CKA × conf × gap^0.5, then softmax with temperature
- T=0.1: Q0.6B 30%, LFM 28%, Q1.7B 42%
- T=0.2: Q0.6B 32%, LFM 31%, Q1.7B 38%

**CKA-amplified routing (CKA² × conf × gap^0.5):**
- LFM wins 13/16 spans — CKA dominates

**R2's formula: u_(m,t) = d_m × (0.60 × gap_n + 0.10 × conf_n + 0.30 × c_(m,t))**
- With CKA as c_(m,t), this gives reasonable balance
- The 0.30 weight on compatibility is critical for preventing Q1.7B domination

**Inverse effectiveness weighting confirmed:**
- w(span) = exp(student_entropy / T), T=1.0
- Span 0 gets 3.38× more KD weight than span 15
- This aligns with cross-domain principle: strongest help where student is weakest

### C. Anchor Probe (RUNNING — partial results available)

Running 3K continuation × 4 variants from 60K checkpoint (BPT baseline: 3.5726):
1. `control`: CE-only (no teachers) — **Step 500 eval: BPT=3.6447** (+0.07 from fresh optimizer)
2. `q06_only`: Q0.6B logit KD — pending
3. `lfm_only`: LFM logit KD — pending
4. `q06_lfm_consensus`: Q0.6B + LFM logit KD — pending

**Early observation:** The control variant shows BPT=3.6447 at step 500 — a +0.07 regression from the fresh optimizer/LR schedule. This establishes the baseline that KD variants must beat. By step 1000-1500, the control should recover closer to 3.57 as the optimizer adapts.

**Prediction:** If CKA data is predictive, LFM-only should show the earliest BPT improvement (highest compatibility, lowest gap). Q0.6B-only should be moderate. Q0.6B+LFM consensus should be best but marginal over the better single teacher.

[FULL RESULTS WILL BE AVAILABLE FOR R4 IF NEEDED]

### D. Cross-Domain Research (agents still completing)

[RESULTS WILL BE ADDED WHEN AGENTS COMPLETE]

### E. Summary of ALL Data Available for R3 Design

| Data Source | Key Finding | Design Implication |
|-------------|-------------|-------------------|
| Committee map | Q0.6B 87.5% with R1 formula | Formula is broken, need ZPD/CKA-based |
| Committee map | LFM gap=2.08, lowest | LFM is closest teacher to student |
| Committee map | 14.1% disagreement | Consensus-first is correct |
| Committee map | >92% vocab overlap | Cross-tokenizer is NOT the hard problem |
| CKA probe | LFM CKA=0.87, highest | LFM most compatible, confirms co-anchor |
| CKA probe | LFM CKA increases with context | LFM best for late spans |
| CKA probe | Q1.7B CKA decreases to 0.67 | Q1.7B risky for late spans, gate it |
| Routing analysis | Gap×conf → Q1.7B monopoly | Need CKA in formula for balance |
| Routing analysis | ZPD at k=1.0 → balanced | Gap calibrated to student entropy works |
| 60K benchmarks | HS=29.0, PIQA=59.4 | These are KD targets (commonsense gap) |
| 60K benchmarks | ARC-E=45.4, WG=51.3 | Already competitive, less KD needed |

---

## QUESTIONS FOR ROUND 3

1. **Does the CKA data change any R2 design decisions?** Specifically: should CKA replace the local-cosine c_(m,t) in R2's routing formula? Or use both?

2. **Anchor probe results**: [pending] — which teacher gives the best BPT improvement as a solo anchor? Does consensus (both) beat the best single teacher?

3. **Is ZPD-based routing better than R2's formula?** R2 uses u = d × (gap_n + conf_n + compat). ZPD uses score ∝ 1/|gap - target|. Which principle should win?

4. **How to handle Q1.7B CKA divergence?** CKA drops from 0.85 → 0.67 across spans. Should Q1.7B be restricted to spans where CKA > 0.75? Or is the high gap value on later spans still useful despite low compatibility?

5. **Should EmbeddingGemma semantic KD use CKA?** We only tested CKA on generative teachers. EmbeddingGemma is an encoder — CKA between its representations and student's may be informative for calibrating L_sem strength.

6. **Confidence scores**: What specific new evidence would push each outcome to 9/10?
   - O1 (Intelligence): Need positive anchor probe results showing BPT drops
   - O2 (Improvability): Need proof that per-teacher diagnostics enable surgical fixes
   - O3 (Democratized): Need evidence that adding/removing a teacher is clean
   - O4 (Data Efficiency): Need BPT/benchmark gains from KD vs continued CE
   - O5 (Inference): Need evidence that KD doesn't slow inference

---

## YOUR TASK FOR ROUND 3

Given the new CKA data and (when available) anchor probe results:

**Phase A**: How does CKA data affect the R2 design? Any corrections?

**Phase B**: What final probes/research would push remaining outcomes to 9/10?

**Phase C**: Produce the FINAL refined design with:
- Updated routing formula incorporating CKA
- Any changes from anchor probe results
- Implementation specification detailed enough to code directly
- EVERY parameter, dimension, formula, schedule specified

**Target: push ALL 5 outcomes toward 9/10.** Each +1 requires SPECIFIC new evidence.
