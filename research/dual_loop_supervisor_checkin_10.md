# Dual-Loop Supervisor Check-In #10

**Date:** 2026-07-07
**Loops completed:** W-Loop B11 (iterations 101-105, halted by kill), Q-Loop B15 (iterations 99-105)
**Status:** FMD PROTOTYPE KILL — natural-continuation margin training failed to produce benchmark margins. Direction alive, formulation dead. Portfolio pivot.

---

## What Happened

### Q-Loop B15: Adversarial Attack On The Pivot

7 iterations attacking Functional Margin Distillation and Disagreement Router before implementation. Key outputs:

1. **FMD is prior art (Iteration 99/103):** Pairwise ranking losses are well-studied (RKD, CRD, Margin-MSE, PLaD, DPO, RRHF, SLiC-HF). The possible novelty is the CONJUNCTION: byte-native + multi-teacher + cross-tokenizer functional margins + disagreement routing + single-GPU. Each component alone is prior art.

2. **Concrete loss design (Iteration 100):** Specified two data modes — Mode A: labeled benchmark MCQ (easiest, most diagnostic); Mode B: unlabeled text converted to contrastive choices (closer to manifesto). Teacher-wrong filtering is mandatory (Qwen 38% on HellaSwag, 36% on ARC-Easy in B10 sample).

3. **Multi-teacher tokenizer problem (Iteration 101):** Sequence-level byte-normalized NLLs sidestep token alignment but discard token-level dark knowledge. Multi-Level Optimal Transport (arXiv:2412.14528) is direct prior art for cross-tokenizer KD.

4. **FMD_SHADOW_288 design (Iteration 102):** Proposed 288 benchmark-style train examples + 144 held-out + 5 strong baselines (MSE adapter, label-only, token KD, random margins, length/position controls). Precommitted PASS/MARGINAL/FAIL tokens.

5. **Disagreement density audit (Iteration 104):** Must measure useful disagreement, oracle routing ceiling, best-teacher gap, and noise sensitivity BEFORE building a router. Pass: >=15% useful disagreement + oracle ceiling beats best teacher by >=5pp.

6. **Process changes (Iteration 105):** Split ADMISSION_EVIDENCE vs MOONSHOT_EVIDENCE. Run a PORTFOLIO (FMD + teacher audit + disagreement audit + counterfactual sketch), not serial obsession. One repair cycle max without functional evidence. Add a novelty gate.

### W-Loop B11: FMD Prototype Smoke

Created `code/margin_distillation.py` — complete prototype with:
- `MarginStudent`: frozen codec + trainable projection + tiny causal reasoner + byte decoder
- Unlabeled natural-continuation candidate sets from shard data
- Qwen teacher black-box margin targets
- Pairwise RankNet + SmoothL1 margin regression loss
- Functional-margin benchmark evaluation (reusing B10 infrastructure)

**Smoke result: FAIL_MARGIN_PROTOTYPE**

| Benchmark | Qwen teacher | Baseline untrained | Margin-trained | Delta |
|---|---:|---:|---:|---:|
| HellaSwag | 38.0% | 34.0% | 22.0% | **-12.0pp** |
| PIQA | 76.0% | 52.0% | 52.0% | **0.0pp** |
| ARC-Easy | 36.0% | 30.0% | 30.0% | **0.0pp** |

Training loss decreased substantially (1.19→0.56 over 10 steps, 50 examples). The model learned the shard-continuation ranking task. But that learning did NOT transfer to benchmark MCQ margins — it actively HURT HellaSwag by 12pp.

**Hard kill fired.** Iterations 106-110 blocked.

---

## Supervisor Assessment

### The Formulation Is Dead. The Direction Has One More Chance.

This is an important distinction. W-Loop B11 tested a SPECIFIC formulation:

```
Unlabeled natural-continuation margins → benchmark MCQ discrimination
```

That path failed for an identifiable reason: **domain mismatch between training and evaluation data.** The model learned to rank arbitrary text continuations by corpus-naturalness, which is a different skill than discriminating benchmark answer choices. The HellaSwag regression suggests the continuation-naturalness prior actively fought the answer-discrimination task.

Q-Loop B15 predicted this (Iteration 102):

> "The first W-Loop should not try to prove the moonshot. It should test one narrow question."

The narrow answer: **No, unlabeled continuation margins don't create benchmark margins in 10 steps on a tiny student.**

### What This Does NOT Kill

The broader FMD thesis — "train on teacher decision margins, not hidden states" — is not dead from this evidence. B11 used:
- Unlabeled shard-derived continuations (not benchmark-style MCQ data)
- A tiny randomly initialized student (not pretrained S0/Wide7)
- Only Qwen as teacher (38% on HellaSwag — barely above chance)
- 50 examples, 10 gradient steps

A different formulation using benchmark-style candidate sets (B15's Mode A / FMD_SHADOW_288) has not been tested. That is the ONE remaining repair allowed under B15's "one repair cycle" rule.

### What This DOES Kill

1. The claim that margin-facing losses automatically produce benchmark-facing margins
2. The assumption that shard-derived contrastive pairs are equivalent to benchmark MCQ pairs
3. Any hope of a quick, easy pivot from coordinate inheritance to FMD

### The Deeper Pattern

This is the project's **6th consecutive direction death**:

| # | Direction | Kill mechanism |
|---|---|---|
| 1 | Gradient KD | ≤0.7pp HellaSwag |
| 2 | Brainseed v0 | All scorers worse than codec-only |
| 3 | Evidence-Native v0 | Evidence training hurt evidence use |
| 4 | Evidence-Native v1 | Internalization gate +0.47pp |
| 5 | Coordinate-Inheritance v0/v1/v2 | Main inherited WORSE than random on benchmarks |
| 6 | FMD prototype (unlabeled) | -12pp HellaSwag, 0pp PIQA/ARC |

Q-Loop B15 named the risk: "baseline laundering — after a favorite mechanism dies, the project pivots to an ordinary baseline and gives it moonshot language." The supervisor must confront whether the repeated failures indicate:

(a) We haven't found the right formulation yet, or
(b) The byte-codec scaffold itself cannot support KD at this scale.

**Assessment:** Evidence points to (a) more than (b). Every dead direction trained on a PROXY metric (NLL, embedding MSE, or corpus-continuation ranking) and evaluated on FUNCTIONAL benchmarks. None has directly trained on benchmark-facing MCQ margins. B15's FMD_SHADOW_288 is the first design that does this. It deserves exactly one shot.

However: if FMD_SHADOW_288 also fails, the scaffold hypothesis (b) must be tested directly — e.g., can ANY training procedure improve this student's benchmark margins?

### Narrative Gate

**One-sentence story:** "We pivoted from hidden-coordinate copying to teacher-margin training, ran the first smoke, and the shard-derived margins still didn't make the byte student better at choosing answers."

**Survives "isn't that obvious?":** Yes — the training loss decreased, so the negative result is diagnostic. The model learned shard preferences, not benchmark discrimination.

**Survives "so what?":** Only as kill discipline. Not as a moonshot result.

**Narrative alive?** No. The pivot has not yet produced a positive functional result. The narrative remains dead until a benchmark-facing training formulation shows held-out improvement over same-budget baselines.

---

## Decisions

### D1: FMD_SHADOW_288 Gets Exactly One More Shot

Per B15's "one repair cycle if failure identifies a specific fixable artifact" rule:
- The fixable artifact is **data mismatch** (shard continuations vs benchmark MCQ)
- FMD_SHADOW_288 uses benchmark-style train data with benchmark-style evaluation
- If this also fails: FMD direction is demoted. Do not repair again.

### D2: Adopt The Portfolio Approach

Per B15's process changes, the next batch runs a small evidence board, not a serial deep-dive:

| Probe | Purpose | Dependency |
|---|---|---|
| FMD_SHADOW_288 | Test margin objective on benchmark-style data with strong baselines | None — run first |
| Teacher-margin data audit | Measure teacher accuracy, margin quality, hard-negative quality on 200+ examples | None — run in parallel |
| Disagreement density audit | Measure useful disagreement between Qwen + one other teacher | Needs second teacher loaded |
| Scaffold capacity check | Can ANY short training procedure improve this student's benchmark margins? (even label-only CE) | None — run in parallel |

### D3: Add Scaffold Capacity Check

This is new. After 6 kills, we need to know if the problem is the objectives or the student. A simple test:
- Take the exact B11 student architecture
- Train with supervised label-only CE on benchmark train examples
- If even THIS doesn't improve benchmark margins, the student/scaffold is the bottleneck, not the KD objective

### D4: Split Evidence Labels

Per B15:
- `ADMISSION_EVIDENCE` = can earn a second experiment
- `MOONSHOT_EVIDENCE` = byte-native, held-out, strong-baseline-beating, artifact-controlled

Nothing in the project currently qualifies as MOONSHOT_EVIDENCE.

---

## Confidence Table

| Claim | Confidence | Evidence |
|---|---|---|
| Unlabeled continuation margins → benchmark margins | **5%** | B11 FAIL_MARGIN_PROTOTYPE, -12pp HellaSwag |
| Benchmark-style margin training can work (FMD_SHADOW_288) | 30% | Untested, but fixes the identified data mismatch |
| The byte-codec scaffold can learn benchmark discrimination at all | 50% | No direct evidence either way — scaffold capacity check needed |
| Disagreement router has useful fuel | 25% | Untested — needs density audit |
| Next direction will be the moonshot | 10% | 6 consecutive kills lowers base rate |

---

## Launch Orders

### Q-Loop B16 (Iterations 106-112): Attack After A Second Kill

**Goal:** Now that BOTH coordinate inheritance and FMD prototype have failed, interrogate the deeper pattern. 7 iterations:

1. **Is the scaffold the bottleneck?** After 6 kills with different objectives but the same codec→student→Qwen-head scaffold, should we test the scaffold directly? What would a scaffold capacity check prove?
2. **FMD_SHADOW_288 design review:** Attack B15's proposed experiment. What are its failure modes? Is benchmark-style training just supervised fine-tuning with extra steps?
3. **The 6-kill pattern:** Is there a common root cause? Every dead direction used a proxy training signal. Is the lesson "train directly on the evaluation metric" or is it "this student can't learn"?
4. **Scale question:** B11 used 50 examples and 10 steps with a tiny random student. Is this a fair test of ANY KD method, or is the scale too small to conclude anything?
5. **Competitive reality check:** CBD got 42.65% HellaSwag at 138M via chain KD. What did they do differently? What's their student architecture? Is there a structural lesson?
6. **Alternative portfolios:** If FMD_SHADOW_288 fails, what's left? Rank the remaining candidates: counterfactual minimal pairs, Error Atlas, teacher debate, direct S0 fine-tuning, abandon KD entirely.
7. **Honest moonshot probability:** Given 6 kills, what is the realistic probability that Eklavya produces a paradigm-shifting result? Should the project pivot to a different moonshot entirely?

### W-Loop B12 (Iterations 111-120): Portfolio Probe

**Goal:** Run the evidence board from D2. Four parallel probes:

1. **FMD_SHADOW_288** — B15's design: 288 benchmark-style train examples (96 each from HellaSwag/PIQA/ARC-Easy train-safe), 144 held-out eval, margin ranking loss with teacher targets, 5 strong baselines (label-only CE, MSE adapter, random margins, length/position control, untrained baseline). Precommitted PASS/MARGINAL/FAIL tokens per B15.
2. **Teacher-margin data audit** — Score 200+ examples per benchmark with full Qwen teacher. Measure: teacher accuracy, margin quality, teacher-wrong rate, hard-negative quality. No training — data quality check only.
3. **Scaffold capacity check** — Train the SAME B11 student architecture with simple supervised label-only CE on 288 benchmark train examples, 10+ steps. If even supervised training can't move benchmark margins, the scaffold is the bottleneck.
4. **Disagreement density audit** (if feasible) — Load a second teacher (Mamba or Granite), score 200+ examples, measure useful disagreement rate and oracle routing ceiling.

**Kill conditions:**
- FMD_SHADOW_288: FAIL if <+2pp over same-budget label-only CE on >=2/3 benchmarks
- Scaffold capacity: FAIL_SCAFFOLD if label-only CE also can't move margins → pivot away from this student architecture
- Teacher audit: FAIL_TEACHER if Qwen accuracy <50% on >=2/3 benchmarks → teacher is too weak
- Disagreement: FAIL if useful disagreement <8% or oracle ceiling <+2pp over best teacher

---

## Dual-Loop Status

| Loop | Last Batch | Status | Next |
|------|-----------|--------|------|
| W-Loop | B11 (KILL: FMD prototype) | Natural-continuation margins → benchmark: FAIL | B12: Portfolio probe (4 parallel probes) |
| Q-Loop | B15 (attack the pivot) | Pivot alive but prior-art-laden; portfolio approach mandated | B16: Attack after second kill |

**The dual-loop is now in its hardest phase: it must find something that works, not just kill things that don't.** 6 kills prove the falsification machinery is strong. The question is whether the invention machinery can match it.
