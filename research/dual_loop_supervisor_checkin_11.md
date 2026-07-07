# Dual-Loop Supervisor Check-In #11

**Date:** 2026-07-07
**Loops completed:** W-Loop B12 (portfolio probe, 4 diagnostics), Q-Loop B16 (attack after second kill)
**Status:** PROJECT-LEVEL INFLECTION — FAIL_SCAFFOLD proves the student architecture is the bottleneck. All 6 prior objective-blame kills were misattributed. The substrate is untrainable.

---

## What Happened

### Q-Loop B16: The Sharpest Analysis Yet

7 iterations interrogating the 6-kill pattern. Core verdict: **scaffold capacity is THE question, not the objective.** Key outputs:

1. **Scaffold bottleneck hypothesis (Iteration 106):** After 6 kills with different objectives but the same scaffold, the constant substrate is under indictment. "Stop asking whether the next objective is clever until you've asked whether the student can learn under the least clever objective."

2. **FMD_SHADOW_288 is MCQ fine-tuning risk (Iteration 107):** If benchmark labels supply the gold answer, the teacher adds only hard-negative selection and margin strength. Must beat label-only CE to prove teacher value.

3. **Single root cause (Iteration 108):** Every dead direction improved a proxy that was not the functional benchmark. But the deeper cause may be student capacity, not objective choice.

4. **CBD structural lesson (Iteration 109):** CBD gets 42.65% HellaSwag at 138M through chain-init — coordinate continuity and inherited representation. "CBD transfers an already-formed feature basis. You are asking a scratch byte scaffold to infer that basis from a few proxy losses."

5. **Honest moonshot probability (Iteration 111):** Current scaffold → paradigm shift: 1-3%. "After six kills, the burden is no longer to invent a better loss. The burden is to prove the student can learn function at all."

6. **Terminal board framing (Iteration 112):** W-Loop B12 is the terminal admission board. Decision matrix clearly specified per outcome.

### W-Loop B12: The Terminal Admission Board

Four diagnostic probes. Results:

#### Probe A: FAIL_SCAFFOLD (THE decisive result)

| Benchmark | Untrained | Label-CE trained (50 steps) | Delta |
|---|---:|---:|---:|
| HellaSwag | 20.83% | 20.83% | **0.0pp** |
| PIQA | 56.25% | 54.17% | **-2.1pp** |
| ARC-Easy | 22.92% | 14.58% | **-8.3pp** |

**The MarginStudent cannot learn even from direct supervised labels.** 0/3 benchmarks improved. ARC-Easy REGRESSED 8pp. 50 gradient steps with cross-entropy on gold labels did not produce any held-out improvement.

**This kills ALL future objective experimentation on this scaffold.** If labels can't move it, no margin loss, ranking loss, disagreement router, or teacher signal can.

#### Probe B: MARGINAL_TEACHER

| Benchmark | Qwen acc | Positive margin | Confident wrong |
|---|---:|---:|---:|
| HellaSwag | 49.5% | 49.5% | 49.5% |
| PIQA | 67.5% | 67.5% | 28.5% |
| ARC-Easy | 34.0% | 34.0% | **65.5%** |

Qwen is usable on PIQA, borderline on HellaSwag, and **actively poisonous on ARC-Easy** (65.5% confident-wrong). This makes Qwen unsuitable as sole teacher, especially for ARC.

#### Probe C: SKIPPED_BY_FAIL_SCAFFOLD

FMD_SHADOW_288 was conditionally dependent on Probes A+B passing. Probe A FAIL_SCAFFOLD hard-killed it. **Critically: the FMD repair shot was NOT consumed.** If a new scaffold passes capacity check, FMD can still be tested.

#### Probe D: PASS_DISAGREEMENT (The One Bright Spot)

| Benchmark | Qwen acc | SmolLM2-360M acc | Top-1 disagreement | Useful disagreement | Oracle ceiling | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 49.5% | 58.5% | 25.0% | 17.0% | 62.5% | +4.0pp |
| PIQA | 67.5% | 71.5% | 20.0% | 20.0% | 79.5% | +8.0pp |
| ARC-Easy | 34.0% | 54.5% | 49.0% | 34.5% | 61.5% | +7.0pp |
| **Aggregate** | **50.3%** | **61.5%** | **31.3%** | **23.8%** | **67.8%** | **+6.3pp** |

**Teacher disagreement fuel is abundant.** 23.8% useful disagreement, oracle ceiling beats best single teacher by +6.3pp. This validates the Eklavya thesis — teachers disagree informatively, and an oracle router would materially improve over any single teacher.

Also: **SmolLM2-360M is dramatically stronger than Qwen3-0.6B** (61.5% vs 50.3% aggregate). The project may have been using a suboptimal primary teacher.

---

## Supervisor Assessment

### The Root Cause Was Always The Scaffold

This is the clearest diagnostic result the project has produced. For 6 kills, the dual-loop blamed objectives:

| Kill | Blamed | Real cause |
|---|---|---|
| Gradient KD | Objective too weak | Scaffold can't learn |
| Brainseed | Wrong scorer/readout | Scaffold can't learn |
| Evidence-Native v0/v1 | Wrong architecture/corpus | Scaffold can't learn |
| Coordinate Inheritance v0/v1/v2 | Wrong coordinate transfer | Scaffold can't learn |
| FMD prototype | Wrong training data | Scaffold can't learn |
| **Probe A** | **N/A — tested directly** | **CONFIRMED: scaffold can't learn** |

Every time we changed the objective, the scaffold was the constant that silently ate the signal. The project spent 6 kill cycles learning this.

### What FAIL_SCAFFOLD Means

The MarginStudent architecture:
- Frozen codec encoder (CausalByteTransformer, 4 layers, 256-dim)
- Trainable input projection
- 2-layer global reasoner
- 1-layer byte decoder
- Random initialization

This combination cannot learn MCQ discrimination in 50 gradient steps from direct supervised labels. The representation capacity may be too small, the initialization too far from useful, or the codec-to-student interface too lossy. **The specific cause doesn't matter yet — what matters is that NO objective can succeed on this substrate.**

### What PASS_DISAGREEMENT Means

The Eklavya thesis is NOT dead. Teacher disagreement is real, abundant, and informative. An oracle router over Qwen + SmolLM2-360M would reach 67.8% aggregate — far above either teacher alone. This is genuine complementarity, not noise.

But disagreement fuel without a trainable student is "fuel without an engine" (B12's exact words).

### The Path Forward

Q-Loop B16 Iteration 110 ranked the options clearly:

**If scaffold capacity fails (which it did):**
1. **Change student architecture / drop current codec scaffold** — Highest leverage
2. **Direct S0/Wide7 fine-tuning on a stronger or pretrained base** — New capacity floor
3. **Abandon KD and pivot to another moonshot** — Serious option
4. Counterfactual curriculum — Only after learnable scaffold exists
5. Error Atlas — Cannot patch a student that can't absorb patches
6. Teacher debate — Adds complexity before capacity

The supervisor concurs with this ranking. The next batch must change the student, not the objective.

### Concrete Options For A New Student

| Option | Description | Pros | Cons |
|---|---|---|---|
| **A. Use pretrained S0/Wide7** | The actual 121M trained model, not a tiny random MarginStudent | Already trained, has language model capacity | Still byte-level, may not help with MCQ |
| **B. CBD-style chain-init** | Pretrained Qwen/SmolLM → intermediate → Sutra student | Proven to work at 138M (CBD), transfers coordinates | Needs same-tokenizer chain or byte interface design |
| **C. Direct SmolLM2-360M fine-tuning** | Take SmolLM2-360M, fine-tune with teacher margins | Already 61.5% aggregate, token-level model | Not byte-native, not Sutra architecture |
| **D. Hybrid: pretrained core + byte interface** | Pretrained transformer core + byte codec input adapter | Combines pretrained capacity with byte interface | Design complexity, may not work |
| **E. Abandon KD, different moonshot** | Pivot to CTI, renormalization, CWC per CLAUDE.md | Fresh start, no sunk cost | Abandons all Eklavya infrastructure |

### Supervisor's Recommendation

**Option A first (cheapest), then B if needed.**

Before designing a new student from scratch, test whether the EXISTING trained S0/Wide7 (121M params, already language-modeling capable) can learn MCQ discrimination from labels. This is the same scaffold capacity check but on a model that actually has language model capacity.

If S0/Wide7 can learn labels: the MarginStudent was simply too small/random. Use S0 as the student for FMD + disagreement routing.

If S0/Wide7 cannot learn labels: the byte-level architecture itself may be the problem. Consider Option B (chain-init) or Option E (project pivot).

### FMD Repair Shot Preserved

Because Probe C was SKIPPED (not FAILED), the one FMD repair shot from B15's one-repair rule is still available. It should be tested on whatever scaffold passes the capacity check.

---

## Decisions

### D1: Stop All Objective Experimentation On MarginStudent

No more margin losses, ranking losses, KD objectives, or teacher signals on the B11 MarginStudent architecture. It can't learn labels. Nothing else will help.

### D2: Run S0/Wide7 Scaffold Capacity Check

The next W-Loop must test whether the trained S0 model (121M, eval_bpb 1.900, HellaSwag 26.3% zero-shot) can learn MCQ discrimination from supervised labels. Same test as Probe A but on a real model.

### D3: Upgrade Teacher Portfolio

SmolLM2-360M (61.5%) is dramatically stronger than Qwen3-0.6B (50.3%). Future KD should use SmolLM2-360M as primary teacher (or at minimum as co-teacher). Also: install Mamba CUDA kernels to unlock Mamba-790M as a structurally diverse SSM teacher.

### D4: Preserve Disagreement Infrastructure

PASS_DISAGREEMENT is real and reusable. The disagreement density audit code and results should be preserved for the next student that passes capacity check.

### D5: FMD Repair Shot Still Available

One FMD_SHADOW_288 experiment is still allowed on the next scaffold that passes capacity check. Don't waste it on another untrainable student.

---

## Confidence Table

| Claim | Confidence | Evidence |
|---|---|---|
| MarginStudent can learn benchmark discrimination | **2%** | FAIL_SCAFFOLD: 0/3 benchmarks, -8pp ARC regression |
| The problem is the scaffold, not the objective | **90%** | 6 objective kills + direct scaffold capacity fail |
| Teacher disagreement fuel exists | **85%** | PASS_DISAGREEMENT: 23.8% useful, +6.3pp oracle gap |
| Qwen3-0.6B is a good primary teacher | 25% | MARGINAL_TEACHER: only PIQA passes, ARC poisoned |
| S0/Wide7 can learn MCQ discrimination | 40% | Untested but has 121M params + LM pretraining |
| Eklavya KD can work with a proper student | 30% | Disagreement exists, but no student has learned yet |
| Current line produces paradigm shift without major pivot | **5%** | 6 kills + scaffold bottleneck |

---

## Launch Orders

### Q-Loop B17 (Iterations 113-119): The Architecture Question

**Goal:** Now that the scaffold is proven untrainable, interrogate the architecture question. 7 iterations:

1. **Why did the MarginStudent fail?** Capacity too small? Random init too far? Codec interface too lossy? Can we diagnose which?
2. **Can S0/Wide7 learn MCQ?** The real 121M byte model has language ability but only 26.3% HellaSwag. Is supervised MCQ training on a pretrained byte model even expected to work?
3. **The CBD structural lesson revisited.** CBD uses chain-init with same-tokenizer models. What would a byte-native chain-init look like? Is it even possible across tokenizers?
4. **SmolLM2-360M as student baseline.** It's already 61.5% aggregate. Fine-tuning it with teacher margins is the easiest path to numbers. But is it Sutra? Is it byte-native? Does it serve the manifesto?
5. **Is byte-native a load-bearing constraint?** The Vision says byte-level is a means, not a goal. Should we drop byte-native and use token-level students?
6. **What would a hostile reviewer say now?** 6 kills + scaffold FAIL + no positive functional result. The disagreement pass is interesting but has no engine. What is the honest narrative?
7. **Decision: pivot KD approach, pivot student, or pivot moonshot entirely?** Make a concrete recommendation.

### W-Loop B13 (Iterations 121-130): S0/Wide7 Capacity Check + Teacher Upgrade

**Goal:** Test the real trained models. 4 probes:

1. **S0 scaffold capacity check** — Same protocol as Probe A but using the actual trained S0 model (or Wide7). Supervised label-only CE on 288 benchmark examples, 50+ steps, 144 held-out eval. If S0 can learn: the MarginStudent was just too weak. If S0 can't: byte-level architecture itself may be the bottleneck.
2. **SmolLM2-360M as strong-teacher baseline** — Score 200+ examples with SmolLM2 as primary teacher. Compare margin quality against Qwen. If SmolLM2 is clearly better, upgrade all future teacher infrastructure.
3. **FMD_SHADOW_288 on S0 (conditional on S0 capacity pass)** — The preserved repair shot. Same design as B15 Iteration 102, but on S0 instead of MarginStudent. Only if S0 passes capacity check.
4. **Mamba kernel install (if feasible)** — Install mamba-ssm CUDA kernels, then re-run disagreement density with Mamba-790M for true cross-architecture diversity.

**Kill conditions:**
- S0 capacity: FAIL if <+2pp on >=2/3 benchmarks → byte architecture is the bottleneck
- FMD on S0: FAIL if <+3pp over label-only CE → teacher margins don't add value even on a capable student
- If BOTH fail: recommend project-level pivot per Q-Loop B16

---

## Dual-Loop Status

| Loop | Last Batch | Status | Next |
|------|-----------|--------|------|
| W-Loop | B12 (FAIL_SCAFFOLD + PASS_DISAGREEMENT) | MarginStudent can't learn; disagreement fuel exists | B13: S0/Wide7 capacity check + teacher upgrade |
| Q-Loop | B16 (terminal board verdict) | Scaffold is the bottleneck; honest probability <5% | B17: Architecture question |

**The dual-loop has reached its most honest moment.** It proved the scaffold is the bottleneck — not the objectives, not the teachers, not the losses. The disagreement fuel is real but needs an engine. The next batch determines whether the engine exists (S0) or must be built from scratch (chain-init / architecture change / moonshot pivot).
