# Dual-Loop Supervisor Check-in #12

Date: 2026-07-07
Loops covered: Q-Loop B17 (iterations 113-119), W-Loop B13 (4 probes)

## Evidence Summary

### W-Loop B13: S0/Wide7 Capacity Check + Teacher Upgrade

| Probe | Token | Key numbers |
|---|---|---|
| 1. Wide7 native fine-tune (BINDING) | `FAIL_S0_CAPACITY` | Train 38.5%→73.6%, held-out: HellaSwag 0pp, PIQA -2.1pp, ARC-Easy +8.3pp. 1/3 pass, needed 2/3. |
| 1b. Wide7 frozen residual head | `FAIL_S0_CAPACITY` | Train 38.5%→60.1%, held-out: HellaSwag -4.2pp, PIQA -8.3pp, ARC-Easy +4.2pp. 0/3 pass. |
| 1c. Wide7 50-step native | `FAIL_S0_CAPACITY` | Train 38.5%→58.0%, held-out: HellaSwag 0pp, PIQA +2.1pp marginal, ARC-Easy 0pp. 0/3 pass. |
| 2. SmolLM2-360M teacher audit | `UPGRADE_TEACHER` | SmolLM2: HellaSwag 56.0% (+6.5pp over Qwen), ARC-Easy 56.5% (+22.5pp!), PIQA 65.0% (-2.5pp). |
| 3. FMD on S0 | `SKIPPED_BY_FAIL_S0_CAPACITY` | Repair shot preserved. |
| 4. Mamba kernels | `DEFER_MAMBA_DISAGREEMENT` | CUDA DLL import failure. |

**Critical observation**: Wide7 (119M params, eval_bpb 1.293) can memorize 288 training examples (train acc 73.6% at 100 steps) but cannot generalize MCQ discrimination to held-out sets. This was tested THREE ways (frozen head, 50-step native, 100-step native) and all three show the same pattern: train overfits, held-out flat or regresses on 2/3 benchmarks.

### Q-Loop B17: The Architecture Question (7 iterations)

Verdict: `PIVOT_STUDENT_NOT_MOONSHOT_YET`

Key conclusions per iteration:
- **I113 (MarginStudent failure analysis)**: Primary lesion = missing semantic initialization + insufficient capacity. The shadow scaffold was never a real student.
- **I114 (S0/Wide7 MCQ capacity)**: Literature supports that pretrained LMs can be adapted, but 26.3% HellaSwag base is weak. 50 steps is a short diagnostic, not a fair fine-tune.
- **I115 (CBD structural lesson)**: Cross-tokenizer chain-init is not weight-copy — it's a new alignment research program. BPE-to-byte transfer must learn coordinate translation.
- **I116 (SmolLM2 as student)**: SmolLM2 is the fastest mechanism control but it's NOT Sutra evidence. Correct role: "mechanism control, not manifesto proof."
- **I117 (Byte-native load-bearing?)**: "Byte-native is a preferred mechanism, not an invariant. It earns mainline status only if S0/Wide7 can learn function." → S0/Wide7 FAILED. Byte-native is demoted.
- **I118 (Hostile reviewer)**: "Rigorous negative-results repo with no positive functional student result." One impressive thing: kill discipline + PASS_DISAGREEMENT. 
- **I119 (Decision)**: Rank A > B > C. Pivot student, not moonshot. Run token-level control. Reserve moonshot pivot as trigger if both fail.

Honest probability table from Q-Loop:

| Claim | Probability |
|---|---:|
| S0/Wide7 passes MCQ capacity | 35-45% → **ACTUAL: FAIL** |
| Token-level SmolLM2 shows Eklavya residual over label-only | 20-35% |
| Byte-native Eklavya reaches paradigm evidence without chain-init | 3-7% |
| Current line reaches public moonshot result without student pivot | ~0% |

## Supervisor Audit

### What Codex got right

1. Used the strongest available checkpoint (Wide7, BPB 1.293) for the binding test — a hostile reviewer cannot object that we tested the wrong model.
2. Ran THREE S0/Wide7 tests (frozen head, 50-step native, 100-step native) providing robust evidence that this is a structural issue, not a training-budget issue.
3. Correctly skipped FMD — preserving the repair shot for a student that actually passes capacity.
4. SmolLM2 teacher audit was clean and decisive: UPGRADE_TEACHER on 2/3 benchmarks with massive ARC-Easy improvement (+22.5pp over Qwen's poisoned 34%).
5. Q-Loop maintained adversarial tension through all 7 iterations with genuine steelman-then-attack discipline.

### What Codex missed or could challenge

1. **50 steps and 100 steps are both SHORT for full fine-tuning.** Q-Loop I114 noted this: "50 steps should not be treated as a fair final fine-tune. It is a cheap capacity gate." The binding 100-step result is better but still short. A hostile reviewer could demand 500-1000 steps to fully diagnose whether this is budget failure or structural failure. However: train accuracy reaching 73.6% at 100 steps proves the optimizer works — the generalization gap is the real issue, and more steps risks further overfitting on 288 examples.

2. **ARC-Easy actually PASSED at 100 steps (+8.3pp).** The binding result is 1/3 pass, not 0/3. This is not a total wipeout — ARC-Easy generalized while HellaSwag/PIQA did not. A kinder interpretation: the byte model CAN learn some MCQ discrimination, but 288 diverse-benchmark examples is too few to learn all benchmark formats. A focused single-benchmark fine-tune might pass. However: this would be benchmark engineering, not a moonshot capacity test.

3. **Q-Loop's probability estimate was wrong.** Q-Loop estimated 35-45% chance of S0 capacity pass. Actual result: fail. The Q-Loop was appropriately uncertain but its estimate was optimistic given the evidence.

4. **Neither loop addressed training data volume.** Both loops kept using 288 examples. At 121M params, 288 examples is ludicrously small for learning general MCQ function. But the question was capacity, not optimization — and the train overfitting proves the model has capacity to absorb signals, just not to generalize them from this tiny set.

### Narrative Gate

**Honest one-sentence narrative given ONLY what survived this checkpoint:**

"We tested our best trained byte model on the simplest possible task — learn benchmark answers from labels — and it memorized the training set but couldn't generalize, while our teacher analysis found strong complementary fuel waiting for an engine."

**"Isn't that obvious?"**: Partially. A 121M byte model fine-tuned on 288 MCQ examples failing to generalize is not shocking. But the PATTERN (memorization without generalization across three test configurations) is informative — it separates "not enough steps" from "wrong learning geometry."

**"So what?"**: As moonshot evidence, this is boring. As decision evidence, it's load-bearing: it prevents further wasted work on byte-native KD objectives and redirects to the decisive test of the Eklavya PROTOCOL on a capable substrate.

**Narrative verdict: DEAD for moonshot, ALIVE for decision-making.** The project has a paper's worth of negative results (rigorous falsification with kill discipline) but no moonshot story yet. The next result must be positive or the moonshot pivots.

## Decisions

### D1: Accept FAIL_S0_CAPACITY as binding. Byte-native student is DEMOTED from mainline.

The evidence is robust: three configurations, same pattern, on the strongest available byte checkpoint. The byte-native student path does not support MCQ function transfer at this scale/training state.

Byte-native is NOT killed permanently — it becomes a future architecture research direction (chain-init, cross-tokenizer bridge, larger scale). But it is no longer the active mainline student for Eklavya experiments.

### D2: Accept UPGRADE_TEACHER. SmolLM2-360M is primary teacher.

SmolLM2 dramatically outperforms Qwen on 2/3 benchmarks (especially ARC-Easy: 56.5% vs 34.0%). Qwen retained for PIQA co-teaching and disagreement routing only.

### D3: FMD repair shot REMAINS preserved.

Skipped twice (B12 by FAIL_SCAFFOLD, B13 by FAIL_S0_CAPACITY). Never consumed. Opens ONLY on a student that passes label-only capacity gate first.

### D4: Launch token-level SmolLM2 mechanism control (DECISIVE EKLAVYA TEST).

This is the make-or-break experiment for the Eklavya protocol:
- **Question**: Does multi-teacher disagreement produce residual held-out gains BEYOND label-only fine-tuning on a competent student?
- **Student**: SmolLM2-135M (same parameter class as Sutra, already has benchmark function)
- **Teachers**: SmolLM2-360M (primary) + Qwen3-0.6B (disagreement partner)
- **Baselines required**: (1) SmolLM2-135M zero-shot, (2) label-only fine-tune, (3) single-teacher KD, (4) best-teacher imitation
- **Pass condition**: Multi-teacher disagreement routing beats ALL baselines by >=3pp on >=2/3 benchmarks
- **If PASS**: Eklavya protocol is validated. Byte-native becomes an architecture return problem.
- **If FAIL**: Eklavya KD is not a moonshot. Pivot to CTI/renormalization/CWC.

### D5: Establish terminal decision board.

| Result | Action |
|---|---|
| SmolLM2 mechanism control shows Eklavya residual >=3pp | Eklavya protocol survives. Design byte-return path or accept token-level identity. |
| SmolLM2 label-only improves but no Eklavya residual | Eklavya is ordinary fine-tuning. Pivot moonshot entirely. |
| SmolLM2 nothing improves | Both student and protocol are dead. Full moonshot pivot. |

There are no more "one more test" options after D4. This is the terminal Eklavya test.

## Launch Orders

### Q-Loop B18: Attack the token-level pivot (7 iterations)

Focus: Is SmolLM2 mechanism control a moonshot or engineering? What baselines kill the claim? What would make the narrative paradigm-shifting?

Iterations:
- I120: Is SmolLM2-135M the right control student? (size, capability, fairness of comparison)
- I121: What baselines are truly needed to prove Eklavya adds value? (label-only, single-teacher, best-teacher imitation, random routing)
- I122: How should disagreement routing work in practice? (architecture, loss, training)
- I123: Can SmolLM2 mechanism results transfer back to byte-native? (what would that look like?)
- I124: What would make the narrative survive "that's just fine-tuning"?
- I125: Hostile reviewer reads the full repo including the token-level pivot. What do they say?
- I126: Final decision framework — what exactly constitutes Eklavya evidence vs ordinary KD?

### W-Loop B14: SmolLM2 mechanism control experiment (10 iterations)

This is the DECISIVE experiment. Probe design:

**Probe 1: SmolLM2-135M baseline evaluation (NO TRAINING)**
- Load SmolLM2-135M (HuggingFaceTB/SmolLM2-135M)
- Score 200+ examples per benchmark (HellaSwag, PIQA, ARC-Easy) with continuation NLL
- Report zero-shot accuracy, margin quality
- This establishes the untrained floor

**Probe 2: SmolLM2-135M label-only fine-tune**
- LoRA or full fine-tune on 288 train-safe examples with label CE
- 100-200 steps
- Held-out evaluation on 144 examples
- This is the baseline that Eklavya must beat

**Probe 3: SmolLM2-135M single-teacher KD**
- Train with SmolLM2-360M soft labels (KL divergence on choice NLLs)
- Same budget as Probe 2
- This separates "any teacher helps" from "disagreement helps"

**Probe 4: SmolLM2-135M multi-teacher disagreement routing**
- Use BOTH SmolLM2-360M and Qwen3-0.6B
- Route based on disagreement: when teachers disagree, weight toward the teacher that's right (using held-out validation)
- Same budget as Probe 2
- This IS the Eklavya test

**Precommitted verdict tokens:**
```
PASS_EKLAVYA_MECHANISM — disagreement routing beats BOTH label-only AND single-teacher KD by >=3pp on >=2/3 benchmarks
MARGINAL_EKLAVYA — disagreement routing beats single-teacher but not label-only, or <3pp
FAIL_EKLAVYA_MECHANISM — no Eklavya residual → protocol is dead, pivot moonshot
```

**Kill condition:** If Probe 2 (label-only) shows no improvement over zero-shot, SmolLM2-135M is too small/weak. Use SmolLM2-360M as student with a larger teacher, or pivot.

## Accumulated Kill Record (Updated)

| # | Direction | Kill token | Batch | Root cause |
|---|---|---|---|---|
| 1 | Gradient KD | FAIL | B3 | Proxy improved, benchmark flat |
| 2 | Brainseed | FAIL | B5 | Same pattern |
| 3 | Evidence-Native v0 | FAIL | B7 | Same pattern |
| 4 | Evidence-Native v1 | FAIL | B8 | Same pattern |
| 5 | Coordinate-Inheritance | FAIL | B9-B10 | Compatibility proxy, not function |
| 6 | FMD prototype | FAIL_MARGIN_PROTOTYPE | B11 | HellaSwag -12pp, train loss dropped |
| 7 | MarginStudent scaffold | FAIL_SCAFFOLD | B12 | 0/3 benchmarks, architecture bottleneck |
| 8 | S0/Wide7 capacity | FAIL_S0_CAPACITY | B13 | Memorizes train (73.6%), held-out flat (1/3 pass) |

**Pattern**: Kills 1-6 were objective/loss failures on a bad scaffold. Kill 7 proved the scaffold was the bottleneck. Kill 8 proved even the real trained byte model can't generalize MCQ from small supervision.

**What survives**: PASS_DISAGREEMENT (fuel exists), UPGRADE_TEACHER (SmolLM2 >> Qwen), FMD repair shot (never consumed). The Eklavya PROTOCOL has never been tested on a capable student. D4 tests that.

## State After This Check-in

- **Active moonshot**: Eklavya protocol test on token-level student (TERMINAL TEST)
- **Byte-native**: DEMOTED. Future architecture research, not current mainline.
- **Teachers**: SmolLM2-360M (primary), Qwen3-0.6B (PIQA/disagreement partner)
- **Student for test**: SmolLM2-135M (same param class as Sutra, has existing benchmark function)
- **FMD repair shot**: Still preserved, waiting for label-capacity pass
- **Moonshot pivot trigger**: FAIL_EKLAVYA_MECHANISM on D4
