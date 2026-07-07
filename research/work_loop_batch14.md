# W-Loop B14: TERMINAL Eklavya Mechanism Control — SmolLM2-135M

**Verdict: MARGINAL_EKLAVYA (harness) / FAIL_EKLAVYA_MECHANISM (honest)**
**Date: 2026-07-07**
**Elapsed: 4502s (~75 min)**
**Kill #9: Eklavya routing mechanism — no residual over single-teacher KD**

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Student | SmolLM2-135M (134.5M params, BPE, pretrained 2T tokens) |
| Teachers | SmolLM2-360M (361.8M) + Qwen3-0.6B (596.0M) |
| LoRA | rank 16, alpha 32, dropout 0.05, 1.84M trainable / 136.4M total |
| Training | 150 steps, batch=12, lr=2e-4, AdamW, grad_clip=1.0 |
| Alpha | 0.5 CE + 0.5 KD (all KD conditions), 1.0 CE (label-only) |
| Max length | 768 tokens |
| Eval | held-out split, n=48 per benchmark (train-safe, zero source overlap) |
| Benchmarks | HellaSwag, PIQA, ARC-Easy |
| Seed | 20260707 |

---

## Teacher Cache

| Metric | Train (n=288) | Eval (n=144) |
|--------|---------------|--------------|
| SmolLM2-360M accuracy | 61.1% | 60.4% |
| Qwen3-0.6B accuracy | 51.4% | 49.3% |
| Top-1 disagreement | 33.0% | 29.9% |
| Useful disagreement | 27.8% | 20.8% |
| Oracle ceiling | 70.1% | 65.3% |
| Oracle gap over best teacher | +9.0pp | +4.9pp |

Disagreement fuel EXISTS. Oracle gap is real. The question is whether any routing mechanism can harvest it.

---

## Accuracy Table (held-out, n=48 per benchmark)

| Benchmark | Zero-shot | Label-only | Single-T(360M) | Oracle route | Non-oracle | Random route |
|-----------|-----------|------------|----------------|--------------|------------|--------------|
| HellaSwag | 39.58% | 41.67% | 39.58% | 41.67% | **43.75%** | 39.58% |
| PIQA | **75.00%** | 68.75% | **75.00%** | **75.00%** | 70.83% | **75.00%** |
| ARC-Easy | 50.00% | 54.17% | **62.50%** | 58.33% | 56.25% | 60.42% |
| **Mean** | 54.86% | 54.86% | **59.03%** | 58.33% | 56.94% | 58.33% |

---

## Verdicts (precommitted tokens)

| Condition | Verdict | Benchmarks passing threshold |
|-----------|---------|------------------------------|
| Label-only vs zero-shot (>=3pp) | MARGINAL_LABEL_ONLY | 1/3 (ARC-Easy +4.17pp) |
| Single-teacher vs label-only (>=2pp) | PASS_SINGLE_TEACHER | 2/3 (ARC-Easy +8.33pp, PIQA +6.25pp) |
| Non-oracle vs both (>=3pp over both) | MARGINAL_EKLAVYA | 0/3 |
| Oracle vs single-teacher | MARGINAL_EKLAVYA | 0/3 |
| **Primary** | **MARGINAL_EKLAVYA** | -- |

---

## Critical Delta Analysis

### Non-oracle routing vs single-teacher (THE EKLAVYA TEST)
| Benchmark | Delta | CI 95% | Verdict |
|-----------|-------|--------|---------|
| HellaSwag | +4.17pp | [0.0, +10.4] | Only win, barely significant |
| PIQA | -4.17pp | [-12.5, +4.2] | Loss |
| ARC-Easy | -6.25pp | [-16.7, +2.1] | Loss |
| **Aggregate** | **-2.08pp** | -- | **Routing HURTS** |

### Oracle routing vs single-teacher (THE CEILING)
| Benchmark | Delta |
|-----------|-------|
| HellaSwag | +2.08pp |
| PIQA | +0.00pp |
| ARC-Easy | -4.17pp |
| **Aggregate** | **-0.69pp** |

**Even perfect routing doesn't beat single teacher. The ceiling is below the floor.**

### Random routing vs non-oracle (FALSIFICATION)
| Benchmark | Non-oracle | Random | Delta |
|-----------|-----------|--------|-------|
| HellaSwag | 43.75% | 39.58% | +4.17pp (routing helps) |
| PIQA | 70.83% | 75.00% | -4.17pp (routing hurts) |
| ARC-Easy | 56.25% | 60.42% | -4.17pp (routing hurts) |
| **Mean** | **56.94%** | **58.33%** | **-1.39pp (random beats learned)** |

**The learned router is WORSE than random routing on aggregate.**

---

## Training Histories

| Condition | First loss | Final loss | Final batch acc |
|-----------|-----------|------------|-----------------|
| label_only | 1.066 | 0.320 | 100% |
| single_teacher | 0.371 | 0.372 | 83% |
| oracle_route | 0.647 | 0.371 | 92% |
| non_oracle | 0.421 | 0.362 | 100% |
| random_route | 0.597 | 0.383 | 83% |

Label-only reaches 100% batch accuracy -- memorization. Single-teacher maintains 83% -- better generalization signal.

---

## Why the Mechanism Fails

1. **Oracle ceiling is below single-teacher floor**: The multi-teacher routing CONCEPT doesn't work at this setup, not just the implementation. No possible router can beat single-teacher.

2. **Conflicting teacher signals cancel out**: SmolLM2-360M and Qwen3-0.6B push the student in different directions. The consistent signal from one good teacher (SmolLM2-360M) beats the averaged/routed signal from two disagreeing teachers.

3. **The disagreement oracle gap is a mirage**: The teacher cache shows +9pp oracle gap, but this measures the ceiling of TEACHER accuracy, not STUDENT accuracy. The student LoRA can't convert teacher-level routing into student-level accuracy gains.

4. **Label-only overfits**: 100% final batch accuracy with PIQA regression (-6.25pp). The CE-only objective memorizes training examples without generalizing.

5. **n=48 is noisy but the pattern is clear**: Wide CIs on individual benchmarks, but the systematic ranking (single-teacher > random >= oracle > non-oracle on aggregate) is consistent.

---

## Limitations

1. Train-safe held-out split is not public benchmark validation.
2. Oracle routing is a label-leaking ceiling, not a deployable method.
3. Non-oracle routing uses cached teacher choice distributions; teachers are not loaded during training.
4. All conditions use the same LoRA configuration, step budget, optimizer, alpha, and data split.

---

## Evidence Hierarchy Classification (Q-Loop B18 I126)

| Level | Status |
|-------|--------|
| 1. INVALID_EKLAVYA_TEST | No -- test is valid |
| 2. FAIL_TOKEN_STUDENT_CAPACITY | No -- student CAN learn (single-teacher proves this) |
| 3. ORDINARY_FINE_TUNING | Label-only is MARGINAL over zero-shot |
| 4. ORDINARY_KD | Single-teacher KD WORKS (+4.17pp mean over zero-shot) |
| **5. MARGINAL_EKLAVYA** | **HERE -- routing > label-only but < single-teacher** |
| 6. PASS_EKLAVYA_MECHANISM | NOT reached |
| 7. STRONG_EKLAVYA | NOT reached |
| 8. MOONSHOT_CANDIDATE | NOT reached |

---

## Terminal Decision

### Precommitted continuation bar (from check-in #12):
- Non-oracle routing >=3pp over BOTH label-only AND single-teacher: **NOT MET (0/3)**
- On >=2/3 benchmarks: **NOT MET (0/3)**
- Disagreement-slice lift: **MOOT (oracle ceiling fails)**
- Consensus not damaged: **AMBIGUOUS**

### Honest verdict:
The harness generously coded MARGINAL because routing beats label-only by ~2pp. But the terminal test was "beats BOTH baselines by >=3pp." The Eklavya-specific residual (routing minus single-teacher) is NEGATIVE (-2.08pp aggregate). The oracle ceiling itself fails (-0.69pp vs single-teacher). Random routing beats the learned router.

**TERMINAL VERDICT: FAIL_EKLAVYA_MECHANISM**

The Eklavya protocol -- multi-teacher disagreement routing as a learning mechanism -- has no measurable residual over single-teacher KD at this scale. Kill #9.

---

## Kill Record (cumulative)

| # | What | When | Why |
|---|------|------|-----|
| 1-6 | Byte-native objective variants | B7-B11 | No improvement over baselines |
| 7 | MarginStudent scaffold | B12 | Can't learn even supervised labels |
| 8 | S0/Wide7 byte capacity | B13 | Memorizes train, held-out flat |
| **9** | **Eklavya routing mechanism** | **B14** | **No residual over single-teacher KD** |

---

## NARRATIVE SECTION

**Gossip-magazine headline**: "Researchers test if two AI teachers can teach a small model better than one -- they can't."

**Survives "isn't that obvious?"**: Partially -- the KD literature shows multi-teacher benefits at larger scales. At this scale, the result is not obvious ex ante.

**Survives "that's trivial?"**: Yes -- we ran a proper controlled experiment with precommitted verdicts, oracle ceilings, and random controls. This is honest science.

**The honest narrative**: Single-teacher KD works. Multi-teacher routing adds nothing. The disagreement signal exists in the teacher outputs but cannot be harvested into student accuracy at 135M-param LoRA scale. This is a clean negative result with clear methodology. It is NOT a moonshot.

**What this means for the project**: Eklavya as designed is dead. The dual-loop methodology is the real artifact. Pivot trigger activated.
