# CTI Precommit Specification v0

**Date: 2026-07-07**
**Status: DRAFT — requires Codex design-gate review before any experiments**
**Origin: Q-Loop B19 I131 + B20 analysis**

---

## 1. What Is CTI?

CTI (Compute Thermodynamics of Intelligence) asks: **Is there a predictive law relating
compute spent to functional distortion achieved, and do different interventions change that
law's parameters in classifiable ways?**

CTI is NOT:
- "Loss follows a power law with compute" (that's Kaplan et al. 2020)
- A manifesto or philosophy
- A claim about universal intelligence

CTI IS:
- A precommitted forecasting program
- Testable on one GPU with small models
- Falsifiable: if it doesn't predict better than naive baselines, it's dead

---

## 2. Distortion Definition (D)

**Precommitted BEFORE any experiment:**

Primary scalar for law fitting:
- **D_func = 1 - mean_accuracy** (held-out MCQ forced-choice accuracy, mean over benchmarks)

Supporting diagnostics (tracked but NOT used for law fitting):
- D_proxy = training loss (NLL/BPB)
- D_cal = calibration error (ECE)
- D_gap = |train_accuracy - held_out_accuracy| (generalization gap)
- D_margin = mean margin between gold and best-wrong choice NLL

The key CTI claim: **D_proxy and D_func can diverge** — an intervention that reduces proxy
distortion may not reduce functional distortion. This is exactly what Eklavya's B14 showed
(label-only memorized train but didn't improve held-out). CTI must predict WHICH compute
trajectories reduce D_func vs only D_proxy.

---

## 3. Compute Accounting (C)

Primary variable for law fitting:
- **C = cumulative FLOPs** (forward + backward, estimated from model size and step count)

Supporting variables (tracked):
- Wall-clock GPU seconds
- Peak VRAM
- Trainable parameter count
- Data tokens processed

For pretrained models (SmolLM2), C includes only LOCAL compute (fine-tuning FLOPs), not
pretraining. Pretraining budget is treated as a fixed "birth" property of the model family.

---

## 4. Law Forms (precommitted)

```
Power law:        D_func(C) = D_inf + k * C^(-alpha)
Broken power law: D_func(C) = D_inf + k * C^(-alpha_1) * sigmoid((C - tau) / beta)
Null:             No stable extrapolation beats naive baselines
```

Free parameters fit on SEEN compute points. Prediction tested on HELD-OUT compute points.

---

## 5. Model/Task Grid (one RTX 5090)

### Models
| Family | Sizes | Birth type | Local? |
|--------|-------|-----------|--------|
| SmolLM2 | 135M, 360M | Pretrained BPE (2T tokens) | Yes (cached) |
| Random tiny transformer | 1M-10M | Random init | Yes (train from scratch) |
| Tiny byte transformer | 5M-20M | S0-style random init | Yes (train from scratch) |
| 2-layer MLP | 50K-500K | Random init | Yes |

### Tasks
| Family | Examples |
|--------|----------|
| MCQ forced-choice | HellaSwag, PIQA, ARC-Easy (existing pipeline) |
| Modular arithmetic | Addition mod p (grokking tasks) |
| Sparse parity | k-sparse parity on n bits |
| One non-language | CIFAR-10 or a simple vision task (control) |

### Compute Schedules
Log-spaced step budgets: 10, 30, 100, 300, 1000, 3000, 10000 steps
(where feasible given model/task size)

### Interventions
| Type | Levels |
|------|--------|
| Architecture | random tiny vs pretrained vs byte |
| Objective | label-only CE vs KD vs reconstruction |
| Data quality | 25% / 50% / 100% labels, shuffled-label control |
| Training regime | full fine-tune vs LoRA vs frozen-probe |

---

## 6. Baseline Forecasters

CTI must beat ALL of these:

| Baseline | Description |
|----------|-------------|
| B0: Last-point | Predict next D_func = current D_func |
| B1: Linear extrapolation | Linear fit on log(C) vs D_func |
| B2: Per-task independent power law | Fit separate power law per task, no shared structure |
| B3: Proxy-only forecast | Use D_proxy trajectory to predict D_func |
| B4: Random intervention ranking | Randomly rank which intervention is best |

---

## 7. Prediction Protocol

1. **Fit phase**: Use the FIRST 30% of compute points (steps 10, 30, 100 of a 1000-step run)
2. **Predict phase**: Forecast D_func at remaining 70% of compute points
3. **Intervention ranking**: Given 2+ interventions at 30% budget, predict which will have
   lower D_func at full budget
4. **Evaluation**: Mean absolute prediction error on held-out points + intervention ranking
   accuracy

**Cross-validation**: For each task family, hold out one task or one model size. Fit on
the rest, predict on held-out.

---

## 8. Verdict Tokens

| Token | Meaning | Criteria |
|-------|---------|----------|
| INVALID_CTI | Experiment flawed | Bad compute accounting, data leak, wrong metric |
| NO_PREDICTIVE_LAW | CTI doesn't predict | Forecast error >= all baselines on >=2/3 task families |
| PROXY_ONLY_LAW | Predicts proxy but not function | CTI predicts D_proxy but NOT D_func better than baselines |
| PASS_CTI_LAW_0 | First positive result | CTI predicts D_func better than ALL baselines on >=2/3 task families AND correctly classifies >=1 intervention as constant-shift vs exponent-shift |
| STRONG_CTI | Robust law | PASS_CTI_LAW_0 + prediction transfers to held-out task family + intervention ranking accuracy >=70% |
| MOONSHOT_CTI | Paradigm-shifting | STRONG_CTI + law saves real compute (predicted optimal intervention beats default) + works across language and non-language tasks |

---

## 9. Kill Discipline

- If NO_PREDICTIVE_LAW after full grid: CTI is dead. Pivot to renormalization or CDMD.
- If PROXY_ONLY_LAW: CTI is interesting but not the manifesto claim. Publish narrow result, consider
  renormalization lane for the functional-distortion gap.
- FMD repair shot from Eklavya: NOT APPLICABLE to CTI. Do not carry forward.
- Maximum experiment budget before first gate: 48 GPU-hours (2 days on RTX 5090).
- If no PASS_CTI_LAW_0 within 48 GPU-hours: downgrade to exploratory.

---

## 10. Eklavya Salvage Integration

The Eklavya arc provides FREE data for CTI:
- 9 kills with known compute budgets and functional distortion measurements
- Train/held-out divergence patterns (proxy moved, function didn't)
- Teacher disagreement oracle ceiling data
- Multiple interventions (label-only, KD, routing) on same student

This data can seed the CTI law fitting WITHOUT new experiments. The first CTI artifact
should be: **fit D_func(C) curves to Eklavya's existing data and check if the law form
predicts the 9 kill outcomes**.

---

## 11. What This Spec Does NOT Cover (future iterations)

- Thermodynamic interpretation (entropy, free energy, temperature) — earn the language
  AFTER the law predicts
- Renormalization lane — separate precommit spec needed
- Public claims — no CTI language until PASS_CTI_LAW_0
- Theory — empirical law FIRST, theory after

---

## 12. Narrative Precommit

**Allowed after PASS_CTI_LAW_0:**
"A precommitted compute-distortion law predicted which training interventions would improve
held-out performance before they finished running, on one consumer GPU."

**Forbidden until STRONG_CTI:**
"Universal law of intelligence"
"Compute thermodynamics"
"We discovered how intelligence scales"

**Normal-person headline target:**
"A laptop predicted which AI training ideas were worth the electricity before they finished."
