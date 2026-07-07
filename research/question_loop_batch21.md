# Q-Loop B21: CTI Direction Refinement - Post-Salvage

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I141-I147
**Status:** analysis-only adversarial review; no model, dataset, GPU, or web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_13.md`
3. `research/CTI_PRECOMMIT_SPEC.md`
4. `research/work_loop_batch15.md`
5. `research/question_loop_batch20.md`
6. `research/question_loop_batch19.md`, with I131-I133 re-read explicitly after the first full read clipped in terminal output.

Binding state:

- Eklavya is dead as a mainline mechanism after the terminal SmolLM2 control. The pivot is CTI, not Eklavya repair.
- CTI is not allowed to mean "loss follows a power law." It only matters if it forecasts held-out functional distortion and intervention value before full-budget results are visible.
- CTI-0 is informative but not positive evidence. It found proxy/function divergence in B14 salvage data: final proxy loss anti-aligned with final held-out `D_func` (`Spearman rho = -0.8208`), step-30 proxy ranking had only `4/9` pairwise agreement with final functional ranking, proxy alpha varied from about `0.026` to `0.150`, and proxy power-law `R2` values were weak (`0.03` to `0.55`).
- CTI-1 as currently designed: random tiny transformer on modular arithmetic plus SmolLM2-135M LoRA on MCQ, log-spaced checkpoints `10, 30, 100, 300, 1000, 3000`, `D_func` at every checkpoint, fit on early points, predict later points, beat all baselines or die.

Current strongest position entering B21:

```text
CTI-0 does not prove a law, but it proves the next measurement target:
training proxy can move opposite to held-out function. CTI-1 should test whether
early functional compute-distortion curves can forecast later function and
intervention ranking better than naive baselines on one GPU.
```

---

## I141: Is CTI Just Rediscovering Overfitting?

### Steelman

The strongest CTI defense is not that overfitting exists. Everyone knows that. The defense is that CTI turns the overfitting warning into a decision problem:

```text
Given partial compute traces, can we forecast which intervention will reduce
held-out function rather than merely reduce proxy loss?
```

B15 gives a clean reason to ask that question. In the salvage board, `label_only` had the best final proxy loss and perfect final batch accuracy, yet the worst final held-out `D_func`. `single_teacher` had the best held-out function without being the proxy winner. That is exactly the failure mode a one-GPU lab cares about: not "does training loss go down?" but "which run deserves more electricity?"

So CTI can be different from ordinary overfitting theory if it adds all four of these:

1. early-budget forecasting, not post-hoc explanation;
2. held-out functional distortion, not proxy loss;
3. intervention ranking, not just curve fit;
4. decision value under a fixed compute budget.

Under that version, CTI is a resource-allocation law for small labs. It asks whether there is a stable geometry of improvement that lets the lab stop bad runs early and fund good runs before the answer is obvious.

### Attack

The hostile reviewer says:

```text
You discovered validation curves.
```

That attack is serious. "Training loss can improve while validation accuracy gets worse" is the first week of generalization theory. Bias-variance tradeoff, early stopping, cross-validation, regularization, train/test gap monitoring, learning-curve extrapolation, hyperparameter optimization, and overfit diagnostics already exist. The B15 board is especially vulnerable because `label_only` looking best on train and worst on held-out is the ordinary overfitting story, not a new physics story.

The phrase "proxy-function divergence" does not create novelty by itself. It is just another name for train/validation mismatch unless CTI forecasts something a standard validation workflow would not.

The hardest version of the objection:

```text
If CTI uses held-out D_func at checkpoints, then the obvious baseline is:
evaluate held-out D_func early and continue the intervention with the best value
or best trend. Why do I need CTI?
```

If CTI cannot beat that baseline, then it is not a law. It is a dashboard.

The "intervention taxonomy" also needs teeth. Calling one intervention a constant shift and another an exponent shift is not new unless that classification is predicted early and transfers. Otherwise it is curve-description after the fact.

### What Survived

CTI survives only as a decision forecaster, not as an explanation of overfitting.

The novelty wedge must be narrowed to:

```text
CTI predicts later held-out functional distortion and final intervention ranking
from early compute traces better than strong validation-curve and resource-allocation
baselines, with enough reliability to save real compute.
```

Required hardening:

- Add an explicit validation-curve baseline: choose by early `D_func` value and slope at step 100.
- Add an early-stopping/resource-allocation baseline: continue the current best early `D_func` run.
- Do not call proxy/function divergence itself novel. Treat it as the motivating failure mode.

### Next Sharpest Objection

Even if CTI is a forecaster rather than overfitting theory, the current law form may be wrong. B15's proxy power-law fits are weak, and CTI-1's early fit window may underidentify any curve with more than two parameters.

---

## I142: The Power-Law Form May Be Wrong

### Steelman

The defense is that B15 did not test the real law. It fit sparse noisy minibatch proxy losses, not held-out `D_func(C)`. B14 lacked intermediate held-out evaluations, so B15 could not fit the object CTI actually cares about.

CTI-1 fixes that by measuring `D_func` at every log checkpoint:

```text
10, 30, 100, 300, 1000, 3000
```

The power law in `CTI_PRECOMMIT_SPEC.md` can be treated as a first candidate, not a dogma. The spec also allows a broken power law and a null. Modular arithmetic may reveal a transition curve; MCQ LoRA may reveal monotone adaptation; shuffled-label controls should reveal no functional improvement. If the family is wrong, the null should win.

### Attack

The hostile reviewer says:

```text
Your curve family is underidentified and your fallback form is vague.
```

With only early checkpoints `10, 30, 100`, a three-parameter power law

```text
D_func(C) = D_inf + k * C^(-alpha)
```

is barely identified. It can interpolate early points while producing unstable extrapolations at `300, 1000, 3000`. If `D_func` is accuracy-derived, it is bounded, discrete, and often non-smooth. If the board includes grokking, the curve may be flat, then jump. If the board includes LoRA on MCQ slices, the curve may be noisy and nonmonotone.

B15's weak proxy `R2` values matter because they reveal the danger of forcing a grand law onto short, noisy traces. The answer should not be "add a broken power law" unless the broken-law form is itself precommitted tightly. The current broken form in the spec is also suspicious as written:

```text
D_inf + k * C^(-alpha_1) * sigmoid((C - tau) / beta)
```

For early `C << tau`, the sigmoid term is near zero, pushing the curve near `D_inf` before learning has happened. That is not the natural delayed-generalization shape. A better transition form would need a baseline plateau, a transition time, and a late asymptote, or a mixture/hazard interpretation. Otherwise "broken power law" becomes an escape hatch.

The deeper problem: if CTI changes the functional form after seeing CTI-0, it will look like p-hacking. If CTI lets the worker choose whichever curve wins after seeing held-out checkpoints, it is just automated post-hoc fitting.

### What Survived

The power law should be demoted from "the CTI law" to "one candidate in a precommitted forecaster race."

CTI-1 should precommit a small curve-selection rule before running:

| Candidate | Why it belongs |
|---|---|
| Last point / persistence | The minimum useful baseline. |
| Linear in `log C` | Simple smooth trend. |
| Saturating exponential or bounded logistic | Natural for bounded accuracy/distortion. |
| Power law with regularized `D_inf` | Continuity with scaling-law prior. |
| Plateau-transition-asymptote model | Necessary for grokking-style delayed generalization. |
| Isotonic or monotone-constrained smoother | Strong nonparametric baseline if monotonicity is justified. |

The winner should be chosen by a locked model-selection rule with penalties, not by researcher taste. If no candidate produces calibrated prediction intervals and baseline-beating forecasts, the correct token is `NO_PREDICTIVE_LAW`.

### Next Sharpest Objection

Even with a better forecaster race, CTI-1 may be too small. Two model/task families and six checkpoints can generate pretty tables without supporting `PASS_CTI_LAW_0`.

---

## I143: CTI-1 Is Too Small To Prove Anything

### Steelman

The strongest defense is that CTI-1 is a first gate, not the final moonshot. Its job is to collect the data B14 lacked and decide whether the direction deserves more compute.

That makes a small board rational:

- random tiny transformer on modular arithmetic gives a low-noise local-compute lab;
- SmolLM2-135M LoRA on MCQ preserves continuity with the B14 failure mode;
- six log-spaced checkpoints are enough to test whether early curves have any forecasting signal;
- the 48 GPU-hour cap prevents another long arc of ungrounded theory.

Under this defense, CTI-1 does not need to prove universality. It needs to answer:

```text
Is there any precommitted forecast signal beyond naive baselines?
```

### Attack

The hostile reviewer says:

```text
N=2 with curve fitting.
```

That attack lands if CTI-1 keeps only the two headline boards. Checkpoints are not independent samples; they are correlated observations from the same run. Interventions on the same task are not independent universes. A random modular arithmetic board and a pretrained MCQ LoRA board are so different that success on one and noise on the other cannot support a shared law.

The current spec says `PASS_CTI_LAW_0` requires success on `>=2/3 task families`, but the B15 CTI-1 design names two core task families. If only two boards actually run, the pass rule becomes ambiguous or impossible. If a third optional board is added after seeing early results, the reviewer will call it researcher degrees of freedom.

The "fit first 30%, predict remaining 70%" phrase also overstates the statistical separation. It is held out in time, not held out in data, model family, task family, or intervention. If the same eval split and same researcher decisions shape the whole board, the later points are not blind validation of the idea.

Minimum evidence for `PASS_CTI_LAW_0` should be stronger than "curve error looks lower on two runs":

- at least three predeclared boards;
- at least one held-out intervention or task not used to choose the curve-selection rule;
- paired comparison against baselines across the same forecast targets;
- prediction intervals, not just point estimates;
- a final intervention-ranking decision that is locked before full-budget values are opened.

### What Survived

CTI-1 survives as a kill/discovery gate, not as a public pass unless hardened.

Revised status vocabulary:

| Token | Meaning |
|---|---|
| `CTI_SIGNAL` | Single-seed or two-board run shows forecast signal worth replication. |
| `NO_PREDICTIVE_LAW` | Fails strong baselines on the predeclared boards. |
| `PROXY_ONLY_LAW` | Proxy forecasts work but `D_func` forecasts/rankings fail. |
| `PASS_CTI_LAW_0` | Multi-board, locked, baseline-beating functional forecast with stable ranking evidence. |

If the repo does not want a new token, then `PASS_CTI_LAW_0` must be withheld from the minimal two-board CTI-1. The honest verdict after a promising minimal run would be "signal, replicate."

### Next Sharpest Objection

The baseline board is still too weak. CTI should not get credit for beating last-point, log-linear, proxy-only, and random ranking if standard learning-curve and budget-allocation methods would do as well.

---

## I144: The Baseline Forecasters Are Too Weak

### Steelman

The current baseline set is useful as a floor:

| Baseline | What it protects against |
|---|---|
| Last point | CTI must beat "nothing changes." |
| Linear extrapolation | CTI must beat the simplest trend. |
| Per-task independent power law | CTI must beat unshared ordinary curve fitting. |
| Proxy-only forecast | CTI must show function, not proxy. |
| Random intervention ranking | CTI must beat chance. |

Those baselines are appropriate for a first spec because they make failure easy to detect and keep the experiment understandable.

### Attack

The hostile reviewer says:

```text
You chose baselines a weak learning-curve paper would beat.
```

For CTI's actual claim, the relevant competitor is not random ranking. The competitor is a practical lab manager with early validation curves and standard budget allocation:

```text
Run all interventions to step 100, keep the one with best held-out D_func or best
early trend, stop the rest.
```

If CTI cannot beat that, it has no electricity-saving value.

Stronger baselines CTI-1 should face:

| Baseline | Why it is hard |
|---|---|
| Early `D_func` rank | Choose the best intervention at step 100. |
| Early `D_func` slope | Choose the best value plus improvement trend over `10, 30, 100`. |
| Successive-halving policy | Allocate budget to early winners, the standard resource-allocation posture. |
| Regularized curve ensemble | Average power, log-linear, exponential, logistic, and plateau-transition candidates with penalties. |
| Bayesian/hierarchical learning-curve forecaster | Share structure across interventions while preserving uncertainty. |
| Gaussian-process or spline over `log C` | Strong nonparametric forecasting baseline. |
| Gap-aware forecaster | Use `D_proxy`, `D_gap`, margin, and calibration as features without CTI's law language. |
| Task-specific known-theory baseline | For grokking boards, use train/test gap, weight norm, or task progress measures. |

The baseline question should be split:

1. Forecast accuracy: who predicts later `D_func` best?
2. Decision value: who chooses the best final intervention with least compute?

CTI has to win both or the headline fails. A curve that predicts values slightly better but does not improve the continue/kill decision is not useful. A ranking that wins only because all interventions tie is not useful.

### What Survived

The baseline board must be upgraded before CTI-1 can grant a positive token.

Minimum hard baseline set:

```text
B0 persistence
B1 linear/log trend
B2 regularized candidate-curve ensemble
B3 early-D_func rank/slope policy
B4 proxy/gap/margin feature forecaster
B5 successive-halving/resource-allocation policy
B6 task-specific grokking baseline on modular boards
B7 random ranking sanity control
```

CTI's own forecaster must be specified as a distinct algorithm. If it is just "the best of the same candidate curves," then CTI has no separate content.

### Next Sharpest Objection

The modular arithmetic board may not test a general compute-distortion law. It may only rediscover known grokking and phase-transition dynamics.

---

## I145: Modular Arithmetic May Be A Grokking Confound

### Steelman

Modular arithmetic is attractive because it gives CTI a clean low-noise lab. Exact-answer `D_func` is unambiguous. The task is cheap. Proxy/function divergence is expected and measurable. If CTI cannot handle delayed generalization on a grokking-style task, it probably cannot handle messy language adaptation either.

The strongest defense is:

```text
Grokking is not a confound; it is a stress test. A real compute-distortion
forecaster should predict phase-like transitions, not avoid them.
```

Under that view, modular arithmetic is a useful hard case for broken laws and intervention classification.

### Attack

The hostile reviewer says:

```text
You picked a task where delayed generalization is already famous, then claimed
to discover delayed generalization.
```

That attack is valid unless CTI beats grokking-specific baselines. Modular addition has a literature and known diagnostics. If the CTI result is "training accuracy improves before test accuracy," it is not new. If the result is "weight decay changes the transition," it is not enough. If the result is "a broken curve fits a phase transition after it happens," it is post-hoc description.

There is also a practical failure mode. With checkpoints only through `3000`, the grokking transition may not occur. Then every forecaster predicts high distortion and looks decent. Or the transition may occur after step 100 but before step 300, making early prediction almost impossible from `10, 30, 100` unless the model includes real order parameters. Or the transition may be tuned accidentally by task size and regularization until it lands conveniently in the forecast window.

The shuffled-label control is also too easy. A control that never generalizes can make CTI's ranking look good without testing the law. "Real beats shuffled" is not an intervention taxonomy; it is sanity checking.

### What Survived

Modular arithmetic should remain in CTI-1, but it cannot carry the main claim by itself.

Hardening rules:

- Treat modular arithmetic as the phase-transition board, not the universal-law board.
- Add or preserve at least one non-grokking monotone task board where ordinary learning-curve extrapolation is expected to work.
- On modular arithmetic, require CTI to beat a grokking-specific baseline using known order-parameter-like signals, not just last-point or random.
- Predeclare task sizes, data fractions, weight decay, and max steps so the transition is not tuned after inspection.
- Do not count shuffled-label separation as positive CTI evidence except for validity.

### Next Sharpest Objection

Even if the task mix is improved, a single-seed one-GPU run cannot support exponent/ranking claims. Checkpoints are not replication.

---

## I146: Single GPU Does Not Excuse Single-Seed Claims

### Steelman

The single RTX 5090 constraint is not a bug; it is part of the democratization thesis. CTI should be useful precisely because the lab cannot run frontier-scale sweeps. The first gate has to fit inside roughly 48 GPU-hours, or it repeats the Eklavya mistake of spending too long before knowing whether the direction has signal.

There is a reasonable staged position:

```text
Use single-seed CTI-1 to kill or detect signal. Require replication only if the
signal appears.
```

For cheap modular tasks, multiple seeds are feasible. For SmolLM2 LoRA, maybe the first run can be deterministic and use bootstrap over eval examples while replication waits for a follow-up.

### Attack

The hostile reviewer says:

```text
You cannot classify exponents, intervention rankings, or compute laws from one
random seed.
```

They are right. Random initialization, data order, LoRA adapter initialization, batch composition, MCQ slice composition, and optimizer noise can all flip early trends and final rankings. Checkpoints give dense-looking plots, but they are repeated measures from one trajectory. They do not estimate run-to-run variance.

B14 already shows how dangerous tiny slices are. With `n=48` per benchmark, a movement of a few examples can look like several percentage points. A final ranking among interventions can be determined by noise at exactly the scale CTI wants to forecast.

Single seed can kill a law: if nothing forecasts anything even in the easiest setting, stop. But single seed cannot pass a law. A pass requires stability under rerun.

### What Survived

The one-GPU constraint survives, but the evidence ladder must separate signal from pass.

Minimum seed policy:

| Board | Minimum for `CTI_SIGNAL` | Minimum for `PASS_CTI_LAW_0` |
|---|---:|---:|
| Cheap modular/algorithmic board | 3 seeds | 5 seeds or 3 seeds plus a held-out task-size transfer |
| SmolLM2-135M LoRA MCQ | 1 seed plus bootstrap CIs | 3 seeds or 2 seeds plus blind larger eval split |
| Any final intervention ranking claim | 2 independent confirmations | 3 independent confirmations or one locked replication batch |

The report should use prediction intervals and paired tests against baselines. If the interval overlaps the baseline, no positive token.

### Next Sharpest Objection

Measuring `D_func` at every checkpoint fixes B14's missing-data flaw, but it creates a new danger: the held-out functional metric can become the new proxy and leak into design decisions.

---

## I147: D_func At Every Checkpoint Can Become Validation Leakage

### Steelman

The current CTI-1 design correctly fixes B14's fatal flaw. B14 only had final held-out function, so CTI-0 could diagnose proxy/function divergence but could not forecast `D_func(C)`. Evaluating `D_func` at every log checkpoint gives the law the right target.

This is the strongest version of CTI after B21's earlier attacks:

```text
Use early held-out functional measurements, not proxy loss, to forecast later
held-out functional measurements and final intervention ranking under a locked
baseline board.
```

That is much better than B14 salvage.

### Attack

The hostile reviewer says:

```text
You made validation accuracy the new training signal.
```

If the same held-out split is evaluated at every checkpoint, used to fit the law, used to rank interventions, used to update the spec after a batch, and used to announce the result, then `D_func` is no longer a blind functional measure. It is a validation proxy. That is still better than minibatch training loss, but it is not final evidence.

"Fit on first 30%, predict remaining 70%" only hides later time points. It does not hide the data distribution, benchmark slices, task choices, intervention menu, or curve-selection logic. In a multi-loop repo, leakage can happen at the research-process level even if the model weights never see the held-out labels.

The correct structure is three-way:

| Split | Used for |
|---|---|
| Train/proxy | Optimization and `D_proxy`. |
| Forecast validation | Checkpoint `D_func` used by CTI forecasters and early ranking. |
| Blind final test | Opened only after forecasts, rankings, and continue/kill decisions are locked. |

For MCQ, the blind final test should be larger than the tiny B14 slices if possible. For modular arithmetic, it should be a separate held-out set or held-out task variant, not the same grid inspected throughout development.

### What Survived

`D_func` at every checkpoint remains mandatory, but CTI-1 needs split hygiene.

Required change:

```text
Every checkpoint logs D_func_forecast on a predeclared forecast-validation split.
The final verdict uses D_func_blind on a separate split opened only after the
forecast file and intervention-ranking file are written.
```

If blind final testing is too expensive for CTI-1, then the verdict cannot be `PASS_CTI_LAW_0`. It can only be `CTI_SIGNAL` or `NO_PREDICTIVE_LAW`.

### Next Sharpest Objection

After all hardening, CTI is still narrative-fragile. A truthful current headline must say that the lab has designed a forecasting test, not that a laptop has already predicted which training ideas are worth electricity.

---

## B21 Verdict

CTI should continue, but the current CTI-1 design is not yet hostile-reviewer-proof enough to award `PASS_CTI_LAW_0`.

The correct near-term posture:

```text
Run CTI-1 as a locked signal-or-kill gate. If it shows signal, replicate and
harden before any public CTI claim. If it fails strong baselines, kill CTI or
downgrade it to a narrow learning-curve tool.
```

The biggest required spec changes before a positive verdict:

1. Add strong validation-curve, learning-curve, and resource-allocation baselines.
2. Precommit a curve-selection rule rather than privileging power laws.
3. Require at least three predeclared boards or withhold `PASS_CTI_LAW_0`.
4. Separate forecast-validation `D_func` from blind-final `D_func`.
5. Separate single-seed `CTI_SIGNAL` from replicated `PASS_CTI_LAW_0`.

---

## Ranking Of CTI-1's Biggest Risks

| Rank | Risk | Why it can kill CTI-1 | Required mitigation |
|---:|---|---|---|
| 1 | Weak baselines | Beating last-point/random does not prove decision value. | Add early-`D_func`, curve-ensemble, gap-feature, and successive-halving baselines. |
| 2 | Underpowered evidence | Two boards, six checkpoints, and one seed look like `N=2` curve fitting. | Predeclare at least three boards and separate `CTI_SIGNAL` from `PASS_CTI_LAW_0`. |
| 3 | Validation leakage | Repeated held-out `D_func` can become the new proxy. | Use forecast-validation and blind-final splits. |
| 4 | Wrong or underidentified law form | Three early points cannot support flexible power/broken-law claims. | Precommit a regularized forecaster race with penalties and prediction intervals. |
| 5 | Known-overfitting dismissal | Proxy/function divergence alone is ordinary generalization failure. | Frame CTI as intervention forecast and resource allocation, not overfitting discovery. |
| 6 | Grokking confound | Modular arithmetic may only rediscover known delayed generalization. | Treat modular as phase stress test and beat grokking-specific baselines. |
| 7 | Single-seed instability | Rankings and exponents can flip across seeds. | Require seed replication for any pass token. |
| 8 | MCQ eval noise | B14-style tiny slices make few-example changes look large. | Increase eval size or use bootstrap CIs and blind final confirmation. |
| 9 | Pretraining compute ambiguity | SmolLM2 local LoRA inherits massive external pretraining. | State the result as conditional adaptation forecasting; keep at least one local-only board. |
| 10 | Narrative overclaim | "Laptop predicted worthwhile electricity" is false before a locked forecast succeeds. | Forbid public CTI language until replicated baseline-beating functional forecasts exist. |

---

## NARRATIVE ATTACK

### 1. Strongest "that's obvious" dismissal

```text
Training loss and held-out performance can diverge. More compute often follows
learning curves. Use validation curves, early stopping, and hyperparameter
search. You renamed standard generalization practice "compute thermodynamics."
```

This dismissal kills any CTI story that rests on B15 alone. The salvage result is a clue, not a breakthrough. Proxy/function anti-alignment in five B14 conditions is interesting because it motivates a forecasting test, not because it teaches the field that overfitting exists.

The honest response:

```text
Correct. CTI is not the discovery of proxy/function divergence. CTI only becomes
non-obvious if it forecasts later held-out functional improvement and final
intervention ranking earlier and better than standard validation-curve and
resource-allocation baselines.
```

### 2. Strongest "that's trivial" dismissal

```text
You ran learning-curve extrapolation on toy tasks and a pretrained small model,
then curve-fit six checkpoints with grand physics words. The result is either a
toy grokking plot or ordinary LoRA validation monitoring.
```

This dismissal kills CTI if the board stays small, single-seed, and weak-baseline. It also kills CTI if "thermodynamics" appears before a genuine predictive law.

The honest response:

```text
The first CTI result is trivial unless it is locked, blind-tested, replicated,
and decision-useful. The word "thermodynamics" is unearned until the forecast
changes what compute the lab spends and the blind final result confirms the
forecast.
```

### 3. What the result would need to be for the narrative to be unkillable

The unkillable version is not:

```text
We fitted a power law.
```

It is:

```text
A locked one-GPU forecaster saw only the first 30% of several training runs,
predicted the later blind functional distortion curves with calibrated intervals,
chose which intervention deserved the remaining compute, beat strong learning-curve
and successive-halving baselines, and the blind final test confirmed the choice
across multiple seeds and task families.
```

Minimum unkillable ingredients:

1. At least three predeclared boards, including one local-only model board and one pretrained-adaptation board.
2. `D_func_forecast` at checkpoints plus separate `D_func_blind` for the final verdict.
3. Strong baselines: early validation rank/slope, regularized curve ensemble, gap-feature forecaster, successive halving, and grokking-specific baseline where relevant.
4. Multi-seed confirmation for the final ranking.
5. A real compute-saving decision: CTI stops or deprioritizes an intervention that would have wasted substantial budget, and selects the final winner before the full run is visible.
6. No public "universal law" language. The first allowed public line is narrower:

```text
A precommitted one-GPU forecaster predicted which small AI training runs improved
held-out function before they finished, and it beat standard early-stopping and
learning-curve baselines.
```

Current gossip-magazine version, if told today:

```text
After nine honest failures, a one-GPU lab found a clue in the wreckage: the run
that learned the training proxy best generalized worst. Now they are testing
whether early functional curves can predict which future runs deserve the power
bill. They have not proved it yet.
```
