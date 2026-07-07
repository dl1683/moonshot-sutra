# Q-Loop B22: CTI After PROXY_ONLY_LAW - Kill or Pivot?

**Date:** 2026-07-07
**Role:** Question Loop worker
**Iterations:** I148-I154
**Status:** analysis-only adversarial review; no model, dataset, GPU, or web runs.

---

## Grounding

Read in the requested order:

1. `research/VISION.md`
2. `research/CTI_PRECOMMIT_SPEC.md`
3. `research/work_loop_batch16.md`
4. `research/question_loop_batch21.md`
5. `research/work_loop_batch15.md`

Binding facts from CTI-1 Board 1:

- The run is valid as a measurement. No invalidity token is available from the current evidence: no pretrained model was loaded, `D_func` was measured at every checkpoint, and predictions were locked before steps 300, 1000, and 3000 were opened.
- Verdict token: `PROXY_ONLY_LAW`.
- CTI power-law MAE: `0.226068`, worse than `b0_last_point` at `0.215232` and worse than `b3_proxy_only` at `0.194245`.
- Every non-random forecaster predicted step-3000 winner `quarter_data`; actual winner was `label_only`.
- `label_only` had a grokking-like curve: `D_func` stayed near `0.99` through step 100, dropped to `0.4427` at step 300, regressed to `0.6366` at step 1000, then collapsed to `0.0070` at step 3000.
- `quarter_data` was the early apparent winner by step-100 `D_func` (`0.9152` vs `0.9821` for `label_only`) but never generalized: held-out accuracy fell to `4.76%` by step 3000 while condition-train accuracy stayed at `100%`.
- `shuffled_labels` behaved as a clean negative control: near-chance held-out accuracy throughout while proxy loss improved.

Current strongest position to attack:

```text
The vanilla CTI power law failed on the first clean board. Either CTI is dead,
or the result says compute-distortion curves are regime/phase objects rather
than smooth scalar power laws.
```

---

## I148: Kill CTI Because Power Laws Do Not Predict Phase Transitions

### Steelman

The hardest anti-CTI position is simple:

```text
The first clean CTI board falsified the law.
```

This was not a messy salvage artifact. Board 1 measured exact held-out modular addition accuracy over the full held-out split at every checkpoint. The forecast was locked before the held-out future points were visible. The negative control worked. The actual failure was not bookkeeping; it was the law.

The law form predicted gradual improvement. The actual winning intervention hid almost all of its functional improvement until after the fit window, then produced a double transition: improvement at step 300, regression at step 1000, near-solution at step 3000. A smooth power law fit on steps 10, 30, and 100 predicted `label_only` step-3000 `D_func = 0.9695`; actual was `0.0070`. That is not a small calibration error. It is a category error.

If the most interesting phenomenon is sudden functional reorganization, and CTI cannot predict sudden functional reorganization, the hostile reviewer says CTI is dead. It is not "compute thermodynamics"; it is an underfit curve.

### Attack

This kills the privileged vanilla power law, not automatically the whole research program.

The original spec did not only ask "does one power law fit?" It asked whether compute spent relates to functional distortion and whether interventions change that relation in classifiable ways. Board 1 found classifiable ways, but they are not the old classes:

| Intervention | Observed shape | Meaning |
|---|---|---|
| `label_only` | delayed transition, regression, recovery | latent generalization phase |
| `quarter_data` | early proxy win, held-out decay | memorization trap / support failure |
| `shuffled_labels` | proxy improvement with chance function | pure proxy-only memorization |

That is not a positive CTI result, but it is not nothing. It says compute is not sufficient as a scalar state variable. The missing variable is regime.

The sharper conclusion:

```text
D_func(C) = D_inf + k*C^-alpha is dead as CTI's privileged law form.
CTI survives only if it becomes a regime-aware forecasting program.
```

### What Survived

Survived:

- The measurement program survived.
- The proxy/function distinction survived.
- The need for locked early forecasts survived.
- The old smooth power-law law form did not survive.

CTI can continue only if it demotes scalar MAE-on-power-law to one candidate and adds a precommitted regime forecast:

```text
Given early traces, predict both the curve class and the final functional outcome.
```

### Next Sharpest Objection

Maybe the real result is not numerical forecasting at all. Maybe CTI should abandon scalar law fitting and become a classifier of compute-distortion curve shapes.

---

## I149: The Grokking Result Is The Result

### Steelman

The best pro-pivot argument is that Board 1 discovered the object CTI should have been studying:

```text
Training interventions do not merely shift a shared curve. They induce different
curve shapes.
```

For a one-GPU lab, shape class may matter more than point prediction. The practical decision is not "what is `D_func` at exactly step 1000?" The practical decision is:

- Is this run in a monotone improvement regime?
- Is this run in a delayed generalization regime?
- Is this run only memorizing the proxy?
- Is this run data-support limited, so more compute will worsen held-out function?
- Should the lab continue, stop, or collect different data?

Board 1 makes this concrete. `quarter_data` looked best early, reached `100%` condition-train accuracy, and then got worse on held-out. `label_only` looked weak early, then became the real winner. `shuffled_labels` showed that proxy improvement alone is worthless. A shape taxonomy would be more decision-useful than pretending all three traces are the same power law with different exponents.

### Attack

Post-hoc shape naming is cheap.

The locked forecasters did not predict the shape that mattered. They all picked `quarter_data`, because at step 100 it had the best early held-out `D_func` and the steepest apparent improvement. If the shape classifier only says "that was grokking" after step 300 or step 3000, it does not save electricity. It is just a caption on a plot.

The shape-pivot only becomes CTI-grade if the class is forecast early enough to change allocation. The minimum question is:

```text
Could a pre-step-300 classifier have warned that quarter_data was a memorization
trap and that label_only had latent transition potential?
```

There is a clue but not a proof. At step 100, `quarter_data` had `D_gap = 0.8344`: train accuracy `91.92%`, held-out accuracy `8.48%`. That is a huge trap signature. `label_only` had train accuracy `10.18%`, held-out accuracy `1.79%`, and only `D_gap = 0.0840`: not good, but not yet exhausted. A gap-aware classifier might have distrusted `quarter_data` despite its better early held-out score.

But that still would not separate `label_only` from `shuffled_labels` cleanly, because both were weak early. More order parameters are needed.

### What Survived

Curve-shape classification survives as the correct pivot target, but only under a hard rule:

```text
No curve class counts unless it is predicted from early-state features before
the decisive transition is visible.
```

Board 2 must log more than `C` and `D_func`:

- `D_gap`
- train accuracy and held-out accuracy separately
- proxy slope and proxy curvature
- margin statistics
- calibration
- effective data support / examples per class or residue
- simple weight/optimizer state diagnostics where cheap

### Next Sharpest Objection

Maybe modular arithmetic is the wrong test. Grokking tasks are pathological; CTI's real domain may be monotone adaptation, not phase transitions.

---

## I150: Board 1 Is The Wrong Test

### Steelman

The strongest defense of CTI is that Board 1 was a stress test, not the core market.

The democratization moonshot is not "forecast every grokking transition in modular arithmetic." It is "help a small lab choose which real training interventions deserve scarce compute." The real use case may be SmolLM2 LoRA on MCQ, byte-model adaptation, KD, routing, or other monotone-ish learning curves. In those settings, early functional checkpoints may forecast later functional checkpoints well enough to beat baselines.

On this defense, Board 1 should be discounted:

```text
Grokking is non-generic. Do not kill CTI until Board 2 tests monotone adaptation.
```

### Attack

CTI cannot choose modular arithmetic in the precommit and then call it irrelevant after failure.

The spec explicitly included modular arithmetic and grokking-style tasks. B21 already warned that modular arithmetic might be a phase-transition board. Board 1 was not a surprise category; it was a known hard case. If CTI's claimed law cannot handle it, CTI must narrow its domain.

Also, avoiding grokking avoids the exact failure mode CTI was created to address: proxy/function divergence. The small lab does not merely need a law for easy monotone curves. It needs to know when early proxy and early validation signals are traps.

The valid distinction is not "Board 1 does not count." The valid distinction is:

```text
Board 1 falsifies smooth-curve CTI on phase-transition tasks. Board 2 can still
test whether a separate monotone-adaptation CTI exists.
```

### What Survived

Board 1 remains binding evidence, but it should be classified as a phase-stress failure, not as complete death of every CTI variant.

Before Board 2, the spec must split domains:

| Domain | Expected behavior | Required forecaster |
|---|---|---|
| Monotone adaptation | gradual improvement or saturation | smooth curve / ensemble baseline |
| Delayed generalization | plateau then transition | transition-hazard or order-parameter model |
| Memorization trap | proxy improves, function stalls or worsens | gap/support classifier |
| Impossible/noisy | no forecastable signal from early state | abstention with wide intervals |

### Next Sharpest Objection

Maybe the result is not "wrong law form" but "no early information exists." If all forecasters fail, perhaps the task is fundamentally unpredictable from the first 100 steps.

---

## I151: No Forecaster Won, So The Board Is Uninformative

### Steelman

This objection is brutal because every forecaster failed the decision.

The best MAE was `b3_proxy_only`, but it still picked `quarter_data` as the step-3000 winner. CTI, last point, log-linear, independent power law, and proxy-only all missed the actual best intervention. If no method could predict the winner from the first 100 steps, then Board 1 may not reveal a better law. It may reveal non-identifiability.

The hostile reviewer says:

```text
There is no law to find from these observations. The transition time is hidden.
You cannot forecast a phase change from three early scalar points.
```

### Attack

The board is uninformative only for forecasters that use the wrong information.

The locked forecasters were mostly scalar curve extrapolators. They used early `D_func`, log-compute trend, or proxy trajectory. They did not use the most obvious trap signal: the relationship between training accuracy and held-out accuracy.

At step 100:

| Intervention | Train acc | Held-out acc | `D_gap` | Interpretation |
|---|---:|---:|---:|---|
| `label_only` | `10.18%` | `1.79%` | `0.0840` | weak but not exhausted |
| `quarter_data` | `91.92%` | `8.48%` | `0.8344` | likely memorization trap |
| `shuffled_labels` | `10.18%` | `1.00%` | `0.0918` | weak with wrong labels |

A crude gap-aware rule would not necessarily predict `label_only` as final winner, but it should have refused to trust `quarter_data` as the winner. That matters. Board 1 does not prove unpredictability. It proves `D_func(C)` alone is under-specified.

The right verdict is:

```text
Early scalar curves were non-identifying. Early state vectors may still contain
forecastable phase-risk signals.
```

### What Survived

Board 1 remains informative because it identifies a missing feature family.

CTI-2 must include an abstention/uncertainty target:

- If the early state is non-identifying, the forecaster must say "phase-risk / no confident ranking."
- A wrong confident ranking should be penalized more than a wide interval.
- Decision value should reward not wasting compute on a false early winner.

Surviving research question:

```text
Can early gap/support/order-parameter features predict when an apparent early
winner is a memorization trap?
```

### Next Sharpest Objection

Maybe `quarter_data`, not `label_only`, is the real finding. The democratization lesson may be data support beats compute quantity.

---

## I152: Quarter_Data Is The Real Finding

### Steelman

The most practical result in Board 1 may be:

```text
More compute can make function worse when data support is insufficient.
```

`quarter_data` hit `100%` condition-train accuracy by step 300, but held-out accuracy was only `11.69%`. By step 3000, it was still perfect on the condition train set and had fallen to `4.76%` held-out accuracy. Compute improved proxy and worsened function.

That is the democratization story in plain language:

```text
Do not buy more electricity for a bad data regime.
```

For a small lab, discovering that a quarter-data run is a compute sink may be more useful than predicting the exact grokking step of the full-data run.

### Attack

Do not overclaim "data quality beats compute."

This intervention was not a demonstrated high-quality data subset. It was reduced support on modular addition. The lesson is narrower:

```text
Insufficient coverage of the task's structure can turn compute into memorization.
```

That is useful, but it is also close to known sample-complexity intuition. And the full-data `label_only` run eventually solved the task, so compute was not the enemy. Compute was useful when the data support was adequate and destructive when it was not.

The actual law cannot be "less data is better" or "more compute is bad." It must be conditional:

```text
The sign of compute depends on whether the intervention has enough structural
support to convert optimization into generalization.
```

### What Survived

`quarter_data` survives as the cleanest Board 1 warning for the moonshot:

```text
Proxy improvement plus high train accuracy plus poor held-out function is an
electricity sink.
```

Before Board 2, CTI must make data support an explicit variable. For modular arithmetic, this means coverage over residues/pairs. For MCQ or language adaptation, this means benchmark/task coverage, example diversity, label quality, teacher coverage, and whether the validation split is structurally close to the train/adaptation set.

### Next Sharpest Objection

Maybe CTI only needs a new law form: plateau-transition-asymptote or grokking-aware transition models.

---

## I153: CTI Needs Phase-Transition-Aware Law Forms

### Steelman

The constructive pivot is straightforward:

```text
Replace vanilla power laws with phase-aware curve families.
```

Candidate families:

| Candidate | Use |
|---|---|
| Plateau-transition-asymptote | delayed generalization / grokking |
| Broken power law | different pre/post transition slopes |
| Hazard model over transition time | probability of transition by compute budget |
| Mixture of regimes | monotone, delayed, trap, no-signal |
| Bounded logistic over log compute | saturating functional improvement |
| Abstaining ensemble | wide interval when early state is non-identifying |

A better `label_only` model would not extrapolate the step-100 plateau forever. It would say: this run may be in a delayed transition regime; final distortion has a wide interval; do not confidently rank the early `quarter_data` winner above it.

### Attack

Changing the law form after failure is p-hacking unless the new form is locked before the next board.

There are two technical dangers:

1. A plateau-transition model cannot estimate transition time from only steps 10, 30, and 100 unless it has additional state variables or strong priors.
2. Board 1 was not a single clean transition. `label_only` improved at step 300, regressed at step 1000, then solved by step 3000. A one-sigmoid law cannot capture that path.

This means the true failure may be deeper than "wrong curve." It may be:

```text
Compute alone is not the state variable.
```

The law needs state:

```text
D_func future = f(C, early_state, intervention, data_support, optimization_state)
```

Without that, adding sigmoids is just a prettier hindsight fit.

### What Survived

Phase-aware law forms survive only as part of a richer precommitted forecaster.

Minimum CTI-2 target:

```text
Predict regime class, transition risk, final D_func interval, and intervention
ranking from early compute plus early state features.
```

The old scalar target can remain as a scoreboard, but it cannot be the whole claim.

### Next Sharpest Objection

The narrative may now be too damaged. A failed first law may make CTI look like another grand theory that shrinks after contact with data.

---

## I154: The Narrative May Be Dead

### Steelman

The gossip-magazine story is currently ugly:

```text
A laptop tried to predict which AI training idea was worth the electricity and
failed because the winner changed phase after the forecast window.
```

That sounds depressing, not paradigm-shifting. The power-law claim failed. The forecast picked the wrong intervention. The best baseline was proxy-only. A hostile reviewer can say:

```text
You ran a known grokking toy problem, failed to predict grokking, and now want
to rename the failure as a phase theory.
```

If CTI keeps the same branding and simply adds flexible forms, the narrative dies.

### Attack

The honest narrative is not dead if the failure is treated as a kill of the old claim.

The compelling version is:

```text
The first law failed cleanly. That failure exposed the real object: small-lab
training decisions are governed by curve regimes, not by a single smooth
compute law. The next experiment asks whether those regimes have early warning
signatures.
```

That is not a victory lap. It is a sharper research program.

But this only works if CTI accepts the downgrade:

- no "universal law" language;
- no "thermodynamics" language in public;
- no positive token from Board 1;
- no Board 2 under the old scalar-power-law contract;
- no post-hoc curve rescue counted as evidence.

### What Survived

The narrative survives as a pivot story, not as a CTI success story.

The first public-safe sentence would be:

```text
Our first locked compute-distortion forecast failed on a grokking board; the
failure showed that early scalar curves can miss phase transitions, so the next
test is whether phase-risk signatures can be detected early enough to save
compute.
```

### Next Sharpest Objection

If Board 2 also fails after adding regime features, then CTI should be killed as a law program and retained only as a diagnostic dashboard.

---

## Recommendation

**Verdict: PIVOT-CONTINUE.**

Kill:

```text
CTI as a privileged smooth power law D_func(C) = D_inf + k*C^-alpha.
```

Continue:

```text
CTI as a precommitted one-GPU functional forecasting program, but only after
pivoting to regime/phase-aware prediction.
```

Pivot direction:

```text
Regime-aware compute-distortion forecasting:
early state -> curve class, transition risk, final D_func interval, and
intervention-ranking decision.
```

This is not a cosmetic change. It changes the object of prediction from one scalar curve to a conditional decision law.

---

## What Must Change Before Board 2

Board 2 should not run under the old CTI contract. Minimum required changes:

1. **Demote vanilla power law.** Keep it as a baseline, not as CTI's privileged form.
2. **Add regime labels.** Predefine at least: monotone improvement, delayed transition, memorization trap, proxy-only/no-generalization, noisy/unforecastable.
3. **Predict intervals and abstention.** A forecaster may output "phase risk / no confident ranking"; overconfident wrong rankings should be punished.
4. **Add state features.** Log `D_gap`, train accuracy, held-out accuracy, proxy slope, proxy curvature, margin, calibration, effective data support, and cheap optimizer/weight diagnostics where available.
5. **Add hard baselines.** Include early-`D_func` rank/slope, gap-aware heuristic, gap/support feature forecaster, successive halving, regularized curve ensemble, and task-specific grokking baseline for algorithmic boards.
6. **Separate Board 2 domain.** If Board 2 is SmolLM2 LoRA/MCQ, treat it as a monotone-adaptation test, not as proof that Board 1 did not matter.
7. **Keep blind hygiene.** Forecasts, rankings, curve-class predictions, and abstentions must be locked before later checkpoints and blind-final measurements are opened.
8. **Require decision value.** The score is not only MAE. The forecaster must improve continue/kill allocation versus strong baselines.

Positive token discipline:

```text
Board 1 cannot support PASS_CTI_LAW_0.
Board 2 can at most support CTI_SIGNAL unless it is locked, blind, baseline-hard,
and decision-useful.
```

Kill rule after pivot:

```text
If regime-aware CTI cannot beat gap-aware/resource-allocation baselines on Board 2
and one replication or held-out board, kill CTI as a law program. Keep only the
diagnostic dashboard.
```

---

## NARRATIVE ATTACK

### The hostile headline

```text
They said a laptop could predict which AI training run was worth the electricity.
The first clean test picked the wrong run.
```

That headline is fair. Board 1 did not merely underperform by MAE. It missed the intervention ranking. The power-law forecaster saw early `quarter_data` improvement and chose it. The actual winner was `label_only`, because it underwent a late functional transition.

### The "that's obvious" attack

```text
This is grokking. The field already knows modular arithmetic can sit on a
plateau and then suddenly generalize. You did not predict the transition; you
rediscovered it after it happened.
```

The only honest answer is:

```text
Correct. Board 1 is not a positive discovery of grokking. It is a falsification
of scalar CTI and a forcing function for a stricter phase-aware test.
```

### The "that's depressing" attack

```text
If the practical answer is "phase transitions are hard," then a small lab has
less predictive power than before, not more.
```

The response:

```text
Only if we stop at the failed power law. The useful question is now whether
phase-risk has cheap early warning signs. Board 1 suggests one: huge train/held-out
gap under reduced data support. That does not solve the problem, but it gives
Board 2 a sharper target.
```

### The honest gossip-magazine story now

```text
A one-GPU lab tried to forecast which tiny training run deserved more compute.
The first locked forecast failed: the early winner was a memorization trap, and
the apparent loser later snapped into generalization. That failure killed the
simple power-law story. The next test is harsher: can the lab detect phase-risk
early enough to avoid wasting electricity?
```

### The narrative can still win if

The story becomes compelling only if the next board does all of this:

- predicts "memorization trap" before the trap is obvious;
- refuses overconfident rankings when phase risk is high;
- beats early-validation and successive-halving baselines;
- saves actual compute by deprioritizing a bad run or preserving a latent winner;
- reports the result without pretending Board 1 was a success.

### The narrative dies if

The narrative dies if the project says:

```text
The power law failed, so we added a more flexible curve and now CTI is alive.
```

That is not a moonshot. That is curve-fitting after loss.

Final narrative verdict:

```text
CTI is not dead, but the old CTI story is. The live story is phase-aware
small-lab forecasting or nothing.
```
