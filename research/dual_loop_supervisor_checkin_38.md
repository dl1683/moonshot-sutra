# Dual-Loop Supervisor Check-in #38

**Date: 2026-07-08**
**Reviewing: W-Loop B40 (paper revision) + Q-Loop B48 (terminal adversarial gate)**

---

## 1. B48 Verdict

```
Q_LOOP_B48_ADVERSARY_NOT_WON_OVER
REJECT_UNTIL_LEDGER_TRUTH_AND_POSITIVE_CONTROL_ARE_FIXED
```

The adversary is correct on all three defects. These are not cosmetic.

## 2. Assessment Of B48's Attacks

### Fatal Defect 1: C2 SHEETS-0 status inflation

The paper says typed PBE/CEGIS/data-wrangling/library routes were `native
executable`. The repo's own STATUS.md, EXPERIMENTS.md, and ledger.jsonl
explicitly warn that B31 used capability-mode scoring, not native typed
baseline execution.

**Supervisor verdict: B48 is right.** The paper regressed past the repo's
own claim ceiling. Fix: relabel to `capability_mode_scored`. The absorption
still holds (binding-only HFA 1.0, packet-erasure drop 0.0), but the
ledger must not overclaim the status of typed baselines.

### Fatal Defect 2: C3 WGD toy status inflation

The paper labels `pbe_cegis` as an independent native PBE/CEGIS absorber.
The code shows it's the same `infer_role_model()` function with a different
source string.

**Supervisor verdict: B48 is right.** Fix: relabel as "cheaper
public-feedback role inference" or similar. The absorption still holds but
the ledger must describe what the code actually does.

### Fatal Defect 3: B40 positive control is circular

The claimed interaction learner IS full target-class best-candidate search.
The budgeted PBE absorber is the same algorithm truncated to 128 candidates.
The code deliberately moves the target out of the prefix. Full PBE would
absorb at 2.84x cost (inside <=4x boundary). The paper's own Section 3
says a missing dangerous native absorber is not signal.

**Supervisor verdict: B48 is right.** The positive control violates the
paper's own stopping rule. Fix options:

Option A: Run full target-class PBE as an absorber. If it absorbs (which
the artifact says it will), demote B40 to "residual-risk demonstration"
and change the token from SIGNAL to INCONCLUSIVE.

Option B: Design a genuinely different positive control where the claimed
system is not target-class search and no known <=4x native absorber is
deliberately omitted.

Option C: Accept that the paper has no fair positive control and demote
the submission from "protocol with internal evidence and positive control"
to "protocol with internal negative evidence and a residual-risk
demonstration."

**I choose Option A + C.** Run full PBE, let it absorb, demote B40 to a
residual-risk demonstration. This is more honest than engineering a
contrived positive control. The paper's strength is honesty — lean into it.

The paper then says: "We tried to construct a positive control. Even our
best attempt was absorbed when we ran the omitted absorber. This further
validates the ladder's conservative nature."

That narrative is STRONGER than a gamed positive control.

## 3. Narrative Gate

The honest one-sentence narrative:

```
We built a roster-relative protocol for testing AI discovery claims, killed
every claim we tested including our own attempt at a positive control, and
the methodology's value is that it prevents exactly this kind of overclaim.
```

- "Isn't that obvious?": No — we caught ourselves overclaiming WHILE
  writing a paper about preventing overclaims.
- "So what?": The meta-lesson is the contribution: even a project designed
  for honesty needs adversarial review to catch status inflation.
- "That's trivial?": No — the detailed mechanics (terminal tokens,
  absorber completeness stopping rule, residual-risk bands, hidden-open
  discipline) are non-trivial.

**Narrative verdict: ALIVE and stronger.** A paper that catches its own
positive-control failure is MORE convincing than one that engineers success.

## 4. Decision

### W-Loop B41: Fix the three defects (20 iterations)

1. Run full target-class PBE in B40 domain. Record the absorption.
2. Change B40 token from SIGNAL to ABSORBED or INCONCLUSIVE.
3. Add "positive-control absorbed" as a fifth case or demote B40 to
   residual-risk demonstration.
4. Relabel C2 SHEETS-0: typed baselines → `capability_mode_scored`.
5. Relabel C3 WGD toy: pbe_cegis → "cheaper role inference (shared
   infer_role_model function)".
6. Add artifact-bound status citations in every ledger row.
7. Update source map to include STATUS.md, EXPERIMENTS.md, ledger.jsonl.
8. Keep the paper short — no new padding.

### Q-Loop B49: Second terminal adversarial gate (14 iterations)

Fresh adversarial review of the fixed paper. Same protocol as B48.

## 5. Supervisor Verdict

```
B48 ADVERSARY NOT WON OVER. THREE REAL DEFECTS FOUND.

ALL THREE ARE THE PAPER VIOLATING ITS OWN STANDARD.
THAT IS THE MOST DAMAGING KIND OF REJECTION.

THE FIX IS HONESTY, NOT ENGINEERING:
1. RELABEL C2/C3 LEDGERS ACCURATELY
2. RUN FULL PBE ON B40 — LET IT ABSORB
3. DEMOTE POSITIVE CONTROL TO RESIDUAL-RISK DEMO

THE PAPER BECOMES STRONGER BY BEING CAUGHT.

W-LOOP B41: FIX THE THREE DEFECTS (20 ITER)
Q-LOOP B49: SECOND TERMINAL GATE (14 ITER)

THE DUAL-LOOP CONTINUES.
```
