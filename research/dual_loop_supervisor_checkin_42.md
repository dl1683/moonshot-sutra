# Dual-Loop Supervisor Check-in #42

**Date: 2026-07-08**
**Reviewing: W-Loop B44 (hostile E3 toy) + Q-Loop B52 (attack E3 signal)**

---

## 1. Results

### W-Loop B44: HOSTILE TOY KILLS TEACHER IDENTITY CLAIM

The hostile absorber test produced the clearest result in 14 kill cycles:

| Method | Mean hidden acc | vs E3 |
|---|---:|---:|
| B13 exact domain tool | 1.0000 | +14.12pp |
| **B15 nuisance oracle** | **0.8588** | **0.00pp** |
| **E3 source-specific lessons** | **0.8588** | — |
| B10+ enhanced augmentation | 0.7512 | -10.75pp |
| All ordinary baselines | ~0.50 | ~-35pp |

**Per-seed: E3 = B15 in 50/50 seeds. Zero seeds where E3 beats B15.**

Token: `B44_E3_HOSTILE_NUISANCE_ORACLE_ABSORBS_REQUIRES_INFERENCE_GATE`

### Q-Loop B52: E3 SIGNAL REAL BUT TOY

7 iterations attacking from different angles. Key verdicts:
- I463: 35pp gap tests hand-authored labels vs no labels, not tomography
- I464: >95% of E3's cost is researcher-authored geometry
- I465: No recipe for finding natural heterogeneous teachers
- I466: Active learning baseline too weak to be honest absorber
- I467: Geophysical analogy is exposition, not evidence
- I468: Novelty claim not isolated from sensor fusion literature
- I469: Mission gate not passed — E3 requires expensive scaffolding

Token: `Q_LOOP_B52_E3_SIGNAL_REAL_BUT_TOY_AND_EXPENSIVE_NATURAL_PATH_UNSPECIFIED`

## 2. Supervisor Audit

### The convergence is remarkable

Both loops independently reached the same conclusion:
- W-Loop proved empirically that teacher identity adds nothing (B15 = E3)
- Q-Loop predicted theoretically that the researcher pays for >95% of E3

The 13-kill pattern is confirmed for E3's current form. Supplied geometry
gets absorbed. This is Kill #14 at the mechanism level — the CURRENT E3
toy doesn't test what it claims to test.

### What's genuinely dead

- "Teacher identity is necessary for hidden-transform transfer" — dead.
  B15 proves this with zero ambiguity across 50 seeds.
- "Teacher tomography discovers hidden structure" — dead in this toy.
  The XOR rule is hard-coded, not discovered.
- "E3 is a paradigm shift" — not with supplied geometry.

### What's genuinely alive

- The DIRECTION survives because the kill is specific. It kills
  supplied-geometry E3, not discovery-based E3.
- E3 beats B10+ by 10.75pp. This means the packet FORMAT has value
  beyond raw augmentation — the nuisance correction matters. The question
  is whether E3 can DISCOVER the nuisance correction.
- The B10+/B15 gap (10.75pp) is the information content of knowing the
  nuisance structure vs. just having the transformation generator. This
  is the object E3 must learn to infer.

### The critical next question

**Can E3 infer the composition rule from teacher behavior on calibration
data, without being told the rule?**

This is the inference gate. If a simple function over teacher margins on
~32 calibration examples can rediscover the XOR composition rule, then
E3's inference claim survives and the direction is worth pursuing into
natural domains. If it can't, E3 joins the graveyard.

## 3. Mission-Drift Check

- "Does this make intelligence cheaper?" → The kill result is informative.
  It tells us exactly what E3 must do to serve the mission: discover
  geometry cheaply, not receive it from researchers.
- "Positive capability or documenting failure?" → This is an honest kill,
  not a capability. But it's a PRODUCTIVE kill — it constrains the search
  space and points toward the inference gate.
- "Process or product?" → Product. 50-seed sweep with clear verdict.
  The repo carries the proof.

## 4. Narrative Gate

**Headline:** "We tested our method against its own geometry. It tied.
The geometry was doing the work, not the teacher disagreement."

- "Obvious?" → In hindsight yes, but the B43 shuffled controls made it
  LOOK like teacher identity mattered. B44 revealed that shuffling
  destroyed the geometry, not the identity.
- "Trivial?" → No — the B10+/B15 gap shows the nuisance correction has
  real information content. The question shifts from "does teacher
  tomography work?" to "can teacher tomography discover the correction?"
- Mission test → Not yet. But the path is narrower and sharper.

**Verdict: DIRECTION ALIVE. MECHANISM NEEDS INFERENCE GATE.**

## 5. Steering for Next Batch

### W-Loop B45 (10 iterations):
1. **INFERENCE GATE.** Replace the hard-coded `infer_packet_rule` with a
   genuine inference step:
   - Take teacher margins on calibration examples (where true labels are
     known)
   - Try to discover the composition rule (which function of teacher
     margins predicts the true ranking?)
   - Options: logistic regression over teacher margin products,
     correlation analysis, mutual information, exhaustive search over
     small function families
   - Compare: inferred composition rule vs. hand-authored XOR rule vs.
     random composition vs. no composition
2. If the inferred rule matches the hand-authored rule on calibration data
   AND transfers to hidden transforms, E3's inference claim survives.
3. If the inferred rule fails to match, or matches on calibration but
   doesn't transfer, the inference is trivial or lucky.
4. Run B15 nuisance oracle against the INFERRED version too — does
   inferred-E3 still tie with B15?

### Q-Loop B53 (7 iterations):
1. Attack the inference gate design. Is logistic regression over teacher
   margins a trivial baseline? Could the composition be discovered by
   ANY method with access to calibration labels?
2. If the inference gate passes: what does the natural-domain version look
   like? Where do calibration labels come from in a real task?
3. Is the "infer the composition" step equivalent to standard feature
   selection? If so, E3 reduces to "do feature selection over teacher
   outputs" — which is known and not a paradigm shift.
4. Competitive check: has anyone done composition-rule discovery over
   teacher outputs? Multi-teacher feature selection?

## 6. Repo State

Repo hygiene pass due NEXT batch (3 W-Loop iterations since cleanup:
B42, B43, B44). Will trigger the hygiene agent concurrently with B45.

Files added this batch:
- code/e3_teacher_tomography.py (modified with hostile absorbers)
- experiments/e3_teacher_tomography_hostile_result.json
- experiments/e3_teacher_tomography_hostile_smoke.json
- research/work_loop_batch44.md
- research/question_loop_batch52.md
