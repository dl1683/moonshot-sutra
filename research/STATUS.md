# Project Status

**Last updated:** 2026-07-08
**Current state:** Terminal gate passed — "The Absorption Ladder" accepted at current claim ceiling.
**Fixed points:** The five sacred outcomes in `research/VISION.md`.

## Current Verdict

**Terminal gate passed. Adversary won over.**

Token: `Q_LOOP_B49_ADVERSARY_WON_OVER_ACCEPT_AT_CURRENT_CLAIM_CEILING`

The methodology paper (`research/methodology_paper.md`, 259 lines) passed a
fresh-eyes adversarial gate after two revision cycles. The paper is a
roster-relative methodology proposal with internal negative case evidence
and an absorbed positive-control attempt.

The absorption ladder — the evaluation methodology that produced 13 honest kills
across two moonshot arcs — is the publishable contribution. Paper in progress:
`research/methodology_paper.md`.

```text
WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY (hard domain, 64 rules, 2^64 space)
```

### FrameSeed Arc (Kills #12)

Two domains were tested and both absorbed:

| Domain | Evidence | Terminal token | Internal status |
|---|---|---|---|
| Boolean FRAMESEED-0 | `experiments/frameseed0_b28_hidden_hfa.json` | `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION` | Absorbed by exact finite teaching/search |
| Typed SHEETS-0 | `experiments/frameseed_sheets0_b31_hidden_hfa.json` | `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING` | Absorbed once exact bindings and typed pipeline substrate are granted |

The milestone report is `research/frameseed_milestone_report.md`. The final
supervisor gate is `research/dual_loop_supervisor_checkin_32.md`. The fresh-eyes
adversarial review is `research/question_loop_batch40.md`.

## Public Claim Ceiling

The negative result is strong enough for internal direction-killing. It is not a
publication-grade claim that every named typed prior-art baseline was natively
implemented.

- Boolean FRAMESEED-0 is a clean absorption by finite teaching/search.
- SHEETS-0 is a conservative packet-erasure and granted-binding demolition.
  B31 used capability-mode scoring for typed baselines, not native learned PBE,
  CEGIS, schema-matching, or MDL-library execution.
- Future public text must say that the packet was unnecessary once exact
  bindings and typed pipeline substrate were granted. Do not say native typed
  baselines empirically solved SHEETS-0 unless a later artifact implements them.

## Full Kill Record

The numbered record below preserves the repo's own kill numbering where it is
explicit. Kills 1-8 come from the accumulated Eklavya-era record, #9 and #10 are
explicitly named by supervisor gates, #11 is the PCCP-H absorption slot between
CTI and FrameSeed, and #12 is the final FrameSeed gate.

| # | Direction or mechanism | Binding evidence | Terminal status | Root cause |
|---:|---|---|---|---|
| 1 | Gradient KD | B3 record summarized in supervisor #12 | `FAIL` | Proxy improved while benchmark function stayed flat |
| 2 | Brainseed | `research/work_loop_batch5.md`, supervisor #4 | `BRAINSEED_DEAD_AS_BIRTH_ARTIFACT` | Learned scorers lost to codec-only scoring |
| 3 | Evidence-Native v0 | `research/work_loop_batch6.md`, Q8, supervisor #5 | `FAIL` | Evidence-conditioned judgment did not beat controls |
| 4 | Evidence-Native v1 | B8 record summarized in supervisor #12 | `FAIL` | Internalization gate did not clear the moonshot bar |
| 5 | Coordinate-Inheritance | B9-B10, supervisor #10/#12 | `FAIL` | Compatibility proxy did not become benchmark function |
| 6 | FMD prototype | B11, supervisor #10/#12 | `FAIL_MARGIN_PROTOTYPE` | Training signal hurt HellaSwag while proxy loss moved |
| 7 | MarginStudent scaffold | `research/work_loop_batch12.md`, supervisor #11/#12 | `FAIL_SCAFFOLD` | Student could not learn supervised labels under the gate |
| 8 | S0/Wide7 byte capacity | `research/work_loop_batch13.md`, supervisor #12 | `FAIL_S0_CAPACITY` | Memorized train data, held-out function stayed flat |
| 9 | Eklavya routing mechanism | `research/work_loop_batch14.md`, supervisor #13 | `FAIL_EKLAVYA_MECHANISM` | No residual over single-teacher KD; oracle ceiling failed |
| 10 | CTI smooth compute law | `research/work_loop_batch16.md`, `research/work_loop_batch17.md`, supervisor #14 | `PROXY_ONLY_LAW` / dead | Smooth `D(C)` law was beaten by trivial forecasters and proxy/function divergence |
| 11 | PCCP-H as discovery paradigm | `research/adversarial_review_final.md`, current PCCP status | Discovery absorbed; methodology retained | B1-B3 discovery absorbed by exhaustive baselines; finite after-frame witness was real but not a paradigm |
| 12 | FrameSeed / Intelligence Vaccines | `research/work_loop_batch28.md`, `research/work_loop_batch31.md`, B32/B40 | `FRAMESEED_CURRENT_FORM_ABSORBED` | Supplied frames collapsed into teaching sets, bindings, and typed synthesis substrate |
| 13 | World Grammar Discovery (WGD) | `code/wgd0_measurement.py`, `code/wgd0_b38_hard_domain.py`, B35-B38 | `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY` | Grammar discovery absorbed by schema/binding (toy) and GF(2) constraint solving (hard domain) |

Side demotions not counted as numbered kills:

- Neural CWC was killed as a mainline candidate during direction finding, but it
  did not consume a full measurement arc.
- Chain-init remains a weak positive signal and strong baseline/fallback, not an
  active mainline.
- Evidence-native survives only as a future stronger test under stricter
  evidence, control, and geometry gates.

## What Was Learned

1. Perfect hidden HFA is not signal. Both FrameSeed L3 systems reached perfect
   hidden accuracy and were still absorbed.
2. Packet compactness is not enough. A compact object can still be only a
   teaching set, binding list, or pipeline hint.
3. Packet-erasure is decisive. In SHEETS-0, binding-only HFA was 1.0 and
   packet-erasure drop was 0.0.
4. Supplied geometry keeps getting absorbed. PCCP-H and FrameSeed both failed
   when the experiment designer supplied the ontology, verifier, operation
   grammar, binding schema, or frame.
5. The reusable artifact is the methodology: precommit tokens, absorber-first
   design, equal-information baselines, hidden-open discipline, cost ledgers,
   ablations, audits, and strict claim ceilings.

## Current Direction: Methodology Paper

The absorption ladder methodology IS the deliverable. Paper draft:
`research/methodology_paper.md`.

Four case studies:
1. Boolean FrameSeed-0 → absorbed by teaching dimension
2. Typed SHEETS-0 → absorbed by schema/binding discovery
3. WGD toy (16 rules) → absorbed by schema/binding (even toy baselines)
4. WGD hard (64 rules, 2^64 space) → absorbed by GF(2) constraint discovery

Key progression: domain difficulty escalated across all 4 (brute search finally
failed at case 4), but structured constraint methods still absorbed.

## Reusable Methodology

Use `research/METHODOLOGY_TEMPLATE.md` for the next direction's precommit and
harness design. It preserves:

- absorption ladder;
- precommitted terminal tokens;
- equal-information baseline parity;
- hidden-open discipline;
- split RNG and manifest hashing;
- blind constructor and provenance logging;
- packet-erasure / component-erasure ablations;
- native-vs-proxy baseline labels;
- role/schema/name/unit/row permutation audits;
- cost ledgers for frame, binding, program, parser, human labor, examples,
  counterexamples, verifiers, and residual teaching bits;
- strict claim ceilings.

FrameSeed harnesses that should be reused as patterns, not as active claims:

- `code/frameseed0_harness.py` - Boolean audit harness template.
- `code/frameseed_sheets0_harness.py` - typed domain audit harness template.

## Operating Rules

- Start from `research/VISION.md`.
- Treat every mechanism as replaceable.
- Think first, formalize first, test small.
- CPU-first experiments unless explicitly authorized otherwise.
- Give ordinary baselines first refusal.
- Distinguish native execution, proxy absorber, capability-mode scoring, and
  formal proof/lower bound.
- Do not narrate mixed evidence into a win.
- The loop stops only when a hostile fresh-eyes reviewer cannot move the token.

## Artifact Index

### Active Canon

- `research/VISION.md` - first-principles vision and five sacred outcomes.
- `research/STATUS.md` - current state and kill record.
- `research/METHODOLOGY_TEMPLATE.md` - reusable absorption/precommit/harness
  framework.
- `research/frameseed_milestone_report.md` - B32 milestone report.
- `research/dual_loop_supervisor_checkin_32.md` - final FrameSeed supervisor
  gate.
- `research/question_loop_batch40.md` - fresh-eyes adversarial review.
- `research/work_loop_batch33.md` - post-FrameSeed cleanup and direction-prep
  log.

### FrameSeed Specs, Reports, And Evidence

- `research/frameseed_0_precommit_spec.md` - Boolean FRAMESEED-0 hardened
  precommit.
- `research/frameseed_sheets_0_spec.md` - typed SHEETS-0 hardened precommit.
- `research/question_loop_batch33.md` - adversarial pre-test that predicted the
  teaching/synthesis absorption route.
- `research/work_loop_batch28.md` - Boolean hidden measurement report.
- `research/work_loop_batch31.md` - SHEETS-0 hidden measurement report.
- `experiments/frameseed0_b28_hidden_hfa.json` - Boolean hidden measurement.
- `experiments/frameseed_sheets0_b31_hidden_hfa.json` - SHEETS hidden
  measurement.
- `code/frameseed0_measurement.py` - Boolean measurement runner.
- `code/frameseed_sheets0_measurement.py` - SHEETS measurement runner.

### Historical PCCP-H Record

- `research/PCCP_PRECOMMIT_SPEC.md` - PCCP-H specification.
- `research/PCCP_THEOREM_DRAFT.md` - three-part theorem draft.
- `research/adversarial_review_final.md` - hostile fresh-eyes review.
- `research/DEEP_RETHINK.md` - historical kill log.
- `code/pccp0_witness.py` - after-frame witness and B1 absorption.
- `code/pccp0_b2_relations.py` - B2 relation absorption.
- `code/pccp0_b3_decomposition.py` - B3 decomposition absorption.

### WGD Specs, Reports, And Evidence

- `research/wgd_0_precommit_spec.md` - WGD-0 hardened precommit.
- `research/question_loop_batch41.md` - direction-finding that proposed WGD.
- `research/question_loop_batch46.md` - methodology paper outline.
- `research/dual_loop_supervisor_checkin_33.md` - WGD approval gate.
- `research/dual_loop_supervisor_checkin_36.md` - WGD kill and paper pivot.
- `code/wgd0_harness.py` - WGD-0 audit harness with 20 native absorbers.
- `code/wgd0_measurement.py` - WGD-0 hidden measurement runner.
- `code/wgd0_b38_hard_domain.py` - hard domain (64 rules) measurement.
- `experiments/wgd0_b37_hidden_measurement.json` - toy domain hidden result.
- `experiments/wgd0_b38_hidden_measurement.json` - hard domain hidden result.

### Methodology Paper

- `research/methodology_paper.md` - full draft (in progress).

### Historical Logs

- `research/work_loop_batch*.md` - W-Loop iterations.
- `research/question_loop_batch*.md` - Q-Loop iterations.
- `research/dual_loop_supervisor_checkin_*.md` - supervisor assessments.
