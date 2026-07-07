# Experiments

**Purpose:** human-readable index of material experiments and gates.

The JSONL companion file, `experiments/ledger.jsonl`, is the append-only
machine-readable ledger. Keep model identities and operational details out of
commit messages; this public index uses generic descriptions.

## Current Verdicts

| ID | Date | Track | Verdict | Summary |
|----|------|-------|---------|---------|
| `byte_marginal_kd_disconnect` | 2026-07-03 | Byte-KD | Failed as mainline | Byte prediction improved sharply, but downstream judgment barely moved. |
| `brainseed_v0_scorers` | 2026-07-07 | Brainseed | Dead | Learned scorers all lost to codec-only scoring. |
| `chain_init_compat_probe` | 2026-07-07 | Chain-init | Weak positive signal | Inherited-coordinate layers beat random layers in a compatibility probe, but absolute quality is not benchmark-ready. |
| `evidence_native_v0_proto` | 2026-07-07 | Evidence-native | Failed v0 gates | The first learned evidence judge lost to no-evidence/shuffled conditions and the best same-retriever dumb baseline. |
| `evidence_native_survival_test` | 2026-07-07 | Evidence-native | Open | Broader direction survives only if a stronger judge shows control-resistant learned judgment geometry. |
| `pccp_h_absorption_record` | 2026-07-07 | PCCP-H | Absorbed as discovery paradigm | Finite after-frame witness remained useful, but B1-B3 discovery and novelty were absorbed by exhaustive baselines. |
| `frameseed_boolean_b28` | 2026-07-07 | FrameSeed | Absorbed by teaching dimension | Boolean supplied-frame packet solved perfectly, but exact finite teaching/search also solved perfectly. |
| `frameseed_sheets_b31` | 2026-07-07 | FrameSeed/SHEETS | Absorbed by schema binding | Typed packet was unnecessary once exact charged bindings and typed pipeline substrate were granted. |
| `frameseed_arc_b32_b40` | 2026-07-07 | FrameSeed arc | Current form absorbed | Two domains, two absorptions; methodology preserved and direction redirects toward frame discovery. |

## Experiment Notes

### `byte_marginal_kd_disconnect`

Two byte-level distillation routes improved byte prediction but did not produce
meaningful downstream judgment transfer. This killed byte-marginal KD as the
mainline mechanism.

Primary docs:

- `research/STATUS.md`
- `research/DEEP_RETHINK.md`
- `research/INDEPENDENT_ANALYSIS.md`

### `brainseed_v0_scorers`

Brainseed v0 was tested as a frozen downstream scorer family. Ridge, MLP,
bilinear, and learned-cosine variants all performed worse than codec-only
scoring. Zero-cost chart rescues did not recover the track.

Verdict:

`BRAINSEED_DEAD_AS_BIRTH_ARTIFACT`

Primary docs:

- `research/work_loop_batch5.md`
- `research/dual_loop_supervisor_checkin_4.md`

### `chain_init_compat_probe`

A chain-init compatibility probe found a weak positive signal: copied inherited
coordinate layers were more compatible with codec-derived inputs than random
layers. This promotes chain-init to strong baseline/fallback, not to the
moonshot mainline.

Primary docs:

- `research/work_loop_batch5.md`
- `research/dual_loop_supervisor_checkin_4.md`

### `evidence_native_v0_proto`

Evidence-Native v0 ran end to end and failed the first prototype gates. The
learned judge did not show reliable evidence-conditioned judgment: retrieved
evidence did not beat no-evidence, shuffled evidence, or the best same-retriever
dumb baseline.

Verdict:

`FAIL_EVIDENCE_NATIVE_FIRST_PROTOTYPE`

Primary docs:

- `research/work_loop_batch6.md`
- `research/question_loop_batch8.md`
- `research/dual_loop_supervisor_checkin_5.md`

### `evidence_native_survival_test`

The broader direction is not dead because v0 tested a small, weak architecture.
The next serious run must show that evidence training changes model behavior
relative to no-evidence-trained controls, dumb baselines, leakage controls, and
geometry probes.

Primary docs:

- `research/dual_loop_supervisor_checkin_5.md`
- `research/STATUS.md`

### `pccp_h_absorption_record`

PCCP-H preserved an important audit and verification methodology, but it is not
the current discovery paradigm. The finite after-frame witness remains a narrow
result; B1, B2, and B3 discovery were absorbed by equal-information exhaustive
baselines, and B3 synthesis value was real but novelty-absorbed.

Primary docs:

- `research/PCCP_PRECOMMIT_SPEC.md`
- `research/PCCP_THEOREM_DRAFT.md`
- `research/adversarial_review_final.md`
- `research/STATUS.md`

### `frameseed_boolean_b28`

Boolean FRAMESEED-0 reached perfect hidden HFA, but exact finite teaching/search
and other declared absorbers also reached perfect HFA. The terminal token was:

`FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`

Primary docs:

- `research/frameseed_0_precommit_spec.md`
- `research/work_loop_batch28.md`
- `experiments/frameseed0_b28_hidden_hfa.json`

### `frameseed_sheets_b31`

Typed SHEETS-0 reached perfect hidden typed HFA, but packet-erasure drop was 0.0
and binding-only HFA was 1.0. This is a conservative granted-binding and typed
pipeline-substrate absorption, not a public claim that every named typed prior-
art baseline was natively executed.

Terminal token:

`FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING`

Primary docs:

- `research/frameseed_sheets_0_spec.md`
- `research/work_loop_batch31.md`
- `experiments/frameseed_sheets0_b31_hidden_hfa.json`
- `research/question_loop_batch40.md`

### `frameseed_arc_b32_b40`

The FrameSeed arc is complete as current-form supplied packet transmission.
Boolean and SHEETS both absorbed. The surviving artifact is the methodology:
absorption ladder, precommitted terminal tokens, equal-information baseline
parity, hidden-open discipline, packet-erasure, cost ledgers, and strict claim
ceilings.

Arc token:

`FRAMESEED_CURRENT_FORM_ABSORBED_METHODOLOGY_PRESERVED_REDIRECT_TO_FRAME_DISCOVERY`

Primary docs:

- `research/frameseed_milestone_report.md`
- `research/dual_loop_supervisor_checkin_32.md`
- `research/question_loop_batch40.md`
- `research/METHODOLOGY_TEMPLATE.md`
- `research/work_loop_batch33.md`
