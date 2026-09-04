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
| `pccp_h_absorption_record` | 2026-07-07 | PCCP-H | Absorbed as discovery paradigm | Finite after-frame witness remained useful, but B1-B3 discovery and novelty were absorbed by exhaustive baselines. |
| `frameseed_boolean_b28` | 2026-07-07 | FrameSeed | Absorbed by teaching dimension | Boolean supplied-frame packet solved perfectly, but exact finite teaching/search also solved perfectly. |
| `frameseed_sheets_b31` | 2026-07-07 | FrameSeed/SHEETS | Absorbed by schema binding | Typed packet was unnecessary once exact charged bindings and typed pipeline substrate were granted. |
| `wgd_b37_b38` | 2026-07-08 | WGD | Absorbed by constraint discovery | Toy schema/binding and hard GF(2) constraint solving absorbed grammar discovery. |
| `e3_teacher_tomography_b43` | 2026-07-08 | E3 | Positive control only | Friendly E3 toy beat ordinary baselines, but geometry parity was still untested. |
| `e3_teacher_tomography_b44` | 2026-07-08 | E3 | Supplied-geometry toy absorbed | Exact tool beat E3 and nuisance oracle matched E3; inference gate required. |

## Embedding Reboot (September 2026)

| ID | Date | Track | Verdict | Summary |
|----|------|-------|---------|---------|
| `eklavya_embed_e1` | 2026-09-03 | Eklavya-Embedding | Passes kill criterion (thin margin) | Untrained student (149M), 2 teachers, MS MARCO 500 pairs, 4 arms. Tomography MRR 0.561 > B3+0.01. Gain +0.260 vs +0.236. BUT: 200-pair replication reverses arm ordering. |
| `eklavya_embed_v2r2` | 2026-09-03 | Eklavya-Embedding | **Narrow negative** (reclassified from Kill #15) | Frozen encoder + residual head, single teacher, B4c absorber. Response-delta = B4c (+0.0000). Ceiling-saturated (baseline nDCG 0.93). Terminal criterion not executed. Corrected adjudication (E1.5) required. |
| `eklavya_embed_e1_200` | 2026-09-04 | Eklavya-Embedding | Contradicts 500-pair E1 | 200-pair replication: single-teacher KD wins (+0.241), tomography third (+0.176). Signal is seed/split-dependent. |
| `eklavya_vision_v1` | 2026-09-04 | Eklavya-Vision | **Inconclusive** (Codex FAIL) | 5-arm CIFAR-100 with DINOv2-small. Codex evidence gate: exploratory debugging, not valid evidence. Catastrophic forgetting masks method differences. Probe-target misalignment. Neither "tomography dead" nor "standard KD wins" established. Method question OPEN. |

## Experiment Notes

### `byte_marginal_kd_disconnect`

Two byte-level distillation routes improved byte prediction but did not produce
meaningful downstream judgment transfer. This killed byte-marginal KD as the
mainline mechanism.

Primary docs:

- `research/STATUS.md`
- `research/DEEP_RETHINK.md`

### `brainseed_v0_scorers`

Brainseed v0 was tested as a frozen downstream scorer family. Ridge, MLP,
bilinear, and learned-cosine variants all performed worse than codec-only
scoring. Zero-cost chart rescues did not recover the track.

Verdict: `BRAINSEED_DEAD_AS_BIRTH_ARTIFACT`

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

Verdict: `FAIL_EVIDENCE_NATIVE_FIRST_PROTOTYPE`

Primary docs:

- `research/work_loop_batch6.md`
- `research/question_loop_batch8.md`
- `research/dual_loop_supervisor_checkin_5.md`

### `pccp_h_absorption_record`

PCCP-H preserved an important audit and verification methodology, but it is not
the current discovery paradigm. The finite after-frame witness remains a narrow
result; B1, B2, and B3 discovery were absorbed by equal-information exhaustive
baselines.

Primary docs:

- `research/dual_loop_supervisor_checkin_23.md`
- `research/DEEP_RETHINK.md`
- `research/STATUS.md`

### `frameseed_boolean_b28`

Boolean FRAMESEED-0 reached perfect hidden HFA, but exact finite teaching/search
and other declared absorbers also reached perfect HFA.

Terminal token: `FRAMESEED_T3_ABSORBED_BY_TEACHING_DIMENSION`

Primary docs:

- `research/work_loop_batch28.md`
- `research/dual_loop_supervisor_checkin_32.md`
- `research/question_loop_batch40.md`

### `frameseed_sheets_b31`

Typed SHEETS-0 reached perfect hidden typed HFA, but packet-erasure drop was 0.0
and binding-only HFA was 1.0. This is a conservative granted-binding and typed
pipeline-substrate absorption, not a public claim that every named typed prior-
art baseline was natively executed.

Terminal token: `FRAMESEED_SHEETS0_ABSORBED_BY_SCHEMA_BINDING`

Primary docs:

- `research/work_loop_batch31.md`
- `research/dual_loop_supervisor_checkin_32.md`
- `research/question_loop_batch40.md`

### `wgd_b37_b38`

World Grammar Discovery escalated from toy schema/binding tests to a hard GF(2)
constraint domain. Brute search failed on the hard case, but structured
constraint discovery still absorbed the proposed grammar-discovery claim.

Terminal token: `WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY`

Primary docs:

- `research/work_loop_batch37.md`
- `research/work_loop_batch38.md`
- `research/dual_loop_supervisor_checkin_36.md`
- `research/question_loop_batch46.md`

### `e3_teacher_tomography_b43`

The first E3 toy experiment produced a strong friendly signal: source-specific
lesson packets reached 0.8588 mean hidden accuracy across 50 seeds while ordinary
baselines sat near chance. This proved the intended object can work in a
controlled toy, not that teacher tomography discovered geometry.

Primary artifacts:

- `code/e3_teacher_tomography.py` (deleted 2026-09-04, preserved in git history)
- `code/compare_ablations.py` (deleted 2026-09-04, preserved in git history)
- `experiments/e3_teacher_tomography_smoke.json`
- `experiments/e3_teacher_tomography_result.json`
- `experiments/e3_teacher_tomography_result_50seed.json`
- `research/work_loop_batch43.md`
- `research/dual_loop_supervisor_checkin_41.md`

### `e3_teacher_tomography_b44`

The hostile equal-geometry absorber test killed the supplied-geometry E3 claim:
B13 exact domain tool reached 1.0 hidden accuracy, and B15 nuisance oracle matched
E3 exactly across 50/50 seeds. E3 survives only as an inference-gate direction.

Primary artifacts:

- `code/e3_teacher_tomography.py` (deleted 2026-09-04, preserved in git history)
- `experiments/e3_teacher_tomography_hostile_smoke.json`
- `experiments/e3_teacher_tomography_hostile_result.json`
- `research/work_loop_batch44.md`
- `research/question_loop_batch52.md`
- `research/dual_loop_supervisor_checkin_42.md`
