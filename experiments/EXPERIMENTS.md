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