# WORK LOOP - Batch 13: S0/Wide7 Capacity Check + Teacher Upgrade

Date: 2026-07-07

## Grounding

Read first, in order:

1. `research/VISION.md`
2. `research/dual_loop_supervisor_checkin_11.md`
3. `research/work_loop_batch12.md`
4. `research/question_loop_batch16.md`
5. `code/s0_training.py`
6. `code/s0_option_c_training.py`
7. `code/margin_distillation.py`
8. `tmp_margin_distillation_b12/scaffold_capacity_50/scaffold_capacity.json`

Primary metric throughout: benchmark-facing forced-choice functional margins and
MCQ accuracy, not BPB, NLL, or training loss.

## Artifacts

- `code/s0_mcq_capacity.py`
- `tmp_work_loop_b13/s0_capacity.json`
- `tmp_work_loop_b13/s0_capacity_head.pt`
- `tmp_work_loop_b13_native/s0_native_finetune_capacity.json`
- `tmp_work_loop_b13_native100/s0_native_finetune_capacity.json`
- `tmp_work_loop_b13/smollm2_teacher_audit.json`
- `tmp_work_loop_b13/mamba_kernel_probe.json`
- `tmp_work_loop_b13/s0_native_finetune_b13.py`

Implementation note: `code/s0_mcq_capacity.py` is the canonical B13 harness for
the frozen residual-head capacity probe and SmolLM2 audit. The Windows sandbox
wrapper blocked later `apply_patch` updates to add the native fine-tune fallback,
so the native full-model fallback was run from the workspace-local temporary
runner `tmp_work_loop_b13/s0_native_finetune_b13.py`. That runner imports the
canonical B13 helpers and writes exact JSON artifacts, but should be promoted
into `code/s0_mcq_capacity.py` before reuse.

## Checkpoint Selection

The original S0 checkpoint exists:

- `C:/sutra_fast/checkpoints/s0_full/s0_best.pt`
- step: 10000
- eval_bpb: 1.9004
- config: P4, D=576, 30 layers, about 121M params

The stronger trained byte model also exists:

- `C:/sutra_fast/checkpoints/wide7_scout/s0_best.pt`
- step: 10000
- eval_bpb: 1.2932
- config: P4, D=1152, 7 layers, about 121M params

Because the mission allowed S0 or Wide7 and the consequence is a kill gate, the
binding capacity probe used Wide7. A hostile reviewer would not accept killing
the byte path while ignoring the better trained byte checkpoint.

## Probe 1: S0/Wide7 Scaffold Capacity Check

Question: Can a real trained byte model learn MCQ discrimination from supervised
labels?

Setup:

- Candidate: `wide7_scout/s0_best.pt`.
- Data: 288 train-safe examples, 96 each from HellaSwag, PIQA, ARC-Easy.
- Held-out eval: 144 disjoint train-safe examples, 48 each.
- Split source overlap: 0 for each benchmark.
- Max bytes per context+choice: 768.
- Binding adaptation: full native Wide7 fine-tune, 100 supervised choice-CE
  steps over the native byte-NLL continuation scorer.
- Supplemental checks:
  - frozen Wide7 with zero-initialized residual MCQ head, 80 steps;
  - full native Wide7 fine-tune, 50 steps.

The frozen residual head starts at exactly the native Wide7 byte-NLL baseline:
score = `-native_nll + residual(hidden)`, with residual initialized to zero.
The native fallback then tested the stronger claim: can the Wide7 byte LM itself
adapt under the exact forced-choice scorer?

### Binding Result: 100-Step Native Full Fine-Tune

| Benchmark | Untrained Wide7 | Label-CE trained | Delta | Verdict contribution |
|---|---:|---:|---:|---|
| HellaSwag | 25.00% | 25.00% | +0.00pp | FAIL |
| PIQA | 52.08% | 50.00% | -2.08pp | FAIL |
| ARC-Easy | 29.17% | 37.50% | +8.33pp | PASS |

Training-set accuracy moved from 38.54% to 73.61%. The model can absorb the
training labels, but the held-out movement is not broad. It passed only 1/3
benchmarks, below the precommitted >=2/3 gate.

Precommitted token:

```text
FAIL_S0_CAPACITY
```

Interpretation: Wide7 is not optimizer-dead. It memorized or fit the train-safe
training slice. But under the 100-step fair-chance budget it did not show broad
held-out MCQ scaffold capacity. The strongest signal was ARC-Easy only; HellaSwag
was flat and PIQA regressed.

### Supplemental Result: Frozen Residual Head

| Benchmark | Untrained Wide7 | Label-head trained | Delta |
|---|---:|---:|---:|
| HellaSwag | 25.00% | 20.83% | -4.17pp |
| PIQA | 52.08% | 43.75% | -8.33pp |
| ARC-Easy | 29.17% | 33.33% | +4.17pp |

Training-set accuracy moved from 38.54% to 60.07%, but held-out regressed on
HellaSwag and PIQA. Frozen Wide7 representations plus a small residual head did
not pass capacity.

### Supplemental Result: 50-Step Native Full Fine-Tune

| Benchmark | Untrained Wide7 | Label-CE trained | Delta |
|---|---:|---:|---:|
| HellaSwag | 25.00% | 25.00% | +0.00pp |
| PIQA | 52.08% | 54.17% | +2.08pp |
| ARC-Easy | 29.17% | 29.17% | +0.00pp |

Training-set accuracy moved from 38.54% to 57.99%, but held-out movement was
only marginal on PIQA and flat elsewhere.

### Probe 1 Verdict

The binding verdict is `FAIL_S0_CAPACITY`.

This is not the same as "no gradient can move Wide7." The gradients moved train
accuracy. The failure is more specific and more damaging: short-budget supervised
MCQ training did not generalize across benchmark families even on the strongest
available trained byte checkpoint.

Consequence: Probe 3 is not opened.

## Probe 2: SmolLM2-360M Teacher Quality Audit

Question: Is SmolLM2-360M a better primary teacher than Qwen3-0.6B?

Setup:

- Teacher: `HuggingFaceTB/SmolLM2-360M`.
- Scoring: full continuation NLL/token.
- Examples: 200 train-safe examples per benchmark.
- Reference: B12 Probe B Qwen numbers.

| Benchmark | Qwen B12 acc | SmolLM2 acc | Delta | Qwen mean margin | SmolLM2 mean margin | Confident wrong |
|---|---:|---:|---:|---:|---:|---:|
| HellaSwag | 49.5% | 56.0% | +6.5pp | -0.0263 | +0.0911 | 41.5% |
| PIQA | 67.5% | 65.0% | -2.5pp | +0.1920 | +0.1543 | 30.5% |
| ARC-Easy | 34.0% | 56.5% | +22.5pp | -1.2170 | +0.1613 | 41.5% |

Hard-negative notes:

- HellaSwag hard negatives were not dominated by shortest or longest endings
  (shortest 20.5%, longest 32.5%).
- PIQA is binary, so shortest/longest overlap is expected and not diagnostic.
- ARC-Easy improved dramatically over Qwen, but hard negatives skewed longest
  72.5%; future ARC teacher use should keep length diagnostics visible.

Precommitted token:

```text
UPGRADE_TEACHER
```

Interpretation: SmolLM2 clears the >=+5pp accuracy gate on 2/3 benchmarks
and improves margin quality on 2/3. Qwen remains stronger on this PIQA sample,
so the upgrade is not "SmolLM2 dominates everywhere." The actionable rule is:
use SmolLM2 as primary/default teacher for HellaSwag and ARC-Easy, and keep Qwen
available as a PIQA co-teacher or disagreement partner.

## Probe 3: FMD_SHADOW_288 On S0

Status:

```text
SKIPPED_BY_FAIL_S0_CAPACITY
```

Reason: Probe 3 was explicitly conditional on `PASS_S0_CAPACITY`. Probe 1
failed after both frozen-head and native full-model adaptation checks. Running
teacher-margin training now would launder a teacher-objective experiment into a
student-capacity failure.

The preserved FMD repair shot is not consumed. It should only be used if a future
student passes a label-only capacity gate.

## Probe 4: Mamba Kernel Install + Cross-Architecture Disagreement

Attempted command:

```text
pip install mamba-ssm causal-conv1d
```

Result:

- `pip` reported existing installs for `mamba-ssm 2.2.6.post3` and
  `causal-conv1d 1.5.3.post1`.
- During resolution it attempted to build a newer `mamba-ssm` and failed in the
  isolated build environment because `torch` was not visible.
- Runtime import then failed:

```text
ImportError: DLL load failed while importing selective_scan_cuda: The specified procedure could not be found.
```

Precommitted operational decision:

```text
DEFER_MAMBA_DISAGREEMENT
```

Interpretation: Mamba-790M remains blocked by the local CUDA extension/toolchain,
not by VRAM or benchmark logic. No Mamba disagreement audit was run.

## Batch Verdict

| Probe | Token | Action |
|---|---|---|
| 1. S0/Wide7 capacity | `FAIL_S0_CAPACITY` | Do not run FMD on this scaffold. Treat current byte-student path as the bottleneck. |
| 2. SmolLM2 audit | `UPGRADE_TEACHER` | Use SmolLM2 as the default stronger teacher, with Qwen retained for PIQA/disagreement. |
| 3. FMD on S0 | `SKIPPED_BY_FAIL_S0_CAPACITY` | Repair shot not consumed; no teacher-objective experiment on failed capacity. |
| 4. Mamba kernels | `DEFER_MAMBA_DISAGREEMENT` | CUDA extension import is broken; defer cross-architecture SSM audit. |

## Recommendation

The current byte-native KD line should not receive another objective repair on
S0/Wide7. B12 showed the tiny MarginStudent cannot learn labels. B13 now shows
the strongest available trained byte checkpoint can fit the train split but does
not pass broad held-out MCQ capacity under direct labels.

That is a project-level architecture/initialization problem, not a teacher-loss
problem. The next serious move should be one of:

1. change the student birth mechanism, probably chain-init or inherited
   representation;
2. test a token-level capable student baseline such as SmolLM2 as the functional
   control;
3. pivot the main moonshot away from byte-native KD if the democratization
   invariant is better served elsewhere.

Teacher disagreement remains valuable fuel, and SmolLM2 is now the better
primary teacher. But B13 did not find the engine.

## NARRATIVE SECTION

One-sentence story: We tested the real trained byte model instead of the tiny
MarginStudent, and even Wide7 failed to turn direct supervised MCQ labels into
broad held-out benchmark gains.

Does it survive "isn't that obvious?": Yes. The strongest objection after B12
was that the failed scaffold was tiny and random. B13 removed that objection by
using the trained 121M byte checkpoint and then giving it both a residual-head
probe and a native full-model fine-tune probe.

Does it survive "so what?": Yes. The result closes FMD on S0 by rule and forces
the project away from objective tinkering. The positive teacher result matters
only after a capable student exists.

If boring, say so: This is boring as moonshot evidence and valuable as falsifying
evidence. The project now has teacher fuel, but the current byte student path is
not an engine.
