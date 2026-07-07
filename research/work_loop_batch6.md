# WORK LOOP - Batch 6: Evidence-Native Prototype Run

Date: 2026-07-07
Iterations: 51-60

## Executive Verdict

```text
FAIL_EVIDENCE_NATIVE_FIRST_PROTOTYPE
FAIL_RETRIEVED_OVER_CLOSED_BOOK_GATE
FAIL_RETRIEVED_OVER_DUMB_BASELINE_GATE
FAIL_SHUFFLED_CONTROL_COLLAPSE_GATE
NO_SEED42_OR_GOLD_CURRICULUM_ESCALATION_PRECONDITION_FAILED
```

The evidence-native prototype ran end to end after audit repairs. The learned judge trained, but it did **not** learn useful evidence-conditioned judgment on the precommitted test. Retrieved evidence was essentially tied with no evidence, worse than shuffled evidence, worse than wrong-topic evidence, and 3.81pp below the best same-retriever dumb baseline.

Honest headline:

**The judge read the evidence and got worse; the overlap rule beat it.**

This is not a moonshot signal. It is a falsification of this first evidence-native implementation. The measured data support the adversarial story: retrieval artifacts and benchmark/candidate heuristics are stronger than the learned judge, and in this run the retrieved evidence itself does not produce a reliable lift.

## Artifact Status

The user-requested target directory `C:/sutra_fast/evidence_native/` was readable but not writable in this sandbox. It already contained pre-patch artifacts from an earlier run, contrary to the launch note. I inspected them, but they are **not** the final result because they lacked candidate randomization and used a weaker shuffled control.

Attempts to write or snapshot under `C:/sutra_fast/evidence_native/` failed with access denied. `C:/tmp/evidence_native_b6` also failed with access denied. Final audited artifacts are therefore in the writable workspace directory:

```text
tmp_evidence_native_b6_final2/metrics.json
tmp_evidence_native_b6_final2/evidence_records.json
tmp_evidence_native_b6_final2/corpus_manifest.json
tmp_evidence_native_b6_final2/evidence_judge.pt
```

A smoke artifact also exists at:

```text
tmp_evidence_native_smoke_b6/metrics.json
```

## Iterations 51-52: Code Audit and Repairs

Read in order:

1. `research/VISION.md`
2. `research/DEEP_RETHINK.md`
3. `research/dual_loop_supervisor_checkin_4.md`
4. `research/question_loop_batch7.md`
5. `research/work_loop_batch5.md`
6. `code/evidence_native_sutra.py`

### Architecture

The architecture is coherent for a first falsification prototype:

```text
bytes(context + evidence + candidate)
  -> frozen semantic codec encoder patch states
  -> PatchProjection(256 -> d_model)
  -> small causal GlobalReasoner
  -> last+mean pooled judgment head
  -> one scalar per candidate
  -> listwise cross-entropy over candidates
```

Default model size is not 121M. It is a small prototype:

| Component | Params |
|---|---:|
| Frozen codec encoder | 4,263,168 |
| Projection | 393,472 |
| Reasoner | 1,180,928 |
| Judgment head | 131,585 |
| Total | 10,067,329 |
| Trainable | 5,542,017 |

### Repairs Made Before Final Run

The initial script compiled, but the audit found control issues that were material enough to fix before treating the run as final.

1. **Candidate order was not randomized.**
   - Added deterministic candidate shuffling for HellaSwag and PIQA.
   - Updated labels after shuffling.
   - Stored `choice_order` in examples.
   - Added `--no-randomize-choices` only as an escape hatch.

2. **`shuffled` evidence was random corpus evidence, not shuffled retrieved evidence.**
   - Replaced it with retrieved doc IDs deranged within dataset.
   - This preserves the retrieved-evidence distribution while breaking example pairing.

3. **Passage separator used `chr(258)` inside a UTF-8 string.**
   - Replaced it with ASCII `[PASSAGE]` text so passage boundaries do not become accidental UTF-8 byte content.

4. **Control metadata was not copied into `metrics.json`.**
   - Added self-describing control metadata to final metrics.

Compiled after repairs:

```powershell
python -m py_compile code/evidence_native_sutra.py
```

Result: pass.

### Remaining Audit Caveats

The dumb baseline set is useful but incomplete. It includes majority label, shortest candidate, unigram frequency, nearest-neighbor train-label transfer, and BM25 evidence-overlap ranking. It does **not** include the Q-Loop's requested <=20M learned classifier baselines or Qwen3-0.6B-with-evidence.

The leakage audit is a first-pass exact/12-gram audit, not a reviewer-proof MinHash/SimHash provenance audit.

The teacher cache is only partially compatible after candidate randomization: `278/1024` train records attach teacher scores. The run used the default `--teacher-alpha 0.0`, so those optional teacher scores did not affect training or evaluation.

## Iteration 53: Smoke Test

Command:

```powershell
python code/evidence_native_sutra.py --train-hellaswag 32 --train-piqa 32 --eval-hellaswag 64 --eval-piqa 64 --epochs 2 --progress --allow-downloads --output-dir tmp_evidence_native_smoke_b6
```

Smoke result:

| Metric | Value |
|---|---:|
| Epoch 1 loss | 1.0602 |
| Epoch 2 loss | 0.8945 |
| Epoch 1 train acc | 39.1% |
| Epoch 2 train acc | 68.8% |
| Retrieved eval | 38.28% |
| No evidence eval | 32.03% |
| Shuffled eval | 37.50% |
| Wrong-topic eval | 39.84% |
| Gold eval | 37.50% |
| Best dumb baseline | BM25 evidence-overlap ranker |

Smoke verdict: mechanics pass, loss decreases, CUDA run fits. The tiny smoke already warned that retrieved evidence did not beat the dumb baseline and shuffled did not collapse.

## Iterations 54-55: Full Training Run

Final audited command:

```powershell
python code/evidence_native_sutra.py --progress --allow-downloads --output-dir tmp_evidence_native_b6_final2
```

Run configuration:

| Field | Value |
|---|---:|
| Seed | 20260707 |
| Device | cuda |
| Train HellaSwag | 512 |
| Train PIQA | 512 |
| Eval HellaSwag | 1024 |
| Eval PIQA | 1024 |
| Corpus docs | 2200 |
| Top-k evidence | 3 |
| Epochs | 6 |
| Train evidence kind | retrieved |
| Candidate order randomized | true |
| Shuffled evidence | retrieved doc IDs deranged within dataset |

Training curve:

| Epoch | Loss | Train Acc |
|---:|---:|---:|
| 1 | 1.0678 | 37.5% |
| 2 | 0.8849 | 55.9% |
| 3 | 0.6747 | 66.9% |
| 4 | 0.5501 | 72.3% |
| 5 | 0.4094 | 80.3% |
| 6 | 0.2959 | 87.4% |

The model clearly fit the small training set. Generalization and evidence sensitivity failed.

## Iterations 56-57: Result Analysis

### Main Evaluation

| Evidence condition | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|
| Retrieved | 36.67% | 23.83% | 49.51% |
| None | 36.72% | 23.83% | 49.61% |
| Shuffled | 37.50% | 24.71% | 50.29% |
| Wrong-topic | 39.06% | 25.68% | 52.44% |
| Gold / label-conditioned retrieval | 37.65% | 24.12% | 51.17% |

### Dumb Baselines

| Baseline | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|
| Majority label | 34.91% | 23.14% | 46.68% |
| Shortest candidate | 39.31% | 24.90% | 53.71% |
| Unigram frequency | 38.67% | 25.98% | 51.37% |
| Nearest-neighbor train label | 37.16% | 26.46% | 47.85% |
| BM25 evidence-overlap ranker | **40.48%** | **26.86%** | **54.10%** |

### Precommitted Gates

| Gate | Required | Measured | Verdict |
|---|---:|---:|---|
| Retrieved beats no evidence by >=5pp | +5.00pp | **-0.05pp** | FAIL |
| Retrieved beats best dumb baseline by >=3pp | +3.00pp | **-3.81pp** | FAIL |
| Shuffled much worse than retrieved | retrieved - shuffled >=3.00pp | **-0.83pp** | FAIL |

Additional questions:

1. **Does evidence-native beat closed-book by >=5pp?** No. Retrieved is 36.67%; no evidence is 36.72%.
2. **Does it beat dumb baselines by >=3pp?** No. It loses to BM25 overlap by 3.81pp.
3. **Does shuffled evidence hurt significantly?** No. Shuffled is 0.83pp better than retrieved.
4. **Does wrong-topic evidence hurt?** No. Wrong-topic is 2.39pp better than retrieved.
5. **Does gold evidence help more than retrieved?** Slightly, but not meaningfully. Gold is 0.98pp above retrieved and still below wrong-topic and BM25 overlap.
6. **Leakage audit status?** Not clean. The audit found 13 exact context hits, 2 exact choice hits, and 1 long 12-gram hit across 2048 eval examples. Most exact context hits are short PIQA prompts such as `Question: bowl\nAnswer:`, but the HellaSwag choice/12-gram hits are real caveats. Because the result is negative, leakage does not rescue a positive claim.
7. **Model parameter count?** 10,067,329 total; 5,542,017 trainable. This is a prototype, not a 121M model.

## Iteration 58: Additional Controls

Skipped by precondition. The first run did not show a signal. No seed-42 rerun or gold-train/eval-retrieved curriculum is justified under the launch plan.

## Iteration 59: Q-Loop B7 Prediction Audit

| Q-Loop B7 prediction | Status | Evidence |
|---|---|---|
| Retrieved evidence will lift HellaSwag above closed-book | REFUTED | HellaSwag retrieved = 23.83%; no evidence = 23.83%. |
| Dumb baselines will explain more of the lift than expected | CONFIRMED | There was no lift; BM25 overlap beats the learned judge by 3.81pp overall and 3.03pp on HellaSwag. |
| Shuffled controls will not reduce all the way to no-evidence | CONFIRMED | Shuffled = 37.50%, above both retrieved and no-evidence. |
| Small classifiers will be competitive | UNCLEAR | Not measured. Strong non-neural baselines already beat the judge, so the concern remains live. |
| Qwen3-0.6B-with-evidence will beat the first Sutra judge | UNCLEAR | Not measured in this batch. Given the first judge is below BM25 overlap, the prediction remains plausible but untested. |

## Iteration 60: Synthesis

The home-run story does not survive the first audited run. The learned geometry does not beat retrieval artifacts; it does not even show positive evidence sensitivity. Training accuracy climbs to 87.4%, but evaluation accuracy stays near weak baseline levels and improves under wrong-topic evidence.

The result is worse than `retrieval quality explains all gains`. There are no gains to explain. The same corpus and retriever produce a cheap overlap baseline that beats the judge, while shuffled/wrong-topic controls fail in the wrong direction.

### What This Means

Evidence-native remains conceptually attractive, but this implementation should not be promoted. The current recipe is mostly a benchmark-shaped supervised candidate scorer over weak evidence. It has not learned to bind retrieved evidence to candidate correctness.

The most likely failure causes are:

1. **Evidence quality is too weak or misaligned.** If gold/label-conditioned retrieval barely helps, the input evidence is not carrying decisive information in a usable form.
2. **The judge architecture is too small and too easy to overfit.** It fits training but does not learn transferable evidence use.
3. **Candidate artifacts dominate.** Shortest-candidate, unigram, nearest-neighbor, and BM25 overlap are all competitive or stronger.
4. **The task mix is hostile to a naive shared judge.** PIQA drives aggregate accuracy toward binary-choice artifact baselines; HellaSwag remains near chance.
5. **The evidence controls are telling on the model.** Wrong-topic and shuffled should hurt if evidence binding is real. They do not.

### Recommendation

Do not run more seeds of this exact prototype as moonshot evidence. The next useful step is not more training on the same serialization.

Recommended fork:

1. **Demote this prototype to a negative-result artifact.** Keep it as the first falsifier for evidence-native judgment.
2. **If evidence-native continues, fix retrieval first.** Build an evidence-quality oracle slice where retrieved/gold evidence demonstrably helps a dumb or teacher judge, then test whether a learned small judge adds value above that.
3. **Add the missing small learned baselines before another claim.** Logistic/ridge/MLP and <=20M cross-encoder baselines must run under identical evidence.
4. **Return moonshot priority to the fallback with positive signal unless a redesigned evidence pipeline clears an oracle-evidence gate.** Batch 5 chain-init had weak but real compatibility signal; this evidence-native run has no positive gate.

Final verdict:

```text
EVIDENCE_NATIVE_V0_FAILED
NO_CONTROL_RESISTANT_LIFT
RETRIEVAL_AND_ARTIFACT_BASELINES_DOMINATE
NEXT_DIRECTION_CHAIN_INIT_OR_RETRIEVAL_QUALITY_ORACLE_BEFORE_MORE_JUDGE_TRAINING
```
