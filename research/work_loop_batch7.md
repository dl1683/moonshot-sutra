# WORK LOOP - Batch 7: Evidence-Native v1 Architecture Redesign

Date: 2026-07-07
Iterations: 61-70

## Executive Verdict

```text
FAIL_EVIDENCE_NATIVE_V1_INTERNALIZATION_GATE
FAIL_EVIDENCE_NATIVE_V1_EVIDENCE_USE_GATE
FAIL_EVIDENCE_NATIVE_V1_BASELINE_GATE
FAIL_EVIDENCE_NATIVE_V1_SENSITIVITY_GATE
FAIL_EVIDENCE_NATIVE_V1_GOLD_CEILING_GATE
```

Evidence-Native v1 was implemented as a new factorized judge in `code/evidence_native_v1.py`. It fixes the main v0 architectural flaw: context, evidence, and candidate are encoded separately, and candidate/context states cross-attend to evidence states before scoring.

The d=256, 2-layer factorized suite used 4,000 training records, 2,048 eval records, and 3 seeds x 2 matched training conditions. It failed every precommitted gate. Most importantly, the internalization gate is negative:

```text
M_evidence(none) - M_none(none) = -0.31pp
```

By the launch rule, this fails the steelman claim that evidence-conditioned training changed internal judgment geometry.

Honest headline:

**The redesigned judge separated evidence from candidates, attended to it explicitly, and still did not learn a useful evidence-conditioned geometry.**

## Artifact Status

The requested target `C:/sutra_fast/evidence_native_v1/` is not writable in this sandbox. Creation failed with access denied, matching the Batch 6 constraint. Final artifacts were written under the workspace:

```text
tmp_evidence_native_v1_d256/suite_metrics.json
tmp_evidence_native_v1_d256/suite_metrics_with_predictions.json
tmp_evidence_native_v1_d256/evidence_records.json
tmp_evidence_native_v1_d256/corpus_manifest.json
tmp_evidence_native_v1_d256/dumb_baselines.json
tmp_evidence_native_v1_d256/seed_*/M_*/metrics.json
```

A lean smoke/secondary run also exists:

```text
tmp_evidence_native_v1_smoke/
tmp_evidence_native_v1_full2/
```

## Iteration 61: v1 Architecture

Implemented `code/evidence_native_v1.py` as a new file, leaving v0 unchanged.

Primary architecture:

```text
context bytes   -> shared SemanticCodec encoder -> projection -> segment reasoner -> context states
candidate bytes -> shared SemanticCodec encoder -> projection -> segment reasoner -> candidate states
evidence bytes  -> shared SemanticCodec encoder -> projection -> segment reasoner -> evidence states

candidate states cross-attend to evidence states
context states   cross-attend to evidence states
candidate states cross-attend to context states

pooled interaction features -> scalar score per candidate -> listwise CE
```

Primary d=256 parameter count:

| Component | Params |
|---|---:|
| Frozen codec encoder | 4,263,168 |
| Segment projection | 393,472 |
| Segment reasoner | 1,180,928 |
| Cross-attention | 789,504 |
| Judgment head | 1,447,937 |
| Total | 12,172,417 |
| Trainable | 7,647,105 |

This is a stronger factorized architecture than v0, but still not a 121M judge. I treated it as the architecture-redesign test at matched prototype scale. The lean d=128 run was kept only as a smoke/secondary control.

## Iteration 62: Clean External Corpus

Corpus policy changed from v0. Raw benchmark train contexts and choices are not inserted as evidence documents.

Corpus manifest:

| Field | Value |
|---|---:|
| Docs | 5,000 |
| SHA256 normalized | `05ce8188caac6625050b6b134fe22b25b43149e3ef6576540a35a63de9752467` |
| External source | `common-pile/wikimedia_filtered`, `stackexchange_filtered`, `project_gutenberg_filtered`, `news_filtered` via `C:/sutra_fast/data/shards_diverse` |
| External docs scanned | 7,000 |
| External docs kept | 7,000 |
| Decontam removals | 0 by 12-gram set intersection against train+eval |
| Train rationale docs generated | 4,000 before corpus cap |
| Raw benchmark train-as-corpus | false |
| Benchmark choices-as-corpus | false |

Caveat: the requested Qwen3-0.6B rationale generation was not executed in this run. Gold evidence and train rationale docs are deterministic oracle/template rationales. This is weaker than the requested true Qwen gold condition and is marked as a limitation, not hidden.

## Iteration 63: Expanded Training Data

Prepared:

| Split | Count |
|---|---:|
| HellaSwag train | 2,000 |
| PIQA train | 2,000 |
| Total train | 4,000 |
| HellaSwag validation eval | 1,024 |
| PIQA validation eval | 1,024 |
| Total eval | 2,048 |

Candidate order is randomized deterministically. Teacher/template soft scores are attached for all records.

## Iterations 64-65: Training Implementation and Smoke

Compile passed:

```powershell
python -m py_compile code\evidence_native_v1.py
```

Smoke command:

```powershell
python code\evidence_native_v1.py --mode suite --train-hellaswag 24 --train-piqa 24 --eval-hellaswag 32 --eval-piqa 32 --max-corpus-docs 160 --max-external-docs 120 --max-rationale-docs 48 --shard-docs 180 --shard-bytes 2000000 --epochs 1 --batch-size 4 --eval-batch-size 8 --d-model 128 --layers 1 --heads 4 --kv-heads 2 --seeds 20260707 --bootstrap-samples 100 --output-dir C:\sutra_fast\evidence_native_v1 --fallback-output-dir tmp_evidence_native_v1_smoke --allow-downloads --progress
```

Smoke result: mechanics passed, both matched models trained/evaluated, output fallback worked, and cross-attention path was non-crashing.

## Iterations 66-68: Full Matched Suite

Primary command:

```powershell
python code\evidence_native_v1.py --mode suite --reuse-records --train-hellaswag 2000 --train-piqa 2000 --eval-hellaswag 1024 --eval-piqa 1024 --epochs 2 --batch-size 16 --eval-batch-size 32 --d-model 256 --layers 2 --heads 4 --kv-heads 2 --ffn-mult 2.0 --seeds 20260707,42,12345 --bootstrap-samples 500 --output-dir C:\sutra_fast\evidence_native_v1 --fallback-output-dir tmp_evidence_native_v1_d256 --allow-downloads --progress
```

Training fit the data but did not produce evidence-use generalization.

| Seed | Model | Epoch 2 Train Acc | Peak VRAM |
|---:|---|---:|---:|
| 20260707 | M_evidence | 55.55% | 2.57 GB |
| 20260707 | M_none | 58.65% | 2.57 GB |
| 42 | M_evidence | 52.73% | 2.57 GB |
| 42 | M_none | 54.20% | 2.57 GB |
| 12345 | M_evidence | 59.38% | 2.57 GB |
| 12345 | M_none | 61.10% | 2.57 GB |

## Main Results

Mean accuracy across 3 seeds:

| Model | Evidence | Overall | HellaSwag | PIQA |
|---|---|---:|---:|---:|
| M_evidence | retrieved | 38.57% | 26.66% | 50.49% |
| M_evidence | none | 38.40% | 26.20% | 50.59% |
| M_evidence | shuffled | 38.48% | 26.43% | 50.52% |
| M_evidence | wrong_topic | 38.56% | 26.43% | 50.68% |
| M_evidence | gold | 38.56% | 26.56% | 50.55% |
| M_none | retrieved | 38.69% | 26.46% | 50.91% |
| M_none | none | 38.70% | 26.50% | 50.91% |
| M_none | shuffled | 38.72% | 26.40% | 51.04% |
| M_none | wrong_topic | 38.75% | 26.73% | 50.78% |
| M_none | gold | 38.67% | 26.43% | 50.91% |

Dumb baselines:

| Baseline | Overall | HellaSwag | PIQA |
|---|---:|---:|---:|
| majority_label | 34.91% | 23.14% | 46.68% |
| shortest_candidate | 39.31% | 24.90% | 53.71% |
| unigram_frequency | 38.28% | 25.39% | 51.17% |
| nearest_neighbor_train_label | 38.72% | 25.88% | 51.56% |
| BM25 evidence-overlap ranker | 39.31% | 26.86% | 51.76% |
| gold overlap oracle rule | 37.30% | 31.25% | 43.36% |

## Gate Results

| Gate | Required | Measured | 95% paired bootstrap CI | Verdict |
|---|---:|---:|---:|---|
| INTERNALIZATION: M_evidence(none) - M_none(none) | >= +2.00pp | -0.31pp | [-1.33pp, +0.77pp] | FAIL |
| EVIDENCE_USE: M_evidence(retrieved) - M_none(retrieved) | >= +3.00pp | -0.11pp | [-1.16pp, +0.92pp] | FAIL |
| BASELINE: M_evidence(retrieved) - best dumb | >= +5.00pp | -0.73pp | not paired | FAIL |
| SENSITIVITY: M_evidence(retrieved) - M_evidence(shuffled) | >= +3.00pp | +0.10pp | [-0.20pp, +0.39pp] | FAIL |
| GOLD_CEILING: M_evidence(gold) HellaSwag | >= 35.00% | 26.56% | not paired | FAIL |

The critical gate is internalization. It is negative, so the evidence-native steelman does not survive this run.

## Iteration 69: Internalization Probes

The precondition for deeper representation probes was not met. Because INTERNALIZATION failed, I did not spend additional compute on hidden-representation comparisons, counterfactual flip probes, paraphrase invariance, or span masking.

Probe-like condition evidence from the main controls is already unfavorable:

- Retrieved vs shuffled for M_evidence is only +0.10pp.
- Gold/template rationale evidence is not better than retrieved in aggregate.
- Wrong-topic evidence is effectively tied with retrieved.
- M_none with no evidence is slightly better than M_evidence with no evidence.

## Iteration 70: Synthesis

v1 fixed the most obvious architecture complaint from v0, expanded training to 4K examples, ran the matched twin comparison over 3 seeds, and used an external/rationale corpus instead of raw benchmark-train evidence. The result still does not show evidence-conditioned internalization.

The direction fails the launch rule:

```text
If INTERNALIZATION fails -> evidence-native direction is DEAD.
```

Strictly phrased: **Evidence-native v1 as implemented here is dead as a moonshot mainline.** The caveats are real: this was a 12.17M factorized judge, not a 121M judge, and Qwen-generated rationales were not executed. But the burden was to show that evidence training changes the model even without evidence at test time. The measured sign goes the wrong way.

Recommendation: demote evidence-native to an application-layer RAG/reranking research line unless a future run first supplies a true Qwen/human gold-evidence ceiling and then clears internalization with a much larger model. The mainline should return to chain-init, inherited coordinates, or larger Sutra-family anchors.
