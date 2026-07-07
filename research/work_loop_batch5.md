# WORK LOOP - Batch 5: Brainseed Final Diagnostics + Chain-Init Prototype

Date: 2026-07-07

Artifacts:
- `code/brainseed_batch5_diagnostics.py`
- `code/chain_init_codec_probe.py`
- `research/work_loop_batch5.md`
- `C:/sutra_fast/brainseed_batch5/batch5_zero_cost_diagnostics.json`
- `C:/sutra_fast/brainseed_batch5/batch5_slices_diagnostics.json`
- `C:/sutra_fast/brainseed_batch5/batch5_scorers_diagnostics.json`
- `C:/sutra_fast/brainseed_batch5/scorer_cache.pt`
- `C:/sutra_fast/chain_init_probe/chain_init_token_end_layers4.json`
- `C:/sutra_fast/chain_init_probe/chain_init_patch_boundary_layers4.json`

Commands run:

```powershell
python -m py_compile code/brainseed_batch5_diagnostics.py
python code/brainseed_batch5_diagnostics.py --mode zero_cost --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --num-sequences 2 --batch-size 1 --seq-len 64 --device cpu --linear-train-cap 32 --no-artifacts
python code/brainseed_batch5_diagnostics.py --mode zero_cost --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/brainseed_batch5 --num-sequences 64 --batch-size 4 --seq-len 256 --device cuda --json
python code/brainseed_batch5_diagnostics.py --mode slices --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/brainseed_batch5 --num-sequences 64 --batch-size 4 --seq-len 256 --device cuda --json
python code/brainseed_batch5_diagnostics.py --mode scorers --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_brainseed_batch5_smoke --device cuda --teacher-device cuda --extract-hellaswag 2 --extract-piqa 2 --eval-hellaswag 4 --eval-piqa 4 --mlp-epochs 1 --bilinear-epochs 1 --cosine-epochs 1 --progress --json
python code/brainseed_batch5_diagnostics.py --mode scorers --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/brainseed_batch5 --device cuda --teacher-device cuda --extract-hellaswag 512 --extract-piqa 512 --eval-hellaswag 1024 --eval-piqa 1024 --mlp-epochs 120 --bilinear-epochs 120 --cosine-epochs 120 --progress
python -m py_compile code/chain_init_codec_probe.py
python code/chain_init_codec_probe.py --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir tmp_chain_init_smoke --device cuda --layers 2 --readout token_end --num-sequences 2 --seq-len 128 --batch-size 1 --max-positions-per-sequence 32 --json
python code/chain_init_codec_probe.py --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/chain_init_probe --device cuda --layers 4 --readout token_end --num-sequences 32 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --json
python code/chain_init_codec_probe.py --codec-checkpoint C:/sutra_fast/codec_phase1.5/codec_final.pt --output-dir C:/sutra_fast/chain_init_probe --device cuda --layers 4 --readout patch_boundary --num-sequences 32 --seq-len 512 --batch-size 4 --max-positions-per-sequence 128 --json
```

Implementation note: the first scorer smoke exposed a slow full-split `datasets` conversion path. `code/brainseed_batch5_diagnostics.py` now reads the cached Arrow files directly and samples requested rows by index before conversion. That made the 512+512 teacher-margin scorer run practical.

Compute policy honored:
- No codec retraining was run in Track A.
- Iteration 44 chart-quality push was skipped because Iteration 43 did not show any scorer beating codec-only.
- Scorer training was tiny CPU/GPU regression over cached pair features; no training run approached 30 minutes.
- Chain-init prototype loaded Qwen3-0.6B and 4-layer toy reasoners under 12GB VRAM; no Sutra training was launched.

## Executive Verdict

Batch 5 closes Brainseed v0 as a born-knowing path and opens chain-init as the next mainline.

```text
FAIL_ZERO_COST_READOUT_REPAIR
PARTIAL_MIDTOKEN_CHART_SIGNAL_BUT_FIRST_BYTE_WEAK
FAIL_MLP_BILINEAR_COSINE_SCORERS
SKIP_CHART_QUALITY_PUSH_PRECONDITION_FAILED
BRAINSEED_DEAD_AS_BIRTH_ARTIFACT
BRAINSEED_COMPONENT_ONLY_AS_CODEC_DIAGNOSTIC
PASS_CHAIN_INIT_COMPATIBILITY_PROBE_WEAK_SIGNAL
PROMOTE_CHAIN_INIT_TO_BATCH_6_MAINLINE
```

Track A did not find a rescue. The best zero-cost patch-target readout reached 38.94% top-1 on its held-out linear-readout subset, below the precommitted 40% warning line and below the toy-derived ~50% usefulness cliff. Offset slicing shows Phase 1.5 did learn something real at mid-token positions, but the first-byte slice is still only 20.08% top-1. Stronger scorers did worse than codec-only on both HellaSwag and PIQA.

Track B produced a weak but real compatibility signal: copied Qwen layers beat random layers through codec inputs on token-space next-token loss. This is not downstream competence and not byte perplexity, but it is the first positive evidence in this batch that direct weight inheritance through the codec interface carries usable structure.

## Iteration 41: Zero-Cost Baselines

### Register
Run no-retrain alternatives on the Phase 1.5 codec:
- current patch state baseline;
- nearest preceding token-end state;
- local causal mean/max pooling over the last 4/8 byte hidden states;
- previous-token-end alignment embedding plus current raw hidden state with a small ridge linear readout.

### Design-Gate
Passed. The nearest-token-end baseline is reported in two forms:
- against the patch target, which is the drop-in Gate A question;
- against its own previous-token target, which is a sanity control showing the token-end chart is still strong.

The linear readout uses train/eval split over collected patch anchors, not codec training.

### Evidence-Gate
64 sequences, 256 bytes each, 4096 patch records.

| Method | Patch target top-1 | Patch target top-10 | Notes |
|---|---:|---:|---|
| Current patch state | 37.89% | 57.28% | Reproduces Batch 4 Gate A |
| Nearest preceding token-end -> patch target | 18.05% | 21.62% | Does not solve patch target |
| Nearest preceding token-end -> own token | 81.37% | 94.03% | Sanity control only |
| Local mean last 4 | 26.64% | 43.53% | Worse than current patch |
| Local mean last 8 | 13.35% | 26.29% | Worse |
| Local max last 4 | 22.07% | 38.48% | Worse |
| Local max last 8 | 6.49% | 15.89% | Worse |
| Prev token-end + current hidden linear | 38.94% | 54.29% | Best, but held-out subset only |

Precommitted check:

```text
any_zero_cost_patch_top1_ge_40 = FALSE
```

### Narrative Gate
Moving the microphone to the previous token end does not fix the fixed-patch chart. A small readout can squeeze out about +1pp over the current patch state, but not enough to cross the 40% readout-change gate, let alone the toy cliff.

## Iteration 42: Offset-Sliced Analysis

### Register
Slice Phase 1.5 patch chart by token offset, token length, and token-frequency proxy.

### Design-Gate
Passed with one caveat: no tokenizer frequency table is present in the repo. Frequency slices therefore use token-id rank as a proxy: `<1000`, `1000-9999`, and `>=10000`. This is useful but not a corpus-frequency claim.

### Evidence-Gate

Offset slices:

| Slice | N | Top-1 | Top-10 |
|---|---:|---:|---:|
| Offset 0 / first byte | 976 | 20.08% | 34.63% |
| Offset 1 | 850 | 31.65% | 57.18% |
| Offset 2 | 790 | 48.86% | 91.52% |
| Offset 3+ | 1480 | 60.81% | 90.07% |
| Second-to-last byte | 863 | 58.40% | 87.25% |
| Last byte | 970 | 83.20% | 95.67% |

Token length slices:

| Slice | N | Top-1 | Top-10 |
|---|---:|---:|---:|
| 1 byte | 155 | 96.13% | 100.00% |
| 2 bytes | 216 | 65.74% | 85.65% |
| 3 bytes | 570 | 60.70% | 75.61% |
| 4 bytes | 774 | 62.02% | 79.33% |
| 5+ bytes | 2381 | 38.26% | 61.95% |

Token-id frequency proxy slices:

| Slice | N | Top-1 | Top-10 |
|---|---:|---:|---:|
| token id <1000 | 1404 | 54.06% | 64.25% |
| token id 1000-9999 | 1372 | 47.23% | 71.06% |
| token id >=10000 | 1320 | 38.71% | 67.88% |

Precommitted verdict is mixed. The script reports average offset 0/1 top-1 = 25.86%, which technically clears the `>25%` mid-token learning rule. But the adversarial reading is sharper: offset 1 is real at 31.65%, while offset 0 remains weak at 20.08%. Phase 1.5 learned genuine mid-token prediction, but not enough at the hardest earliest position.

### Narrative Gate
The translator is not merely speaking at token ends anymore. But at the first byte of a token, it is still guessing too often for a birth artifact.

## Iteration 43: MLP/Bilinear/Cosine Scorers

### Register
Replace the ridge scorer with:
- MLP over PCA-256 pair features, 256 -> 128 -> 1;
- bilinear context^T W candidate with ranks 32 and 64;
- learned per-dimension weighted cosine.

Train on the same requested budget: 512 HellaSwag train + 512 PIQA train examples scored by Qwen3-0.6B. Evaluate on 1024 HellaSwag validation + 1024 PIQA validation.

### Design-Gate
Passed. The scorer target remains Qwen mean completion log-likelihood. The dataset loader was repaired to sample cached Arrow rows directly rather than converting full splits before selection.

### Evidence-Gate

| Scorer | HellaSwag | PIQA |
|---|---:|---:|
| Codec-only baseline | 28.71% | 51.66% |
| MLP PCA-256 | 24.51% | 50.10% |
| Bilinear rank 32 | 23.44% | 49.41% |
| Bilinear rank 64 | 25.20% | 50.88% |
| Learned weighted cosine | 25.98% | 48.44% |

Best lift over codec-only:

| Benchmark | Best method | Best acc | Codec-only | Delta |
|---|---|---:|---:|---:|
| HellaSwag | learned weighted cosine | 25.98% | 28.71% | -2.73pp |
| PIQA | bilinear rank 64 | 50.88% | 51.66% | -0.78pp |

Precommitted check:

```text
hellaswag_any_scorer_beats_codec_only_by_ge_3pp = FALSE
```

### Narrative Gate
The extraction method was not merely too linear. MLP, bilinear, and learned cosine all lose to the raw codec-only score. That points away from "wrong scorer" and toward "chart/extraction signal is not downstream-usable."

## Iteration 44: Chart Quality Push

### Register
Conditional only: run if any scorer beats codec-only.

### Evidence-Gate
Skipped by precondition. No scorer beat codec-only on HellaSwag or PIQA. Therefore no 10K Phase 1.5 extension, 90% patch fraction run, 256-anchor run, or larger codec run was launched.

Verdict:

```text
SKIP_CHART_PUSH_PRECONDITION_FAILED
```

## Iteration 45: Track A Verdict

### Verdict

```text
BRAINSEED_DEAD_AS_BIRTH_ARTIFACT
BRAINSEED_COMPONENT_ONLY_AS_CODEC_DIAGNOSTIC
```

Brainseed is dead as the mainline born-knowing artifact because all downstream scorers lose to codec-only. The codec remains useful as a diagnostic and possibly as a bridge for chain-init/KD, but the Brainseed v0 extraction path has now failed ridge, MLP, bilinear, and learned-cosine variants.

The strongest surviving Brainseed-adjacent fact is not a birth claim. It is the offset-sliced chart result: Phase 1.5 produces measurable causal mid-token signal. That is useful for designing byte-to-token bridges, not enough to justify another Brainseed extraction batch.

## Iteration 46: Research CBD and Chain-Init Methods

### Register
Read `DEEP_RETHINK.md`, `FIELD_SURVEY_JUNE2026.md`, and `RESEARCH_NOTES.md` for chain-init/KD context.

### Synthesis
CBD works because it preserves coordinate continuity:
- same tokenizer throughout the chain;
- same or closely related architecture family;
- small capacity gaps between anchors;
- student initialized from inherited weights rather than random weights.

The byte-native analog is not "extract a compact scorer." It is:

```text
bytes -> learned adapter/codec -> token-space inherited core -> byte or token-space output head
```

The closest literature route is byteifying a pretrained token model: keep the pretrained transformer core, train a byte encoder/decoder around it, then prune/distill. For Sutra, the current codec can serve as the first adapter test because it already maps byte states to Qwen embedding space.

### Minimum viable prototype
Use the Phase 1.5 codec as a frozen byte-to-Qwen-space adapter. Copy a few Qwen3-0.6B decoder layers into a Qwen-shaped toy reasoner. Compare copied layers against random layers with the same Qwen output head on token-space next-token loss.

## Iteration 47: Design Chain-Init-via-Codec

### Qwen3-0.6B config
Local config:

| Field | Value |
|---|---:|
| hidden_size | 1024 |
| intermediate_size | 3072 |
| num_hidden_layers | 28 |
| attention heads | 16 |
| KV heads | 8 |
| head_dim | 128 |
| vocab_size | 151,936 |
| tied embeddings | true |

### Dimension implications
The codec alignment head outputs 1024-dimensional Qwen embedding-space vectors. That matches Qwen3-0.6B decoder layer width directly.

The existing Sutra Wide7 reasoner is 1152-dimensional, so direct Qwen layer copy into the current Sutra body is impossible without either:
- a new 1024-dimensional Sutra reasoner variant;
- learned 1024 -> 1152 and 1152 -> 1024 adapters around copied blocks;
- or a byteified Qwen-core path where the global core stays Qwen-shaped.

### Simplest viable approach
For the prototype:
- freeze Phase 1.5 codec;
- use token-end or patch-boundary codec states;
- project through `alignment_head` to 1024d;
- scale to Qwen embedding mean norm;
- feed copied first-N Qwen decoder layers;
- use copied Qwen norm + LM head;
- compare to random decoder layers with the same copied output head.

This tests whether direct inherited Qwen layers are more compatible with codec states than random layers. It does not test a byte decoder.

## Iteration 48: Implement Minimal Chain-Init Prototype

### Implement
Added `code/chain_init_codec_probe.py`.

Features:
- loads Phase 1.5 codec frozen;
- loads Qwen3-0.6B;
- builds a Qwen-shaped toy reasoner with configurable copied layer count;
- builds a random-layer control with copied Qwen embedding/norm/LM head;
- evaluates token-end and patch-boundary byte-derived streams;
- reports token-space next-token NLL, PPL, and next-token accuracy;
- writes JSON artifacts under `C:/sutra_fast/chain_init_probe/`.

### Dry-Run / Smoke
2-layer token-end smoke passed. Copied layers beat random by 4.14 nats/token on 58 token predictions. This validated the script and justified the 4-layer evidence pass.

## Iteration 49: Evaluate Chain-Init Signal

### Evidence-Gate
4 copied Qwen layers, 32 held-out byte sequences, 512 bytes each.

Token-end readout:

| Model | NLL | PPL | Next-token acc | Tokens |
|---|---:|---:|---:|---:|
| Identity codec -> Qwen LM head | 55.08 | capped huge | 0.03% | 3825 |
| Random 4 layers + codec input | 17.17 | 28.5M | 0.00% | 3825 |
| Chain-init 4 layers + codec input | 15.52 | 5.49M | 0.03% | 3825 |
| Chain-init 4 layers + true teacher embeddings | 11.94 | 153.9K | 0.78% | 3825 |

Patch-boundary readout:

| Model | NLL | PPL | Next-token acc | Tokens |
|---|---:|---:|---:|---:|
| Identity codec -> Qwen LM head | 39.61 | capped huge | 0.76% | 4063 |
| Random 4 layers + codec input | 18.34 | 91.9M | 0.00% | 4063 |
| Chain-init 4 layers + codec input | 16.62 | 16.45M | 0.00% | 4063 |
| Chain-init 4 layers + true teacher embeddings | 12.52 | 273.2K | 0.91% | 4063 |

Precommitted checks:

```text
token_end:      chain_nll_lt_random_nll = TRUE, delta = -1.65 nats/token
patch_boundary: chain_nll_lt_random_nll = TRUE, delta = -1.72 nats/token
```

### Interpretation
This is a positive compatibility signal, not a capability result. Copied Qwen layers are materially better than random layers when fed codec-derived inputs. However, the absolute NLL is enormous because:
- only the first 4 Qwen layers are used, but the copied LM head expects a full-depth final representation;
- codec vectors are normalized approximations to teacher embeddings;
- patch-boundary streams are not true token sequences;
- no byte decoder or adaptation training is present.

HellaSwag proxy was not run because the truncated-layer perplexity is too high for a multiple-choice result to be interpretable. The correct next Track B step is adapter training / deeper inherited core / byte decoder, not claiming benchmark quality from this prototype.

## Iteration 50: Batch 5 Synthesis

### Track A Verdict
Brainseed v0 is dead as a birth artifact.

Evidence:
- zero-cost readouts fail to reach 40% patch top-1;
- offset slices show the earliest token positions remain weak;
- MLP/bilinear/weighted-cosine scorers all lose to codec-only;
- chart-quality push precondition failed.

Brainseed remains a component only as a codec diagnostic: it tells us where the byte-to-token bridge is weak, but it does not produce useful frozen judgment.

### Track B Verdict
Chain-init via codec is alive as a prototype direction.

Evidence:
- copied Qwen layers beat random layers through codec inputs for both token-end and patch-boundary streams;
- the signal appears before any adapter training;
- true teacher embeddings are still much better than codec inputs, so the next bottleneck is adapter quality and inherited-core depth.

This is not yet downstream signal. It is enough to promote chain-init to the Batch 6 mainline because Track A is now exhausted.

### Batch 6 Recommendation
Mainline Batch 6 should be byteified chain-init, not Brainseed.

Recommended Batch 6 steps:
1. Build a 1024d byteified-Qwen reasoner path: frozen codec/alignment adapter -> copied Qwen blocks -> Qwen LM head.
2. Add a tiny adapter calibration stage from codec outputs to raw Qwen embedding distribution: scale + affine/RMSNorm, then low-rank adapter if needed.
3. Evaluate full-depth or deeper copied-core token-space NLL against random/coreless controls.
4. Add a byte decoder only after token-space compatibility is materially improved.
5. Run a 100-example HellaSwag proxy only when token-space NLL is within a sane range of the truncated teacher-embedding upper bound.

Do not run another Brainseed scorer batch unless chain-init later needs Brainseed as an auxiliary diagnostic.

### Updated Confidence Table

| Claim | Confidence | Update |
|---|---:|---|
| Phase 1.5 chart contains real mid-token signal | Moderate | Offset 1/2/3+ slices support it; first byte remains weak |
| Zero-cost readout change can rescue patch chart | Low | Best patch-target method stayed below 40% |
| Brainseed extraction adds useful downstream judgment | Near zero | Ridge, MLP, bilinear, and learned cosine all lose to codec-only |
| Brainseed should remain mainline | Near zero | No positive downstream result after final diagnostics |
| Codec is useful as byte-to-token bridge | Moderate-high | Chart signal plus chain-init compatibility both support it |
| Direct copied Qwen layers are compatible with codec states | Moderate | Copied layers beat random by ~1.6-1.7 nats/token |
| Current chain-init prototype is benchmark-capable | Very low | Absolute NLL is huge; no byte decoder or adaptation |
| Batch 6 should prioritize byteified chain-init | High | Only surviving path with positive signal and CBD-aligned strategy |

### Honest Gossip-Magazine Headline

**The brain scan failed, but the inherited reflexes twitched.**

Longer version: Brainseed could map where the translator was wrong, but could not make a newborn judge better than the translator alone. The first copied Qwen layers, however, already behaved less randomly when fed byte-codec states. The next moonshot is not printing a seed; it is teaching a byte-native body to use inherited coordinates.