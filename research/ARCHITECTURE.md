# Sutra Architecture Reference (Post-Falsification — 2026-03-25)

**Last updated: 2026-03-25 | Update this file whenever the architecture changes.**

This is the single source of truth for ALL architectural design decisions. Every design session MUST read this file. Every component must justify its existence — "inherited from GPT-2" or "everyone does it" is NOT justification.

---

## CRITICAL: Architecture at a Crossroads

**The falsification experiments (F1-F5) completed on 2026-03-25 have challenged core assumptions. The architecture must evolve or be replaced. The 5 OUTCOMES are sacred; the mechanisms are negotiable.**

Three architectures exist:
1. **68M Recurrent** (v0.6.0a/b) — 12-pass state-superposition SSM, GPT-2 tokenizer (50K vocab)
2. **42M P2c Recurrent** (transplanted) — Same core, 16K custom tokenizer (saves 26M params)
3. **51M Dense** (F4 control) — Standard 11-layer transformer, 16K vocab

The falsification data says:
- **Recurrence vs Dense is UNRESOLVED** — Dense runs 10-18x faster, competitive at early training
- **16K tokenizer is the biggest win** — More impactful than any mechanism change
- **Online KD is noise at 68M** — Teacher signals too weak to justify throughput cost
- **Widening helps D8 but not D10/D12** — Bottleneck is architectural, not parametric
- **Elastic compute (D10 < D12) is REAL** — Replicated across all experiments

---

## Active Architectures

### Architecture A: 68M Recurrent (v0.6.0a — training complete, 20K steps)

| Component | Params | % of Total | Justified? |
|-----------|--------|-----------|------------|
| **emb** (50257 × 768) | 38,597,376 | 56.5% | **RESOLVED: Too large. 16K tokenizer adopted.** |
| **stage_bank** (7 × FFN 768→1536→768) | 16,536,583 | 24.2% | **FALSIFIED: 7 banks fragment capacity. Stages never differentiated.** |
| **router** (LocalRouter, window=4, k=8) | 4,723,200 | 6.9% | YES — cross-position communication |
| **scratchpad** (8-slot memory) | 2,373,896 | 3.5% | YES — +3.27% when removed |
| **writer** (BayesianWrite) | 2,360,832 | 3.5% | YES — precision-weighted updates |
| **pos_emb** (2048 × 768) | 1,572,864 | 2.3% | QUESTIONABLE — RoPE might be better |
| **frozen_cache** | 591,168 | 0.9% | **DEAD WEIGHT — delete** |
| **init_mu/init_lam** | 1,181,184 | 1.7% | YES — maps embedding to state space |
| **transition** (SwitchingKernel2) | 258,901 | 0.4% | **FALSIFIED: v0.6.1 showed mode_gate is sequence-global, MI(mode,token)≈0** |
| **gain_probe** | 116,481 | 0.2% | Diagnostic only |
| **LayerNorms** | ~10,759 | 0.0% | YES |
| **TOTAL** | **68,323,243** | | |

**Intelligence fraction: 38.5%** | **Infrastructure/dead weight: 61.5%**

**Best benchmarks (20K steps):** SciQ 48.1%, PIQA 54.5%, LAMBADA 11.2%, ARC-E 31.3%, WinoGrande 51.5%

### Architecture B: 42M P2c Recurrent (16K vocab — active candidate)

Same recurrent core as A but with 16K custom BPE tokenizer. Created via head transplant from v0.6.0a step 5000.

| Component | Params | % of Total | vs 68M |
|-----------|--------|-----------|--------|
| **emb** (16000 × 768) | 12,288,000 | 29.1% | **-68% (was 56.5%)** |
| **stage_bank** (same) | 16,536,583 | 39.2% | Same |
| **router** (same) | 4,723,200 | 11.2% | Same |
| **scratchpad** (same) | 2,373,896 | 5.6% | Same |
| **writer** (same) | 2,360,832 | 5.6% | Same |
| **other** (init, LN, etc.) | 3,891,732 | 9.2% | Same |
| **TOTAL** | **~42,200,000** | | **-38% params** |

**Intelligence fraction: 61.5%** (vs 38.5% on 68M — massive improvement)

**P2c Control Results (1K steps, teacher-free):**
| Step | BPT | D10 | D12 | D8 |
|------|-----|-----|-----|-----|
| 0 | 6.461 | 6.609 | 6.625 | 11.081 |
| 750 | **5.828** | **5.651** | **5.685** | 10.102 |
| 1000 | 5.820 | 6.002 | 6.028 | 10.420 |

**Key finding:** D10=5.65 at P2c step 750 vs D10=6.75 at 68M step 5000. P2c learns 2x faster.

### Architecture C: 51M Dense Transformer (F4 control)

Standard decoder-only transformer. Built to test whether recurrence is even necessary.

| Component | Params | Details |
|-----------|--------|---------|
| Layers | 11 | Standard transformer blocks |
| d_model | 512 | Smaller than recurrent (768) |
| Heads | 8 | Standard multi-head attention |
| FFN | SwiGLU hidden=1536 | 3x expansion |
| Position | RoPE | Rotary embeddings |
| Norm | RMSNorm | Pre-norm |
| Vocab | 16K (tied) | Custom BPE |
| **TOTAL** | **~51M** | |

**Dense Control Results (5K steps, from scratch):**
| Step | CE Loss | tok/s |
|------|---------|-------|
| 0 | 17.65 | 97K |
| 1000 | 11.51 | 95K |
| 3000 | 9.59 | 96K |
| 5000 | **8.99** | **96K** |

**Key finding:** Dense runs at 97K tok/s vs recurrent at ~10K tok/s (10x faster). Inference is 18x faster (1 pass vs 12). **Benchmarks now available — see F4 section below and consolidated table.**

---

## Falsification Results (F1-F5, 2026-03-25)

### F2: Teacher-Free vs KD A/B (1000 steps, 68M)

| Metric | KD-ON | KD-OFF | Delta |
|--------|-------|--------|-------|
| BPT | 6.684 | **6.666** | KD is 0.018 WORSE |
| D10 | 6.682 | **6.663** | KD is 0.019 WORSE |

**VERDICT:** Online KD = noise at 68M. Teacher signals (CKA=0.013) too weak. O4 mechanism FALSIFIED.

### F3: 16K Recurrent (1000 steps, 42M)

| Depth | Start | 1K | Delta |
|-------|-------|----|-------|
| D8 | 11.37 | **5.53** | **-5.84** |
| D10 | 6.76 | **5.22** | **-1.54** |

**VERDICT:** 16K tokenizer = biggest single win. 1.5+ BPT improvement from tokenizer alone.

### F4: Dense Control (5000 steps, 51M, from scratch)

CE=8.99 at 5K steps. 10x training throughput, 18x inference speed. Still improving at 5K.

**Dense Benchmarks (lm-eval, 7 tasks):**
| Task | Dense F4 | P1a-6K | Delta | Winner |
|------|---------|--------|-------|--------|
| ARC-Easy | 30.9% | 32.1% | -1.2% | P1a |
| ARC-Chall (norm) | **21.6%** | 18.2%* | **+3.4%** | Dense |
| HellaSwag | **26.2%** | 25.9% | **+0.3%** | Dense |
| WinoGrande | 49.6% | 51.1% | -1.5% | P1a |
| PIQA | 54.4% | 54.7% | -0.3% | P1a |
| SciQ | 40.5% | 49.0% | -8.5% | P1a (KD) |
| LAMBADA | 2.8% | 6.7% | -3.9% | P1a |

**VERDICT:** Dense beats recurrent on REASONING tasks (ARC-Chall, HellaSwag). Recurrent only wins on KNOWLEDGE tasks where KD provided unfair advantage. At equal wall-clock time, dense gets 10x more training → extrapolated BPT ≈ 4.5. Recurrence thesis seriously challenged.

### F5: Widen-Only (500 steps, 68M)

| Depth | Baseline | Widened | Delta |
|-------|----------|---------|-------|
| D8 | 12.69 | **6.92** | **-5.77** |
| D10 | 6.77 | 6.69 | -0.08 (flat) |

**VERDICT:** More capacity helps D8 (elastic compute) but NOT peak quality. Bottleneck is NOT parametric.

---

## Consolidated Benchmark Table (all versions, latest data)

| Benchmark | v0.5.4-20K | v0.6.0a-20K | v0.6.0b-15K | v0.6.1-1K | P1a-6K | Dense-F4-5K | Pythia-160M |
|-----------|-----------|------------|-----------|----------|--------|------------|------------|
| SciQ | 33.6% | 48.1% | 37.8% | 36.1% | **49.0%** | 40.5% | ~74% |
| PIQA | 54.1% | 54.5% | 52.9% | 54.4% | **54.7%** | 54.4% | ~62% |
| WinoGrande | 49.1% | **51.5%** | 51.2% | 48.9% | 50.8% | 49.6% | ~53% |
| ARC-Easy | 29.7% | 31.3% | 31.0% | 31.1% | **32.1%** | 30.9% | ~43% |
| ARC-Chall (norm) | 20.2% | 17.5% | 18.7% | 16.7% | 18.2%* | **21.6%** | ~28% |
| HellaSwag | 25.8% | 25.7% | 25.9% | 25.8% | 25.8% | **26.2%** | ~30% |
| LAMBADA | 1.8% | **11.2%** | 4.6% | 2.6% | 6.7% | 2.8% | ~32.6% |
| **Best at** | 0/7 | 2/7 | 0/7 | 0/7 | **3/7** | **2/7** | ALL |
| **Throughput** | ~10K | ~10K | ~10K | ~10K | ~2.6K | **97K** | - |
| **Training** | 20K steps | 20K steps | 15K warm | 1K warm | 6K warm+KD | 5K scratch | ~600K |

*P1a ARC-Chall used acc not acc_norm. Dense used acc_norm for fairest comparison.

**Key insight:** Dense F4 at 5K steps from scratch (NO KD, NO warm-start) beats all recurrent models on ARC-Chall and HellaSwag. It loses on knowledge tasks (SciQ -8.5%) where teacher KD gave P1a an advantage. Dense trains at **97K tok/s** — 10x faster than recurrent, 37x faster than P1a with teachers.

---

## Outcome Status (Updated Post-Falsification)

| Outcome | R20 Score | Evidence | What Would RAISE It |
|---------|-----------|----------|---------------------|
| **O1: Intelligence** | 4.5/10 | P1a 6K best on 4/7 but generation still poor. Far below Pythia-160M. | Match Pythia-160M on 4+ tasks, coherent generation |
| **O2: Improvability** | 6.5/10 | Strong trainer diagnosis. No module-swap win demonstrated. | Swap one stage, show improvement without regression |
| **O3: Democratization** | 2.5/10 | Named stages exist. No ABI, no registry, no composition proof. | Any community contributor demo |
| **O4: Data Efficiency** | 4.0/10 | **Online KD falsified at 68M (F2).** Ekalavya v2 Procrustes dead. ALM gives short-horizon lead only. | Working multi-teacher protocol, offline KD, or novel approach |
| **O5: Inference Efficiency** | 5.5/10 | D10<D12 replicated. v0.6.1 tokenwise control falsified. Fixed-depth savings real. | Token-level compute allocation working |

**CRITICAL: These are the scores to IMPROVE. The mechanisms failed, not the goals. Every design proposal must target raising ALL scores.**

---

## Design Decision Audit (Updated)

| Decision | Status | Updated Justification |
|----------|--------|----------------------|
| GPT-2 tokenizer (50K vocab) | **RESOLVED** | Replaced with 16K custom BPE. Saves 26M params. |
| dim=768 | **INHERITED** | Not derived for this scale. Should be questioned. |
| ff_dim=1536 (2x dim) | **INHERITED** | Standard ratio. Not derived. |
| 7 stage graph | **FALSIFIED** | Stages never differentiated. Fragmenting capacity. |
| Top-2 projection | DERIVED | Still justified — bounded compute per pass. |
| Bayesian write | DERIVED | Still justified — precision-weighted updates. |
| Scratchpad (8 slots) | VALIDATED | +3.27% when removed. Load-bearing. |
| LocalRouter (w=4, k=8) | **INHERITED** | Window/k arbitrary. Not tuned. |
| Learned pos embedding | **INHERITED** | RoPE used in F4 dense. Should switch? |
| Weight tying | VALIDATED | Standard, saves 12.3M params (16K) or 38.6M (50K). |
| 12 recurrent passes | **QUESTIONED** | Elastic compute works but 10x throughput cost. Dense competitive. |
| Pheromone routing | **INCONCLUSIVE** | Never isolated. Adds complexity. |
| Controller/transition | **FALSIFIED** | v0.6.1 proved mode_gate is sequence-global. MI(mode,token)≈0. |
| Online KD (Ekalavya) | **FALSIFIED at 68M** | F2: KD-OFF beats KD-ON. May work at different scale. |
| Recurrence thesis | **UNPROVEN** | Dense (F4) competitive at 10x throughput. Not yet falsified but unproven. |

---

## What This File Is For

1. **Design sessions:** Read before every T+L round. Question anything INHERITED/ARBITRARY/FALSIFIED.
2. **Inherited Paradigm Audit:** The Decision Audit makes inherited assumptions visible.
3. **Architecture changes:** When ANY component changes, update this file FIRST.
4. **Falsification tracking:** Record what was tested and what the data says.

## R21 Addendum

This addendum **supersedes** earlier "recurrence vs dense unresolved" language in this file.

R21 selected `EDSR-98M` as the canonical next run: a 16K, 12-layer dense backbone (`d_model=768`, `n_heads=12`, `ff_dim=2048`) with exit heads at layers 4/8/12. The surviving thesis is adaptive depth via successive refinement, not a 12-pass recurrent backbone.
