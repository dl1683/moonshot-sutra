# T+L Round 6 Evidence Brief

## 100M Gate Results (the primary R5 experiment)

### Final metrics (step 5000)

| Model | Params | BPT | kurtosis_avg | kurtosis_max | max_act |
|-------|--------|-----|-------------|-------------|---------|
| Transformer 24x512 | 90.2M | 4.8087 | 0.45 | 1.66 | 44.8 |
| **HEMD-R5-G 24x512** | **93.5M** | **4.7019** | **1.93** | **3.94** | **63.6** |

**BPT delta: +0.11** (passes R5 criterion of >=0.10)
**Stability: FAILS** (kurtosis 2.4x worse, max_act 1.4x worse)

### Full trajectory comparison

| Step | T BPT | H BPT | Delta | T kurtosis_max | H kurtosis_max | T max_act | H max_act |
|------|-------|-------|-------|----------------|----------------|-----------|-----------|
| 500 | 8.28 | 8.50 | -0.22 | 0.89 | 0.63 | 16.3 | 26.8 |
| 1000 | 6.78 | 6.67 | +0.11 | 0.46 | 0.32 | 17.4 | 19.9 |
| 1500 | 6.58 | 5.85 | +0.73 | 0.32 | 0.49 | 20.3 | 23.8 |
| 2000 | 6.05 | 5.72 | +0.33 | 0.24 | 0.76 | 18.8 | 31.2 |
| 2500 | 5.81 | 5.81 | 0.00 | 0.34 | 0.51 | 25.4 | 23.6 |
| 3000 | 5.44 | 5.42 | +0.02 | 0.68 | 1.21 | 28.4 | 41.7 |
| 3500 | 5.46 | 5.21 | +0.25 | 0.96 | 1.70 | 41.8 | 45.4 |
| 4000 | 5.07 | 5.04 | +0.03 | 1.60 | 2.19 | 39.2 | 50.0 |
| 4500 | 4.92 | 4.79 | +0.13 | 1.91 | 5.09 | 36.8 | 73.2 |
| 5000 | 4.81 | 4.70 | +0.11 | 1.66 | 3.94 | 44.8 | 63.6 |

### Key observations

1. **Hybrid learns faster early** (step 1000-1500: +0.11 to +0.73 BPT advantage)
2. **Advantage erodes** during mid-training (step 2000-3000: narrows to 0-0.03)
3. **BPT recovers** during cooldown (step 4000-5000: advantage returns to +0.11)
4. **Instability grows monotonically** — hybrid kurtosis goes from 0.32 at step 1000 to 5.09 at step 4500 to 3.94 at step 5000 (partial cooldown recovery)
5. **max_act follows same pattern** — peaks at 73.2 (step 4500), recovers to 63.6
6. **Both models show some instability** in the 3000-4000 window, but transformer recovers better
7. **The kurtosis explosion at step 4500 (5.09)** is concerning for longer training

### What the gate proved

- **Hybrid direction validated**: the normalized additive hybrid genuinely learns more from data
- **Stability mechanism insufficient**: per-branch RMSNorm + per-channel beta is not enough to prevent activation health degradation at 100M scale
- **The instability is in the fusion, not the branches**: comparing to the stable mean-fusion P-block at 42M (kurtosis 0.82), the problem is how branches combine

### Hypothesis for the instability

The per-channel beta vectors are amplifying branch magnitude differences over training. Even though each branch is normalized, the betas can learn to scale one branch much higher than the other, defeating the normalization. This accumulates across 24 layers.

Possible fixes (for R6):
1. **Clamp betas** to [0.5, 2.0] or use softmax normalization on (beta_a, beta_c)
2. **Post-fusion norm** before the output projection
3. **Replace learned betas** with fixed 0.5 scaling (like the P-block that was stable)
4. **Reduce output projection init** — currently 0.02/sqrt(2L), could try smaller
5. **Use SS-RMSNorm for branch norms** instead of full RMSNorm (fewer per-channel scales)

## MiniPLM Pilot Results (O4 parallel work)

200 windows scored, 10% corpus sample, CPU-only.

- Teacher (Qwen3-1.7B-Base): NLL = 2.281 +/- 0.623
- Reference (Qwen3-0.6B-Base): NLL = 2.545 +/- 0.663
- Difference score: 0.264 +/- 0.113

Source ranking by value (diff score):
1. gutenberg: 0.37 (literature)
2. wikipedia: 0.29 (factual)
3. fineweb_s1: 0.28
4. tinystories: 0.24
5. minipile: 0.24
6. openwebmath: 0.23
7. wildchat: 0.21 (conversational)

Infrastructure validated end-to-end. Ready for full corpus scoring.

## Generation Quality (both models at 5K steps, 90-93M)

Both models produce repetitive, low-quality text at this scale/training length.
This is expected and not a differentiator between variants.

## Questions for Round 6

1. Should we fix the stability and rerun the gate, or accept the BPT win and move to production with stability fixes?
2. Which stability fix is most promising? (clamp betas, post-fusion norm, fixed scaling, smaller init)
3. Should we try a mixed schedule (some N blocks, some A blocks) to limit instability?
4. At what point do we stop iterating the gate and move to production training?
5. What is the O4 priority: full MiniPLM corpus scoring, or start production training first?
