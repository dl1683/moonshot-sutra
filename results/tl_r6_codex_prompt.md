MANDATORY FIRST STEP: Read CLAUDE.md in this repository root. Every rule there is binding.
Then read research/VISION.md — especially the "Design Philosophy" section at the top.
Then read research/TESLA_LEIBNIZ_CODEX_PROMPT.md — it defines your role, your questioning principle, your output format, and how you must think. Follow it exactly.
Then read research/ARCHITECTURE.md — it contains the full architecture evolution through 5 rounds plus the new R6 addendum with 100M gate results.
Then read research/RESEARCH.md — it contains all field research, probe results, and dead ends.

CONTEXT YOU MUST KNOW:
- DESIGN PHILOSOPHY: Nothing is sacred except outcomes. 5 sacred OUTCOMES, mechanisms negotiable.
- We are in Round 6 of the Tesla+Leibniz architecture design loop.
- Round 5 (your previous round's output) is pasted verbatim below.
- New findings since Round 5 are listed after the Round 5 output.

---

## YOUR PREVIOUS ROUND'S OUTPUT (Round 5, Verbatim)

(Read the file results/tl_round5_output.md for the full R5 output. It is 95 lines containing: 6 assumption challenges, 2 research requests, 3 probe requests, confidence scores [5/6/5/3/5], intuitions, and the HEMD-R5-G design proposal with block math.)

---

## NEW FINDINGS SINCE ROUND 5

### Finding 1: 100M Gate Results (THE PRIMARY EXPERIMENT)

The R5 promotion gate was run: 24×512 pure transformer vs 24×512 HEMD-R5-G hybrid. Both at SS-RMSNorm, plain NTP, exits 8/16/24, 5K steps, seq 512, AdamW.

**Final metrics (step 5000):**

| Model | Params | BPT | kurtosis_avg | kurtosis_max | max_act |
|-------|--------|-----|-------------|-------------|---------|
| Transformer 24×512 | 90.2M | 4.8087 | 0.45 | 1.66 | 44.8 |
| **HEMD-R5-G 24×512** | **93.5M** | **4.7019** | **1.93** | **3.94** | **63.6** |

**BPT delta: +0.11** (passes R5 criterion of >=0.10)
**Stability: FAILS** (kurtosis 2.4x worse, max_act 1.4x worse)

**Full trajectory:**

| Step | T BPT | H BPT | Delta | T kurtosis | H kurtosis | T max_act | H max_act |
|------|-------|-------|-------|-----------|-----------|----------|----------|
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

**Key trajectory observations:**
1. Hybrid learns faster early (step 1000-1500: up to +0.73 BPT)
2. Advantage erodes mid-training (step 2000-3000: 0-0.03)
3. BPT advantage recovers during cooldown (step 4500-5000: +0.11-0.13)
4. Instability grows monotonically — kurtosis peaks at 5.09 (step 4500) before partial recovery to 3.94
5. Transformer also shows instability (kurtosis 1.91 at step 4500) but recovers better
6. max_act follows same pattern — hybrid peaks at 73.2, recovers to 63.6 vs transformer's 44.8

**Comparison with prior probes:**
- 42M mean-fusion P-block (9 layers): kurtosis 0.82, max_act 18.15 at step 5000. MUCH more stable.
- 42M concat C-blocks (12 layers): kurtosis 1.4-5.8 at step 5000. Similar instability pattern.
- The N-block (normalized additive) is between P and C on stability: better than concat, worse than mean.

**Hypothesis for instability:** The per-channel β vectors are the problem. Even though each branch is normalized by RMSNorm before β scaling, the β values can grow unbounded over training, defeating the normalization. This accumulates across 24 layers. The P-block (fixed 0.5 scaling, no learned betas) was stable at 42M.

### Finding 2: MiniPLM Pilot Complete (O4 Parallel Work)

200 windows scored on CPU. Qwen3-1.7B-Base teacher, Qwen3-0.6B-Base reference.

| Metric | Value |
|--------|-------|
| Ref NLL (mean ± std) | 2.545 ± 0.663 |
| Teacher NLL (mean ± std) | 2.281 ± 0.623 |
| Diff score (mean ± std) | 0.264 ± 0.113 |
| Top-50% threshold | 0.257 |

Source ranking by value: gutenberg (0.37) > wikipedia (0.29) > fineweb (0.28) > tinystories (0.24) > minipile (0.24) > openwebmath (0.23) > wildchat (0.21).

Infrastructure validated. Pipeline works end-to-end. Ready for full corpus scoring.

### Finding 3: HEMD-R5-G Implementation Details

The actual implemented block has these per-block params at dim=512:
- Attention (GQA 4Q/2KV): 0.33M
- Conv path: 0.33M
- Branch projections + output: 0.52M
- FFN (SwiGLU 1536): 2.36M
- Norms + betas: negligible
- Total per block: 3.54M
- Total model (24 layers): 93.5M (vs R5 estimate of ~90M — 3.5M from branch projections)

The branch norms are full RMSNorm (per-channel weight vector), not SS-RMSNorm (single scalar). This gives each branch its own per-channel normalization.

### Finding 4: Generation Quality at 5K Steps

Both models produce repetitive, low-quality text at this scale/training. Not a differentiator. Expected at 90-93M / 5K steps.

---

## SPECIFIC QUESTIONS FOR THIS ROUND

1. The hybrid wins on BPT but fails on stability. Should we: (a) fix stability and rerun the gate, (b) accept the BPT win and move directly to production training with stability fixes built in, or (c) something else?

2. The per-channel β vectors are the likely instability source. Best fix: (a) clamp betas to [0.5, 2.0], (b) softmax over (beta_a, beta_c) so they sum to 1, (c) post-fusion norm before output projection, (d) remove betas entirely (fixed 0.5 like P-block), (e) use SS-RMSNorm for branch norms instead of full RMSNorm?

3. The P-block was stable at 42M but uses full-dim branches (more params per layer). The N-block uses asymmetric branches (d=256 projected to 512). Could the projection itself be a stability problem? Should we test a "P-block with GQA" variant?

4. At what confidence level should we stop iterating the gate and commit to production training? Current confidence is 5/10 on O1. What would get us to 7/10?

5. O4 priority: the MiniPLM pilot worked. Should we (a) score the full corpus before production training, (b) start production training now and score in parallel, or (c) train a custom 100M Qwen-tokenizer reference first (MiniPLM paper approach)?
