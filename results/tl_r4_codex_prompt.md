This is Tesla+Leibniz Round 4. Read the template at research/TESLA_LEIBNIZ_CODEX_PROMPT.md first — it defines who you are, the 5 outcomes, the questioning principle, and output format.

MANDATORY FIRST STEPS: Read research/VISION.md, research/ARCHITECTURE.md (now includes Round 3 addendum in section 10), research/RESEARCH.md, CLAUDE.md, code/dense_baseline.py. Run nvidia-smi.

## SECTION 4: CONTEXT FOR ROUND 4

### PREVIOUS ROUND OUTPUT (Round 3, verbatim):

**1. Assumption Challenges**
- `42M is the right probe scale.` For: it already killed bad drop-ins cheaply and clearly: DyT lost by +0.75 BPT, TOP lost by +4.61 BPT with kurtosis exploding to 99.3, prior MTP lost by +0.13, prior halting lost by +0.15. Against: several mechanisms in the literature only pay off above this regime: TOP was validated at 340M+, MoR underperforms at 135M, MiniPLM gains get larger at 500M and 1.2B. Resolve with a dual gate: 42M for catastrophic screening, 100M for promotion. Lean: keep 42M, but only as a falsification floor, not as the final selection scale. Confidence: 9/10.
- `All failed auxiliaries are universally dead.` For: three different auxiliary losses have now hurt locally, and TOP was not marginally bad but catastrophically unstable. Against: the same data also supports a capacity-threshold interpretation, because TOP and token-adaptive compute succeed in the literature at larger scales. Distinguish by severity: catastrophic failures at 42M mean "not for the scout"; mild negatives mean "retest at 100M." Lean: TOP, classic MTP, and learned halting are dead for the 100M scout mainline, not necessarily dead forever. Confidence: 8/10.
- `HEMD survives unchanged from Round 2.` For: the broad thesis still lives: local-global mixing is plausible, fixed exits still fit the mission, and the project still needs a modular edge model. Against: the specific Round 2 embodiment is outdated. The field has moved from inter-layer alternation to intra-layer parallel hybrid blocks, and Muon no longer deserves default status. Lean: keep the family, but rewrite it: Hybrid = intra-layer parallel, Elastic = fixed exits only, Memory = deferred, Decoder = retained. Confidence: 7/10.
- `Inter-layer hybrid is the right hybrid.` For: it is simpler, already represented in code, and the live 42M trunk probe can still tell us whether non-attention mixing helps at all. Against: March 2026 evidence is convergent: Falcon-H1 and Hymba both use parallel attention plus SSM inside every block; the systematic analysis says intra-layer beats inter-layer; inter-layer is now the older, weaker baseline. Lean: pivot the architecture hypothesis to intra-layer parallel hybrid. Use the running inter-layer probe only as a lower bound. Confidence: 7/10.
- `Muon should still be the scout optimizer.` For: at 42M, Muon converged much faster early and clearly has real signal up to about step 3000; literature says the efficiency story is real. Against: the only completed local optimizer verdict is negative. AdamW + SS-RMSNorm finished best at 4.91 BPT; Muon + SS-RMSNorm finished worse at 5.01, with 2.7x larger max activations and visible regressions around steps 2000 and 3500-4000. Lean: AdamW for the scout, lower-LR Muon or NorMuon only as side probes. Confidence: 8/10.
- `The scout should still carry memory, teacher ports, or planning losses on day 1.` For: they serve Outcomes 4 and 5 in theory. Against: the project's actual probe pattern is that extra moving parts hurt before the base optimizer, norm, and trunk are stable. The only clearly validated Round 3 additions are SS-RMSNorm and architectural simplification. Lean: the first scout should be brutally simple: data shaping offline, plain NTP online, fixed exits, nothing else. Confidence: 9/10.

**2. Research Requests** (all COMPLETED — results below in NEW EVIDENCE)
- Research NorMuon implementation details for small dense decoders.
- Research the smallest successful intra-layer parallel hybrid blocks that use causal convolution instead of Mamba.
- Research multi-teacher purification for pretraining.
- Research MiniPLM reference-model choice for same-family teacher/reference pairs.

**3. Experiment / Probe Requests** (status below)
- Adopt a two-scale protocol. 42M/5k stays the catastrophic-screen. 100M/5-10k becomes the promotion gate.
- Finish the live trunk-choice probe. (PARTIAL — V1 and V2 done, V3 in progress)
- Run a matched 100M trunk probe: pure transformer vs intra-layer parallel hybrid. (QUEUED after trunk-choice)
- Run a 100M optimizer probe: AdamW vs Muon lr=0.01 vs Muon lr=0.005. (QUEUED)
- Run a 100M data-shaping probe: raw vs filter vs MiniPLM vs both. (QUEUED)
- Run one multi-source probe after plain scout is healthy. (DEFERRED)

**4. Per-Outcome Confidence** (Round 3)
- O1 (Intelligence): 4/10.
- O2 (Improvability): 6/10.
- O3 (Democratization): 5/10.
- O4 (Data Efficiency): 3/10.
- O5 (Inference Efficiency): 5/10 (down from 6/10).

**5-8.** See full Round 3 output at results/tl_round3_output.md and Round 3 addendum in research/ARCHITECTURE.md section 10.

### NEW EVIDENCE SINCE ROUND 3

**A. Trunk-Choice Probe (42M, 5000 steps, inter-layer hybrid)**

| Variant | Block Pattern | Params | Final BPT | Kurtosis Max | Max Act |
|---------|---------------|--------|-----------|-------------|---------|
| V1: pure_transformer | 12x Attention | 49.1M | **4.9131** | 1.1 | 23.9 |
| V2: pure_conv | 12x GatedConv (k=64) | 46.3M | 5.6147 | 2.3 | 30.2 |
| V3: hybrid_3to1 | [H,H,A,H,H,H,A,H,H,H,A,H] | 47.0M | **4.7218** | 1.03 | 19.21 |

Key findings:
- Pure conv (Hyena-style GatedConv, kernel=64) loses to pure transformer by +0.70 BPT at 42M. Conv-only is clearly worse.
- **V3 (inter-layer hybrid) WINS: BPT=4.72 vs V1=4.91 — improvement of 0.19 BPT!**
- V3 also has BETTER activation health: kurtosis_max=1.03 vs V1's 1.11, max_act=19.21 vs V1's 23.91.
- **V3 wins with FEWER parameters** (47.0M vs 49.1M).
- This tests inter-layer hybrid (alternating block types), NOT the intra-layer parallel hybrid proposed in Round 3.
- Conv-only is bad (-0.70 BPT) → conv NEEDS attention for exact retrieval. But hybrid shows attention alone is suboptimal → attention ALSO benefits from conv's efficient local mixing.
- **Strongest positive signal for the hybrid architecture.** Intra-layer parallel (every block has BOTH) should be even stronger since mixing happens at every layer.

**B. NorMuon Algorithm (arxiv:2510.05491, fully characterized)**

Algorithm steps:
1. First-order momentum: M_t = beta1 * M_{t-1} + (1-beta1) * G_t
2. Orthogonalization: O_t = NS5(M_t) (5 Newton-Schulz iterations)
3. Neuron-wise second-order momentum: v_t = beta2 * v_{t-1} + (1-beta2) * mean_cols(O_t . O_t)
4. Row-wise normalization: O_hat_t = O_t / sqrt(v_t + eps)
5. Scaled update: W_{t+1} = W_t - eta*lambda*W_t - eta_hat*O_hat_t where eta_hat = 0.2*eta*sqrt(mn)/||O_hat_t||_F

Key findings:
- Directly fixes the instability we observed: per-neuron normalization prevents the max_act=59.6 spike pattern.
- Memory: m(n+1) per weight matrix — only m extra scalars over Muon. ~50% more efficient than Adam.
- Hyperparameters: beta1=0.95, beta2=0.95, lr=3.6e-4 at 124M, lr=7.5e-4 at 350M.
- Small-scale results: 6% gain at 124M, 15% at 350M, 11.31% at 1.1B over Muon.
- Not yet in any framework — needs custom implementation.

**C. Falcon-H1 0.5B Exact Architecture (from HuggingFace config.json)**

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden dim | 1024 |
| Attention heads | 8 (GQA: 2 KV heads) |
| Mamba heads | 24 |
| Head dim | 64 |
| FFN intermediate | 2048 |
| Mamba d_state | 128 |
| Mamba d_ssm | 1536 |
| Mamba d_conv | **4** |
| Attn:SSM ratio | **1:3** (8:24 heads) |
| muP multipliers | attention_out=0.9375, ssm_out=0.2357, ssm_in=1.25 |
| Tie embeddings | No |

Critical observations:
- Falcon-H1 uses d_conv=4 (Mamba-2 tiny kernel), NOT large-kernel conv like our GatedConv (k=64).
- 1:3 attention:SSM ratio — most computation goes to SSM branch.
- 36 layers at dim=1024 for 0.5B. Much deeper and narrower than our Round 3 spec (14 blocks, dim=768).
- muP-style multipliers for balancing attention vs SSM outputs — critical for training stability.

**D. Hymba 1.5B Architecture (NVIDIA, ICLR 2025)**

| Parameter | Value |
|-----------|-------|
| Layers | 32 |
| Hidden dim | 1600 |
| Attention heads | 25 |
| SSM:Attention parameter ratio | 5:1 |
| Meta tokens | 128 learnable tokens |
| KV sharing | Cross-layer between every 2 consecutive layers |
| Full attention layers | Only 3 (first, middle, last) — rest sliding window |

Combination: Parallel paths -> mean -> output projection. NOT concat-then-project.

**E. Knowledge Purification (Jin et al., Feb 2026)**

- Applies to FINE-TUNING ONLY, NOT pretraining.
- 5 methods: GPT-4 aggregation, Plackett-Luce, PLM classifier, similarity router, RL selection.
- Best: similarity router and RL selection (+5% over naive multi-teacher).
- Key finding: more teachers WITHOUT purification HURTS performance.
- NO pretraining-specific multi-teacher purification exists in literature.
- Implication: for pretraining, one-teacher-per-batch (Round 3 proposal) is still the cleanest approach.

**F. MiniPLM Practical Details (ICLR 2025, fully characterized)**

- Difference Sampling formula: r(x) = log p_teacher(x) / log p_ref(x)
- Reference model: 104M trained on 5B tokens (SMALLER than all students in the paper).
- Works CROSS-FAMILY: Qwen teacher/ref -> Llama and Mamba students confirmed.
- All data scored UPFRONT (no incremental option). Top-50% selected by score.
- 200M student: 41.3% vs 39.9% baseline (+1.4% avg accuracy on 9 downstream tasks).
- Our 22.9B token corpus feasible to score in ~1-2 hours on RTX 5090.
- Question: should reference model be ~100M (matches paper) or Qwen3-0.6B (simpler but 6x larger)?

**G. Design Tension: Depth vs Width at 100M**

Round 3 spec: 14 blocks x dim=768 = ~100M.
Falcon-H1 scaling: 36 layers x dim=1024 = 0.5B.

If we scale down to 100M:
- Option A: 14 blocks x dim=768 (Round 3 spec — wider, shallower)
- Option B: 24 blocks x dim=512 (deeper, narrower — matches hybrid paradigm)
- Option C: 18 blocks x dim=640 (balanced)

Both Falcon-H1 and Hymba favor DEPTH for hybrids.

### QUESTIONS FOR THIS ROUND

1. Given Falcon-H1 d_conv=4 (Mamba-2) vs our GatedConv k=64 (Hyena-style): which conv variant should the parallel hybrid use at 100M?

2. Hybrid block combination: concat-then-project (Round 3), mean (Hymba), or head-interleaved (Falcon-H1)?

3. Depth vs width for the 100M hybrid: 14x768, 24x512, 18x640, or something else?

4. Should the 100M optimizer probe include NorMuon alongside Muon lr=0.01?

5. MiniPLM reference model: train a quick 100M on 5B tokens, or use Qwen3-0.6B?

6. muP multipliers for attention vs conv outputs — incorporate at 100M?

7. Conv-only lost by +0.70 BPT — does this change the attention:conv ratio from Round 3's 1:2?

8. Should we wait for V3 trunk-choice results or is the intra-layer pivot already clear enough?

## SECTION 5: YOUR TASK

Follow the output format from research/TESLA_LEIBNIZ_CODEX_PROMPT.md section 5.
