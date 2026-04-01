# T+L R13 — After Cross-Tokenizer Logit KD: What Actually Works?

## Round: 13
## Previous round: R12 (From-Scratch Cross-Tok KD — FAILED)

---

## CRITICAL EVIDENCE (READ ALL BEFORE DESIGNING)

### 1. CROSS-TOKENIZER LOGIT KD IS DEFINITIVELY DEAD

11 experiments across warm-start AND from-scratch, ALL failed:

**Warm-start (8 experiments, all noise-level):**
- TAID, plain KD, reverse KL, AKL, sparse replay, surface matching, KnowledgePorts — all 0.00-0.01 BPT
- The 60K CE-trained student's representations resist teacher reorganization

**From-scratch (4 experiments today, definitive):**
| Config | 3K Eval BPT | vs CE-WSD |
|--------|------------|-----------|
| CE-WSD (no KD, WSD schedule) | **4.9703** | baseline |
| alpha=0.3 KD + WSD | 4.9507 | -0.02 (NOISE) |
| alpha=0.9 tau=1.0 KD | 8.8844 (1K) | +2.10 WORSE |
| alpha=0.9 tau=0.5 KD | 9.0226 (1K) | +2.24 WORSE |

**Root cause:** DSKD ETA cross-tokenizer alignment is too lossy. Only 92.6% vocab overlap, byte-offset alignment introduces mismatches, top-k truncation on shared vocab loses teacher knowledge. The KD signal through this pipeline is more noise than information.

**What this kills:** ALL logit-level KD with cross-tokenizer alignment. No alpha or temperature will fix this.

### 2. THE LITERATURE GAP

The literature gains (Gemma 2 +7.4pp, ACL 2025 +8.0pp) ALL use **same-tokenizer** models. Nobody has demonstrated comparable gains with cross-tokenizer logit KD from scratch. Our failure is consistent with the literature.

### 3. BONUS FINDING: WSD SCHEDULE

WSD scheduling (LR decay in last 20%) gives -0.42 BPT FREE improvement at 3K steps (4.97 vs 5.39). This should be adopted for ALL future training regardless of KD approach.

### 4. WHAT'S NOT DEAD

The logit pathway is dead, but the teacher model itself is useful. Options that bypass the tokenizer mismatch:

**A. Hidden-State KD** — Match student and teacher intermediate representations via learned projections. No vocabulary dependency. Student dim=768, teacher dim=2048, so a linear projection aligns them. Align at byte-offset positions (same alignment infrastructure, but applied to hidden states not logits).

**B. Byte-Level Representation Matching** — Pool both student and teacher hidden states into byte-span representations, then match. The `compute_state_kd_loss` function already exists using CKA on byte-span pools.

**C. Contrastive Distillation** — Instead of matching distributions, use contrastive learning: corresponding student-teacher hidden states at the same position should be closer than non-corresponding ones.

**D. Attention Transfer** — Match attention patterns (not logits or hidden states). Attention maps are vocabulary-independent and capture structural knowledge.

**E. Pre-compute Teacher Features** — Run teacher inference offline and save hidden states. Use them as targets during student training. Eliminates online teacher overhead.

**F. Train Same-Tokenizer Teacher** — Train a larger model (e.g., 400M-800M) with our 16K tokenizer from scratch, then distill into 197M. Expensive but guarantees same-vocab KD works.

### 5. WHAT WE HAVE

- Architecture: Sutra-24A-197M (24L, 768d, 12h GQA, SwiGLU, SS-RMSNorm, exits at 7/15/23)
- Teacher: Qwen3-1.7B-Base (2048d, cached, ~3.4GB VRAM)
- TeacherAdapter class that returns both logits AND hidden_states
- compute_state_kd_loss function (CKA on byte-span pools)
- compute_cross_tok_logit_kd function (DEAD — don't use)
- WSD LR schedule (validated, -0.42 BPT free)
- 22.9B tokens in 246 shards
- Single RTX 5090 (24GB)
- CE-WSD 3K baseline: BPT=4.9703

### 6. USER DIRECTIVE (MANDATORY)
"No small deviations should be considered. If it's knowledge transfer from some of the best models in the world, we should be growing much quicker."

The bar is DECISIVE, VISIBLE growth. Not 0.02 BPT noise.

---

## QUESTIONS FOR CODEX

1. Which of options A-F (or combination) has the highest chance of producing VISIBLE growth (>0.3 BPT improvement over CE-WSD at 3K)?
2. For hidden-state KD: which teacher layers should map to which student layers? Should we use MSE, cosine, or CKA?
3. For hidden-state KD: how to handle the byte-offset alignment for hidden states? Pool to byte spans? Or use the same token-level causal alignment?
4. What alpha to use for hidden-state KD? (logit KD failed at alpha=0.9 and was noise at alpha=0.3)
5. Should we combine hidden-state KD with CE loss, or use it as an auxiliary loss?
6. Can we pre-compute teacher hidden states offline to avoid online teacher overhead?
7. What's the expected BPT improvement based on representation KD literature?
8. Should we train a same-tokenizer teacher instead? Cost-benefit analysis.

Design a CONCRETE protocol with EXACT parameters. No ranges. Include a 3K smoke test config and rationale for each choice.
