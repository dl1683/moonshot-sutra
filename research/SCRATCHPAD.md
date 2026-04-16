# Scratchpad

Working space for half-finished thoughts, emerging ideas, and in-progress reasoning. Things move OUT of here into RESEARCH.md (when validated) or get deleted (when falsified).

**Rule:** Nothing here is a conclusion. Everything here is a question or a sketch.

---

## EKALAVYA ITER5 — ACTIVE RUN (2026-04-15)

**Design:** Codex T+L prescribed routing + covering + TAID + uncertainty gating + piecewise decay.
**Config:** results/config_ekalavya_iter5_full_6k.json
**Seed:** best.pt from routing run step 250 (eval BPB=1.418)

### Key Parameters
- TAID: β 0.0→0.8 over 600 steps (geometric interpolation)
- UG: exp 1.0→2.0 over 600 steps, clamp=1.5, renormalize
- Alpha: 0.03 peak, warmup 150 steps, piecewise decay [[150,1.0],[1500,0.5],[3000,0.167],[4500,0.0]]
- Beta: 0.02, anchor-only, zero by step 600
- Teachers: SmolLM2-1.7B (anchor) + Pythia-1.4B (aux)
- Aggregation: anchor-confidence routing (JSD>0.02, aux_cap=0.35)

### Probe Data (250 steps, predecessor to full run)
| Step | BPB | CE | KD | Repr | Grad | Ramp | TAID β | UG | Notes |
|------|-----|-----|-----|------|------|------|--------|-----|-------|
| 10 | 1.388 | 0.962 | 0.162 | 0.878 | 0.34 | 0.06 | 0.01 | 0.99/56% | Below baseline |
| 50 | **1.377** | 0.955 | 0.188 | 0.46 | 0.881 | 0.33 | 0.07 | 0.98/51% | New best |
| 80 | **1.376** | 0.954 | 0.222 | 0.58 | 0.721 | 0.53 | 0.11 | 0.98/53% | -0.054 below baseline |
| 100 | **1.388** | 0.962 | 0.216 | 0.62 | 0.595 | 0.66 | 0.13 | 0.98/51% | AM avg here: 1.461 |
| 130 | **1.397** | 0.968 | 0.299 | 0.56 | 0.416 | 0.86 | 0.17 | 0.98/55% | TAID stable where AM degraded |
| 150 | **1.377** | 0.954 | 0.247 | 1.88 | 0.320 | 0.99 | 0.20 | 0.98/52% | Survives peak where AM collapsed |
| 250 | **1.398** | 0.969 | 0.558 | 0.50 | 0.188 | 1.00 | 0.33 | 0.95/48% | Strong end |
| EVAL | **1.409** | — | — | — | — | — | — | — | Post-training eval, -0.009 below baseline |

**Probe verdict:** MARGINAL but stable. 6K launched based on: final eval below baseline, mechanism stability, KD budget only 13% of routing (real test needs 6K).

### Full 6K Run #5 (CURRENTLY ACTIVE — PID 59068)

Run started from step 250, using step_250.pt checkpoint.

| Step | CE | KD | Repr | BPB | TAID_b | UG | Notes |
|------|-----|-----|------|-----|--------|-----|-------|
| 260 | 0.995 | 0.561 | 0.197 | 1.435 | 0.35 | 0.74/57% | |
| 270 | 0.981 | 0.464 | 0.174 | 1.415 | 0.36 | 0.70/52% | |
| 280 | 0.965 | 0.474 | 0.161 | 1.392 | 0.37 | 0.69/52% | |
| 290 | 0.964 | 0.415 | 0.148 | 1.391 | 0.39 | 0.72/56% | |
| 300 | 0.977 | 0.557 | 0.149 | 1.409 | 0.40 | 0.69/53% | |
| 310 | 0.959 | 0.470 | 0.129 | **1.383** | 0.41 | 0.71/55% | New best |
| 320 | 1.002 | 0.594 | 0.144 | 1.445 | 0.43 | 0.78/61% | Hard batch |
| 330 | 0.988 | 0.489 | 0.124 | 1.425 | 0.44 | 0.72/56% | |
| 340 | 0.965 | 0.577 | 0.120 | 1.392 | 0.45 | 0.70/54% | Near-best |
| 350 | 0.965 | 0.588 | 0.118 | 1.392 | 0.47 | 0.73/56% | |
| 360 | 1.003 | 0.791 | 0.130 | 1.448 | 0.48 | 0.70/54% | Hard batch |
| 370 | 0.905 | 0.754 | 0.138 | **1.305** | 0.49 | 0.74/55% | Easy batch, best ever |
| 380 | 0.972 | 0.610 | 0.108 | 1.402 | 0.51 | 0.66/50% | β crosses 0.50 |
| 390 | 0.998 | 0.828 | 0.115 | 1.440 | 0.52 | 0.70/54% | |
| 400 | 1.011 | 0.886 | 0.114 | 1.459 | 0.53 | 0.70/54% | |
| 410 | 0.992 | 0.651 | 0.110 | 1.432 | 0.55 | 0.69/53% | |
| 420 | 1.001 | 0.939 | 0.106 | 1.444 | 0.56 | 0.70/54% | |
| 430 | 0.988 | 0.778 | 0.106 | 1.425 | 0.57 | 0.69/54% | |
| 440 | 0.968 | 0.707 | **0.097** | 1.397 | 0.59 | 0.68/51% | Repr all-time low |

**Running average BPB (260-440): 1.411.** TAID β crossed 0.50 at step 380 — target now majority-teacher. NO degradation (AM collapsed here). Repr monotonically dropping (0.197→0.097). Step 500 eval imminent — kill criterion: eval BPB > 1.430.

### Kill Criteria
| Step | Kill if eval > | Budget relative to routing |
|------|----------------|---------------------------|
| 500 | 1.430 | 0.9x routing |
| 1000 | 1.430 | 2.6x routing |
| 1500 | 1.420 | 4.0x routing |

Step 500 eval expected ~22:50 EST. Running average suggests pass (mean 1.407 << 1.430 kill).

### 6K Success Criteria (eval BPB)
| Eval Step | DECISIVE | POSITIVE | NEUTRAL | KILL |
|-----------|----------|----------|---------|------|
| 500 | ≤1.40 | 1.40-1.418 | 1.418-1.430 | >1.430 |
| 1000 | ≤1.39 | 1.39-1.410 | 1.410-1.420 | >1.430 |
| 1500 | ≤1.38 | 1.38-1.400 | 1.400-1.420 | >1.420 |
| 3000 | ≤1.36 | 1.36-1.390 | 1.390-1.410 | >1.420 |
| 6000 | ≤1.35 | 1.35-1.380 | 1.380-1.410 | >1.418 |

---

## TAID × UNCERTAINTY GATING — WHY IT WORKS (2026-04-15)

**The mechanism chain at step 280:**
```
teacher_logits → softmax/T → t_probs (256-dim byte distribution)
student_logits → softmax    → s_probs (256-dim byte distribution)

TAID target = s_probs^(1-β) × t_probs^β     [geometric interpolation]
  β=0.37 at step 280 → target is 63% student, 37% teacher

UG gate = t_conf × (1 - s_match)^exp         [per-position weight]
  exp=1.47 at step 280
  t_conf = teacher's max prob (how confident is teacher?)
  s_match = student's prob at teacher's top byte (does student agree?)

KL_loss = KL(student || TAID_target) × gate   [gated divergence]
Total = α_eff × T² × KL_loss                  [scaled contribution]
```

**Why TAID+Routing succeeds where AM fails:**

1. **AM failure:** When teachers disagree, AM creates bimodal targets. Student wastes capacity on mode-covering. KD loss 4x higher, CE unstable.

2. **Routing eliminates inter-teacher conflict:** Anchor-confidence routing picks ONE coherent signal per position. JSD>0.02 gate ensures aux only contributes where it agrees with or improves on anchor.

3. **TAID eliminates capacity gap:** Geometric interpolation `p_TAID = p_student^(1-β) × p_teacher^β` creates targets the student can actually reach. At β=0.37, target is mostly student (63%) with gentle teacher influence (37%).

4. **Uncertainty gating eliminates noise:** Focus KD on ~50% of positions where teacher is confident AND student disagrees. The other 50% would be harmful or uninformative signal.

**Theoretical grounding:** Axiomatic Aggregation (2601.09165) proves geometric mean is one of 3 valid aggregation families satisfying convexity, positivity, weight monotonicity, continuity, temperature coherence. TAID is a parameterized geometric mean.

---

## NEXT EKALAVYA ITERATION — PLANNING (2026-04-15)

### Decision Tree (after step 500 eval)
| Step 500 eval BPB | Action | Rationale |
|--------------------|--------|-----------|
| < 1.410 | EXCELLENT: Continue to 6K, plan 3-teacher iter6 | KD accumulating beyond probe |
| 1.410 - 1.420 | GOOD: Continue, step 1000 is real test | Matching probe, need more budget |
| 1.420 - 1.430 | NEUTRAL: Continue cautiously | At routing baseline, watch for degradation |
| > 1.430 | KILL: T+L redesign session | KD mechanism failing |

### Improvements for Iter6
1. **Teacher caching** — Pre-compute covering byte probs. Eliminates 77% wall-time bottleneck. Config: results/config_cache_3teacher_12k.json. ~15-20h with GPU covering.
2. **GPU scatter_add covering** — Implemented in sutra_dyad.py. Expected ~20-30x speedup. Needs GPU validation.
3. **Qwen3-1.7B as 3rd teacher** — Byte BPB 1.055, near-zero correlation with anchor (0.008). Same hidden dim (2048). 151K vocab.
4. **12K steps** — Proportionally stretched schedules from 6K config.
5. **Config ready:** results/config_ekalavya_iter6_3teacher_12k.json

### 3-Teacher Config Summary
```json
{
  "teachers": [
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "weight": 1.0, "role": "anchor"},
    {"id": "EleutherAI/pythia-1.4b", "weight": 1.0, "role": "aux"},
    {"id": "Qwen/Qwen3-1.7B", "weight": 1.0, "role": "aux"}
  ],
  "teacher_aggregation": "anchor_confidence_routing",
  "teacher_aux_weight_cap": 0.35,
  "use_teacher_cache": true,
  "max_steps": 12000
}
```

**VRAM estimate:** ~13GB (student + 3× 4-bit teachers + covering + optimizer). Safe.

### Qwen3-1.7B Teacher Analysis
| Teacher | Byte BPB | Corr w/ Anchor | Vocab | Hidden |
|---------|----------|----------------|-------|--------|
| SmolLM2-1.7B (anchor) | ~1.05 | — | 49K | 2048 |
| Pythia-1.4B (aux) | ~1.10 | 0.067 | 50K | 2048 |
| Qwen3-1.7B (aux) | 1.055 | 0.008 | 151K | 2048 |

Near-zero correlation = maximally complementary. Different tokenizer captures different byte patterns.

---

## KD CEILING ANALYSIS — Theoretical Limits (2026-04-15)

### Teacher Performance (byte BPB)
| Teacher | Byte BPB | Top-1 Conf | Vocab |
|---------|----------|------------|-------|
| SmolLM2-1.7B | ~1.05 | 43.8% | 49K |
| Pythia-1.4B | ~1.10 | 39.6% | 50K |
| Qwen3-1.7B | 1.055 | 39.8% | 151K |
| Ensemble (oracle) | ~0.95 (est.) | — | — |

### Gap Analysis
- Student baseline (CE-only, step 250 eval): **1.430 BPB**
- Dense baseline (CE-only, 10K steps est.): **~1.20 BPB**
- Student-to-best-teacher gap: 1.430 - 1.05 = **0.380 BPB**
- Cross-arch cross-tokenizer KD in literature: closes **5-20%** of gap
- Conservative estimate: 0.038 BPB improvement → eval ~1.392
- Optimistic (multi-teacher routing): 0.095 BPB → eval ~1.335

### Observed So Far
- Routing run step 250 eval: 1.418 (gain = 0.012 BPB, 3.2% of gap)
- Current run step 310 training: 1.383 (12.4% of gap — but training overstates)
- 6K step 500 eval will be the first real measurement

### The Manifesto Metric
Not absolute BPB, but **BPB-per-training-step** ratio:
- If KD at 3K matches CE at 10K → 3.3x data efficiency → manifesto wins
- If KD at 6K matches CE at 6K → KD is overhead → pivot

### What Would Change the Ceiling?
1. More teachers (3→5): Diminishing returns per teacher, but oracle BPB keeps dropping
2. Better routing: Adaptive aux cap based on agreement quality
3. Multi-teacher representation KD: Currently anchor-only
4. Curriculum-aware KD: Focus signal on student's weakest areas
5. Larger student: 188M → 350M increases absorption capacity

---

## MVG SCOUT EXPERIMENT — PENDING (2026-04-15)

**Config:** results/config_mvg_scout_5k.json
**Status:** Blocked on GPU (training running)
**Purpose:** Test minimum viable generation — can Sutra generate coherent text?
**Note:** zeroth_p4_5k experiment showed BPB 1.887 regression (patch_size=4 vs default 2). Config uses patch_size=4 — may need correction.
