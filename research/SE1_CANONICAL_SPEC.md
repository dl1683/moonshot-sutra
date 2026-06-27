# SE1 Canonical Build Specification

This document is the canonical, unified SE1 spec. It reconciles
`research/EKLAVYA_DOCTRINE.md` (concrete SE1 substrate and mechanics) with
`research/GROUND_UP_FUTURE_DESIGN.md` (first-principles protocol and runtime identity).

Status: FROZEN for S0 — R1-R6 complete. Architecture converged. Remaining open
parameters are empirical (determined by runs, not more doctrine).

## Provenance

This spec was produced through multi-round adversarial deliberation.
Key convergence decisions are cited by round.

## 1. What Eklavya Is

Eklavya is a gap-driven lesson protocol. It extracts transferable invariants from
diverse teacher models and compiles them into typed lesson packets that target
specific modules of the student architecture.

Key properties (from DOCTRINE, validated by GROUND_UP):
- Teachers are instruments/sensors, not masters
- The unit of transfer is an invariant, not an output distribution
- Teacher identity (axis) must be preserved until routing and loss application
- Invariants are defined by cross-teacher agreement under perturbation
- Success = retained gain after teacher removal

Key properties adopted from GROUND_UP:
- The core loop is: gap → teacher → probe → lesson → local update → ownership → keep/delete
- No teacher lesson is admitted until a student gap exists (R2 revision: teacher
  profiling can be parallel; teacher lesson admission must be gap-first)
- Each lesson must be minimal — the smallest intervention that changes learner
  behavior under teacher removal with bounded cost

## 2. What Sutra Is

Sutra is a compiled adaptive cognition runtime, not just model weights.

From DOCTRINE: Sutra is a byte-local/patch-global student with local byte encoder,
global reasoner, local byte decoder, and compute controller.

From GROUND_UP (adopted R1): Sutra is a versioned runtime containing weights,
adapters, memory, policies, validators, lesson ledger, and profile cache. The
revision identity includes all of these.

Reconciliation (R2): For the first build, DOCTRINE's model substrate is the
physical architecture. GROUND_UP's revision manifest is the identity model.
Weights alone are not Sutra — but the full runtime need not be sophisticated
at first.

## 3. Build Order

Merged from R2 deliberation:

```text
D0: source/task/validator admission
    - Common Pile license filtering (Public Domain/CC0)
    - Source attribution distribution policy
    - Scout slice preparation

S0: traceable byte/patch substrate scout (121.7M params)
    - Byte-local encoder + patch-global reasoner + byte-local decoder
    - Fixed compute path (no adaptive compute yet)
    - Purpose: byte/patch tradeoff, traceability, basic teacher-free competence
    - Hardware: single RTX 5090, 24GB VRAM

T0: parallel teacher feasibility profiling
    - Teacher cost measurement
    - Surface schema testing
    - Probe plumbing
    - Hardware admission
    - ALLOWED before student gaps
    - NOT ALLOWED: claiming lessons, compiling packets, allocating training weight

G0: gap mapping on real student traces
    - Requires S0 to have trained enough to expose failures
    - Gap = observed student failure on admitted evaluation slices
    - Gap is NOT a design hypothesis — it is an empirical measurement

E1: single-teacher byte-level distillation (anchor decoder 1.7B → S0)
    - Embedding alignment: ByteEncoder patch states → teacher static embeddings
    - Byte KL: first-byte marginals at patch position 0, gap-selected
    - 3-phase schedule: projection warmup → alignment landing → full KD
    - See research/EKLAVYA_E1_PROTOCOL.md for full protocol
    - Code: code/eklavya_cache.py, code/eklavya_training.py

E2: multi-teacher byte-level distillation (5 teachers → S0)
    - PL-style router + arithmetic/log-pool purifier
    - Per-teacher projection ports with warm-start from E1
    - 5-phase curriculum: port warmup → consensus → semantic → disagreement → ownership
    - Multi-teacher gradient budget (per-teacher 0.10, total 0.30)
    - See research/EKLAVYA_E2_PROTOCOL.md for full protocol
    - Code: code/eklavya_e2_cache.py, code/eklavya_e2_router.py,
            code/eklavya_e2_losses.py, code/eklavya_e2_training.py

P0: packet compilation for observed gaps only
    - Each packet targets a named gap
    - Each packet has a named landing zone
    - Each packet has a teacher source identity

G1: integrated runtime (350-500M params)
    - Full 7-interface skeleton with minimal implementations
    - All 7 interfaces present but simple
    - Only proceed if S0 passes basic competence checks

O0: ownership/credit/efficiency gates
    - Teacher-free retained gain evaluation
    - Module credit assignment
    - Normalized efficiency reporting
```

## 4. Staged Bootstrap, Cyclic Improvement

R1/R2 convergence:

Before a traceable student exists:
- Define expected failure classes and validators
- Profile teacher feasibility in parallel
- Cannot claim Eklavya gap evidence

After a traceable student exists:
- Every lesson must be gap-driven
- Gap → teacher → probe → lesson → local update → ownership → keep/delete
- This is the steady-state protocol from GROUND_UP

Rule: **staged bootstrap, cyclic improvement after first observable student.**

## 5. Teacher Admission

R2 convergence:

Allowed BEFORE student gaps:
- Teacher feasibility profiling
- Teacher cost measurement
- Surface schema testing
- Probe plumbing
- Hardware admission
- Source/license checks

Forbidden BEFORE student gaps:
- Claiming a lesson
- Compiling final packets
- Allocating training weight to teacher signal
- Promoting a teacher as useful

Rule: **teacher profiling can be parallel; teacher lesson admission must be gap-first.**

## 6. Interface Skeleton

R2 convergence, revised R4-R5: all 7 interfaces present in G1. See Section 9
for concrete S0 implementations and parameter counts.

```text
I0 input_surface:    byte encoder (2-layer local mixer + nonlinear patch aggregator)
I1 compact_state:    lower 8-10 transformer blocks + pooled state heads
I2 reasoning_step:   upper 20-24 transformer blocks + learned scratch tokens
I3 verification:     triage head at S0 (2-6M); grows to 20-40M at G1; routes to
                     exact validators for authority decisions
I4 memory:           non-parametric ANN index (HNSW/FAISS) over lesson ledger,
                     teacher traces, and failure records; read-only at S0
I5 compute_governor: rule-based at S0; early exit + MoD + self-verification at G1
I6 output_surface:   local autoregressive byte decoder with causal cross-attention
                     to nearby patch states

Update ports:        LoRA/adapters on I0/I3/I4/I5/I6 only at first
```

Deferred from G1:
- Autonomous memory writes (I4 write gate)
- Sleep/precompute mode
- Learned long-horizon compute policy
- Deployment profile machinery

Rule: **full interface skeleton in G1; minimal implementations; defer advanced runtime behavior.**

## 7. Scale

R2 convergence:

```text
S0 substrate scout:  ~122M params (30x576 deep-thin default)
    Purpose: byte/patch tradeoff, traceability, basic teacher-free competence
    NOT for proving the full runtime thesis
    121.7M for frozen dimensions (30x576 deep-thin). Earlier 180-250M estimate
    was arithmetic error — do not inflate to hit old target.

G1 integrated build:  350-500M params
    Purpose: enough capacity for all 7 interfaces, multiple lesson types,
             verifier, memory query, compute governor, adapters
    Only proceed if S0 passes
```

Rule: **do not use 485M for the first substrate scout; do not expect 180M to prove the full runtime thesis.**

## 8. Patch Size

DOCTRINE scouts P4 and P8 as first two fixed-patch falsifiers.
GROUND_UP agrees on P4/P8.

Decision: scout P4 (high-resolution control) and P8 (strong decoder test) as the
first two fixed-patch variants. Adaptive/learned patching is deferred to post-S0.

R3 literature review:
- BLT: P6/P8 scaling is strong but reported at 400M+. P8 starts worse at smaller
  scale and improves as model/data scale grow.
- MEGABYTE: validates fixed patching + local/global split, but uses P8 with 758M
  global + 262M local model, not a 200M scout.
- Charformer: at 134-206M, downsampling 2/3 beats 4 in reported tables. Argues
  against over-compressing the first scout.

Revised decision: P4 as primary S0 architecture, P8 as falsifier/control.
P6 as engineering compromise if budget allows three runs.

## 9. S0 Concrete Architecture (R3, revised R4)

Based on R3 deliberation with BLT/MEGABYTE/Charformer literature review.
R4 stress-tested VRAM, encoder, decoder, verifier, and memory specs.

### VRAM Budget (R4 validated)

```text
200M params, bf16, AdamW:
  weights bf16:       200M × 2  = 0.40 GB
  gradients bf16:     200M × 2  = 0.40 GB
  Adam m/v fp32:      200M × 8  = 1.60 GB
  fp32 master copy:   200M × 4  = 0.80 GB
  param total:                    3.20 GB

Activations (FlashAttention + checkpointing, seq=1024):
  microbatch 16:  ~3.8-6.4 GiB (depends on checkpoint granularity)
  microbatch 32:  ~4.7-9.8 GiB

Effective batch ≥64 via microbatch 16 × grad accum 4. Fits 24GB GPU.
If tight, reduce microbatch first — model size and context are not the bottleneck.
```

### Byte Encoder (~3.4M params)

R4 correction: single Conv1d(k=5) is too weak. At P4, kernel=5 barely covers one
UTF-8 codepoint plus boundary. Misses escapes, indentation, identifier fragments,
byte-pair-like local morphology. Upgraded to local transformer/conv-attention mixer.

```python
byte_emb = nn.Embedding(260, 256)
local_mixer = LocalByteMixer(
    d_model=256, n_layers=2, kernel_or_window=8,
    # 2-layer windowed attention or depthwise-conv + cross-attention hybrid
    # Forbidden: global attention, teacher lookup, hidden reasoning
)
# R5: nonlinear patch aggregation replaces bare Linear projection
patch_agg = PatchAggregator(
    # gated MLP or local latent cross-attention over byte states + boundary features
    byte_dim=256, patch_size=P, out_dim=D,  # P=4, D=576 (default) or 768 (ablation)
)
entropy_head = nn.Sequential(nn.Linear(D, 256), nn.GELU(), nn.Linear(256, 1))
residual_head = nn.Linear(256, 1)
```

Raw bytes, patch spans, and byte side state preserved through the pipeline.

### Global Reasoner (~100-130M params)

Standard causal Transformer first. SSM/hybrid deferred to post-S0.

R5 revision: MobileLLM (2024) showed deep-thin beats wide-shallow at sub-billion
scale. Their 125M: d=576, 30 layers. Their 350M: d=960, 32 layers. Deep-thin
gives more reasoning depth at matched FLOPs (30×576 ≈ 18×768 in FLOPs but with
+12 layers of iterative refinement).

| Field | Default (R5) | Ablation |
|---|---|---|
| D | 576 | 768 |
| layers | 30-32 | 22 |
| heads | 9 | 12 |
| kv_heads | 3 (GQA) | 4 (GQA) |
| FFN | SwiGLU, ~2.66x hidden (1536) | SwiGLU (~2048) |
| context | train 4096 bytes/P4 = 1024 patches; eval up to 8192 bytes | same |
| memory | bf16, FlashAttention, activation checkpointing, grad accumulation | same |

If P8 suffers from narrow dimension, try 30×640 before reverting to shallow-wide.

Layer sharing (MobileLLM-LS): secondary ablation only, not primary S0. Sharing
upper reasoning blocks only; forbidden for I0/local encoder, first 4 blocks,
and final 2-4 output-facing blocks. Reason: byte/patch models need module
specialization for lesson localization and credit assignment.

### Byte Decoder (~10.5M params)

Autoregressive over bytes inside each patch. Teacher-forced during training;
byte-by-byte generation within current patch at inference.

R9/R10 causal alignment: hidden[i] predicts bytes of patch i+1 (not patch i).
This ensures no target leakage — the conditioning hidden state never encodes
the bytes being predicted. Logits shape is (B, N-1, P, 256).

```python
prev_byte_emb = nn.Embedding(260, 256)
cond = nn.Linear(D, d_local)  # initial conditioning
local_decoder = TinyCausalTransformer(
    d_model=384, n_layers=4, n_heads=6, context=P,
    # R5: add local causal cross-attention to current + 1-2 nearby patch states
    # + byte-side residual/context states from encoder
    cross_attn_to_patch_states=True,
)
lm_head = nn.Linear(384, 256)
```

R5 decoder conditioning: bare `Linear(D, d_local)` is a baseline only. The
canonical S0 decoder adds small causal cross-attention to current and nearby
patch states (~0.7-0.9M params per cross-attn module). Full global-sequence
cross-attention at every byte is NOT default — that weakens the patch
compression test.

R4 constraint: P4 vs P8 comparison MUST use identical decoder architecture and
capacity. If P8 only wins with a stronger decoder, the win belongs to decoder
capacity, not patch size. Experiment matrix:

```text
primary:    P4 same decoder, P8 same decoder
secondary:  P8 stronger decoder, P4 same stronger decoder (compute-matched control)
```

### All 7 Interface Specs (G1 minimal, R4-revised)

| Interface | Implementation | Params |
|---|---|---:|
| I0 input_surface | Embedding + local mixer (2-layer windowed attn/conv) + chunker + proj | 18-30M |
| I1 compact_state | Lower 8-10 Transformer blocks + pooled state heads (deep-thin) | 30-45M |
| I2 reasoning_step | Upper 20-24 Transformer blocks + learned scratch tokens (deep-thin) | 70-90M |
| I3 verification | Triage head only (see below); serious verification via exact validators | 2-6M (S0), 20-40M (G1) |
| I4 memory | Query/key projections + non-parametric ANN index (see below) | 1-4M trainable (S0) |
| I5 compute_governor | Rule-based first, then small MLP policy over entropy/margin/risk | <1-3M |
| I6 output_surface | Local autoregressive byte decoder, 3-4 tiny Transformer layers | 25-45M |
| Update ports | LoRA/adapters on I0/I3/I4/I5/I6 only at first | 5-15M |

### I3 Verification Scope (R4 revision)

At S0, I3 is a triage/routing head — NOT a real verifier for code or math:

```text
I3 CAN do (triage):           I3 CANNOT do (authority):
  pass/fail/unknown verdict     syntactic validity without parser
  error span guess              proof correctness
  repair class routing          semantic code correctness
  escalation decision           any verification requiring execution
  "call exact validator" flag
```

Real verification requires exact validators, execution, parsers, schema checkers,
or teacher-generated verifier traces. At G1, I3 grows to 20-40M and gains
structured verification heads, but still routes to external validators for
authority decisions.

### I4 Memory Specification (R4 revision)

I4 must NOT be just attention over a learned embedding table — that is parameters
with an awkward interface, not memory. A real I4 requires:

```text
index:     non-parametric ANN index (HNSW/FAISS-style vector index)
content:   lesson ledger entries, teacher traces, exact-oracle cases,
           failure/repair records, provenance-stamped snippets
           NO hidden web corpus unless explicitly admitted

trainable: query adapter, key/value projection adapter,
           reranker, conflict detector, write gate

outputs:   retrieved records, confidence, conflict set,
           provenance, write/no-write decision
```

I4 differs from more layers because it retrieves mutable, provenance-bearing
records not stored in weights. At S0, I4 is read-only with a minimal index
over lesson ledger entries. At G1, write gates and conflict detection activate.

### Non-KD Identity: Concrete Example (R3)

Boundary lesson: student confuses `model_id` with `rnodel_id` or `O_RDONLY` with
`0_RDONLY`.

Vanilla KD: add more examples, distill teacher logits over final strings.

Eklavya packet:
- lesson_type: boundary_lesson
- target_invariant: preserve byte-level identity of confusable spans
- teacher_sources: exact string oracle + verifier teacher + decoder margin teacher
- landing_zones: I0, I6, I3, I5

Losses applied per landing zone:
- I0: BCE(residual_flag_pred, residual_flag_target)
- I6: weighted CE over exact bytes in confusable span
- I6/I2: contrastive margin: max(0, margin - logp(correct) + logp(confusable))
- I3: CE(verdict), CE(error_span), CE(repair_class)
- I5: CE(action = verify_or_repair) + lambda * compute_cost

Ownership test:
1. Teacher removed — no teacher logits or packet lookup at inference
2. Test on NEW unseen confusable identifiers/paths/schemas
3. Required: exact byte accuracy up, logp margin up, I0 residual flags activate,
   I3 localizes error, I5 escalates only when warranted
4. Controls: plain CE fine-tune, naive KD on final strings, global-only update
5. Pass only if targeted route beats controls AND ablating local adapters removes
   the specific gain

## 10. Non-KD Identity Summary

R2 convergence: the system degenerates into vanilla multi-teacher KD if these
are removed:

```text
REQUIRED for non-KD identity:
  teacher axis preserved (no early averaging)
  named landing zones (packets target specific modules)
  local update ports (constrained adapters, not global finetuning)
  teacher-free ownership tests (retained gain under teacher removal)
  verifier or exact-check lesson path
  compute-governor lesson path
  memory/retrieval lesson path (even if read-only)
```

## 11. Utility and Metrics

R1 convergence: GROUND_UP's normalized utility replaces DOCTRINE's raw
net_retained_gain scalar.

Rule: raw capability, dollars, FLOPs, latency, and damage must not be subtracted
without normalization. DOCTRINE's formula is a checklist, not a scalar metric.

Required metrics before any efficiency claim:
- p95 FLOPs/token
- mean FLOPs/token
- hard-case quality delta (bottom-10% by verifier confidence)
- easy-case speedup
- total quality retention vs fixed-compute baseline
- update locality after compute policy changes

## 12. Terminology

| DOCTRINE term | GROUND_UP term | Canonical term |
|---|---|---|
| invariant packet | lesson packet | lesson packet |
| teacher family | teacher measurement function | teacher measurement function |
| teacher signature tensor | teacher surface schema | teacher surface |
| landing zone | Sutra interface + update port | landing zone |
| global reasoner | I1 compact_state + I2 reasoning_step | global reasoner |
| local byte encoder/decoder | I0 input_surface + I6 output_surface | byte encoder/decoder |
| compute controller | compute governor | compute governor |
| verifier head | I3 verification | verifier |
| packet ledger | lesson ledger | lesson ledger |
| retained gain | ownership | retained gain (ownership) |
| promotion gates G0-G5 | G1 acceptance gates | promotion gates |
| net_retained_gain | realized_lesson_value | normalized retained gain |

## 13. What's Kept and Discarded

### Kept from DOCTRINE
- SE1 byte-preserving patch-global substrate
- Teacher axis and source-specific accounting
- Concrete packet payloads and landing-zone losses
- Comparison suite and retained-gain controls
- Patch-size scout doctrine (P4/P8)
- Readiness boundary: no training until slices, teachers, thresholds, and controls exist
- Hardware-realistic first scale (121.7M for S0)

### Discarded from DOCTRINE
- Named teacher roster as canonical (keep as candidate examples only)
- Raw net_retained_gain scalar
- Teacher profiling not anchored to observed student gaps
- "David story" / world-class rhetoric
- Parameter priors as hard commitments

### Kept from GROUND_UP
- Sutra as versioned runtime, not just weights
- Eklavya as gap-driven lesson protocol
- Teacher/tool/oracle boundary
- Causal lesson inference ladder
- Update port doctrine and rollback
- Module credit assignment
- Normalized efficiency reporting
- Failure semantics and reset criteria

### Discarded/Deferred from GROUND_UP
- "Ignore inherited SE1" as permanent rule (revised to: "do not inherit uncritically")
- Sleep/precompute mode for G1
- Full memory-write/update ecosystem before basic ownership is proven
- 485M reference build (unless justified by measured need at G1)
- H0-H8 as blocking bureaucracy (converted to minimum build gates)
- Self-declared "no remaining doctrine flaws" audit

## 14. Source Admission Summary

Based on license research (2026-06-27):

### Admitted (Public Domain / MIT)
- common-pile/arxiv_abstracts (CC0) — D0 base text
- common-pile/caselaw_access_project (Public Domain) — D0 base text
- common-pile/biodiversity_heritage_library (Public Domain) — D0 base text
- openai/gsm8k (MIT) — D2 reasoning
- epfl-dlab/JSONSchemaBench (MIT) — D4 exact verifier

### Reclassified: candidate_admit_after_attribution_manifest
- roneneldan/TinyStories (CDLA-Sharing-1.0) — D5 generation guardrail
  CDLA-Sharing Section 3.5 explicitly exempts Results (model weights)
  Remaining requirements: frozen revision, attribution manifest, leakage testing

### Still Held
- allenai/ai2_arc (CC-BY-SA-4.0) — eval/benchmark only, not training
- databricks/databricks-dolly-15k (CC-BY-SA-3.0) — held
- common-pile/arxiv_papers (mixed licenses) — held

### Rejected
- allenai/scifact (CC-BY-NC-2.0) — non-commercial, rejected

## 15. Convergence Assessment (R6)

R6 verdict: **converging, not oscillating.** The spec is buildable for S0.

R3-R5 did not randomly move the target — each round fixed a specific gap:
- R3: concrete architecture from literature (BLT/MEGABYTE/Charformer)
- R4: underpowered encoder, verifier overclaim, memory underspec, decoder confound
- R5: sub-billion shape (MobileLLM deep-thin), patch projection, decoder conditioning
- R6: confirmed convergence, no further architecture changes before first run
- R7: config validation, next-patch cross-attention removed, 121.7M param count
- R8: patch-local byte mixer (zero cross-patch leakage), dead head freeze
- R9: identified same-patch target leakage — hidden[i] predicted patch i (not i+1)
- R10: validated causal shift fix — hidden[i] now predicts patch i+1, GO for burn-in
- R11: burn-in success criteria — warmup=100 for 500-step run, healthy BPB 3.5-5.0,
  hard-fail if BPB>7.0 or NaN/Inf. Per-position accuracy required (position 0 = global
  context test). Automated verdict via burnin_verdict.py.

### S0 Training Stability Recipe (R6)

```text
pre-norm RMSNorm
residual scaling or careful init
warmup schedule
gradient clipping
bf16 with fp32 optimizer states
FlashAttention
activation checkpoint by 2-4 layer blocks
no teacher-packet multi-loss soup during base S0
```

### Open Parameters (Empirical, Not Theoretical)

These should be determined by runs, not more deliberation:

```text
D: 576 primary; 640/768 only if measured failure
layers: 30 vs 32
patch size: P4 primary, P8 falsifier
patch aggregator: gated MLP vs local latent cross-attention
decoder cross-attn: current patch only vs current + previous 1-2 patches
checkpoint granularity: 1/2/4 layers
microbatch/grad accumulation ratio
loss weights for later packet phase
adapter ranks and landing-zone placement
```

### S0 Success Criteria

S0 does NOT need to prove full Eklavya multi-teacher transfer. It needs to answer:

1. Can this byte/patch substrate train stably?
2. Can it reconstruct/generate bytes with reasonable fidelity?
3. Does P4 vs P8 show measurable tradeoffs?
4. Does the architecture produce useful traceable failures for G0 gap mapping?

If S0 answers these, it has succeeded regardless of absolute quality.

### Escalation Ladder (D=576 Concerns)

If multi-lesson learning at G1 causes interference at D=576:

```text
1. Measure: gradient cosine between losses, adapter ablation,
   CKA/subspace drift, per-lesson retained gain
2. Try 30x640 (first escalation, minimal disruption)
3. Try 22x768 (second escalation, MobileLLM tradeoff reversed)
4. Only then consider new architecture
```

Do not widen preemptively. The 576→640→768 ladder is the fallback plan.

### Post-Burn-In Decision Tree (R12b)

**GO** (BPB 3.5-5.0, no hard/soft fails):
- Resume from 500-step checkpoint (do not restart)
- Full 50K with warmup=1000, eval_every=500
- Milestones: 5K≤3.6, 10K≤3.1, 25K≤2.6, 50K≤2.3
- Launch P8 after P4 clears 5K gate (eval BPB≤4.2)

**CONDITIONAL GO** (BPB 5.0-6.0, or soft concerns):
- Extend burn-in to 1500 steps (do not start full run)
- High gap>0.7: reduce LR to 2e-4, increase weight_decay to 0.15
- Grad clipping >75%: reduce LR to 1e-4, warmup=200
- Eval jitter: increase eval_batches to 100 or full sweep
- Promote to GO if BPB≤5.0 at 1500, pos-0 acc≥0.03

**NO-GO** (hard fails):
- NaN: fp32 debug + anomaly detection at lr=1e-4
- BPB>7.0: run test_overfit.py to isolate arch vs data
- Pos-0 failure: re-verify causal alignment
- Escalate to D=640 ONLY after clean data + stable optimizer still fails
  at BPB>5.5@1500 or BPB>3.8@10K

**Stop training** if eval BPB worsens across two milestone windows,
no ≥0.1 BPB improvement over 5K after 25K, or train/eval gap>1.0.
