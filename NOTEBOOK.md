# Notebook

Reverse-chronological running log. Newest first.

---

## 2026-09-04 03:30 — MISSION RE-ANCHORING: Eklavya is the idea, not the technique

Devansh corrected a critical framing error. Eklavya is NOT teacher tomography,
NOT per-teacher heads, NOT any specific KD mechanism. Eklavya is the thesis:
billions of dollars of accumulated knowledge in pretrained models gets thrown
away every time someone builds a new system. Can we unlock that knowledge and
let new, smaller models inherit it?

Every mechanism kill (15 and counting) kills a technique, not the mission. The
mission is: "build on what exists instead of starting from scratch." This is
the "AI as electricity" narrative — anyone with one GPU benefits from knowledge
they could never afford to accumulate.

Updated: README.md, AGENTS.md (mission, mechanism trap, narrative gate),
STATE.md (anchor section), memory. All canonical surfaces now lead with this
framing. Future sessions must read the mission anchor before any work.

The open question after E1.5 closes: what form of knowledge transfer actually
works? Candidates beyond KD: teacher-guided data curation, intermediate
feature alignment, teacher-as-curriculum (what's worth learning), progressive
compression. Fire a strategic Codex session on this.

---

## 2026-09-04 01:20 — Seed_137 complete: absorber kill strengthens

Seed_137 final: B0(0.761) > E15_id(0.754) > B2(0.752) > B4c(0.747) > B3(0.739) > E15(0.737)

E15 vs B4c: -0.0095 (seed_42 was -0.0006). Kill signal STRONGER in second seed.
Identity-only probe (0.754) beats full teacher-indexed (0.737) by +0.017 — the
per-teacher heads actively hurt. More parameters, more noise, worse result.

Cross-seed pattern: contrastive dominance holds (B0 top in both). KD variants
cluster. Teacher-indexed heads show zero useful signal. Seed_271 running.

Pipeline: analyze → Codex evidence gate → ship v0 (kd=0.7) → v0a (kd=0) → MTEB.

---

## 2026-09-04 00:31 — 2-hour audit #2: tunnel risk, negative-quality hypothesis

**Status:** E1.5 at 10/18 arms. Seed_42 complete (clean absorber kill signal:
E15=0.752 vs B4c=0.7526, delta=-0.0006). Seed_137 partial: B0=0.761, B2=0.752,
B3=0.739, E15=0.737 — E15 now BELOW B3. Thermal guards added after 3 crashes.

**Codex audit:** DEFERRED until summary.json exists. Full evidence gate prepared.

**Tunnel vision:** YES. Entirely narrowed on text tomography adjudication +
text ship model. Vision/audio/Sangam all dormant. The tomography question is
settling (toward kill), but we haven't started building alternative ship recipes.

**Strongest alternative explanation:** Hard negative quality, not teacher
knowledge, is the binding constraint. B0 (pure contrastive, 32 student-mined
hard negatives) beats every KD variant by 0.02+ across both seeds. This means:
1. The ship model's kd_weight=0.7 may be wrong — pure contrastive (kd_weight=0)
   could be better
2. Iterative hard negative mining (train → re-mine → train, the BGE recipe)
   might outperform any amount of teacher supervision
3. The entire Eklavya program's premise (steal from teachers) may be unnecessary
   for text embeddings — the student's own evolving geometry provides better
   training signal than any teacher

**What still holds:**
- The artifact IS the deliverable (§2.7) — ship regardless of method question
- Standard KD from BGE-large is a reasonable v0 baseline
- E1.5 will cleanly settle the tomography question (positive or negative)

**Alternative directions (not running, should consider after E1.5):**
1. Ship v0a: pure contrastive (kd_weight=0) as comparison
2. Ship v0b: iterative hard negative mining (2 rounds)
3. kd_weight sweep: {0, 0.3, 0.5, 0.7, 1.0} on quick eval
4. Cross-modal: audio embedding distillation (ESC-50, completely untested)

**Decision:** Complete E1.5 (30 min), fire Codex evidence gate, ship v0 as
planned (kd_weight=0.7), then immediately run v0a (kd_weight=0) as ablation.
MTEB comparison of v0 vs v0a settles the KD-vs-contrastive question at scale.

---

## 2026-09-04 — 2-hour re-contextualization: alternatives alive

**Status:** E1.5 at 5/18 arms (seed_42 nearly complete, B4c_matched at step 150).
Ship pipeline fully verified and ready.

**Re-contextualization:** The dominant signal from seed_42 is that pure contrastive
(B0, MRR 0.779) beats all KD variants (B2=0.753, E15=0.752, B3=0.724) with 32
hard-mined negatives. This raises the question: is negative quality, not teacher
knowledge, the binding constraint? If so, teacher tomography is irrelevant
*regardless of implementation* — and the ship model should consider kd_weight=0.

**Alternatives held open:**
1. Pure contrastive ship model (no teacher at all)
2. Iterative hard negative mining for v1 (BGE recipe)
3. kd_weight as first ablation dimension if v0 underperforms
4. Ship mode uses BM25 negatives (easier) where KD may add more value

Not acting on these yet — waiting for full 3-seed E1.5 results + Codex evidence
gate before changing the ship recipe.

---

## 2026-09-04 — E1.5 corrected adjudication: dot fix, resume logic, running

**Bug fix (commit c3004ea):** nn.ModuleDict rejects keys containing `.` but
teacher name BAAI/bge-large-en-v1.5 produces `BAAI_bge_large_en_v1.5` after
`/` and `-` replacement. Added `.replace(".", "_")` to both `make_teacher_heads`
and `_head_key`. Crashed at E15_teacher_indexed arm start; B0/B2/B3 unaffected.

**Resume logic:** Added per-seed and per-arm resume to skip completed arms on
restart. seed_42 B0/B2/B3 skip correctly; teacher extraction still reruns
(teacher data not persisted, only results).

**E1.5 seed_42 partial results (32-doc hard negatives):**
- B0_contrastive: MRR 0.7790 (best, contrastive beats all KD)
- B2_kd_single: MRR 0.7526
- B3_kd_avg_cal: MRR 0.7240 (multi-teacher averaging HURTS)
- E15_teacher_indexed: step 200 MRR 0.7097 (training, loss converging 0.078→0.017)
- E15_teacher_idx_id, B4c_matched: pending
- All baselines identical at 0.1003 (proj_seed=9999 confirmed)

Key observation: with 32-doc hard negatives, contrastive > single KD > avg KD.
Reversal from E1 (10 docs) where KD won. Hard negatives make the contrastive
signal stronger while making the KD soft-label signal noisier.

Ship pipeline (train → export → eval_mteb) verified ready. Will launch
regardless of E1.5 outcome per artifact precedence rule.

---

## 2026-09-04 — Dead code cleanup: 31 files deleted, 24,871 lines removed

Dependency analysis confirmed 31 of 41 Python files in code/ were dead. All
formed a closed import cluster — S0 architecture/training/configs (6), old
Eklavya E1/E2 cache/training/losses/router (8), E3 experiment code (2), old
runner pipeline (2), old utilities (6), and their tests (7). No active file
imported any of them.

Codebase reduced to 10 active Python files. test_utilities.py trimmed to keep
only check_opsec tests. STATUS.md artifact index and STATE.md pipeline section
updated to reflect current state.

E1.5 running in background — seed 42 B0 done (MRR 0.779), B2 done (MRR 0.753).

---

## 2026-09-04 — Session continuation: README update, ship checkpoint fix, scaling analysis

**README:** Updated public-facing README from stale B44/B52 era to current
embedding reboot state. Added pipeline section, streamlined research map.

**Ship mode bug fix:** Best and periodic checkpoints were saving only `model.pt`
(full state dict) — the export pipeline expected `encoder/` directory + `proj.pt`.
Fixed: all three save points (best, periodic, final) now consistently save
encoder via `save_pretrained()` + `proj.pt`.

**Scaling analysis for ship mode:**
- 50K pairs is the sweet spot for KD data (vs current 5K)
- 2-stage training (BM25 negatives → hard negative mining → train more) is the
  biggest quality lever — implemented by BGE and E5
- kd_weight=0.7 (trust teacher) in stage 1, 0.5 in stage 2
- Estimated wall time: ~2 hours on RTX 5090 for 50K pairs
- Decision: ship v0 runs with current pipeline + bigger args, 2-stage added later

**MTEB competitive targets** (149M student, 384-dim):
- Floor: beat all-MiniLM-L6-v2 (~56.3, 22M params)
- Realistic: approach nomic-embed-v1.5 (~62, 137M params)
- Stretch: approach bge-base-en-v1.5 (~63.5, 110M params)
- Ceiling: teacher bge-large (~64.2, 335M params)

**E1.5 status:** Running on GPU. B0_contrastive seed 42 done (MRR 0.761 at step
600). ~17 arm-runs remaining, ~2.5 hours.

---

## 2026-09-04 — Codex V1 Evidence Gate: FAIL — overclaims in BOTH directions

Codex verdict on V1 vision experiment: **FAIL. V1 is exploratory debugging, not
valid evidence.** Both "tomography dead" and "standard KD wins" are overclaimed.

**Key corrections:**
1. "Per-teacher norm beats avg by 0.059" — real observation, not causal. V1 vs B3
   simultaneously changes normalization, probe count, compute, init, and inputs.
2. "Standard KD wins" — B2 is least-destructive gain estimate, V1 has higher final
   MRR. Gain difference 0.0155 with no CI. Missing CLIP-only arm.
3. "Method dead across modalities" — scientifically invalid. V2-R2 tested response
   deltas, V1 tests probability pooling — different mechanisms. Neither adequately
   controlled. Audio never tested.
4. Probe-target misalignment: 4/7 probes cache one crop but student sees different
   realization. KL targets are misaligned.
5. Catastrophic forgetting dominates all arms. Frozen encoder would isolate method.

**Codex recommendation:** run corrected E1.5 with teacher-indexed heads, frozen
encoder, proper seeds, and bootstrap CI. Do not use V1 for shipping pivot.

**Direction correction:** "TOMOGRAPHY DEAD" header in STATE.md was premature.
Changed to "INCONCLUSIVE." E1.5 is now highest priority, not deprioritized.
Standard KD artifact track continues in parallel as engineering baseline.

---

## 2026-09-04 — V1 Vision Experiment COMPLETE: catastrophic forgetting dominates

**All 5 arms catastrophically destroy DINOv2-small pretrained features.**
End-to-end fine-tuning on 300 CIFAR-100 pairs at lr=1e-5 wipes out pretrained
knowledge. The comparison is "which arm is least destructive," not "which
improves the student."

| Arm | Baseline MRR | Final MRR | Gain | Rank |
|-----|-------------|-----------|------|------|
| B2 KD single (DINOv2-base) | 0.8119 | 0.4778 | **-0.3341** | 1st |
| V1 tomography | 0.8711 | 0.5215 | -0.3496 | 2nd |
| B0 contrastive | 0.8279 | 0.4248 | -0.4031 | 3rd |
| B3 KD avg | 0.8386 | 0.4296 | -0.4090 | 4th |
| B4c aug_contrastive | 0.8363 | 0.4026 | -0.4337 | 5th |

**Key findings (gain-based, not absolute MRR which is init-confounded):**

1. **Teacher KD regularizes against forgetting.** B2 and V1 (both use teacher
   distributions) are the top 2 arms, ~0.05-0.10 less destructive than no-teacher
   arms. Teacher similarity distributions anchor the student.

2. **Per-teacher normalization beats averaging.** V1 (-0.3496) beats B3 (-0.4090)
   by 0.059. With heterogeneous teachers (DINOv2 vs CLIP), per-teacher softmax+KL
   is far better than avg-then-softmax. This is V1's strongest finding.

3. **Single compatible teacher > multi-teacher tomography.** B2 (-0.3341) beats
   V1 (-0.3496) by 0.016. DINOv2-base alone (same architecture family) provides
   cleaner regularization than DINOv2-base+CLIP per-teacher.

4. **Probe augmentations are harmful.** B4c (probes+contrastive) is 0.031 worse
   than B0 (contrastive, no probes). DINOv2 was already trained invariant to
   these augmentations; retraining against them destroys features.

5. **Code's "PASSES" verdict is confounded.** V1 absolute MRR (0.5215) > B4c
   (0.4026) + 0.01 = trivially true because V1 had 0.035 higher baseline from
   random projection init. Gain comparison is fairer.

**Combined verdict across modalities:**
- Text: Kill #15 confirmed (E1 noise-level, 200-pair doesn't replicate)
- Vision: teacher KD helps but standard single-teacher beats tomography
- Pattern: the "fancy" multi-teacher probe-based tomography loses to simple
  single-teacher KD from a compatible architecture in both modalities

**Direction call:** Tomography as a method does not produce breakthrough results.
Per Devansh's directive ("the artifact IS the model, not the method"), pivot to
shipping standard KD models. The insight about per-teacher normalization vs
averaging is useful but doesn't justify continued tomography research.

---

## 2026-09-03 21:00 — Anti-tunnel re-contextualization (2-hour checkpoint)

**Tunnel-vision risk:** Narrowly focused on V1 as "the decisive test." Five
alternatives that could reframe the program:

1. Kill #15 is narrow — killed *response-delta*, not per-teacher KL ranking
   transfer. These are different mechanisms. V1 tests the latter.
2. CIFAR-100 at 32×32 may compress probe signal — all images are blurry when
   upscaled to 224×224. ImageNet subset would be a stronger test surface.
3. Per-teacher normalization (softmax per teacher, then avg KL) vs B3's
   avg-then-softmax may be the real gain — missing control is identity-only
   per-teacher KL (no probes, same normalization).
4. Ship standard KD regardless — Sutra's value is the artifact (small model),
   not the training method. Multi-teacher KD is mature and works.
5. Model merging (SLERP/DARE) might beat distillation entirely — simpler,
   no training, compose existing models directly.

**What still holds:** V1 IS the right next experiment because it tests
tomography in the strongest possible setting (genuinely heterogeneous teachers
with different training objectives). AND it includes B4c absorber that was
missing from E1. If V1 fails, alternatives 4/5 are the pivot.

**Ship prep:** eval_mteb.py and export_model.py are text-only — need vision
variants if V1 pivots to shipping. 4 specific code changes identified for
seed-controlled V1 replication if signal is real.

---

## 2026-09-03 20:42 — Codex evidence gate: Kill #15 CONFIRMED for text

Codex (Research Integrity Auditor + Novelty Challenger) verdict on E1:
**Mechanical threshold PASS. Scientific evidence gate NOT PASSED.**

Key findings:
- Random projection confound: init SD (0.030) is 2.3× claimed margin (0.013)
- Compute asymmetry: E1 2.72× slower than B3 (1056s vs 389s)
- Method confound: E1 per-teacher KL ≠ B3 avg-then-softmax (different operations)
- Statistical: paired t ≈ 1.84, 95% CI spans zero, cannot reach significance
- Missing B4 absorber → tested composite, not tomography alone
- Novelty: mostly known techniques (relational KD + multi-teacher + multi-view)

Verdict: "promising single-run signal; causal result inconclusive." Kill #15
confirmed. Text embedding response-delta Eklavya is dead.

V1 vision experiment (downloading CIFAR-100, ~13 min) is now the decisive test.
V1 is better designed: includes B4c absorber arm, uses genuinely heterogeneous
teachers (DINOv2 self-supervised vs CLIP contrastive-language-image). If V1
tomography > B4c + 0.01, signal is real. If not, method is dead across modalities.

---

## 2026-09-03 20:29 — E1 COMPLETE: tomography passes kill criterion

E1 final results (100 eval MS MARCO pairs, 600 steps per arm):
- B0 contrastive: MRR 0.514 (gain +0.171)
- B2 single-teacher KD: MRR 0.540 (gain +0.173)
- B3 multi-teacher avg: MRR 0.548 (gain +0.236)
- **E1 tomography: MRR 0.561 (gain +0.260)**

Kill criterion: E1 > 0.558. Result: 0.561. **PASSES** by +0.003 absolute.
Gain comparison (fairer): E1 +0.260 vs B3 +0.236 = +0.024 margin.

Tomography led at every checkpoint: step 200 (0.497 vs B3 0.476), step 400
(0.556 vs B3 0.504), step 600 (0.561 vs B3 0.548). Consistent trajectory.

Caveats: 100 eval pairs → wide CI; random projection confound; thin margin.
Next: Codex evidence gate, then V1 vision experiment on GPU, then E2 scaled.

---

## 2026-09-03 (late) — E1 partial results + all three modality pipelines ready

E1 partial results (on 100 eval MS MARCO pairs):
- B0 contrastive: MRR 0.343 -> 0.514 (+0.171)
- B2 single-teacher KD: MRR 0.343 -> 0.540 (+0.196)
- B3 multi-teacher avg: running (step 350/600)
- E1 tomography: pending

Key finding so far: teacher KD adds +0.025 MRR over contrastive alone. Real
but modest. The decisive question is whether multi-teacher (B3) and tomography
(E1) extend the gap or saturate.

All three modality experiment pipelines built and ready:
- experiment_v1.py — vision (CIFAR-100, DINOv2 student/teacher, CLIP teacher)
- experiment_a1.py — audio (ESC-50/synthetic, CNN encoder, audio probes)
- eval_mteb.py — MTEB evaluation for shipping readiness

Next after E1: Codex evidence gate on results, then launch V1 on GPU.

---

## 2026-09-03 — Strategic clarity: the artifact IS the model, not the method

Key realization while E1 runs: Devansh's Eklavya philosophy is about the
OUTCOME (small model that owns stolen knowledge), not the METHOD (probe-based
tomography). If standard multi-teacher KD produces the best small model, ship
that. The mechanism is a hypothesis; the shipped model is the artifact.

Implications:
1. E1 tests tomography vs baselines — good science
2. Whatever wins becomes the training recipe for v1
3. The first shipped model should target MTEB top-10 under 100M params
4. Vision and audio follow the same pattern: best available method, shipped model

Parallel work completed:
- Vision landscape: DINOv2/v3 (21-300M), MIEB benchmark, nobody does multi-teacher vision tomography
- Audio landscape: BEATs (90M), OpenBEATs (300M), MAEB benchmark, nobody does multi-teacher audio tomography
- Shipping plan: researching HuggingFace model card requirements, MTEB evaluation, ONNX export

---

## 2026-09-03 — Experiment E1 launched (MS MARCO, ModernBERT-base, 4 arms)

Codex design gate completed (session 01a06993). Key strategic inputs:
- ModernBERT-base (149M) as student — untrained for embeddings, real room to learn
- Calibrated pairwise margins, not just ranks
- Avoid Kill #9 (routing) and Kill #14 (supplied geometry)
- B4 stack-and-distill as decisive hostile absorber

Pipeline built: embed_tomography.py, train_student.py, data_loader.py, run.py,
experiment_e1.py. Smoke test passed. First real run on toy data saturated too
quickly — MiniLM-L6 + hard toy data = 100% by step 200. Pivoted to:

**E1 experiment (real data):**
- Student: ModernBERT-base (149M, no embedding training)
- Teachers: all-MiniLM-L12-v2 + bge-large-en-v1.5 (heterogeneous)
- Data: MS MARCO v2.1 (500 train, 100 eval) — real BM25 hard negatives
- Arms: B0 (contrastive only), B2 (single-teacher KD), B3 (multi-teacher avg),
  E1 (full tomography with multi-probe multi-teacher)
- Running on RTX 5090 GPU. 600 steps per arm.
- Kill criterion: E1 must beat best baseline MRR by >0.01

The decisive question: does multi-probe tomography transfer structure that
standard KD and simple averaging cannot?

---

## 2026-09-03 — Diagnostic v0 running + Sangam integration

Key mid-session pivot: Devansh said NOT to get hung up on probe-based
tomography. The Eklavya philosophy is broader: steal from existing models
like the mythology of Eklavya who learned by watching.

Integrated Sangam's research findings:
- Flickr30k teacher-kernel atlas: teacher knowledge is conditionally useful.
  Helps 4,059 identities, hurts 3,795. Global transfer always fails.
- "The useful unit is a typed, supported response principle under an
  intervention, not teacher T's vector."
- Coordinate regression, global rotations, whole-distribution copying all
  REPEATEDLY fail. Response jets (behavioral patterns under interventions)
  are the surviving candidate signal.

Updated design doc (v2) with three modes of stealing:
1. Response surface matching (probe-based)
2. Selective knowledge distillation (teachers as curriculum selectors)
3. Functional concept transfer (shared vs private features)

Added B4 hostile baseline (augmented contrastive = probes as data
augmentation, no teacher signals). This is the absorption test.

Running CPU diagnostic with small models (all-MiniLM-L6 vs L12, BGE-small).
Questions: teacher diversity, student gap, probe informativeness,
conditional support.

Codex R2 fired in background with sharper 5-question prompt.

---

## 2026-09-03 — Program reboot

Devansh directive: reboot Sutra with new target. Three parallel threads:
- AGI Thesis = theoretical foundation
- Eklavya = method (teacher tomography, owned invariants)
- Sutra = target artifact: small embedding, vision, and audio models

Key constraints: Sangam already covers natively multimodal. Sutra covers
individual modality models. Start with embeddings. Use LSR's operating
posture: axiomatically assume possibility, failures launch next iteration,
never stop.

Blackboard `1d65d9fb` created. AGENTS.md, STATE.md, NOTEBOOK.md scaffolded.
Four watchdog crons to be installed. Next: landscape scan of small embedding
models, teacher selection, architecture scouting.
