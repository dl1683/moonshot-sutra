# Notebook

Reverse-chronological running log. Newest first.

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
