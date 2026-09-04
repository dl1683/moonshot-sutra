# Notebook

Reverse-chronological running log. Newest first.

---

## 2026-09-04 20:45 -- Codex R4: Strategic Pivot to Attribution + E16

**Codex R4 strategic review (outputs/codex_strategic_r4.txt):**

Key directives:
1. **Unfrozen B0 baseline is NON-NEGOTIABLE.** Ship mode proves nothing about KD
   without a matched kd_weight=0 comparison. Code fixed: teacher loading skipped
   when kd_weight=0 (committed 03b09b3).
2. **Fade demoted to appendix.** E16 Boundary Inheritance (teacher-curated negatives)
   promoted to mainline: preserves winning contrastive objective, avoids KD interference.
3. **P2 implementation gap:** provenance manifest compares vs trained B0, not raw
   student. Current data is exploratory evidence, not the formal estimand.
4. **Distance-3+ drift confirmed.** No more provenance dimensions until unfrozen
   control and E16 preflight are complete.

**Deeper provenance analysis (union across teachers):**
- 40 donor-private queries (Q1): B2 rescues 20/40 (50%), B3 rescues 14/40 (35%)
- 32 B0-only queries (Q2): B2 loses 15/32, B3 loses 23/32
- **Net: B2 = +5 queries, B3 = -9 queries**
- On shared-Q1 (28 queries): B0 RR=0.368, B2 RR=0.729, B3 RR=0.683
- KD IS transferring knowledge but creating interference. Net effect barely positive
  for single-teacher, net negative for multi-teacher.

**Implication:** teacher knowledge should enter through DATA SELECTION (E16), not
loss modification (KD). This avoids interference entirely while preserving the
contrastive objective that already wins.

**Revised priority sequence:**
1. Finish ship v0 (step ~1350/3000)
2. Run matched unfrozen B0 (kd_weight=0) → MTEB eval on both
3. E16 preflight design with Codex
4. Fade only as appendix experiment

---

## 2026-09-04 19:30 -- P2 Gate PASS + Ship Mode Progress

**P2 Gate Result: DECISIVE PASS (17%)**
Provenance manifest seed 42 complete (SHA 8848c1552151b6a6). Both teachers show
Q1 (donor-private) = 34/200 = 17%, massively exceeding the 5% threshold.

Quadrant analysis (seed 42):
- MiniLM-L12: Q1=17%, Q2=13%, Q3=53%, Q4=17%. Teacher MRR=0.819
- BGE-large: Q1=17%, Q2=10%, Q3=56%, Q4=17%. Teacher MRR=0.828
- B0 baseline MRR=0.779

KD rescue rates on Q1 (donor-private queries):
- B2 (standard KD): 18/34 = 53% rescued
- B3 (calibrated avg): 13-14/34 = 38-41% rescued

Key insight: standard KD already transfers 53% of donor-private signal on frozen
encoder, despite losing aggregate MRR to B0. The fade experiment will test whether
this transferred signal persists when KD pressure is removed.

Seeds 137+271 computing on CPU for complete 3-seed manifest.

**Ship mode v0 progress (unfrozen KD):**
- Step 500: MRR=0.471, hit@1=0.264, hit@5=0.774 (+41% over baseline)
- Step 1000: MRR=0.506, hit@1=0.296, hit@5=0.816 (+51% over baseline)
- Loss trending: 0.728 → 0.542 (step 1000)
- Training was stalled ~4h due to orphaned GPU processes; resumed after cleanup

**Process cleanup:** Killed 5 orphaned provenance_manifest processes, 1 orphaned
GPU mining script, 1 dead ship mode process. Freed significant CPU and GPU.

---

## 2026-09-04 16:00 -- Fade Pilot Code Hardened + Provenance Re-run

**Fade pilot code improvements (experiment_e1.py):**
- Arm metadata logged at start: frozen status, trainable param count, phase, eval_every
- LR and phase recorded in every log entry (was missing — Codex R3 required)
- Projection checkpoint saved at fade_step and final step (steps 200, 600)
- Manipulation check added: post-hoc B3 vs B0 comparison at step 200 across seeds
- MRR@10 printed alongside MRR in step-level output

**Provenance manifest crashed:** First CPU run (PID 34328) loaded both teachers
simultaneously, consumed 14K CPU seconds and 7.9GB RAM, then died silently —
likely OOM from memory pressure with ship mode running concurrently. No output
file written.

**Re-run with fixes:** Memory-safe version loads one teacher at a time, scores all
queries, deletes, loads next teacher. Progress printed every 20 queries. Limited
to 4 CPU threads (OMP/MKL). Running seed 42 only for fast P2 gate answer.

**Ship mode status:** Step 350/3000, loss 0.641-0.665, GPU 99%/88°C, ~3h remaining.

---

## 2026-09-04 15:00 -- SANGAM-E0-TDPA Protocol Frozen + Status Check

**SANGAM-E0-TDPA protocol written** (outputs/sangam_e0_tdpa_protocol.json):
Full typed donor-private provenance assay protocol for Sangam. Frozen per Codex
Round 3 §9. Key design elements:
- Fail-closed cache validator (103/103 shards required, all terminal status checks)
- 3 media-identity-disjoint folds (seed 42, ±10% balance)
- 3 typed candidate pools: image↔text, video↔text, temporal-video (32 candidates each)
- 3 scorers: Qwen teacher (from cache), Sangam recipient (live), OOF label-only baseline
- Q1-Q4 quadrant analysis per type and difficulty stratum
- 4 admission gates: G1 (Q1≥5%), G2 (turnover≥20%), G3 (teacher advantage CI>0), G4 (contamination≤2%)
- Cluster bootstrap (10K draws) by media_identity
- Contamination ledger: alternate-positive, duplicate-identity, cross-fold leakage
- Execution: ONCE on fit-only evidence after 103/103 cache validated and hash-frozen
- No execution until cache complete (currently 40/103)

**Parallel compute status:**
- Ship mode v0 (GPU, PID 29400): step 250/3000, loss 0.641, ~4h remaining
- Second ship mode (GPU, PID 13316): n_train=20000, steps=5000, competing for GPU
- Provenance manifest (CPU, PID 36628): running, 10K+ CPU seconds consumed
- Two Sangam semantic descent processes also running (CPU)
- GPU at 99%/88°C

**Gate question:** Provenance manifest will answer whether teachers actually rank
eval queries correctly. If P2 (donor-private prevalence) ≈ 0, fade experiment
is pointless regardless of infrastructure quality.

---

## 2026-09-04 14:30 -- Codex Round 3: Convergence Spec Delivered

Codex Round 3 (session 01a06bed, outputs/codex_round3_convergence.txt).
Three rounds complete. Key convergence points:

**IMPORTANT CORRECTION: E1.5 was FROZEN (confirmed).** Summary.json records
`"frozen": true` from args. The `[FROZEN]` banner prints to stdout, not log.jsonl
— its absence from logs doesn't prove unfrozen. All fade arms must also be frozen.

**Manifest spec (outputs/E_provenance_v1/candidate_manifest.json):**
- Inputs only — never teacher or student outputs in the manifest
- Same 200 eval query identities and 32-candidate recipe as E1.5
- New canonical realization with full texts, hashes, SHA-256
- Each query: id, texts, candidates[32] with document_text_sha256
- Label correction: record ALL selected passages (not just first)
- All 9 fade arms consume this manifest; legacy E1.5 = descriptive only

**Predeclared estimands (frozen, per Codex Round 3 §4):**
- P1: paired mean RR10_donor - RR10_raw (per teacher)
- P2: donor-private prevalence Pr(C_d=1, C_raw=0) — KEY GATE
- P3: B0-resistant donor-private (held-out seed, both other B0s wrong)
- P4: recipient-private/corruption prevalence
- P5: donor-recipient top-10 turnover (1 - Jaccard)
- P6: MiniLM-only / BGE-only / shared / B3-rescue decomposition
- P7: unresolved-top, duplicate-positive, contamination rates
- F1 (primary): mean paired RR10_B3fade - RR10_B0sched at step 600
- F2 (secondary): B2 fade delta
- F3: retained gain on strict carrier-private cases
- F4: recipient preservation on recipient-correct/donor-wrong
- F5: paired curve at steps 100,200,300,400,500,600
- F6: legacy LOSO treatment strata (secondary, separate from provenance)

**Success criteria:** F1 lower 95% bound > +0.005 MRR@10. Plus carrier
provenance, F3 positive, F4 no worse than -0.005.

**Manipulation check:** B3 fade must beat B0 at step 200 in NEW runs.
If early advantage doesn't reproduce, pilot fails its premise.

**Sangam TDPA prework (buildable now):**
Protocol, fail-closed cache validator, 3 identity-disjoint folds,
typed candidate pools, baseline interfaces, ranking adapter, Q1-Q4 analyzer.
No execution until 103/103 cache.

---

## 2026-09-04 14:00 -- Provenance Manifest + Evaluator Improvements

**Provenance manifest script (code/provenance_manifest.py):** Running on CPU while
ship mode uses GPU. For each of 3 seeds: reconstructs E1.5 eval candidate pools
(deterministic), scores each 32-candidate pool with MiniLM-L12 and BGE-large,
produces per-query teacher rankings. Includes baseline verification (MRR match
with saved E1.5 values) and SHA-256 hash. Output: outputs/provenance_manifest.json.

**MRR@10 added to evaluator:** `evaluate()` now computes MRR@{cutoff} (default 10)
alongside existing MRR. Per-query records include `rr@10`. Backward compatible.

**Paired bootstrap CIs added to fade analysis:** `paired_bootstrap_ci()` computes
95% CI on per-query RR differences (fade vs B0) pooled across all seeds. 10K
bootstrap samples. Reports significance (CI excludes zero).

**Ship mode v0:** Step 100/3000, loss 0.728→0.707, kd_weight=0.7, BGE-large teacher.
GPU at 89C/99%. First eval at step 500.

**Codex Round 3:** Running in background. Convergence round — concrete buildable spec
for provenance manifest, predeclared estimands, fade pilot approval.

---

## 2026-09-04 12:45 -- Codex Round 2: Sharp Corrections Accepted

Codex Round 2 (session 01a06bde, 180 lines, outputs/codex_round2_response.txt).
Most impactful round yet — changed my mind on 4 points:

**CORRECTION 1: My "provenance" quadrants are MISLABELED.**
No teacher was scored on eval queries. Q1-Q4 compare B2-student vs B0-student
outcomes, not teacher-correct vs student-wrong. Correct labels: "KD-recipient
rank 1, B0-recipient not rank 1" — TREATMENT EFFECTS, not provenance.
Calling Q1 "inheritance" presupposes the conclusion.

**CORRECTION 2: B2 is NOT the best fade candidate.**
B2 trails B0 at step 200 in E1.5 (confirmed by learning curves). B3
(calibrated average) leads: B3=0.705 vs B0=0.672 mean at step 200. B3->B0
tests the observed early advantage; B2->B0 tests a non-advantage. Changed
primary fade arm to B3_fade_200.

**CORRECTION 3: Fade and provenance answer DIFFERENT questions.**
Fade: "Does transient teacher supervision improve trajectory?"
Provenance: "Did teacher possess correct information student lacked?"
Without provenance, positive fade = maybe optimization, not inheritance.
Without fade, positive provenance = knowledge exists, carrier untested.
They're COMPLEMENTARY — run concurrently, interpret jointly.

**CORRECTION 4: "Judgments vs representations" is false dichotomy.**
Deeper framing (accepted): "Eklavya is an inheritance assay and compiler.
It locates donor-private capability, compiles it into the cheapest viable
carrier — data, ordering, weights, modules, or persistent artifacts — removes
the donor, and verifies retained capability."

**Codex's corrected plan:**
1. Ship mode finish (artifact only)
2. Materialize immutable candidate manifest + separate teacher rankings
3. Predeclare joint estimands (MRR@10, donor-correct/B0-wrong gain, etc.)
4. Provenance scoring (CPU) + fade pilot (GPU) concurrently
5. Use B3->B0 for observed early-advantage test; B2->B0 as secondary
6. SANGAM-E0-TDPA after Qwen cache is 103/103 (currently 40/103)

**Codex's critique of fade code (partially valid):**
- ~~frozen/unfrozen mismatch~~ — WRONG: E1.5 also ran unfrozen (verified)
- Circular subgroup selection — VALID, fixed with leave-one-seed-out
- Should use B3 not B2 — VALID, changed
- Lacks MRR@10 and paired CIs — VALID, will add
- "Helped" set selection inflation on 8 queries — VALID caveat

**Jaccard reinterpretation:** 0.773/0.414 is a "strong routing hypothesis,"
not proof that data selection will preserve help and eliminate harm. The
0.414 hurt Jaccard is NOT noise — it's highly structured (12/29 overlap vs
~2 expected by chance). "Selecting only Q1 examples does not localize the
parameter update — Q1 gradients can spill into Q2 behavior."

Full output: outputs/codex_round2_response.txt

---

## 2026-09-04 12:15 -- Codex Round 1 + Sangam Bridge: Convergence Synthesis

**ROUND 1 OUTPUT** (outputs/codex_round1_assay_debate.txt, 127 lines):
Codex revised conservation field to "ANISOTROPIC conservation field" — local
donation and global conservation coexist. Key changes from Round 0:
- Jaccard result IS real: teacher imposes repeatable, query-specific bias
- BUT B2 and E15 share MiniLM-L12 — Jaccard may be MiniLM-shaped, not BGE-shaped
- Causal cassettes WITHDRAWN as near-term recommendation
- Developmental inheritance: testable but probably can't beat BGE-base (109M)
- Central claim must NARROW: "compact descendant may retain donor-held
  capability from sealed packet at lower lifecycle cost"
- DO NOT launch full E16 — run donor-provenance gate first, then teacher fade

**SANGAM BRIDGE** (outputs/codex_sutra_sangam_bridge.txt, 147 lines):
Codex read Sangam's full codebase. CRITICAL FINDING: **Sangam already observed
the identical conservation-field pattern independently.**
- Teacher atlas: helped 4,059 identities, hurt 3,795
- Helped hardest 4 deciles, regressed all 6 easier deciles
- None of 57 global maps transferred Qwen's advantage
- 384-D deficit is UPSTREAM (feature formation), not final-vector capacity

**Strategic conclusion: "Treat Qwen as a frozen cartographer, not a force field."**
- Qwen cache (103 shards) should change WHICH EXAMPLES student studies
- NOT what representation student is forced to imitate
- Student trains with ordinary supervised/contrastive objective, Qwen-free
- Proposed SANGAM-E0-TDPA: Typed Donor-Private Provenance Assay (no training)
- Inheritance must be DOMAIN-TYPED: Qwen admitted for visual-semantic, explicitly
  unsupported for acoustic/synchronized-AV/physical-time/evidence-revision
- Kill criteria: 5% stable Q1, 20% top-k turnover, 2% max false-negative rate

**CONVERGENCE across Sutra + Sangam:**
The same pattern — net negative aggregate, positive on hard queries, negative
on easy — operates across:
- Text-only: BGE→ModernBERT (Sutra E1.5)
- Multimodal: Qwen→384-dim (Sangam teacher atlas)
This suggests a UNIVERSAL pattern in KD, not a setup artifact. The conservation
field is real; the question is whether selective inheritance can extract the
hard-query benefit without the easy-query corruption.

**Eklavya reframed:** Not a new loss function. A METHODOLOGY for teacher-guided
data curation. Teacher's most valuable output = its JUDGMENTS about training
data, not its REPRESENTATIONS. This is a DEEPENING, not a narrowing — it
resolves why all 15 loss-based mechanisms failed while data-selection survived.

**NEW: Teacher decomposition analysis (B2=MiniLM-only vs E15=MiniLM+BGE):**
Codex Round 1 flagged that B2 and E15 share MiniLM-L12. Decomposition confirms:
- ~85% of E15's help is MiniLM-shared (33/39 queries per seed)
- Only ~15% is BGE-specific (5-7 queries, NOT stable across seeds: 0 in all 3)
- BGE adds more corruption than help (13-16 hurt vs 5-7 help per seed)
- BGE interferes with MiniLM help on 5-9 queries per seed
- MiniLM-only hurt (B2 hurts, E15 doesn't): 12-17 queries — BGE rescues from MiniLM
- The Jaccard of 0.773 IS predominantly MiniLM-shaped, NOT BGE-shaped

**Design implication:** Teacher fade should use B2 (MiniLM-only), not E15.
MiniLM is the stable help channel. BGE is net-negative and unstable.
For E16 data selection: MiniLM-L12 may be better teacher than BGE-large.

**CAUTION: Fade prediction analysis (using E1.5 learning curves):**
B2 does NOT lead B0 at step 200 in E1.5:
- Seed 42: B0=0.657, B2=0.649 (B0 leads)
- Seed 137: B0=0.667, B2=0.657 (B0 leads)
- Seed 271: B0=0.692, B2=0.694 (essentially tied)
This differs from E1 (where B2 led B0 dramatically at step 200). The harder
32-candidate negatives in E1.5 provide enough gradient signal that the teacher's
soft distribution doesn't accelerate early learning. Fade may converge to B0
in aggregate. The signal, if it exists, will be in per-query decomposition:
does fade preserve B2's help on ~38-43 teacher-helped queries?

Round 2 Codex dialogue in progress. Key questions:
- Should teacher fade come BEFORE or AFTER donor-provenance gate?
- Does the cross-project convergence strengthen or weaken the case?
- Is "methodology for data curation" a deepening or a retreat?
- Should E16 use MiniLM-L12 instead of BGE-large as data curator?
- B2 doesn't lead at step 200 in E1.5 — does this change the fade design?

---

## 2026-09-04 11:30 -- 4-Quadrant Provenance + Codex Cross-Seed Synthesis

**NEW ANALYSIS: 4-Quadrant Provenance (Codex-recommended, now computed).**
Using B2 (teacher-assisted) vs B0 (student-only), "correct" = rank 1:

| Quadrant | Seed 42 | Seed 137 | Seed 271 | Meaning |
|----------|---------|----------|----------|---------|
| Q1 (teacher helps, student fails) | 22 | 24 | 23 | INHERITANCE |
| Q2 (student wins, teacher hurts) | 32 | 27 | 32 | CORRUPTION |
| Q3 (both correct) | 100 | 100 | 97 | REDUNDANT |
| Q4 (neither correct) | 46 | 49 | 48 | BOTH FAIL |
| Net Q1-Q2 | **-10** | **-3** | **-9** | NET NEGATIVE |

**Net inheritance is NEGATIVE in all 3 seeds.** Teacher corrupts more queries
than it helps. 50% of queries are redundant (Q3). Only 8 queries are in Q1
across all 3 seeds (4%): msmarco_404, 444, 474, 492, 501, 519, 536, 579. On
these 8, teacher improvement is massive (+0.6 MRR), but drowned by corruption.

**Codex cross-seed session (01a06bc8, completed) key findings:**
1. E15-B2 help Jaccard = **0.773**; hurt Jaccard = only **0.414**. The
   beneficial knowledge channel is MECHANISM-STABLE; corruption is
   mechanism-dependent. Strongest argument for E16 — change the mechanism
   entirely to inherit stable help without unstable harm.
2. Among 23 hard queries: 12 consistently helped, 1 consistently hurt, 10
   inconsistent. Among 8 always-hard: 5 helped, 0 hurt. Teacher knowledge
   genuine for hard queries.
3. Net E15 effect decomposed: consistently helped +0.0374, consistently hurt
   -0.0431, inconsistent -0.0182. Most net harm from INCONSISTENT bucket.
4. Sign agreement 59% vs 20.3% expected under independence. Structural signal.
5. E16 needs positive-aware exclusion — current loader can surface unlabeled
   positives as "negatives." More knowledgeable teacher → worse contamination.
6. **Conditional GO for E16 as selective boundary inheritance** (not blanket
   teacher top-k). 6-arm selector comparison with BM25 control, preflight
   without training, then 3 data replicates x 2 seeds. T-consistent must beat
   best no-teacher selector with lower 95% CI above +0.010 MRR@10.

**Synthesis: Conservation field interpretation is INCOMPLETE.** The help
channel is structural (Jaccard 0.773); the harm channel is mechanism-dependent
(0.414). E16 doesn't change the loss — it changes the DATA. If harm is
mechanism-dependent and E16 uses a fundamentally different mechanism
(data-selection not loss-modification), it may inherit help without harm. But
the 4-quadrant analysis shows only 4% consistent inheritance — the corridor
is narrow.

---

## 2026-09-04 11:00 -- Codex Deep-Think: Program-Level Reframe

Codex deep-think session (01a06bbd, 6 questions, xhigh effort). The most
important Codex output this program has received. Key synthesis:

**"The teacher behaves like a conservation field, not a knowledge donor."**
Teacher accelerates early learning and preserves existing geometry but restricts
later plasticity. B0 wins because it's FREE to reorganize. The experiment doesn't
ask "can student acquire donor-only knowledge" — it asks "which supervision
best adapts an ALREADY-KNOWLEDGEABLE representation." ModernBERT already has
substantial pretrained knowledge; our ranking task exposes it, not creates it.

**The deepest methodological problem: "optimized the transfer mechanism before
building an inheritance assay."** A proper assay requires:
1. Donor possesses capability C
2. Recipient demonstrably LACKS C
3. Ordinary training under matched budget cannot reveal C
4. Bounded, sealed donor packet transferred
5. Donor removed
6. Recipient exhibits C on hidden compositional tests
7. Repeats across recipient architectures
8. Recipient-only capabilities retained
9. Lifecycle cost beats direct reacquisition

**Per-query teacher effects are METHOD-INVARIANT (Pearson 0.86-0.90).**
B2, B3, E15, E15_id help/hurt the SAME queries. Not mechanism-specific; shared
teacher-supervision bias profile. Suggests evaluating in 4 provenance quadrants:
(1) donor-correct/recipient-wrong, (2) recipient-correct/donor-wrong,
(3) both correct, (4) neither correct. Inheritance = quadrant 1 wins.

**Ship mode sampling inefficiency:** With replacement sampling at 5000 steps from
20K pool, only ~4,424 distinct queries consumed (22.1%). No kd_weight=0 control.
Cannot attribute results to KD; cannot kill unfrozen KD from failure either.

**Three novel mechanisms proposed:**
1. Developmental inheritance (prune/contract the donor, not independent student)
2. Causal capability cassettes (transplant specific donor-only computations)
3. Executable textbook (teacher compiles persistent teaching artifact)

**Meta-pattern from 15 kills (4 recurring failure modes):**
1. Proxy improvement without functional transfer
2. Conditional redundancy (B0 gets first refusal)
3. Human-supplied structure masquerading as inheritance
4. Information in wrong carrier

**E15_id mean advantage: +0.005, SD 0.0036 vs E15 SD 0.0073.** Identity-only
is MORE STABLE. Extra probes add no reliable teacher-private information.

**Codex's call:**
- Let ship mode finish (artifact), but note it's not a KD adjudication
- Run cheap E16 channel-admission tests BEFORE 24 runs
- Begin designing donor-private capability assay NOW
- Move program center from output imitation to parameter/module inheritance

Full output: outputs/codex_deep_think.txt. Entries on blackboard 539efcd4.

---

## 2026-09-04 10:45 -- Codex: Selective E16 + Statistical Correction

Codex heterogeneous treatment follow-up (session 01a06bc1). Processed the
per-query + learning curve evidence. Key outputs:

**Statistical correction on heterogeneous treatment effect:**
My "E15 wins zero easy queries" was partly artifact: rank-1 can't improve by
definition. B0-vs-B0 across seeds shows similar ceiling pattern. But leave-one-
seed-out analysis (difficulty defined by other two seeds) STILL shows the effect:
- Easy: -0.056/-0.072/-0.067 (E15 hurts)
- Medium: -0.005/+0.024/+0.032 (E15 mixed)
- Hard: +0.106/+0.131/+0.094 (E15 helps, only 12-14 queries)

Evidence predicts TARGETED E16 benefit only if teacher selects boundaries that
differ materially from student mining. Indiscriminate teacher top-k not predicted
to work.

**Three distinct mechanisms (Codex taxonomy):**
| Mechanism | What teacher changes | Hypothesis |
|-----------|---------------------|------------|
| E16       | Negative identities | Teacher knows which boundaries matter |
| Weighted B0 | Gradient magnitude | Teacher knows where capacity should go |
| Curriculum | Timing/order | Teacher knows when to introduce examples |

Codex: don't run curriculum before E16 (schedule-dependent, not predicted by
evidence). Weighted B0 is a cheap diagnostic but not prerequisite.

**KEY: Revised E16 design = "Selective Boundary Inheritance"**
Replace all-query teacher top-k with ROUTED teacher arm:
1. Start from student-mined negatives
2. Substitute teacher negatives ONLY where teacher-student disagree
3. Teacher must rank positives safely above proposed negatives
4. Semi-hard band (not just hardest)
5. Retain uniform/self-mined replay floor for all queries

Add pre-training manifest gate (before spending 24 runs):
- Teacher/student top-k turnover by difficulty stratum
- Teacher-corrects-student vs student-corrects-teacher counts
- Eligible-query rate, negative hardness distribution
- False-negative and duplicate-positive rates
If these show low turnover or no teacher advantage, don't run.

Add predeclared secondary outcomes: easy-query regression rate, hard-query rescue
rate, rank-bucket transition matrix, matched learning curves.

Entries e90-e97 on blackboard 539efcd4.

---

## 2026-09-04 10:15 -- Cross-Seed Consistency: Teacher Boundary Knowledge is Real

Follow-up analysis on the heterogeneous treatment data. KEY QUESTION: does the
teacher help the SAME queries across training seeds, or is it noise?

**Result: 18 queries consistently helped, 20 consistently hurt, 80 neutral, 82
inconsistent across all 3 seeds.** Teacher's net effect is ~zero because help and
hurt cancel. But 41% of queries have inconsistent fates — help vs hurt depends
on training seed, suggesting significant noise in teacher knowledge transfer.

**Hard queries: teacher helps CONSISTENTLY.** Of 23 queries that are hard (B0
rank 6+) in at least one seed: msmarco_404 (rank 4-6 to rank 1 ALL seeds),
msmarco_421 (rank 3-7 to rank 1 ALL seeds), msmarco_536 (rank 3-9 to rank 1
ALL seeds), msmarco_559 (rank 3-6 to rank 1 ALL seeds). Genuine, reproducible
boundary knowledge.

**B0 difficulty is 71.5% stable across seeds.** Query hardness is structural,
not noise. Training randomness changes outcomes for 28.5% of queries.

**E15 and B2 help the SAME queries (Jaccard 0.68-0.75).** Different KD mechanisms
(teacher-indexed heads vs standard KD) access the same underlying teacher
knowledge. The mechanism doesn't matter for WHICH queries benefit.

**Implication for E16:** Teacher boundary knowledge is genuine and consistent for
hard queries. The problem is uniform application — it helps hard queries but
damages easy ones. E16's data-selection approach naturally concentrates teacher
knowledge where it helps (harder training examples) without gradient interference
on easy queries. The high E15-B2 Jaccard suggests the knowledge channel is
robust — any mechanism that targets the right queries should work.

**Risk for E16:** 82 inconsistent queries (41%) suggest much of teacher's apparent
knowledge is training-noise interaction, not deep structure. If teacher-mined
negatives are mostly from the "inconsistent" band rather than the "consistently
hard" band, E16 may not help.

---

## 2026-09-04 09:30 -- Deep Per-Query Analysis: Heterogeneous Treatment Effect

CPU analysis while ship mode runs on GPU. Mined overlooked signal in E1.5 data.

**Finding 1: KD helps hard queries, CORRUPTS easy queries.**
Across all 3 seeds, consistently:
- EASY (B0 rank-1, ~130/200): E15 wins 0, loses ~30. Mean delta -0.135.
- MEDIUM (B0 rank 2-5, ~55/200): E15 wins ~30, loses ~13. Mean delta +0.195.
- HARD (B0 rank 6+, ~15/200): E15 wins ~11, loses ~3. Mean delta +0.135.
Net negative because easy queries dominate (66%). The teacher genuinely helps
on hard queries but KD loss pulls representation uniformly, corrupting already-
correct answers.

**Finding 2: KD arms learn fast then stall/regress. B0 keeps learning.**
B3 has HIGHEST MRR at step 200 (0.713) but STALLS to 0.724 at step 600.
B0 has LOWEST MRR at step 200 (0.657) but CLIMBS to 0.779 (highest final).
E15 regresses from 0.754 to 0.737 in seed 137 after step 400.
Teacher gradient helps early (head start) but becomes conflicting later.

**Finding 3: B4c catastrophe = near-uniform rank distribution.**
Seed 271: 19 rank-1, 18 rank-32, everything between. Complete training
collapse into a local minimum. Seeds 42/137: 128/123 rank-1.

**Finding 4: Identity probe beats full teacher-indexed heads (seed 137).**
E15_id wins 30 vs E15 18 per-query. Extra parameters add noise.

**Implication for E16:** These findings directly validate E16's mechanism.
The teacher's value is in difficulty assessment (which queries/negatives are
hard), not in its output distributions (loss modification). E16 uses the
teacher ONCE for data selection, then removes it — avoiding gradient
interference. Two simpler alternatives also motivated: teacher-weighted
contrastive loss (per-query weighting by teacher-student agreement) and
teacher-guided curriculum (sort by difficulty).

Codex E16 design gate: CONDITIONAL GO (separate session, just completed).
Codex follow-up sessions fired with this new evidence.

---

## 2026-09-04 06:00 -- E1.5 Codex Evidence Gate: INCONCLUSIVE

Codex Research Integrity Auditor reviewed E1.5 results. Verdict: INCONCLUSIVE.

Key corrections (adopted verbatim):
- "KILL confirmed" overclaims the evidence. Valid-seed CI (df=1) includes both
  0 and +0.005 threshold. Cannot resolve terminal kill.
- All 3 seeds use the SAME 200 eval query IDs. These are training-randomness
  replicates on one fixed split, not independent data replications.
- B0 dominance is scoped to "frozen-encoder, fixed-split configuration."
  The frozen-encoder restriction may prevent representation changes needed
  for knowledge inheritance.
- B4c instability (seed_271) is genuine evidence of algorithmic fragility.
- Combined evidence is "consistently unfavorable" to tested mechanisms but
  says nothing adverse about the axiomatic Eklavya mission.

Claim ceiling: "On one fixed MS MARCO evaluation set under frozen-encoder
training, direct contrastive supervision outperformed every tested KD arm
across three training runs."

Reclassified: KILL -> INCONCLUSIVE (narrow negative) in all canonical surfaces.
Ship mode (unfrozen encoder, standard KD) remains the correct next artifact.

Full verdict: outputs/E1_5_text/codex_evidence_gate.txt

Strategic Codex recommendation (separate session): E16 Boundary Inheritance.
Teacher-guided negative mining is the highest-leverage untested mechanism.
The teacher picks training comparisons (positive-aware hard negatives, false
negative relabeling); student trains with pure B0 contrastive loss. Teacher
moves upstream from the loss into the training distribution. 5 ranked
mechanisms: (1) teacher-guided mining, (2) teacher-generated data, (3) weight
inheritance via pruning, (4) teacher-as-curriculum, (5) intermediate features.

---

## 2026-09-04 05:15 -- E1.5 COMPLETE: B0 wins all seeds (narrow negative)

All 18 arms (6 arms x 3 seeds) complete. Summary.json regenerated with all seeds.

Cross-seed results (MRR, mean across 3 seeds):
1. B0_contrastive: 0.769 (no teachers, pure contrastive)
2. E15_teacher_idx_id: 0.750 (identity probe only)
3. B2_kd_single: 0.748 (single teacher KD)
4. E15_teacher_indexed: 0.745 (per-teacher heads + teacher distributions)
5. B3_kd_avg_cal: 0.735 (calibrated multi-teacher avg)
6. B4c_matched: 0.573 (same heads, gold targets -- broken on seed_271)

E15 vs B4c (valid seeds 42, 137): deltas = -0.001, -0.010. Mean = -0.005.
Teacher-indexed heads add NOTHING vs matched architecture with gold targets.

B4c seed_271 catastrophic failure (MRR=0.22 vs 0.75 on other seeds) is genuine.
B0 winning all seeds is the strongest signal -- pure contrastive beats all KD.

Codex evidence gate fired (b0xctj7kg). Thermal guard fix committed (a346a3f).

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
