# Manufacturing, Process Engineering & Operations Research Mental Models for Knowledge Transfer

*A portfolio of ALIVE frameworks for when classical KD plateaus.*
*Context: Sutra-Dyad, 188M byte-level. Classical KD = "match teacher logits everywhere with equal fervor" has stalled.*
*Written for: a reader 3 months from now who picks ONE and runs with it.*

---

## TL;DR — The most absent manufacturing insight in ML-KD

**Classical KD is a shop floor with no andon cord, no SPC chart, no bottleneck analysis,
and no load-leveling.** It is what manufacturing looked like in 1910 — push as much
material through as possible, hope for the best, inspect at the end, scrap whatever
fails QC. The Toyota Production System spent fifty years figuring out that this is
the wrong way to make cars. ML-KD has not yet had its Ohno moment.

The single most absent insight is **Theory of Constraints** (Goldratt, 1984): in any
production system, exactly one resource is the binding bottleneck at any given moment,
and *every other resource subordinates to it or wastes its capacity producing inventory
that piles up in front of the bottleneck.* In ML-KD this is screamingly visible once
you look: the student has a small number of *concept-bottleneck positions* per batch
(positions where it is genuinely confused, where teacher signal would actually move the
loss), and the rest are either trivially correct (no signal) or hopelessly out of
distribution (no traction). Classical KD spreads the gradient budget uniformly across
all positions — meaning it spends ~95% of the supervisory budget on positions that are
not the bottleneck. The bottleneck positions get 5% of the gradient *and* are diluted
by averaging with the wasted 95%. No factory would be run this way for a week.

A KD system that explicitly identifies the bottleneck (top-k highest-regret positions per
batch) and **subordinates all other resources to feeding the bottleneck** (concentrate
teacher signal, increase teacher diversity, add longer context) would extract more
intelligence per gradient step than any uniform method, by a margin that should be
visible within 2K steps.

**Runner-up**: Jidoka / andon cord. Classical KD has no mechanism for the *student* to
stop the line and say "this teacher's signal is hurting me right now." It must accept
every teacher byte as ground truth. A student that can refuse a bad teacher signal at
specific positions (and have that refusal logged and inspected) is a student that can
recover from bad data without manual intervention.

---

## Model 1 — Theory of Constraints: Find the bottleneck, subordinate everything else

**The model in one sentence.** Goldratt's *The Goal*: in any production system, throughput
is limited by exactly one bottleneck at a time, and the only operationally rational policy
is to (1) identify it, (2) exploit it (run it 100%), (3) subordinate all other resources
to keeping it fed and unblocked, (4) elevate it (add capacity), and (5) when it stops
being the bottleneck, find the new one — repeat forever.

**ML translation.**
- **Throughput** = bits-per-byte reduction per unit of GPU-hour spent.
- **Resources** = positions in the batch, teachers in the panel, gradient budget,
  optimizer steps, VRAM, attention heads, MLP capacity.
- **The bottleneck at any given step** = the small set of positions where the student's
  current loss is high, the gradient norm is high, AND there is teacher signal with
  enough confidence to actually move the parameters. Three conditions must coincide:
  *student doesn't know it*, *student is plastic to it*, *teacher knows it*. Few
  positions per batch satisfy all three. Those are the bottlenecks.
- **Wasted capacity** = the other ~95% of positions. Trivially-correct bytes contribute
  near-zero gradient. Hopelessly-OOD bytes contribute noise. Both consume forward-pass
  compute and dilute the averaged loss.

**Sutra thought experiment.** Add a *bottleneck identifier* as a hot path in the trainer.
At each step:
1. Forward pass on batch. Compute per-position loss `L_p` and per-position gradient norm
   proxy `g_p` (e.g., ||student_logit - teacher_logit||).
2. Compute per-position teacher coverage `c_p` = max teacher confidence at p.
3. **Bottleneck score** `B_p = L_p × g_p × c_p`. A position is a bottleneck iff all three
   are high simultaneously.
4. Pick the top 5% of positions by `B_p`. These are the "constraint."
5. **Exploit**: zero-out gradient contribution from non-constraint positions for half the
   training run (or weight them at 0.1×). This concentrates the optimizer on the
   constraint.
6. **Subordinate**: route extra teacher diversity *only* to constraint positions —
   non-constraint positions get the cheapest teacher; constraint positions get the full
   teacher panel ensemble.
7. **Elevate**: when the constraint loss drops below the non-constraint loss, the
   constraint has shifted — recompute and repeat.

This is *concentration of force*. It is what every factory manager learns in their first
week and what no KD paper I have read has ever actually implemented as a hot-path
gradient policy. (Curriculum learning is adjacent but operates on whole examples and on
a slow schedule; this is per-position, per-step, with explicit bottleneck identification.)

**What it reframes about KD.** The plateau is the consequence of the system running its
non-bottleneck resources at 100% and its bottleneck at <5% utilization. The student is
not actually being trained on the things it doesn't know. It is being trained ~95% on
things it already knows (cheap entropy reduction on easy bytes) and ~5% on things it
genuinely needs (the actual bottleneck). The total loss curve looks like learning, but
the *capability frontier* moves only as fast as the underutilized 5% allows. Classical
KD has been running the wrong machine flat-out for years.

This also explains why "more teachers" doesn't help once you have a few. Adding teachers
adds capacity to non-bottleneck resources. The bottleneck is *student plasticity at
high-regret positions*, not teacher availability.

**Testable prediction.** Run two students, identical except:
- (A) Standard uniform KD across all positions.
- (B) Bottleneck-concentrated KD: 80% of gradient comes from top-5% positions by
  `B_p = L_p × g_p × c_p`, the rest of the batch contributes 20%.

Within 2000 steps at the same compute budget, (B) should show:
- Strictly lower BPB on the *hardest decile* of held-out positions (the bytes the student
  was failing at when training started).
- Equal or slightly worse BPB on the easiest decile (we explicitly stopped training there).
- Net BPB gain of 0.05-0.15 if the bottleneck hypothesis is correct.

If (B) is not strictly better on the hard decile, the bottleneck identification is
wrong (probably `c_p` is the weak factor — teachers don't actually know what the student
needs). If (B) is worse overall but better on hard decile, we have rediscovered the
classical exploration/exploitation tradeoff and need to schedule the concentration ratio.
Either outcome is informative.

---

## Model 2 — Jidoka & the Andon Cord: Any worker can stop the line

**The model in one sentence.** Toyota's second pillar: *autonomation* — machines (and
workers) have the authority and the means to halt production the instant a defect is
detected, because the cost of stopping the line is always less than the cost of building
defective inventory on top of a problem and discovering it three workstations later.

**ML translation.**
- **The line** = the training loop, processing batch after batch.
- **The worker** = the student model, processing each batch position by position.
- **The defect** = a teacher signal that, if accepted, would move the student in a
  direction inconsistent with what the student already strongly believes from other
  evidence (high cross-teacher disagreement, teacher confidence collapse, teacher OOD
  signal, teacher hallucination at this position).
- **The andon cord** = a per-position gating mechanism that lets the student *refuse* a
  teacher signal at position `p` if the signal fails a sanity check, and log that
  refusal for later inspection.
- **Stop the line** = set the KD loss contribution from that (teacher, position) pair
  to zero for this step *and write a row to a defect log*.

**Sutra thought experiment.** Add a defect detector and an andon log to the KD pipeline.
At each step, for each (teacher t, position p):
1. Compute *cross-teacher consensus* `C_p` = 1 - JSD(teacher_t logits, ensemble logits
   minus t). High when t agrees with the rest of the panel.
2. Compute *student prior* `S_p` = student logits from previous checkpoint (cached).
3. Compute *teacher self-confidence* `Q_t,p` = -entropy(teacher_t logits at p).
4. **Defect score** `D_t,p = high if (C_p low) AND (Q_t,p low) AND (KL(S_p, teacher_t)
   high)`. I.e., this teacher disagrees with all other teachers, is itself unsure, and
   wants the student to move sharply.
5. If `D_t,p` exceeds threshold, set `w_t,p = 0` for this step *and write* (step, batch_id,
   position, teacher, defect_score, top-3 teacher tokens, student top-3) to
   `andon_log.jsonl`.
6. After every 1000 steps, *inspect the andon log*: cluster the defects, see if a
   particular teacher dominates, see if particular byte n-grams dominate. This is
   genchi genbutsu — go and see.

**What it reframes about KD.** Classical KD is the assembly line where every worker
must accept every part handed to them, even if it is visibly defective. The whole
quality-management literature exists because this model produces shit cars. It also
produces shit students. The reason ML-KD has not figured this out is cultural: we treat
the teacher as ground truth, when in fact the teacher is *a noisy upstream supplier*
whose output should be sample-inspected at every step and rejected when it fails QC.

The deeper reframe: **the defect log itself is the most valuable artifact of the
training run.** It tells you *which teachers are bad at what*, *which positions are
intrinsically ambiguous*, and *which byte patterns trigger teacher hallucination*. None
of this information is captured by classical KD, which silently averages defects into
the gradient. After 100K steps you have a million-row defect log. That is the
curriculum for the next student, and the QC report for each teacher.

**Testable prediction.** Run a 5K-step Sutra-Dyad training with andon-cord KD.
- Expect 2-8% of (teacher, position) pairs to be flagged as defective in any given batch.
  (If <0.5% you have not set the threshold tightly enough; if >20% the defect detector
  is broken.)
- Expect one teacher in the panel to account for >40% of defects. (Manufacturing data
  is always Pareto-distributed; supplier quality always is.)
- Expect held-out BPB to improve by 0.03-0.08 vs identical training without the andon
  cord, *primarily* because the worst teacher's worst signals are no longer corrupting
  the student. The benefit should be larger on the *tail* of the loss distribution
  (hard examples) than on the mean.
- Inspecting the andon log should reveal at least one systematic failure mode of one
  teacher that we can describe in a single English sentence ("teacher X hallucinates
  a closing brace whenever an opening brace appeared more than 200 bytes ago"). If we
  can't, the defect detector is too noisy to be useful.

---

## Model 3 — Statistical Process Control: 6-sigma signals only, please

**The model in one sentence.** Shewhart and Deming's insight: every process has *common-cause
variation* (the inherent noise of the system, normally distributed around the mean) and
*special-cause variation* (an actual signal — a real change in the process), and you cannot
manage the former by reacting to it; treating noise as signal *increases* total variance
because every adjustment is itself a perturbation. Only react to deviations beyond ~3σ
(or, for high-stakes capability claims, 6σ — about 3.4 events per million).

**ML translation.**
- **The process** = the KD training loop.
- **The output metric** = held-out BPB (and downstream eval scores).
- **Common-cause variation** = batch-to-batch BPB jitter, eval-set sampling noise,
  optimizer noise, dropout noise. For Sutra-Dyad at this scale, this is on the order of
  ±0.005 BPB per eval. *Anything within ±3 × 0.005 = ±0.015 BPB of the baseline is
  noise.*
- **Special-cause variation** = a real change (a new mechanism actually working, a real
  regression from a bug, a teacher whose signal genuinely helps). For us to claim it,
  the change must be ≥6σ ≈ 0.03 BPB and reproducible across seeds.
- **The control chart** = a running record of every eval BPB with the upper and lower
  control limits drawn at ±3σ from a stable baseline.

**Sutra thought experiment.** Build a real SPC dashboard for Sutra-Dyad before any more
KD experiments. Specifically:
1. Run the *current* Sutra-Dyad training 5 times from scratch with different seeds to
   the same step count. Record eval BPB at every checkpoint.
2. Compute σ across seeds at each step. This is your common-cause variation envelope.
3. Plot UCL = mean + 3σ and LCL = mean − 3σ as horizontal lines on every future training
   curve.
4. **Decision rule**: any new mechanism's BPB curve must spend *consecutive* checkpoints
   below the LCL of the baseline curve before we declare it a real win. A single
   below-LCL point is noise. Two consecutive is signal. Three is publishable.
5. Adopt Western Electric rules: 1 point beyond 3σ, 2 of 3 beyond 2σ, 4 of 5 beyond 1σ,
   8 in a row on one side of the centerline. Each is a different signature of
   special-cause variation.

**What it reframes about KD.** Most KD "wins" reported in the literature, including some
of our own past Sutra results, are within ±1σ of the noise floor and were celebrated as
signal. This is not malice — it is the absence of a control chart. Every reported "+0.01
BPB improvement from KD trick X" was almost certainly noise. The plateau is partly real
and partly an artifact of measuring through the noise floor: we can't see the real
improvements because we're celebrating the false ones, and the real ones (which would
need to clear 6σ ≈ 0.03 BPB) are being demanded of mechanisms that genuinely only deliver
±0.005 BPB per increment.

The SPC reframe is brutal: *we have been running our experiments at insufficient
statistical power for years, and most of our learning is from the few experiments where
the effect happened to be larger than noise.* We do not know which of our smaller-effect
mechanisms actually work.

**Testable prediction.** After establishing the noise floor with 5 seeds, re-run the last
3 KD mechanisms we believed "worked." Predictions:
- At least 1 of 3 will be inside the ±1σ envelope = noise.
- At least 1 of 3 will be in the ±1-3σ band = ambiguous, needs more seeds.
- At most 1 of 3 will clear the 3σ threshold = real but small.
- 0 of 3 will clear 6σ at the seed counts we have actually run.

If this prediction holds, we should *retire* every KD mechanism that has not been
demonstrated at 3σ across at least 3 seeds, and commit to never reporting another
result without a control chart. This is also the answer to "decisive margins" feedback:
6σ on a real eval is the only definition of decisive that survives an adversarial
reading.

---

## Model 4 — Heijunka (Load Leveling): Don't push waves the student can't absorb

**The model in one sentence.** Toyota's heijunka: instead of producing in batches that
match incoming demand spikes (which causes bullwhip effects, bottlenecks, and overtime
chaos downstream), you *level* the production schedule — small mixed batches at a steady
takt time — so that every downstream station gets a smooth, predictable, absorbable load.
The total work is the same; the *variance* is what kills you.

**ML translation.**
- **The production schedule** = the sequence of (teacher, signal_type, difficulty) tuples
  delivered to the student each step.
- **The bullwhip** = when a single source dominates a batch (a long stretch of code, then
  a long stretch of math, then a long stretch of natural language) the student's parameters
  oscillate; the gradient at step 100 is fighting the gradient at step 200, and net
  progress is small.
- **Takt time** = the steady rate at which the student can absorb new information without
  destabilizing previously-learned patterns. (Empirically: when EMA loss is smooth, you're
  at takt; when it's spiky, you're above takt and the student is being whipped.)
- **Mixed small batches** = each batch should contain a *representative cross-section* of
  domains, difficulties, and teachers — not a homogeneous chunk.

**Sutra thought experiment.** Replace the current shard-shuffled training schedule with a
heijunka scheduler:
1. Tag every training byte with (domain, teacher, difficulty_bucket).
2. Define a target mix per batch — e.g., for an 8K-byte batch: 2K code, 2K natural
   language, 1K math, 1K dialog, 2K general — and within each, mix easy/medium/hard
   in 4:3:3 ratio.
3. The data loader must honor this mix on every batch, not just on average. (This is the
   crucial difference from random shuffling — random achieves the right mix on average
   but with high variance per batch.)
4. Track *gradient cosine similarity between consecutive batches.* Heijunka should keep
   this above 0.7 (consecutive batches point in similar directions). Random shuffling
   typically gives 0.4-0.6 (bullwhip).
5. Adjust the mix dynamically: if the student's loss on a given domain is rising, *reduce*
   that domain's share next batch (overproduction, glut); if loss is falling fast, *hold
   steady* (let the student fully absorb before moving on). This is closed-loop heijunka.

**What it reframes about KD.** The standard mental model is "give the student lots of
data, the optimizer will figure it out." The factory floor mental model is "the optimizer
is a downstream worker, and it has a finite absorption rate per step. Exceed that rate
and you get destructive interference, not faster learning." The plateau in classical KD
may be partly *self-inflicted bullwhip* — we are pushing more variance through the
optimizer than it can level out, and the parameters are spending most of their plasticity
budget undoing the previous batch instead of consolidating learning.

This connects to Liquid AI's "data-mixing matters" results, but goes further: it's not
just the *long-run* mixture that matters, it's the *per-batch variance of the mixture*.
Two training runs with identical long-run mixtures but different per-batch variances will
produce different students.

**Testable prediction.** Run 3 students, identical except for batch composition:
- (A) Random shuffle of all shards.
- (B) Block-shuffled (1K bytes from one source, then 1K from another, etc.) — *high*
  per-batch variance.
- (C) Heijunka-mixed: every batch contains the target mix in fixed ratios.

Predictions:
- Gradient cosine similarity between consecutive batches: B (0.3) < A (0.5) < C (0.8).
- Loss curve smoothness (lower std of step-to-step loss change): B < A < C.
- Final BPB after 5K steps: B (worst) < A < C (best), with the gap C-A being small
  (~0.02 BPB) but real, and the gap A-B being larger (~0.05-0.10).
- Most importantly: *time-to-target-BPB* should be 15-25% faster for C than A,
  because C wastes less of its gradient budget on undoing the previous batch.

If C is not faster, then either the optimizer's leveling capacity is higher than we
think (it's already absorbing the bullwhip), or the per-batch mixture is dominated by
within-batch averaging effects we haven't modeled. Either way, informative.

---

## Model 5 — Poka-Yoke: Make the wrong thing physically impossible

**The model in one sentence.** Shigeo Shingo's mistake-proofing principle: the right
response to "workers occasionally make mistake X" is not "train workers harder" or
"add an inspection step" — it is to *redesign the part or the fixture so that mistake X
becomes physically impossible to make.* The USB-C cable that plugs in either way is
poka-yoke. The IV bag fitting that *cannot* connect to an oxygen line is poka-yoke. The
asymmetric SIM card slot is poka-yoke.

**ML translation.**
- **The mistake** = a class of failure modes the student keeps falling into despite KD.
  Examples: hallucinating a closing bracket without an opening one. Predicting whitespace
  after whitespace ad infinitum. Generating UTF-8 byte sequences that are not valid UTF-8.
  Confusing two domains because their byte patterns overlap (Python `def` vs English
  "def").
- **The training-time poka-yoke** = an architectural or loss-function change that makes
  the failure *structurally impossible*, not merely *statistically discouraged*.
- **Why this matters more for byte-level than token-level** = byte-level models can
  generate physically invalid outputs (malformed UTF-8). Token-level models cannot.
  The byte-level setting is *full of opportunities* for poka-yoke that the token-level
  setting forecloses.

**Sutra thought experiment.** Catalog the top 10 systematic Sutra-Dyad failure modes from
generation samples, and for each, design a poka-yoke:
1. **Failure**: invalid UTF-8 byte sequences in output. **Poka-yoke**: at the output
   layer, mask out byte values that would make the current incomplete UTF-8 sequence
   invalid. Compute the mask from the trailing bytes of the context. Cost: O(1) per
   token. Effect: invalid UTF-8 becomes physically impossible at decode time.
2. **Failure**: unmatched brackets in code. **Poka-yoke**: maintain a small stack of open
   brackets in the model's *input representation* (not the parameters), penalize closing
   bytes that don't match top-of-stack. This is a *structural* constraint encoded at
   the data layer.
3. **Failure**: domain confusion (Python keyword vs English word). **Poka-yoke**: include
   a learned domain-tag byte at the start of every shard during training, so the student
   *cannot* see code without first seeing the code-domain marker.
4. **Failure**: teacher signal corrupting student belief at OOD positions. **Poka-yoke**:
   gate teacher KD loss by `confidence(teacher) × in-distribution-score(position)`. Below
   threshold, the teacher physically cannot contribute gradient. (This is the andon cord
   from Model 2, but now codified into the loss function rather than as an alarm.)
5. **Failure**: optimizer instability when one teacher is much sharper than others.
   **Poka-yoke**: temperature-normalize all teachers to the same effective entropy
   *before* mixing them into the KD target. Sharp teachers physically cannot dominate
   the mixture.

**What it reframes about KD.** Most ML-KD papers frame failure modes as things to
*discourage* (add a penalty term, weight more carefully, train longer). Poka-yoke says:
discouragement is the wrong frame. Every discouragement is a probabilistic constraint
that the model can violate when other gradients overwhelm it. The right move is to make
the failure *structurally impossible* via the architecture, the data representation, or
the masking — not via the loss landscape. A KD student with 5 well-chosen poka-yokes will
outperform a student with 5 well-tuned penalty terms, because the penalty terms compete
for gradient budget while the poka-yokes do not consume gradient at all.

The deeper reframe: *the gradient is precious; spend it on capability, not on enforcing
constraints that should be structural.* The current paradigm wastes 10-30% of training
signal teaching the model not to do things it can be physically prevented from doing.

**Testable prediction.** Implement poka-yoke #1 (UTF-8 validity mask) and poka-yoke #5
(teacher temperature normalization) on top of the current Sutra-Dyad KD.
- Generation samples: invalid UTF-8 rate should drop from current ~0.3% to 0% (this is
  trivially true by construction; the question is whether overall quality also improves).
- Eval BPB: should *improve* by 0.01-0.03, not because of UTF-8 directly but because
  the gradient previously spent on penalizing invalid UTF-8 is now free for other
  capability.
- Generation coherence: with teacher temperature normalization, sample diversity should
  increase (no single sharp teacher dominating), measured by output token-distribution
  entropy.
- If poka-yoke #1 *worsens* eval BPB, it means the model was using the freedom to
  generate invalid bytes as a useful internal scratch space, which would itself be a
  fascinating finding (and would suggest poka-yoke is wrong for that specific failure).
  Either outcome is informative.

---

## CROSS-POLLINATION: How these connect to each other and to existing portfolios

### Within manufacturing (this portfolio)
- **Theory of Constraints + SPC**: Bottleneck identification needs SPC to distinguish
  "this position is genuinely the bottleneck" from "this position has high variance and
  the noise floor is fooling the bottleneck score." Implement SPC first; then bottleneck
  identification is interpretable.
- **Jidoka + Heijunka**: The andon cord generates defect data; heijunka is the policy
  that *uses* defect data to level future batches (avoid sending more bytes from a
  domain whose defect rate is spiking). Together they form a closed-loop quality system.
- **Poka-yoke + Theory of Constraints**: Once a class of failures is poka-yoke'd into
  impossibility, the bottleneck shifts. After UTF-8 poka-yoke, the next bottleneck is
  probably semantic coherence — and you only see this clearly *after* the trivial
  failure mode is structurally eliminated.
- **SPC + Heijunka**: Heijunka's per-batch variance reduction is the operational
  implementation of SPC's "reduce common-cause variation." They are the same idea at
  two scales (per-batch vs over training).
- **All five together**: a complete TPS-style training factory. Heijunka levels the
  input. Poka-yoke prevents structural failures. Andon flags substantive failures.
  Bottleneck analysis concentrates the gradient. SPC distinguishes signal from noise.

### Cross-portfolio resonances
- **Economics (#10) + Theory of Constraints**: The "Hayekian teacher market" sets prices
  via revealed preferences. Theory of Constraints sets prices via bottleneck location.
  These are dual: Hayek says "the bottleneck reveals itself through prices," Goldratt
  says "find the bottleneck and price-discriminate around it." Combine: market prices
  set by bottleneck shadow prices. This is the linear-programming dual that economists
  and operations researchers have known is the same theorem since Dantzig.
- **Information theory (#2) + SPC**: 6σ on BPB and "the channel capacity of the BPB
  measurement" are the same number under different names. Shannon would call SPC's
  control limits the *measurement noise floor*.
- **Dynamical systems (#7) + Heijunka**: Bullwhip is dynamical-systems instability in
  the data → optimizer transfer function. Heijunka is low-pass filtering of the input
  signal to match the optimizer's bandwidth. Same mechanism, different vocabulary.
- **Biology (#4) + Jidoka**: Andon cord = the immune system rejecting a non-self
  signal. The cytokine that says "stop, this is foreign" is the andon. Defect log =
  immunological memory.
- **Cogsci (#6) + Poka-yoke**: Affordances are environmental poka-yokes — a door
  handle physically affords pulling, not pushing. Architectural poka-yokes are the
  ML equivalent of designing an environment with the right affordances for the
  student to fall into the right behaviors.
- **Computer science (#11) + SPC**: Type systems are compile-time poka-yoke; SPC
  is runtime quality control. Together they cover the full software-quality spectrum,
  and together they cover the full ML-quality spectrum.
- **Art/design (#15) + Heijunka**: Composition (in painting, music, writing) is
  heijunka — the artist levels the perceptual load on the audience by mixing
  intensity with rest. The optimizer is also an audience.

### The deepest cross-portfolio insight
Manufacturing, economics, biology, and dynamical systems all agree on the same meta-pattern:
*systems with closed feedback loops on quality, finite resources, and bottleneck
identification outperform systems without these features by margins that compound over
time.* Classical KD has none of the four. Building any one of them in is plausible to
yield 0.05-0.15 BPB; building all four creates a *system* that is qualitatively different
from "KD with tricks" — it is a *training factory*, and the manufacturing literature has
a hundred years of evidence that training factories outperform craft workshops by
factors of 3-10× in throughput at constant quality.

The Sutra-Dyad bottleneck is not the architecture. It is that we are running a craft
workshop, and the BPB plateau is the throughput ceiling of every craft workshop ever
built. The exit from the plateau is not a better tool; it is a different mode of
production.
