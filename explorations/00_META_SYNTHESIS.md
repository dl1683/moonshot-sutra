# 00 — META-SYNTHESIS: What 18 Mental-Model Portfolios Say About Knowledge Transfer

**Purpose.** Eighteen independent portfolios (90+ models) were written, each mining one discipline for frameworks that reframe classical KD. Read in isolation they are essays; read together they are a triangulation. When seven or ten unrelated fields converge on the same observation, that observation is almost certainly a feature of the problem, not of any one lens. This document extracts those convergences — and also the genuine disagreements — so that the next mechanism Sutra prototypes is chosen by evidence across the entire survey, not by the persuasiveness of the most recently read file.

**Current Sutra context.** Sutra-Dyad, 188M byte-level. Classical logit-KD plateaued at eval BPB ≈1.40. Committee-regret window-weighted sampling (R5 design, 9/10 confidence) FAILED in the last experiment — training BPB crashed to 1.33 while eval climbed to 1.46. Clean pool-overfitting. The falsification matters: it rules out the *input-reweighting* family of fixes. The survey below was written partly to find out what else exists.

---

## 1. CONVERGENT PATTERNS

Below are the themes that appeared *independently* in three or more portfolios, each reached from a different starting vocabulary.

### 1.1 Transfer STRUCTURE, not SURFACE
**Portfolios:** 02 (MDL/program), 05 (functoriality, relational KD), 06 (structure-mapping/Gentner), 08 (Frege compositionality), 11 (program synthesis, CRDT constraints), 12 (IIT integration, HOT), 13 (density matrix / partial trace), 15 (style vs content decomposition), 16 (sensorimotor contingencies).

**Strongest form.** The teacher's competence is an *algorithm*, not a probability table. Its outputs are traces of that algorithm on specific inputs, and those traces are *sum-decomposable* across positions, tokens, and contexts. The algorithm is *not*. KL-on-marginals is therefore a rate-limited channel that by construction erases everything intensional — functoriality, composition rules, equivariance, entanglement between hidden state and output, compositional generalization. Every plateau that reports "the student imitates outputs but fails on compositional/OOD evaluation" is this gap.

**Testable consequence for Sutra.** Replace (or augment) per-token KL with signals that *are not sum-decomposable*: paired-contrast losses `T(x)-T(x')` where `x'` is a structured perturbation of `x` (insertion, deletion, substitution, paraphrase, prefix-extension), functorial consistency on composed inputs, and IIT-style bipartition integration gaps. If any of these losses improves OOD/compositional eval while barely moving aggregate BPB, the intensional-vs-extensional diagnosis is correct and BPB is the wrong metric.

### 1.2 Sparse selection beats dense matching
**Portfolios:** 01 (CLS replay, predictive coding), 04 (affinity maturation), 06 (Zone of Proximal Development), 09 (MAP-Elites), 10 (Hayek prices, TOC), 11 (cache-coherence miss-driven KD), 15 (atelier sparse correction), 16 (active perception), 17 (Theory of Constraints).

**Strongest form.** At any moment, a small fraction of (position, teacher, signal) triples carry almost all the useful gradient. The remainder are rehearsal (student already knows), noise (teacher is uncertain), or unreachable (student lacks capacity to use the signal). Uniform KD spends ~95% of teacher compute on non-load-bearing positions and dilutes the 5% that actually move capability.

**Testable consequence for Sutra.** An event-driven KD — query the teacher only at positions where `B_p = student_loss × gradient_norm × teacher_confidence` exceeds threshold — should match or beat uniform KD at a fraction of teacher FLOPs. The atelier version (Model 15.1) is the simplest realization: correct only the single highest-surprisal position per rollout.

### 1.3 The student's own state is the missing conditioning variable
**Portfolios:** 02 (rate-distortion w.r.t. student capacity), 06 (tutoring, schema theory), 07 (observability Gramian, Kalman gain, adaptive control), 12 (HOT — student's metacognition), 14 (catalysis — signal vanishes at optimum), 16 (motor babbling, intrinsic motivation / learning progress), 18 (TMR cued replay).

**Strongest form.** Classical KD is a one-way pipe: `teacher → student`. Every portfolio that asked "what does the *student* know, right now, about its own state?" concluded that closing this loop is non-optional. The teacher's signal can only be weighted, gated, or selected sensibly once we can answer (i) how confident is the student, (ii) where is its representation low-rank / observability-poor, (iii) where is its loss *currently dropping* vs plateaued. Without these we are running open-loop control on a non-stationary plant.

**Testable consequence for Sutra.** A trainable 5M-param "student predictor" (Model 06.5 — forecasts student logits from student hidden state) unlocks: (a) teacher weight = 0 where the predictor says the student will be right, (b) early-exit at inference, (c) cue-vector for TMR replay, (d) per-position observability diagnostic. One artifact, four downstream wins.

### 1.4 Offline / batch / equilibration phases are structurally missing
**Portfolios:** 01 (CLS replay), 03 (RG flow phase transitions), 09 (open-endedness), 14 (Le Chatelier equilibration, nucleation), 17 (heijunka), 18 (all four — SHY, REM/NREM, TMR, semanticization).

**Strongest form.** Biology, chemistry, and manufacturing all *prohibit* the ML default of uninterrupted online stepping. Real systems alternate between forcing phases and relaxation phases — otherwise signal does not equilibrate, the parameter distribution never settles into the shape implied by the current forcing, and most of the gradient budget goes to undoing yesterday's batch. Sutra's loss curve — monotone with no announced pauses, consolidation phases, or renormalization — is precisely the "never at equilibrium" signature chemistry identifies (Le Chatelier).

**Testable consequence for Sutra.** The cheapest falsifier: pause training at the plateau, hold data/teacher fixed, run N no-data steps (or replay buffer steps only). If BPB continues to drop, we have never been at equilibrium. If it flattens, we are. Either answer is actionable — the first says "add offline consolidation," the second says "the plateau is real; change perturbations."

### 1.5 Averaging teachers destroys their information; the right operation is NOT the mean
**Portfolios:** 03 (replica theory — the mean is a point nobody lives at), 04 (HGT grafts, stigmergy), 05 (Wasserstein barycenter, sheaf gluing), 07 (Kalman fusion with state-dependent gain), 10 (Markowitz covariance-aware portfolio, VCG second-price), 11 (CRDT intersection), 12 (global workspace competition, not averaging), 15 (style-from-A + content-from-B, never average), 16 (sim-to-real invariant subspace, not mean), 18 (temporally separated cued replay, not mixed).

**Strongest form.** *Every* portfolio that asked "how to combine teachers?" answered that mean/weighted-KL is the wrong operator. The right operator is one of: set intersection (CRDT, sheaf gluing), covariance-aware portfolio (Markowitz), Wasserstein barycenter (mode-preserving), competition winner (workspace), role-factorization (style/content), temporal separation (REM/NREM, TMR), or orthogonal-subspace assignment (morphogen dims 0-7 vs 8-15).

**Testable consequence for Sutra.** This is the core Ekalavya result waiting to be harvested. Any mechanism that *stops averaging* — even the simplest, e.g. "each batch uses a single teacher chosen by who is most confident at that position" (VCG-style, Model 10.2) — is likely to outperform all the averaging variants that have failed to deliver > 0.01 BPB wins in prior runs. The field has been optimizing weights in a parameterization where the optimum is unreachable.

### 1.6 Negative evidence (refusals, runners-up, suppressions) is being thrown away
**Portfolios:** 02 (off-diagonal / rate-distortion), 08 (Grice maxims, top-k structure), 12 (partial trace discards off-diagonal), 13 (density matrix off-diagonal blocks), 15 (pre-softmax floor, top-2 gap, negative space).

**Strongest form.** After softmax normalization, the teacher's pre-softmax logit magnitudes collapse to "small probability" and the *ordered rejection structure* is erased. But that structure — the runner-up token, the floor logit, the gap between top-1 and top-2, the absolute magnitude of rejection — is computed *for free* by every teacher forward pass and already sitting in memory. Classical KD normalizes it away before training.

**Testable consequence for Sutra.** Add two near-free losses: (i) top-1/top-2 margin matching (Model 15.2), (ii) pre-softmax floor matching (Model 15.5). Cost: two reductions per position. Expected: improved calibration, reduced hallucination on long generations, and (per the Grice / IIT portfolios) higher transferable structure even when aggregate BPB is flat.

### 1.7 Knowledge is laws (Jacobians, equivariances), not values
**Portfolios:** 05 (functoriality morphisms, parallel transport on fiber bundles), 06 (structure mapping — relations not surface), 07 (MPC trajectory), 08 (distributional derivatives, compositionality), 11 (program synthesis spec strength), 13 (quantum channel structure), 16 (sensorimotor contingencies).

**Strongest form.** Two models can have identical pointwise distributions and completely different Jacobians; the Jacobian carries most of what generalizes. Sobolev training, contingency loss, paired-contrast distillation, and functorial KD are all the same operation under different vocabularies. This is the same claim as 1.1 viewed analytically rather than algebraically.

**Testable consequence for Sutra.** Already covered by 1.1's prescription — paired-contrast / finite-difference losses. The convergence across seven vocabularies is itself the evidence that this is a load-bearing omission, not a boutique improvement.

### 1.8 Constraints / priors are more efficient than data
**Portfolios:** 04 (Waddington landscape shapes valleys), 08 (Chomsky — poverty of stimulus, UG), 09 (NEAT complexification), 11 (type constraints, CRDT intersection), 15 (constraint-driven creativity — OuLiPo/Bauhaus), 17 (poka-yoke — structural impossibility beats discouragement), 18 (SHY renormalization resets the operating point).

**Strongest form.** When a small architectural or data prior is imposed, the same teacher signal goes much further, because the hypothesis space is pre-shaped along the right axes. The plateau is often not a data problem or a teacher problem — it is a *hypothesis-space problem*. Pushing more teacher signal onto an unshaped manifold wastes gradient on carving the manifold from scratch.

**Testable consequence for Sutra.** Hierarchical attention decay + byte-bracket-aware local bias (Model 08.1) is a one-line architectural change with no teacher cost. If after adding it the same KD schedule shows larger gaps between teachers, the Chomskian diagnosis holds — teachers were fighting over an unconstrained space, now they are disambiguating a shaped one.

### 1.9 Plasticity is non-uniform across layers and time
**Portfolios:** 01 (critical period plasticity windows), 03 (RG — UV layers converge first, IR last), 04 (morphogenesis — different valleys at different times), 07 (adaptive control — non-stationary plant), 14 (catalysis applied at transition state only), 18 (REM/NREM stage order matters).

**Strongest form.** All parameters trained at one LR schedule is a biologically and physically incoherent default. Different layers have different "developmental times," different teacher signals should dominate at different training phases, and the teacher's marginal value is concentrated at transition states — not at steady state.

**Testable consequence for Sutra.** Log per-layer gradient norm across training. If the spectrum is naturally staggered, amplify it with a per-layer schedule + per-phase teacher. If uniform, the architecture lacks temporal differentiation and a critical-period-style forced schedule is a candidate intervention.

### 1.10 The loss landscape is *shaped* by the teacher, not *targeted*
**Portfolios:** 03 (thermodynamic coupling, spin-glass ultrametric), 04 (Waddington morphogens), 14 (catalysis — teacher vanishes at optimum, Le Chatelier — equilibrium is teacher-independent), 15 (atelier — teacher is a correction field, not a target), 18 (dream recombination — teacher becomes a generative manifold).

**Strongest form.** The teacher should ideally *not* appear in the final-state loss. A teacher that leaves a fingerprint at the optimum is stoichiometrically consumed; a teacher that reshapes the barrier between random-init and competent-state while vanishing at the optimum is catalytic. Ekalavya turnover number — student updates enabled per teacher forward-pass — is the right efficiency metric, and it is ~1 in current KD.

**Testable consequence for Sutra.** Scale the KD term by `||∇_θ L_CE||²` (or by the Hessian trace, or by the gradient-variance indicator of criticality) so teacher pressure naturally vanishes in flat basins. Chemistry (14.1), physics (03.3), and art (15.1) all independently predict this is the right shape.

### 1.11 Quality-diversity, not scalar optimum
**Portfolios:** 04 (HGT chimeras, niche construction), 09 (MAP-Elites, novelty search, open-endedness), 15 (style/content with multiple voices), 16 (intrinsic motivation — self-generated curriculum), 18 (semanticization prototypes).

**Strongest form.** Classical KD produces one student that is mediocre-everywhere. EC learned two decades ago that scalar fitness traps the search in deceptive basins. A MAP-Elites archive over behavioral descriptors, distilled at the end into a final student, may Pareto-dominate scalar-best at matched compute — especially on tail behaviors.

**Testable consequence for Sutra.** Maintain an archive indexed by `(output_entropy, teacher-agreement-vector)` over training checkpoints; the tail slices (code/math/long-context) should be won by archive-distilled rather than scalar-best.

### 1.12 There is a fundamental finite-query ceiling (no-cloning / observability / Gramian)
**Portfolios:** 02 (rate-distortion at fixed student rate), 07 (observability Gramian rank), 12 (IIT — sum-decomposable losses have zero gradient toward irreducible subspace), 13 (no-cloning bound).

**Strongest form.** At some teacher-query budget, the marginal BPB-per-query drops to near zero. This is structural, not a tuning failure. The cure is adding informationally distinct *channels* (hidden-state alignment, multi-basis tomography, off-diagonal probes), not more queries in the same channel.

**Testable consequence for Sutra.** Plot marginal BPB-per-query vs query count. If we are on the cloning floor, adding a second channel (say, hidden-state cosine loss on two middle layers) should produce a visible kink upward. If not, we are bandwidth-limited, not channel-limited.

---

## 2. DIVERGENT PERSPECTIVES

The portfolios do not agree on everything. The genuine disagreements below are where the field's future design space is actually open.

### 2.1 Match marginals vs match programs
- **Marginal camp.** Classical KD, info-theoretic IB, rate-distortion: the student should match `p_T(y|x)`, with more or less relevance weighting.
- **Program camp.** MDL (02.1), program synthesis (11.3), compositionality (08.5), category theory (05.1), Bell-violation (13.5): distill the *combinator* / *computational trace*, not the output.
- **What's at stake.** If the marginal camp is right, adding more teacher bandwidth / better marginal losses / better reweighting schemes will eventually close the gap. If the program camp is right, those will not — and we need structurally different losses.
- **What the recent data says.** Regret sampling (a marginal-reweighting scheme) failed on pool overfitting. This is mild evidence against the marginal camp, but not dispositive — the failure could also be a data-coverage pathology. The only dispositive test is the paired-contrast / functorial / compositional probe from §1.1.

### 2.2 Teacher-as-oracle vs teacher-as-catalyst vs teacher-as-environment
- **Oracle camp.** Most portfolios implicitly: the teacher is a source of targets that the student consumes.
- **Catalyst camp.** Chemistry (14): the teacher modifies the landscape and vanishes at the optimum; turnover number, not per-step cost, is the efficiency metric.
- **Environment camp.** Biology stigmergy (04.2), manufacturing TOC (17.1) partially, niche construction (04.5): the teacher is a medium the student reads/writes through, not a partner it queries.
- **What's at stake.** Oracle framing leads to endless "match more teacher bits better." Catalyst framing justifies *reducing* teacher usage drastically. Environment framing dissolves the teacher entirely into a shared stigmergic buffer.
- **What the recent data says.** Classical, dense teacher querying has hit a plateau. That is evidence the oracle framing is at least rate-limited; it is consistent with either of the other two being more productive.

### 2.3 Knowledge lives in weights vs lives in environment vs lives in dynamics
- **Weights.** HGT grafts (04.4), semanticization prototypes (18.4).
- **Environment.** Stigmergy (04.2), training-as-niche (04.5), replay buffers.
- **Dynamics.** RG flow (03.1), MPC trajectory (07.3), recurrent re-entry (12.5), sensorimotor contingencies (16.3).
- **What's at stake.** Each implies a very different artifact. Weights-view => composable architectures (Ekalavya as chimera). Environment-view => persistent training-time memory structures. Dynamics-view => the student's *computational trajectory* must be shaped, not its endpoint.
- **What the recent data says.** Regret sampling tried to reshape the input environment (which tokens get seen) and failed to transfer. That is weak evidence against a pure environment view; it does not distinguish weights from dynamics.

### 2.4 Reduce variance (converge fast) vs inject difficulty (converge slow but better)
- **Variance-reduction.** Heijunka (17.4), consistent-mixture data loading, SPC.
- **Difficulty injection.** Desirable difficulty (06.4), constraint-driven creativity (15.4), domain randomization (16.1), novelty search (09.2), motor babbling (16.2).
- **What's at stake.** Heijunka predicts smooth loss curves and faster convergence. Bjork predicts slower early training but better late representations. These are incompatible under one training run, though potentially compatible across phases.
- **What the recent data says.** Ambiguous. Sutra's train-eval gap in the last run (train crashed to 1.33 while eval went up to 1.46) is *exactly* the Bjork signature of easy-route memorization. That's moderate evidence that Sutra has been too low-difficulty, not too high-variance.

### 2.5 More plasticity vs more rigidity (soundness vs precision)
- **Plasticity.** Open-endedness (09.5), affinity maturation (04.1), evolutionary KD.
- **Rigidity.** Abstract interpretation (11.1) — soundness first, precision second. Poka-yoke (17.5) — make failures structurally impossible.
- **What's at stake.** Plasticity risks catastrophic drift; rigidity risks brittle, uncreative students. Is the right answer "start loose and tighten" or "start tight and loosen"?
- **What the recent data says.** The pool-overfitting failure favors rigidity at least initially — our 188M student has enough plasticity to *memorize the training pool* while failing eval. Tightening (soundness coverage, poka-yoke constraints) would have prevented this specific failure.

### 2.6 Feedforward sufficient vs recurrence necessary
- **Feedforward.** The current Sutra-Dyad architecture; most classical KD.
- **Recurrence.** RPT / binding (12.5), MPC closed-loop (07.3), re-entry sweeps, adaptive-depth test-time compute.
- **What's at stake.** A feedforward student distilling from test-time-compute teachers (o1, R1, Universal Transformer) may be *architecturally incapable* of inheriting the teacher's binding sweep, no matter how good the loss.
- **What the recent data says.** Silent — we have not tested recurrent architectures. This is a gap in the evidence itself.

---

## 3. THE CROSS-PORTFOLIO GRAND UNIFICATION ATTEMPT

Given the specific falsification — **regret sampling overfit the training pool while eval BPB rose** — what composite of mental models *best explains this failure and suggests the fix*?

### The composite diagnosis (4 mechanisms)

**(A) Principal-agent / Goodhart (10.3).** The student, given capacity, gamed the measured KL on the re-weighted pool by finding shortcut features specific to the oversampled positions. The measured signal (training pool BPB) was not aligned with the latent goal (eval/test generalization). This is the textbook Goodhart failure mode; our training loss drop and eval loss rise are the *predicted* empirical signature.

**(B) Absence of difficulty / rigidity (06.4 Bjork, 11.1 abstract interpretation).** The regret-weighted pool *concentrated easy-to-absorb teacher agreement*, lowering the effective difficulty of each gradient step. The student took the easy route (fast train drop) at the cost of robust representations (eval rise). Bjork's prediction: fast-dropping training loss *is a warning*, not a success indicator.

**(C) Missing offline / equilibration phase (Chapter 1.4, 14.5 Le Chatelier, 18 all).** The pool reweighting is a sustained perturbation with no relaxation window. The student never equilibrates to the current forcing before the forcing is intensified. Per 14.5, "cranking the KD pressure shifts the equilibrium until internal stresses exactly counter the pressure" — which manifests as "the student fits the sampling distribution rather than the underlying data."

**(D) Bottleneck misidentification (17.1 TOC, 10.1 Hayek, 16.4 active perception).** "Regret" (student high-loss) is a poor proxy for "bottleneck" because it does not require (i) teacher confidence at that position *and* (ii) student plasticity toward that signal. Positions with high student loss but low teacher confidence are *noise magnets*; upsampling them trains the student on teacher hallucinations. The TOC / bottleneck score `L_p × g_p × c_p` requires all three factors; regret used only one.

### Why these four compose

- (A) says the *metric* we reweighted was wrong.
- (B) says the *difficulty* of the reweighted distribution was too low.
- (C) says the *dynamics* had no relaxation window.
- (D) says the *definition of "hard position"* used only one of three necessary factors.

Any one of these alone could have caused the failure. All four are coherent with the data. The rest of the survey says all four are real and recurring.

### The minimal composite experiment

**Name: "Rigid Bottleneck + Offline Relaxation"** (~6 GPU-hours, no new infrastructure).

1. **Bottleneck identifier (D).** At each step, compute per-position `B_p = student_loss_p × ||teacher_logit - student_logit||_p × teacher_top1_prob_p` (requires all three > threshold). Only the top 5-10% of positions by `B_p` contribute to KD gradient. Others contribute CE only, at 0.2× weight.
2. **Soundness-first rigidity (B, A).** For the top-`B_p` positions, the KD loss is *not* forward KL on full softmax. It is (i) top-2 token-set match (Jaccard on top-2), (ii) top-1/top-2 margin match, (iii) pre-softmax floor match. Three bounded, calibration-friendly terms. Total cost: three reductions per active position. No full-softmax KL — the student cannot game it by smoothing.
3. **Offline relaxation (C).** Every 500 training steps, freeze the data loader for 50 steps. Replay *only* buffered high-`B_p` positions from the last window. No new data, no teacher calls (use cached logits). This is the equilibration pause Le Chatelier and SHY both demand.
4. **Train-vs-held-out divergence gate (A).** Track the ratio of `KL(student || teachers_in_loss)` vs `KL(student || held_out_teacher)`. If the ratio widens beyond a threshold, trigger a sleep/renormalization phase (SHY — multiplicatively shrink all weights by 0.97 and a brief CE-only pass) before resuming KD.

**Expected signature if the composite is right.** Training BPB drops slower than uniform KD (rigidity removes the easy route). Eval BPB drops *faster*. The train-vs-held-out KL ratio stays bounded. The improvement concentrates on positions that *both* the student was wrong on *and* the teacher was confident on.

**Expected failure modes and what each would tell us.**
- If training and eval BPB both drop slowly but together: bottleneck identification is too aggressive (`B_p` threshold too high).
- If training drops fast and eval lags: top-2/margin losses are still gameable; need stronger soundness constraint.
- If eval BPB is flat: the 188M student lacks capacity at the bottleneck positions; accommodation (06.2 schema theory) is needed — allocate LoRA capacity aligned to the residual direction at those positions.

---

## 4. ANTI-PATTERNS (multi-portfolio warnings)

These are patterns that classical KD does by default and that ≥3 portfolios independently warn against. They are the cheap subtractions.

### 4.1 Don't match full softmax uniformly
Warned by: 01 (predictive coding wants residuals), 02 (rate-distortion bounds), 03 (high-entropy teacher tokens are thermal noise), 06 (ZPD says most tokens are out-of-band), 11 (cache-miss-driven), 12 (IIT — marginals are sum-decomposable), 15 (all five creative models), 17 (TOC — 95% of positions are non-bottleneck).

### 4.2 Don't average teachers (see §1.5)
Nine portfolios. This is the single most unanimous anti-pattern in the survey.

### 4.3 Don't train without offline / equilibration phase (see §1.4)
Six portfolios. Biology, chemistry, manufacturing, and cognitive science all make this impossible to ignore.

### 4.4 Don't optimize a single scalar
Warned by: 09 (QD), 10 (Markowitz, Hayek), 11 (SPC, control chart semantics — a scalar with no confidence interval is useless), 15 (style/content decomposition), 17 (SPC demands noise floor measurement).

### 4.5 Don't normalize away the rejection structure (see §1.6)
Five portfolios warn that pre-softmax information is load-bearing and discarded for free.

### 4.6 Don't use BPB as the only progress metric
Warned by: 01 (generation quality), 06 (fast training loss is suspicious), 11 (spec bandwidth), 12 (BPB averages out integration gaps), 15 (BPB blind to compositional skill), 17 (SPC — most "BPB wins" are inside the noise floor).

### 4.7 Don't keep teacher pressure constant
Warned by: 03 (phase transitions), 06 (critical periods), 07 (adaptive control), 14 (catalysis — teacher should vanish at optimum), 15 (atelier — sparse correction).

### 4.8 Don't treat the student as stateless / passive
Warned by: 06, 07 (observability), 12, 16 (every model), 18 (TMR needs student state cues). This is the same point as §1.3 but framed as prohibition.

---

## 5. BEST 3 MECHANISMS TO PROTOTYPE NEXT

Selection criteria: novel (not another KL variant), tractable (≤1 week, existing infra), informative-on-failure, disjoint from each other. Every candidate is drawn from multiple portfolios so its expected value does not depend on a single lens being right.

### 5.1 "The Atelier Bottleneck" — sparse on-policy teacher correction at top-B_p positions only
**Portfolios fused.** 15.1 (atelier sparse correction), 17.1 (TOC bottleneck), 11.2 (cache-miss KD), 08.3 (Wittgenstein on-policy), 16.2 (babbling-weighted KD), 06.1 (ZPD gating).

**Proposed experiment.** Replace the KD inner loop: (i) student generates `k=8` bytes autoregressively from each context, (ii) teacher computes surprisal on the student's samples, (iii) the *single* position with max surprisal-student × student_confidence is chosen, (iv) KD loss applied only at that position + radius 2. Total teacher forward cost: O(1) positions per context, down from O(k). Train for 3K steps alongside standard-KD baseline at matched teacher-FLOPs.

**Expected failure signature.** If the BPB gap is zero, bottleneck identification via `surprisal × student_confidence` is no better than uniform — the sparse-selection hypothesis (§1.2) is at least partially wrong. If BPB improves but generation quality does not, we have overfit to the specific selected positions. If both improve, we have the first confirmation that Sutra's failure is not loss design but *position selection*.

**Why now.** Six portfolios independently recommend it. Cost is strictly *less* than current KD (fewer teacher queries). Failure is diagnostic regardless of direction.

### 5.2 "Soundness-First Rejection Structure" — top-2 margin + pre-softmax floor KD instead of full KL
**Portfolios fused.** 15.2 (runner-up gap), 15.5 (negative space / pre-softmax floor), 11.1 (abstract interpretation — soundness before precision), 08.4 (Gricean pragmatic KD), 12.1 (IIT — preserve structure), 13.3 (density matrix — preserve off-diagonal).

**Proposed experiment.** Replace forward-KL KD with a 3-component loss: (i) top-2 set-overlap loss (Jaccard on top-2 token sets), (ii) top-1/top-2 margin MSE `((ℓ_1 − ℓ_2)_T − (ℓ_1 − ℓ_2)_S)²`, (iii) pre-softmax floor matching `(percentile_1(ℓ_T) − percentile_1(ℓ_S))²`. Weight 0.05 total. Run alongside standard KD baseline. The student cannot game this by smoothing — smoothing reduces margin and floor, so both terms penalize it.

**Expected failure signature.** If ECE (calibration) does not improve, the pre-softmax floor contains no useful signal. If BPB ties but calibration improves and hallucination rate drops, we have improved exactly the dimensions BPB is blind to (matching the §4.6 anti-pattern). If BPB improves measurably *and* train-vs-eval gap narrows (vs the last regret-sampling run), we have found a route around the pool-overfitting failure mode — the student can't game a loss built on rejection structure.

**Why now.** Pre-softmax logits are already computed. Cost is two extra reductions per position. Five portfolios independently argue this is where the teacher's editorial decisions live, and Sutra has never trained on them.

### 5.3 "Student Self-Predictor + TMR Sleep" — train a 5M-param forecaster of the student's own output; use it to schedule cued replay
**Portfolios fused.** 06.5 (cognitive tutoring — student-model), 12.3 (HOT — metacognitive head), 18.3 (TMR cued replay), 16.5 (intrinsic motivation — learning progress), 07.1 (observability).

**Proposed experiment.** (i) Train a 5M-param MLP/small-transformer that, from student hidden state at position p, predicts student logits at p+1. Target: AUC > 0.7 on held-out tokens for "will the student be right?" This is Sutra's first piece of self-modeling. (ii) Every 500 training steps, run a 50-step "TMR sleep": buffer replay sampled with `P ∝ 1 - predictor_confidence` (where the predictor says the student will be wrong) AND learning-progress-positive clusters. No new data entering during sleep.

**Expected failure signature.** If the predictor AUC is <0.6, student behavior is too high-dimensional to cheaply forecast — the whole "self-model" line of work is too expensive at 188M. That alone closes out §1.3 at this scale. If AUC > 0.7 but TMR sleep doesn't improve BPB, the issue is cue quality, not the predictor. If AUC > 0.7 AND TMR yields a BPB gain, we have a single artifact that simultaneously enables: self-aware early exit (Outcome 5), targeted-weakness replay (Outcome 2 — surgical improvability), cued multi-teacher scheduling (Ekalavya via §1.5 without averaging), and a Gramian-like observability diagnostic.

**Why now.** The highest-leverage single artifact in the entire survey. Four portfolios all say "close the loop on the student's state" and this is the minimum-viable realization. Even if the mechanism fails, the predictor itself is useful standalone.

### Why these three and not others
- They are **disjoint**: position selection (5.1), loss-object redesign (5.2), state-conditioning (5.3).
- Each draws from **≥5 portfolios** — no single lens is load-bearing.
- Each **fails informatively**: the failure modes are interpretable and move the design space, unlike "KL tweak didn't work."
- Together they constitute the minimal implementation of the grand-unification composite (§3).

Excluded: RG-flow KD (high ceiling, high implementation cost — a month not a week), sheaf-gated multi-teacher (architecturally invasive), evolutionary archive (needs a different training loop entirely), recurrent re-entry (architectural change). All are on the shelf for later rounds.

---

## 6. THE META-MODEL: What Is Knowledge Transfer?

Across 18 lenses, here is the consensus every portfolio would sign.

**Knowledge transfer is the controlled reshaping of one system's state-space by another system's already-paid computational work, through a bounded channel, such that the receiving system acquires not the sender's outputs but the sender's *invariants*.** The invariants are the lawful relationships between perturbations and consequences — the Jacobian, the functorial composition rule, the equivariance group, the sensorimotor contingency, the program, the access policy — and they are what enables the receiver to handle inputs the sender never demonstrated. Every successful transfer mechanism across every discipline operates by identifying such invariants, isolating them from the idiosyncratic output values in which they were embedded, and re-instantiating them in a substrate of different shape. The sender's *outputs* are evidence of the invariants; they are not the invariants. Transfer that treats outputs as the target produces mimicry; transfer that treats outputs as evidence produces competence.

**The irreducible core of knowledge transfer** — the part every portfolio recognizes, in its own vocabulary — is a closed loop of four operations: *selection* (what to transmit, where, when), *projection* (through what channel, with what loss), *equilibration* (offline relaxation during which the receiver integrates and the signal is not pushed), and *verification* (comparing the receiver's own productions against the sender's lawful structure). Classical KD does projection well. It does not do selection (uniform positions), equilibration (always online), or verification (no receiver-state feedback). The empirical consequence — a plateau at BPB ~1.40 that falsifies every input-reweighting scheme — is the exact symptom predicted by a pipeline with one of four loop-operations present.

The good news is that every portfolio supplies cheap, testable mechanisms for each missing operation. The bad news is that any single mechanism added alone will not move the needle much, because the loop is still open in three places. The prescription is structural: build the minimum-viable closed loop first (§5's three prototypes, or §3's composite experiment), validate each edge of the loop independently, and only then ask which specific mechanism inside each edge works best. That reorders a decade of KD research practice — the field has been trying to perfect one edge while three others remained broken.

The single sentence to remember: **knowledge is a set of invariants; transfer is a closed-loop protocol; classical KD optimizes one leaf of an open loop.** Everything else in this survey is how to close the loop.
