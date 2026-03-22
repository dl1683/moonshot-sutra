**1. Assumption Challenges With Both-Sides Analysis**

1. `12` passes is the right depth. For: full-vocab truncation at step `8300` shows `12` passes = `7.59` BPT, `10` = `18.39`, `8` = `20.40`, so late recurrence is doing real work. Against: collapse metrics show `91.5%` of total improvement happens in passes `7-11`, with a phase change at `10->11`; most passes are underused fixed-point drift, not twelve useful refinements. Resolve with: the same shared-core architecture trained with random-depth and `Dmax` in `{4,6,8,12}` at matched average compute. Lean: keep recurrence, drop `12` as doctrine, design around `Dmax=8`. Confidence `8/10`.

2. Pure weight sharing is the right parameter strategy. For: on a single `24GB` laptop GPU, sharing is the only way recurrence is affordable, and repeatedly training one strong block is usually a better use of scarce core capacity. Against: current collapse is exactly what sub-`200M` recursive LM literature predicts when pass identity is absent; pure sharing is probably too symmetric here. Resolve with: same shared core, same training recipe, compare `none` vs pass-conditioned control. Lean: mostly shared core plus tiny pass/mode conditioning. Confidence `9/10`.

3. The current `7`-stage bank is the right way to realize multi-mode processing. For: explicit roles are good for debugging and future modularity, and earlier repo evidence says content types followed different stage paths. Against: `StageBank` costs `16.54M` params, `24.2%` of the whole model and about `58.7%` of the non-embedding core, yet it is not a real stable ABI; it is mostly seven expensive MLPs blended back into one state. Resolve with: one shared content block plus explicit external modules and lightweight control. Lean: keep decomposition as interfaces, not as seven heavyweight FFN banks. Confidence `9/10`.

4. Dynamic multi-mode processing should stay explicit. For: this is the part of Sutra that is actually differentiated; real intelligence does lookup, synthesis, verification, and revision concurrently. Against: too many heavyweight modes on a `68M` budget will fragment capacity and create dead branches. Resolve with: a top-2 mode simplex over a small set of lightweight modes that conditions a single shared core. Lean: non-negotiable concept, lightweight implementation. Confidence `9/10`.

5. The current scratchpad design is already the memory answer. For: Probe D is clearly positive; the saved JSON and probe code show `full=5.8353`, `no_read=6.0260` (`+3.27%`), `no_write=6.0201` (`+3.17%`), and `removed=6.0260`, so sequence-specific writes and readable memory both matter. Against: recall only moves from `0.74` to `0.72` top-100, and the mechanism is an `8`-slot soft-attention EMA workspace, not exact retrieval. Resolve with: keep it as the general/workspace end of memory and add a separate precise memory path. Lean: keep scratchpad, demote it from “memory solution” to “general workspace.” Confidence `9/10`.

6. A precise-general retrieval spectrum is optional. For the spectrum: the biggest competitive gap is knowledge, not coarse reasoning; `SciQ 25.9%` vs `74.0%` and `LAMBADA ~1%` vs `32.6%` are memory-architecture failures. Against implementing it naively: exact memory can hijack a small model, become a brittle cache, or cost too much. Resolve with: a learned mixture over workspace memory, exact episodic memory, and compact parametric associative memory. Lean: mandatory design axis. Confidence `10/10`.

7. `BayesianWrite` and `lambda` already tell the right semantic story. For: bounded writes are useful; this path fixed real NaN problems and prevents residual soup. Against: the collapse probe shows `lambda` spiking from `1.89` to `4.60` on the final pass, which looks like deferred budget dumping, not calibrated certainty, and monotone precision makes contradiction-driven revision hard. Resolve with: separate bounded state update from halting/budget signals. Lean: keep a bounded writer, retire `lambda` as “confidence.” Confidence `8/10`.

8. The current router plus pheromone is the right communication design. For: some routing is clearly needed for code, long dependencies, and cross-token coordination; the router is only `6.9%` of params. Against: current “sparse” routing still builds a full `T x T` score matrix before top-k, and pheromone adds positive feedback to a system already showing attractor behavior. Resolve with: local causal mixing plus memory-mediated long-range recall, with pheromone removed. Lean: local routing stays, global routing should mostly go through memory, pheromone should go. Confidence `8/10`.

9. The current training methodology is basically right. For: attached history helps, weak intermediate supervision helps, and full-vocab intermediate CE is decisively catastrophic; the repo already found the Goldilocks fact. Against: sampled `L_step` is biased because negatives come from final-pass top logits, fixed `12`-depth starves early passes of real final-loss signal, and from-scratch resets burn tokens. Resolve with: random-depth mandatory, soft intermediate objective only, additive warm-start probes first. Lean: the next gain is training-recipe repair plus simplification, not another heavy module. Confidence `9/10`.

10. The current tokenizer/embedding front-end is acceptable as a final design. For: keeping GPT-2 BPE for immediate probes isolates architectural questions and preserves checkpoint compatibility. Against: `emb + pos_emb` consume `40.17M` params, `58.8%` of the model, and SVD shows the learned embedding is genuinely full-rank, so warm-start factorization is not viable. Resolve with: keep tokenizer fixed in warm-start probes, but move to factored embeddings from scratch in the clean next line. Lean: freeze tokenizer short-term, redesign embeddings long-term. Confidence `8/10`.

11. Recurrence at `68M` has already earned ideological status. For: full-vocab truncation proves iterative refinement is doing real work, and small recursive LMs can work when the recipe is right. Against: the current model is still near-random on most benchmarks, so recurrence has not earned a free pass; it may only be valuable once paired with exact memory and lighter control. Resolve with: compare `D=1` and random-depth `D>1` inside the same Sutra family, not against a generic dense baseline. Lean: recurrence should stay, but only as shared-core iterative refinement coupled to precise memory. Confidence `7/10`.

12. Early architecture gates should be benchmark-only. For benchmarks: they matter in the end. Against benchmark-only gating now: both `v0.5.4` and `v0.6.0a` sit near floor on several tasks, so small architectural improvements will be invisible there; BPT alone is also insufficient because it can improve local fluency without fixing knowledge. Resolve with: gate on a triad of full-vocab BPT, generation quality, and exact/knowledge recall probes. Lean: benchmarks stay secondary until the memory path exists. Confidence `8/10`.

**2. Research Requests For Claude To Execute**

- Survey exact-memory backends for small causal LMs under `24GB`: `ARMT`, smallest credible `PKM` variants, runtime exact episodic tables, and swappable on-device fact stores. I need equations, footprint, warm-start story, and failure modes.
- Survey mode-conditioned shared cores below `200M`: pass-conditioned `adaLN`, mode-conditioned `adaLN`, low-rank mode adapters, and top-2 control simplices. I care about real small-model evidence, not generic transformer tricks.
- Survey halting/elastic-compute training for recursive LMs that combine random-depth, gain prediction, and compute penalties without pass collapse.
- Survey module-local distillation: `Pythia-160M`/`70M`-style hidden-state transfer for the core and `BGE`/`E5`-style alignment for retrieval modules, with hard-token selection and low-coefficient schedules.
- Survey function-preserving growth specifically for adding a new memory module to a live causal LM, then for later migrating to a clean factored-embedding line.
- Survey tokenizer/embedding co-design for small recurrent LMs so the clean restart does not just inherit GPT-2’s front-end tax.

**3. Experiment/Probe Requests With Methodology**

- `GPU when free:` random-depth on `v0.6.0a`. Sample `D∈{1..12}` from a deep-biased distribution, keep everything else fixed, warm-start `3-5K` steps, and log full-vocab per-pass BPT, pass cosine, generation diversity, and `lambda`/stability profiles. Success: earlier passes become materially useful without hurting final BPT.
- `GPU when free:` build `Sutra-core-lite`: one shared SwiGLU block, top-2 control simplex, current bounded writer, scratchpad, no `7`-bank, no pheromone. Compare `D=1`, `Dmax=4`, and `Dmax=8` at matched average compute. Success: `D>1` wins on BPT, generation, and recall inside the same family.
- `GPU when free:` precise-memory spectrum probe on `Sutra-core-lite`: workspace-only vs workspace+exact episodic buffer vs workspace+ARMT sidecar vs all three. Evaluate held-out BPT, synthetic copy/entity recall, LAMBADA-style suffixes, and factual prompts. Success: exact memory lifts knowledge without collapsing fluency.
- `GPU when free:` mode-control probe: pass-only conditioning vs pass+mode-conditioned control on the same shared core. Track mode entropy, top-2 mode utilization by token class, pass specialization, BPT, and generation. Success: explicit modes improve both quality and diagnosability.
- `GPU when free:` soft auxiliary shootout after random-depth works: current sampled `L_step` vs detached final-state alignment vs residual prediction. Success: one auxiliary keeps late-pass gains while reducing collapse more than the current biased sampled CE.
- `GPU when free:` router simplification probe: current router vs local-only mixer vs local+m emory-mediated long-range. Measure throughput, BPT, and recall. Success: remove full `T x T` scoring without losing long-range behavior.
- `CPU now:` token-type recall audit on current checkpoints. Bucket failures into entities, numbers, repeated rare words, code identifiers, and generic function words; compare pass disagreement and final errors. Success: tells the controller when to go precise vs general.
- `CPU later:` re-run disagreement/gain calibration after the best anti-collapse fix. Success: the signal still predicts future gain once the model is no longer collapsed.

**4. Per-Outcome Confidence (1-10) With Justification**

- Outcome 1 (Intelligence): `6/10` — late recurrence is real, the knowledge failure is now clearly diagnosed as missing precise memory, and the proposed design reallocates capacity away from stage duplication and later away from the embedding tax; but the key fixes have not yet been run.
- Outcome 2 (Improvability): `7/10` — the repo already localizes failures unusually well, and a `core / workspace / exact-memory / controller / readout` split is a real ABI candidate, not just rhetoric.
- Outcome 3 (Democratization): `6/10` — swappable exact-memory backends, verifiers, and domain fact stores make outside contribution plausible, but the interface is not frozen and composition is not yet demonstrated.
- Outcome 4 (Data Efficiency): `6/10` — selective teacher absorption, exact memory, and later factored embeddings are concrete levers now; this is no longer vague. The missing piece is a positive canary.
- Outcome 5 (Inference Efficiency): `5/10` — random-depth plus predicted-gain halting and cheap precise lookup should reduce average depth, but the halting signal is only validated on the collapsed model and no token retirement exists yet.

**5. What Would Raise My Confidence**

- Outcome 1: `Sutra-core-lite` with exact memory beats the current family on generation quality and on at least one knowledge-sensitive probe, not just BPT.
- Outcome 2: a memory-module or controller-module swap improves one failure mode without degrading the rest of the model.
- Outcome 3: a domain-specific memory or verifier backend plugs in cleanly and composes with the base model.
- Outcome 4: a narrow teacher path reaches the same BPT in fewer tokens, or an exact fact-store backend lifts factual tasks without more raw-text training.
- Outcome 5: easy tokens retire after `2-3` passes with no quality loss, while hard tokens still use the tail productively.

**6. What Would Lower My Confidence**

- Outcome 1: exact memory fails to lift knowledge/recall, or recurrence still collapses after random-depth and control simplification.
- Outcome 2: every meaningful gain still requires whole-model retraining and no stable module interface survives.
- Outcome 3: interfaces keep changing across versions or module improvements do not compose cleanly.
- Outcome 4: teacher signals keep acting like noise, or added memory/fact modules do not improve intelligence per training token.
- Outcome 5: most tokens still need near-max depth, or exact memory queries are so expensive that they erase the compute win.

**7. Design Proposal**

Build a **Shared-Core Cognitive Spectrum Sutra**. The base model should stop pretending one mechanism can do both fuzzy reasoning and exact recall.

- Use **one shared recurrent content block** repeated with `Dmax≈8`, not a `7`-FFN bank. Condition it with a **top-2 control simplex** and pass identity through lightweight `adaLN`-style modulation. This serves Outcomes `1,2,3,5`.
- Make the control simplex explicit: `compose`, `route`, `retrieve`, `verify`. Different positions can hold different top-2 mode mixtures on the same pass, so the superposition thesis survives without duplicating the whole compute core. This serves Outcomes `1,2,3`.
- Implement a **continuous retrieval spectrum** with three backends behind one interface: a **workspace scratchpad** for diffuse discourse state, an **exact episodic buffer** for lossless in-context recall, and a **compact parametric associative memory** sidecar for factual lookup. Let the controller produce a learned `rho_precise` that mixes them per token/pass. This serves Outcomes `1,2,3,4,5`.
- Remove pheromone. Use **local causal mixing** for nearby syntax and let **memory handle long-range recall** instead of building a full `T x T` score matrix every pass. This serves Outcomes `1,5`.
- Keep a bounded writer, but split state update from halting. Replace `lambda-as-confidence` with **stability/conflict tracking plus predicted residual gain**. Train with random-depth and a small compute cost so early stopping becomes learned pressure, not a hardcoded hack. This serves Outcomes `1,2,5`.
- Keep intermediate objectives **soft only**. The long-term replacement for sampled `L_step` should be **residual prediction** or **detached final-state alignment**, never intermediate full-vocab CE. This serves Outcomes `1,5`.
- Design **multi-source learning by module**. Distill the shared core from a small LM only on hard tokens, align retrieval modules to embedding teachers, and let domain experts ship swappable fact stores or verifiers. This serves Outcomes `2,3,4`.
- Treat **function-preserving growth** as a hard rule. New exact-memory backends should enter zero-gated, train locally first, then unfreeze the core with a restart schedule. This serves Outcomes `2,3,4`.
- Do **not** retrofit factorized embeddings into the live line. Once the control/memory architecture is proven, start a clean line with **factored embeddings from scratch** so the model stops spending ~`40M` params on the front-end. This serves Outcomes `1,4,5`.

The near-term path is additive and warm-start-friendly: random-depth, control simplification, exact-memory sidecar, pheromone removal. The clean next line is the same ABI plus factored embeddings. That is the path that preserves Sutra’s distinctive thesis while directly attacking the architecture’s actual bottleneck: it has general reasoning machinery, but no precise memory and too much wasted capacity.