**Core**
Based on [CLAUDE.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/CLAUDE.md), [code/launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py), [code/train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py), [code/spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py), [research/VISION.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/VISION.md), [research/SCRATCHPAD.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/SCRATCHPAD.md), [results/codex_outer_loop_budget.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_outer_loop_budget.md), and [results/codex_v060_design_review.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md), the right rule is:

Every transition must be an identity-preserving refinement, not a behavioral discontinuity.

That means four invariants for all v0.6 stages:
- New pathways start at zero influence.
- New controllers run in shadow mode before they act.
- New losses are zero-weight or zero-penalty at the parent operating point.
- When gates are closed, the child checkpoint should reproduce the parent checkpoint within numerical noise.

**Q1: Perpetual Warm-Start**
The GatedLayerNorm pattern in [code/launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) is the first member of a broader family. You need five more variants.

- `VerifyHead`: do not introduce it as an immediate actor. Train it first in shadow mode against stop-worthiness, not correctness. The target should be marginal future gain (`CE_t - CE_best_future`), which matches the failure noted in [results/codex_v060_design_review.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md). Its control gate starts at zero, freezing disabled, and only after AUROC is good does it begin to influence `done`.
- `Token freezing`: use soft freezing before hard freezing. Start with a blend `mu <- (1-z) * mu_new + z * mu_old` where `z=0` reproduces v0.5.4. Then mask only Stage 5 writes, then Stage 4 routing, then transition updates. Done tokens remain readable as context the whole time.
- `Budget accounting`: begin as an observer, not a penalty. First log token-step spend against the v0.5.4 baseline of `8*T`. Then use an overspend-only penalty with slack set to current average spend, so parent behavior has near-zero extra loss. Only later tighten the slack and raise `lambda_cost`.
- `Spectrum memory`: do not replace the router branch in one shot. Run old global retrieval and new spectrum memory in parallel with a residual blend gate `g_spec`, initialized to zero. Bias the scale predictor to the scratchpad extreme so the new path starts as “just gist.” Open coarse levels first, then intermediate, then exact leaf access.
- `Raising step ceiling`: treat passes `9+` as dead branches at first. Add pass-conditioned projections with zero-init, keep `max_passes_ceiling > 8` but gate late passes off, then open pass 9, then 10, then 11/12 while keeping average compute fixed. This avoids throwing the model straight into the OOD regime flagged in [results/codex_v060_design_review.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md).
- `Pheromone / core dynamics under new pass regime`: the current delayed rule in [code/launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) is tied to fixed inner-step count. Re-express it in normalized pass-fraction form, but behind a zero-init pass-conditioning gate so old checkpoints still behave like fixed-8 models when the gate is closed.

The staged roadmap should be:
- `v0.6.0`: add verifier, inter-step supervision, soft freeze gate, shadow budget accountant. Keep scratchpad and fixed 8-step ceiling. No spectrum yet.
- `v0.6.1`: keep average compute flat, raise ceiling to 10 then 12, open late passes one at a time, and only now let hard freezing become real.
- `v0.6.2`: add prefix-causal spectrum memory with old global branch blended in, then progressively transfer traffic from old router-global to spectrum.
- `v0.6.3`: add active-set compaction and true wall-clock elasticity only after the semantics are already stable.

**Q2: Multi-Source Knowledge Enrichment**
Yes, Sutra can absorb from many teachers without becoming dependent, but only if you distill stage contracts, not teacher architectures.

Recommended mapping:
- `GPT-2 / Pythia -> Stage 7 readout`: use top-k soft-label KD on final and intermediate logits. This is the cleanest fit because they solve the same autoregressive problem.
- `DeBERTa / BERT -> Stage 4/5`: do not align to full-sequence token states, because that leaks future information into a causal student. Instead distill prefix-only chunk/span summaries and low-rank affinity sketches inside the visible prefix. That teaches memory content and routing geometry, not impossible clairvoyance.
- `sentence-transformers -> Stage 5`: align scratchpad/tree summaries to normalized sentence or chunk embeddings and add a contrastive loss so memory states become semantically organized.

Best stack:
- `Stage-mapped distillation` is the backbone.
- `Soft-label KD` is strongest for Stage 7.
- `Representation alignment` is good for pooled memory states, not raw token-by-token imitation.
- `Contrastive learning` is a regularizer for semantic geometry, not the main teacher signal.

To avoid dependence:
- Teachers are frozen and never queried at inference.
- Teacher losses attach to Sutra’s existing stage outputs or temporary projection heads, not to runtime-only adapters.
- Teacher weights ramp up from zero, then ramp back down.
- Finish every distillation stage with a teacher-free consolidation run so the checkpoint proves it stands alone.

To fit a single RTX 5090:
- Never load Sutra and a large teacher together for full training.
- Run one teacher at a time in an offline or rolling-cache pass, store compressed artifacts, then train Sutra alone on those artifacts.
- Store only compressed targets: top-k logits for AR teachers, pooled fp16 embeddings for semantic teachers, and low-rank affinity sketches for encoder teachers.
- Distill on a curated subset of high-value data, not the full corpus. Knowledge enrichment is a shaping signal, not the whole objective.

**Integrated v0.6 + teacher roadmap**
- `v0.6.0 from v0.5.4 step 15K`: warm-start with verifier and soft freezing only. Train stop-worthiness in shadow mode from Sutra’s own future-step regret. No teacher dependence yet beyond optional offline diagnostics.
- `v0.6.1`: add AR-teacher KD from GPT-2/Pythia to Stage 7 while raising the ceiling at matched average compute. This enriches readout exactly where extra passes matter.
- `v0.6.2`: add spectrum memory and then distill DeBERTa/BERT prefix-span summaries plus sentence-transformer chunk embeddings into Stage 4/5 memory states.
- `v0.6.3`: anneal all teacher losses, run teacher dropout, finish with CE + verify + budget only, and keep the teachers completely absent. That is the “doctor studied at many universities” checkpoint.

If you want, the next useful step is an exact loss-and-gating spec for `v0.6.0 -> v0.6.3` against [code/launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py), [code/train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py), and [code/spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py).