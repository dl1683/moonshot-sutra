Verdict: the `scratch` recommendation was not lazy, but it was too categorical.

1. It was justified for scientific cleanliness, because `768 -> 1024` is not the same kind of mutation as `v0.5 -> v0.5.4` at fixed width. The current warm-start loader in [launch_v054.py:219](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/launch_v054.py#L219) only copies exact-shape tensors, so at `768 -> 1024` it would transfer almost nothing: about `13k` params, `0.013%` of the new `105.3M` model. And width changes behavior in the actual forward pass, not just shapes: final logits divide by `sqrt(dim)` in [launch_v054.py:201](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/launch_v054.py#L201), router attention divides by `sqrt(D)` in [sutra_v05_ssm.py:215](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L215), and scratchpad read does the same in [scratchpad.py:59](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/scratchpad.py#L59). So “just warm-start it” was not trivial. But saying “do scratch, not warm-start” was too strong. With a real widening morph, preserving the 69M model is worth doing.

2. Concrete widening plan:
- Instantiate `SutraV054(dim=1024, ff_dim=2048)`.
- Copy every old tensor into the upper-left/prefix block of the new tensor. All `69.38M` old params are shape-compatible this way, which is `65.9%` of the new model.
- For vectors: copy prefix, zero tail. For LN tails use `weight=1`, `bias=0`.
- For square `dim x dim` and projection weights: copy `[:768, :768]`, zero cross-blocks, small init in the new-new block.
- For embeddings, pos embeddings, scratch memory: copy first `768` dims, small-noise or zero tail.
- For `StageBank` hidden layers (`1536 -> 2048`), use Net2Wider duplication for the extra `512` hidden units and divide the paired outgoing columns by replication count. That preserves function better than dead zero rows.
- Add compatibility scalars for the three `sqrt(dim)` sites. Easiest version: initialize router/scratchpad/output with old-scale compensation, then anneal to native `1024` over `1k-2k` steps.
- Use a split or gated norm during transition. Full `LayerNorm(1024)` is not an exact subcase of `LayerNorm(768)`.

3. Preserve vs relearn:
- Preserve: embeddings/tied LM head in [launch_v054.py:101](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/launch_v054.py#L101) and [launch_v054.py:201](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/launch_v054.py#L201), stage specializations in [sutra_v05_ssm.py:88](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L88), router/write/scratchpad behavior in [sutra_v05_ssm.py:191](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L191) and [scratchpad.py:37](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/scratchpad.py#L37), and transition/init priors.
- Relearn: the extra `256` state channels, extra `512` FF channels, width-dependent calibration, and FineWeb-specific distributional stats.

4. Actual risk: moderate.
- The main risk is discontinuity, not shape mismatch.
- This model is recurrent for `max_steps=8`, so small scaling errors compound.
- `top2_project` makes stage selection partly discrete, so small evidence shifts can flip active stages.
- New channels may stay dead, leaving you with a padded 69M model.
- Warm-start can also bias mechanism evaluation: a mechanism may fail because it clashes with the inherited basin, not because it is bad at `1024`.

5. Revised scaling plan:
- Mainline: do a `1024` widened warm-start on FineWeb. That is now my recommended production path.
- Control: also run a `1024` scratch baseline, but as a control, not the only plan.
- Retest killed mechanisms one at a time only after the widened baseline is stable.
- Decision rule: if widened warm-start wins early slope and actually activates the new dims, keep it as the lineage. If scratch catches or surpasses once calibration settles, keep warm-start as bootstrap only.

So the revised answer is: not `scratch instead of warm-start`, but `widened warm-start plus a scratch control`. If you want, I can patch a real `768 -> 1024` widening loader into the repo next.