**Findings**

1. Blocker: pass indexing and tensor conventions are inconsistent in the spec. It says “inter-step supervision from pass 0” but also “after pass 12,” “pass `12` vs `1..11`,” and “residual-after-8” without defining whether indices are 0-based or 1-based. See [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L9), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L141), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L318), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L446).  
Exact fix: define one convention and use it everywhere. I’d write:
- internal pass index is `p in {0..11}`
- `mu_hist` shape is `(B, T, 12, D)`
- `pi_hist` shape is `(B, T, 12, 7)`
- `probe_pred` shape is `(B, T, 12)`
- “pass 8” means `p=7`
- final pass means `p=11`

2. Blocker: the sampled objective is missing exact math in two places. “sampled margin” is never defined, and the spec says to “collect sampled margins” during the loop even though candidate IDs are built only after final logits exist. See [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L54), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L112), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L138).  
Exact fix: add:
- `candidate_ids` are built once from final-pass logits after `p=11`
- `sampled_ce_hist` and `sampled_margin_hist` are computed post hoc from `mu_hist`
- per-pass sampled scores use `F.linear(LN(mu_p), emb[candidate_ids]) / sqrt(dim)`
- `sampled_margin_p = score[target_slot_0] - max(score[negative_slots])`
- random-table fill must exclude the target and already-selected negatives

3. Major: forced-freeze/cache semantics are not precise enough for consistent implementation. “FrozenPrefixCache” implies prefix-causal behavior, but `force_freeze_mask` is not defined as prefix-only, and “later-token BPT delta on tokens after frozen positions” is underspecified. See [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L73), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L145), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L299).  
Exact fix: add:
- `force_freeze_after_pass=4` means tokens update through `p=3`/human pass 4 and are frozen starting `p=4`/human pass 5
- `force_freeze_mask` has shape `(B, T)`
- cache reads for query position `t` may use only frozen positions `j < t`
- “later-token BPT delta” is computed only on tokens with at least one frozen earlier position in the same sequence

4. Major: evaluation protocol is still too loose for proceed/stop gates. Bucket fixation, ranking population, baseline fit/eval split, and JSON artifact schema are all left open. See [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L252), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L268), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L284), [spec](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060a_buildable_spec.md#L365).  
Exact fix: add:
- easy/hard buckets are computed once from pass-1 full CE on the fixed eval slice and held fixed for all later passes
- `residual_after_8 = CE_pass8 - min(CE_pass9..CE_pass12)`
- top-decile recall is over all evaluated tokens on that fixed slice
- logistic baseline is fit on the first half of the eval slice and scored on the second half using features `[margin, margin_slope]`
- each result JSON must include `experiment`, `step`, `n_tokens`, `metrics`, `thresholds`, `status`

**Answers**

1. Is it complete enough to hand to a developer and say “build this”?  
Not yet. It is close, but the four fixes above should be made first.

2. Are there ambiguities that would cause implementation mistakes?  
Yes. The main ones are pass numbering, sampled-margin math/order, freeze/cache semantics, and eval-gate definitions.

3. Given the three FAILs, is from-scratch 12-pass the right approach?  
Yes. The evidence is aligned:
- uniform convergence means controller/freezing is premature: [experiment_convergence.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/experiment_convergence.json#L4)
- steps 9-12 are OOD beyond the trained 8-pass ceiling: [experiment_dense12.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/experiment_dense12.json#L21)
- full per-pass readout is too expensive, so sampled supervision is the right constraint-aware substitute: [experiment_readout_cost.json](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/experiment_readout_cost.json#L9)

4. Is 8-10 hours realistic?  
Slightly optimistic. With the spec cleaned up and heavy reuse of `v0.5.4`, maybe. For actual implementation plus debug/OOM/parity/eval wiring, `10-14` hours is the safer estimate.

5. Any final objections?  
No architectural objection. The direction is correct: probe-only, from-scratch, dense-12 first. The only remaining objections are spec precision issues above.

Not approved yet. Exact remaining fixes needed are the four items above. After those are patched into the final spec, it is ready for implementation.