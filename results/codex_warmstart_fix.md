1. Best option: `B` in principle, `A` in production.

The first-principles rule for warm-starting is: the new model must contain the old model as an exact subcase. Current v0.5.4 does not. It inserts six fresh `LayerNorm`s in the recurrent core at [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L95) and applies them at [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L135), [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L145), and [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L159). The trainer only lowers LR and delays Grokfast at [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py#L43) and [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py#L232); that cannot fix a broken first forward pass.

I ran a direct checkpoint probe on `results/checkpoints_v053/step_5000.pt`. On the same sample batch, v0.5.3 was `5.9006` BPT and current warm-started v0.5.4 was `10.5686` BPT, with only `45.3%` top-1 agreement. That is an initialization discontinuity, not a training-speed issue.

`C` is not viable as stated: `LayerNorm` has no running stats to initialize. Fitting `gamma/beta` cannot undo per-sample mean-centering. `D` is also not enough: I tested an ungated RMS-style swap and it was still `10.4184` BPT on the same checkpoint probe. The problem is not “LayerNorm specifically”; it is “non-identity normalization inserted into a trained recurrent system.”

2. Exact implementation spec: use a homotopy wrapper, not plain LN.

Use `y = x + alpha * (Norm(x) - x)`. That is the only version that makes `alpha=0` an exact identity path. Do not gate only `gamma/beta`; gate the whole normalization branch.

Spec:
1. Replace each fresh Peri-LN in [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L95) with a `BlendNorm` wrapper around `nn.LayerNorm`.
2. Use one shared scalar `peri_alpha` for all six wrappers. Shared is better than six independent gates because it gives a coherent system-level continuation from v0.5.3 to v0.5.4.
3. In `warmstart_v054()` at [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L196), set `peri_alpha = 0.0`.
4. In training, set `peri_alpha` before each forward pass:
   `0.0` for steps `0..249`
   cosine ramp `0.0 -> 1.0` for steps `250..2249`
   `1.0` after that
5. Delay Grokfast until the ramp has started. I would set `GROKFAST_DELAY = 500` in this variant so optimizer adaptation is not confounded with an architecture jump.

On my checkpoint probe, the same blend with `alpha=0` stayed at `5.8991` BPT, essentially preserving v0.5.3. That is the key property you need.

3. The `+5.9%` Peri-LN Chrome result is not valid evidence for warm-start.

It is valid only for the regime it tested: scratch training at `dim=128`, `300` steps, in [chrome_v054_ablation.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/chrome_v054_ablation.py#L35), [chrome_v054_ablation.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/chrome_v054_ablation.py#L39), and [chrome_v054_ablation.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/chrome_v054_ablation.py#L342). It says Peri-LN can help when the whole system co-adapts from random init. It does not say you can splice LN into a trained fixed point and keep function continuity.

4. Yes: deploy Grokfast-only now, and save Peri-LN for either gated warm-start or scratch-only.

The Chrome results already point that way. In [chrome_v054_ablation.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/chrome_v054_ablation.json#L17), `peri_ln` helps from scratch; in [chrome_v054_ablation.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/chrome_v054_ablation.json#L53), `peri_pheromone` is basically flat; in [chrome_v054_ablation.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/chrome_v054_ablation.json#L71), full v0.5.4 is weaker than Peri-LN alone; and in [chrome_v054_ablation.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/chrome_v054_ablation.json#L89), the clear production win is Grokfast.

So my recommendation is:
1. Ship `v0.5.3 + Grokfast` as the immediate recovery path.
2. If you still want warm-started Peri-LN, implement the gated blend version above and validate it separately.
3. Treat ungated Peri-LN as scratch-only until proven at production scale.

If you want, I can patch the gated `BlendNorm` version directly into [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) and [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py) next.