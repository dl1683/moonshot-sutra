**Bottom Line**

`[v053_metrics.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/v053_metrics.json)` does **not** tell you where Sutra stands vs Pythia-70M. It is an internal Sutra-only training eval. The honest position today is: **v0.5.3 is improving on its own held-out MiniPile eval, but you do not yet have a valid v0.5.3 vs Pythia-70M head-to-head number.**

**Are The BPT Numbers Comparable?**

- `[v053_metrics.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/v053_metrics.json)` says Sutra v0.5.3 hit `5.9579` BPT at step `5000`.
- `[sutra_v05_train.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_train.py)` shows that number comes from `evaluate()`: 20 **random** batches sampled from held-out MiniPile tokens. Good for training tracking, not competitor benchmarking.
- `[pythia_baseline.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/pythia_baseline.json)` is **Pythia-160M**, not 70M, on a different 50K-char corpus.
- `[clean_benchmark_pythia.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/clean_benchmark_pythia.json)` is **Pythia-70M** on `[clean_benchmark.txt](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/clean_benchmark.txt)`, a separate tiny clean benchmark.

So: **no, these BPTs are not apples-to-apples**. Different model sizes, different datasets, different evaluation paths.

Also, `[clean_eval.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/clean_eval.py)` evaluates each model with its own tokenizer. That makes **BPB** the fair cross-model metric; **BPT is only fair when tokenizer and text are fixed**.

**What Benchmark Should We Use?**

Use a **single matched benchmark**:
- Same exact text
- Same exact script
- Same checkpoint policy
- Report **BPB primary**, BPT secondary

The repo already has the right shape for this in `[clean_eval.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/clean_eval.py)` and `[head_to_head.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/eval/head_to_head.py)`. For a fair head-to-head, I’d use a `head_to_head.py` style eval on a **larger fixed held-out corpus**, not the current 5.3KB clean benchmark. The clean benchmark is fine as a sanity check, too small for a strong claim. For public claims, add a standard `lm-eval` sweep, which `[CLAUDE.md](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/CLAUDE.md)` already requires.

**What Is The Honest Gap?**

- If you naively compare `5.9579` vs Pythia-70M’s `3.5592`, Sutra looks `67.4%` worse in BPT.
- But that comparison is **invalid** because the evals are different.

The closest honest matched evidence already in the repo is `[clean_eval_run4.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/clean_eval_run4.json)` vs `[clean_benchmark_pythia.json](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/clean_benchmark_pythia.json)`:
- Sutra: `9.3393` BPT, `2.3166` BPB
- Pythia-70M: `3.5592` BPT, `0.8524` BPB

That is roughly:
- `2.6x` worse in BPT
- `2.7x` worse in BPB

So the honest statement is: **v0.5.3 has no valid matched Pythia-70M number yet, and the last matched clean evidence says Sutra is still far behind, not close.**