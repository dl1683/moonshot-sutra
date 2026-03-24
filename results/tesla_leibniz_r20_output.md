**Revised Scores**

| Outcome | Score | Evidence |
|---|---:|---|
| O1 Intelligence | **4.5/10** | P1a step 5000 is a real improvement (`D10=6.746`, `D12=6.760`), but generation is still poor across the project record, and no matched dense control exists. BPT compression is not enough to claim stronger intelligence. |
| O2 Improvability | **6.5/10** | The project is strong at trainer diagnosis and probe design, but weak on demonstrated model-local improvability. No clean module swap, ABI, or component-isolated improvement has been proven. |
| O3 Democratization | **2.5/10** | Adapter packs, registry, and frozen-stage composition are still design intent, not demonstrated capability. |
| O4 Data Efficiency | **4.0/10** | P1a 5K is promising, but there is still no teacher-free continuation control from the same checkpoint. At step 5050, teacher CKA is still tiny (`0.013/0.017`). Causal KD benefit is unproven. |
| O5 Inference Efficiency | **5.0/10** | There is real depth structure, but v0.6.1 falsified tokenwise control. Current gains come from fixed late-pass concentration, not earned learned compute allocation. |

The auditor’s range is basically right. My point estimates are `4.5 / 6.5 / 2.5 / 4.0 / 5.0`.

**Audit Response**

1. **Accept.** The workflow is anti-falsification biased. The static template literally discourages dense baselines and killing directions. That is a methodological bug and should be overridden immediately.
2. **Accept.** O1=5 was generous. P1a 5K shows better `D10/D12`, but generation remains weak, and benchmark evidence is still too soft to score intelligence higher.
3. **Accept.** O2=8 was inflated by trainer debuggability. Good instrumentation is not the same as proven architectural improvability.
4. **Accept.** O4=5 was not earned. Without a teacher-free continuation from the same checkpoint, KD causality is still a hypothesis.
5. **Accept.** O5=7 was inflated. v0.6.1 falsified tokenwise control; current evidence only supports fixed-depth late-pass usefulness.
6. **Accept.** The “26M core” story is accounting, not causal validation. It remains a plausible hypothesis, not a demonstrated root cause.
7. **Accept.** Matched dense control is the most important missing experiment. It moves from “later branch” to mandatory falsifier.
8. **Accept.** The project has been overproducing design and underproducing decisive falsifiers. R20 should reverse that.
9. **Accept.** At this scale, the recurrence thesis is unproven. I cannot honestly argue the dense-vs-P2c prior below the auditor’s concern without a real control.
10. **Mostly accept.** I differ only slightly on point estimates, not direction.

**Falsifying Experiments**

1. **Teacher-free continuation control**
- Parent checkpoint: **P1a step 5000**. If step 6000 beats it at the next deterministic eval, use the better of `{5000, 6000}`.
- Arms: `KD-on` and `KD-off`, both restoring the **same optimizer state, LR schedule, RNG, and dataloader position**.
- Length: **1000 steps** minimum.
- Eval: steps `0 / 500 / 1000` with fixed 97-window deterministic `D1/D4/D8/D10/D12`, fixed generation panel, and watcher tasks `SciQ, PIQA, ARC-E, ARC-C, HellaSwag, WinoGrande, LAMBADA`.
- Falsifier: if `KD-off` is within **0.05 BPT** at `D10` and `D12` and is not clearly worse on the watcher set, O4 drops and KD leaves the critical path. If `KD-on` wins by **>=0.10** on `D10` or `D12` and wins a majority of watcher tasks, KD survives.

2. **Widen-only canary**
- Parent checkpoint: same best P1a checkpoint as above.
- Change: keep tokenizer, recurrence, losses, and data fixed; widen only the recurrent compute path by expanding StageBank FFNs from **1536 -> 2304** with a **zero-gated additive branch** so the checkpoint is function-preserving at step 0.
- Training: **500 steps teacher-free**.
- Schedule: first **200 steps** train only new width and gates; next **300 steps** unfreeze all with old:new LR ratio `0.1:1.0`.
- Eval: steps `0 / 250 / 500`, same deterministic depth slice and generation panel.
- Falsifier: if widening does **not** improve `D10` or `D12` by at least **0.05** versus the paired teacher-free baseline, the “core too small” story loses primary status.

3. **Matched dense control**
- Architecture: plain decoder-only dense transformer, **12 layers**, `d_model=512`, `n_heads=8`, **SwiGLU FFN=1536**, **RMSNorm**, **RoPE**, **tied 16K embeddings**. Total budget is about **49.3M params**, which is close enough to the planned recurrent budget.
- Data/recipe: same **16K retokenized data**, same sequence length, same optimizer/schedule, same token budget. Run **teacher-free first** to isolate architecture. If recurrence survives, repeat with matched KD later.
- Training length: **10K steps**, with decision gates at **2K, 5K, 10K**.
- Eval: fixed validation slice, same watcher tasks, same generation panel.
- Falsifier: if dense beats the recurrent 16K arm on **4/7 watcher tasks**, or matches BPT while giving clearly better generation, recurrence loses primary-track status at this scale.

4. **One extra cheap falsifier worth adding**
- Run a **same-scaffold D=1 16K control** once the 16K line exists.
- Gate: `D>1` must beat `D=1` on **compute-normalized** BPT and generation, not just raw BPT.

**P1a Decision**

Do **not** wait for 15K.

Step 5000 already cleared the R19 transplant gate, and the extra 10K steps would cost about **63 hours** while the biggest unanswered questions are causal, not incremental. The fact pattern is: `D10/D12` improved, `D8` kept worsening, and quality is still concentrating into late depth. That is useful, but it does not justify another 2.6 days before controls.

My recommendation is:

- Let the live run reach the **next deterministic eval at step 6000**.
- Keep the better of `{5000, 6000}` as the parent.
- Then **stop P1a** and reallocate the GPU to the falsifiers above.

**Updated Branch Order**

Old order: `P2a -> P2b -> P2c -> P2d`

New order: `P1a stop@6K -> teacher-free control -> widen-only canary -> paired 16K recurrent vs matched dense -> only then factorized recovery / widen+teachers`

**Research / Probe Requests**

- Remove the anti-falsification clause from the workflow template.
- Standardize one eval bundle for every branch: deterministic depth slice, fixed generation panel, watcher tasks.
- Save fork manifests with optimizer state, RNG state, and dataloader offset so controls are actually paired.
- Prepare the 16K dense config now; the retokenization backlog is no longer the blocking issue.