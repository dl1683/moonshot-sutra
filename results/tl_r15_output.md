**B) Pivot away from standard matching-KD.**  
Run exactly one final KD falsification run: **from-scratch Predictive Coding Distillation (PCD)** on hidden states. If it misses the gates below, stop KD-loss research for O4 and move to teacher-guided data selection/curriculum. Do **not** spend R15 on more InfoNCE/logit/alpha tuning.

**Exact config**  
All unlisted trainer/data/optimizer hyperparameters stay **identical to the CE-WSD from-scratch baseline**.

- Init: step `0`, from scratch
- Total steps: `6000`
- Eval cadence: every `500` steps
- Control: matched `CE-WSD 6K` run required for final decision
- Teacher: `Qwen3-1.7B` only
- Surface: hidden-state prediction, not matching logits
- Layer pairs: `7→8`, `15→16`, `23→24`
- Pooling: `byte_span_pool`, `n_spans=32`
- Student prediction heads: `3 x Linear(768, 2048, bias=False)`
- Precision heads: `3 x LayerNorm(768) -> Linear(768, 1, bias=True)`, bias init `-2.2`
- Head LR: `6e-4`
- Head WD: projectors `0.01`, precision heads `0.0`
- Target: teacher pooled states, `L2`-normalized, `stopgrad`
- Prediction: student pooled states through projector, `L2`-normalized
- Active spans: top `20%` spans by **student normalized entropy** in the current batch
- Span entropy: mean token entropy inside each byte span, normalized by `log(vocab_size)`
- Loss per active span: `Huber(pred, target, delta=0.5)`
- Layer weights: `[0.25, 0.50, 0.25]`
- Total loss:  
  `L = L_CE + λ_pcd * Σ_l w_l * mean_active[ sigmoid(precision_l) * Huber(pred_l, tgt_l) ]`
- `λ_pcd` schedule:
  - steps `1-500`: `0.00 -> 0.30`
  - steps `501-1800`: hold `0.30`
  - steps `1801-3000`: `0.30 -> 0.10`
  - steps `3001-6000`: `0.10 -> 0.03`

**Kill criteria**
- step `500`: kill if `BPT > 7.61`
- step `1000`: kill if `BPT > 6.56`
- step `1500`: kill if `BPT >= CE-WSD`
- step `3000`: kill if `BPT > 4.80`
- step `6000`: kill line if `ΔBPT > -0.20` vs matched `CE-WSD 6K`
- Auxiliary kill: if active-span rate is `<5%` or `>35%` for 2 evals in a row

**Success criteria**
- step `3000`: `BPT <= 4.75`
- step `6000`: `ΔBPT <= -0.25` vs matched `CE-WSD 6K`
- No eval after step `1000` may trail control
- Plus one precision-side win: `ECE -10%`, or probe accuracy `+1pp`, or mid-layer `CKA +0.05`

**Rationale**
- `R5-R11`: warm-start KD is dead.
- `R12`: cross-tokenizer logit KD is dead.
- `R13`: hidden-state transfer is the only real signal.
- `R14`: higher fixed alpha improved KD loss while hurting LM; the problem is **global matching pressure**, not lack of optimization.
- PCD is fundamentally different: it makes the student **predict teacher states**, applies KD only on the **student’s hardest spans**, and learns **local precision** instead of forcing a fixed global alpha.
- That matches the literature direction in the scratchpad: from-scratch > warm-start, hidden states contain more transferable signal than output logits, and student-entropy gating is the best selector.

**Confidence**
- O1 Intelligence: `5/10`
- O2 Improvability: `8/10`
- O3 Democratization: `7/10`
- O4 Data Efficiency: `6.5/10`
- O5 Inference Efficiency: `8/10`

If this misses the `3K` gate, end KD-mechanism work. The next move should be **teacher-guided data curriculum**, not R13/R14-style KD.