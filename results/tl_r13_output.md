**Verdict**

R13 should be `single-teacher, single-surface, multi-depth hidden-state KD from scratch`: Qwen3-1.7B teaches Sutra only through intermediate states, pooled into byte spans, matched at the student’s natural interfaces `7/15/23`. No logit KD. No vocab alignment. No semantic Gram loss in the smoke test. No attention transfer.

1. The best next move is `A + C`, with `B` as the alignment scaffold: hidden-state KD plus contrastive matching on byte-span-pooled states. `D` still inherits a token-grid alignment problem, and `F` is a slower fallback, not the next experiment.

2. Layer mapping: student `7 -> teacher 8`, student `15 -> teacher 16`, student `23 -> teacher 24` (0-indexed transformer blocks). Use cosine-based contrastive matching, not MSE, not pure CKA. MSE over-constrains scale; CKA is too global and already showed “head-start only” behavior.

3. Alignment should be byte-span pooling, not token-to-token ETA. Use `32` equal byte spans per window. That keeps the signal local without recreating the tokenizer mismatch.

4. Hidden-state KD weight: `alpha_state_total` ramps `0.06 -> 0.30` over steps `1-400`, holds at `0.30` for `401-2200`, then decays `0.30 -> 0.03` over `2201-3000`. Depth weights are exact: `0.30 / 0.50 / 0.20` for layers `7 / 15 / 23`.

5. Keep CE as the primary objective. Use hidden-state KD as an auxiliary term:
   `L = L_CE + alpha_state * L_state`
   Do not use `(1-alpha) * CE + alpha * KD`. That mixture made sense for logits; it is wrong for representation shaping.

6. Pre-computing teacher hiddens is not the first smoke-test path. Full-corpus cache is too large. Cache only pooled spans if you later freeze a fixed curriculum. Scale:
   `3 layers x 32 spans x 2048 x fp16 = 384 KB/window`
   `96k-window 3K scout cache ~= 36 GB`
   `full-corpus pooled cache ~= 17 TB`
   `full hidden-state cache ~= 281 TB`
   So: online teacher for the first 3K; offline pooled cache only after a positive result.

7. Expected 3K outcome: `4.65-4.78` BPT. Strong pass is `<= 4.70`. Minimum continue bar is `<= 4.78`. Kill it if final BPT is `> 4.82`.

8. Do not train a same-tokenizer teacher now. It has the best long-run ceiling, but it delays time-to-signal by an entire extra training project. Only start that if two hidden-state scouts fail.

**3K Smoke Protocol**

```python
{
  "name": "scratch_state_contrast_q17_3k",
  "teacher": "Qwen/Qwen3-1.7B-Base",

  "total_steps": 3000,
  "batch_size": 4,
  "grad_accum": 8,
  "seq_len": 512,

  "lr": 3e-4,
  "min_lr": 1e-5,
  "warmup": 200,
  "eval_every": 500,

  "dim": 768,
  "n_layers": 24,
  "n_heads": 12,
  "ff_dim": 2304,
  "exit_layers": [7, 15, 23],
  "norm_type": "ss_rmsnorm",
  "block_schedule": ["A"] * 24,
  "n_q_heads": 6,
  "n_kv_heads": 3,
  "head_dim": 64,
  "conv_kernel_size": 4,

  "kd_surface": "state_only",
  "student_kd_layers": [7, 15, 23],
  "teacher_kd_layers": [8, 16, 24],
  "n_spans": 32,

  "state_loss": "span_infonce_cosine",
  "contrastive_temperature": 0.07,
  "depth_weights": [0.30, 0.50, 0.20],

  "alpha_state_start": 0.06,
  "alpha_state_max": 0.30,
  "alpha_state_min": 0.03,
  "alpha_state_ramp_end": 400,
  "alpha_state_decay_start": 2200,
  "alpha_state_decay_end": 3000,

  "projector": "3 independent linear projectors, 768->2048, bias=False",
  "projector_lr": 6e-4,
  "projector_weight_decay": 0.01,

  "teacher_dtype": "float16",
  "logit_kd": false,
  "semantic_kd": false,
  "attention_transfer": false
}
```

Use student states `exit_hidden[7]`, `exit_hidden[15]`, and `final_norm(hidden)`; do not match every layer. That is the whole point: teach through the architecture’s actual interfaces.

Implementation hook points are [DenseTransformer.forward](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L1400), [TeacherAdapter](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L1566), [compute_state_kd_loss](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L1724), and [train_kd_scratch](/C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L6626). Keep `byte_span_pool`; replace the CKA core with span-level contrastive cosine and add projector param groups exactly like the warm-start KD path.

If this misses `4.82`, stop. The next move is not another cross-tokenizer teacher-side online variant; it is either same-tokenizer teacher training or offline enriched-data KD.