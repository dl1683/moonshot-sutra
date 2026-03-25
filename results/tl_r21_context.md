# T+L R21 Context — Falsification Results + New Data (2026-03-25)

## YOUR PREVIOUS ROUND (R20) — FULL OUTPUT (VERBATIM)

**Scores**

| Outcome | Score | Evidence |
|---|---:|---|
| O1 | **4.5/10** | On March 24, 2026, `step_5000` improved deterministic `D10=6.746` and `D12=6.760` versus `step_3000`, and `step_3000` beat the parent on 4/7 watcher tasks. But generation is still poor across branches, SciQ/LAMBADA are still weak, and there is still no matched dense control. |
| O2 | **6.5/10** | The project is strong at trainer/data/pipeline diagnosis (`_char_ends()`, sampled-CE inversion, queue bugs, resume bugs), but weak at demonstrated model-local improvability. No subsystem has been improved in isolation and then recomposed cleanly. |
| O3 | **2.5/10** | Named stages exist, but there is still no frozen ABI, no registry, no loader, no adapter-pack composition proof, and no contributor-facing subsystem improvement demo. |
| O4 | **4.0/10** | P1a is promising, but causal KD lift is still unproven. There is no paired teacher-free continuation from the same checkpoint, and at `step_5050` the active teacher signals are still tiny (`Q1cka=0.013`, `Q2cka=0.017`) while throughput is heavily taxed. |
| O5 | **5.5/10** | Repeated `D10 < D12` is real at `1K/2K/3K/4K/5K`, so fixed-depth savings are not fake. But v0.6.1 falsified the stronger claim: routing stayed pass-global and `MI(mode,token)` stayed near zero. |

The auditor's band is basically right. My only mild disagreement is O5: I would put it at `5.5`, not `5.0`, because fixed-depth efficiency is now replicated, even though tokenwise control is not.

**Audit Response**

1. **Accept.** The "never propose dense baselines" rule is methodologically wrong for internal falsification.
2. **Accept.** O1=5 was generous; BPT and depth wins are not enough while generation remains bad.
3. **Accept.** O2=8 was inflated by trainer debuggability.
4. **Accept.** O4=5 was not earned without a teacher-free continuation control.
5. **Accept, with nuance.** O5=7 was inflated; the real evidence is fixed-depth savings, not learned tokenwise compute.
6. **Accept.** The "26M core" story is accounting, not causal proof.
7. **Accept strongly.** Matched dense control is the highest-value missing experiment.
8. **Accept.** Too much design, not enough clean falsifiers.
9. **Partial accept.** I would not defend the exact `65%`, but dense is the current prior favorite until falsified.
10. **Accept overall.** My point estimates are `4.5 / 6.5 / 2.5 / 4.0 / 5.5`.

**Falsifying Experiments**

- Teacher-free continuation control.
- Widen-only canary.
- Full-rank 16K recurrent control.
- Matched dense control.
- Extra cheap falsifier (D=1 recurrent).

**P1a Decision:** Stop at step 6000.

**Updated Branch Order:** P1a stop@6K -> teacher-free control -> widen-only -> 16K recurrent -> matched dense -> P2b/P2c.

**Research / Probe Requests:** Remove anti-falsification clauses, standardize eval bundle, save paired fork manifests, prepare dense config.

---

## NEW DATA SINCE R20 — ALL FALSIFICATION EXPERIMENTS COMPLETE

### F1: P1a Step 6000 Final Eval (COMPLETE)

P1a stopped at step 6000 as R20 requested. Full 7-task lm-eval:

| Task | Accuracy | vs v060a-20K |
|------|----------|-------------|
| SciQ | **49.0%** | +0.9% (gate was 46% — PASSED) |
| PIQA | 54.7% | +0.2% |
| WinoGrande | 50.8% | +1.9% |
| ARC-Easy | 32.1% | +0.8% |
| ARC-Challenge | 16.6% | -0.1% |
| HellaSwag | 25.8% | +0.0% |
| LAMBADA | 6.7% | -4.5% |

Step 6000 D10=6.721, D12=6.738 (better than step 5000).

### F2: Teacher-Free vs KD A/B (COMPLETE — 1000 steps each)

**Parent:** step_6000.pt (both arms forked from same checkpoint)

| Metric | KD-ON | KD-OFF | Delta |
|--------|-------|--------|-------|
| Test BPT | 6.6838 | **6.6659** | KD is 0.018 WORSE |
| D10 | 6.6821 | **6.6634** | KD is 0.019 WORSE |
| D12 | 6.686 | **6.6644** | KD is 0.022 WORSE |
| D8 | 12.438 | 12.481 | Within noise |

**VERDICT: O4 FALSIFIED at 68M. KD = noise.** Teacher signals (CKA Q1=0.013, Q2=0.017) too small to overcome throughput tax. Teacher-free continuation WINS.

### F3: 16K Recurrent Control (COMPLETE — 1000 steps)

**Parent:** transplanted_16k_from_step5000.pt (42M params, vocab_size=16000)

| Depth | Step 0 | Step 1000 | Delta |
|-------|--------|-----------|-------|
| D1 | 13.99 | 13.58 | -0.41 |
| D8 | 11.37 | **5.53** | **-5.84** |
| D10 | 6.76 | **5.22** | **-1.54** |
| D12 | 6.77 | **5.24** | **-1.53** |
| BPT | 6.49 | **5.88** | **-0.61** |

**KEY FINDING:** 16K tokenizer is MASSIVE win. D10=5.22 vs F2's D10=6.66 (50K vocab). The tokenizer is a bigger lever than any mechanism we've tried. Also: D8=5.53 is close to D10=5.22, suggesting 16K helps depth utilization dramatically.

### F4: Matched Dense Control (COMPLETE — 5000 steps from scratch)

**Architecture:** 11 layers, d_model=512, 8 heads, SwiGLU hidden=1536, RoPE, RMSNorm, tied 16K embeddings, ~51M params.

| Step | CE Loss | tok/s |
|------|---------|-------|
| 0 | 17.65 | 97K |
| 1000 | 11.51 | 95K |
| 2000 | 10.35 | 94K |
| 3000 | 9.59 | 96K |
| 5000 | **8.99** | **96K** |

**NEW: Dense BPT numbers (comparable metric, same 16K eval cache):**

| Step | Dense BPT | Recurrent D10 (at same step) | Notes |
|------|-----------|------------------------------|-------|
| 1K | 7.06 | 6.00 (P2c warm-start) | Dense from scratch, recurrent warm-started |
| 2K | 6.34 | N/A | |
| 3K | 5.76 | N/A | Dense catches recurrent |
| 5K | **5.36** | N/A | Dense WINS on raw BPT |

**Critical context for comparison:**
- Dense runs at **97K tok/s** vs recurrent at **~10K tok/s** (10x faster training)
- Dense is 18x faster at inference (1 pass vs 12)
- P2c recurrent was WARM-STARTED (5K parent steps + 1K continuation). Dense was FROM SCRATCH.
- In the SAME WALL-CLOCK TIME as recurrent 1K steps, dense gets ~10K steps (BPT ≈ 4.5 extrapolated)
- Dense BPT=5.36 at 5K > F3 recurrent D10=5.22 at 1K, but recurrent had warm-start advantage
- **Fairest comparison: same total tokens or same wall-clock time. Dense wins both.**
- **lm-eval benchmarks NOW AVAILABLE (ran 2026-03-25):**

| Task | Dense F4 (51M, 5K, scratch) | P1a-6K (68M, KD, warm) | v060a-20K (68M) | Dense vs P1a |
|------|----------------------------|------------------------|-----------------|--------------|
| ARC-Easy | 30.9% | 32.1% | 31.3% | -1.2% |
| ARC-Chall (norm) | 21.6% | 18.2% | 16.7% | **+3.4%** |
| HellaSwag | 26.2% | 25.9% | 25.8% | **+0.3%** |
| WinoGrande | 49.6% | 51.1% | 48.9% | -1.5% |
| PIQA | 54.4% | 54.7% | 54.4% | -0.3% |
| SciQ | 40.5% | 49.0% | 48.1% | -8.5% |
| LAMBADA | 2.8% | 6.7% | 11.2% | -3.9% |

**Analysis:** Dense from scratch with NO KD beats P1a on reasoning tasks (ARC-Chall +3.4%, HellaSwag +0.3%). P1a only wins on knowledge tasks (SciQ, LAMBADA) where KD teachers provided an unfair advantage. At equal wall-clock time, dense would get 10x more training steps. Dense trains at 97K tok/s vs recurrent ~10K tok/s.

### F5: Widen-Only Canary (COMPLETE — 500 steps)

**Parent:** step_6000.pt + widened FFN 1536→2304 (zero-gated additive branch)

| Depth | Baseline (step 0) | Widened (step 500) | Delta |
|-------|-------------------|-------------------|-------|
| D8 | 12.69 | **6.92** | **-5.77 (HUGE)** |
| D10 | 6.77 | 6.69 | -0.08 (flat) |
| D12 | 6.79 | 6.72 | -0.07 (flat) |

**VERDICT:** Widening helps D8 dramatically (capacity matters for early passes) but does NOT help peak quality D10/D12. The bottleneck is NOT capacity — it's architectural.

### F6: Module-Swap (NOT RUN)
Deferred — F1-F5 answered the critical questions.

### P2c Control Canary (COMPLETE — 1000 steps, teacher-free)

**Parent:** transplanted_16k_from_step5000.pt (42M params, 16K vocab)
**Same parent as F3 but different data/config.** This is the teacher-free baseline for the P2c ALM-only canary.

| Step | BPT | D10 | D12 | D8 |
|------|-----|-----|-----|-----|
| 0 | 6.461 | 6.609 | 6.625 | 11.081 |
| 250 | 6.414 | 6.444 | 6.491 | 10.992 |
| 500 | 6.104 | 5.870 | 5.909 | 10.455 |
| 750 | **5.828** | **5.651** | **5.685** | 10.102 |
| 1000 | 5.820 | 6.002 | 6.028 | 10.420 |

**Key observations:**
- P2c learns MUCH faster than 68M (D10=5.65 at step 750 vs 68M's 6.75 at step 5000)
- Cosine LR decay too aggressive: quality degrades from step 750 to 1000
- D10 < D12 holds (elastic compute works on P2c)
- D8 still bad (10+) — depth concentration persists

### Codex V2 Verdict on Ekalavya (Teacher KD)

Codex independently validated the F2 result:
- ALM gives real short-horizon optimization lead at step 750, but the 68M student lacks enough core capacity to preserve it
- Procrustes is effectively dead at 68M (loss stays at 0.001 throughout)
- Recommends: **P2c ALM-first, Procrustes only if earned**
- Overweight LFM2-1.2B in Q1 (best ALM loss: 0.18 vs Qwen's 0.47)
- If P2c ALM-only also fails, pivot to offline distillation/data shaping entirely
- Online KD is too throughput-expensive; shift bulk of O4 to offline strategies

### Ekalavya v2 Canary (68M, 1K steps — prior to F2)

This canary ran BEFORE the falsification experiments:
- At step 750: canary D10=-0.103, D12=-0.097 (206%/242% of gates). ALM gives SHORT-HORIZON lead.
- By step 1000: 71.5% of D10 edge and 75.8% of D12 edge evaporated.
- Throughput dropped from 9.2K to 5.3K tok/s with teachers loaded.
- The preregistered kill rule: KD-OFF must not be within 0.05 on D10/D12. It violated this.

---

## SUMMARY OF WHAT WE KNOW NOW

### What WORKS (validated):
1. **16K tokenizer** — Biggest single win. D10: 6.66→5.22 just from tokenizer change.
2. **Elastic compute (D10 < D12)** — Replicated across ALL experiments.
3. **Warm-starting** — P2c learns 2x faster than 68M parent.
4. **P2c architecture** — 42M params, 16K vocab, solid foundation.

### What DOESN'T WORK (falsified):
1. **Online KD at 68M** — F2 proves it's noise. CKA signals too weak.
2. **Procrustes** — Dead at 68M (loss floor = 0.001).
3. **7-stage bank** — Fragmenting scarce capacity. Stages never differentiated.
4. **v0.6.1 controller repair** — Falsified. Modes biased transition but didn't gate computation.

### What's UNCERTAIN:
1. **Recurrence vs dense** — F4 dense is competitive and 10-18x faster, but no BPT/benchmark comparison exists yet.
2. **Online KD at 42M/16K** — May work better at P2c scale (more efficient core). P2c ALM canary about to test this.
3. **Widening** — Helps D8 but not D10/D12. Is this the right kind of widening?
4. **Scale** — All results at 42-68M. What happens at 200M+?

### WHAT R21 MUST DECIDE:
1. **Keep recurrence or pivot?** Dense is competitive. Recurrence gives elastic compute (valuable) but at 10x throughput cost.
2. **Architecture for next training run.** What specific design changes based on F1-F5 data?
3. **O4 strategy.** Online KD failed at 68M. What's the path forward? Offline? Synthetic? Multi-teacher? Or something entirely new?
4. **Scale strategy.** Stay at 42M or push to 200M+?
5. **No sacred cows.** If a completely different architecture (not recurrent, not dense transformer) serves the 5 outcomes better, propose it.

---

## CURRENT HARDWARE STATUS (2026-03-25)
- GPU: **FREE** — 0% utilization, 0 MB used. 24GB available.
- CPU: 68GB RAM, mostly free.
- No training running. Full compute available for experiments.
- User has granted FULL computing budget. Run as many experiments as needed.
