# Sutra Architecture Reference

**Status: Clean slate (2026-03-25)**

This file is the architecture source of truth for Sutra. Every T+L round reads it, questions it, and updates it only when local evidence justifies the change.

---

## Current Architecture

**None decided.** The next T+L design session will propose an architecture from first principles, informed by the 5 outcomes and field research in RESEARCH.md.

---

## What We Know Works (Process Knowledge, Not Architecture Commitments)

These are lessons from prior experimentation. They inform the search but do not constrain it — any of these could be wrong due to implementation bugs.

- **16K custom BPE tokenizer** — Consistently the biggest single efficiency win across all experiments. Keep unless there's a strong reason to change.
- **Warm-starting** — Continuing from checkpoints consistently outperforms training from scratch.
- **WSD learning rate schedule** — Warmup-Stable-Decay with warm restart showed good training dynamics.
- **Fixed early exits** — Post-hoc threshold calibration on fixed exits showed compute savings without learned halting overhead. Worth exploring again.

## What We Tried and Failed (Caveat: Possibly Implementation Issues)

At 98M scale, the following were tested and lost to a plain dense AR control. However, we had multiple bugs during development, so these failures may not be universal truths:

- MTP D=1 (multi-token prediction) — lost to control, 10% slower
- Learned halting controller — lost to control, 21% slower
- N-gram memory fusion — lost to control, gate barely opened
- All GPU-side auxiliary objectives that competed for backbone capacity

**The right response is not "these don't work" but "our implementations didn't work at 98M scale." A fresh T+L session should decide whether to retry any of these with better implementations or at larger scale.**

---

## Hardware Constraints

- **GPU:** Single NVIDIA RTX 5090 Laptop (24GB VRAM)
- **RAM:** 68GB system
- **Data:** ~22.9B tokens available, 246 shards, 18 sources, 16K custom BPE tokenizer
- **Target:** Trainable from scratch on this hardware, deployable on edge

---

## Token Accounting Reference

- Config template: batch_size=8, grad_accum=4, seq_len=512 = **16,384 tokens/optimizer step**
- Step 1000 = 16.4M tokens
- Step 5000 = 81.9M tokens
- Step 10000 = 163.8M tokens
- Chinchilla-optimal for 98M params = ~2B tokens

---

## Design Decision Audit

*To be populated by the next T+L session.*

| Decision | Status | Confidence | Evidence |
|----------|--------|------------|----------|
| *None yet* | | | |

---

## Per-Outcome Confidence

| Outcome | Score | Why |
|---------|-------|-----|
| O1: Intelligence | 0/10 | No current model. Fresh start. |
| O2: Improvability | 0/10 | No current model. |
| O3: Democratization | 0/10 | No package, no composability proof. |
| O4: Data Efficiency | 0/10 | No current approach decided. |
| O5: Inference Efficiency | 0/10 | No current model. |

---

## Training Priorities

*To be set by the next T+L session.*
