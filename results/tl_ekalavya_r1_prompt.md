# Tesla+Leibniz Round 1: Ekalavya Protocol Design

MANDATORY FIRST STEPS:
1. Read research/TESLA_LEIBNIZ_CODEX_PROMPT.md — your full persona and task spec
2. Read research/VISION.md — the 5 non-negotiable outcomes
3. Read research/ARCHITECTURE.md — architecture locked at Sutra-24A-197M, RMFD system from R8
4. Read research/RESEARCH.md Section 6.4 — ALL multi-teacher KD literature (20+ papers)
5. Read code/dense_baseline.py lines 1458-1700 (KD components) and 4574-5048 (phased training loop)
6. Read CLAUDE.md — project constitution

## CONTEXT

Architecture is LOCKED: Sutra-24A-197M (768d, 24L, 12h GQA, SwiGLU, 196.7M params).
ALL design energy focuses on Ekalavya Protocol (multi-teacher cross-architecture KD).

We are independent researchers (one person, single RTX 5090). We need something EXCEPTIONALLY WEIRD — multi-teacher cross-architecture KD is that thing.

Current state: 60K control baseline just completed (BPT=3.5726). Checkpoint ready as warm-start.

Available teachers (locally cached):
- Qwen/Qwen3-1.7B-Base (~3.4GB FP16, decoder, hidden=2048)
- LiquidAI/LFM2.5-1.2B-Base (~2.4GB FP16, hybrid)
- Qwen/Qwen3-0.6B-Base (~1.2GB FP16, decoder, hidden=1024)
- google/embeddinggemma-300m (~0.6GB FP16, embedding)
VRAM: Student ~5GB total, remaining ~19GB for teachers. All fit.

Existing infrastructure: TeacherAdapter, byte_span_pool, CKA losses, train_kd_phased.

MISSING (what you design): routing, gradient conflict resolution, multi-depth matching, teacher curriculum, alpha scheduling. NO mechanism is sacred — only the OUTCOME matters (learning from every model simultaneously). Propose whatever achieves this best.

## YOUR TASK

Design the complete Ekalavya Protocol with EXTREME GRANULARITY. Follow Phase A/B/C from your persona document. Include cross-domain research directions (biology, physics, neuroscience).

Output format: assumption challenges, research requests, probe requests, per-outcome confidence scores, design specification.
