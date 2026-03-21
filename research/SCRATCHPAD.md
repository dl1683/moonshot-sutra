# Sutra Scratchpad — Active Unvalidated Ideas

Ideas graduate to RESEARCH.md only after Codex validation + empirical signal.
See git history for archived ideas from earlier sessions.

---

## PRIORITY 1: Telescopic Memory (Codex-designed, v0.5.5 candidate)

Multi-resolution memory: discourse (gist) → chunk (paragraph) → token (exact recall).
The model learns WHICH zoom level each position needs via successive refinement.

**Codex design (results/codex_telescopic_memory.md):**
- Level 0: current scratchpad (gist, 8 slots, EMA)
- Level 1: chunk cache (64-token windows, compressed summaries)
- Level 2: exact token retrieval (immutable snapshots, restricted to top-2 chunks)
- Zoom: successive refinement — discourse picks chunks, chunks restrict exact search
- Cost: ~1.5M params (3%), **12.8x cheaper** compute than current global router
- Warm-startable from v0.5.4 via zero-init gates

**Stage integration:**
- Stage 4 → telescopic global read
- Stage 5 → writes to all levels
- Stage 6 → owns zoom depth (uncertainty-driven)
- Stage 7 → low verify forces deeper re-read

**Key Codex corrections to initial sketch:**
- Three memories must be hierarchy (parent→child), not independent
- Zoom query needs compute penalty or it collapses to always-coarse
- Hidden states aren't "exact" — need immutable leaf snapshots with source IDs
- Explains why multi-timescale scratchpad failed: fixed decays aren't query-conditional zoom

**Score**: Turing 6/10 upside. Breakthrough if proven rate-distortion optimal.
**Status**: Codex-designed. Queue for dim=1024 scale-up or v0.5.5 Chrome.

---

## Scale-Up Ideas (for dim=1024)

### Contractive Hyperspherical Core
Normalized state geometry: keep tokens on a sphere, cosine interactions, bounded updates.
Scale-invariant because governing quantities are angles/norms, not width.
Source: Codex architecture analysis + nGPT research.
Status: UNTESTED. Design phase.

### RG/MERA Stage Pyramid
Explicit multiscale graph: token → phrase → segment → document nodes.
Wavelet up/down passes, shared local update rule across scales.
Source: 15-domain research synthesis (wavelets, RG, MERA, holography).
Status: UNTESTED. Strong theoretical support.

### Spatially Coupled Predictive-Coding Graph
Sparse error-only communication on expander graphs, BP-style iterative updates.
Replace dense scratchpad broadcasts with sparse residual packets.
Source: coding theory (LDPC/BP), predictive coding, spatial coupling.
Status: UNTESTED. Best replacement for current scratchpad/router.

### Grokfast at Scratch Training
+11% at dim=128, diverges at dim=768 warm-start. Retest from scratch at dim=1024.
May need much lower lambda (0.01-0.1) at larger scale.
Status: DEFERRED. Module ready at code/grokfast.py.

---

## Mechanisms to Retest at Scale

All killed at 69M but may work at 105M+ where model has capacity for learned control:

| Mechanism | Result at 69M | Retest at |
|-----------|--------------|-----------|
| Complex embeddings | -36% | dim=1024 scratch |
| CfC time constants | -14% | dim=1024 scratch |
| Lambda halting | AUROC 0.36 | dim=1024 scratch |
| Error scratchpad | -0.4% (late 2.2x better) | dim=1024 with delayed start |
| Surprise bank | -1.7% to -2.1% | dim=1024 scratch |
| Grokfast | +11% dim=128, diverges dim=768 | dim=1024 scratch |
| Dendritic neurons | untested at production | dim=1024 scratch |
| nGPT hypersphere | untested at production | dim=1024 scratch |
| Tropical routing | untested at production | dim=1024 scratch |

---

## Data Ideas

- NCA pre-pre-training: 164M NCA tokens gave 6% LM improvement in literature. Test before language data.
- Curriculum: start with TinyStories (narrative coherence) → educational → full mix
- Compression ratio as eval metric (r=-0.95 correlation with benchmarks per COLM 2024)

---

## Open Questions

1. Why does Grokfast diverge at dim=768? Is it gradient magnitude scaling? Or warm-start specific?
2. Can we predict dim=768 behavior from dim=128 probes? What scaling law would work?
3. The delayed-start principle: is step 3 universal or should it adapt per mechanism?
4. Net2Net widening: will the extra dimensions actually activate, or stay dead?
