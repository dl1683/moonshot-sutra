# Dual-Loop Supervisor Check-in #13: TERMINAL VERDICT

Date: 2026-07-07
Loops covered: Q-Loop B18-B19 (14 iterations), W-Loop B14 (terminal Eklavya test)

---

## THE VERDICT

**FAIL_EKLAVYA_MECHANISM. Kill #9. Eklavya protocol is dead.**

The terminal test on SmolLM2-135M — a capable, pretrained, BPE student with proven benchmark performance — shows that multi-teacher disagreement routing has NO residual over single-teacher KD. The oracle ceiling itself fails. The learned router is worse than random.

**MOONSHOT PIVOT TRIGGERED.**

---

## Evidence: W-Loop B14 Terminal Test

### Accuracy Table (held-out, n=48 per benchmark)

| Benchmark | Zero-shot | Label-only | Single-T | Oracle | Non-oracle | Random |
|-----------|-----------|------------|----------|--------|------------|--------|
| HellaSwag | 39.6% | 41.7% | 39.6% | 41.7% | 43.8% | 39.6% |
| PIQA | 75.0% | 68.8% | 75.0% | 75.0% | 70.8% | 75.0% |
| ARC-Easy | 50.0% | 54.2% | 62.5% | 58.3% | 56.3% | 60.4% |
| **Mean** | **54.9%** | **54.9%** | **59.0%** | **58.3%** | **56.9%** | **58.3%** |

### Terminal test criteria vs results

| Criterion | Required | Actual | Met? |
|-----------|----------|--------|------|
| Non-oracle >= 3pp over label-only | On >=2/3 benchmarks | 0/3 (all ~+2pp) | NO |
| Non-oracle >= 3pp over single-T | On >=2/3 benchmarks | 0/3 (aggregate -2.08pp) | NO |
| Oracle ceiling > single-T | Necessary for mechanism validity | -0.69pp aggregate | NO |
| Learned router > random | Necessary for signal | -1.39pp aggregate | NO |

**All four criteria fail. This is not marginal — the mechanism has no signal.**

---

## Evidence: Q-Loop B18 + B19

### B18 (7 iterations): TOKEN_CONTROL_VALID_BUT_NOT_MOONSHOT_YET
- Established the 7-level evidence hierarchy (INVALID -> FAIL_CAPACITY -> ORDINARY -> MARGINAL -> PASS -> STRONG -> MOONSHOT_CANDIDATE)
- Designed the 10-arm baseline board
- Set the minimum continuation bar: >=3pp over ALL strict baselines, >=2/3 benchmarks
- Ruled that SmolLM2-135M = "terminal token-level Eklavya protocol control," NOT Sutra proof

### B19 (7 iterations): PIVOT_PLAYBOOK_READY_CTI_FIRST_IF_EKLAVYA_FAILS
- Pre-designed pivot playbook before results arrived
- Scored 5 pivot directions from CLAUDE.md Nobel-track list:
  - **CTI (Compute Thermodynamics of Intelligence): 21/25** — #1
  - Renormalization: 19/25 — #2
  - CDMD: 18/25 — #3
  - CWC: 15/25 — #4
  - ENI: 15/25 — #5
- Identified salvageable assets: dual-loop methodology, evidence hierarchy, kill discipline, baseline board
- Identified sunk costs: Eklavya identity, FMD repair shot, byte-native mainline

---

## Supervisor Assessment

### What Codex got right
- The harness design was excellent: 5 conditions, oracle ceiling, random control, precommitted verdicts, teacher cache, bootstrap CIs. This is model experiment design.
- The evidence hierarchy from B18 is a genuine methodological contribution.
- The pivot playbook from B19 was ready BEFORE the results — disciplined contingency planning.
- Kill discipline is impeccable: 9 honest kills, each with clear evidence and precommitted criteria.

### What slipped past Codex's own skepticism
- The MARGINAL_EKLAVYA verdict is generous. The honest read is FAIL because:
  - The Eklavya-specific residual (routing minus single-teacher) is NEGATIVE
  - The oracle ceiling FAILS — no implementation can succeed
  - Random beats learned routing
- The n=48 per benchmark is acknowledged in limitations but the PATTERN is unmistakable

### The narrative gate

**Gossip-magazine headline**: "After 9 failed experiments, researchers prove that one good teacher beats two arguing ones — and pivot to something bigger."

**Survives "isn't that obvious?"**: The kill count and methodology survive. The result itself is publishable as negative results. The dual-loop methodology is the real artifact.

**Survives "that's trivial?"**: Yes — 9 precommitted kills with oracle ceilings and random controls is not trivial. The methodology is rigorous.

**But the project has no positive result.** The narrative is "impressive methodology, no breakthrough." This is exactly what B18 I125 predicted: "rigorous negative-results repo." That's not a moonshot.

---

## Kill Record (final for Eklavya arc)

| # | What | When | Evidence |
|---|------|------|----------|
| 1-6 | Byte-native objective variants | B7-B11 | No BPT improvement over baselines |
| 7 | MarginStudent scaffold | B12 | Can't learn supervised labels |
| 8 | S0/Wide7 byte capacity | B13 | Memorizes train, held-out flat |
| **9** | **Eklavya routing mechanism** | **B14** | **No residual over single-teacher; oracle ceiling fails; random beats learned** |

---

## Decision: MOONSHOT PIVOT

### What we're pivoting FROM
- Eklavya multi-teacher disagreement routing as the core mechanism
- Byte-native architecture as mainline
- Custom small models (S0, Wide7, MarginStudent) as students
- The entire Eklavya->Sutra pipeline

### What we're pivoting TO
Per Q-Loop B19's playbook: **CTI (Compute Thermodynamics of Intelligence)**
- Universal compute-distortion law: D(C) = D_inf + k*C^(-alpha)
- THE manifesto in equation form: Intelligence = Geometry, not Scale
- Demonstrable on a single RTX 5090
- CLAUDE.md rates it: Success=25%, Impact=9/10

### What we salvage
1. **Dual-loop methodology** — the process works; the direction was wrong
2. **Evidence hierarchy** (7 levels) — reusable for ANY research direction
3. **Kill discipline** (precommitted verdict tokens) — proven rigorous
4. **Baseline board pattern** (10-arm) — reusable
5. **Teacher cache infrastructure** — reusable if KD appears in CTI
6. **SmolLM2-135M/360M as calibrated test beds** — known quantities

### What is sunk cost (do NOT carry forward)
1. Eklavya identity / branding
2. FMD repair shot (never consumed, now irrelevant)
3. Byte-native mainline assumption
4. S0/Wide7 custom architectures
5. MarginStudent scaffold
6. All coordinate-inheritance code

---

## Launch Orders

### Immediate
1. Commit all B14 results and this check-in
2. Update MEMORY.md with terminal verdict
3. Begin CTI research phase: internet survey of compute scaling laws, Kaplan/Chinchilla/broken neural scaling, grokking-as-phase-transition literature

### Next Dual-Loop Cycle
- **Q-Loop B20**: CTI Direction — What exactly is CTI? What's the one experiment? What's the headline?
- **W-Loop B15**: CTI Preliminary — First compute-distortion measurement on SmolLM2 family

### Before any CTI implementation
- Codex design gate: CTI experiment plan must pass Codex review
- Evidence hierarchy: Define CTI-specific verdict tokens
- Baseline board: Define what "no signal" looks like for CTI

---

## Probability Update

| Claim | Check-in #12 est. | Actual |
|-------|-------------------|--------|
| Token-level SmolLM2 shows Eklavya residual over label-only | 20-35% | ~50% (routing beats label-only by ~2pp, but not significant at n=48) |
| Token-level SmolLM2 shows Eklavya residual over single-teacher | (not separately estimated) | **0% — routing LOSES to single-teacher** |
| Eklavya protocol survives terminal test | ~20% | **FAIL** |

| Forward claim | Estimate |
|---------------|----------|
| CTI produces a measurable, replicable compute-distortion law on SmolLM2 family | 40-50% |
| CTI produces a PARADIGM-SHIFTING result worthy of moonshot narrative | 15-20% |
| The dual-loop methodology is publishable as a research methodology paper | 70% |
