# Dual-Loop Supervisor Check-in #37

**Date: 2026-07-08**
**Reviewing: W-Loop B39 (methodology paper draft) + Q-Loop B47 (adversarial review)**

---

## 1. What Was Delivered

### W-Loop B39: The Paper

`research/methodology_paper.md` — 3205 lines. Title: "The Absorption Ladder:
How to Honestly Test AI Discovery Claims." Structure: 17 main sections + 8
appendices. All numbers pulled from frozen JSON measurement artifacts. Four
case studies with real metrics. Twenty-iteration draft log in Appendix F.

### Q-Loop B47: The Adversarial Review

`research/question_loop_batch47.md` — 536 lines. Fourteen iterations
(I407-I420), each attacking the previous one. Final verdict:
`Q_LOOP_B47_ADVERSARY_NOT_WON_OVER`.

## 2. Supervisor Assessment of B47's Attacks

B47 is the sharpest adversarial review this project has produced. Every
attack lands. Sorting by severity:

### Fatal (paper cannot be published without addressing)

1. **Core claim overclaims (I420).** The paper says "not credible until
   strongest boring explanations have failed." But a finite absorber roster
   cannot certify "strongest." The paper violates its own claim-ceiling
   discipline. Fix: roster-relative claim, not universal claim.

2. **No positive control (I408).** The ladder has only been seen saying no.
   A methodology that can only reject is a rejection machine, not an
   evaluation method. Fix: constructed positive-control domain or honest
   demotion to "protocol proposal."

3. **Self-reference loop (I415).** The anti-self-deception method is
   validated by self-audit. Fix: at minimum, acknowledge the limit
   explicitly and constrain the claim ceiling accordingly.

### Serious (paper is weakened without addressing)

4. **Absorber completeness has no stopping rule (I409).** "Strongest" is
   unknowable. Fix: auditable roster standard with residual-risk statement.

5. **All-in cost not robustness-tested (I410).** The 0.967 cost ratio is a
   property of serialization conventions, not a robust scientific result.
   Fix: sensitivity analysis under alternative encodings.

6. **Scope exceeds evidence (I412).** Four synthetic CPU-only cases cannot
   support broad AI-discovery framing. Fix: narrow scope or add external
   case.

7. **B38 scale illusion (I413).** 2^64 candidate space but only 256 test
   cases. Fix: sampling-validity argument.

8. **Appendices are padding (I416-I417).** Generated checklists damage
   credibility. Fix: compress or fill with actual evidence.

### Important (should address for quality)

9. **Not distinguished from good practice (I414).** Fix: related-work delta.
10. **Not actionable for external evaluators (I418).** Fix: domain
    onboarding recipe.
11. **No field-facing case with stakes (I419).** Fix: external reanalysis
    or scope narrowing.
12. **Native absorber status implicit (I411).** Fix: filled status ledgers.
13. **Internal negatives treated as validation (I407).** Fix: separate
    methodology-validation from case-study reporting.

## 3. Narrative Gate

The honest one-sentence narrative given ONLY what survived this checkpoint:

```
We built a rigorous methodology for testing AI discovery claims — it killed
every claim we tested across 13 directions — but the paper selling the
methodology overclaims its own epistemic authority and lacks a positive
control or external case.
```

- "Isn't that obvious?": The methodology itself is not obvious — precommit
  tokens, hidden-open discipline, native absorber theater, all-in cost
  accounting are non-trivial. But the PAPER's version overclaims.
- "So what?": The methodology IS useful. The paper needs revision, not
  abandonment.
- "That's trivial?": No — B47 confirmed the methodology is real. The
  attacks are about the PAPER's framing, not the methodology itself.

**Narrative verdict: ALIVE but the paper needs surgery, not cosmetics.**

## 4. Decision: Paper Revision Required

The hard clock said 2 batches remaining, and both are done. But per
Invariant #2, the loop does not stop until the adversary is won over. B47
is not won over.

### The Revision Plan

B47's minimum revision list is the right scope. Prioritized:

**Must-do (changes that affect the paper's core claim):**

1. Rewrite core claim: roster-relative, not universal. The paper is about
   making ordinary explanations executable within a declared roster, not
   about certifying that ALL ordinary explanations have been exhausted.

2. Add positive-control section: construct a domain where narrow signal
   survives the ladder. CPU-only, small, but demonstrating the ladder
   can say "signal relative to declared roster."

3. Confront self-audit limit: explicit section acknowledging the project
   self-grades, with a bound on what self-audit can claim.

4. Add absorber-completeness stopping rule: residual-risk framework with
   explicit "absorber omission lowers claim ceiling" protocol.

**Should-do (structural improvements):**

5. Cut/compress appendices F, G, H. Evidence or checklists, not padding.
6. Add filled absorber-status ledgers for each case study.
7. Add cost robustness section with sensitivity bands.
8. Add related-work delta: what makes this more than good practice.
9. Add domain onboarding recipe (8-step workflow).

**Defer (important but not blocking submission):**

10. External case study (the most demanding ask — requires domain expertise
    outside this project's scope). Address in limitations + future work.
11. B38 sampling-validity argument (can add, but secondary to claim fixes).

### Batch Allocation

**W-Loop B40: Paper revision (20 iterations)**

Codex takes the paper and B47's attacks and revises. Priority order:
items 1-4 (must-do), then 5-9 (should-do). The paper should shrink, not
grow — compression is the goal.

**Q-Loop B48: Final adversarial gate (14 iterations)**

Fresh adversarial review of the revised paper. This is the terminal gate.
If the adversary is won over, the moonshot is complete. If not, the loop
continues.

## 5. Supervisor Verdict

```
B47 ADVERSARY NOT WON OVER. PAPER NEEDS SURGERY.

THE METHODOLOGY IS REAL. THE PAPER OVERCLAIMS.

TOP 3 FIXES:
1. ROSTER-RELATIVE CLAIM (NOT UNIVERSAL)
2. POSITIVE CONTROL (LADDER CAN SAY YES)
3. CONFRONT SELF-AUDIT LIMIT

W-LOOP B40: REVISE PAPER (20 ITER)
Q-LOOP B48: TERMINAL ADVERSARIAL GATE (14 ITER)

THE DUAL-LOOP CONTINUES.
```
