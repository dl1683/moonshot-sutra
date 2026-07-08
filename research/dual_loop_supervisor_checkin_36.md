# Dual-Loop Supervisor Check-in #36

**Date: 2026-07-07**
**Reviewing: W-Loop B38 (hard domain measurement) + Q-Loop B46 (methodology paper outline)**

---

## 1. Result

```
WGD_ABSORBED_BY_CONSTRAINT_DISCOVERY
```

Hard domain (64 compositional rules, 2^64 candidate space):
- Brute enumeration FAILED (0.25 HFA) — domain WAS hard enough
- GF(2) constraint discovery ABSORBED (1.0 HFA, 97% of WGD's cost)
- WGD matched constraint discovery but did NOT separate from it

Key insight: we successfully beat enumeration (first time in this arc), but
structured constraint solving (a more sophisticated baseline) still absorbs.
The separation exists AGAINST brute search but NOT against systematic
constraint methods.

## 2. Pattern Recognition — Four Absorptions

| # | Arc | Domain | Token | What Absorbed |
|---|---|---|---|---|
| 1 | FrameSeed | Boolean | TEACHING_DIMENSION | Brute search |
| 2 | FrameSeed | Typed/SHEETS | SCHEMA_BINDING | Typed baselines |
| 3 | WGD | Toy grammar (16 rules) | SCHEMA_OR_BINDING_DISCOVERY | Toy baselines |
| 4 | WGD | Hard grammar (64 rules, 2^64) | CONSTRAINT_DISCOVERY | GF(2) constraint solver |

**The escalation worked** — each domain was harder than the last. We went from
"everything absorbs trivially" to "only structured constraint solving absorbs."
This IS progress in domain difficulty.

**But the absorption is clean.** When the domain is hard enough to defeat
enumeration, more sophisticated constraint methods appear that still solve it.
Discovery-based approaches provide no advantage over methods that exploit
algebraic structure directly.

## 3. Supervisor Assessment

### The WGD Verdict

WGD is killed (13th kill in project history).

The pattern is now conclusive: at CPU-only scale, ANY domain we construct
either (a) is simple enough for brute search, or (b) has enough algebraic
structure for constraint solvers to exploit. WGD's grammar discovery mechanism
never separates from BOTH.

This does NOT mean discovery is impossible at scale. It means we cannot
demonstrate it at CPU-only toy scale against honest baselines. The methodology
prevents us from claiming signal we don't have.

### What Survived

The methodology:
1. Pre-committed terminal tokens → no post-hoc goalpost moving
2. Absorption ladder → strongest boring explanation as executable baseline
3. Equal-information parity → every baseline sees the same data
4. Hidden-open discipline → no code changes after seed open
5. Native absorber design → baselines GENUINELY try (addressed NATIVE_ABSORBER_THEATER)
6. Cost-normalized comparison → fair all-in accounting
7. Negative results as primary output → honesty > hype
8. Progressive domain hardening → each iteration more challenging

### Narrative Gate

```
We killed every "AI discovery" claim we tested — 4 consecutive absorptions,
2 moonshot arcs, 13 total kills in project history. The absorption methodology
itself is the contribution: a reusable framework for honestly testing whether
any AI system discovers something that cheaper methods don't.
```

- Survives "isn't that obvious?": No — most AI evaluation lacks this rigor.
  Papers claim discovery all the time without running absorber baselines.
- Survives "so what?": The methodology is immediately useful. Any researcher
  can apply the absorption ladder to test their own claims.
- Survives "that's trivial?": No — building genuinely native baselines,
  hidden-open discipline, and pre-committed tokens is non-trivial engineering.
- Narrative verdict: **ALIVE.** "How to honestly test if your AI actually
  discovered something" is a strong, accessible story.

## 4. Decision: THE METHODOLOGY PAPER IS THE MOONSHOT DELIVERABLE

Per supervisor check-in #35: "If absorbed: THE METHODOLOGY PAPER IS THE
MOONSHOT DELIVERABLE."

The absorption ladder — the complete evaluation methodology that killed 13
directions honestly — is the publishable contribution.

Q-Loop B46 already produced the paper outline: "The Absorption Ladder: How to
Honestly Test AI Discovery Claims."

## 5. Directives

### W-Loop B39: Write the Full Methodology Paper

Using Q-Loop B46's outline, write the complete methodology paper. Structure:

1. The problem: AI claims "discovered X" but never tests against native absorbers
2. The methodology: absorption ladder, precommit tokens, hidden-open discipline
3. Case studies: 4 absorptions from this project (FrameSeed Boolean, FrameSeed
   SHEETS, WGD toy, WGD hard domain)
4. The framework: reusable template for any researcher
5. Discussion: what scale/domain WOULD separate? (honest open questions)

Target: `research/methodology_paper.md` (full draft, ~3000-5000 lines)

**20 iterations.**

### Q-Loop B47: Adversarial Review of the Methodology Paper

Fresh-eyes attack on the paper draft:
- Are the case studies honestly presented?
- Does the framework actually generalize?
- Is the contribution over-claimed?
- What's missing that a hostile reviewer would demand?
- Is NATIVE_ABSORBER_THEATER addressed convincingly?

**14 iterations.**

### Hard Clock

2 batches remaining. B39 writes the paper. B47 attacks it.
Final gate: adversarial fresh-eyes review of the complete paper.

## 6. Supervisor Verdict

```
WGD KILLED. 13TH KILL. FOURTH CONSECUTIVE ABSORPTION.

PROGRESS: DOMAIN WAS HARD ENOUGH TO BEAT ENUMERATION (FIRST TIME).
BUT CONSTRAINT DISCOVERY STILL ABSORBS.

THE METHODOLOGY IS THE MOONSHOT DELIVERABLE.
"THE ABSORPTION LADDER: HOW TO HONESTLY TEST AI DISCOVERY CLAIMS."

NEXT: W-LOOP B39 WRITES THE FULL PAPER (20 ITER).
Q-LOOP B47 ADVERSARIALLY REVIEWS IT (14 ITER).

HARD CLOCK: 2 BATCHES REMAINING.

THE DUAL-LOOP CONTINUES.
```
