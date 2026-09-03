# AGENTS.md — Operating Process (local, git-ignored)

Process only. Mission and state live in `README.md`, `research/VISION.md`,
`research/STATUS.md`, `STATE.md`, and `experiments/EXPERIMENTS.md`. Read this
before any work. Authority: project memory → this file → global `CLAUDE.md` →
repo state docs → code.

## Mission (Devansh, 2026-09-03)

Eklavya is the method: extract owned invariants from existing models through
teacher tomography — behavior signatures under controlled probes, not output
mimicry. Sutra is the target artifact: small, efficient models across
embedding, vision, and audio modalities that win through better geometry, not
scale. Sangam already covers natively multimodal (video/audio/everything
together); Sutra pushes the individual modality models.

The AGI Thesis ("The Intelligence Transition") is the theoretical foundation:
accumulated capability that survives model replacement, governed transitions,
evidence-bound persistence.

## Axiom (from LSR, user-directed)

Axiomatically assume what we are doing is possible. If something fails, use
the failure to launch the next iteration. Never stop working. Every kill
narrows the search and makes the next attempt sharper. The 14 kills in Sutra's
history are the narrowing, not defeat.

## The R^n trap adapted: the KD trap

The recurring failure mode: every distillation approach so far stayed too
close to ordinary KD (mimic outputs, average teachers, supplied geometry).
Every Codex review, audit, and direction dialogue MUST ask: "Is the current
work building owned invariants in the student, or disguising teacher
dependence? If the latter, that IS the tunnel."

## Five Sacred Outcomes (from VISION.md — fixed points)

1. Genuine Intelligence — actually capable, not benchmark-shaped
2. Improvability — failures found, understood, repaired surgically
3. Democratized Development — reproducible, modifiable, extendable
4. Data Efficiency — learns more from less through better structure
5. Inference Efficiency — cheap to run, deployable widely

## Blackboard (mandatory)

Board `1d65d9fb` is the reasoning board for this program. Every session and
sub-agent: `bb_get_state` on `1d65d9fb` first; add findings with provenance;
`bb_synthesis` before concluding. Every Codex prompt includes: "Use the
blackboard MCP: bb_list first; record what you find; bb_synthesis before
concluding."

## Autonomy mandate

Always working; take the highest-leverage thing and do it. No "blocked on a
human" state — converge with Codex, decide, execute. The user steers by
giving a direction or saying stop. A closed line is a pivot, not a park.
Exception: GPU runs need explicit user approval.

## Think before you run

Before any run: state the expectation, what each outcome implies, and the
confound that would make the result meaningless. Before interpreting any
result, test the single simplest confound that could explain every row.

## Evidence gates (hard)

1. Retained gain after teacher removal is the soul test. No teacher at inference.
2. Control-adjusted gain vs matched baselines (CE-only, best single teacher,
   naive average, no-packet ablations).
3. Lifecycle cost accounting: teacher signature cost + training + inference +
   storage + validation + update.
4. The cheapest direct remedy is run as a control arm, not imagined.
5. Held-out tasks for headline numbers; model versions pinned.
6. Nothing external until gates pass with controls actually run.

## Narrative gate (gossip-magazine test)

Every direction needs a one-sentence "so what" a non-expert finds exciting.
A small embedding model that matches models 10x its size through better-shaped
lessons from many teachers — that's the David story. If the wow reduces to a
known trick, it's fake.

## Codex discipline

Codex = the real `codex` CLI, never simulated. Invoke:
`codex exec -s workspace-write --skip-git-repo-check -C "<dir>" -o "<out>" "<prompt>"`
Background runs need `< /dev/null`. Point Codex at files (it reads the repo).
Design gate before non-trivial work. Evidence gate before claims. PR gate
after each coherent block. Direction changes are 2-3 round dialogues.

## Cadence

Four session-scoped watchdogs from `agentic-setup`, re-installed every session:
20-min liveness · hourly ops/leverage · 2-hour adversarial audit + anti-tunnel
(Codex) · 2-hour entropy/consolidation sweep (separate sub-agent).

## Flywheel

Nothing closes without depositing its transferable residue into the sibling
project it fits. Eklavya findings → Sangam. Representation science → LSR.
Accumulation theory → AGI Thesis. Cross-project mechanisms are load-bearing
signals.

## Hardware

RTX 5090 laptop, 24 GB VRAM. Battery ~63% health; sustained GPU load can
hard-crash. Explicit user approval per GPU run; one job at a time; launch
detached with checkpoints; short bursts survive. Qwen3-0.6B for local
prototyping. Windows env: `PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8`.

## Hygiene and git

No new files by default; one canonical runner per artifact; config explores
variation. Delete aggressively. If it's not in ledger.jsonl and EXPERIMENTS.md,
it didn't happen. One idea per commit, message ends `Committed by Devansh`.
Never leak model names, providers, costs, or routing. Never `git add -A`.
AGENTS.md, internal/, .codex_* are never committed.

## Research & Context Directories

Codex and sub-agents read these directly:
- Domain research: `C:\Users\devan\OneDrive\Desktop\Projects\Market Reports\Open Exploration\`
- Portfolio meta: `C:\Users\devan\OneDrive\Desktop\Projects\_meta\`
- Project memory: `C:\Users\devan\.claude\projects\C--Users-devan-OneDrive-Desktop-Projects\memory\`
- Sangam (multimodal sibling): `C:\Users\devan\OneDrive\Desktop\Projects\sangam\`
- LSR (representation science): `C:\Users\devan\OneDrive\Desktop\Projects\Latent-Space-Reasoning\`
- AGI Thesis: `C:\Users\devan\OneDrive\Desktop\Projects\AGI Thesis\`
- Eklavya prototype: `C:\Users\devan\OneDrive\Desktop\Projects\AI Moonshots\Eklavya\`
