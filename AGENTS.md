# AGENTS.md — Operating Process (local, git-ignored)

Process only. Mission and state live in `README.md`, `research/VISION.md`,
`research/STATUS.md`, `STATE.md`, and `experiments/EXPERIMENTS.md`. Read this
before any work. Authority: project memory → this file → global `CLAUDE.md` →
repo state docs → code.

## Mission (Devansh, 2026-09-04 — corrected framing)

**Eklavya is not a technique. Eklavya is the idea.**

There are billions of dollars of accumulated knowledge sitting in pretrained
models — language understanding, visual reasoning, semantic structure — that
gets thrown away every time someone builds a new system. Every new setup
requires starting from scratch. Eklavya asks: can we unlock that accumulated
knowledge and let new, smaller, cheaper models inherit it? Can AI development
stop being "start from scratch every time" and start being "build on what
already exists"?

If this works, a person with one GPU can benefit from the knowledge inside
models they could never afford to train. AI stops being something only the
richest labs can build and starts being something anyone can build on top of.
This is the "AI as electricity / AI as vaccines" thesis from the portfolio
mission.

**Eklavya = the framework for knowledge inheritance from pretrained models.**
Teacher tomography was one mechanism we tried — it failed (15 kills). Standard
KD is another — it works but is well-understood and incremental. The mission
survives every mechanism kill. The question is always: what is the RIGHT way
to transfer accumulated knowledge, not WHETHER it can be transferred.

**Sutra** is the current exploration ground: compact text/vision/audio embedding
models that inherit from larger teachers. Sangam covers natively multimodal;
Sutra pushes individual modality models.

The AGI Thesis ("The Intelligence Transition") is the theoretical foundation:
accumulated capability that survives model replacement, governed transitions,
evidence-bound persistence.

## Axiom (from LSR, user-directed)

Axiomatically assume what we are doing is possible. If something fails, use
the failure to launch the next iteration. Never stop working. Every kill
narrows the search and makes the next attempt sharper. The 14 kills in Sutra's
history are the narrowing, not defeat.

## The mechanism trap (formerly "KD trap")

The recurring failure mode: confusing *a specific transfer mechanism* with
*the Eklavya mission*. Tomography, output mimicry, averaged distributions,
per-teacher heads — these are mechanisms. They can and should be killed when
they fail. But killing a mechanism is NOT killing the mission. The mission
(unlock accumulated knowledge for inheritance) survives every mechanism kill.

Every Codex review MUST ask two questions:
1. "Is this mechanism genuinely transferring knowledge the student couldn't
   get on its own?" (mechanism test)
2. "If this mechanism is dead, what OTHER form of knowledge transfer should
   we try next?" (mission continuity)

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

The Eklavya narrative: "There's billions of dollars of knowledge locked in AI
models that nobody can reuse — every new model starts from scratch. We figured
out how to let a small model inherit what the big ones already know, so anyone
with a laptop can build on the world's best AI instead of starting over."

If the current mechanism doesn't serve this narrative, kill the mechanism and
find one that does. The narrative is the filter, not the technique.

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
