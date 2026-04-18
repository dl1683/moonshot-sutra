# Dynamical Systems & Control Theory Mental Models for Knowledge Transfer

*A portfolio of ALIVE frameworks for when classical KD plateaus.*
*Context: Sutra-Dyad, 188M byte-level. Classical KD = "match teacher logits" has stalled.*
*Written for: a reader 3 months from now who picks ONE and runs with it.*

---

## TL;DR — The most absent control-theoretic insight in ML-KD

**Observability.** Classical KD assumes that if the student matches the teacher's outputs
well enough, the student's *internal state* must also be close to the teacher's. This is
an unverified observability claim. In control theory, observability is a *property of the
pair (dynamics, observation map)* that you check before you bother estimating. Most
teacher-student pairs used in KD are provably **unobservable**: the teacher's output
manifold is a low-dimensional projection that leaves entire subspaces of the student's
state uncorrected by the loss. Training bangs against the observable subspace while
the unobservable modes drift freely — which is exactly what a plateau looks like.

Before the next KD run, ask: *what is the observability Gramian of my distillation
setup?* If it's rank-deficient, no amount of data fixes it — you need either additional
observation channels (hidden-state alignment, multi-teacher, auxiliary losses) or a
reparameterized student whose unobservable modes collapse to a point.

**Runner-up**: Model Predictive Control. Next-token KD is open-loop, horizon-1. An MPC
student would plan N bytes ahead, compare its planned trajectory to the teacher's, and
re-plan — closing a feedback loop that current KD doesn't even have.

---

## Model 1 — Observability & the Distillation Gramian

**The model in one sentence.** A linear system (A, C) is observable iff the observability
Gramian has full rank; unobservable states are invisible to *any* output-based estimator,
no matter how clever or how much data you throw at it.

**ML translation.** Model the student as a dynamical system with state s_t (hidden
activations), dynamics s_{t+1} = f_\theta(s_t, x_t), and observation y_t = g_\theta(s_t)
(logits). KD supervises y_t only. The *distillation Gramian* W_O = \sum_k (\partial y /
\partial s)^T (\partial y / \partial s) evaluated along training trajectories tells you
which directions in state space the loss can see. Directions in the nullspace of W_O
are ghosts — they can take any value and the loss never notices. If the teacher's useful
information lives partially in those ghost directions, you have a fundamental data-
efficiency ceiling that more teacher samples cannot break.

**Sutra thought experiment.** Pick a 10K-step Sutra-Dyad checkpoint. For a batch of
contexts, compute the Jacobian of the output logits w.r.t. the last-layer residual
stream. SVD it. Plot the singular value spectrum. If the spectrum has a sharp cliff at
rank R << d_model, then (d_model − R) dimensions of the student's state are currently
unobservable under the KD loss. Those dimensions are where "dead capacity" lives. Adding
a hidden-state matching loss (a second observation channel) should *provably* reduce
the nullspace — and the predicted magnitude of the improvement is computable from how
much teacher variance lives in the previously unobservable subspace.

**What it reframes about KD.** KD plateaus are not always about loss landscapes or
capacity — sometimes they are about **structural unidentifiability**. The cure is not
more data or more epochs; it is adding observation channels until the Gramian has full
rank over the directions that carry teacher information. This also explains *why*
multi-teacher KD sometimes works dramatically better than single-teacher: each teacher
adds an independent observation channel, and their Gramians *union* into a richer
observable subspace. Ekalavya is implicitly a Gramian-expansion protocol.

**Smallest testable prediction.** At a fixed Sutra checkpoint, measure the effective
rank of the output Jacobian. Then measure which residual-stream directions teacher
hidden states populate. The KD loss's improvement from adding hidden-state matching
should be proportional to the fraction of teacher variance in the *nullspace* of the
output Jacobian — not proportional to total teacher variance. If that proportionality
holds (even roughly), Gramian analysis is a diagnostic tool for designing KD losses.

---

## Model 2 — The Student as a Kalman Filter

**The model in one sentence.** A Kalman filter fuses a noisy model prediction with a
noisy measurement into an optimal posterior estimate; the Kalman gain weights them
inversely to their covariances.

**ML translation.** Reframe the student not as "a model being trained" but as *an
estimator* of the true next-byte distribution. At every step, the student has a prior
(its own prediction from its internal dynamics) and a measurement (the teacher's
prediction). The optimal fusion is not a fixed-weight convex combination — it's a
*state-dependent* Kalman gain K_t that depends on (a) the student's current uncertainty
about its own prediction, (b) the teacher's reliability on this specific input, and
(c) the covariance between the two. Classical KD uses a constant gain (alpha). Kalman-KD
would estimate K_t per token.

**Sutra thought experiment.** Implement a per-token Kalman gain in Sutra-Dyad. The
student maintains an uncertainty estimate (e.g., from dropout variance, last-layer
Jacobian, or a calibrated ensemble head). The teacher's reliability is estimated from
agreement among Ekalavya's multi-teacher ensemble. The KD loss weight becomes
K_t = P_student / (P_student + R_teacher) — high when the student is unsure *and* the
teachers agree, low when teachers disagree or the student is already confident. Easy
tokens (student already confident) get near-zero KD weight; hard tokens where teachers
speak with one voice get near-full teacher imitation; ambiguous tokens where teachers
disagree get partial weight scaled by *their* disagreement.

**What it reframes about KD.** Temperature-softmax-and-KL is a crude, global, single-
number approximation to what should be a per-token state-dependent fusion. Distillation
is *estimation* — and estimation without uncertainty is a dead mechanism. This also
gives a principled story for when to *stop* distilling: when the student's prior is
narrower than the teacher's posterior, the Kalman gain drives to zero automatically.
The teacher stops contributing exactly when it should.

**Smallest testable prediction.** At a matched compute budget, Kalman-gated KD should
beat constant-alpha KD by at least 0.03 BPB on held-out text, with the gain concentrated
on high-entropy tokens (where the gain matters most). Furthermore, the *learned*
distribution of Kalman gains across tokens should be bimodal — close to 0 for easy
bytes, close to 1 for rare/hard bytes — mirroring the uncertainty structure of natural
text. If gains are unimodal (flat), the estimator is not actually adapting, and the
Kalman framing buys nothing.

---

## Model 3 — MPC: The Student Plans, Then Distills the Plan

**The model in one sentence.** Model Predictive Control solves a finite-horizon optimal
control problem at every step, applies only the first action, then re-plans — trading
computation for closed-loop robustness to model error.

**ML translation.** Current next-token KD is horizon-1 open-loop. An MPC-KD student would
*roll out* its own predictions N bytes ahead, compare the full predicted trajectory to
the teacher's N-byte continuation, and only then take one step. The loss is on the
trajectory, not the point. This forces the student to internalize the teacher's
*dynamics* (how predictions evolve as context accumulates), not just its *marginals*.

Concretely: at position t, the student samples (or argmaxes) its own continuation
ŷ_{t+1:t+N}. It feeds this continuation back and computes its own logits at each step.
The teacher, on the *same* rollout ŷ, computes its logits. The KD loss is now
\sum_k KL(teacher(ŷ_{<t+k}) || student(ŷ_{<t+k})), summed over the horizon. Then a
single gradient step is taken on the position-t parameters, the rollout is discarded,
and the student moves to t+1.

**Sutra thought experiment.** Apply MPC-KD on Sutra-Dyad at N=8 bytes of horizon
(affordable at 188M). The key observation: the teacher grades the *student's own
trajectory* — so compounding errors (where the student drifts off-distribution) are
directly supervised. Current KD never sees the student's own errors propagate; MPC-KD
sees them and corrects them. Expect: stronger generation quality, reduced exposure
bias, and (critically) faster learning of long-range coherence than open-loop KD of
matched token count — because each MPC step is an N-step supervisory signal.

**What it reframes about KD.** Classical KD is the MPC horizon-1 special case. By
raising N, you smoothly interpolate between teacher-forcing KD (N=1) and full
trajectory imitation (N→∞, approaching behavioral cloning of rollouts). The optimal
horizon is a hyperparameter with a clean physical interpretation: *how far does the
teacher's useful signal project into the future along the student's own dynamics?*
Plateau in classical KD may mean: the useful horizon is larger than 1, and you are
simply not looking far enough.

**Smallest testable prediction.** MPC-KD with N=4 should reduce Sutra's BPB below
classical KD at matched token budget, AND should reduce off-distribution degradation
in long-form generation more than the BPB improvement alone predicts. If BPB improves
but long-form quality doesn't, the mechanism is wrong. If both improve together, MPC-KD
is genuinely closing the exposure-bias gap.

---

## Model 4 — Lyapunov Functions for Training Progress

**The model in one sentence.** A Lyapunov function V is a scalar "potential" that
decreases along trajectories of a dynamical system; its existence *proves* stability
without solving the system, and its level sets describe basins of attraction.

**ML translation.** Modern training has no Lyapunov function. Loss *usually* decreases
but is not monotone, oscillates across steps, and provides no guarantee of convergence
to any specific point. A distillation Lyapunov function V(student, teacher) would be a
scalar quantity — not necessarily the loss — that provably decreases along training
trajectories when the KD mechanism is working, and plateaus or diverges when it isn't.
V gives you a *stability certificate* and an early-warning system for pathological
runs, without needing 10K-step ablations.

**Sutra thought experiment.** Propose V = D_KL(teacher || student) + lambda ·
||hidden_teacher − align(hidden_student)||^2 − beta · H(student), where H is student
entropy. If the optimizer is a true Lyapunov descent on V, then dV/dt < 0 along
training. Measure dV/dt empirically at each step in Sutra-Dyad. When training plateaus,
one of three things will be true: (a) V has a local minimum at the plateau (mechanism
is correctly designed, student has converged to the Lyapunov-optimal point under
current constraints), (b) V is still decreasing but the loss isn't (loss is a bad
proxy for progress, keep training), or (c) V is not decreasing (Lyapunov violation —
the training dynamics are not what you think they are, and a structural fix is needed
before more compute).

**What it reframes about KD.** "Is training working?" becomes a *provable* question
instead of a vibes question. The Lyapunov function is the object that separates
"genuine plateau at the best-we-can-do point" from "pathological plateau from broken
mechanism" — a distinction Sutra has repeatedly failed to make, burning compute in
case (c) while thinking it was in case (a). A good V also constrains architecture
choices: mechanisms that don't admit a Lyapunov function are mechanisms you cannot
prove will ever converge.

**Smallest testable prediction.** Instrument Sutra with a candidate V. Plot V and
loss over training. When runs plateau (e.g., step 10K), V will either (a) also
plateau — then the plateau is real and more data/capacity is needed, not more steps;
or (b) keep decreasing — then loss is lying and there is hidden progress; or (c) have
been non-monotone all along — then the mechanism has a bug, and we can point to the
step where V first violated monotonicity. Any of these three outcomes is more actionable
than "loss flattened, unclear why."

---

## Model 5 — Adaptive Control: The Plant Is Non-Stationary

**The model in one sentence.** Classical control assumes a fixed plant and designs a
controller for it; adaptive control simultaneously identifies the plant's changing
parameters and controls it, with stability guaranteed only if identification is faster
than plant drift.

**ML translation.** In KD we assume the student (plant) is fixed-architecture with
only its weights as unknowns. But the *effective* dynamics of the student change
dramatically over training: early on the network is near-linear and noisy; mid-training
it develops sharp feature detectors; late training it's a tight nonlinear map. A single
KD recipe (fixed alpha, fixed temperature, fixed teacher mixture) is a single controller
applied to a drifting plant — guaranteed to be mis-tuned for most of training. Adaptive-
control-KD would continuously re-identify the student's current regime (via probes on
gradient norms, curvature, Jacobian rank, or teacher-student agreement) and re-tune the
KD hyperparameters in closed loop.

**Sutra thought experiment.** Add a lightweight "plant identifier" that runs every 500
steps: measure the student's output Jacobian rank, gradient noise scale, and the
fraction of tokens where student and teacher already agree within tolerance. Feed these
into a rule (or tiny neural controller) that sets temperature, alpha, and the teacher
mixture weights for the next 500 steps. Expect: early training uses high temperature
and low alpha (student needs to explore), mid-training lowers temperature and raises
alpha (exploit teachers hardest), late training lowers alpha again (student's own
signal is good, teachers add less). The *learned* schedule will likely look nothing
like the hand-designed cosine schedules currently used.

**What it reframes about KD.** KD hyperparameters are not constants — they are a
*control policy*. Every fixed-schedule KD run is implicitly assuming a plant model that
almost certainly doesn't match the true learning dynamics. The plateau may be partly
a control-tuning failure: the controller is stuck applying yesterday's recipe to a
plant that evolved last week.

**Smallest testable prediction.** A minimal adaptive scheme that adjusts only
temperature based on Jacobian rank (one knob, one measurement) should beat a fixed-
temperature baseline by >0.02 BPB on Sutra-Dyad at matched compute — and the learned
temperature schedule should be visibly non-monotonic, falsifying the "monotone cooling
is optimal" folk wisdom.

---

## Cross-Pollination — How these models talk to each other

- **Observability + Kalman.** The Kalman filter *is* the optimal estimator for an
  observable linear system. If the distillation Gramian is rank-deficient, Kalman-KD
  has a fundamental ceiling no matter how well gains are estimated — you must first
  add observation channels (Model 1) before fancier estimators (Model 2) pay off.
  Practical implication: run Gramian analysis BEFORE implementing Kalman-KD.

- **MPC + Lyapunov.** MPC stability proofs rely on terminal Lyapunov functions — the
  horizon-end cost must itself be a Lyapunov function for the closed-loop dynamics to
  guarantee the whole trajectory descends V. If Sutra adopts MPC-KD, the natural design
  for V is the terminal KD loss, and the horizon N is constrained by how far you can
  trust V to remain a valid descent direction. These two models are *one* model in
  disguise.

- **Adaptive control + everything else.** Every fixed hyperparameter in the other four
  models (N in MPC, lambda in Lyapunov, threshold in Gramian analysis, gain in Kalman)
  is a candidate for adaptive re-tuning. The adaptive-control frame says: never commit
  to a constant when a cheap online measurement can set it better.

- **Observability + physics RG (from 03).** An RG trajectory through teacher layers
  provides *multiple observation channels* — each layer is an additional output the
  student can be supervised on. RG-KD and Gramian analysis answer different questions
  about the same underlying fix: "how do we escape structural unidentifiability?"
  RG says *use the teacher's internal flow*; observability says *measure how much
  it actually helps and where*.

- **Kalman + info-theory (from 02).** The Kalman gain is the information-theoretic
  optimal fusion weight under Gaussian assumptions. Non-Gaussian generalizations
  (particle filters, variational filters) connect directly to the variational / rate-
  distortion framings in the info-theory portfolio. Same skeleton, richer distributions.

- **Lyapunov + biology (from 04).** Lyapunov functions are the mathematical formalism
  for homeostasis — a system actively defending a setpoint against perturbation. A
  biological "homeostatic KD" mechanism and a Lyapunov-certified KD are the same idea
  at different levels of description. Useful: biology inspires the *candidate* V;
  control theory gives the *proof* it works.

---

## Pick one. Ship it.

If you only have 2 weeks: **Model 1 (Observability Gramian)** — pure diagnostic, no
training changes, tells you whether your KD loss is even capable of succeeding. Worst
case, you learn something. Best case, you discover a structural reason for the plateau
that no amount of tuning would have fixed.

If you have 6 weeks: **Model 3 (MPC-KD)** — the biggest potential BPB win and the
cleanest story ("we closed the feedback loop KD never had"). Compounding-error
reduction alone could be a paper.

If you have 3 months and want the deepest reframing: **Model 4 (Lyapunov-certified
training)** — changes how you think about every future run, not just this one. Distinguishes
real plateaus from pathological ones. Pays dividends across all of Sutra's future work.
