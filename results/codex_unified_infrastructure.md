I wrote the unified theory into [research/RESEARCH.md#L5347](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5347). The core claim is:

`u(a | s) = E[D_future(s) - D_future(T_a(s))] / c(a)`

Sutra’s three threads are the same law at three timescales:
- token-time: choose stage, zoom, reroute, continue, or freeze by marginal future distortion reduction per cost
- training-time: choose which teacher supervises which token and which stage contract by the same gain/cost rule
- version-time: open new modules only when they add positive gain while preserving old behavior

Teacher absorption is therefore sparse and conditional, not global imitation. The right question is not “which teacher is best?” but “which teacher-stage pair reduces this token’s remaining distortion most cheaply?” I mapped that in [research/RESEARCH.md#L5451](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5451). A teacher is “fully absorbed” when its conditional marginal gain is effectively zero on held-out data and teacher dropout causes negligible regression.

Perpetual warm-start becomes an information-preservation law, not just a training convenience. The key invariants are in [research/RESEARCH.md#L5522](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5522): identity at zero gate, zero-influence introduction, gain-gated activation, and teacher-free consolidation. That makes version transitions gain-neutral by construction.

The full token lifecycle is in [research/RESEARCH.md#L5577](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5577): arrive, cheap coarse processing, gain-driven stage allocation, gain-driven memory zoom, verify residual gain, optional teacher absorption during training where residual gain is highest, then freeze when no action has enough positive utility.

My bottom-line answer is in [research/RESEARCH.md#L5634](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5634): Sutra is simultaneously a model, an infrastructure, and a learning framework, but the deepest description is “a modular intelligence infrastructure for monotone accumulation of useful predictive structure under a single rate-distortion control law.”

Score: this is the closest Sutra has come to the manifesto’s paradigm-shift bar. I scored the synthesis at 9.0-9.7 on unifying power / alignment / upside in [research/RESEARCH.md#L5693](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5693). The remaining gap is proof: marginal-gain targets, causal memory, and disciplined warm-start execution still need to work empirically.

I also attempted the required live Codex CLI design gate, but the local `codex` binary could not reach its backend and returned connection-refused errors, so this synthesis is grounded in the repo docs and prior Codex reviews rather than a fresh external Codex pass.