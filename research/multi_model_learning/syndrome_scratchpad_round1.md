The round-1 multi-slot acting design is superseded.

Canonical plan now lives in [research/RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md) under the revised LDPC syndrome probe section.

The stripped-down first test is:

- one shadow-only scalar `syndrome_energy` head
- fixed cached CPU batches at `dim=768`
- frozen recurrent core
- matched-parameter control with shuffled targets
- no acting read/write path
- no Stage 7 claim
- no proceed-gate coupling

Only metric:

- `Spearman(syndrome_energy, future_gain)` on hard late token-pass pairs

Kill rule:

- if `Spearman < 0.10`, kill the entire LDPC branch

This file remains only as a pointer so the round-2 review chain has a stable target.
