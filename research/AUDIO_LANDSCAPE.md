# Audio Encoder Landscape (September 2026)

## Benchmark: MAEB (Massive Audio Embedding Benchmark)

MAEB evaluates 50+ models across 30 tasks spanning speech, music, and
environmental sounds in 100+ languages. Integrated into the MTEB ecosystem.
Key finding: **no single model dominates all tasks**. Contrastive audio-text
models (CLAP variants) excel at environmental sound but fail multilingual
speech; speech-pretrained models show the opposite pattern.

Top overall: LCO-Embedding-Omni-7B (52.2% avg), Qwen2-Audio-7B (2nd overall,
1st on audio-only tasks). These are too large for Eklavya's target.

Other benchmarks: AudioSet (sound events), ESC-50 (environmental sounds),
HEAR (speech/music/environment), X-ARES (cross-domain).

## Small Audio Encoders (<500M params)

| Model | Params | Type | Strengths | License |
|-------|--------|------|-----------|---------|
| BEATs-base | 90M | Iterative audio tokenizer + encoder | Strong on AudioSet; acoustic tokenizers as self-supervised targets | MIT |
| OpenBEATs-base | 90M | BEATs + open data | Matches BEATs, fully open-source, reproducible | Apache-2.0 |
| OpenBEATs-Large | 300M | Scaled BEATs (ViT-Large) | SOTA on 6 bioacoustics + 2 env sound datasets; beats 1B+ models at 1/4 size | Apache-2.0 |
| CLAP (LAION) | 158M | Contrastive audio-text (HTSAT-tiny) | Zero-shot classification, audio retrieval | Apache-2.0 |
| ParaCLAP | 276M | Dual-encoder CLAP | Stronger than original CLAP | Open |
| AudioMAE | ~86M | Masked autoencoder on spectrograms | Strong general representation, self-supervised | Apache-2.0 |
| EAT | ~86M | Efficient Audio Transformer | Self-supervised, spectrogram patches | Open |
| AST | 87M | Audio Spectrogram Transformer | ImageNet-pretrained ViT adapted to audio | Apache-2.0 |
| Whisper-small encoder | 244M | ASR encoder (repurposed) | Strong speech features, can be repurposed for audio understanding | MIT |

## Large Audio Encoders (potential teachers)

| Model | Params | Type | Notes |
|-------|--------|------|-------|
| Dasheng-1.2B | 1.2B | Masked autoencoder, 272K hrs | SOTA on HEAR benchmark, trained on diverse audio |
| Dasheng-0.6B | 600M | Smaller Dasheng variant | Still very strong |
| Qwen2-Audio-7B | 7B | Audio-language model | #1 on MAEB audio-only tasks, reranking 80.8% |
| GLAP | 855M | Scaled CLAP | Stronger contrastive audio-text |

## How Audio Distillation Works Today

Standard approaches:
1. **Feature-level KD**: student matches teacher's intermediate representations
   (temporal knowledge distillation — distill attention patterns from
   transformers into lightweight CNNs for on-device deployment)
2. **Cross-model KD (CMKD)**: CNN/Transformer cross-architecture transfer
   (e.g., CNN student learns from Transformer teacher or vice versa)
3. **Data-free KD**: generate synthetic spectrograms via model inversion when
   training data is unavailable (Feature-Rich Audio Model Inversion)
4. **Self-supervised + KD**: combine masked autoencoder pretraining with
   teacher distillation signal
5. **Multi-representation KD**: transfer multiple levels of abstraction
   simultaneously (low-level acoustic + high-level semantic)

Frontier work (2025-2026):
- Edge-optimized distillation: 86M teacher -> 0.26M student for voice control
- Ensemble-guided distillation for acoustic scene classification on edge
- ICME 2025 Audio Encoder Challenge: OpenBEATs won with scaled BEATs

## Eklavya Angle for Audio

**The gap**: Audio distillation today is single-teacher, single-objective
(feature matching or logit KD). Nobody extracts multi-teacher behavioral
invariants across audio domains.

**The opportunity**: Audio models have even more heterogeneous geometries than
text embeddings. A BEATs model (iterative tokenizer), a CLAP model
(contrastive audio-text), and a Dasheng model (masked autoencoder) learn
fundamentally different representations. Their agreement under controlled
perturbations (time-shift, pitch-shift, noise injection, speed change) would
reveal domain-agnostic acoustic invariants.

**Audio-specific probes**:
- Time shift (phase invariance)
- Pitch shift (+/- semitones)
- Speed change (time stretch without pitch change)
- Noise injection (SNR levels)
- Channel reduction (stereo to mono)
- Frequency masking (band-pass)
- Reverb/echo addition

These are the audio analogues of text paraphrase/negation probes — controlled
perturbations that reveal what the teacher considers invariant.

**Target artifact**: A sub-100M audio encoder that matches 300M+ models on
MAEB by stealing invariants from heterogeneous teachers.

## Teacher Candidates for RTX 5090 (24GB)

| Teacher | Params | Geometry | Fits in 24GB | License |
|---------|--------|----------|-------------|---------|
| OpenBEATs-Large | 300M | Iterative tokenizer | Yes (easily) | Apache-2.0 |
| Dasheng-0.6B | 600M | Masked autoencoder | Yes | Apache-2.0 |
| CLAP (LAION, 158M) | 158M | Contrastive audio-text | Yes | Apache-2.0 |
| ParaCLAP | 276M | Contrastive (stronger) | Yes | Open |

All four fit comfortably on 24GB for sequential inference. Teacher diversity:
iterative tokenizer vs masked autoencoder vs contrastive audio-text — three
genuinely different learning paradigms.

**Student candidate**: BEATs-base (90M) or AudioMAE (~86M) — small enough to
iterate fast, large enough to have capacity for stolen knowledge.
