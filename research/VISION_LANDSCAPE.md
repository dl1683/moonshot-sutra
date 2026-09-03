# Vision Model Landscape for Eklavya

**Status:** Scouting (2026-09-03)
**Target:** Small vision encoders (<500M params) via Eklavya teacher tomography

## Small Vision Encoders (<500M params)

| Model | Params | Architecture | ImageNet LP | Training | License |
|---|---|---|---|---|---|
| DINOv2 ViT-S/14 | 21M | ViT-S, self-supervised | ~81% | Self-distillation from ViT-g | Apache 2.0 |
| DINOv2 ViT-B/14 | 86M | ViT-B, self-supervised | ~84% | Self-distillation from ViT-g | Apache 2.0 |
| DINOv3 ViT-S/16 | 21M | ViT-S, self-supervised | 81.4% | Self-distillation | Apache 2.0 |
| DINOv3 ViT-B/16 | 86M | ViT-B, self-supervised | 85.0% | Self-distillation | Apache 2.0 |
| DINOv3 ViT-L/16 | 300M | ViT-L, self-supervised | 87.4% | Self-distillation | Apache 2.0 |
| SigLIP 2 ViT-B/16 | 86M | ViT-B, contrastive | Strong | Contrastive + captioning + self-distillation | Apache 2.0 |
| SigLIP 2 ViT-L/16 | 303M | ViT-L, contrastive | Strong | Contrastive + captioning + self-distillation | Apache 2.0 |
| SigLIP 2 So400m/14 | 400M | Custom, contrastive | Strong | Contrastive + captioning + self-distillation | Apache 2.0 |
| TinyViT-5M/11M/21M | 5-21M | Compact ViT | 79-84% | Fast pretraining distillation from large ViT | MIT |
| InternViT-300M | 300M | ViT, supervised | Strong | Part of InternVL2 | Apache 2.0 |
| MobileViT v2 | ~5-10M | Mobile-optimized | ~78% | Supervised + KD | MIT |

## Key Benchmarks

1. **ImageNet-1K linear probe** — the standard for vision backbone quality
2. **ImageNet-1K zero-shot** — for CLIP/SigLIP-style models
3. **MIEB** (Massive Image Embedding Benchmark) — 130 tasks across 38 languages,
   8 categories; the MTEB of vision. Published Apr 2025, ICCV 2025.
4. **Dense prediction** — segmentation, depth, surface normals (ADE20K, NYUv2)
5. **Transfer learning** — fine-tuning on downstream classification tasks

## How Vision Distillation Works Today

Standard approaches (what Eklavya must beat):

1. **Self-distillation (DINO/DINOv2/v3):** Student and teacher are the same
   architecture. Teacher is EMA of student. Both see augmented views. Student
   matches teacher's token distributions via cross-entropy. Smaller models
   distilled from the giant model post-training.

2. **Logit distillation (TinyViT):** Large teacher generates dense logits on
   training data, sparsified and stored on disk. Student trained to match
   these logits. Fast and scalable. 2-4pp gains over non-distilled.

3. **Feature distillation (ViTKD):** Student matches teacher's intermediate
   features. Shallow layers mimicked directly, deep layers generated. Both
   shallow and deep layers matter (unlike CNNs where deep > shallow).

4. **Contrastive distillation (SigLIP 2):** Combined training with contrastive
   (image-text), captioning, self-distillation, and masked prediction losses.
   Multi-objective training produces stronger features.

## Where Eklavya Could Add Value

The gap: all current methods transfer either logits, features, or EMA-averaged
parameters. Nobody transfers *behavioral invariants under controlled visual
perturbations*.

**Vision probes (the Eklavya angle):**
- **Geometric:** rotation, flip, crop, scale — teacher ranking should be
  invariant to viewpoint changes
- **Color/texture:** grayscale, color jitter, style transfer — which aspects
  of appearance does the teacher rely on?
- **Occlusion:** mask patches, random erasing — how robust is teacher's
  ranking to missing information?
- **Semantic:** swap objects, change backgrounds — what semantic features
  drive teacher's similarity?
- **Resolution:** downsample, upsample — does teacher preserve ranking at
  low resolution?

The measurement object: given image q, gallery D = {d_1..d_k}, probes g_k(q):
```
signature_T(q, D, G) = {g_j: [sim_T(g_j(q), d_i) for d_i in D] for g_j in G}
```
Same as embedding tomography but with visual transforms instead of text probes.

**Strongest boring explanation:** The probes are just standard data
augmentation. If a student trained with the same augmented views (without
teacher signals) matches the tomography student, the teacher's behavioral
pattern was cosmetic.

## Teacher Candidates (RTX 5090, 24GB)

| Teacher | Params | Geometry | Why heterogeneous |
|---|---|---|---|
| DINOv3 ViT-L/16 | 300M | Pure vision, self-supervised | No language bias, patch-level features |
| SigLIP 2 So400m/14 | 400M | Vision-language contrastive | Language-grounded semantics |
| DINOv3 ViT-H+ | 840M | Self-supervised giant | Highest-quality vision features (fits in 24GB for inference) |

All loadable on RTX 5090 for inference (one at a time for 840M).
Student target: DINOv3 ViT-S (21M) or ViT-B (86M).

## Comparison to Embedding Landscape

| Dimension | Text Embeddings | Vision Encoders |
|---|---|---|
| Standard distillation | Logit/feature KD | Self-distillation (DINO), logit KD (TinyViT) |
| Multi-teacher | Rare (Jina v5 = single-teacher) | Not standard |
| Probe-based | Nobody | Nobody |
| Smallest competitive | ~22M (MiniLM-L6) | ~21M (DINOv2/v3 ViT-S) |
| Gold benchmark | MTEB | MIEB + ImageNet LP |
| Matryoshka | Common | Not standard |

## Next Steps (when embedding E1 completes)

1. If embedding tomography shows signal: adapt probes to visual domain
2. Student: DINOv3 ViT-S (21M) — smallest, most room to learn
3. Teachers: DINOv3 ViT-L + SigLIP 2 So400m (heterogeneous geometries)
4. Eval: MIEB subset + ImageNet linear probe
5. Control: standard self-distillation with same augmentations
