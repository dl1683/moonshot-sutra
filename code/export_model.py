"""Export a trained ModernBERT student to sentence-transformers format.

Takes a checkpoint directory with student weights and creates a model that
can be loaded via SentenceTransformer() and pushed to HuggingFace.

Usage:
  python code/export_model.py \
    --checkpoint outputs/E2/best_arm/checkpoint.pt \
    --base_model answerdotai/ModernBERT-base \
    --dim 384 \
    --out_dir outputs/eklavya-embed-v1

  # Then push to HuggingFace:
  python -c "from sentence_transformers import SentenceTransformer; \
    m = SentenceTransformer('outputs/eklavya-embed-v1'); \
    m.push_to_hub('iqidis/eklavya-embed-v1')"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def build_sentence_transformer(
    base_model: str = "answerdotai/ModernBERT-base",
    dim: int = 384,
    projection_weights: dict | None = None,
    encoder_path: str | None = None,
):
    """Build a SentenceTransformer model from components.

    If encoder_path is provided, loads trained encoder weights from that
    directory (saved via save_pretrained). Otherwise uses the base model.
    """
    from sentence_transformers import SentenceTransformer, models

    model_source = encoder_path if encoder_path and os.path.isdir(encoder_path) else base_model
    transformer = models.Transformer(model_source, max_seq_length=512)
    hidden_dim = transformer.get_word_embedding_dimension()

    pooling = models.Pooling(
        hidden_dim,
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )

    dense = models.Dense(
        in_features=hidden_dim,
        out_features=dim,
        bias=True,
        activation_function=nn.Identity(),
    )

    if projection_weights is not None:
        dense.linear.weight.data = projection_weights["weight"]
        dense.linear.bias.data = projection_weights["bias"]

    normalize = models.Normalize()

    model = SentenceTransformer(modules=[transformer, pooling, dense, normalize])
    return model


def load_checkpoint_weights(checkpoint_path: str, device: str = "cpu") -> tuple[dict | None, str | None]:
    """Load projection weights and encoder path from a training checkpoint.

    Handles two formats:
    1. Ship mode directory: encoder/ subdir + proj.pt file
    2. Raw state_dict .pt file: extract proj.weight/proj.bias

    Returns (projection_weights_dict, encoder_path_or_None).
    """
    encoder_path = None

    if os.path.isdir(checkpoint_path):
        encoder_dir = os.path.join(checkpoint_path, "encoder")
        if os.path.isdir(encoder_dir):
            encoder_path = encoder_dir
        proj_pt = os.path.join(checkpoint_path, "proj.pt")
        model_pt = os.path.join(checkpoint_path, "model.pt")
        if os.path.exists(proj_pt):
            proj = torch.load(proj_pt, map_location=device, weights_only=True)
            return proj, encoder_path
        if os.path.exists(model_pt):
            checkpoint_path = model_pt
        else:
            print(f"Warning: no model.pt or proj.pt found in {checkpoint_path}")
            return None, encoder_path

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    if "proj.weight" in state:
        return {"weight": state["proj.weight"], "bias": state["proj.bias"]}, encoder_path
    if "model_state_dict" in state:
        proj_state = {
            k.replace("proj.", ""): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("proj.")
        }
        if proj_state:
            return proj_state, encoder_path
    for key in state:
        if "proj" in key and "weight" in key:
            bias_key = key.replace("weight", "bias")
            if bias_key in state:
                return {"weight": state[key], "bias": state[bias_key]}, encoder_path

    print("Warning: could not find projection weights in checkpoint")
    return None, encoder_path


def export_model(
    checkpoint_path: str | None,
    base_model: str,
    dim: int,
    out_dir: str,
    model_card_extra: dict | None = None,
):
    """Export trained model to sentence-transformers format."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    proj_weights = None
    encoder_path = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        proj_weights, encoder_path = load_checkpoint_weights(checkpoint_path)
        if encoder_path:
            print(f"  Trained encoder found: {encoder_path}")

    print(f"Building SentenceTransformer ({encoder_path or base_model}, dim={dim})")
    model = build_sentence_transformer(base_model, dim, proj_weights, encoder_path)

    model.save(out_dir)
    print(f"Saved to {out_dir}")

    write_model_card(out_dir, base_model, dim, checkpoint_path, model_card_extra)

    test_model(out_dir)


def write_model_card(out_dir: str, base_model: str, dim: int,
                     checkpoint_path: str | None, extra: dict | None):
    """Write a model card for HuggingFace."""
    teachers = extra.get("teachers", []) if extra else []
    training_method = extra.get("method", "multi-teacher knowledge distillation") if extra else "knowledge distillation"
    metrics = extra.get("metrics", {}) if extra else {}

    teacher_section = ""
    if teachers:
        teacher_lines = "\n".join(f"- {t}" for t in teachers)
        teacher_section = f"""
## Training

**Teachers used:**
{teacher_lines}

**Method:** {training_method}

Teacher models are NOT used at inference time. All knowledge is absorbed
into the student's parameters.
"""

    metrics_section = ""
    if metrics:
        rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
        metrics_section = f"""
## Performance

| Metric | Score |
|--------|-------|
{rows}
"""

    card = f"""---
language: en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
  - sentence-transformers
  - feature-extraction
  - sentence-similarity
  - knowledge-distillation
---

# Sutra Embedding Model

A {dim}-dimensional sentence embedding model based on {base_model},
trained via knowledge distillation.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("iqidis/sutra-embed-v0")
embeddings = model.encode(["Hello world", "How are you?"])
print(embeddings.shape)  # (2, {dim})
```
{teacher_section}
{metrics_section}
## Limitations

- English-only (inherits from base model)
- Max sequence length: 512 tokens
- Performance may vary on domain-specific text not well represented
  in the training data

## Citation

```
@misc{{sutra2026,
  title={{Sutra Embedding Models}},
  author={{Devansh}},
  year={{2026}},
  url={{https://github.com/iqidis}}
}}
```
"""

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("Model card written")


def test_model(model_dir: str):
    """Quick test that the exported model works."""
    from sentence_transformers import SentenceTransformer

    print("\nTesting exported model...")
    model = SentenceTransformer(model_dir)

    test_sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The cat sat on the mat.",
    ]

    embeddings = model.encode(test_sentences)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Norm check: {(embeddings**2).sum(axis=1)}")

    from numpy import dot
    sim_01 = dot(embeddings[0], embeddings[1])
    sim_02 = dot(embeddings[0], embeddings[2])
    print(f"  Sim(ML, DL): {sim_01:.4f}")
    print(f"  Sim(ML, cat): {sim_02:.4f}")

    if sim_01 > sim_02:
        print("  Sanity check PASSED (related sentences more similar)")
    else:
        print("  Sanity check WARNING (unrelated sentence ranked higher)")


def main():
    parser = argparse.ArgumentParser(description="Export Eklavya model to sentence-transformers")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--base_model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--out_dir", default="outputs/eklavya-embed-v1")
    parser.add_argument("--teachers", nargs="*", default=[])
    parser.add_argument("--method", default="multi-teacher knowledge distillation")
    args = parser.parse_args()

    extra = {
        "teachers": args.teachers,
        "method": args.method,
    }

    export_model(args.checkpoint, args.base_model, args.dim, args.out_dir, extra)


if __name__ == "__main__":
    main()
