"""Eklavya Vision Tomography — teacher signature extraction for vision encoders.

Visual probes: controlled image transformations that reveal how teachers rank
images. The invariant is the ranking response surface under perturbation, just
as for text embeddings but with geometric/photometric transforms instead of
linguistic probes.

Probe families:
  - identity: original image
  - hflip: horizontal flip (should be invariant for most tasks)
  - crop: random crop + resize (spatial robustness)
  - color_jitter: brightness/contrast/saturation changes
  - grayscale: remove color (tests color vs structure dependence)
  - rotate: small rotation (tests rotation robustness)
  - blur: Gaussian blur (tests detail dependence)

Usage:
  python code/vision_tomography.py extract \
    --teachers facebook/dinov2-vits14 google/siglip2-base-patch16-224 \
    --data data/image_pairs.jsonl --out data/vision_signatures.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T


@dataclass(frozen=True)
class VisualProbe:
    probe_id: str
    transform: object  # torchvision transform


def get_probe_transforms(image_size: int = 224, seed: int = 0) -> list[VisualProbe]:
    """Standard set of visual probes."""
    return [
        VisualProbe("identity", T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("hflip", T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("crop", T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.5, 0.8)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("color_jitter", T.Compose([
            T.Resize((image_size, image_size)),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("grayscale", T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("rotate", T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        VisualProbe("blur", T.Compose([
            T.Resize((image_size, image_size)),
            T.GaussianBlur(kernel_size=7, sigma=(1.0, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
    ]


@dataclass
class VisionSignature:
    pair_id: str
    query_image_path: str
    candidate_image_paths: list[str]
    probes: list[str]  # probe_ids
    teacher_sigs: dict[str, dict[str, list[float]]]  # teacher -> probe_id -> [sim_scores]
    gold_idx: int = 0


def load_vision_encoder(model_name: str, device: str = "cpu"):
    """Load a vision encoder (DINOv2, SigLIP, etc.) for feature extraction."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor


@torch.no_grad()
def encode_image(model, processor, image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Encode a single image to a normalized embedding vector."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    # Use CLS token or pooled output depending on model
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output
    else:
        emb = outputs.last_hidden_state[:, 0]
    return F.normalize(emb, p=2, dim=-1)


@torch.no_grad()
def encode_images_batch(model, processor, images: list[Image.Image], device: str = "cpu") -> torch.Tensor:
    """Encode a batch of images."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        embs = outputs.pooler_output
    else:
        embs = outputs.last_hidden_state[:, 0]
    return F.normalize(embs, p=2, dim=-1)


@torch.no_grad()
def compute_vision_signature(
    model,
    processor,
    query_image: Image.Image,
    candidate_images: list[Image.Image],
    probes: list[VisualProbe],
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Compute teacher similarity scores under each visual probe."""
    # Encode candidates once (no transform variation on candidates)
    identity_probe = probes[0]  # assume first is identity
    cand_tensors = [identity_probe.transform(img) for img in candidate_images]
    cand_batch = torch.stack(cand_tensors).to(device)

    # For DINOv2-style models, process differently
    cand_embs = encode_images_batch(model, processor, candidate_images, device)

    sig = {}
    for probe in probes:
        probed_image = probe.transform(query_image)
        # Convert tensor back to PIL for processor
        probed_pil = T.ToPILImage()(probed_image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                                     + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1))
        q_emb = encode_image(model, processor, probed_pil, device)
        sims = (q_emb @ cand_embs.T).squeeze(0).cpu().tolist()
        sig[probe.probe_id] = sims

    return sig


def main():
    parser = argparse.ArgumentParser(description="Eklavya Vision Tomography")
    parser.add_argument("--teachers", nargs="+", default=["facebook/dinov2-vits14"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Vision tomography pipeline scaffolded.")
    print(f"Teachers: {args.teachers}")
    print("Requires image dataset (CIFAR-100, ImageNet subset, or custom).")
    print("Use --data with image pairs in JSONL format.")
    print()
    print("Next steps:")
    print("  1. Create image pair dataset (query image + candidate images)")
    print("  2. Extract teacher signatures with visual probes")
    print("  3. Train student vision encoder to match teacher response surfaces")
    print("  4. Evaluate on MIEB or ImageNet linear probe")


if __name__ == "__main__":
    main()
