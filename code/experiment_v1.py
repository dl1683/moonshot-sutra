"""Eklavya Experiment V1 — Vision embedding tomography vs standard KD.

Student: DINOv2-ViT-S/14 (21M, pretrained but not tuned for retrieval)
Teachers: DINOv2-ViT-B/14 (86M) + CLIP-ViT-B/32 (86M, heterogeneous objective)
Data: CIFAR-100 image retrieval pairs (class membership = relevance)
Probes: identity, hflip, crop, color_jitter, grayscale, rotate, blur

Arms:
  V1: Full tomography (multi-probe, multi-teacher KL on ranking distributions)
  B0: Contrastive-only baseline (InfoNCE, no teacher)
  B2: Standard single-teacher KD (identity probe only, best teacher)
  B3: Multi-teacher average KD (average teacher scores, identity probe only)

Usage:
  python code/experiment_v1.py --device cuda --steps 600 --out_dir outputs/V1_cifar100
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms as T
from PIL import Image


VISUAL_PROBES = [
    ("identity", T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("hflip", T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("crop", T.Compose([
        T.RandomResizedCrop(224, scale=(0.5, 0.8)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("color_jitter", T.Compose([
        T.Resize((224, 224)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("grayscale", T.Compose([
        T.Resize((224, 224)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("rotate", T.Compose([
        T.Resize((224, 224)),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
    ("blur", T.Compose([
        T.Resize((224, 224)),
        T.GaussianBlur(kernel_size=7, sigma=(1.0, 2.0)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])),
]

IDENTITY_TRANSFORM = VISUAL_PROBES[0][1]


def build_retrieval_pairs(
    dataset, n: int = 500, n_candidates: int = 8, seed: int = 42, split: str = "train"
) -> list[dict]:
    """Build image retrieval pairs from a classification dataset."""
    rng = random.Random(seed)

    by_class: dict[int, list[int]] = {}
    targets = dataset.targets if hasattr(dataset, "targets") else [t for _, t in dataset]
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    classes = sorted(by_class.keys())
    pairs = []
    for i in range(n):
        cls = classes[i % len(classes)]
        members = by_class[cls]
        if len(members) < 2:
            continue
        q_idx, pos_idx = rng.sample(members, 2)

        neg_classes = [c for c in classes if c != cls]
        neg_cls_sample = rng.sample(neg_classes, min(n_candidates - 1, len(neg_classes)))
        neg_indices = [rng.choice(by_class[c]) for c in neg_cls_sample]

        cand_indices = [pos_idx] + neg_indices
        rng.shuffle(cand_indices)
        gold_idx = cand_indices.index(pos_idx)

        pairs.append({
            "id": f"{split}_{i}",
            "query_idx": q_idx,
            "candidate_indices": cand_indices,
            "gold_idx": gold_idx,
            "query_class": cls,
        })

    return pairs


class VisionEncoder:
    """Generic wrapper for vision encoders (DINOv2, CLIP, etc.)."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoModel, AutoImageProcessor

        self.device = device
        self.name = model_name

        if "clip" in model_name.lower():
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.mode = "clip"
        else:
            self.model = AutoModel.from_pretrained(model_name).to(device).eval()
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.mode = "generic"

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> torch.Tensor:
        if self.mode == "clip":
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            embs = self.model.get_image_features(**inputs)
        else:
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                embs = outputs.pooler_output
            else:
                embs = outputs.last_hidden_state[:, 0]
        return F.normalize(embs, p=2, dim=-1)


class VisionStudent(nn.Module):
    """Small vision encoder with trainable projection for retrieval."""

    def __init__(self, model_name: str = "facebook/dinov2-vits14", dim: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, dim)
        self.dim = dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=pixel_values)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]
        projected = self.proj(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(next(self.encoder.parameters()).device)
        return self.forward(pixel_values)

    def encode_tensors(self, tensors: torch.Tensor) -> torch.Tensor:
        return self.forward(tensors.to(next(self.encoder.parameters()).device))


@torch.no_grad()
def extract_vision_signatures(
    teachers: dict[str, VisionEncoder],
    dataset,
    pairs: list[dict],
    probes: list[tuple[str, T.Compose]],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Extract teacher signatures for all pairs under all probes."""
    sigs = {}
    total = len(pairs)

    for pi, pair in enumerate(pairs):
        if (pi + 1) % 50 == 0:
            print(f"    Signatures: {pi + 1}/{total}")

        query_img = dataset[pair["query_idx"]][0]
        if not isinstance(query_img, Image.Image):
            query_img = T.ToPILImage()(query_img)

        cand_images = []
        for ci in pair["candidate_indices"]:
            img = dataset[ci][0]
            if not isinstance(img, Image.Image):
                img = T.ToPILImage()(img)
            cand_images.append(img)

        pair_sigs = {}
        for tname, teacher in teachers.items():
            cand_embs = teacher.encode(cand_images)
            tsig = {}
            for probe_name, probe_tf in probes:
                probed_img = apply_probe_to_pil(query_img, probe_tf)
                q_emb = teacher.encode([probed_img])
                sims = (q_emb @ cand_embs.T).squeeze(0).cpu().tolist()
                tsig[probe_name] = sims
            pair_sigs[tname] = tsig

        sigs[pair["id"]] = pair_sigs

    return sigs


def apply_probe_to_pil(img: Image.Image, transform: T.Compose) -> Image.Image:
    """Apply a probe transform to a PIL image, returning a PIL image."""
    tensor = transform(img)
    # Undo normalization to get back to [0,1] range
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    return T.ToPILImage()(tensor)


def compute_vision_tomography_loss(
    student: VisionStudent,
    query_img: Image.Image,
    cand_images: list[Image.Image],
    teacher_sigs: dict[str, dict[str, list[float]]],
    probes: list[tuple[str, T.Compose]],
    tau: float = 0.05,
) -> torch.Tensor:
    """KL divergence between student and teacher vision ranking distributions under probes."""
    cand_embs = student.encode_images(cand_images)
    device = cand_embs.device
    loss = torch.tensor(0.0, device=device)
    n = 0

    for probe_name, probe_tf in probes:
        probed_img = apply_probe_to_pil(query_img, probe_tf)
        q_emb = student.encode_images([probed_img])
        student_sims = (q_emb @ cand_embs.T).squeeze(0)
        student_log_dist = F.log_softmax(student_sims / tau, dim=0)

        for tname, tsig in teacher_sigs.items():
            if probe_name in tsig:
                target_sims = torch.tensor(tsig[probe_name], dtype=torch.float32, device=device)
                target_dist = F.softmax(target_sims / tau, dim=0)
                kl = F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)
                loss = loss + kl
                n += 1

    return loss / max(n, 1)


def compute_vision_kd_loss(
    student: VisionStudent,
    query_img: Image.Image,
    cand_images: list[Image.Image],
    teacher_scores: list[float],
    tau: float = 0.05,
) -> torch.Tensor:
    """Standard KD: match one teacher's identity ranking."""
    cand_embs = student.encode_images(cand_images)
    q_emb = student.encode_images([query_img])
    student_sims = (q_emb @ cand_embs.T).squeeze(0)
    student_log_dist = F.log_softmax(student_sims / tau, dim=0)
    device = cand_embs.device
    target = torch.tensor(teacher_scores, dtype=torch.float32, device=device)
    target_dist = F.softmax(target / tau, dim=0)
    return F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)


def compute_vision_contrastive_loss(
    student: VisionStudent,
    query_img: Image.Image,
    cand_images: list[Image.Image],
    gold_idx: int,
    tau: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss for images."""
    cand_embs = student.encode_images(cand_images)
    q_emb = student.encode_images([query_img])
    sims = (q_emb @ cand_embs.T).squeeze(0) / tau
    device = sims.device
    target = torch.tensor(gold_idx, device=device)
    return F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))


@torch.no_grad()
def evaluate_vision(student: VisionStudent, dataset, pairs: list[dict]) -> dict:
    hits1 = hits5 = 0
    mrr_sum = 0.0
    for pair in pairs:
        query_img = dataset[pair["query_idx"]][0]
        if not isinstance(query_img, Image.Image):
            query_img = T.ToPILImage()(query_img)

        cand_images = []
        for ci in pair["candidate_indices"]:
            img = dataset[ci][0]
            if not isinstance(img, Image.Image):
                img = T.ToPILImage()(img)
            cand_images.append(img)

        q_emb = student.encode_images([query_img])
        c_embs = student.encode_images(cand_images)
        sims = (q_emb @ c_embs.T).squeeze(0)
        ranked = sims.argsort(descending=True).tolist()
        gold = pair["gold_idx"]
        if ranked[0] == gold:
            hits1 += 1
        if gold in ranked[:5]:
            hits5 += 1
        rank = ranked.index(gold) + 1
        mrr_sum += 1.0 / rank
    n = len(pairs)
    return {"hit@1": hits1/n, "hit@5": hits5/n, "mrr": mrr_sum/n, "n": n}


def run_vision_arm(
    arm_name: str,
    student: VisionStudent,
    dataset,
    train_pairs: list[dict],
    eval_pairs: list[dict],
    teacher_sigs: dict | None,
    probes: list[tuple[str, T.Compose]],
    steps: int,
    lr: float,
    tau: float,
    out_dir: str,
    arm_type: str = "tomography",
):
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name} ({arm_type})")
    print(f"{'='*60}")
    sys.stdout.flush()

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.1)

    base = evaluate_vision(student, dataset, eval_pairs)
    print(f"  Baseline: Hit@1={base['hit@1']:.4f}  Hit@5={base['hit@5']:.4f}  MRR={base['mrr']:.4f}")
    sys.stdout.flush()

    arm_dir = os.path.join(out_dir, arm_name)
    Path(arm_dir).mkdir(parents=True, exist_ok=True)
    log_f = open(os.path.join(arm_dir, "log.jsonl"), "w")

    t0 = time.time()
    running_loss = 0.0

    for step in range(1, steps + 1):
        idx = (step - 1) % len(train_pairs)
        pair = train_pairs[idx]

        query_img = dataset[pair["query_idx"]][0]
        if not isinstance(query_img, Image.Image):
            query_img = T.ToPILImage()(query_img)

        cand_images = []
        for ci in pair["candidate_indices"]:
            img = dataset[ci][0]
            if not isinstance(img, Image.Image):
                img = T.ToPILImage()(img)
            cand_images.append(img)

        optimizer.zero_grad()

        if arm_type == "tomography":
            loss = compute_vision_tomography_loss(
                student, query_img, cand_images,
                teacher_sigs[pair["id"]], probes, tau=tau,
            )
        elif arm_type == "kd_single":
            tid = list(teacher_sigs[pair["id"]].keys())[0]
            loss = compute_vision_kd_loss(
                student, query_img, cand_images,
                teacher_sigs[pair["id"]][tid]["identity"], tau=tau,
            )
        elif arm_type == "kd_avg":
            scores_lists = [t["identity"] for t in teacher_sigs[pair["id"]].values()]
            avg_scores = [sum(s) / len(s) for s in zip(*scores_lists)]
            loss = compute_vision_kd_loss(
                student, query_img, cand_images,
                avg_scores, tau=tau,
            )
        elif arm_type == "contrastive":
            loss = compute_vision_contrastive_loss(
                student, query_img, cand_images,
                pair["gold_idx"], tau=tau,
            )
        else:
            raise ValueError(f"Unknown arm type: {arm_type}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % 50 == 0:
            avg = running_loss / 50
            entry = {"step": step, "loss": round(avg, 6), "elapsed_s": round(time.time() - t0, 1)}
            if step % 200 == 0:
                m = evaluate_vision(student, dataset, eval_pairs)
                entry.update(m)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}")
            sys.stdout.flush()
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

    final = evaluate_vision(student, dataset, eval_pairs)
    result = {
        "arm": arm_name,
        "type": arm_type,
        "steps": steps,
        "baseline": base,
        "final": final,
        "gain_hit1": final["hit@1"] - base["hit@1"],
        "gain_mrr": final["mrr"] - base["mrr"],
    }
    print(f"\n  RESULT: Hit@1 {base['hit@1']:.4f} -> {final['hit@1']:.4f} ({result['gain_hit1']:+.4f})")
    print(f"          MRR   {base['mrr']:.4f} -> {final['mrr']:.4f} ({result['gain_mrr']:+.4f})")
    sys.stdout.flush()

    with open(os.path.join(arm_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log_f.close()
    return result


def main():
    parser = argparse.ArgumentParser(description="Eklavya V1 -- Vision Embedding Tomography")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--out_dir", default="outputs/V1_cifar100")
    parser.add_argument("--student", default="facebook/dinov2-vits14")
    parser.add_argument("--teachers", nargs="+",
                        default=["facebook/dinov2-vitb14", "openai/clip-vit-base-patch32"])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading CIFAR-100...")
    sys.stdout.flush()
    cifar = datasets.CIFAR100(root="data/cifar100", train=True, download=True)

    print("Building retrieval pairs...")
    sys.stdout.flush()
    all_pairs = build_retrieval_pairs(cifar, n=args.n_train + args.n_eval, seed=42)
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]
    print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval pairs")

    probes = VISUAL_PROBES

    print("\nExtracting teacher signatures...")
    sys.stdout.flush()
    teachers = {}
    for tname in args.teachers:
        print(f"  Loading {tname}")
        sys.stdout.flush()
        teachers[tname] = VisionEncoder(tname, device=args.device)

    teacher_sigs = extract_vision_signatures(teachers, cifar, train_pairs, probes)

    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted signatures for {len(teacher_sigs)} pairs")
    sys.stdout.flush()

    config = {
        "student": args.student,
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "dataset": "CIFAR-100",
        "device": args.device,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    results = {}

    # ARM B0: Contrastive only
    student_b0 = VisionStudent(args.student, dim=256).to(args.device)
    results["B0_contrastive"] = run_vision_arm(
        "B0_contrastive", student_b0, cifar, train_pairs, eval_pairs,
        teacher_sigs=None, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="contrastive",
    )
    del student_b0
    torch.cuda.empty_cache()

    # ARM B2: Single-teacher KD
    student_b2 = VisionStudent(args.student, dim=256).to(args.device)
    results["B2_kd_single"] = run_vision_arm(
        "B2_kd_single", student_b2, cifar, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_single",
    )
    del student_b2
    torch.cuda.empty_cache()

    # ARM B3: Multi-teacher average KD
    student_b3 = VisionStudent(args.student, dim=256).to(args.device)
    results["B3_kd_avg"] = run_vision_arm(
        "B3_kd_avg", student_b3, cifar, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_avg",
    )
    del student_b3
    torch.cuda.empty_cache()

    # ARM V1: Full tomography
    student_v1 = VisionStudent(args.student, dim=256).to(args.device)
    results["V1_tomography"] = run_vision_arm(
        "V1_tomography", student_v1, cifar, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="tomography",
    )
    del student_v1
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT V1 SUMMARY (Vision)")
    print("=" * 60)
    print(f"{'Arm':<20} {'Hit@1':>8} {'MRR':>8} {'Gain Hit@1':>12} {'Gain MRR':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['final']['hit@1']:>8.4f} {r['final']['mrr']:>8.4f} "
              f"{r['gain_hit1']:>+12.4f} {r['gain_mrr']:>+10.4f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    tomo = results["V1_tomography"]
    best_baseline = max(results["B0_contrastive"]["final"]["mrr"],
                        results["B2_kd_single"]["final"]["mrr"],
                        results["B3_kd_avg"]["final"]["mrr"])
    margin = tomo["final"]["mrr"] - best_baseline
    print(f"\nTomography vs best baseline MRR margin: {margin:+.4f}")
    if margin > 0.01:
        print("VERDICT: Vision tomography shows signal. Proceed to V2.")
    elif margin > -0.01:
        print("VERDICT: Inconclusive. Need more data/harder eval.")
    else:
        print("VERDICT: Vision tomography absorbed. Investigate why.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
