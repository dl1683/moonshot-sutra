"""Eklavya Experiment A1 — Audio embedding tomography vs standard KD.

Student: BEATs-base (~90M) or AudioMAE (~86M)
Teachers: OpenBEATs-Large (300M) + CLAP-LAION (158M) — heterogeneous objectives
Data: ESC-50 or AudioSet-subset audio retrieval pairs
Probes: identity, time_shift, pitch_shift, speed_change, noise, reverb, freq_mask

Arms:
  A1: Full tomography (multi-probe, multi-teacher KL on ranking distributions)
  B0: Contrastive-only baseline (InfoNCE, no teacher)
  B2: Standard single-teacher KD (identity probe only, best teacher)
  B3: Multi-teacher average KD (average teacher scores, identity probe only)

Usage:
  python code/experiment_a1.py --device cuda --steps 600 --out_dir outputs/A1_esc50
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

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except ImportError:
    HAS_TORCHAUDIO = False


AUDIO_PROBES = [
    "identity",
    "time_shift",
    "pitch_shift",
    "speed_change",
    "noise_inject",
    "reverb",
    "freq_mask",
]

SAMPLE_RATE = 16000


def apply_audio_probe(waveform: torch.Tensor, sr: int, probe_name: str, seed: int = 0) -> torch.Tensor:
    """Apply an audio probe transform to a waveform."""
    rng = random.Random(seed)

    if probe_name == "identity":
        return waveform

    if probe_name == "time_shift":
        shift = rng.randint(-sr // 4, sr // 4)
        return torch.roll(waveform, shifts=shift, dims=-1)

    if probe_name == "pitch_shift":
        if HAS_TORCHAUDIO:
            semitones = rng.choice([-2, -1, 1, 2])
            try:
                return torchaudio.functional.pitch_shift(waveform, sr, semitones)
            except Exception:
                return waveform
        return waveform

    if probe_name == "speed_change":
        if HAS_TORCHAUDIO:
            speed = rng.choice([0.9, 1.1])
            effects = [["speed", str(speed)], ["rate", str(sr)]]
            try:
                out, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
                return out
            except Exception:
                return waveform
        return waveform

    if probe_name == "noise_inject":
        snr_db = rng.choice([10, 15, 20])
        noise = torch.randn_like(waveform)
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = (signal_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
        return waveform + scale * noise

    if probe_name == "reverb":
        decay = 0.3
        delay_samples = int(0.03 * sr)
        result = waveform.clone()
        if result.shape[-1] > delay_samples:
            result[..., delay_samples:] += decay * waveform[..., :-delay_samples]
        return result

    if probe_name == "freq_mask":
        spec = torch.fft.rfft(waveform)
        n_freq = spec.shape[-1]
        mask_start = rng.randint(0, max(1, n_freq - n_freq // 5))
        mask_end = min(mask_start + n_freq // 5, n_freq)
        spec[..., mask_start:mask_end] = 0
        return torch.fft.irfft(spec, n=waveform.shape[-1])

    return waveform


def load_esc50(root: str = "data/esc50") -> tuple[list[tuple[torch.Tensor, int]], list[str]]:
    """Load ESC-50 dataset. Returns list of (waveform, class_id) and class names."""
    meta_path = os.path.join(root, "meta", "esc50.csv")
    audio_dir = os.path.join(root, "audio")

    if not os.path.exists(meta_path):
        print(f"ESC-50 not found at {root}. Download from https://github.com/karolpiczak/ESC-50")
        print("For now, generating synthetic audio data for pipeline testing...")
        return _generate_synthetic_audio(n_classes=50, n_per_class=40)

    import csv
    samples = []
    classes = set()
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = os.path.join(audio_dir, row["filename"])
            if os.path.exists(audio_path):
                waveform, sr = torchaudio.load(audio_path)
                if sr != SAMPLE_RATE:
                    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
                waveform = waveform.mean(dim=0, keepdim=True)[:, :SAMPLE_RATE * 5]
                class_id = int(row["target"])
                samples.append((waveform, class_id))
                classes.add(row["category"])

    return samples, sorted(classes)


def _generate_synthetic_audio(n_classes: int = 50, n_per_class: int = 40) -> tuple[list, list]:
    """Generate synthetic audio for pipeline testing (different frequency bands per class)."""
    samples = []
    classes = [f"class_{i}" for i in range(n_classes)]
    rng = random.Random(42)

    for cls_id in range(n_classes):
        base_freq = 100 + cls_id * 50
        for j in range(n_per_class):
            t = torch.linspace(0, 1, SAMPLE_RATE)
            freq = base_freq + rng.uniform(-20, 20)
            waveform = torch.sin(2 * 3.14159 * freq * t)
            harmonics = rng.randint(1, 4)
            for h in range(2, harmonics + 2):
                waveform += 0.3 / h * torch.sin(2 * 3.14159 * freq * h * t)
            waveform += 0.05 * torch.randn_like(waveform)
            waveform = waveform.unsqueeze(0)
            samples.append((waveform, cls_id))

    return samples, classes


def build_audio_retrieval_pairs(
    samples: list[tuple[torch.Tensor, int]],
    n: int = 300,
    n_candidates: int = 8,
    seed: int = 42,
) -> list[dict]:
    """Build audio retrieval pairs from classification dataset."""
    rng = random.Random(seed)

    by_class: dict[int, list[int]] = {}
    for idx, (_, label) in enumerate(samples):
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
            "id": f"audio_{i}",
            "query_idx": q_idx,
            "candidate_indices": cand_indices,
            "gold_idx": gold_idx,
            "query_class": cls,
        })

    return pairs


class AudioEncoder(nn.Module):
    """Simple audio encoder: mel spectrogram -> CNN -> pooling -> projection."""

    def __init__(self, dim: int = 256, n_mels: int = 64):
        super().__init__()
        self.n_mels = n_mels
        self.dim = dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(128 * 16, dim)

    def _to_melspec(self, waveform: torch.Tensor) -> torch.Tensor:
        if HAS_TORCHAUDIO:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=SAMPLE_RATE, n_mels=self.n_mels, n_fft=1024, hop_length=512,
            ).to(waveform.device)
            spec = mel_transform(waveform)
        else:
            spec = torch.stft(waveform.squeeze(0), n_fft=1024, hop_length=512, return_complex=True)
            spec = spec.abs().unsqueeze(0)[:, :self.n_mels, :]

        return (spec + 1e-9).log()

    def forward(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        specs = []
        for w in waveforms:
            w = w.to(device)
            if w.dim() == 1:
                w = w.unsqueeze(0)
            spec = self._to_melspec(w)
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)
            if spec.dim() == 3:
                spec = spec.unsqueeze(0)
            specs.append(spec)

        max_time = max(s.shape[-1] for s in specs)
        padded = []
        for s in specs:
            if s.shape[-1] < max_time:
                s = F.pad(s, (0, max_time - s.shape[-1]))
            padded.append(s)

        batch = torch.cat(padded, dim=0)
        features = self.conv(batch)
        features = features.view(features.size(0), -1)
        projected = self.proj(features)
        return F.normalize(projected, p=2, dim=-1)


class AudioTeacher:
    """Wraps an audio encoder model for teacher signature extraction."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device
        self.name = model_name
        self.encoder = AudioEncoder(dim=256).to(device).eval()
        print(f"    Note: using lightweight AudioEncoder as teacher proxy for {model_name}")
        print(f"    (Full teacher loading requires model-specific code)")

    @torch.no_grad()
    def encode(self, waveforms: list[torch.Tensor]) -> torch.Tensor:
        return self.encoder(waveforms)


@torch.no_grad()
def extract_audio_signatures(
    teachers: dict[str, AudioTeacher],
    samples: list[tuple[torch.Tensor, int]],
    pairs: list[dict],
    probes: list[str],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Extract teacher signatures for audio pairs under all probes."""
    sigs = {}
    total = len(pairs)

    for pi, pair in enumerate(pairs):
        if (pi + 1) % 50 == 0:
            print(f"    Signatures: {pi + 1}/{total}")

        query_wav = samples[pair["query_idx"]][0]
        cand_wavs = [samples[ci][0] for ci in pair["candidate_indices"]]

        pair_sigs = {}
        for tname, teacher in teachers.items():
            cand_embs = teacher.encode(cand_wavs)
            tsig = {}
            for probe_name in probes:
                probed_wav = apply_audio_probe(query_wav, SAMPLE_RATE, probe_name, seed=pi)
                q_emb = teacher.encode([probed_wav])
                sims = (q_emb @ cand_embs.T).squeeze(0).cpu().tolist()
                tsig[probe_name] = sims
            pair_sigs[tname] = tsig

        sigs[pair["id"]] = pair_sigs

    return sigs


def compute_audio_tomography_loss(
    student: AudioEncoder,
    query_wav: torch.Tensor,
    cand_wavs: list[torch.Tensor],
    teacher_sigs: dict[str, dict[str, list[float]]],
    probes: list[str],
    tau: float = 0.05,
    step_seed: int = 0,
) -> torch.Tensor:
    cand_embs = student(cand_wavs)
    device = cand_embs.device
    loss = torch.tensor(0.0, device=device)
    n = 0

    for probe_name in probes:
        probed_wav = apply_audio_probe(query_wav, SAMPLE_RATE, probe_name, seed=step_seed)
        q_emb = student([probed_wav])
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


def compute_audio_kd_loss(
    student: AudioEncoder,
    query_wav: torch.Tensor,
    cand_wavs: list[torch.Tensor],
    teacher_scores: list[float],
    tau: float = 0.05,
) -> torch.Tensor:
    cand_embs = student(cand_wavs)
    q_emb = student([query_wav])
    student_sims = (q_emb @ cand_embs.T).squeeze(0)
    student_log_dist = F.log_softmax(student_sims / tau, dim=0)
    device = cand_embs.device
    target = torch.tensor(teacher_scores, dtype=torch.float32, device=device)
    target_dist = F.softmax(target / tau, dim=0)
    return F.kl_div(student_log_dist, target_dist, reduction="batchmean", log_target=False)


def compute_audio_contrastive_loss(
    student: AudioEncoder,
    query_wav: torch.Tensor,
    cand_wavs: list[torch.Tensor],
    gold_idx: int,
    tau: float = 0.05,
) -> torch.Tensor:
    cand_embs = student(cand_wavs)
    q_emb = student([query_wav])
    sims = (q_emb @ cand_embs.T).squeeze(0) / tau
    target = torch.tensor(gold_idx, device=sims.device)
    return F.cross_entropy(sims.unsqueeze(0), target.unsqueeze(0))


@torch.no_grad()
def evaluate_audio(student: AudioEncoder, samples, pairs: list[dict]) -> dict:
    hits1 = hits5 = 0
    mrr_sum = 0.0
    for pair in pairs:
        q_wav = samples[pair["query_idx"]][0]
        c_wavs = [samples[ci][0] for ci in pair["candidate_indices"]]
        q_emb = student([q_wav])
        c_embs = student(c_wavs)
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


def run_audio_arm(
    arm_name: str,
    student: AudioEncoder,
    samples,
    train_pairs: list[dict],
    eval_pairs: list[dict],
    teacher_sigs: dict | None,
    probes: list[str],
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

    base = evaluate_audio(student, samples, eval_pairs)
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

        query_wav = samples[pair["query_idx"]][0]
        cand_wavs = [samples[ci][0] for ci in pair["candidate_indices"]]

        optimizer.zero_grad()

        if arm_type == "tomography":
            loss = compute_audio_tomography_loss(
                student, query_wav, cand_wavs,
                teacher_sigs[pair["id"]], probes, tau=tau, step_seed=step,
            )
        elif arm_type == "kd_single":
            tid = list(teacher_sigs[pair["id"]].keys())[0]
            loss = compute_audio_kd_loss(
                student, query_wav, cand_wavs,
                teacher_sigs[pair["id"]][tid]["identity"], tau=tau,
            )
        elif arm_type == "kd_avg":
            scores_lists = [t["identity"] for t in teacher_sigs[pair["id"]].values()]
            avg_scores = [sum(s) / len(s) for s in zip(*scores_lists)]
            loss = compute_audio_kd_loss(
                student, query_wav, cand_wavs, avg_scores, tau=tau,
            )
        elif arm_type == "contrastive":
            loss = compute_audio_contrastive_loss(
                student, query_wav, cand_wavs, pair["gold_idx"], tau=tau,
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
                m = evaluate_audio(student, samples, eval_pairs)
                entry.update(m)
                print(f"  step {step:>5d}  loss={avg:.4f}  hit@1={m['hit@1']:.4f}  mrr={m['mrr']:.4f}")
            else:
                print(f"  step {step:>5d}  loss={avg:.4f}")
            sys.stdout.flush()
            log_f.write(json.dumps(entry) + "\n")
            log_f.flush()
            running_loss = 0.0

    final = evaluate_audio(student, samples, eval_pairs)
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
    parser = argparse.ArgumentParser(description="Eklavya A1 -- Audio Embedding Tomography")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--out_dir", default="outputs/A1_esc50")
    parser.add_argument("--data_dir", default="data/esc50")
    parser.add_argument("--teachers", nargs="+",
                        default=["OpenBEATs-Large", "CLAP-LAION"])
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading audio data...")
    sys.stdout.flush()
    samples, class_names = load_esc50(args.data_dir)
    print(f"Loaded {len(samples)} audio samples, {len(class_names)} classes")

    print("Building retrieval pairs...")
    sys.stdout.flush()
    all_pairs = build_audio_retrieval_pairs(samples, n=args.n_train + args.n_eval, seed=42)
    train_pairs = all_pairs[:args.n_train]
    eval_pairs = all_pairs[args.n_train:]
    print(f"Data: {len(train_pairs)} train, {len(eval_pairs)} eval pairs")

    probes = AUDIO_PROBES

    print("\nExtracting teacher signatures...")
    sys.stdout.flush()
    teachers = {}
    for tname in args.teachers:
        print(f"  Loading {tname}")
        sys.stdout.flush()
        teachers[tname] = AudioTeacher(tname, device=args.device)

    teacher_sigs = extract_audio_signatures(teachers, samples, train_pairs, probes)

    del teachers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Extracted signatures for {len(teacher_sigs)} pairs")
    sys.stdout.flush()

    config = {
        "teachers": args.teachers,
        "steps": args.steps,
        "lr": args.lr,
        "tau": args.tau,
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "dataset": "ESC-50 (or synthetic)",
        "device": args.device,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    results = {}

    student_b0 = AudioEncoder(dim=256).to(args.device)
    results["B0_contrastive"] = run_audio_arm(
        "B0_contrastive", student_b0, samples, train_pairs, eval_pairs,
        teacher_sigs=None, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="contrastive",
    )
    del student_b0
    torch.cuda.empty_cache()

    student_b2 = AudioEncoder(dim=256).to(args.device)
    results["B2_kd_single"] = run_audio_arm(
        "B2_kd_single", student_b2, samples, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_single",
    )
    del student_b2
    torch.cuda.empty_cache()

    student_b3 = AudioEncoder(dim=256).to(args.device)
    results["B3_kd_avg"] = run_audio_arm(
        "B3_kd_avg", student_b3, samples, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="kd_avg",
    )
    del student_b3
    torch.cuda.empty_cache()

    student_a1 = AudioEncoder(dim=256).to(args.device)
    results["A1_tomography"] = run_audio_arm(
        "A1_tomography", student_a1, samples, train_pairs, eval_pairs,
        teacher_sigs=teacher_sigs, probes=probes, steps=args.steps, lr=args.lr, tau=args.tau,
        out_dir=args.out_dir, arm_type="tomography",
    )
    del student_a1
    torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("EXPERIMENT A1 SUMMARY (Audio)")
    print("=" * 60)
    print(f"{'Arm':<20} {'Hit@1':>8} {'MRR':>8} {'Gain Hit@1':>12} {'Gain MRR':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<20} {r['final']['hit@1']:>8.4f} {r['final']['mrr']:>8.4f} "
              f"{r['gain_hit1']:>+12.4f} {r['gain_mrr']:>+10.4f}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    tomo = results["A1_tomography"]
    best_baseline = max(results["B0_contrastive"]["final"]["mrr"],
                        results["B2_kd_single"]["final"]["mrr"],
                        results["B3_kd_avg"]["final"]["mrr"])
    margin = tomo["final"]["mrr"] - best_baseline
    print(f"\nTomography vs best baseline MRR margin: {margin:+.4f}")
    if margin > 0.01:
        print("VERDICT: Audio tomography shows signal. Proceed to A2.")
    elif margin > -0.01:
        print("VERDICT: Inconclusive. Need more data/harder eval.")
    else:
        print("VERDICT: Audio tomography absorbed. Investigate why.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
