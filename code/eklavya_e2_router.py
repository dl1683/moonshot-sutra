"""Eklavya E2 — Teacher routing and byte distribution purification.

Components:
  - MultiTeacherBatch: joins position manifest with per-teacher records
  - TeacherRouterV0: simplified Plackett-Luce score → softmax weights
  - purify_byte_target: arithmetic / log-pool / route-to-best purifier
  - disagreement_jsd: Jensen-Shannon divergence across teacher distributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from eklavya_e2_cache import (
    TeacherSpec, TEACHER_REGISTRY,
    PositionRecord, E2KLRecord,
    SparseByteDist,
)


# ---------------------------------------------------------------------------
# MultiTeacherBatch — joins position manifest + per-teacher records
# ---------------------------------------------------------------------------

@dataclass
class TeacherSlice:
    """One teacher's data for a batch of positions."""
    spec: TeacherSpec
    top_bytes: torch.Tensor    # [B, K] uint8→long
    top_probs: torch.Tensor    # [B, K] float32
    tail_probs: torch.Tensor   # [B] float32
    entropies: torch.Tensor    # [B] float32
    logp_golds: torch.Tensor   # [B] float32
    valid_mask: torch.Tensor   # [B] bool — False if teacher had no record


@dataclass
class MultiTeacherBatch:
    """Batch of positions with per-teacher signals joined by position_id."""
    position_ids: torch.Tensor   # [B] int32
    gold_bytes: torch.Tensor     # [B] uint8→long
    student_nlls: torch.Tensor   # [B] float32
    student_entropies: torch.Tensor  # [B] float32
    reason_masks: torch.Tensor   # [B] uint8
    teachers: dict[str, TeacherSlice]

    @property
    def batch_size(self) -> int:
        return self.position_ids.shape[0]

    @property
    def n_teachers(self) -> int:
        return len(self.teachers)


def build_multi_teacher_batch(
    positions: list[PositionRecord],
    teacher_kl: dict[str, list[E2KLRecord]],
    teacher_specs: list[TeacherSpec],
    K: int = 16,
) -> MultiTeacherBatch:
    """Join position manifest with per-teacher KL records by position_id.

    Args:
        positions: list of PositionRecords (batch)
        teacher_kl: {teacher_name: list of E2KLRecords} — must be indexed
            such that records can be looked up by position_id
        teacher_specs: which teachers to include
        K: top-K for byte distributions
    """
    B = len(positions)
    pos_ids = torch.tensor([p.position_id for p in positions], dtype=torch.int32)
    golds = torch.tensor([p.gold_byte for p in positions], dtype=torch.long)
    nlls = torch.tensor([p.student_nll for p in positions], dtype=torch.float32)
    ents = torch.tensor([p.student_entropy for p in positions], dtype=torch.float32)
    masks = torch.tensor([p.reason_mask for p in positions], dtype=torch.uint8)

    pid_set = {p.position_id: i for i, p in enumerate(positions)}

    teachers = {}
    for spec in teacher_specs:
        if not spec.has_kl:
            continue
        name = spec.name
        kl_recs = teacher_kl.get(name, [])

        top_bytes = torch.zeros(B, K, dtype=torch.long)
        top_probs = torch.zeros(B, K, dtype=torch.float32)
        tail_probs = torch.zeros(B, dtype=torch.float32)
        entropies = torch.zeros(B, dtype=torch.float32)
        logp_golds = torch.zeros(B, dtype=torch.float32)
        valid = torch.zeros(B, dtype=torch.bool)

        kl_by_pid = {r.position_id: r for r in kl_recs}

        for pid, batch_idx in pid_set.items():
            rec = kl_by_pid.get(pid)
            if rec is None:
                continue
            valid[batch_idx] = True
            top_bytes[batch_idx] = torch.from_numpy(rec.top_bytes[:K].astype(np.int64))
            top_probs[batch_idx] = torch.from_numpy(rec.top_probs[:K].astype(np.float32))
            tail_probs[batch_idx] = rec.tail_prob
            entropies[batch_idx] = rec.entropy
            logp_golds[batch_idx] = rec.logp_gold

        teachers[name] = TeacherSlice(
            spec=spec,
            top_bytes=top_bytes,
            top_probs=top_probs,
            tail_probs=tail_probs,
            entropies=entropies,
            logp_golds=logp_golds,
            valid_mask=valid,
        )

    return MultiTeacherBatch(
        position_ids=pos_ids,
        gold_bytes=golds,
        student_nlls=nlls,
        student_entropies=ents,
        reason_masks=masks,
        teachers=teachers,
    )


# ---------------------------------------------------------------------------
# TeacherRouterV0 — Plackett-Luce style score → softmax
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    alpha: float = 1.0
    beta: float = 0.5
    tau: float = 1.0


@dataclass
class RouteResult:
    """Routing decision for one position."""
    weights: dict[str, float]
    jsd: float
    route_entropy: float


def _zscore(values: list[float]) -> list[float]:
    if len(values) <= 1:
        return [0.0] * len(values)
    arr = np.array(values, dtype=np.float64)
    mu = arr.mean()
    std = arr.std()
    if std < 1e-12:
        return [0.0] * len(values)
    return ((arr - mu) / std).tolist()


def route_teachers(
    teacher_dists: dict[str, SparseByteDist],
    gold_byte: int,
    priors: dict[str, float],
    config: RouterConfig | None = None,
) -> RouteResult:
    """Compute per-teacher routing weights for a single position.

    PL-style score:
        score_t = log(prior_t) + alpha * z(log q_t[gold]) - beta * z(H_t)

    Args:
        teacher_dists: {teacher_name: SparseByteDist}
        gold_byte: ground truth byte value (0-255)
        priors: {teacher_name: prior weight}
        config: router hyperparameters
    """
    if config is None:
        config = RouterConfig()

    names = sorted(teacher_dists.keys())
    if not names:
        return RouteResult(weights={}, jsd=0.0, route_entropy=0.0)

    log_q_golds = []
    entropies = []
    for name in names:
        dist = teacher_dists[name]
        mask = dist.top_bytes == gold_byte
        if mask.any():
            p_gold = float(dist.top_probs[mask][0])
        else:
            p_gold = float(dist.tail_prob) / max(1, 256 - len(dist.top_bytes))
        log_q_golds.append(np.log(max(p_gold, 1e-10)))

        full = _sparse_to_full(dist)
        ent = -float(np.sum(full * np.log(full)))
        entropies.append(ent)

    z_logq = _zscore(log_q_golds)
    z_ent = _zscore(entropies)

    scores = []
    for i, name in enumerate(names):
        prior = priors.get(name, 1.0 / len(names))
        s = np.log(max(prior, 1e-10)) + config.alpha * z_logq[i] - config.beta * z_ent[i]
        scores.append(s)

    scores_arr = np.array(scores) / config.tau
    scores_arr -= scores_arr.max()
    exp_scores = np.exp(scores_arr)
    weights = exp_scores / exp_scores.sum()

    w_dict = {name: float(weights[i]) for i, name in enumerate(names)}

    jsd = _compute_jsd(teacher_dists, w_dict)
    rent = -np.sum(weights * np.log(weights + 1e-10))

    return RouteResult(weights=w_dict, jsd=jsd, route_entropy=float(rent))


def _compute_jsd(
    teacher_dists: dict[str, SparseByteDist],
    weights: dict[str, float],
) -> float:
    """Jensen-Shannon divergence across weighted teacher distributions."""
    mixture = np.zeros(256, dtype=np.float64)
    teacher_fulls = {}

    for name, dist in teacher_dists.items():
        full = np.full(256, float(dist.tail_prob) / max(1, 256 - len(dist.top_bytes)),
                       dtype=np.float64)
        for b, p in zip(dist.top_bytes, dist.top_probs):
            full[int(b)] = float(p)
        full = np.maximum(full, 1e-12)
        full /= full.sum()
        teacher_fulls[name] = full
        mixture += weights.get(name, 0.0) * full

    mixture = np.maximum(mixture, 1e-12)
    h_mix = -np.sum(mixture * np.log(mixture))
    h_avg = sum(
        weights.get(name, 0.0) * (-np.sum(f * np.log(f)))
        for name, f in teacher_fulls.items()
    )
    return float(h_mix - h_avg)


def disagreement_jsd(
    teacher_dists: dict[str, SparseByteDist],
    weights: dict[str, float] | None = None,
) -> float:
    """Convenience: compute JSD with uniform weights if none given."""
    if weights is None:
        n = len(teacher_dists)
        weights = {name: 1.0 / n for name in teacher_dists}
    return _compute_jsd(teacher_dists, weights)


# ---------------------------------------------------------------------------
# Byte distribution purification
# ---------------------------------------------------------------------------

def _sparse_to_full(dist: SparseByteDist) -> np.ndarray:
    """Expand sparse byte dist to full 256-dim distribution."""
    n_top = len(dist.top_bytes)
    n_tail = max(1, 256 - n_top)
    full = np.full(256, float(dist.tail_prob) / n_tail, dtype=np.float64)
    for b, p in zip(dist.top_bytes, dist.top_probs):
        full[int(b)] = float(p)
    full = np.maximum(full, 1e-12)
    full /= full.sum()
    return full


def _full_to_sparse(full: np.ndarray, K: int = 16) -> SparseByteDist:
    """Convert full 256-dim distribution back to sparse top-K."""
    top_k_idx = np.argsort(full)[::-1][:K]
    top_bytes = top_k_idx.astype(np.uint8)
    top_probs = full[top_k_idx].astype(np.float32)
    tail_prob = float(1.0 - top_probs.sum())
    return SparseByteDist(top_bytes=top_bytes, top_probs=top_probs,
                          tail_prob=max(tail_prob, 0.0))


def purify_byte_target(
    teacher_dists: dict[str, SparseByteDist],
    route: RouteResult,
    mode: Literal["arithmetic", "log_pool", "route"] = "arithmetic",
    K: int = 16,
) -> SparseByteDist | None:
    """Purify teacher distributions into a single target.

    Modes:
        arithmetic: weighted average (safe default)
        log_pool: weighted geometric mean (sharper, rewards agreement)
        route: pick the highest-weight teacher only
    """
    if not teacher_dists:
        return None

    names = sorted(teacher_dists.keys())
    weights = route.weights

    if mode == "route":
        best = max(names, key=lambda n: weights.get(n, 0.0))
        return teacher_dists[best]

    fulls = {name: _sparse_to_full(teacher_dists[name]) for name in names}

    if mode == "arithmetic":
        mixture = np.zeros(256, dtype=np.float64)
        for name in names:
            w = weights.get(name, 0.0)
            mixture += w * fulls[name]
        mixture = np.maximum(mixture, 1e-12)
        mixture /= mixture.sum()

    elif mode == "log_pool":
        log_mixture = np.zeros(256, dtype=np.float64)
        for name in names:
            w = weights.get(name, 0.0)
            log_mixture += w * np.log(fulls[name])
        mixture = np.exp(log_mixture)
        mixture = np.maximum(mixture, 1e-12)
        mixture /= mixture.sum()

    else:
        raise ValueError(f"Unknown purification mode: {mode}")

    return _full_to_sparse(mixture, K)


# ---------------------------------------------------------------------------
# Batch-level routing and purification (tensor ops)
# ---------------------------------------------------------------------------

def route_batch(
    batch: MultiTeacherBatch,
    priors: dict[str, float] | None = None,
    config: RouterConfig | None = None,
) -> list[RouteResult]:
    """Route all positions in a batch. Returns per-position RouteResults."""
    if priors is None:
        priors = {spec.name: spec.prior for spec in TEACHER_REGISTRY}

    results = []
    for i in range(batch.batch_size):
        teacher_dists = {}
        for name, ts in batch.teachers.items():
            if not ts.valid_mask[i]:
                continue
            teacher_dists[name] = SparseByteDist(
                top_bytes=ts.top_bytes[i].detach().cpu().numpy().astype(np.uint8),
                top_probs=ts.top_probs[i].detach().cpu().numpy().astype(np.float32),
                tail_prob=float(ts.tail_probs[i].item()),
            )
        gold = int(batch.gold_bytes[i].item())
        results.append(route_teachers(teacher_dists, gold, priors, config))
    return results


def purify_batch(
    batch: MultiTeacherBatch,
    routes: list[RouteResult],
    mode: Literal["arithmetic", "log_pool", "route"] = "arithmetic",
    K: int = 16,
) -> list[SparseByteDist | None]:
    """Purify all positions in a batch."""
    results = []
    for i in range(batch.batch_size):
        teacher_dists = {}
        for name, ts in batch.teachers.items():
            if not ts.valid_mask[i]:
                continue
            teacher_dists[name] = SparseByteDist(
                top_bytes=ts.top_bytes[i].detach().cpu().numpy().astype(np.uint8),
                top_probs=ts.top_probs[i].detach().cpu().numpy().astype(np.float32),
                tail_prob=float(ts.tail_probs[i].item()),
            )
        results.append(purify_byte_target(teacher_dists, routes[i], mode, K))
    return results
