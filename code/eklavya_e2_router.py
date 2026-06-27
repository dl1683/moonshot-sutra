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
    mode: str = "oracle_gold"
    alpha: float = 1.0
    beta: float = 0.5
    tau: float = 1.0
    agreement_gamma: float = 0.5
    student_delta: float = 0.25

    def __post_init__(self):
        import math
        if not (isinstance(self.tau, (int, float)) and math.isfinite(self.tau)
                and self.tau > 0):
            raise ValueError(f"tau must be finite and positive, got {self.tau}")


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


_VALID_ROUTER_MODES = frozenset({
    "oracle_gold",
    "gold_free_entropy",
    "gold_free_agreement",
    "gold_free_student_jsd",
})


def _pairwise_agreement_scores(
    names: list[str],
    fulls: dict[str, np.ndarray],
) -> list[float]:
    """Per-teacher outlier score: mean JSD to every other teacher.

    Higher = more isolated from the ecology.  Used as a penalty in gold-free
    routing (teachers that disagree with everyone are less trustworthy).
    """
    n = len(names)
    if n <= 1:
        return [0.0] * n
    scores = []
    for i, name_i in enumerate(names):
        jsd_sum = 0.0
        for j, name_j in enumerate(names):
            if i == j:
                continue
            m = 0.5 * (fulls[name_i] + fulls[name_j])
            m = np.maximum(m, 1e-12)
            h_m = -np.sum(m * np.log(m))
            h_i = -np.sum(fulls[name_i] * np.log(fulls[name_i]))
            h_j = -np.sum(fulls[name_j] * np.log(fulls[name_j]))
            jsd_sum += h_m - 0.5 * (h_i + h_j)
        scores.append(jsd_sum / (n - 1))
    return scores


def _teacher_student_jsd(
    teacher_full: np.ndarray,
    student_probs: np.ndarray,
) -> float:
    """JSD between a single teacher distribution and the student distribution."""
    s = np.maximum(student_probs, 1e-12)
    s = s / s.sum()
    m = 0.5 * (teacher_full + s)
    m = np.maximum(m, 1e-12)
    h_m = -np.sum(m * np.log(m))
    h_t = -np.sum(teacher_full * np.log(teacher_full))
    h_s = -np.sum(s * np.log(s))
    return float(h_m - 0.5 * (h_t + h_s))


def route_teachers(
    teacher_dists: dict[str, SparseByteDist],
    gold_byte: int | None,
    priors: dict[str, float],
    config: RouterConfig | None = None,
    student_probs: np.ndarray | None = None,
    student_entropy: float | None = None,
) -> RouteResult:
    """Compute per-teacher routing weights for a single position.

    Modes:
        oracle_gold:           log(prior) + alpha * z(logq_gold) - beta * z(H)
        gold_free_entropy:     log(prior) - beta * z(H)
        gold_free_agreement:   log(prior) - beta * z(H) - gamma * z(A)
        gold_free_student_jsd: log(prior) - beta * z(H) - gamma * z(A) + delta * U_s * z(D)
    """
    if config is None:
        config = RouterConfig()

    if config.mode not in _VALID_ROUTER_MODES:
        raise ValueError(
            f"Unknown router mode: {config.mode!r}. "
            f"Valid: {sorted(_VALID_ROUTER_MODES)}"
        )

    names = sorted(teacher_dists.keys())
    if not names:
        return RouteResult(weights={}, jsd=0.0, route_entropy=0.0)

    is_gold_free = config.mode.startswith("gold_free")

    if config.mode == "gold_free_student_jsd":
        if student_probs is None:
            raise ValueError(
                "gold_free_student_jsd mode requires student_probs"
            )
        if student_entropy is None:
            raise ValueError(
                "gold_free_student_jsd mode requires student_entropy"
            )

    fulls = {name: _sparse_to_full(teacher_dists[name]) for name in names}
    entropies = [-float(np.sum(fulls[n] * np.log(fulls[n]))) for n in names]
    z_ent = _zscore(entropies)

    if not is_gold_free:
        if gold_byte is None:
            raise ValueError("oracle_gold mode requires gold_byte")
        log_q_golds = []
        for name in names:
            dist = teacher_dists[name]
            mask = dist.top_bytes == gold_byte
            if mask.any():
                p_gold = float(dist.top_probs[mask][0])
            else:
                p_gold = float(dist.tail_prob) / max(1, 256 - len(dist.top_bytes))
            log_q_golds.append(np.log(max(p_gold, 1e-10)))
        z_logq = _zscore(log_q_golds)

    agreement_scores = None
    if config.mode in ("gold_free_agreement", "gold_free_student_jsd"):
        agreement_scores = _pairwise_agreement_scores(names, fulls)

    student_jsd_scores = None
    if config.mode == "gold_free_student_jsd" and student_probs is not None:
        student_jsd_scores = [
            _teacher_student_jsd(fulls[n], student_probs) for n in names
        ]

    z_agree = _zscore(agreement_scores) if agreement_scores is not None else None
    z_sjsd = _zscore(student_jsd_scores) if student_jsd_scores is not None else None

    u_s = 0.0
    if student_entropy is not None:
        u_s = student_entropy / np.log(256)

    scores = []
    for i, name in enumerate(names):
        prior = priors.get(name, 1.0 / len(names))
        s = np.log(max(prior, 1e-10))

        if config.mode == "oracle_gold":
            s += config.alpha * z_logq[i] - config.beta * z_ent[i]
        elif config.mode == "gold_free_entropy":
            s -= config.beta * z_ent[i]
        elif config.mode == "gold_free_agreement":
            s -= config.beta * z_ent[i]
            s -= config.agreement_gamma * z_agree[i]
        elif config.mode == "gold_free_student_jsd":
            s -= config.beta * z_ent[i]
            s -= config.agreement_gamma * z_agree[i]
            s += config.student_delta * u_s * z_sjsd[i]

        scores.append(s)

    scores_arr = np.array(scores) / config.tau
    np.nan_to_num(scores_arr, copy=False, nan=0.0, posinf=0.0, neginf=-30.0)
    scores_arr -= scores_arr.max()
    exp_scores = np.exp(scores_arr)
    denom = exp_scores.sum()
    if not np.isfinite(denom) or denom < 1e-30:
        weights = np.full(len(names), 1.0 / len(names))
    else:
        weights = exp_scores / denom

    w_dict = {name: float(weights[i]) for i, name in enumerate(names)}

    jsd = _compute_jsd(teacher_dists, w_dict)
    rent = max(0.0, float(-np.sum(weights * np.log(weights + 1e-10))))

    return RouteResult(weights=w_dict, jsd=jsd, route_entropy=rent)


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
    if not teacher_dists:
        return 0.0
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
    np.nan_to_num(full, copy=False, nan=1e-12, posinf=1e-12, neginf=1e-12)
    full = np.maximum(full, 1e-12)
    total = full.sum()
    if not np.isfinite(total) or total < 1e-12:
        return np.full(256, 1.0 / 256, dtype=np.float64)
    full /= total
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
        np.nan_to_num(mixture, copy=False, nan=1e-12, posinf=1e-12, neginf=1e-12)
        mixture = np.maximum(mixture, 1e-12)
        mixture /= mixture.sum()

    elif mode == "log_pool":
        log_mixture = np.zeros(256, dtype=np.float64)
        for name in names:
            w = weights.get(name, 0.0)
            log_mixture += w * np.log(fulls[name])
        np.clip(log_mixture, -50.0, 50.0, out=log_mixture)
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
    student_logits: torch.Tensor | None = None,
) -> list[RouteResult]:
    """Route all positions in a batch. Returns per-position RouteResults.

    Args:
        student_logits: [B, 256] student logits for gold-free routing modes.
            Required when config.mode starts with 'gold_free_student_jsd'.
    """
    if priors is None:
        priors = {spec.name: spec.prior for spec in TEACHER_REGISTRY}
    if config is None:
        config = RouterConfig()

    needs_student = config.mode == "gold_free_student_jsd"
    if needs_student and student_logits is None:
        raise ValueError(
            "gold_free_student_jsd mode requires student_logits in route_batch"
        )
    if student_logits is not None and student_logits.ndim != 2:
        raise ValueError(
            f"student_logits must be [B, 256], got shape {list(student_logits.shape)}"
        )

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

        gold = int(batch.gold_bytes[i].item()) if not config.mode.startswith("gold_free") else None

        s_probs = None
        s_ent = None
        if student_logits is not None:
            s_logits_i = student_logits[i].detach().cpu().float()
            s_probs_t = torch.softmax(s_logits_i, dim=-1)
            s_probs = s_probs_t.numpy().astype(np.float64)
            s_ent = -float(np.sum(s_probs * np.log(np.maximum(s_probs, 1e-12))))
        elif not config.mode.startswith("gold_free"):
            s_ent = float(batch.student_entropies[i].item())

        results.append(route_teachers(
            teacher_dists, gold, priors, config,
            student_probs=s_probs, student_entropy=s_ent,
        ))
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
