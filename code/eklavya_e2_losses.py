"""Eklavya E2 — Multi-teacher loss functions and gradient budgeting.

Components:
  - MultiTeacherProjectionPorts: per-teacher align/semantic heads
  - E2 KL loss: purified sparse byte distribution → student logits
  - Semantic loss: cosine alignment for embedding teachers
  - Multi-teacher gradient budget: per-teacher + total caps
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from eklavya_e2_cache import TeacherSpec, TEACHER_REGISTRY, SparseByteDist
from eklavya_training import topk_tail_kl, AlignProjection


# ---------------------------------------------------------------------------
# Per-teacher projection ports
# ---------------------------------------------------------------------------

class SemanticTeacherPort(nn.Module):
    """Projection for embedding/semantic teachers (cosine loss)."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(student_dim)
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.norm(h)), dim=-1)


class MultiTeacherProjectionPorts(nn.Module):
    """Container for all per-teacher projection heads."""

    def __init__(self, student_dim: int = 576,
                 teachers: list[TeacherSpec] | None = None):
        super().__init__()
        if teachers is None:
            teachers = TEACHER_REGISTRY

        self.teacher_names = []
        self.align_ports = nn.ModuleDict()
        self.semantic_ports = nn.ModuleDict()

        for spec in teachers:
            self.teacher_names.append(spec.name)

            if spec.has_align:
                self.align_ports[spec.name] = AlignProjection(
                    student_dim, spec.hidden_dim)

            if spec.has_semantic:
                self.semantic_ports[spec.name] = SemanticTeacherPort(
                    student_dim, spec.hidden_dim)

    def warm_start_from_e1(self, e1_align_proj: AlignProjection,
                            anchor_name: str = "t0_anchor_decoder"):
        """Copy E1's single AlignProjection into the anchor teacher port."""
        if anchor_name in self.align_ports:
            self.align_ports[anchor_name].load_state_dict(
                e1_align_proj.state_dict())

    def get_align_projection(self, teacher_name: str,
                              patch_states: torch.Tensor) -> torch.Tensor:
        if teacher_name not in self.align_ports:
            raise ValueError(f"No align port for {teacher_name}")
        return self.align_ports[teacher_name](patch_states)

    def get_semantic_projection(self, teacher_name: str,
                                 hidden_states: torch.Tensor) -> torch.Tensor:
        if teacher_name not in self.semantic_ports:
            raise ValueError(f"No semantic port for {teacher_name}")
        return self.semantic_ports[teacher_name](hidden_states)


# ---------------------------------------------------------------------------
# E2 KL loss — purified sparse target → student logits
# ---------------------------------------------------------------------------

def e2_topk_tail_kl(
    student_logits: torch.Tensor,
    purified: SparseByteDist,
    T: float = 2.0,
) -> torch.Tensor:
    """KL divergence from purified sparse byte distribution to student.

    Same structure as E1 topk_tail_kl but takes SparseByteDist directly.
    """
    top_bytes = torch.from_numpy(purified.top_bytes.astype('int64')).to(student_logits.device)
    top_probs = torch.from_numpy(purified.top_probs).to(student_logits.device)
    tail_prob = torch.tensor(purified.tail_prob, device=student_logits.device)

    return topk_tail_kl(student_logits, top_bytes, top_probs, tail_prob, T)


def e2_batch_kl_loss(
    student_logits_batch: torch.Tensor,
    purified_targets: list[SparseByteDist | None],
    T: float = 2.0,
) -> torch.Tensor:
    """Compute mean KL loss across a batch of purified targets.

    Skips positions where purified target is None (insufficient teachers).
    """
    losses = []
    for i, target in enumerate(purified_targets):
        if target is None:
            continue
        loss = e2_topk_tail_kl(student_logits_batch[i], target, T)
        if torch.isfinite(loss):
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=student_logits_batch.device,
                            requires_grad=True)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Semantic cosine loss
# ---------------------------------------------------------------------------

def semantic_cosine_loss(
    student_proj: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """1 - cosine_similarity, averaged over batch.

    Both inputs should be L2-normalized already (semantic port does this).
    teacher_embedding should also be normalized.
    """
    teacher_norm = F.normalize(teacher_embedding, dim=-1)
    cos_sim = (student_proj * teacher_norm).sum(dim=-1)
    return (1.0 - cos_sim).mean()


# ---------------------------------------------------------------------------
# Multi-teacher gradient budget
# ---------------------------------------------------------------------------

@dataclass
class GradientBudgetReport:
    ce_grad_norm: float
    per_teacher_norms: dict[str, float]
    per_teacher_scales: dict[str, float]
    total_teacher_norm_before: float
    total_teacher_norm_after: float
    total_scale: float
    ce_teacher_cosines: dict[str, float] | None = None
    pairwise_coherence: float | None = None


def _collect_grads(params) -> dict:
    return {id(p): p.grad.detach().clone() for p in params if p.grad is not None}


def _clear_grads(params):
    for p in params:
        p.grad = None


def _flat_grad_vec(grad_dict: dict, params: list) -> torch.Tensor:
    parts = []
    for p in params:
        pid = id(p)
        if pid in grad_dict:
            parts.append(grad_dict[pid].float().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel()))
    return torch.cat(parts) if parts else torch.zeros(1)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm()
    nb = b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a * b).sum() / (na * nb))


def apply_multi_teacher_gradient_budget(
    params: list,
    ce_loss: torch.Tensor,
    teacher_losses: dict[str, torch.Tensor],
    per_teacher_cap: float | dict[str, float] = 0.10,
    total_teacher_cap: float = 0.30,
    scaler=None,
    retain_graph: bool = False,
    compute_coherence: bool = True,
) -> GradientBudgetReport:
    """Multi-teacher gradient budgeting.

    Algorithm:
      1. Save existing accumulated grads.
      2. Backward CE, capture CE grads.
      3. For each teacher loss:
         a. Backward, capture grads.
         b. Scale to per_teacher_cap * CE_norm.
      4. Sum scaled teacher grads.
      5. Scale total teacher grads to total_teacher_cap * CE_norm.
      6. Restore: saved + CE + capped teacher grads.
    """
    params = list(params)

    saved = _collect_grads(params)
    _clear_grads(params)

    if ce_loss.requires_grad:
        ce_retain = bool(teacher_losses) or retain_graph
        if scaler is not None:
            scaler.scale(ce_loss).backward(retain_graph=ce_retain)
        else:
            ce_loss.backward(retain_graph=ce_retain)
        ce_grads = _collect_grads(params)
        ce_norm = sum(g.float().norm().item() ** 2 for g in ce_grads.values()) ** 0.5
        _clear_grads(params)
    else:
        ce_grads = {}
        ce_norm = 0.0

    teacher_names = sorted(teacher_losses.keys())
    per_teacher_norms = {}
    per_teacher_scales = {}
    total_teacher_grads = {}
    per_teacher_grads_raw = {} if compute_coherence else None

    for i, name in enumerate(teacher_names):
        loss = teacher_losses[name]
        is_last = (i == len(teacher_names) - 1)

        if scaler is not None:
            scaler.scale(loss).backward(retain_graph=not is_last or retain_graph)
        else:
            loss.backward(retain_graph=not is_last or retain_graph)

        t_norm = sum(
            p.grad.float().norm().item() ** 2
            for p in params if p.grad is not None
        ) ** 0.5
        per_teacher_norms[name] = t_norm
        if compute_coherence:
            per_teacher_grads_raw[name] = _collect_grads(params)

        cap = per_teacher_cap.get(name, 0.10) if isinstance(per_teacher_cap, dict) else per_teacher_cap
        scale = 1.0
        if ce_loss.requires_grad:
            effective_ce = max(ce_norm, 1e-6)
            if t_norm > cap * effective_ce:
                scale = cap * effective_ce / t_norm
        per_teacher_scales[name] = scale

        for p in params:
            if p.grad is None:
                continue
            pid = id(p)
            scaled = p.grad.detach() * scale if scale != 1.0 else p.grad.detach().clone()
            if pid in total_teacher_grads:
                total_teacher_grads[pid].add_(scaled)
            else:
                total_teacher_grads[pid] = scaled
        _clear_grads(params)

    total_norm_before = sum(
        g.float().norm().item() ** 2 for g in total_teacher_grads.values()
    ) ** 0.5

    total_scale = 1.0
    if ce_loss.requires_grad:
        effective_ce_total = max(ce_norm, 1e-6)
        if total_norm_before > total_teacher_cap * effective_ce_total:
            total_scale = total_teacher_cap * effective_ce_total / total_norm_before
        for pid in total_teacher_grads:
            total_teacher_grads[pid].mul_(total_scale)

    total_norm_after = sum(
        g.float().norm().item() ** 2 for g in total_teacher_grads.values()
    ) ** 0.5

    for p in params:
        parts = []
        if id(p) in saved:
            parts.append(saved[id(p)])
        if id(p) in ce_grads:
            parts.append(ce_grads[id(p)])
        if id(p) in total_teacher_grads:
            parts.append(total_teacher_grads[id(p)])
        for g in parts:
            if p.grad is not None:
                p.grad.add_(g)
            else:
                p.grad = g.clone()

    ce_teacher_cosines = None
    pairwise_coherence = None
    if compute_coherence and per_teacher_grads_raw and ce_grads:
        if len(teacher_names) >= 1:
            ce_vec = _flat_grad_vec(ce_grads, params)
            ce_teacher_cosines = {}
            for name in teacher_names:
                t_vec = _flat_grad_vec(per_teacher_grads_raw[name], params)
                ce_teacher_cosines[name] = _cosine_sim(ce_vec, t_vec)

        if len(teacher_names) >= 2:
            teacher_vecs = {
                name: _flat_grad_vec(per_teacher_grads_raw[name], params)
                for name in teacher_names
            }
            pair_count = 0
            cos_sum = 0.0
            for ii in range(len(teacher_names)):
                for jj in range(ii + 1, len(teacher_names)):
                    cos_sum += _cosine_sim(
                        teacher_vecs[teacher_names[ii]],
                        teacher_vecs[teacher_names[jj]])
                    pair_count += 1
            pairwise_coherence = cos_sum / pair_count if pair_count > 0 else 0.0

    del per_teacher_grads_raw

    return GradientBudgetReport(
        ce_grad_norm=ce_norm,
        per_teacher_norms=per_teacher_norms,
        per_teacher_scales=per_teacher_scales,
        total_teacher_norm_before=total_norm_before,
        total_teacher_norm_after=total_norm_after,
        total_scale=total_scale,
        ce_teacher_cosines=ce_teacher_cosines,
        pairwise_coherence=pairwise_coherence,
    )
