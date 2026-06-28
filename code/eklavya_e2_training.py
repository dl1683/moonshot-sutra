"""Eklavya E2 — Multi-teacher KD training loop.

Six-phase curriculum:
  E2.0: Teacher feasibility profiling (no student update)
  E2.1: Projection-port warmup (500-1000 steps, freeze S0)
  E2.2: Low-conflict consensus KD (1500-3000 steps)
  E2.3: Semantic geometry landing (2000-4000 steps)
  E2.4: Disagreement-routed KD (6000-12000 steps)
  E2.5: Ownership and ablation (no teachers, measure retained gain)

Usage:
    python eklavya_e2_training.py \
        --student-checkpoint checkpoints/e1/e1_best.pt \
        --cache-dir eklavya_e2_cache \
        --output-dir checkpoints/e2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from s0_architecture import SutraS0
from eklavya_training import (
    EklavyaDataset, AlignProjection, overlap_pool,
)
from eklavya_e2_cache import (
    TeacherSpec, TEACHER_REGISTRY,
    E2CacheView,
    PositionRecord,
    SparseByteDist,
)
from eklavya_e2_router import (
    route_teachers, purify_byte_target, disagreement_jsd,
    RouterConfig, RouteResult, _dist_is_valid,
)
from eklavya_e2_losses import (
    MultiTeacherProjectionPorts,
    e2_topk_tail_kl, semantic_cosine_loss,
    apply_multi_teacher_gradient_budget,
)


# ---------------------------------------------------------------------------
# E2 Config
# ---------------------------------------------------------------------------

@dataclass
class E2Config:
    # Loss weights
    lambda_align_max: float = 0.05
    lambda_kl_max: float = 0.10
    lambda_semantic_max: float = 0.02
    lambda_calibration: float = 0.01
    kl_temperature: float = 2.0

    # Learning rates
    base_lr: float = 1e-5
    port_lr: float = 3e-4
    weight_decay_base: float = 0.05
    weight_decay_ports: float = 0.01

    # Curriculum phase durations
    port_warmup_steps: int = 750
    consensus_steps: int = 2000
    semantic_landing_steps: int = 3000
    disagreement_steps: int = 8000

    # Training
    batch_size: int = 4
    seq_len: int = 4096
    grad_accum: int = 2
    max_grad_norm: float = 1.0

    # Gradient budgets
    per_teacher_grad_cap: float = 0.10
    total_teacher_grad_cap: float = 0.30

    # Router
    router_alpha: float = 1.0
    router_beta: float = 0.5
    router_tau: float = 1.0
    router_mode: str = "oracle_gold"
    router_agreement_gamma: float = 0.5
    router_student_delta: float = 0.25
    purifier_mode: str = "arithmetic"

    # JSD thresholds for teacher admission
    jsd_low: float = 0.05
    jsd_high: float = 0.20

    # Paths
    checkpoint_dir: str = "checkpoints/e2"
    log_file: str = "logs/e2_train.jsonl"
    checkpoint_every: int = 1000
    log_every: int = 10

    cache_dir: str = "eklavya_e2_cache"
    data_dir: str = "data/shards_bytes_full"

    # Eval
    eval_every: int = 500
    eval_batches: int = 20

    # Resume
    resume_from: Optional[str] = None

    # Ablation config (controls training behavior, saved in checkpoints)
    ablation_id: str = "A2"
    teacher_include: Optional[list[str]] = None
    teacher_exclude: list[str] = field(default_factory=list)
    disable_router: bool = False
    shuffle_teacher_targets: bool = False
    shuffle_seed: int = 1234

    # Loss warmup
    align_warmup_steps: int = 1000
    kl_warmup_steps: int = 1500
    semantic_warmup_steps: int = 500

    # AMP dtype (match S0: bfloat16 by default, GradScaler disabled for bf16)
    dtype: str = "bfloat16"

    # CE-only mode: bypass phase freezing, train all params from step 0
    ce_only: bool = False

    # A7: disable gradient budgeting (teacher gradients flow uncapped)
    disable_gradient_budget: bool = False

    # A8: no phased admission (all teachers active from first student-updating phase)
    no_phased_admission: bool = False

    # Static weight mode when router is disabled (A5 variants)
    static_weight_mode: str = "uniform"
    static_weights: Optional[dict[str, float]] = None

    # BLD mode: single-teacher byte KL baseline (no E2 machinery)
    bld_mode: bool = False
    bld_kl_weight: float = 0.10


# ---------------------------------------------------------------------------
# Ablation config validation
# ---------------------------------------------------------------------------

_ABLATION_RULES: dict[str, dict] = {
    "A0": {
        "desc": "CE-only continuation (no teacher losses)",
        "required": {"ce_only"},
        "forbidden": {"teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets"},
    },
    "A1": {
        "desc": "Single-teacher (anchor only)",
        "required": {"teacher_include"},
        "forbidden": {"ce_only", "disable_router", "shuffle_teacher_targets"},
        "require_anchor_only": True,
    },
    "A2": {
        "desc": "Full E2 system (oracle upper bound)",
        "required": set(),
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets"},
        "require_router_mode": "oracle_gold",
    },
    "A3": {
        "desc": "Leave-one-out (best non-anchor)",
        "required": {"teacher_exclude"},
        "forbidden": {"ce_only", "teacher_include", "disable_router",
                      "shuffle_teacher_targets"},
    },
    "A4": {
        "desc": "No semantic teachers",
        "required": {"teacher_exclude"},
        "forbidden": {"ce_only", "teacher_include", "disable_router",
                      "shuffle_teacher_targets"},
        "require_semantic_excluded": True,
    },
    "A5": {
        "desc": "No router (uniform weights)",
        "required": {"disable_router"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "shuffle_teacher_targets"},
        "require_static_weight_mode": "uniform",
    },
    "A5a": {
        "desc": "No router (protocol prior weights)",
        "required": {"disable_router"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "shuffle_teacher_targets"},
        "require_static_weight_mode": "prior",
    },
    "A5b": {
        "desc": "No router (tuned static weights)",
        "required": {"disable_router"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "shuffle_teacher_targets"},
        "require_static_weight_mode": "custom",
    },
    "A5c": {
        "desc": "No router, best-2 teachers (X-Token comparison)",
        "required": {"disable_router", "teacher_include"},
        "forbidden": {"ce_only", "teacher_exclude",
                      "shuffle_teacher_targets"},
        "require_static_weight_mode": "prior",
        "require_teacher_count": 2,
    },
    "A6": {
        "desc": "Shuffled teacher targets (falsification)",
        "required": {"shuffle_teacher_targets"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router"},
    },
    "A7": {
        "desc": "No gradient budget (uncapped teacher gradients)",
        "required": {"disable_gradient_budget"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets"},
    },
    "A8": {
        "desc": "No phased admission (all teachers from step 0)",
        "required": {"no_phased_admission"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets",
                      "disable_gradient_budget"},
    },
    "BLD": {
        "desc": "Single-teacher byte KL baseline (no E2 machinery)",
        "required": {"bld_mode"},
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets",
                      "disable_gradient_budget", "no_phased_admission"},
    },
    "A9a": {
        "desc": "Gold-free router (entropy only)",
        "required": set(),
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets", "bld_mode"},
        "require_router_mode": "gold_free_entropy",
    },
    "A9b": {
        "desc": "Gold-free router (entropy + agreement)",
        "required": set(),
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets", "bld_mode"},
        "require_router_mode": "gold_free_agreement",
    },
    "A9c": {
        "desc": "Gold-free router (entropy + agreement + student JSD)",
        "required": set(),
        "forbidden": {"ce_only", "teacher_include", "teacher_exclude",
                      "disable_router", "shuffle_teacher_targets", "bld_mode"},
        "require_router_mode": "gold_free_student_jsd",
    },
}

_FLAG_ACTIVE: dict[str, object] = {
    "teacher_include": lambda cfg: cfg.teacher_include is not None,
    "teacher_exclude": lambda cfg: len(cfg.teacher_exclude) > 0,
    "disable_router": lambda cfg: cfg.disable_router,
    "shuffle_teacher_targets": lambda cfg: cfg.shuffle_teacher_targets,
    "ce_only": lambda cfg: cfg.ce_only,
    "disable_gradient_budget": lambda cfg: cfg.disable_gradient_budget,
    "no_phased_admission": lambda cfg: cfg.no_phased_admission,
    "bld_mode": lambda cfg: cfg.bld_mode,
}


def validate_ablation_config(cfg: E2Config) -> None:
    """Refuse to train if ablation_id doesn't match actual flags.

    Raises ValueError on mismatch. Unknown ablation IDs pass with a warning.
    """
    rules = _ABLATION_RULES.get(cfg.ablation_id)
    if rules is None:
        raise ValueError(
            f"Unknown ablation_id '{cfg.ablation_id}'. "
            f"Valid IDs: {sorted(_ABLATION_RULES.keys())}"
        )

    errors: list[str] = []

    if rules.get("require_anchor_only"):
        if (cfg.teacher_include is None
                or set(cfg.teacher_include) != {"t0_anchor_decoder"}):
            errors.append(
                f"{cfg.ablation_id} requires --teachers t0_anchor_decoder "
                "(exactly the anchor teacher, no others)"
            )

    if rules.get("require_semantic_excluded"):
        if set(cfg.teacher_exclude) != {"t3_semantic_embedding"}:
            errors.append(
                f"{cfg.ablation_id} requires --exclude-teachers to be exactly "
                "t3_semantic_embedding (no other teachers excluded)"
            )

    required_mode = rules.get("require_router_mode")
    if required_mode is not None:
        if cfg.router_mode != required_mode:
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) requires "
                f"--router-mode {required_mode} but got {cfg.router_mode}"
            )

    required_swm = rules.get("require_static_weight_mode")
    if required_swm is not None:
        if cfg.static_weight_mode != required_swm:
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) requires "
                f"--static-weight-mode {required_swm} but got "
                f"{cfg.static_weight_mode}"
            )
        if required_swm == "custom" and not cfg.static_weights:
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) requires "
                f"--static-weights to be set"
            )
        if required_swm == "custom" and cfg.static_weights:
            known = {t.name for t in TEACHER_REGISTRY}
            unknown = set(cfg.static_weights.keys()) - known
            if unknown:
                errors.append(
                    f"{cfg.ablation_id}: unknown teacher names in "
                    f"--static-weights: {unknown}"
                )
            vals = list(cfg.static_weights.values())
            if not all(isinstance(v, (int, float)) and math.isfinite(v)
                       and v >= 0 for v in vals):
                errors.append(
                    f"{cfg.ablation_id}: static weights must be finite "
                    f"and non-negative, got {cfg.static_weights}"
                )
            elif sum(vals) <= 0:
                errors.append(
                    f"{cfg.ablation_id}: static weights must have "
                    f"positive sum, got sum={sum(vals)}"
                )

    required_tc = rules.get("require_teacher_count")
    if required_tc is not None:
        actual = len(cfg.teacher_include) if cfg.teacher_include else 0
        if actual != required_tc:
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) requires exactly "
                f"{required_tc} teachers but got {actual}"
            )
        if cfg.teacher_include:
            known = {t.name for t in TEACHER_REGISTRY}
            unknown = set(cfg.teacher_include) - known
            if unknown:
                errors.append(
                    f"{cfg.ablation_id}: unknown teacher names in "
                    f"--teachers: {unknown}"
                )

    for flag in rules.get("required", set()):
        if not _FLAG_ACTIVE[flag](cfg):
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) requires "
                f"--{flag.replace('_', '-')} but it is not set"
            )

    for flag in rules.get("forbidden", set()):
        if _FLAG_ACTIVE[flag](cfg):
            errors.append(
                f"{cfg.ablation_id} ({rules['desc']}) forbids "
                f"--{flag.replace('_', '-')} but it is set"
            )

    known = {t.name for t in TEACHER_REGISTRY}
    if cfg.teacher_exclude:
        unknown_exc = set(cfg.teacher_exclude) - known
        if unknown_exc:
            errors.append(
                f"Unknown teacher names in --exclude-teachers: {unknown_exc}"
            )

    if errors:
        msg = (f"Ablation config validation failed for {cfg.ablation_id}:\n"
               + "\n".join(f"  - {e}" for e in errors))
        raise ValueError(msg)

    print(f"Ablation config OK: {cfg.ablation_id} ({rules['desc']})")


# ---------------------------------------------------------------------------
# Sigmoid ramp for loss scheduling
# ---------------------------------------------------------------------------

def sigmoid_ramp(step: int, warmup: int, steepness: float = 10.0) -> float:
    """Smooth sigmoid ramp from 0 to 1 centered at warmup/2."""
    if warmup <= 0:
        return 1.0
    x = steepness * (step / warmup - 0.5)
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# E2 Phase Management
# ---------------------------------------------------------------------------

class E2Phase:
    PORT_WARMUP = "E2.1_port_warmup"
    CONSENSUS = "E2.2_consensus"
    SEMANTIC = "E2.3_semantic"
    DISAGREEMENT = "E2.4_disagreement"
    OWNERSHIP = "E2.5_ownership"


def get_e2_phase(step: int, cfg: E2Config) -> str:
    t1 = cfg.port_warmup_steps
    t2 = t1 + cfg.consensus_steps
    t3 = t2 + cfg.semantic_landing_steps
    t4 = t3 + cfg.disagreement_steps

    if step < t1:
        return E2Phase.PORT_WARMUP
    elif step < t2:
        return E2Phase.CONSENSUS
    elif step < t3:
        return E2Phase.SEMANTIC
    elif step < t4:
        return E2Phase.DISAGREEMENT
    else:
        return E2Phase.OWNERSHIP


# ---------------------------------------------------------------------------
# Mmap-backed lookup wrappers (drop-in for dict[int, Record])
# ---------------------------------------------------------------------------

class _MmapPidLookup:
    """Dict-like {position_id: Record} backed by memory-mapped file."""

    def __init__(self, reader, pid_to_idx: dict[int, int]):
        self._reader = reader
        self._pid_to_idx = pid_to_idx

    def get(self, pid: int, default=None):
        idx = self._pid_to_idx.get(pid)
        if idx is None:
            return default
        rec = self._reader[idx]
        if hasattr(rec, "is_valid") and not rec.is_valid():
            return default
        return rec

    def __contains__(self, pid: int) -> bool:
        return pid in self._pid_to_idx

    def __len__(self) -> int:
        return len(self._pid_to_idx)


class _MmapLocLookup:
    """Dict-like {(shard_id, seq_offset): [PositionRecord]} backed by mmap."""

    def __init__(self, reader, loc_index: dict[tuple[int, int], list[int]]):
        self._reader = reader
        self._loc_index = loc_index

    def get(self, key, default=None):
        indices = self._loc_index.get(key)
        if indices is None:
            return default
        return [self._reader[i] for i in indices]

    def __contains__(self, key) -> bool:
        return key in self._loc_index

    def __getitem__(self, key):
        indices = self._loc_index.get(key)
        if indices is None:
            raise KeyError(key)
        return [self._reader[i] for i in indices]

    def __len__(self) -> int:
        return len(self._loc_index)


# ---------------------------------------------------------------------------
# E2 Trainer
# ---------------------------------------------------------------------------

class E2Trainer:
    def __init__(self, cfg: E2Config, student: SutraS0,
                 ports: MultiTeacherProjectionPorts,
                 cache, device: torch.device):
        self.cfg = cfg
        self.student = student
        self.ports = ports
        self.device = device
        self._cache_view = cache if isinstance(cache, E2CacheView) else None

        self.router_config = RouterConfig(
            mode=cfg.router_mode,
            alpha=cfg.router_alpha,
            beta=cfg.router_beta,
            tau=cfg.router_tau,
            agreement_gamma=cfg.router_agreement_gamma,
            student_delta=cfg.router_student_delta,
        )
        self._last_route_stats: dict = {}

        if self._cache_view is not None:
            self._init_from_view(cache)
        else:
            self.positions = cache["positions"]
            self.teacher_data = cache["teachers"]
            self.manifest = cache["manifest"]
            self._index_positions()
            self._index_teacher_records()

    def _init_from_view(self, view: E2CacheView):
        """Initialize from memory-mapped E2CacheView."""
        self.manifest = view.manifest
        self.positions = view._positions
        self.positions_by_loc = _MmapLocLookup(
            view._positions, view._loc_index)

        self.teacher_data = {}
        for name in view._teacher_names:
            self.teacher_data[name] = {
                "embedding_table": view.embedding_table(name),
            }

        self.teacher_kl_by_pid = {}
        self.teacher_align_by_pid = {}

        for name in view._teacher_names:
            kl_reader = view._kl_readers.get(name)
            kl_pid_idx = view._kl_pid_idx.get(name, {})
            align_reader = view._align_readers.get(name)
            align_pid_idx = view._align_pid_idx.get(name, {})

            if self.cfg.shuffle_teacher_targets and kl_reader and kl_pid_idx:
                rng = random.Random(self.cfg.shuffle_seed)
                pids = list(kl_pid_idx.keys())
                shuffled = pids[:]
                rng.shuffle(shuffled)
                shuffled_idx = {shuffled[i]: i for i in range(len(pids))}
                self.teacher_kl_by_pid[name] = _MmapPidLookup(
                    kl_reader, shuffled_idx)
            elif kl_reader:
                self.teacher_kl_by_pid[name] = _MmapPidLookup(
                    kl_reader, kl_pid_idx)

            if self.cfg.shuffle_teacher_targets and align_reader and align_pid_idx:
                rng = random.Random(self.cfg.shuffle_seed + 1)
                pids = list(align_pid_idx.keys())
                shuffled = pids[:]
                rng.shuffle(shuffled)
                shuffled_idx = {shuffled[i]: i for i in range(len(pids))}
                self.teacher_align_by_pid[name] = _MmapPidLookup(
                    align_reader, shuffled_idx)
            elif align_reader:
                self.teacher_align_by_pid[name] = _MmapPidLookup(
                    align_reader, align_pid_idx)

    def _index_positions(self):
        """Build lookup from shard/seq → position records."""
        self.positions_by_loc = {}
        for p in self.positions:
            key = (p.shard_id, p.seq_offset)
            self.positions_by_loc.setdefault(key, []).append(p)

    def _index_teacher_records(self):
        """Build per-teacher position_id → record lookups.

        When shuffle_teacher_targets is True, remap position_id keys so
        teacher signals come from random positions. This breaks spatial
        coherence while preserving the marginal teacher distribution
        (A6 ablation falsification test).
        """
        self.teacher_kl_by_pid = {}
        self.teacher_align_by_pid = {}
        for tname, td in self.teacher_data.items():
            kl_records = td.get("kl_records", [])
            align_records = td.get("align_records", [])

            if self.cfg.shuffle_teacher_targets and kl_records:
                rng = random.Random(self.cfg.shuffle_seed)
                kl_pids = [r.position_id for r in kl_records]
                shuffled_pids = kl_pids[:]
                rng.shuffle(shuffled_pids)
                self.teacher_kl_by_pid[tname] = {
                    orig_pid: rec
                    for orig_pid, rec in zip(shuffled_pids, kl_records)
                }
            else:
                self.teacher_kl_by_pid[tname] = {
                    r.position_id: r for r in kl_records
                }

            if self.cfg.shuffle_teacher_targets and align_records:
                rng = random.Random(self.cfg.shuffle_seed + 1)
                align_pids = [r.position_id for r in align_records]
                shuffled_pids = align_pids[:]
                rng.shuffle(shuffled_pids)
                self.teacher_align_by_pid[tname] = {
                    orig_pid: rec
                    for orig_pid, rec in zip(shuffled_pids, align_records)
                }
            else:
                self.teacher_align_by_pid[tname] = {
                    r.position_id: r for r in align_records
                }

    def stage_embeddings(self, phase: str):
        """Move active teachers' embedding tables to GPU, inactive to CPU."""
        active_names = {s.name for s in self.get_active_teachers(phase)}
        for tname, td in self.teacher_data.items():
            emb = td.get("embedding_table")
            if emb is None:
                continue
            if tname in active_names and emb.device != self.device:
                td["embedding_table"] = emb.to(self.device)
            elif tname not in active_names and emb.device != torch.device("cpu"):
                td["embedding_table"] = emb.cpu()

    def compute_teacher_losses(
        self, logits: torch.Tensor, patch_states: torch.Tensor,
        shard_ids: torch.Tensor, seq_starts: torch.Tensor,
        phase: str, step: int,
    ) -> dict[str, torch.Tensor]:
        """Compute all teacher losses for the current batch.

        Returns dict mapping loss_name → weighted scalar tensor.
        """
        cfg = self.cfg
        self._last_route_stats = {}
        weights = self.get_loss_weights(step, phase)
        active = self.get_active_teachers(phase)
        P = self.student.cfg.patch_size
        B = logits.shape[0]
        Nm1 = logits.shape[1]

        teacher_losses: dict[str, torch.Tensor] = {}

        batch_positions: list[tuple[int, PositionRecord]] = []
        for b in range(B):
            key = (int(shard_ids[b]), int(seq_starts[b]))
            for pos in self.positions_by_loc.get(key, []):
                batch_positions.append((b, pos))

        if not batch_positions:
            return teacher_losses

        # --- Alignment losses (per-teacher) ---
        if weights["align"] > 0:
            for spec in active:
                if not spec.has_align or spec.name not in self.ports.align_ports:
                    continue
                align_idx = self.teacher_align_by_pid.get(spec.name, {})
                td = self.teacher_data.get(spec.name, {})
                emb_table = td.get("embedding_table")
                if not align_idx or emb_table is None:
                    continue

                losses_list = []
                sampled = random.sample(batch_positions,
                                        min(32, len(batch_positions)))
                for b, pos in sampled:
                    arec = align_idx.get(pos.position_id)
                    if arec is None:
                        continue
                    if arec.byte_start + arec.byte_len > patch_states.shape[1] * P:
                        continue
                    if arec.token_id >= emb_table.shape[0]:
                        continue
                    student_span = overlap_pool(
                        patch_states[b], arec.byte_start,
                        arec.byte_start + arec.byte_len, P)
                    if student_span is None:
                        continue
                    z_s = self.ports.get_align_projection(
                        spec.name, student_span.unsqueeze(0))
                    z_s = F.layer_norm(z_s.squeeze(0), (z_s.shape[-1],))
                    teacher_emb = emb_table[arec.token_id].to(dtype=z_s.dtype)
                    z_t = F.layer_norm(teacher_emb, (teacher_emb.shape[-1],))
                    losses_list.append(F.mse_loss(z_s, z_t))

                if losses_list:
                    teacher_losses[f"align_{spec.name}"] = (
                        weights["align"] * torch.stack(losses_list).mean())

        # --- KL + calibration losses (purified from all teachers) ---
        if weights["kl"] > 0:
            kl_losses = []
            cal_losses = []
            route_jsds = []
            route_entropies = []
            route_weight_sums: dict[str, float] = {}
            compute_cal = weights["calibration"] > 0
            sampled = random.sample(batch_positions,
                                    min(16, len(batch_positions)))
            for b, pos in sampled:
                teacher_dists: dict[str, SparseByteDist] = {}
                for spec in active:
                    if not spec.has_kl:
                        continue
                    kl_idx = self.teacher_kl_by_pid.get(spec.name, {})
                    rec = kl_idx.get(pos.position_id)
                    if rec is None:
                        continue
                    teacher_dists[spec.name] = SparseByteDist(
                        top_bytes=rec.top_bytes,
                        top_probs=rec.top_probs.astype(np.float32),
                        tail_prob=rec.tail_prob,
                    )

                if not teacher_dists:
                    continue

                logit_idx = pos.patch_idx - 1
                if logit_idx < 0 or logit_idx >= Nm1:
                    continue
                student_logit = logits[b, logit_idx, 0]

                priors = {s.name: s.prior
                          for s in active if s.name in teacher_dists}
                ptotal = sum(priors.values())
                if ptotal > 0:
                    priors = {k: v / ptotal for k, v in priors.items()}

                s_probs_np = None
                s_ent_float = None
                if self.router_config.mode == "gold_free_student_jsd":
                    s_p = F.softmax(student_logit.detach().float(), dim=-1)
                    s_probs_np = s_p.cpu().numpy().astype(np.float64)
                    s_ent_float = -float(
                        np.sum(s_probs_np * np.log(np.maximum(s_probs_np, 1e-12))))

                if cfg.disable_router:
                    teacher_dists = {k: v for k, v in teacher_dists.items()
                                     if _dist_is_valid(v)}
                    if not teacher_dists:
                        continue
                    n_t = len(teacher_dists)
                    if cfg.static_weight_mode == "prior":
                        ptotal = sum(priors.get(k, 1.0 / n_t) for k in teacher_dists)
                        static_w = {k: priors.get(k, 1.0 / n_t) / ptotal
                                    for k in teacher_dists}
                    elif cfg.static_weight_mode == "custom" and cfg.static_weights:
                        wtotal = sum(cfg.static_weights.get(k, 1.0 / n_t)
                                     for k in teacher_dists)
                        static_w = {k: cfg.static_weights.get(k, 1.0 / n_t) / wtotal
                                    for k in teacher_dists}
                    else:
                        static_w = {k: 1.0 / n_t for k in teacher_dists}
                    static_jsd = disagreement_jsd(teacher_dists)
                    r_ent = -sum(w * math.log(max(w, 1e-12))
                                for w in static_w.values())
                    route_result = RouteResult(
                        weights=static_w, jsd=static_jsd,
                        route_entropy=r_ent)
                    purified = purify_byte_target(
                        teacher_dists, route_result, mode="arithmetic")
                else:
                    gold = pos.gold_byte if not self.router_config.mode.startswith("gold_free") else None
                    route_result = route_teachers(
                        teacher_dists, gold, priors,
                        self.router_config,
                        student_probs=s_probs_np,
                        student_entropy=s_ent_float)
                    purified = purify_byte_target(
                        teacher_dists, route_result, mode=cfg.purifier_mode)
                if purified is None:
                    continue

                raw_jsd = disagreement_jsd(teacher_dists)
                if phase == E2Phase.CONSENSUS and raw_jsd > cfg.jsd_low:
                    continue
                if phase == E2Phase.SEMANTIC and raw_jsd > cfg.jsd_high:
                    continue

                route_jsds.append(route_result.jsd)
                route_entropies.append(route_result.route_entropy)
                for tname, tw in route_result.weights.items():
                    route_weight_sums[tname] = route_weight_sums.get(tname, 0) + tw

                loss = e2_topk_tail_kl(student_logit, purified,
                                       cfg.kl_temperature)
                if torch.isfinite(loss):
                    kl_losses.append(loss)

                if compute_cal:
                    student_p = F.softmax(student_logit, dim=-1)
                    student_H = -(student_p * torch.log(student_p + 1e-10)).sum()
                    full_q = np.zeros(256, dtype=np.float32)
                    full_q[purified.top_bytes] = purified.top_probs
                    n_tail = max(1, 256 - len(purified.top_bytes))
                    tail_mask = np.ones(256, dtype=bool)
                    tail_mask[purified.top_bytes] = False
                    full_q[tail_mask] = purified.tail_prob / n_tail
                    target_H = -float(np.sum(full_q * np.log(full_q + 1e-10)))
                    if math.isfinite(target_H) and torch.isfinite(student_H):
                        cal_losses.append(
                            (student_H - target_H) ** 2)

            if kl_losses:
                teacher_losses["kl_purified"] = (
                    weights["kl"] * torch.stack(kl_losses).mean())

            if cal_losses:
                teacher_losses["calibration"] = (
                    weights["calibration"] * torch.stack(cal_losses).mean())

            if route_jsds:
                n_routes = len(route_jsds)
                avg_weights = {k: v / n_routes
                               for k, v in route_weight_sums.items()}
                self._last_route_stats = {
                    "mean_jsd": sum(route_jsds) / n_routes,
                    "mean_route_entropy": sum(route_entropies) / n_routes,
                    "n_routed": n_routes,
                    "avg_teacher_weights": avg_weights,
                }
            else:
                self._last_route_stats = {}

        # --- Semantic losses (per-teacher) ---
        if weights["semantic"] > 0:
            for spec in active:
                if not spec.has_semantic:
                    continue
                if spec.name not in self.ports.semantic_ports:
                    continue
                td = self.teacher_data.get(spec.name, {})
                emb_table = td.get("embedding_table")
                align_idx = self.teacher_align_by_pid.get(spec.name, {})
                if emb_table is None or not align_idx:
                    continue

                sem_losses = []
                sem_sampled = random.sample(batch_positions,
                                            min(32, len(batch_positions)))
                for b, pos in sem_sampled:
                    arec = align_idx.get(pos.position_id)
                    if arec is None:
                        continue
                    if arec.token_id >= emb_table.shape[0]:
                        continue
                    student_span = overlap_pool(
                        patch_states[b], arec.byte_start,
                        arec.byte_start + arec.byte_len, P)
                    if student_span is None:
                        continue
                    z_s = self.ports.get_semantic_projection(
                        spec.name, student_span.unsqueeze(0))
                    teacher_emb = emb_table[arec.token_id].to(dtype=z_s.dtype)
                    if teacher_emb.norm().item() < 1e-8:
                        continue
                    teacher_emb = F.normalize(
                        teacher_emb.unsqueeze(0), dim=-1)
                    sem_losses.append(
                        semantic_cosine_loss(z_s, teacher_emb))

                if sem_losses:
                    teacher_losses[f"semantic_{spec.name}"] = (
                        weights["semantic"] * torch.stack(sem_losses).mean())

        return teacher_losses

    def compute_bld_kl_loss(
        self, logits: torch.Tensor,
        shard_ids: torch.Tensor, seq_starts: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """BLD-mode: raw top-k tail KL from anchor teacher only.

        No routing, no purification, no alignment, no semantic loss.
        Returns a scalar KL loss or None if no anchor data found.
        """
        anchor_name = None
        specs = self.manifest.get("teacher_specs", TEACHER_REGISTRY)
        for spec in specs:
            if isinstance(spec, TeacherSpec) and spec.role.name == "ANCHOR":
                anchor_name = spec.name
                break
        if anchor_name is None:
            for name in self.teacher_kl_by_pid:
                anchor_name = name
                break
        if anchor_name is None:
            return None
        kl_idx = self.teacher_kl_by_pid.get(anchor_name, {})
        if not kl_idx:
            return None

        B, Nm1 = logits.shape[0], logits.shape[1]
        batch_positions: list[tuple[int, PositionRecord]] = []
        for b in range(B):
            key = (int(shard_ids[b]), int(seq_starts[b]))
            for pos in self.positions_by_loc.get(key, []):
                batch_positions.append((b, pos))

        if not batch_positions:
            return None

        sampled = random.sample(batch_positions,
                                min(16, len(batch_positions)))
        kl_losses = []
        for b, pos in sampled:
            rec = kl_idx.get(pos.position_id)
            if rec is None:
                continue
            logit_idx = pos.patch_idx - 1
            if logit_idx < 0 or logit_idx >= Nm1:
                continue
            student_logit = logits[b, logit_idx, 0]
            dist = SparseByteDist(
                top_bytes=rec.top_bytes,
                top_probs=rec.top_probs.astype(np.float32),
                tail_prob=rec.tail_prob,
            )
            if not _dist_is_valid(dist):
                continue
            loss = e2_topk_tail_kl(student_logit, dist,
                                    self.cfg.kl_temperature)
            if torch.isfinite(loss):
                kl_losses.append(loss)

        if kl_losses:
            return torch.stack(kl_losses).mean()
        return None

    def get_active_teachers(self, phase: str) -> list[TeacherSpec]:
        """Which teachers are active in each phase."""
        if self.cfg.ce_only or self.cfg.bld_mode:
            return []
        all_specs = self.manifest.get("teacher_specs", [])
        if isinstance(all_specs, list) and all_specs:
            if isinstance(all_specs[0], TeacherSpec):
                specs = all_specs
            else:
                specs = TEACHER_REGISTRY
        else:
            specs = TEACHER_REGISTRY

        if self.cfg.no_phased_admission:
            if phase == E2Phase.PORT_WARMUP:
                active = [s for s in specs if s.role.name == "ANCHOR"]
            else:
                active = [s for s in specs if s.name in self.teacher_data]
        elif phase == E2Phase.PORT_WARMUP:
            active = [s for s in specs if s.role.name == "ANCHOR"]
        elif phase == E2Phase.CONSENSUS:
            active = [s for s in specs
                      if s.role.name in ("ANCHOR", "CONTROL") and s.has_kl]
        elif phase == E2Phase.SEMANTIC:
            active = [s for s in specs
                      if s.role.name in ("ANCHOR", "CONTROL", "SEMANTIC")]
        elif phase == E2Phase.DISAGREEMENT:
            active = [s for s in specs if s.name in self.teacher_data]
        else:
            active = []

        if self.cfg.teacher_include is not None:
            active = [s for s in active if s.name in self.cfg.teacher_include]
        if self.cfg.teacher_exclude:
            active = [s for s in active
                      if s.name not in self.cfg.teacher_exclude]
        return active

    def get_loss_weights(self, step: int, phase: str) -> dict:
        """Compute current loss weights with sigmoid ramp."""
        phase_start = self._phase_start(phase)
        local_step = step - phase_start

        w = {
            "align": self.cfg.lambda_align_max * sigmoid_ramp(
                step, self.cfg.align_warmup_steps),
            "kl": 0.0,
            "semantic": 0.0,
            "calibration": 0.0,
        }

        if phase in (E2Phase.CONSENSUS, E2Phase.SEMANTIC, E2Phase.DISAGREEMENT):
            w["kl"] = self.cfg.lambda_kl_max * sigmoid_ramp(
                local_step, self.cfg.kl_warmup_steps)

        if phase in (E2Phase.SEMANTIC, E2Phase.DISAGREEMENT):
            w["semantic"] = self.cfg.lambda_semantic_max * sigmoid_ramp(
                local_step, self.cfg.semantic_warmup_steps)
            w["calibration"] = self.cfg.lambda_calibration

        return w

    def _phase_start(self, phase: str) -> int:
        if phase == E2Phase.PORT_WARMUP:
            return 0
        elif phase == E2Phase.CONSENSUS:
            return self.cfg.port_warmup_steps
        elif phase == E2Phase.SEMANTIC:
            return self.cfg.port_warmup_steps + self.cfg.consensus_steps
        elif phase == E2Phase.DISAGREEMENT:
            return (self.cfg.port_warmup_steps + self.cfg.consensus_steps
                    + self.cfg.semantic_landing_steps)
        return 0

    def configure_freeze(self, phase: str):
        """Set requires_grad based on phase.

        In CE-only mode, all student params are trainable from step 0
        and ports are disabled (no teacher losses).
        """
        if self.cfg.ce_only or self.cfg.bld_mode:
            for p in self.student.parameters():
                p.requires_grad = True
            for p in self.ports.parameters():
                p.requires_grad = False
            return

        if phase == E2Phase.PORT_WARMUP:
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.ports.parameters():
                p.requires_grad = True

        elif phase == E2Phase.CONSENSUS:
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.student.encoder.parameters():
                p.requires_grad = True
            for p in self.ports.parameters():
                p.requires_grad = True

        elif phase in (E2Phase.SEMANTIC, E2Phase.DISAGREEMENT):
            for p in self.student.parameters():
                p.requires_grad = True
            for p in self.ports.parameters():
                p.requires_grad = True

        elif phase == E2Phase.OWNERSHIP:
            for p in self.student.parameters():
                p.requires_grad = False
            for p in self.ports.parameters():
                p.requires_grad = False

    def build_optimizer(self) -> torch.optim.AdamW:
        """Build optimizer with separate LR groups for student vs ports."""
        port_params = list(self.ports.parameters())
        port_ids = {id(p) for p in port_params}

        base_params = [p for p in self.student.parameters()
                       if p.requires_grad and id(p) not in port_ids]

        groups = []
        if base_params:
            groups.append({
                "params": base_params,
                "lr": self.cfg.base_lr,
                "weight_decay": self.cfg.weight_decay_base,
            })

        trainable_ports = [p for p in port_params if p.requires_grad]
        if trainable_ports:
            groups.append({
                "params": trainable_ports,
                "lr": self.cfg.port_lr,
                "weight_decay": self.cfg.weight_decay_ports,
            })

        return torch.optim.AdamW(groups, betas=(0.9, 0.95))

    def total_steps(self) -> int:
        return (self.cfg.port_warmup_steps + self.cfg.consensus_steps
                + self.cfg.semantic_landing_steps + self.cfg.disagreement_steps)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_e2(student: SutraS0, eval_loader: DataLoader,
                device: torch.device, cfg: E2Config) -> dict:
    """Evaluate student CE on held-out shards."""
    student.eval()
    total_loss = 0.0
    total_tokens = 0
    P = student.cfg.patch_size
    amp_dtype = getattr(torch, cfg.dtype)

    for i, batch in enumerate(eval_loader):
        if cfg.eval_batches > 0 and i >= cfg.eval_batches:
            break
        byte_ids, _, _ = batch
        byte_ids = byte_ids.to(device)
        B, T = byte_ids.shape
        N = T // P

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = student(byte_ids, return_aux=False)
            logits = out["logits"]
            targets = byte_ids.reshape(B, N, P)[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, 256),
                                   targets.reshape(-1))

        predicted = B * (T - P)
        total_loss += loss.item() * predicted
        total_tokens += predicted

    student.train()
    if total_tokens == 0:
        print("  WARNING: eval produced 0 tokens (empty loader or all batches dropped)")
        return {"eval_loss": float("inf"), "eval_bpb": float("inf")}
    avg_loss = total_loss / total_tokens
    return {
        "eval_loss": avg_loss,
        "eval_bpb": avg_loss / math.log(2),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def train_e2(cfg: E2Config, student_ckpt_path: str, cache_dir: str):
    """Main E2 training loop."""
    validate_ablation_config(cfg)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(student_ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    student = SutraS0(model_cfg).to(device)
    student.load_state_dict(ckpt["model"])
    print(f"Loaded student from step {ckpt.get('step', '?')}")

    cache_view = E2CacheView(cache_dir)
    try:
        cache_errors = cache_view.validate(data_dir=cfg.data_dir)
        if cache_errors:
            for e in cache_errors:
                print(f"  CACHE ERROR: {e}")
            raise ValueError(
                f"E2 cache validation failed with {len(cache_errors)} error(s)")
        print("Cache validation OK")
        _train_e2_inner(cfg, student, model_cfg, cache_view, device, ckpt)
    finally:
        cache_view.close()


def _train_e2_inner(cfg: E2Config, student: SutraS0, model_cfg,
                     cache_view: E2CacheView, device: torch.device, ckpt: dict):
    """Inner training loop — separated for resource cleanup."""
    print(f"Cache: {cache_view.manifest['n_positions']} positions, "
          f"{cache_view.manifest['teacher_count']} teachers (mmap-backed)")

    specs = cache_view.manifest["teacher_specs"]

    cache_prov = cache_view.manifest.get("provenance", {})
    cache_seq_len = cache_prov.get("seq_len")
    if cache_seq_len is not None and cache_seq_len != cfg.seq_len:
        raise ValueError(
            f"Cache was built with seq_len={cache_seq_len} but training "
            f"config uses seq_len={cfg.seq_len}. Position lookups will fail."
        )

    ports = MultiTeacherProjectionPorts(model_cfg.d_model, specs).to(device)

    if "align_proj" in ckpt:
        e1_proj = AlignProjection(model_cfg.d_model, 2048)
        e1_proj.load_state_dict(ckpt["align_proj"])
        ports.warm_start_from_e1(e1_proj)
        print("Warm-started anchor port from E1 AlignProjection")

    trainer = E2Trainer(cfg, student, ports, cache_view, device)

    total = trainer.total_steps()
    os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    amp_dtype = getattr(torch, cfg.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16)) if device.type == "cuda" else None
    step = 0
    current_phase = None
    optimizer = None
    best_eval_bpb = float("inf")
    log_fh = open(cfg.log_file, "a")

    print(f"E2 training: {total} steps across 4 phases")
    print(f"  E2.1 port warmup: {cfg.port_warmup_steps}")
    print(f"  E2.2 consensus:   {cfg.consensus_steps}")
    print(f"  E2.3 semantic:    {cfg.semantic_landing_steps}")
    print(f"  E2.4 disagreement:{cfg.disagreement_steps}")

    all_shards = sorted(Path(cfg.data_dir).glob("*.bin"))
    cache_shard_range = cache_view.manifest.get("shard_range", [0, len(all_shards)])
    cache_start, cache_end = cache_shard_range[0], cache_shard_range[1]
    n_cached = cache_end - cache_start
    if n_cached < 2:
        raise ValueError(
            f"Cache covers only {n_cached} shards ({cache_start}-{cache_end}). "
            f"Need at least 2 (1 train + 1 eval)."
        )
    n_eval = min(2, max(1, n_cached // 10))
    train_range = (cache_start, cache_end - n_eval)
    print(f"  Cache covers shards [{cache_start}, {cache_end}), "
          f"train [{train_range[0]}, {train_range[1]}), "
          f"eval [{train_range[1]}, {cache_end})")

    train_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len,
                                   model_cfg.patch_size, shard_range=train_range)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True,
                              drop_last=True, persistent_workers=True,
                              prefetch_factor=4)

    eval_range = (train_range[1], cache_end)
    eval_dataset = EklavyaDataset(cfg.data_dir, cfg.seq_len,
                                  model_cfg.patch_size, shard_range=eval_range)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=1, pin_memory=True,
                             drop_last=True)

    # Resume from checkpoint
    if cfg.resume_from:
        resume_ckpt = torch.load(cfg.resume_from, map_location="cpu", weights_only=False)
        student.load_state_dict(resume_ckpt["model"])
        if "ports" in resume_ckpt:
            ports.load_state_dict(resume_ckpt["ports"])
        step = resume_ckpt["step"] + 1
        if "best_eval_bpb" in resume_ckpt:
            best_eval_bpb = resume_ckpt["best_eval_bpb"]
        resumed_phase = resume_ckpt.get("phase")
        if resumed_phase is not None:
            current_phase = resumed_phase
            trainer.configure_freeze(current_phase)
            trainer.stage_embeddings(current_phase)
            optimizer = trainer.build_optimizer()
            if "optimizer" in resume_ckpt and resume_ckpt["optimizer"] is not None:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
        if scaler is not None and "scaler" in resume_ckpt and resume_ckpt["scaler"] is not None:
            scaler.load_state_dict(resume_ckpt["scaler"])
        if "rng_state" in resume_ckpt:
            torch.set_rng_state(resume_ckpt["rng_state"].cpu())
            if device.type == "cuda" and "cuda_rng_state" in resume_ckpt:
                torch.cuda.set_rng_state(resume_ckpt["cuda_rng_state"].cpu())
        if "py_rng_state" in resume_ckpt:
            random.setstate(resume_ckpt["py_rng_state"])
        if "np_rng_state" in resume_ckpt:
            np.random.set_state(resume_ckpt["np_rng_state"])
        print(f"Resumed from step {step} phase {current_phase} "
              f"(best eval BPB: {best_eval_bpb:.3f})")

    t0 = time.time()
    data_iter = iter(train_loader)
    consecutive_ce_only = 0
    CE_ONLY_FAIL_THRESHOLD = 200
    warmup_signal_steps = 0
    warmup_total_steps = 0

    if cfg.ce_only and current_phase is None:
        current_phase = "CE_ONLY"
        trainer.configure_freeze(current_phase)
        optimizer = trainer.build_optimizer()
        print(f"\n[Step {step}] CE-only mode — all student params trainable, no teachers")

    if cfg.bld_mode and current_phase is None:
        current_phase = "BLD"
        trainer.configure_freeze(current_phase)
        optimizer = trainer.build_optimizer()
        print(f"\n[Step {step}] BLD mode — all student params trainable, "
              f"single-teacher byte KL (weight={cfg.bld_kl_weight})")

    while step < total:
        if not cfg.ce_only and not cfg.bld_mode:
            phase = get_e2_phase(step, cfg)

            if phase == E2Phase.OWNERSHIP:
                print(f"\n[Step {step}] Phase E2.5: ownership/ablation — stopping training")
                break

            if phase != current_phase:
                if optimizer is not None:
                    if step % cfg.grad_accum != 0:
                        all_params = list(student.parameters()) + list(ports.parameters())
                        has_grad = any(p.grad is not None
                                       for p in all_params if p.requires_grad)
                        if has_grad:
                            if scaler is not None:
                                scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                            if scaler is not None:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                current_phase = phase
                trainer.configure_freeze(phase)
                trainer.stage_embeddings(phase)
                optimizer = trainer.build_optimizer()
                active = trainer.get_active_teachers(phase)
                print(f"\n[Step {step}] Phase {phase}")
                print(f"  Active teachers: {[t.name for t in active]}")
                if (current_phase == E2Phase.CONSENSUS
                        and warmup_total_steps > 0):
                    ratio = warmup_signal_steps / warmup_total_steps
                    print(f"  Warmup signal coverage: "
                          f"{warmup_signal_steps}/{warmup_total_steps} "
                          f"({ratio:.0%})")
                    if ratio < 0.1:
                        print(f"  WARNING: <10% of warmup steps had teacher "
                              f"signal — projection ports may be untrained")
        else:
            phase = current_phase

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        byte_ids, shard_ids, seq_starts = batch
        byte_ids = byte_ids.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = student(byte_ids, return_aux=False)
            logits = out["logits"]
            patch_states = out["patch_states"]
            B_out, Nm1, Pp, V = logits.shape
            N = Nm1 + 1
            targets = byte_ids.reshape(B_out, N, Pp)[:, 1:]
            ce_loss = F.cross_entropy(logits.reshape(-1, V),
                                       targets.reshape(-1))

            teacher_losses = trainer.compute_teacher_losses(
                logits, patch_states, shard_ids, seq_starts, phase, step)

            bld_kl_loss = None
            if cfg.bld_mode:
                bld_kl_loss = trainer.compute_bld_kl_loss(
                    logits, shard_ids, seq_starts)

        grad_report = None
        if teacher_losses:
            teacher_losses = {
                k: v for k, v in teacher_losses.items()
                if torch.isfinite(v)
            }
        if bld_kl_loss is not None:
            total_loss = (ce_loss + cfg.bld_kl_weight * bld_kl_loss) / cfg.grad_accum
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        elif teacher_losses and not cfg.disable_gradient_budget:
            active_specs = trainer.get_active_teachers(phase)
            per_caps = {}
            for lname in teacher_losses:
                for spec in active_specs:
                    if spec.name in lname:
                        per_caps[lname] = spec.per_teacher_grad_cap
                        break
                else:
                    per_caps[lname] = cfg.per_teacher_grad_cap

            grad_report = apply_multi_teacher_gradient_budget(
                list(student.parameters()) + list(ports.parameters()),
                ce_loss / cfg.grad_accum,
                {k: v / cfg.grad_accum for k, v in teacher_losses.items()},
                per_teacher_cap=per_caps,
                total_teacher_cap=cfg.total_teacher_grad_cap,
                scaler=scaler,
            )
        elif teacher_losses and cfg.disable_gradient_budget:
            total_loss = ce_loss / cfg.grad_accum + sum(
                v / cfg.grad_accum for v in teacher_losses.values())
            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
        else:
            scaled_ce = ce_loss / cfg.grad_accum
            if scaled_ce.requires_grad:
                if scaler is not None:
                    scaler.scale(scaled_ce).backward()
                else:
                    scaled_ce.backward()

        if phase == E2Phase.PORT_WARMUP:
            warmup_total_steps += 1
            if teacher_losses:
                warmup_signal_steps += 1

        expects_teachers = (not cfg.ce_only and not cfg.bld_mode
                            and phase != E2Phase.PORT_WARMUP)
        if expects_teachers and not teacher_losses:
            consecutive_ce_only += 1
            if consecutive_ce_only >= CE_ONLY_FAIL_THRESHOLD:
                raise RuntimeError(
                    f"E2 received NO teacher signal for {consecutive_ce_only} "
                    f"consecutive steps (phase={phase}, step={step}). "
                    f"Likely cache/seq_len mismatch — aborting."
                )
        else:
            consecutive_ce_only = 0

        if not math.isfinite(ce_loss.item()):
            fail_entry = {"step": step, "phase": str(phase),
                          "HARD_FAIL": "non-finite CE loss",
                          "ce_loss": ce_loss.item()}
            log_fh.write(json.dumps(fail_entry) + "\n")
            log_fh.flush()
            log_fh.close()
            raise RuntimeError(
                f"E2 HARD FAIL: non-finite CE loss at step {step} "
                f"(phase={phase})")

        if (step + 1) % cfg.grad_accum == 0:
            all_params = list(student.parameters()) + list(ports.parameters())
            has_grad = any(p.grad is not None
                          for p in all_params if p.requires_grad)
            if has_grad:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    all_params, cfg.max_grad_norm)
                if not math.isfinite(grad_norm.item()):
                    fail_entry = {"step": step, "phase": str(phase),
                                  "HARD_FAIL": "non-finite grad_norm",
                                  "grad_norm": grad_norm.item()}
                    log_fh.write(json.dumps(fail_entry) + "\n")
                    log_fh.flush()
                    log_fh.close()
                    raise RuntimeError(
                        f"E2 HARD FAIL: non-finite grad_norm at step {step} "
                        f"(phase={phase})")
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            optimizer.zero_grad()

        if step % cfg.log_every == 0:
            elapsed = time.time() - t0
            _ln2 = math.log(2)
            tl_nats = {k: v.item() for k, v in teacher_losses.items()}
            tl_bits = {k: v / _ln2 for k, v in tl_nats.items()}
            entry = {
                "step": step,
                "phase": phase,
                "ablation_id": cfg.ablation_id,
                "ce_loss": ce_loss.item(),
                "bpb": ce_loss.item() / _ln2,
                "teacher_losses_nats": tl_nats,
                "teacher_losses_bits": tl_bits,
                "elapsed": round(elapsed, 1),
                "lr": optimizer.param_groups[0]["lr"] if optimizer else 0,
            }
            if trainer._last_route_stats:
                entry["route_stats"] = trainer._last_route_stats
            if grad_report is not None:
                entry["grad_budget"] = {
                    "ce_grad_norm": round(grad_report.ce_grad_norm, 6),
                    "total_teacher_before": round(grad_report.total_teacher_norm_before, 6),
                    "total_teacher_after": round(grad_report.total_teacher_norm_after, 6),
                    "total_scale": round(grad_report.total_scale, 6),
                    "per_teacher_scales": {k: round(v, 4) for k, v in grad_report.per_teacher_scales.items()},
                }
            elif cfg.disable_gradient_budget and teacher_losses:
                entry["grad_budget"] = {"enabled": False}
            if bld_kl_loss is not None:
                entry["bld_kl_loss"] = bld_kl_loss.item()
                entry["bld_kl_bits"] = bld_kl_loss.item() / _ln2
            log_fh.write(json.dumps(entry) + "\n")
            log_fh.flush()

            if step % (cfg.log_every * 10) == 0:
                bpb = ce_loss.item() / _ln2
                t_str = " ".join(f"{k}={v:.4f}"
                                 for k, v in tl_bits.items())
                bld_str = ""
                if bld_kl_loss is not None:
                    bld_str = f" | BLD_KL {bld_kl_loss.item() / _ln2:.4f}"
                print(f"  step {step:>6d} | CE {ce_loss.item():.4f} | "
                      f"BPB {bpb:.3f} | {t_str} (bits){bld_str} | {elapsed:.0f}s")

        if step > 0 and step % cfg.eval_every == 0:
            eval_metrics = evaluate_e2(student, eval_loader, device, cfg)
            eval_entry = {"step": step, "phase": phase, **eval_metrics}
            log_fh.write(json.dumps(eval_entry) + "\n")
            log_fh.flush()
            print(f"  EVAL step {step}: loss {eval_metrics['eval_loss']:.4f} "
                  f"bpb {eval_metrics['eval_bpb']:.3f}")

            if eval_metrics["eval_bpb"] < best_eval_bpb:
                best_eval_bpb = eval_metrics["eval_bpb"]
                best_path = os.path.join(cfg.checkpoint_dir, "e2_best.pt")
                best_dict = {
                    "step": step,
                    "phase": phase,
                    "model": student.state_dict(),
                    "model_cfg": model_cfg,
                    "ports": ports.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "rng_state": torch.get_rng_state(),
                    "py_rng_state": random.getstate(),
                    "np_rng_state": np.random.get_state(),
                    "config": cfg.__dict__,
                    "best_eval_bpb": best_eval_bpb,
                }
                if device.type == "cuda":
                    best_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
                torch.save(best_dict, best_path)
                print(f"  New best eval BPB {best_eval_bpb:.3f} — saved {best_path}")

        accum_aligned = (step + 1) % cfg.grad_accum == 0
        if (accum_aligned and step >= cfg.checkpoint_every
                and step % cfg.checkpoint_every < cfg.grad_accum):
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"e2_step{step}.pt")
            save_dict = {
                "step": step,
                "phase": phase,
                "model": student.state_dict(),
                "model_cfg": model_cfg,
                "ports": ports.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scaler": scaler.state_dict() if scaler else None,
                "rng_state": torch.get_rng_state(),
                "py_rng_state": random.getstate(),
                "np_rng_state": np.random.get_state(),
                "config": cfg.__dict__,
                "best_eval_bpb": best_eval_bpb,
            }
            if device.type == "cuda":
                save_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
            torch.save(save_dict, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        step += 1

    log_fh.close()
    print(f"\nE2 training complete at step {step}. Best eval BPB: {best_eval_bpb:.3f}")

    final_path = os.path.join(cfg.checkpoint_dir, "e2_final.pt")
    torch.save({
        "step": step,
        "phase": current_phase,
        "model": student.state_dict(),
        "model_cfg": model_cfg,
        "ports": ports.state_dict(),
        "config": cfg.__dict__,
        "best_eval_bpb": best_eval_bpb,
    }, final_path)
    print(f"Final checkpoint: {final_path}")


def _parse_static_weights(s: Optional[str]) -> Optional[dict[str, float]]:
    if s is None:
        return None
    result = {}
    for pair in s.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid static weight: {pair!r} (expected name:weight)")
        name, weight = pair.rsplit(":", 1)
        result[name.strip()] = float(weight.strip())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eklavya E2 multi-teacher KD")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cache-dir", default="eklavya_e2_cache")
    parser.add_argument("--output-dir", default="checkpoints/e2")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to E2 checkpoint to resume from")
    parser.add_argument("--ablation-id", default="A2",
                        help="Ablation label (saved in checkpoints)")
    parser.add_argument("--teachers", nargs="+", default=None,
                        help="Include only these teachers (training-time)")
    parser.add_argument("--exclude-teachers", nargs="+", default=None,
                        help="Exclude these teachers (training-time)")
    parser.add_argument("--disable-router", action="store_true",
                        help="Use arithmetic mean instead of PL router")
    parser.add_argument("--shuffle-teacher-targets", action="store_true",
                        help="Shuffle cache position-teacher alignment")
    parser.add_argument("--ce-only", action="store_true",
                        help="CE-only continuation (no teacher losses, for A0)")
    parser.add_argument("--disable-gradient-budget", action="store_true",
                        help="Skip gradient budgeting (uncapped teacher grads, for A7)")
    parser.add_argument("--no-phased-admission", action="store_true",
                        help="All teachers active from first student-updating phase (for A8)")
    parser.add_argument("--router-mode", default="oracle_gold",
                        choices=["oracle_gold", "gold_free_entropy",
                                 "gold_free_agreement", "gold_free_student_jsd"],
                        help="Router scoring mode (default: oracle_gold)")
    parser.add_argument("--router-agreement-gamma", type=float, default=0.5,
                        help="Agreement penalty weight for gold-free routing")
    parser.add_argument("--router-student-delta", type=float, default=0.25,
                        help="Student-teacher JSD weight for gold-free routing")
    parser.add_argument("--static-weight-mode", default="uniform",
                        choices=["uniform", "prior", "custom"],
                        help="Weight mode when router disabled: uniform (A5), "
                             "prior (A5a), custom (A5b)")
    parser.add_argument("--static-weights", type=str, default=None,
                        help="Custom static weights as t0:0.4,t1:0.3,... (for A5b)")
    parser.add_argument("--bld-mode", action="store_true",
                        help="BLD baseline: single-teacher byte KL, no E2 machinery")
    parser.add_argument("--bld-kl-weight", type=float, default=0.10,
                        help="KL loss weight for BLD mode (default: 0.10)")
    parser.add_argument("--shuffle-seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = E2Config(
        checkpoint_dir=args.output_dir, cache_dir=args.cache_dir,
        resume_from=args.resume,
        ablation_id=args.ablation_id,
        teacher_include=args.teachers,
        teacher_exclude=args.exclude_teachers or [],
        disable_router=args.disable_router,
        shuffle_teacher_targets=args.shuffle_teacher_targets,
        shuffle_seed=args.shuffle_seed,
        ce_only=args.ce_only,
        disable_gradient_budget=args.disable_gradient_budget,
        no_phased_admission=args.no_phased_admission,
        static_weight_mode=args.static_weight_mode,
        static_weights=_parse_static_weights(args.static_weights),
        bld_mode=args.bld_mode,
        bld_kl_weight=args.bld_kl_weight,
        router_mode=args.router_mode,
        router_agreement_gamma=args.router_agreement_gamma,
        router_student_delta=args.router_student_delta,
    )
    if args.steps:
        remaining = args.steps
        cfg.port_warmup_steps = min(cfg.port_warmup_steps, remaining)
        remaining -= cfg.port_warmup_steps
        cfg.consensus_steps = min(cfg.consensus_steps, remaining)
        remaining -= cfg.consensus_steps
        cfg.semantic_landing_steps = min(cfg.semantic_landing_steps, remaining)
        remaining -= cfg.semantic_landing_steps
        cfg.disagreement_steps = remaining
    train_e2(cfg, args.student_checkpoint, args.cache_dir)
