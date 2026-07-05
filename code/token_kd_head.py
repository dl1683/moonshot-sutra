"""Tied-embedding token KD head for Phase 2.5.

Projects reasoner hidden states to teacher embedding space, then computes logits
by dotting against frozen teacher embeddings. Only 1.2M trainable params.

Codex R64 design: REJECT 175M full-vocab head. USE tied-embedding approach:
  hidden -> RMSNorm(d_model) -> Linear(d_model, teacher_dim) -> dot(teacher_emb.T)

Combined with dynamic top-k KL (k=64) against teacher's top logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from s0_architecture import RMSNorm


class TiedEmbeddingKDHead(nn.Module):
    """Project hidden states to teacher token space via tied embeddings.

    Uses the teacher's embedding matrix as the output projection, so the
    model is forced to produce representations in the teacher's semantic space.
    Only the d_model->teacher_dim linear layer is trainable (~1.2M params).
    """
    def __init__(self, d_model: int, teacher_dim: int, teacher_embeddings: torch.Tensor):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, teacher_dim, bias=False)
        self.register_buffer("teacher_emb", teacher_embeddings)  # (V, teacher_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute logits over teacher vocabulary.

        Args:
            hidden: (B, N, d_model) — reasoner hidden states at selected positions
        Returns:
            logits: (B, N, V) — unnormalized logits over teacher vocabulary
        """
        h = self.norm(hidden)
        h = self.proj(h)  # (B, N, teacher_dim)
        logits = h @ self.teacher_emb.T  # (B, N, V)
        return logits

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def top_k_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    k: int = 64,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL divergence computed only over teacher's top-k tokens + sampled negatives.

    Focuses learning on discriminative decisions (which continuation is correct?)
    rather than the full distribution (mostly syntax/whitespace).

    Args:
        student_logits: (B, N, V) from TiedEmbeddingKDHead
        teacher_logits: (B, N, V) from teacher model
        k: number of top teacher tokens to include
        temperature: softmax temperature
    Returns:
        loss: scalar
    """
    B, N, V = teacher_logits.shape

    # Get teacher's top-k indices
    _, top_indices = teacher_logits.topk(k, dim=-1)  # (B, N, k)

    # Gather student and teacher logits at top-k positions
    student_top = student_logits.gather(-1, top_indices)  # (B, N, k)
    teacher_top = teacher_logits.gather(-1, top_indices)  # (B, N, k)

    # Softmax over top-k subset
    student_log_probs = F.log_softmax(student_top / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_top / temperature, dim=-1)

    # KL divergence: sum_i p_teacher * (log p_teacher - log p_student)
    loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Scale by T^2 (standard KD temperature scaling)
    return loss * (temperature ** 2)


def find_token_patch_alignment(
    byte_ids: torch.Tensor,
    tokenizer,
    patch_size: int = 4,
) -> list[list[int]]:
    """Find patch indices that align with token boundaries.

    Only returns patches where the patch end aligns with a token end.
    These are the positions where Phase 1 provided supervision and
    the hidden state is most likely to contain token-level information.

    Args:
        byte_ids: (B, T) byte tensor
        tokenizer: teacher tokenizer
        patch_size: P
    Returns:
        List of B lists, each containing patch indices with clean alignment
    """
    B, T = byte_ids.shape
    N = T // patch_size
    batch_indices = []

    for b in range(B):
        bytes_np = byte_ids[b].cpu().numpy().tobytes()
        try:
            text = bytes_np.decode("utf-8", errors="replace")
        except Exception:
            batch_indices.append([])
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_ends = set()
        pos = 0
        for t in tokens:
            token_text = tokenizer.decode([t])
            pos += len(token_text.encode("utf-8"))
            token_ends.add(pos - 1)

        aligned = []
        for i in range(N):
            patch_end = (i + 1) * patch_size - 1
            if patch_end in token_ends:
                aligned.append(i)
        batch_indices.append(aligned)

    return batch_indices


def build_kd_head(
    d_model: int,
    teacher_embeddings_path: str,
    device: str = "cpu",
) -> TiedEmbeddingKDHead:
    """Build KD head from cached teacher embeddings."""
    data = torch.load(teacher_embeddings_path, map_location=device, weights_only=True)
    teacher_emb = data["embeddings"] if isinstance(data, dict) else data
    teacher_dim = teacher_emb.shape[1]
    head = TiedEmbeddingKDHead(d_model, teacher_dim, teacher_emb)
    return head


if __name__ == "__main__":
    d_model = 1152
    teacher_dim = 1024
    V = 151936

    # Simulate
    teacher_emb = F.normalize(torch.randn(V, teacher_dim), dim=-1)
    head = TiedEmbeddingKDHead(d_model, teacher_dim, teacher_emb)

    print(f"Trainable params: {head.trainable_params():,}")
    print(f"Teacher embedding: {teacher_emb.shape}")

    # Test forward
    B, N = 2, 32
    hidden = torch.randn(B, N, d_model)
    logits = head(hidden)
    print(f"Logits shape: {logits.shape}")  # (2, 32, 151936)

    # Test top-k KL loss
    teacher_logits = torch.randn(B, N, V)
    loss = top_k_kl_loss(logits, teacher_logits, k=64)
    print(f"Top-k KL loss: {loss.item():.4f}")
