"""Phase 2: Sutra model with frozen semantic codec replacing ByteEncoder.

The codec's CausalByteTransformer (trained in Phase 1 to retrieve teacher
token embeddings) is FROZEN. Its PatchProjection is trainable (untrained
in Phase 1). The global reasoner and byte decoder are randomly initialized
and trained with byte CE.

Architecture:
  byte_ids → [FROZEN CausalByteTransformer] → patch_hidden(256) →
  [TRAINABLE PatchProjection](256→d_model) → GlobalReasoner → ByteDecoder → logits
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from s0_architecture import S0Config, GlobalReasoner, ByteDecoder, RMSNorm
from semantic_codec import SemanticCodec, CodecConfig


class SutraCodecModel(nn.Module):
    """Sutra with frozen semantic codec encoder.

    Phase 1 trained the codec encoder to retrieve teacher token embeddings.
    Phase 2 freezes the encoder and trains:
    - PatchProjection (codec_dim → d_model)
    - GlobalReasoner (causal transformer over patch states)
    - ByteDecoder (local autoregressive byte prediction)
    """
    def __init__(self, model_cfg: S0Config, codec: SemanticCodec):
        super().__init__()
        self.cfg = model_cfg
        self.codec = codec

        assert model_cfg.patch_size == codec.cfg.patch_size, (
            f"patch_size mismatch: model={model_cfg.patch_size}, codec={codec.cfg.patch_size}"
        )

        # Freeze codec encoder
        for param in self.codec.encoder.parameters():
            param.requires_grad_(False)
        # Discard alignment head (Phase 1 only)
        for param in self.codec.alignment_head.parameters():
            param.requires_grad_(False)

        # Rebuild PatchProjection if d_model doesn't match
        if self.codec.d_model != model_cfg.d_model:
            from semantic_codec import PatchProjection
            self.codec.patch_projection = PatchProjection(
                self.codec.cfg.codec_dim, model_cfg.d_model
            )
            self.codec.d_model = model_cfg.d_model

        self.reasoner = GlobalReasoner(model_cfg)
        self.decoder = ByteDecoder(model_cfg)

    def forward(self, byte_ids: torch.Tensor, return_aux: bool = True):
        B, T = byte_ids.shape
        P = self.cfg.patch_size

        # Codec: frozen encoder + trainable projection
        with torch.no_grad():
            patch_hidden = self.codec.encoder.get_patch_states(byte_ids)
        patch_states = self.codec.patch_projection(patch_hidden)

        # Global reasoning
        hidden = self.reasoner(patch_states)

        # Causal alignment: hidden[i] predicts bytes of patch i+1
        N = hidden.shape[1]
        pred_hidden = hidden[:, :-1]
        target_bytes = byte_ids.reshape(B, T // P, P)[:, 1:]

        # Cross-attention context
        M = N - 1
        prev_padded = F.pad(pred_hidden, (0, 0, 1, 0))[:, :M]
        nearby = torch.stack([prev_padded, pred_hidden], dim=2)

        logits = self.decoder(pred_hidden, target_bytes, nearby)

        return {
            "logits": logits,
            "hidden": hidden,
            "patch_states": patch_states,
        }

    def count_parameters(self) -> dict[str, int]:
        counts = {}
        counts["codec_encoder_frozen"] = sum(
            p.numel() for p in self.codec.encoder.parameters()
        )
        counts["codec_projection_trainable"] = sum(
            p.numel() for p in self.codec.patch_projection.parameters()
        )
        counts["reasoner"] = sum(p.numel() for p in self.reasoner.parameters())
        counts["decoder"] = sum(p.numel() for p in self.decoder.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts


def load_codec_for_phase2(
    checkpoint_path: str,
    d_model: int = 1152,
    device: str = "cpu",
) -> SemanticCodec:
    """Load a Phase 1 trained codec for Phase 2 integration."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    codec_cfg = ckpt.get("config", {})
    cfg = CodecConfig(
        codec_dim=codec_cfg.get("codec_dim", 256),
        codec_layers=codec_cfg.get("codec_layers", 4),
        window_size=codec_cfg.get("window_size", 256),
    )

    codec = SemanticCodec(cfg, d_model=codec_cfg.get("d_model", d_model))
    codec.load_state_dict(ckpt["codec_state_dict"])

    return codec


def build_phase2_model(
    codec_checkpoint: str,
    config_name: str = "wide7",
) -> SutraCodecModel:
    """Build the full Phase 2 model from a Phase 1 codec checkpoint."""
    from s0_configs import ALL_CONFIGS
    model_cfg = ALL_CONFIGS[config_name]()

    codec = load_codec_for_phase2(codec_checkpoint, d_model=model_cfg.d_model)
    model = SutraCodecModel(model_cfg, codec)

    return model


if __name__ == "__main__":
    from s0_configs import s0_wide7
    cfg = s0_wide7()

    # Test with random codec (no checkpoint)
    codec_cfg = CodecConfig()
    codec = SemanticCodec(codec_cfg, d_model=cfg.d_model)
    model = SutraCodecModel(cfg, codec)

    counts = model.count_parameters()
    print("Phase 2 Model Parameter Counts:")
    for k, v in counts.items():
        print(f"  {k}: {v:,} ({v/1e6:.2f}M)")

    # Test forward
    B, T = 2, 256
    byte_ids = torch.randint(0, 256, (B, T))
    out = model(byte_ids)
    print(f"\nLogits shape: {out['logits'].shape}")
    print(f"Hidden shape: {out['hidden'].shape}")
