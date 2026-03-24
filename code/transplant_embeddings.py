"""Transplant embeddings from GPT-2 (50257) to 16K custom tokenizer.

For each token in the 16K vocab:
  1. Decode to text using 16K tokenizer
  2. Encode that text with GPT-2 tokenizer
  3. If single-token match: copy embedding directly
  4. If multi-token match: average the GPT-2 embeddings
  5. If no match: initialize from Gaussian (mean/std of existing embeddings)

Produces a new checkpoint ready for warm-start training with 16K vocab.
Weight-tied output projection handled automatically (emb.weight IS the output matrix).

Usage:
  python code/transplant_embeddings.py \
    --checkpoint results/checkpoints_p1/rolling_latest.pt \
    --tokenizer data/tokenizer_16k/tokenizer.json \
    --output results/checkpoints_p2/transplanted_16k.pt
"""

import sys, argparse, json, torch
import torch.nn as nn
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from launch_v060a import create_v060a


def build_token_map(gpt2_tok, new_tok, new_vocab_size):
    """Map each 16K token ID to a list of GPT-2 token IDs.

    Strategy: decode each 16K token to bytes/text, re-encode with GPT-2.
    Returns: dict[int, list[int]] — new_id -> [gpt2_ids]
    """
    token_map = {}
    exact_matches = 0
    multi_matches = 0
    no_matches = 0

    for new_id in range(new_vocab_size):
        # Decode 16K token to text
        text = new_tok.decode([new_id])
        if not text:
            no_matches += 1
            token_map[new_id] = []
            continue

        # Re-encode with GPT-2
        gpt2_ids = gpt2_tok.encode(text, add_special_tokens=False)
        if len(gpt2_ids) == 1:
            exact_matches += 1
        elif len(gpt2_ids) > 1:
            multi_matches += 1
        else:
            no_matches += 1

        token_map[new_id] = gpt2_ids

    print(f"Token mapping: {exact_matches} exact, {multi_matches} multi, {no_matches} empty")
    return token_map


def transplant(checkpoint_path, tokenizer_path, output_path, dim=768, ff_dim=1536):
    """Create transplanted checkpoint with 16K vocab embeddings."""
    # Load tokenizers
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    new_tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    new_vocab_size = new_tok.vocab_size
    print(f"New vocab: {new_vocab_size}, GPT-2 vocab: {gpt2_tok.vocab_size}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    step = ckpt.get("step", "?")
    bpt = ckpt.get("best_bpt", "?")
    print(f"Loaded checkpoint: step={step}, BPT={bpt}")

    # Extract old embedding weights
    old_emb = state["emb.weight"]  # (50257, 768)
    print(f"Old embedding: {old_emb.shape}")

    # Build token map
    token_map = build_token_map(gpt2_tok, new_tok, new_vocab_size)

    # Create new embedding matrix
    new_emb = torch.zeros(new_vocab_size, dim)
    emb_mean = old_emb.mean(dim=0)
    emb_std = old_emb.std(dim=0)
    transferred = 0

    for new_id in range(new_vocab_size):
        gpt2_ids = token_map[new_id]
        if len(gpt2_ids) == 0:
            # No match — random init from distribution of trained embeddings
            new_emb[new_id] = emb_mean + emb_std * torch.randn(dim) * 0.1
        elif len(gpt2_ids) == 1:
            # Exact match — copy directly
            new_emb[new_id] = old_emb[gpt2_ids[0]]
            transferred += 1
        else:
            # Multi-token — average (weighted could be better but average is safe)
            vecs = old_emb[gpt2_ids]
            new_emb[new_id] = vecs.mean(dim=0)
            transferred += 1

    print(f"Transferred: {transferred}/{new_vocab_size} ({100*transferred/new_vocab_size:.1f}%)")

    # Create new model with correct vocab size
    model = create_v060a(vocab_size=new_vocab_size, dim=dim, ff_dim=ff_dim,
                          max_steps=12, window=4, k_retrieval=8)

    # Load non-embedding weights from old checkpoint
    new_state = model.state_dict()
    loaded_keys = 0
    skipped_keys = []
    for key, value in state.items():
        if key == "emb.weight":
            continue  # We handle this separately
        if key in new_state:
            if new_state[key].shape == value.shape:
                new_state[key] = value
                loaded_keys += 1
            else:
                skipped_keys.append(f"{key}: {value.shape} != {new_state[key].shape}")
        else:
            skipped_keys.append(f"{key}: not in new model")

    # Set new embedding
    new_state["emb.weight"] = new_emb
    loaded_keys += 1

    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys:")
        for sk in skipped_keys[:10]:
            print(f"  {sk}")

    model.load_state_dict(new_state)
    print(f"Loaded {loaded_keys} parameter tensors into new model")
    print(f"New model params: {model.count_params():,}")

    # Save new checkpoint (fresh optimizer — vocab change invalidates old optimizer state)
    new_ckpt = {
        "model": model.state_dict(),
        "step": 0,  # Reset step — this is a new training branch
        "best_bpt": float("inf"),
        "config": {
            "vocab_size": new_vocab_size,
            "dim": dim,
            "ff_dim": ff_dim,
            "max_steps": 12,
            "parent_step": step,
            "parent_bpt": bpt,
            "parent_checkpoint": str(checkpoint_path),
            "tokenizer": str(tokenizer_path),
            "embedding_transfer_rate": transferred / new_vocab_size,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_ckpt, output_path)
    print(f"Saved: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")

    # Verify: quick forward pass
    model.eval()
    x = torch.randint(0, new_vocab_size, (1, 32))
    with torch.no_grad():
        logits, aux = model(x)
    print(f"Verify forward: logits={logits.shape}, no errors")

    return {
        "new_vocab_size": new_vocab_size,
        "transferred": transferred,
        "transfer_rate": transferred / new_vocab_size,
        "old_params": sum(v.numel() for v in state.values()),
        "new_params": model.count_params(),
        "param_savings": sum(v.numel() for v in state.values()) - model.count_params(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default=str(REPO / "data/tokenizer_16k/tokenizer.json"))
    parser.add_argument("--output", default=str(REPO / "results/checkpoints_p2/transplanted_16k.pt"))
    args = parser.parse_args()

    stats = transplant(args.checkpoint, args.tokenizer, args.output)
    print(f"\n{'='*50}")
    print(f"TRANSPLANT COMPLETE")
    print(f"  Vocab: 50257 -> {stats['new_vocab_size']}")
    print(f"  Transfer rate: {stats['transfer_rate']*100:.1f}%")
    print(f"  Param savings: {stats['param_savings']:,} ({stats['param_savings']/stats['old_params']*100:.1f}%)")
