"""Launch v0.5.2: v0.5 + switching kernel + gain clamp + LR=8e-4.

Three surgical changes from v0.5, per Codex synthesis:
1. 2-mode switching transition kernel (+4.1% BPT)
2. BayesianWrite gain clamp(max=10.0) (eliminates NaN)
3. LR=8e-4 (safe with clamp, validated on CPU)

Usage:
    python code/launch_v052.py [--warmstart results/checkpoints_v05/step_XXXXX.pt]
"""

import sys
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
from sutra_v05_ssm import SutraV05, SwitchingKernel2


def create_v052(dim=768, ff_dim=1536, max_steps=8, window=4, k_retrieval=8):
    """Create v0.5.2 model (v0.5 + switching kernel)."""
    model = SutraV05(vocab_size=50257, dim=dim, ff_dim=ff_dim,
                     max_steps=max_steps, window=window, k_retrieval=k_retrieval)
    model.transition = SwitchingKernel2(dim)
    return model


def warmstart_v052(checkpoint_path, **kwargs):
    """Warm-start v0.5.2 from v0.5 checkpoint."""
    model = create_v052(**kwargs)
    new_state = model.state_dict()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    old_state = ckpt["model"] if "model" in ckpt else ckpt

    transferred = 0
    for name in list(new_state.keys()):
        if name in old_state and new_state[name].shape == old_state[name].shape:
            new_state[name] = old_state[name]
            transferred += 1

    # Remap transition.net.* -> transition.base.*
    remap = {
        "transition.net.0.weight": "transition.base.0.weight",
        "transition.net.0.bias": "transition.base.0.bias",
        "transition.net.2.weight": "transition.base.2.weight",
        "transition.net.2.bias": "transition.base.2.bias",
    }
    for old_name, new_name in remap.items():
        if old_name in old_state and new_name in new_state:
            if old_state[old_name].shape == new_state[new_name].shape:
                new_state[new_name] = old_state[old_name]
                transferred += 1

    model.load_state_dict(new_state)
    total = len(new_state)
    print(f"Warm-start: {transferred}/{total} params transferred ({transferred/total*100:.0f}%)")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmstart", default=None, help="v0.5 checkpoint path")
    args = parser.parse_args()

    if args.warmstart:
        model = warmstart_v052(args.warmstart)
    else:
        model = create_v052()

    print(f"v0.5.2 params: {model.count_params():,}")
    print(f"Changes: SwitchingKernel2 + gain_clamp(10) + LR=8e-4")
    print(f"Ready for training with sutra_v05_train.py (update LR to 8e-4)")
