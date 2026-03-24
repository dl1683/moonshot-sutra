#!/bin/bash
# F2: Teacher-free vs teachered A/B test (1K steps each)
# Both arms start from the SAME checkpoint with identical RNG/optimizer state
# Uses --stop-at to preserve LR schedule from parent training
# R20 spec: if KD-off within 0.05 BPT at D10/D12, O4 drops. If KD-on wins >=0.10, KD survives.

REPO="/c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra"
PARENT_CKPT="$REPO/results/checkpoints_p1/rolling_latest.pt"

# Verify parent exists
if [ ! -f "$PARENT_CKPT" ]; then
    echo "ERROR: Parent checkpoint not found: $PARENT_CKPT"
    exit 1
fi

# Read parent step from checkpoint
PARENT_STEP=$(python -c "import torch; c=torch.load('$PARENT_CKPT', weights_only=False, map_location='cpu'); print(c.get('step', 0))")
TARGET_STEP=$((PARENT_STEP + 1000))

echo "=== F2: Teacher-Free vs Teachered A/B Test ==="
echo "Parent step: $PARENT_STEP, stop at: $TARGET_STEP"
echo "LR schedule preserved (cosine with MAX=15000)"
echo ""

# Create separate run dirs and copy parent checkpoint
mkdir -p "$REPO/results/checkpoints_f2_kd_on"
mkdir -p "$REPO/results/checkpoints_f2_kd_off"
cp "$PARENT_CKPT" "$REPO/results/checkpoints_f2_kd_on/rolling_latest.pt"
cp "$PARENT_CKPT" "$REPO/results/checkpoints_f2_kd_off/rolling_latest.pt"

echo "ARM 1 (KD-ON):"
echo "  python code/train_p1_twoteacher.py --run-name f2_kd_on --stop-at $TARGET_STEP"
echo ""
echo "ARM 2 (KD-OFF):"
echo "  python code/train_p1_twoteacher.py --teacher-free --run-name f2_kd_off --stop-at $TARGET_STEP"
echo ""
echo "After both complete, evaluate:"
echo "  python code/train_p1_twoteacher.py --det-eval results/checkpoints_f2_kd_on/step_${TARGET_STEP}.pt,results/checkpoints_f2_kd_off/step_${TARGET_STEP}.pt"
echo ""
echo "Decision criteria (R20):"
echo "  - KD-off within 0.05 BPT at D10+D12 -> O4 drops, KD leaves critical path"
echo "  - KD-on wins >=0.10 at D10 or D12 + majority watcher tasks -> KD survives"
