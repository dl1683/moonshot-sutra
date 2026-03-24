#!/bin/bash
# R20 Falsification Experiment Sequence: F1 → F5
# Run sequentially on GPU. Each experiment fully completes before the next starts.
#
# Total estimated GPU time: ~6-8 hours
# Required: step_6000.pt checkpoint, eval_cache.pt, eval_cache_16k.pt, 16K shards

REPO="/c/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra"
cd "$REPO"

# ============================================================
# F1: SELECT BEST PARENT CHECKPOINT (5K vs 6K)
# ============================================================
# Compare det-eval results. Watcher auto-runs step 6000 eval.
# Step 5000 results (from memory):
#   D10=6.746, D12=6.760
# Step 6000 results: check results/det_eval_step6000.json
#
# Pick the one with LOWER D10. If tie, use step 6000 (more training).

# ============================================================
# F2: TEACHER-FREE vs TEACHERED A/B TEST (1K steps each)
# ============================================================
# Decision criteria (R20):
#   KD-off within 0.05 BPT at D10+D12 → O4 drops, KD leaves critical path
#   KD-on wins >=0.10 at D10 or D12 + majority watcher tasks → KD survives

# Copy parent checkpoint to both run dirs
PARENT_STEP=6000  # or 5000, based on F1 result
cp "results/checkpoints_p1/step_${PARENT_STEP}.pt" results/checkpoints_f2_kd_on/rolling_latest.pt
cp "results/checkpoints_p1/step_${PARENT_STEP}.pt" results/checkpoints_f2_kd_off/rolling_latest.pt

TARGET=$((PARENT_STEP + 1000))

echo "=== F2 ARM 1: KD-ON (with teachers) ==="
python code/train_p1_twoteacher.py \
    --run-name f2_kd_on \
    --q1-queue code/q1_xtkd_queue.json \
    --q2-queue code/q2_cka_queue.json \
    --stop-at $TARGET

echo "=== F2 ARM 2: KD-OFF (teacher-free) ==="
python code/train_p1_twoteacher.py \
    --teacher-free \
    --run-name f2_kd_off \
    --stop-at $TARGET

echo "=== F2 Det-Eval ==="
python code/train_p1_twoteacher.py \
    --det-eval "results/checkpoints_f2_kd_on/rolling_latest.pt,results/checkpoints_f2_kd_off/rolling_latest.pt"

# ============================================================
# F3: RECURRENT 16K TOKENIZER-ONLY CONTROL (1K steps)
# ============================================================
# Tests: does the 16K tokenizer itself improve things, independent of architecture?
# Uses transplanted 16K checkpoint, trains teacher-free on 16K shards.

mkdir -p results/checkpoints_f3_16k_recurrent

echo "=== F3: Recurrent 16K Control ==="
python code/train_p1_twoteacher.py \
    --run-name f3_16k_recurrent \
    --teacher-free \
    --vocab-size 16000 \
    --shard-dir data/shards_16k \
    --init-weights results/checkpoints_p2/transplanted_16k_from_step5000.pt \
    --stop-at 1000

echo "=== F3 Det-Eval (16K cache) ==="
python code/train_p1_twoteacher.py \
    --det-eval results/checkpoints_f3_16k_recurrent/rolling_latest.pt \
    --eval-cache results/eval_cache_16k.pt

# ============================================================
# F4: MATCHED DENSE 16K CONTROL (5K steps)
# ============================================================
# Tests: does recurrence help AT ALL at this scale?
# Dense transformer: 11L/512d/8h/SwiGLU-1536/tied-16K = 45.7M params
# Teacher-free, same 16K data, same token budget.
# Decision: if dense beats recurrent on 4/7 watcher tasks, recurrence loses.

echo "=== F4: Dense 16K Control (5K steps) ==="
python code/dense_baseline.py \
    --run-name dense_f4 \
    --max-steps 5000

echo "=== F4 Det-Eval ==="
python code/dense_baseline.py \
    --det-eval results/checkpoints_dense_f4/step_5000.pt

# ============================================================
# F5: WIDEN-ONLY CANARY (500 steps, teacher-free)
# ============================================================
# Tests: is the "26M core too small" hypothesis real?
# Widen StageBank FFN 1536→2304 with zero-gated additive branch.
# If widening doesn't improve D10/D12 by >=0.05, "core too small" loses status.
#
# NOTE: This requires a code modification to the model architecture.
# Implementation needed before running.

echo "=== F5: Widen-only canary (NEEDS IMPLEMENTATION) ==="
echo "TODO: Implement zero-gated FFN widening in launch_v060a.py"

echo ""
echo "=== ALL FALSIFIERS COMPLETE ==="
echo "Check results/ for all det-eval JSONs."
echo "Decision matrix:"
echo "  F2: KD causal lift (O4)"
echo "  F3: Tokenizer effect isolation"
echo "  F4: Recurrence vs dense (core thesis)"
echo "  F5: Capacity hypothesis (core size)"
