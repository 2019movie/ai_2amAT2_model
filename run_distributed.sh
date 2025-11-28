#!/bin/bash

# -----------------------------
# Default values
# -----------------------------
EPOCHS=10
SAVE_EVERY=1
BATCH_SIZE=32

# -----------------------------
# Parse arguments
# -----------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --save-every)
      SAVE_EVERY="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# -----------------------------
# Detect number of GPUs
# -----------------------------
NUM_GPUS=$(python3 - << 'EOF'
import torch
print(torch.cuda.device_count())
EOF
)

echo "Detected GPUs: $NUM_GPUS"

# -----------------------------
# Branch based on GPU count
# -----------------------------
if [ "$NUM_GPUS" -le 1 ]; then
    echo "Single GPU or CPU detected. Running single-GPU training..."
    python ./ai_2amAT2_model/train-single-jsonLog.py \
        --epochs "$EPOCHS" \
        --save_every "$SAVE_EVERY" \
        --batch_size "$BATCH_SIZE"
else
    echo "Multiple GPUs detected. Running Distributed Data Parallel (DDP)..."
    echo "Launching with --nproc_per_node=$NUM_GPUS"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        ./ai_2amAT2_model/train_multi.py \
        --epochs "$EPOCHS" \
        --save_every "$SAVE_EVERY" \
        --batch_size "$BATCH_SIZE"
fi
