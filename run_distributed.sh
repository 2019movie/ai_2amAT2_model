#!/bin/bash

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
    python ./train-single-jsonLog.py

else
    echo "Multiple GPUs detected. Running Distributed Data Parallel (DDP)..."
    echo "Launching with --nproc_per_node=$NUM_GPUS"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        ./train_multi.py \
        --epochs 10 \
        --save_every 1 \
        --batch_size 32
fi
