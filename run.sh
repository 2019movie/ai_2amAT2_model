#!/bin/bash

# Example usage:
#   bash run.sh single
#   bash run.sh ddp

MODE=$1

if [ "$MODE" = "single" ]; then
    echo "Running single GPU training..."
    python train.py --batch_size=32 --epochs=2 --compile=False

elif [ "$MODE" = "ddp" ]; then
    echo "Running DDP on all visible GPUs..."
    torchrun --standalone --nproc_per_node=4 train.py --distributed

else
    echo "Usage:"
    echo "  bash run.sh single     # single GPU"
    echo "  bash run.sh ddp        # multi-GPU DDP"
fi
