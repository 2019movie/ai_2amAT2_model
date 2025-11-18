# ai_2amAT2_model
# classification model, distributed training with PyTorch with data(CIFAR10
)

This repository contains a complete, assessment-ready machine learning training pipeline using PyTorch, supporting both single-GPU and Distributed Data Parallel (DDP) training with torchrun.

The project trains a simple convolutional neural network (CNN) on the CIFAR-10 dataset,

## Repository structure

project/
│
├── train.py          # Main training script (single GPU + DDP)
├── run.sh            # Bash wrapper for single-GPU or DDP execution
├── README.md         # Documentation
└── requirements.txt  # Python dependencies

## Features

- Supports single-GPU training
- Supports multi-GPU DDP using torchrun
- Supports multi-node DDP
- Clean architecture following PyTorch best practices

## Installation
### Clone the repo
git clone https://github.com/2019movie/ai_2amAT2_model.git
cd ai_2amAT2_model
pip install -r requirements.txt

### Training
#### Single GPU training

Run script:
bash run.sh single

- it executes: python train.py --batch_size=32 --epochs=2 --compile=False

#### multi GPU training (DDP)

run script:
bash run.sh ddp

## Model output
The model will save as "cifar_net.pth"

## Load the model
model = Net()
model.load_state_dict(torch.load("cifar_net.pth"))
model.eval()
