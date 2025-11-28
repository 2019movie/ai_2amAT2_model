# ai_2amAT2_model
# classification model, 
## distributed training with PyTorch with data(CIFAR10)

This repository contains a complete, assessment-ready machine learning training pipeline using PyTorch, supporting both single-GPU and Distributed Data Parallel (DDP) training with torchrun.

The project trains a simple convolutional neural network (CNN) on the CIFAR-10 dataset.

## Repository structure

```markdown
ai_2amAT2_model/
+--train_multi.py             # main training script for multiple GPU
+--train-single-jsonLog.py    # main training script for single GPU/CPU
+--run_distributed.sh         # Bash wrapper for single-GPU or DDP execution
+--README.md                  # Documentation
+--inference.py               # inference script to load ./cifar_net.pth and simple test
L__requirements.txt           # Python dependencies
```

## Features

- Supports single-GPU training
- Supports multi-GPU DDP using torchrun
- Clean architecture following PyTorch best practices

## Installation
### Clone the repo
- git clone https://github.com/2019movie/ai_2amAT2_model.git
- cd ai_2amAT2_model
- python3 -m venv myenv
- source myenv/bin/activate
- pip install -r requirements.txt
- run bash script ./ai_2amAT2_model/run_distributed.sh

### Training
- Run script:
- ./ai_2amAT2_model/run_distributed.sh

#### When Single GPU detected
- it executes: python train.py --batch_size=32 --epochs=2 --compile=False

#### When multi GPU detected (DDP)

- it executes: torchrun --nproc_per_node=[num of GPU] ./ai_2amAT2_model/train_multi.py --epochs 10 --save_every 1 --batch_size 32


## Model output
The model will save as "cifar_net.pth"

## Load the model
model = Net()
model.load_state_dict(torch.load("cifar_net.pth"))
model.eval()