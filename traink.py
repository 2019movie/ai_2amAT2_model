import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms


# ------------------------------------------------------
# Model definition
# ------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ------------------------------------------------------
# Training loop
# ------------------------------------------------------
def train(rank, world_size, args):
    # DDP INITIALISATION
    if args.distributed:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # Distributed sampler if DDP
    train_sampler = DistributedSampler(trainset) if args.distributed else None

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------
    model = Net().to(device)

    if args.compile:
        model = torch.compile(model)

    if args.distributed:
        model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if rank == 0 and (i % 2000 == 1999):
                print(f"[{epoch+1}, {i+1}] loss: {running_loss/2000:.3f}")
                running_loss = 0.0

    # Save only on rank 0 (master)
    if rank == 0:
        torch.save(model.module.state_dict() if args.distributed else model.state_dict(),
                   args.model_path)
        print(f"Model saved to {args.model_path}")

    if args.distributed:
        dist.destroy_process_group()


# ------------------------------------------------------
# Main: argument parsing + spawn workers
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--compile", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="./cifar_net.pth")

    # Automatically detect DDP via torchrun env vars
    parser.add_argument("--distributed", action="store_true", help="Use DDP")

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()


    if args.distributed:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        train(rank, world_size, args)
    else:
        train(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    main()
