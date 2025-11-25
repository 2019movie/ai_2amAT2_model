import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import datetime

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, is_initialized, get_rank, get_world_size


import argparse

# ------------------------------------------------------
# Handle arguments
# ------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    return parser.parse_args()


# ------------------------------------------------------
# Device detection
# ------------------------------------------------------
def get_device_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"CUDA available: {num_gpus} GPU(s). First device: {device_name}")
        return torch.device("cuda"), num_gpus
    else:
        print("Running on CPU")
        return torch.device("cpu"), 0


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def setup_ddp():
    # Detect DDP launch
    ddp = ("RANK" in os.environ) or ("WORLD_SIZE" in os.environ)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if ddp:
        init_process_group(backend="nccl", timeout=torch.timedelta(seconds=600))
        torch.cuda.set_device(local_rank)  # ensure this process uses the correct GPU
        print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}")
    return ddp, rank, world_size, local_rank


def init_json_log(filename="training_log.json"):
    with open(filename, "w") as f:
        json.dump({"logs": []}, f)

def log_json(filename, epoch, batch, loss, device_info, world_size, rank):
    with open(filename, "r+") as f:
        data = json.load(f)
        data["logs"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "device": device_info,
            "world_size": world_size,
            "rank": rank
        })
        f.seek(0)
        json.dump(data, f, indent=4)

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
def train(net, trainloader, criterion, optimizer, testloader, device, classes, epochs=2, world_size=1, rank=0):
    init_json_log()
    for epoch in range(epochs):
        # Ensure proper shuffling across epochs in DDP
        if isinstance(trainloader.sampler, DistributedSampler):
            trainloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999 and rank == 0:
                avg_loss = running_loss / 2000
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                log_json("training_log.json", epoch+1, i+1, avg_loss, str(device), world_size, rank)
                running_loss = 0.0

        if rank == 0:
            evaluate_model(net, testloader, device, classes)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
            print(f"saved: checkpoint_epoch_{epoch+1}.pth")
    if rank == 0:
        print('Finished Training')

# ------------------------------------------------------
# Evaluation
# ------------------------------------------------------
def evaluate_model(net, dataloader, device, classes):
    net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Accuracy: {accuracy:.4f}")

# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    args = parse_args()

    device, num_gpus = get_device_info()

    # Initialize DDP if launched with torchrun
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        init_process_group(backend="nccl")
        rank = get_rank()
        world_size = get_world_size()
        torch.cuda.set_device(rank % max(1, num_gpus))
        print(f"[DDP] world_size={world_size}, rank={rank}")
    else:
        rank, world_size = 0, 1

    try:
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

        # 1) Only rank 0 downloads the dataset
        if rank == 0:
            _ = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
            _ = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

        # 2) All ranks wait until download completes
        if is_initialized():
            torch.distributed.barrier()

        # 3) Now build datasets and loaders on all ranks (download=False)
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=False, transform=transform)
        testset  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=False, transform=transform)

        sampler = DistributedSampler(trainset) if world_size > 1 else None
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

        # After creating the model
        net = Net().to(torch.device("cuda", local_rank))
        if world_size > 1:
            net = DDP(net, device_ids=[local_rank], output_device=local_rank)

        # When synchronizing after dataset download
        if is_initialized():
            torch.distributed.barrier(device_ids=[local_rank])

        #net = Net().to(device)
        #if world_size > 1:
        #    net = DDP(net, device_ids=[rank % max(1, num_gpus)], output_device=rank % max(1, num_gpus))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # If using DistributedSampler, set epoch each loop inside train()
        train(net, trainloader, criterion, optimizer, testloader, device, classes,
              epochs=args.epochs, world_size=world_size, rank=rank)

        if rank == 0:
            torch.save(net.state_dict(), './cifar_net.pth')

    except RuntimeError as e:
        if "Dataset not found or corrupted" in str(e):
            if rank == 0:
                print("Dataset corrupted. Removing ./data and re-downloading...")
                import shutil
                shutil.rmtree("./data", ignore_errors=True)
                _ = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
                _ = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
            if is_initialized():
                torch.distributed.barrier()
            # Rebuild datasets after recovery
            # (Optionally re-run main() or rebuild loaders here)
            raise
        else:
            raise
    finally:
        if is_initialized():
            destroy_process_group()


if __name__ == "__main__":
    main()
