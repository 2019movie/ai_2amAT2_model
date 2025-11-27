# run by: torchrun --nproc_per_node=4 train.py --epochs 4 --save_every 1 --batch_size 32


# ------------------------------------------------------
# Imports
# ------------------------------------------------------
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

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# ------------------------------------------------------
# DDP Setup
# ------------------------------------------------------
def ddp_setup():
    """Initialize Distributed Data Parallel environment."""
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


def init_json_log(filename="training_log.json"):
    if os.environ.get("RANK", "0") == "0":   # only rank 0
        with open(filename, "w") as f:
            json.dump({"logs": []}, f)


def log_json(filename, epoch, batch, loss):
    if os.environ.get("RANK", "0") != "0":   # only rank 0 logs
        return
    with open(filename, "r+") as f:
        data = json.load(f)
        data["logs"].append({
            "epoch": epoch,
            "batch": batch,
            "loss": loss
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
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ------------------------------------------------------
# Trainer Class (DDP)
# ------------------------------------------------------
class Trainer:
    def __init__(self, model, trainloader, testloader, optimizer,
                 criterion, save_every, snapshot_path):
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        
        self.model = model.to(self.gpu_id)
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion

        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.epochs_run = 0

        # Resume if snapshot exists
        if os.path.exists(snapshot_path):
            if self.rank == 0:
                print("Loading snapshot...")
            self._load_snapshot(snapshot_path)

        # Wrap model with DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    # --------------------------
    def _load_snapshot(self, path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(path, map_location=loc)

        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPT_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.rank == 0:
            print(f"Resuming from epoch {self.epochs_run}")

    # --------------------------
    def _run_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # --------------------------
    def _run_epoch(self, epoch):
        self.trainloader.sampler.set_epoch(epoch)

        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images = images.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

            loss = self._run_batch(images, labels)
            running_loss += loss

            # Only rank 0 prints/logs
            if self.rank == 0 and (batch_idx + 1) % 2000 == 0:
                avg = running_loss / 2000
                print(f"[Epoch {epoch+1}, Batch {batch_idx+1}] Loss: {avg:.3f}")
                log_json("training_log.json", epoch+1, batch_idx+1, avg)
                running_loss = 0.0

        if self.rank == 0:
            self.evaluate(epoch)

    # --------------------------
    def evaluate(self, epoch):
        self.model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f"[Rank 0] Eval Epoch {epoch+1} Accuracy: {acc:.4f}")
        self.model.train()

    # --------------------------
    def _save_snapshot(self, epoch):
        if self.rank != 0:
            return
        
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPT_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Checkpoint saved: {self.snapshot_path}")

    # --------------------------
    def train(self, max_epochs):
        init_json_log()  # rank 0 only
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main(epochs, save_every, batch_size):

    ddp_setup()

    # Dataset transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Dataset
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transform)

    # Distributed samplers
    train_sampler = DistributedSampler(trainset)
    test_sampler = DistributedSampler(testset, shuffle=False)

    # DataLoaders
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, sampler=test_sampler, num_workers=2
    )

    # Model, loss, optimizer
    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Trainer
    trainer = Trainer(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        optimizer=optimizer,
        criterion=criterion,
        save_every=save_every,
        snapshot_path="snapshot.pt"
    )

    trainer.train(epochs)

    destroy_process_group()


# ------------------------------------------------------
# CLI Entry
# ------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DDP CIFAR10 Trainer")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args.epochs, args.save_every, args.batch_size)
