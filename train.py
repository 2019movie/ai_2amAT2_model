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
from torch.distributed import init_process_group, destroy_process_group

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
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            try:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("CUDA OOM error. Try reducing batch size.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            running_loss += loss.item()
            if i % 2000 == 1999 and rank == 0:  # only log from rank 0
                avg_loss = running_loss / 2000
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                log_json("training_log.json", epoch+1, i+1, avg_loss,
                         str(device), world_size, rank)
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
    device, num_gpus = get_device_info()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        init_process_group(backend="nccl")
        torch.cuda.set_device(rank % num_gpus)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    sampler = DistributedSampler(trainset) if world_size > 1 else None
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=(sampler is None),
                                              sampler=sampler, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

    net = Net().to(device)
    if world_size > 1:
        net = DDP(net, device_ids=[rank % num_gpus])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, trainloader, criterion, optimizer, testloader, device, classes,
          epochs=2, world_size=world_size, rank=rank)

    if rank == 0:
        torch.save(net.state_dict(), './cifar_net.pth')

    if world_size > 1:
        destroy_process_group()

if __name__ == "__main__":
    main()
