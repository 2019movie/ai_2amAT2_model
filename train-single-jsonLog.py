
# import library
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json

# Use GPU when available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# define helper functions
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Experiment tracking helper functions
def init_json_log(filename="training_log.json"):
    with open(filename, "w") as f:
        json.dump({"logs": []}, f)

def log_json(filename, epoch, batch, loss):
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
def train(net, trainloader, criterion, optimizer, testloader, device, classes, epochs=2):
    init_json_log()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # move to GPU

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                avg_loss = running_loss / 2000
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                log_json("training_log.json", epoch+1, i+1, avg_loss)
                running_loss = 0.0
         evaluate_model(net, testloader, device, classes)
    print('Finished Training')


# ------------------------------------------------------
# Test function
# ------------------------------------------------------
def test(net, testloader, classes):
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

# ------------------------------------------------------
# Evaluation function
# ------------------------------------------------------

def evaluate_model(net, dataloader, device, classes):
    net.eval()  # set to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Accuracy: {accuracy:.4f}")

    # Classification report
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion matrix
    #plot_confusion_matrix(all_labels, all_preds, classes)


# ------------------------------------------------------
# Main entry point
# ------------------------------------------------------
def main():
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4
    
    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------
    
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #net = Net()
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(net, trainloader, criterion, optimizer, testloader, device, classes, epochs=2)

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    test(net, testloader, classes)

if __name__ == "__main__":
    main()
