import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import pandas as pd

# Set the device to use for training and testing (either CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10).to(device)

# Define the training dataset and the data loader
train_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the test dataset and the data loader
test_dataset = datasets.MNIST('../data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        if batch_idx == 0 or (batch_idx + 1) % 64 == 0:
            train_losses.append(loss.item())
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# Define the test function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    wrong_predictions = []
    losses = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item() * data.size(0)
            losses.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # get the indices of the wrongly predicted images
            for i, p in enumerate(pred):
                if p != target[i]:
                    wrong_predictions.append((data[i].cpu(), p, target[i]))
            if epoch == 1:
                confusion_matrix.update(pred.squeeze(), target)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    if epoch == 1:
        print('Confusion Matrix:')
        print(confusion_matrix.compute().cpu().numpy())

    num_to_display = min(1, len(wrong_predictions))
    for i in range(num_to_display):
        data, pred, target = wrong_predictions[i]
        plt.imshow(data.squeeze(), cmap='gray')
        plt.title('Predicted: {}, Actual: {}'.format(pred.item(), target.item()))
        plt.show()

    # plot the moving average of the test loss
    plt.plot(pd.Series(losses).rolling(window=100).mean())
    plt.title('Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


model = Net().to(device)

# Define the optimizer and the learning rate
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network for 5 epochs
train_losses = []
test_losses = []
for epoch in range(1, 2):

    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    plt.plot(train_losses, label='Training loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    xticks = plt.xticks()[0]
    xticks_labels = [int(label) * 64 for label in xticks]
    plt.xticks(xticks, xticks_labels)
    plt.show()
