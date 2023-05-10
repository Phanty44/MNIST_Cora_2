import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.nn import GCNConv, global_max_pool
from torch_geometric.data import Data, DataLoader
from torchvision import datasets, transforms

# Define the Graph Convolutional Network (GCN) architecture
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 256)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.relu(x)

# Load the MNIST dataset as a graph
train_dataset = MNISTSuperpixels(root='.', train=True)
test_dataset = MNISTSuperpixels(root='.', train=False)
print(train_dataset[0])

trainM_dataset = datasets.MNIST('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))
                               ]))


# Define the test dataset and the data loader
testM_dataset = datasets.MNIST('../data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,))
                              ]))

trainM_data = []
for image, label in trainM_dataset:
    edges = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            node_id = i * image.shape[1] + j
            if i > 0:
                edges.append([node_id, node_id - image.shape[1]])
            if j > 0:
                edges.append([node_id, node_id - 1])
            if i < image.shape[0] - 1:
                edges.append([node_id, node_id + image.shape[1]])
            if j < image.shape[1] - 1:
                edges.append([node_id, node_id + 1])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(image.flatten(), dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, y=label)
    trainM_data.append(data)

# Convert the entire MNIST test dataset into graph data
testM_data = []
for image, label in testM_dataset:
    edges = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            node_id = i * image.shape[1] + j
            if i > 0:
                edges.append([node_id, node_id - image.shape[1]])
            if j > 0:
                edges.append([node_id, node_id - 1])
            if i < image.shape[0] - 1:
                edges.append([node_id, node_id + image.shape[1]])
            if j < image.shape[1] - 1:
                edges.append([node_id, node_id + 1])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(image.flatten(), dtype=torch.float).unsqueeze(1)
    data = Data(x=x, edge_index=edge_index, y=label)
    testM_data.append(data)

# Create PyTorch Geometric DataLoader objects
trainM_loader = DataLoader(trainM_data, batch_size=64, shuffle=True)
testM_loader = DataLoader(testM_data, batch_size=64, shuffle=False)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the GCN model and optimizer
model = GCN().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the GCN model
def train(model, optimizer, train_loader):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.cuda())
        loss = F.cross_entropy(out, data.y.cuda())
        loss.backward()
        optimizer.step()

# Evaluate the GCN model on the test set
def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.cuda())
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

for epoch in range(1, 20):
    train(model, optimizer, trainM_loader)
    test_acc = test(model, testM_loader)
    print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')

