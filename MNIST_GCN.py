import torch
import torch.nn.functional as F
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool

# Define the Graph Convolutional Network (GCN) architecture
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 64)
        self.conv2 = GCNConv(64, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_max_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.relu(x)

# Load the MNIST dataset as a graph
train_dataset = MNISTSuperpixels(root='.', train=True)
test_dataset = MNISTSuperpixels(root='.', train=False)

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
    train(model, optimizer, train_loader)
    test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}')
