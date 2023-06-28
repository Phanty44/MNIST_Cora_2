import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.nn import GCNConv, global_mean_pool, SplineConv, GMMConv, max_pool
from torch_geometric.data import Data, DataLoader
from torchvision import datasets, transforms
from torch_geometric.transforms import Cartesian

# Define the Graph Convolutional Network (GCN) architecture
import Plot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = Cartesian(cat=False)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=2, kernel_size=5)
        self.conv2 = GMMConv(64, 128, dim=2, kernel_size=5)

        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index, data.edge_attr))
        x = F.relu(self.conv2(x, edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Load the MNIST dataset as a graph
train_dataset = MNISTSuperpixels(root='.', train=True, transform=transform)
test_dataset = MNISTSuperpixels(root='.', train=False, transform=transform)

# trainM_dataset = datasets.MNIST('../data', train=True, download=True,
#                                 transform=transforms.Compose([
#                                     transforms.ToTensor(),
#                                     transforms.Normalize((0.5,), (0.5,))
#                                 ]))
#
# # Define the test dataset and the data loader
# testM_dataset = datasets.MNIST('../data', train=False,
#                                transform=transforms.Compose([
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5,), (0.5,))
#                                ]))

# trainM_data = []
# for image, label in trainM_dataset:
#     image = image[0]
#     pos = []
#     edges = []
#     edges2 = []
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             node_id = i * image.shape[1] + j
#             if i > 0:
#                 edges.append(node_id)
#                 edges2.append(node_id - image.shape[1])
#             if j > 0:
#                 edges.append(node_id)
#                 edges2.append(node_id - 1)
#             if i < image.shape[0] - 1:
#                 edges.append(node_id)
#                 edges2.append(node_id + image.shape[1])
#             if j < image.shape[1] - 1:
#                 edges.append(node_id)
#                 edges2.append(node_id + 1)
#             pos.append([i, j])
#     edge_index = torch.tensor([edges, edges2], dtype=torch.long)
#     x = torch.tensor(image.flatten(), dtype=torch.float).unsqueeze(1)
#     data = Data(x=x, edge_index=edge_index, y=label, pos=torch.tensor(pos, dtype=torch.int))
#     data = transform(data)
#     trainM_data.append(data)
#
# # Convert the entire MNIST test dataset into graph data
# y_test = []
# testM_data = []
# for image, label in testM_dataset:
#     image = image[0]
#     pos = []
#     edges = []
#     edges2 = []
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             node_id = i * image.shape[1] + j
#             if i > 0:
#                 edges.append(node_id)
#                 edges2.append(node_id - image.shape[1])
#             if j > 0:
#                 edges.append(node_id)
#                 edges2.append(node_id - 1)
#             if i < image.shape[0] - 1:
#                 edges.append(node_id)
#                 edges2.append(node_id + image.shape[1])
#             if j < image.shape[1] - 1:
#                 edges.append(node_id)
#                 edges2.append(node_id + 1)
#             pos.append([i, j])
#     edge_index = torch.tensor([edges, edges2], dtype=torch.long)
#     x = torch.tensor(image.flatten(), dtype=torch.float).unsqueeze(1)
#     data = Data(x=x, edge_index=edge_index, y=label, pos=torch.tensor(pos, dtype=torch.int))
#     data = transform(data)
#     y_test.append(label)
#     testM_data.append(data)


# print(trainM_data[0])
# print(testM_data[0])
# pickle_out = open("Test_GCN.pickle", "wb")
# pickle.dump(testM_data, pickle_out)
# pickle_out.close()
#
# pickle_out = open("Train_GCN.pickle", "wb")
# pickle.dump(trainM_data, pickle_out)
# pickle_out.close()
#
# pickle_out = open("Y_Test_GCN.pickle", "wb")
# pickle.dump(y_test, pickle_out)
# pickle_out.close()

# trainM_data = pickle.load(open("Train_GCN.pickle", "rb"))
# testM_data = pickle.load(open("Test_GCN.pickle", "rb"))
y_test = pickle.load(open("Y_Test_GCN.pickle", "rb"))

#trainM_loader = DataLoader(trainM_data, batch_size=64, shuffle=True)
#testM_loader = DataLoader(testM_data, batch_size=64, shuffle=False)

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

    return loss, out


# Evaluate the GCN model on the test set
def test(model, loader):
    model.eval()
    correct = 0
    predict = []
    for data in loader:
        out = model(data.cuda())
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        predict.extend(pred.cpu())
    return predict, correct / len(loader.dataset)


losses = []
for epoch in range(1, 11):
    loss, output = train(model, optimizer, train_loader)
    y_pred, test_acc = test(model, test_loader)
    print(f'Epoch: {epoch:03d}, Test Acc: {test_acc:.4f}, Loss: {loss:.4f}')

    if epoch % 2 == 0:
        losses.append(loss)

    if epoch == 10:
        print(classification_report(y_test, y_pred))
        Plot.plot_confusion(y_test, y_pred)
        Plot.training_loss(losses[1:])
