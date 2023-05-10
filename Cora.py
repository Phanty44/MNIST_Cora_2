import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GraphConv, TopKPooling, global_mean_pool, Sequential
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report

import Plot

class GraphWeaveNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GraphWeaveNet, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, 64)
        self.conv2 = GraphConv(64, 64)
        self.conv3 = GraphConv(64, 128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc2 = torch.nn.Linear(128, dataset.num_classes)

    def forward(self, dataset):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#Optimizer parameters
learning_rate = 0.01
decay = 5e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='/tmp/cora', name='Cora')
data = dataset[0].to(device)

model = GraphWeaveNet(dataset).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask | data.val_mask], data.y[data.train_mask | data.val_mask])
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, out

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    acc = pred[data.test_mask] == data.y[data.test_mask]
    return acc, out

losses = []
for epoch in range(1, 101):
    loss, outTrain = train()
    accTab, outTest = test()
    acc = accTab.sum().item() / data.test_mask.sum().item()
    losses.append(loss)
    print(f'Epoch: {epoch:03d}, Test Acc: {acc:.4f}, Loss: {loss:.4f}')
    if epoch%100 == 0:
        y_pred = outTest.argmax(dim=1)[data.test_mask].detach().numpy()
        y_test = data.y[data.test_mask].detach().numpy()
        print(classification_report(y_test, y_pred))
        Plot.plot_confusion(y_test, y_pred)

Plot.training_loss(losses[1:])