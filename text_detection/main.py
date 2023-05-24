import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, global_max_pool,global_add_pool, GlobalAttention, GraphMultisetTransformer

from torch_geometric.transforms import ToUndirected
from Mixup import *
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='politifact',
                    choices=['politifact', 'gossipcop'])
parser.add_argument('--feature', type=str, default='spacy',
                    choices=['profile', 'spacy', 'bert', 'content'])
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'SAGE','GIN'])
parser.add_argument('--epoch', type=int, default='100')
parser.add_argument('--alpha', type=float, default='0.1')
parser.add_argument('--mixture', type=str, default='mixup',
                    choices=['mixup','gmt','type1','type2','type3','type4','type5','type6','type7','type8','type9'])
args = parser.parse_args()

path = 'Please replace your path'
train_dataset = UPFD(path, args.dataset, args.feature, 'train', ToUndirected())
val_dataset = UPFD(path, args.dataset, args.feature, 'val', ToUndirected())
test_dataset = UPFD(path, args.dataset, args.feature, 'test', ToUndirected())

avg_num_nodes = int(train_dataset.data.num_nodes / len(train_dataset))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self, model, in_channels, hidden_channels, out_channels,
                 concat=False):
        super().__init__()
        self.concat = concat
        gate_nn = torch.nn.Sequential(torch.nn.Linear(hidden_channels, 2*hidden_channels), torch.nn.BatchNorm1d(2*hidden_channels), torch.nn.ReLU(), torch.nn.Linear(2*hidden_channels, 1))
        nn = torch.nn.Sequential(Lin(hidden_channels, hidden_channels), ReLU(), Lin(hidden_channels, hidden_channels))
        self.pool_cat = GlobalAttention(gate_nn,nn)
        self.pool = GlobalAttention(gate_nn,nn)
        self.pool_gmt = GraphMultisetTransformer(
            in_channels = 3*hidden_channels,
            hidden_channels =2*hidden_channels,
            out_channels = 1*hidden_channels,
            Conv=GCNConv,
            num_nodes=500,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False,
        )
   
        if model == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        elif model == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)

        elif model == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)
        elif model == 'GIN':
            nn_gin = torch.nn.Sequential(torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn_gin, train_eps=True)
            self.conv2 = GINConv(nn_gin, train_eps=True)
            self.conv3 = GINConv(nn_gin, train_eps=True)

        if self.concat:
            self.lin0 = Linear(in_channels, hidden_channels)
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)
        self.lin3 = Linear(2*hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        if args.mixture =='mixup':
            h_node = global_max_pool(h,batch)
            h = h_node
        elif args.mixture == 'gmt':
            h1 = self.conv1(x, edge_index).relu()
            h2 = self.conv2(h1, edge_index).relu() 
            h3 = self.conv3(h2, edge_index).relu() 
            h_all =torch.cat([h1, h2, h3], dim = -1)
            h_node = self.pool_gmt(h_all, batch, edge_index)
            h = h_node

        elif args.mixture == 'type1':
            h_node = global_max_pool(h, batch)
            h_graph = self.pool(h,batch)
            h = torch.add(h_node, h_graph)
        elif args.mixture == 'type2':
            h_node = global_max_pool(h, batch)
            h_graph = self.pool(h,batch)
            h = torch.mul(h_node, h_graph)
        elif args.mixture == 'type3':
            h_node = global_max_pool(h, batch)
            h_graph = self.pool_cat(h,batch)
            h = torch.cat((h_node, h_graph),dim=1)
            h = self.lin3(h)
        elif args.mixture == 'type4':
            h_node = global_add_pool(h, batch)
            h_graph = self.pool(h,batch)
            h = torch.add(h_node, h_graph)
        elif args.mixture == 'type5':
            h_node = global_add_pool(h, batch)
            h_graph = self.pool(h,batch)
            h = torch.mul(h_node, h_graph)
        elif args.mixture == 'type6':
            h_node = global_add_pool(h, batch)
            h_graph = self.pool_cat(h,batch)
            h = torch.cat((h_node, h_graph),dim=1)
            h = self.lin3(h)
        elif args.mixture == 'type7':
            h_node = global_max_pool(h, batch)
            h_graph = global_add_pool(h, batch)
            h = torch.add(h_node, h_graph)
        elif args.mixture == 'type8':
            h_node = global_max_pool(h, batch)
            h_graph = global_add_pool(h, batch)
            h = torch.mul(h_node, h_graph)
        elif args.mixture == 'type9':
            h_node = global_add_pool(h, batch)
            h_graph = global_max_pool(h, batch)
            h = torch.cat((h_node, h_graph),dim=1)
            h = self.lin3(h)

        if self.concat:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(args.model, train_dataset.num_features, 128,
            train_dataset.num_classes, concat=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        data.y = torch.nn.functional.one_hot(data.y, 2).view(-1, 2)
        out, data.y = mixup_data(out, data.y, alpha=args.alpha)
        log_prob = torch.nn.functional.log_softmax(out, dim=1)
        loss = -torch.sum(log_prob * data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = total_examples = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
        total_examples += data.num_graphs

    return total_correct / total_examples

best_score = 0
best_score_test = 0
for epoch in range(args.epoch):
    loss = train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    if val_acc > best_score:
        best_score = val_acc
        best_score_test = test_acc
    else:
        pass
    pass
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
print('-------------------------------------------')
print('Best_score_test:', best_score_test)
print('Bset_score_Val:', best_score)
