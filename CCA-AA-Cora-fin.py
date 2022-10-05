import os
import random

import torch
import torch as th
from torch.optim import Adam, SGD
import numpy as np
import torch_geometric.transforms as T
from tqdm import tqdm
from torch import tensor
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx, dropout_adj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root=r'datas/Cora', name='Cora')


dataset.transform = T.NormalizeFeatures()
dataset

data = dataset[0].to(device)
data

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing, GCNConv

def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)

        self.bn = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        self.act_fn = nn.ReLU()

    def forward(self, _, x):
        x = self.layer1(x)
        if self.use_bn:
            x = self.bn(x)

        x = self.act_fn(x)
        x = self.layer2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers):
        super().__init__()

        self.n_layers = n_layers
        self.convs = nn.ModuleList()

        self.convs.append(GCNConv(in_dim, hid_dim))

        if n_layers > 1:
            for i in range(n_layers - 2):
                self.convs.append(GCNConv(hid_dim, hid_dim))
            self.convs.append(GCNConv(hid_dim, out_dim))

    def forward(self, x, edge_index):

        for i in range(self.n_layers - 1):
            x = F.relu(self.convs[i](x, edge_index))
        x = self.convs[-1](x, edge_index)

        return x


class CCA_AA(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, use_mlp=False):
        super().__init__()
        if not use_mlp:
            self.backbone = GCN(in_dim, hid_dim, out_dim, n_layers)
        else:
            self.backbone = MLP(in_dim, hid_dim, out_dim)

    def get_embedding(self, x, edge_index):
        out = self.backbone(x, edge_index)
        return out.detach()

    def forward(self, x_1, edge_index_1, x_2, edge_index_2):
        h1 = self.backbone(x_1, edge_index_1)
        h2 = self.backbone(x_2, edge_index_2)

        z1 = (h1 - h1.mean(0)) / h1.std(0)
        z2 = (h2 - h2.mean(0)) / h2.std(0)

        return z1, z2


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w.repeat(x.size(0)).view(x.size(0), -1)

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[drop_mask] = 0.

    return x


def drop_feature_weighted_2(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights


def train(model, optimizer, drop_edg_p1, drop_edg_p2, drop_fea_p1, drop_fea_p2):
    drop_weights = degree_drop_weights(data.edge_index).to(device)  # 根据度计算删除重要性
    edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edg_p1, threshold=0.7)  # view_1_edge_index
    edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edg_p2, threshold=0.7)  # view_2_edge_index
    edge_index_ = to_undirected(data.edge_index)
    node_deg = degree(edge_index_[1])
    feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_fea_p1)  # view_1_x
    x_2 = drop_feature_weighted_2(data.x, feature_weights, drop_fea_p2)  # view_2_x
    model.train()
    optimizer.zero_grad()
    z1, z2 = model(x_1, edge_index_1)
    c = th.mm(z1.T, z2)
    c1 = th.mm(z1.T, z1)
    c2 = th.mm(z2.T, z2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N

    loss_inv = -th.diagonal(c).sum()
    iden = th.tensor(np.eye(c.shape[0])).to(device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

    loss.backward()
    optimizer.step()
def seed_torch(seed=12345):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

lambd = 0.001

lr = 0.001

weight_decay = 0
hid_dim = 512
out_dim = 512
n_layers = 2  # GCN
use_mlp = False
drop_edg_p1=0.3
drop_edg_p2 = 0.7

drop_fea_p1 = 0.4
drop_fea_p2 = 0.6
runs =100
epochs = 152
early_stopping = 10
best_max = 0.0

model_acc = []
pbar = tqdm(range(runs), unit='run')
N = dataset[0].x.shape[0]
for _ in pbar:

    seed_torch(438)

    model = CCA_AA(dataset.num_features, hid_dim, out_dim, n_layers, use_mlp).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        drop_weights = degree_drop_weights(data.edge_index).to(device)
        edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edg_p1,
                                          threshold=0.7 )  # view_1_edge_index
        edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=drop_edg_p2,
                                          threshold=0.7 )  # view_2_edge_index
        edge_index_ = to_undirected(data.edge_index)
        # edge_index_1 = dropout_adj(data.edge_index, p=drop_edg_p1)[0]//消融实验
        # edge_index_2 = dropout_adj(data.edge_index, p=drop_edg_p2)[0]

        node_deg = degree(edge_index_[1])
        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        x_1 = drop_feature_weighted_2(data.x, feature_weights, drop_fea_p1)  # view_1_x
        x_2 = drop_feature_weighted_2(data.x, feature_weights, drop_fea_p2)  # view_2_x
        # x_1 = drop_feature(data.x, drop_fea_p1)  # view_1_x//消融实验
        # x_2 = drop_feature(data.x, drop_fea_p2)  # view_2_x
        model.train()
        optimizer.zero_grad()
        z1, z2 = model(x_1, edge_index_1, x_2, edge_index_2)
        c = th.mm(z1.T, z2)
        c1 = th.mm(z1.T, z1)
        c2 = th.mm(z2.T, z2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -th.diagonal(c).sum()
        iden = th.tensor(np.eye(c.shape[0])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        embeds = model.get_embedding(data.x, data.edge_index)
    accs = []

    best_val_acc = 0
    eval_acc = 0

    for _ in range(3):
        log = LogReg(out_dim, dataset.num_classes)
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.00011)
        log.cuda()
        pat_steps = 0
        best_acc = torch.zeros(1)
        best_acc = best_acc.cuda()
        best_val_loss = float('inf')
        val_loss_history = []

        for _ in range(200):
            log.train()
            opt.zero_grad()

            logits = log(embeds)
            loss = torch.nn.CrossEntropyLoss()(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()
            val_loss = torch.nn.CrossEntropyLoss()(logits[data.val_mask], data.y[data.val_mask])
            test_acc = th.sum(th.argmax(logits[data.test_mask], dim=1) == data.y[data.test_mask]).float() / \
                       data.y[data.test_mask].shape[0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = test_acc

            val_loss_history.append(val_loss)
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

        accs.append(best_acc * 100)

    accs = torch.stack(accs)
    model_acc.append(accs.mean())
    model_acc_ = torch.stack(model_acc)

    if(model_acc_.mean().item()>best_max):
        best_max=model_acc_.mean().item()
print(model_acc_.mean(),end=",")
print('best_max:{:.2f}'.format(best_max),end=",")
print(loss,end=",")
print(model_acc_.std())
