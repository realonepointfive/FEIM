from math import ceil
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import SAGEConv, DenseSAGEConv, dense_diff_pool
from torch.distributions import Categorical
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def reset_grad2(t, requires_grad=None):
    tr = t.detach()
    tr.requires_grad = t.requires_grad if requires_grad is None else requires_grad
    return tr


class NodeProfile(nn.Module):
    def __init__(self, in_channels, hidden_channels, normalize=False):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, adj, mask=None):
        h = self.conv(x, adj)
        return h


class T_SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, feat, t_adj, n_adj, i, j, mask=None):
        x1 = self.conv1(feat, t_adj)[:j].relu()
        x1 = F.dropout(x1, p=0.5)
        x2 = self.conv2(x1, n_adj)[:i]
        return x2

    def inference(self, feat, t_adj, n_adj, i, j, mask=None):
        x1 = self.conv1(feat, t_adj)[:j].relu()
        x2 = self.conv2(x1, n_adj)[:i]
        return x2
    

class G_SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()
        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        if lin is True:
            self.lin = torch.nn.Linear(hidden_channels,
                                       out_channels)
        else:
            self.lin = None

    def forward(self, h, adj, mask=None):
        x0 = h.relu()
        x1 = self.conv1(x0, adj).relu().squeeze(0)
        if h.shape[0] > 1:
            x = self.bn1(x1)
        else:
            x = x1
        if self.lin is not None:
            x = self.lin(x).relu()
        return x
    

class Worker(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.phi = nn.Linear(feat_dim, 64)
        self.f_Wspace = nn.Sequential(
            nn.Linear(4*64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.value_function = nn.Linear(4*64, 1)

    def forward(self, h, cand_edge_feat, u_1, c_adj, j):
        cand_edge_embed = self.phi(cand_edge_feat)
        a_list = []
        for i in range(c_adj.shape[1]):
            node_index = c_adj[0][i]
            edge_index = c_adj[1][i]
            a_e = cand_edge_embed[edge_index-j]
            a_h = torch.mul(a_e, h[node_index])
            a_h = F.normalize(a_h, dim=0)
            a_vec = torch.cat((a_e.relu(), a_h, u_1[node_index]))
            a_list.append(a_vec)

        U = torch.stack(a_list, dim=0)
        a_values = self.f_Wspace(U).permute(1, 0)

        a_flat = U.mean(dim=0).unsqueeze(0)
        value = self.value_function(a_flat)
        return value, a_values
    

class Manager(nn.Module):
    def __init__(self, hidden_dim, feat_dim):
        super().__init__()
        num_clusters = 80
        self.gnn1_pool = G_SAGE(hidden_dim, hidden_dim, num_clusters)
        self.gnn1_embed = G_SAGE(hidden_dim, hidden_dim, hidden_dim, lin=False)
        self.gnn2_pool = G_SAGE(hidden_dim, hidden_dim, 20)
        self.gnn2_embed = G_SAGE(hidden_dim, hidden_dim, hidden_dim, lin=False)
        self.gnn3_embed = G_SAGE(hidden_dim, hidden_dim, hidden_dim, lin=False)
        self.lin = torch.nn.Linear(hidden_dim, feat_dim)
        self.value_function = nn.Linear(hidden_dim, 1)

    def forward(self, h, adj, mask):
        if h.shape[0] >= 100:
            s = self.gnn1_pool(h, adj).unsqueeze(0)
            y = self.gnn1_embed(h, adj).unsqueeze(0)
            y, adj, l1, e1 = dense_diff_pool(y, adj, s)
            s = self.gnn2_pool(y.squeeze(0), adj).unsqueeze(0)
            y = self.gnn2_embed(y.squeeze(0), adj).unsqueeze(0)
            y, adj, l2, e2 = dense_diff_pool(y, adj, s)
            l = (l1 + l2).to(device)
            e = (e1 + e2).to(device)
        elif h.shape[0] < 100 and h.shape[0] >= 25:
            s = self.gnn2_pool(h, adj).unsqueeze(0)
            y = self.gnn2_embed(h, adj).unsqueeze(0)
            y, adj, l2, e2 = dense_diff_pool(y, adj, s)
            l = l2.to(device)
            e = e2.to(device)
        else:
            y = h.unsqueeze(0)
            l = 0
            e = 0

        y = self.gnn3_embed(y.squeeze(0), adj).unsqueeze(0)
        y = y.mean(dim=1)
        z = self.lin(y)
        value = self.value_function(y)
        return value, z, l, e
    

class GFN(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gnn_node = NodeProfile(feat_dim, 64).to(device)
        self.x_embed = T_SAGE(feat_dim, 64, 64, lin=False).to(device)
        self.worker = Worker(feat_dim).to(device)
        self.manager = Manager(64, 64).to(device)
        print("GFN")

    def forward(self, obs, training, mask=None):
        feat = obs[0].to(device)
        e_adj = obs[2].to(device)
        t_adj = obs[3].to(device)
        n_adj = obs[4].to(device)
        g_adj = obs[5].to(device)
        c_adj = obs[6].to(device)
        cand_node_num = obs[8]
        input_node_num = obs[9]
        cand_edge_num = obs[10]

        h = self.gnn_node(feat, e_adj, mask)[:input_node_num]

        # u_1
        if training == True:
            x_2 = self.x_embed(feat, t_adj, n_adj, cand_node_num, input_node_num, cand_edge_num)
        else:
            x_2 = self.x_embed.inference(feat, t_adj, n_adj, cand_node_num, input_node_num, cand_edge_num)

        value_M, z, link_loss, ent_loss = self.manager(h, g_adj, mask)
        tran_grad = link_loss + ent_loss

        x_z = torch.mul(x_2, z).to(device)
        x_z = F.normalize(x_z, dim=1)
        u_1 = torch.cat((x_2.relu(), x_z), dim=1)

        cand_edge_feat = feat[input_node_num:(input_node_num + cand_edge_num)]
        value_W, a_values = self.worker(h, cand_edge_feat, u_1, c_adj, input_node_num)
        return value_W, value_M, a_values, tran_grad

    

