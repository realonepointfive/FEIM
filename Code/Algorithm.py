import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SAGE(torch.nn.Module):
    # in_channels = feature_dimension
    # out_channels = num_classes
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(feat_dim, hidden_dim).to(device))
        self.convs.append(SAGEConv(hidden_dim, feat_dim).to(device))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_() 
                # Replace all negative values in the tensor with zeros while leaving the non-negative values unchanged.
                x = F.dropout(x, p=0.5, training=self.training)
        return x

class DQN(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(DQN, self).__init__()
        self.edge_embed = SAGE(feat_dim, hidden_dim).to(device)
        self.layer1 = nn.Linear(feat_dim, 128).to(device)
        self.layer2 = nn.Linear(128, 128).to(device)
        self.layer3 = nn.Linear(128, 1).to(device)

    def forward(self, states):
        x_batch = torch.empty((0, 1000, 50), device=device)
        if isinstance(states, tuple):
            for state in states:
                edge_feat = state[0]
                edge_index = state[1]
                action_range = state[2]
                x = self.edge_embed(edge_feat.to(device), edge_index.to(device))[:action_range].to(device)
                num_rows_to_add = 1000 - x.shape[0]
                zeros_to_add = torch.zeros((num_rows_to_add, x.shape[1]), device=device)
                expanded_x = torch.cat((x, zeros_to_add), dim=0)
                x_batch = torch.cat((x_batch, expanded_x.unsqueeze(0)), dim=0)
        else:
            edge_feat = states[0]
            edge_index = states[1]
            action_range = states[2]
            x = self.edge_embed(edge_feat.to(device), edge_index.to(device))[:action_range].to(device)
            num_rows_to_add = 1000 - x.shape[0]
            zeros_to_add = torch.zeros((num_rows_to_add, x.shape[1]), device=device)
            expanded_x = torch.cat((x, zeros_to_add), dim=0)
            x_batch = torch.cat((x_batch, expanded_x.unsqueeze(0)), dim=0)
        y_batch = F.relu(self.layer1(x_batch))
        y_batch = F.relu(self.layer2(y_batch))
        return self.layer3(y_batch).squeeze(dim=2)
