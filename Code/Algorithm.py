import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import SAGEConv


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


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
    

def reset_grad2(t, requires_grad=None):
    tr = t.detach()
    tr.requires_grad = t.requires_grad if requires_grad is None else requires_grad
    return tr


class Worker(nn.Module):
    def __init__(self, d, k, action_range):
        super(Worker, self).__init__()
        self.k = k
        self.action_range = action_range
        self.f_Wspace = nn.Linear(d, k)

        self.g = nn.Sequential(
            nn.Linear(d, k, bias=False),
        )
        
        self.value_function = nn.Linear(action_range * k, 1)
        self.worker_partial_loss = nn.CosineEmbeddingLoss(reduction='none').to(device)

    def forward(self, o, sum_g_W, sim_thre, reset_value_grad):
        w = self.f_Wspace(sum_g_W) # [batch x k]
        a_flat = o.view(-1, self.action_range * self.k)
        a_w = o.squeeze(0) # [a x k]
        action_values = self.worker_partial_loss(a_w, w, - torch.ones(w.size(0)).to(device)) # [a]
        action_values.unsqueeze(0) # [batch x a]
        work_range = torch.sum(action_values >= sim_thre)
        if reset_value_grad:
            value = self.value_function(reset_grad2(a_flat))
        else:
            value = self.value_function(a_flat)

        return value, action_values, work_range
    
    def reset_states_grad(self, states_W):
        hx, cx = states_W
        return list(map(reset_grad2, hx)), list(map(reset_grad2, cx))


class Manager(nn.Module):
    def __init__(self, d, k, action_range):
        super(Manager, self).__init__()

        self.f_Mspace = nn.Sequential(
            nn.Linear(action_range * k, d),
            nn.ReLU()
        )

        self.f_Mrnn = nn.LSTMCell(d, d)
        
        self.value_function = nn.Linear(d, 1)

    def forward(self, o, states_M, reset_value_grad):
        # action_range * 50
        z = self.f_Mspace(o)  # action_range * 200
        g_hat, cx = states_M = self.f_Mrnn(z, states_M)

        g = F.normalize(g_hat)
        
        if reset_value_grad:
            value = self.value_function(reset_grad2(g_hat))
        else:
            value = self.value_function(g_hat)

        return value, g, z, states_M

    def reset_states_grad(self, states_M):
        hx, cx = states_M
        return list(map(reset_grad2, hx)), list(map(reset_grad2, cx))


class FeudalNet(nn.Module):
    def __init__(self, feat_dim, hidden_dim, d=200, k=50, h=4, action_range=1000):
        super(FeudalNet, self).__init__()
        self.d, self.k, self.h, self.action_range = d, k, h, action_range
        self.g_queue = []
        self.z_queue = []
        self.a_range = []
        self.w_range = []
        self.edge_embed = SAGE(feat_dim, hidden_dim).to(device)
        self.worker = Worker(d, k, action_range).to(device)
        self.manager = Manager(d, k, action_range).to(device)
        self.manager_partial_loss = nn.CosineEmbeddingLoss().to(device)

    def init_weights(self):
        """all submodules are already initialized like this"""
        def default_init(m):
            """Default is a uniform distribution"""
            for module_type in [nn.Linear, nn.LSTMCell]:
                if isinstance(m, module_type):
                    m.reset_parameters()
        self.apply(default_init)

    def forward(self, states, states_M, states_W, reset_value_grad=False):
        o_batch = torch.empty((0, self.action_range, self.k), device=device)
        edge_feat = states[0]
        edge_index = states[1]
        action_range = states[2]

        o = self.edge_embed(edge_feat.to(device), edge_index.to(device))[:action_range].to(device)
        num_rows_to_add = self.action_range - action_range
        zeros_to_add = torch.zeros((num_rows_to_add, self.k), device=device)
        expanded_o = torch.cat((o, zeros_to_add), dim=0)
        o_batch = torch.cat((o_batch, expanded_o.unsqueeze(0)), dim=0)

        value_manager, g, z, states_M = self.manager(o_batch.view(-1, self.action_range * self.k), states_M, reset_value_grad)
        z_prev = self.z_queue[0]
        g_prev = self.g_queue[0]
        sim_thre = self.manager_partial_loss((z - z_prev), g_prev, - torch.ones(g_prev.size(0)).to(device))
        
        self.g_queue.pop(0)
        self.g_queue.append(g)
        self.z_queue.pop(0)
        self.z_queue.append(z)
        
        g_W = torch.stack(self.g_queue, dim=0)
        # sum on h different gt values, note that gt = normalize(hx)
        sum_goal = sum(map(F.normalize, g_W))
        sum_goal_W = reset_grad2(sum_goal, requires_grad=True)

        value_worker, action_values, work_range = self.worker(o_batch, sum_goal_W, sim_thre, reset_value_grad)
        w_range = torch.stack(self.w_range, dim=0)
        a_range = torch.stack(self.a_range, dim=0)
        tran_grad = torch.log(torch.prod(w_range/a_range))
        self.a_range.pop(0)
        self.a_range.append(action_range)
        self.w_range.pop(0)
        self.w_range.append(work_range)

        return value_worker, value_manager, action_values, g, tran_grad, states_M

    def init_state(self, batch_size):
        self.z_queue = [torch.zeros(batch_size, self.d, requires_grad=False).to(device) for _ in range(self.h)]
        self.g_queue = [torch.zeros(batch_size, self.d, requires_grad=False).to(device) for _ in range(self.h)]
        self.a_range = [torch.tensor(1).to(device) for _ in range(self.h)]
        self.w_range = [torch.tensor(1).to(device) for _ in range(self.h)]
        states_M = (torch.zeros(batch_size, self.d).to(device), torch.zeros(batch_size, self.d).to(device))
        return states_M

    def reset_states_grad(self, states_M):
        return self.manager.reset_states_grad(states_M)

    def _intrinsic_reward(self, influence_path):
        rI = torch.zeros(1, 1).to(device)
        for i in range(self.h):
            g_i = F.normalize(self.g_queue[i]) 
            rI += F.cosine_similarity(influence_path, g_i)
        return rI / self.h
