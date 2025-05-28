import os
from re import A
import networkx as nx
import numpy as np
import random
import ast
from collections import namedtuple
from scipy.spatial.distance import cosine
import torch
import wandb

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class FairDiffusionModel():
    def __init__(self, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.G = nx.Graph()
        self.sg = nx.Graph()
        self.uisg = nx.Graph()
        self.l = self.args.msg_limit
        self.t = self.args.seeds_num
        self.num_feats = self.args.feats_num
        self.num_seeds = self.args.seeds_num
        self.neigh_d = self.args.neighbourhood_depth
        self.training_epi = self.args.training_epi
        print("Neighbourhood depth {}".format(self.neigh_d))
        self.data_loader()
        self.inf = 0
        self.cand_nodes = []
        self.cand_edges = []
        self.input_nodes = []
        self.input_edges = []
        self.active_nodes = []
        self.aband_nodes = []
        self.cand_node_vec = []
        self.cand_edge_vec = []
        self.cand_edge_dict = dict()
        self.seeds = self.args.seeds


    def data_loader(self):
        curr_dir = os.path.dirname(__file__)
        self.data_path_prefix = os.path.normpath(os.path.join(curr_dir, self.args.rela_dir))
        self.msg_matrix = np.loadtxt(self.data_path_prefix + self.args.msg_matrix + self.args.data_path_suffix)

        with open(self.data_path_prefix + self.args.graph + self.args.data_path_suffix, 'r', encoding = 'utf-8') as file:
            index = 0
            for line in file:
                useridlist = line.split('</useridlist>')[0].split('<useridlist>')[-1].strip()
                edge_feat = self.msg_matrix[index]
                userid1, userid2 = useridlist.split(' ')
                if self.G.has_edge(userid1, userid2):
                    merged_vector = (self.G.edges[userid1, userid2]['feat'] + edge_feat)/2
                    self.G.edges[userid1, userid2]['feat'] = merged_vector
                else:
                    self.G.add_edge(userid1, userid2)
                    self.G.edges[userid1, userid2]['feat'] = edge_feat
                index += 1

    
    def spread_restart(self, epoch, episode):
        self.active_nodes = []
        self.aband_nodes = []
        self.input_nodes = []
        self.cand_nodes = []
        self.cand_edges = []
        SE_mat_training = np.loadtxt(self.data_path_prefix + self.args.SE_mat_training + self.args.data_path_suffix)
        SE_mat_eval = np.loadtxt(self.data_path_prefix + self.args.SE_mat_eval + self.args.data_path_suffix)
        self.SE_mat = np.concatenate((SE_mat_training, SE_mat_eval[:episode-self.training_epi]), axis=0)
        sg_edges = []
        self.inf = 0

        with open(self.data_path_prefix + self.args.subgraph.format(episode) + self.args.data_path_suffix, 'r', encoding = 'utf-8') as file:
            for line in file:
                node1, node2 = line.split(' ')
                node2 = node2.rstrip('\n')
                sg_edges.append((node1, node2))

        self.sg = self.G.edge_subgraph(sg_edges)
        sg_nodes = list(self.sg.nodes())

        for node in self.sg.nodes():
            self.sg.nodes[node]['active'] = False
            self.sg.nodes[node]['abandoned'] = False
            self.sg.nodes[node]['m'] = 0
            self.sg.nodes[node]['num_cand_edges'] = 0

        seed_nodes = []
        if self.seeds == 'None':
            np.random.seed(episode)
            seed_indices = np.random.randint(0, len(sg_nodes), self.num_seeds)
            for seed_index in seed_indices:
                seed_nodes.append(sg_nodes[seed_index.item()])
        else:
            with open(self.data_path_prefix + self.seeds.format(self.t, episode) + '.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    seed_nodes = ast.literal_eval(line)
                    break

        for node in seed_nodes:
            self.sg.nodes[node]['active'] = True
            self.active_nodes.append(node)


    def msg_recom(self, input_vector, vector_list):
        input_vector = np.array(input_vector)
        vector_list = np.array(vector_list)
        similarities = np.array([1 - cosine(input_vector, row) for row in vector_list])
        sorted_indices = np.argsort(similarities)[::-1]
        act_probs = (similarities - similarities.min())/(1-similarities.min())
        return sorted_indices, act_probs


    def search_cand_nodes(self):
        cand_nodes = []
        for node in self.active_nodes:
            cand_nodes_for_specified_node = [neighbor for neighbor in self.sg.neighbors(node) if not (self.sg.nodes[neighbor]['active'] or self.sg.nodes[neighbor]['abandoned'])]
            cand_nodes.extend(cand_nodes_for_specified_node)
        cand_nodes = list(set(cand_nodes))
        self.cand_nodes = cand_nodes


    def search_cand_edges(self):
        cand_edges = []
        for node in self.cand_nodes:
            active_neighbors = [neighbor for neighbor in self.sg.neighbors(node) if self.sg.nodes[neighbor]['active']]
            for active_node in active_neighbors:
                cand_edges.append(frozenset((node, active_node)))
        self.cand_edges = cand_edges


    def search_k_hop_uisg(self):
        one_hop_ui_nodes = []
        two_hop_ui_nodes = []
        three_hop_ui_nodes = []
        ui_nodes = []

        one_hop_ui_nodes = self.cand_nodes
        ui_nodes += one_hop_ui_nodes
        for node in one_hop_ui_nodes:
            for two_hop_node in self.sg.neighbors(node):
                if not self.sg.nodes[two_hop_node]['active'] and not self.sg.nodes[two_hop_node]['abandoned'] and two_hop_node not in ui_nodes:
                    two_hop_ui_nodes.append(two_hop_node)
        two_hop_ui_nodes = list(set(two_hop_ui_nodes))
        ui_nodes += two_hop_ui_nodes
        for node in two_hop_ui_nodes:
            for three_hop_node in self.sg.neighbors(node):
                if not self.sg.nodes[three_hop_node]['active'] and not self.sg.nodes[three_hop_node]['abandoned'] and three_hop_node not in ui_nodes:
                    three_hop_ui_nodes.append(three_hop_node)
        three_hop_ui_nodes = list(set(three_hop_ui_nodes))
        ui_nodes += three_hop_ui_nodes
        self.input_nodes = ui_nodes

        incident_edges = []
        for node in ui_nodes:
            connected_edges = self.sg.edges(node)
            incident_edges.extend(connected_edges)
        incident_edges = list(set(incident_edges))
        self.uisg = self.sg.edge_subgraph(incident_edges)


    def n_hop_adjmat(self, input_nodes, nodes_dict, neigh_d):
        msg_s = []
        msg_r = []
        one_hop_msg_s = []
        one_hop_msg_r = []
        two_hop_msg_s = []
        two_hop_msg_r = []
        three_hop_msg_s = []
        three_hop_msg_r = []
        adj_list = []
            
        for node in input_nodes:
            one_hop_neigh = []
            for one_hop_node in self.sg.neighbors(node):
                if self.sg.nodes[one_hop_node]['active'] == False and self.sg.nodes[one_hop_node]['abandoned'] == False:
                    one_hop_neigh.append(one_hop_node)
                    one_hop_msg_s.append(nodes_dict[one_hop_node])
                    one_hop_msg_r.append(nodes_dict[node])
            for one_hop_node in one_hop_neigh:
                for two_hop_node in self.sg.neighbors(one_hop_node):
                    if self.sg.nodes[two_hop_node]['active'] == False and self.sg.nodes[two_hop_node]['abandoned'] == False:
                        if two_hop_node not in one_hop_neigh and two_hop_node != node:
                            two_hop_msg_s.append(nodes_dict[two_hop_node])
                            two_hop_msg_r.append(nodes_dict[one_hop_node])
        msg_s = one_hop_msg_s + two_hop_msg_s
        msg_r = one_hop_msg_r + two_hop_msg_r

        adj_list.append(msg_s)
        adj_list.append(msg_r)
        adj_mat = np.array(adj_list)
        return adj_mat
    

    def two_hop_diff_tree(self, cand_nodes, nodes_dict, edges_dict):
        msg_s = []
        msg_r = []

        one_hop_msg_s = []
        one_hop_msg_r = []
        two_hop_msg_s = []
        two_hop_msg_r = []

        adj_list = []

        for node in cand_nodes:
            one_hop_neigh = []
            for one_hop_node in self.sg.neighbors(node):
                if self.sg.nodes[one_hop_node]['active'] == False and self.sg.nodes[one_hop_node]['abandoned'] == False:
                    one_hop_neigh.append(one_hop_node)
                    one_hop_edge = (node, one_hop_node)
                    one_hop_msg_r.append(nodes_dict[node])
                    one_hop_msg_s.append(edges_dict[frozenset(one_hop_edge)])

            for one_hop_node in one_hop_neigh:
                for two_hop_node in self.sg.neighbors(one_hop_node):
                    if self.sg.nodes[two_hop_node]['active'] == False and self.sg.nodes[two_hop_node]['abandoned'] == False and two_hop_node != node:
                        two_hop_edge = (one_hop_node, two_hop_node)
                        two_hop_msg_r.append(nodes_dict[one_hop_node])
                        two_hop_msg_s.append(edges_dict[frozenset(two_hop_edge)])

        msg_s = one_hop_msg_s + two_hop_msg_s
        msg_r = one_hop_msg_r + two_hop_msg_r
        adj_list.append(msg_s)
        adj_list.append(msg_r)
        adj_mat = np.array(adj_list)
        return adj_mat
    

    def NeighborLoader(self):
        input_edges_feat_list = []
        input_nodes_feat_list = []
        input_edges = []
        edges_dict = dict()
        nodes_dict = dict()
        adj_list = []
        msg_s = []
        msg_r = []
        index = 0

        for node in self.input_nodes:
            input_nodes_feat_list.append(np.zeros(self.num_feats))
            nodes_dict[node] = index
            index += 1

        for edge in self.cand_edges:
            input_edges_feat_list.append(self.sg.edges[tuple(edge)]['feat'])
            edges_dict[edge] = index
            input_edges.append(edge)
            index += 1

        for edge in self.uisg.edges():
            if frozenset(edge) in self.cand_edges:
                continue
            else:
                input_edges_feat_list.append(self.sg.edges[edge]['feat'])
                sorted_edge = frozenset(edge)
                edges_dict[sorted_edge] = index
                input_edges.append(sorted_edge)
                index += 1
        self.input_edges = input_edges

        for node in self.input_nodes:
            linked_edges = list(self.sg.edges(node))
            for edge in linked_edges:
                sorted_edge = frozenset(edge)
                msg_s.append(edges_dict[sorted_edge])
                msg_r.append(nodes_dict[node])

        feat_list = input_nodes_feat_list + input_edges_feat_list
        feat_mat = np.array(feat_list)
        adj_list.append(msg_s)
        adj_list.append(msg_r)
        adj_mat_1 = np.array(adj_list)

        adj_mat_2 = self.two_hop_diff_tree(self.cand_nodes, nodes_dict, edges_dict)
        t_feat_mat = None
        target_index = None
        adj_mat_3 = self.n_hop_adjmat(self.cand_nodes, nodes_dict, 1)
        adj_mat_4 = nx.adjacency_matrix(self.uisg, self.input_nodes)

        msg_s = []
        msg_r = []
        adj_list = []
        for edge in self.cand_edges:
            node1, node2 = tuple(edge)
            if node1 in self.cand_nodes:
                msg_s.append(nodes_dict[node1])
            else:
                msg_s.append(nodes_dict[node2])
            msg_r.append(edges_dict[edge])
        adj_list.append(msg_s)
        adj_list.append(msg_r)
        adj_mat_5 = np.array(adj_list)

        return feat_mat, t_feat_mat, adj_mat_1, adj_mat_2, adj_mat_3, adj_mat_4, adj_mat_5, target_index, len(self.cand_nodes), len(self.input_nodes), len(self.cand_edges)
    

    def reward_signal(self, nodes, action_index):
        num_msgs = []
        if action_index == None:
            nodes = list(set(nodes))
            for node in nodes:
                self.sg.nodes[node]['active'] = True
                self.active_nodes.append(node)
            self.seeds=nodes
            num_msg = 0
            num_msgs.append(num_msg)
        else:
            action_index = action_index.tolist()
            if isinstance(action_index, list):
                actions = []
                for a_index in action_index:
                    action = self.input_edges[a_index]
                    actions.append(action)
            else:
                action = self.input_edges[action_index]
                actions = []
                actions.append(action)

            for action in actions:
                edge = tuple(action)
                if edge[0] in self.cand_nodes:
                    node = edge[0]
                else:
                    node = edge[1]

                if self.sg.nodes[node]['active'] == True or self.sg.nodes[node]['abandoned'] == True:
                    continue
                else:
                    sorted_msgs, act_probs = self.msg_recom(self.sg.edges[edge]['feat'], self.SE_mat)
                    top_msgs = [m for m in sorted_msgs[:self.l]]

                    num_msg = 0
                    top_msg_probs = act_probs[top_msgs]
                    success = np.random.uniform(0, 1, self.l) < top_msg_probs
                    num_act = np.sum(success)
                    self.inf += num_act

                    if num_act > 0:
                        self.sg.nodes[node]['active'] = True
                        self.active_nodes.append(node)
                    else:
                        self.sg.nodes[node]['abandoned'] = True
                        self.aband_nodes.append(node)
                    num_msgs.append(self.l)
        return num_msgs, num_act
    

class MDPEnv(FairDiffusionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.data_path_suffix = kwargs["env_args"]["data_path_suffix"]
        self.done = False
        self.num_edges_succ = []
        self.num_edges_fail = []
        self.effe = []
        self.effi = []
        self.num_msg_edge = []
        self.observation = []
        self.ep_msg_num = 0
        self.diff_sg = []
        self.save_dir = str(wandb.run.dir)    


    def obs_gen(self):
        feat_mat, _, adj_mat_1, adj_mat_2, adj_mat_3, adj_mat_4, adj_mat_5, target_index, cand_node_num, input_node_num, cand_edge_num = self.NeighborLoader()

        edge_feat = torch.from_numpy(feat_mat).to(torch.float)
        t_edge_feat = None

        row = torch.from_numpy(adj_mat_1[0]).to(torch.long)
        col = torch.from_numpy(adj_mat_1[1]).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        row = torch.from_numpy(adj_mat_2[0]).to(torch.long)
        col = torch.from_numpy(adj_mat_2[1]).to(torch.long)
        t_index = torch.stack([row, col], dim=0)

        row = torch.from_numpy(adj_mat_3[0]).to(torch.long)
        col = torch.from_numpy(adj_mat_3[1]).to(torch.long)
        node_index = torch.stack([row, col], dim=0)

        adj_mat_4 = adj_mat_4.todense()
        graph_index = torch.from_numpy(adj_mat_4).to(torch.float)

        row = torch.from_numpy(adj_mat_5[0]).to(torch.long)
        col = torch.from_numpy(adj_mat_5[1]).to(torch.long)
        cand_index = torch.stack([row, col], dim=0)

        self.observation = [edge_feat, t_edge_feat, edge_index, t_index, node_index, graph_index, cand_index, target_index, cand_node_num, input_node_num, cand_edge_num]


    def step(self, node, edge_index):
        """ Returns reward, terminated, info """
        num_msgs, num_act = self.reward_signal(node, edge_index)
        reward = 0
        self.ep_msg_num += (len(num_msgs) * self.l)

        self.search_cand_nodes()
        self.search_cand_edges()
        self.search_k_hop_uisg()

        if len(self.cand_nodes) == 0:
            self.done = True
            self.observation = None
        else:
            self.obs_gen()
        return self.observation, reward, self.done


    def reset(self, epoch, episode):
        """ Returns initial observations and states"""
        self.done = False
        self.ep_msg_num = 0
        self.spread_restart(epoch, episode)

        self.search_cand_nodes()
        self.search_cand_edges()
        self.search_k_hop_uisg()

        self.obs_gen()
        return self.observation
