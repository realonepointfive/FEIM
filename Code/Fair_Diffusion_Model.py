import os
import networkx as nx
import numpy as np
import random
import ast
from collections import namedtuple
from scipy.spatial.distance import cosine
import torch

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

class FairDiffusionModel():
    def __init__(self, **kwargs):
        args = kwargs["env_args"]
        if isinstance(args, dict):
            args = convert(args)
        self.args = args
        self.G = nx.Graph()
        self.l = self.args.msg_limit
        self.obs_l = self.args.obs_limit
        self.num_feats = self.args.feats_num
        self.neigh_num = self.args.neighbours_num
        self.seed_sets = []
        self.data_loader()
        self.cand_nodes = []
        self.cand_edges = []
        self.input_edges = []
        self.active_nodes = []
    

    def data_loader(self):
        curr_dir = os.path.dirname(__file__)
        self.data_path_prefix = os.path.normpath(os.path.join(curr_dir, self.args.rela_dir))
        self.SE_matrix = np.loadtxt(self.data_path_prefix + self.args.SE_matrix + self.args.data_path_suffix)
        self.msg_matrix = np.loadtxt(self.data_path_prefix + self.args.msg_matrix + self.args.data_path_suffix)
        with open(self.data_path_prefix + self.args.graph + self.args.data_path_suffix, 'r', encoding = 'utf-8') as file:
            index = 0
            for line in file:
                useridlist = line.split('</useridlist>')[0].split('<useridlist>')[-1].strip()
                edge_feat = self.msg_matrix[index]
                userid1, userid2 = useridlist.split(' ')
                if self.G.has_edge(userid1, userid2):
                    merged_vector = np.maximum(self.G.edges[userid1, userid2]['feat'], edge_feat)
                    self.G.edges[userid1, userid2]['feat'] = merged_vector
                else:
                    self.G.add_edge(userid1, userid2)
                    self.G.edges[userid1, userid2]['feat'] = edge_feat
                index += 1

        # self.node_list = list(self.G.nodes())

        with open(self.data_path_prefix + self.args.node_labels + self.args.data_path_suffix, 'r', encoding = 'utf-8') as file:
            for line in file:
                node, label = line.strip().split('\t')
                self.G.nodes[node]['h_SE'] = int(label)

        with open(self.data_path_prefix + self.args.seed_sets + self.args.data_path_suffix, 'r', encoding = 'utf-8') as file:
            for line in file:
                seeds = ast.literal_eval(line)
                self.seed_sets.append(seeds)


    def spread_restart(self):
        for node in self.G.nodes():
            self.G.nodes[node]['active'] = False
            self.G.nodes[node]['abandoned'] = False
            self.G.nodes[node]['m'] = 0
            self.G.nodes[node]['num_cand_edges'] = 0
        
        # random_seeds = random.sample(self.node_list, k=self.num_seeds)
        random_seeds = self.seed_sets[0]
        for seed in random_seeds:
            self.G.nodes[seed]['active'] = True


    def subevent_recommendation(self, input_vector, vector_list):
        input_vector = np.array(input_vector)
        vector_list = np.array(vector_list)
        similarities = np.array([1 - cosine(input_vector, row) for row in vector_list])
        sorted_indices = np.argsort(similarities)[::-1]
        return sorted_indices


    def search_cand_nodes(self):
        cand_nodes = [] 
        for node in self.active_nodes:
            cand_nodes_for_specified_node = [neighbor for neighbor in self.G.neighbors(node) if not (self.G.nodes[neighbor]['active'] or self.G.nodes[neighbor]['abandoned'])]
            cand_nodes.extend(cand_nodes_for_specified_node)
        cand_nodes = list(set(cand_nodes))
        self.cand_nodes = cand_nodes

    
    def search_cand_edges(self):
        cand_edges = [] 
        for node in self.active_nodes:
            cand_nodes_for_single_node = [neighbor for neighbor in self.G.neighbors(node) if not (self.G.nodes[neighbor]['active'] or self.G.nodes[neighbor]['abandoned'])]
            for cand_node in cand_nodes_for_single_node:
                cand_edges.append((node, cand_node))
        self.cand_edges = cand_edges


    def search_active_nodes(self):
        active_nodes = []
        for node in self.G.nodes():
            if self.G.nodes[node]['active'] == True:
                active_nodes.append(node)
        self.active_nodes = active_nodes


    def EdgeNeighborLoader(self):
        unique_one_hop_edges = []
        unique_two_hop_edges = []
        edges_dict = dict()
        potential_influence_paths = []
        adj_list = []
        message_senders = []
        message_receivers = []
        input_edges_feature_list = []
        one_hop_edges_feature_list = []
        two_hop_edges_feature_list = []
        index = 0
   
        if len(self.cand_edges) > self.obs_l:
            self.input_edges = random.sample(self.cand_edges, self.obs_l)
        else:
            self.input_edges = self.cand_edges
    
        for input_edge in self.input_edges:
            input_edges_feature_list.append(self.G.edges[input_edge]['feat'])
            sorted_edge = frozenset(input_edge)
            edges_dict[sorted_edge] = index
            index += 1
            cand_node = input_edge[1] 
            one_hop_neigh_nodes = list(self.G.neighbors(cand_node))
            one_hop_edges = []
            for one_hop_neigh_node in one_hop_neigh_nodes:
                if self.G.nodes[one_hop_neigh_node]['active'] == True or self.G.nodes[one_hop_neigh_node]['abandoned'] == True:
                    continue
                else:
                    one_hop_edges.append((cand_node, one_hop_neigh_node))
            if len(one_hop_edges) > self.neigh_num[0]:
                input_one_hop_edges = random.sample(one_hop_edges, self.neigh_num[0])
            else: 
                input_one_hop_edges = one_hop_edges

            # avoid repetitive edges
            for one_hop_edge in input_one_hop_edges:
                sorted_edge = frozenset(one_hop_edge)
                if sorted_edge not in unique_one_hop_edges:
                    unique_one_hop_edges.append(sorted_edge)
                one_hop_candidate_node = one_hop_edge[1]
                two_hop_neighbor_nodes = list(self.G.neighbors(one_hop_candidate_node))
                two_hop_edges = []
                # remove edges ending with the target_node or one-hop nodes of the target node 
                for two_hop_neighbor_node in two_hop_neighbor_nodes:
                    if self.G.nodes[two_hop_neighbor_node]['active'] == True or self.G.nodes[two_hop_neighbor_node]['abandoned'] == True:
                        continue
                    elif two_hop_neighbor_node == cand_node:
                        continue
                    else:
                        two_hop_edges.append((one_hop_candidate_node, two_hop_neighbor_node))
                if len(two_hop_edges) > self.neigh_num[1]:
                    input_two_hop_edges = random.sample(two_hop_edges, self.neigh_num[1])
                else: 
                    input_two_hop_edges = two_hop_edges

                for two_hop_edge in input_two_hop_edges:
                    sorted_edge = frozenset(two_hop_edge)
                    if sorted_edge not in unique_two_hop_edges:
                        unique_two_hop_edges.append(sorted_edge)
                    potential_influence_paths.append([frozenset(input_edge), frozenset(one_hop_edge), sorted_edge])
              
        for edge in unique_one_hop_edges:
            one_hop_edges_feature_list.append(self.G.edges[tuple(edge)]['feat'])
            edges_dict[edge] = index
            index += 1
    
        for edge in unique_two_hop_edges:
            if edge not in edges_dict:
                two_hop_edges_feature_list.append(self.G.edges[tuple(edge)]['feat'])
                edges_dict[edge] = index
                index += 1

        for i, path in enumerate(potential_influence_paths):
            if i > 0:
                previous_path = potential_influence_paths[i-1]
                if path[0] == previous_path[0] and path[1] == previous_path[1]:
                    continue
                else:
                    message_senders.append(edges_dict[path[1]])
                    message_receivers.append(edges_dict[path[0]])
            else:
                message_senders.append(edges_dict[path[1]])
                message_receivers.append(edges_dict[path[0]])

        for path in potential_influence_paths:
            message_senders.append(edges_dict[path[2]])
            message_receivers.append(edges_dict[path[1]])
                
        feature_list = input_edges_feature_list + one_hop_edges_feature_list + two_hop_edges_feature_list
        feature_matrix = np.array(feature_list)
        adj_list.append(message_senders)
        adj_list.append(message_receivers)
        adj_matrix = np.array(adj_list)
         
        return feature_matrix, adj_matrix, len(self.cand_edges)   

                
    def recommendation_triggered_by_an_action(self, action):
        edge_index = action.item()
        selected_edge = self.input_edges[edge_index]
        adj_edges = self.G.edges(selected_edge[1])
        # Filter out edges that are not connected to an active node
        cand_edges = [edge for edge in adj_edges if self.G.nodes[edge[1]]['active'] == True]
        self.G.nodes[selected_edge[1]]['num_cand_edges'] = len(cand_edges)
        sorted_subevents = self.subevent_recommendation(self.G.edges[selected_edge]['feat'], self.SE_matrix)
        # Get the indices of the top-l subevents
        top_subevents = [t for t in sorted_subevents[:self.l]]
        
        hitting_subevent = self.G.nodes[selected_edge[1]]['h_SE']

        if hitting_subevent in top_subevents:
            self.G.nodes[selected_edge[1]]['active'] = True
            subevent_index = top_subevents.index(hitting_subevent)
            num_msg = subevent_index + 1
            self.G.nodes[selected_edge[1]]['m'] = num_msg
        else:
            self.G.nodes[selected_edge[1]]['abandoned'] = True
            num_msg = -1
            self.G.nodes[selected_edge[1]]['m'] = num_msg 
        return num_msg


class MDPEnv(FairDiffusionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scenario = kwargs["env_args"]["scenario"]
        self.data_path_prefix = kwargs["env_args"]["data_path_prefix"]
        self.data_path_suffix = kwargs["env_args"]["data_path_suffix"]
        self.done = False
        self.num_edges_succ = []
        self.num_edges_fail = []
        self.effe = []
        self.effi = []
        self.num_msg_edge = []
        self.observation = []
        self.ep_msg_num = 0


    def results_recording(self):
        active_nodes = [node for node in self.G.nodes() if self.G.nodes[node]['active'] == True]
        abandoned_nodes = [node for node in self.G.nodes() if self.G.nodes[node]['abandoned'] == True]
        num_msg_list = [self.G.nodes[node]['m'] for node in active_nodes]

        msg_sum = 0
        for m in num_msg_list:
            msg_sum += int(m)
        msg_sum += self.l*(len(abandoned_nodes))

        self.effe.append(len(active_nodes)-len(abandoned_nodes))
        self.effi.append(float(msg_sum)/len(active_nodes))

        num_msg_edge_one_trial = []
        for node in active_nodes:
            num_msg_edge_one_trial.append((self.G.nodes[node]['m'], self.G.nodes[node]['num_cand_edges'], True))
        for node in abandoned_nodes:
            num_msg_edge_one_trial.append((self.l, self.G.nodes[node]['num_cand_edges'], False))
        self.num_msg_edge.append(num_msg_edge_one_trial)
        num_of_edges_for_succ_one_trial = [self.G.nodes[node]['num_cand_edges'] for node in active_nodes]
        num_of_edges_for_fail_one_trial = [self.G.nodes[node]['num_cand_edges'] for node in abandoned_nodes]
        self.num_edges_succ.append(num_of_edges_for_succ_one_trial)
        self.num_edges_fail.append(num_of_edges_for_fail_one_trial)


    def results_reservation(self, num_steps):
        effe = np.array(self.effe)
        exp_setting = self.algo + '_' + self.graph_name + '_' + self.seed_pattern
        np.savetxt(self.data_path_prefix + exp_setting + '_output\\effe' + str(num_steps) + self.data_path_suffix, effe, fmt='%d')
        effi = np.array(self.effi)
        np.savetxt(self.data_path_prefix + exp_setting + '_output\\effi' + str(num_steps) + self.data_path_suffix, effi, fmt='%f')

        with open(self.data_path_prefix + exp_setting + '_output\\num_edges_succ' + str(num_steps) + self.data_path_suffix, 'w') as file:
            for num_edges_succ_one_trial in self.num_edges_succ:
                for num_edges_succ in num_edges_succ_one_trial:
                    file.write(str(num_edges_succ) + '\n')
                file.write('*****' + '\n')

        with open(self.data_path_prefix + exp_setting + '_output\\num_edges_fail' + str(num_steps) + self.data_path_suffix, 'w') as file:
            for num_edges_fail_one_trial in self.num_edges_fail:
                for num_edges_fail in num_edges_fail_one_trial:
                    file.write(str(num_edges_fail) + '\n')
                file.write('*****' + '\n')

        with open(self.data_path_prefix +  exp_setting + '_output\\num_msg_edge' + str(num_steps) + self.data_path_suffix, 'w') as file:
            for num_msg_edge_one_trial in self.num_msg_edge:
                for num_msg_edge in num_msg_edge_one_trial:
                    file.write(str(num_msg_edge) + '\n')
                file.write('*****' + '\n')

        self.num_edges_succ = []
        self.num_edges_fail = []
        self.effe = []
        self.effi = []
        self.num_msg_edge = []  


    def obs_gen(self):
        feature_matrix, adj_matrix, num_cand_edges  = self.EdgeNeighborLoader()
        edge_feat = torch.from_numpy(feature_matrix).to(torch.float)
        row = torch.from_numpy(adj_matrix[0]).to(torch.long)
        col = torch.from_numpy(adj_matrix[1]).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        action_range = torch.tensor(num_cand_edges)
        self.observation = [edge_feat, edge_index, action_range]


    def step(self, action):
        """ Returns reward, terminated, info """
        num_msg = self.recommendation_triggered_by_an_action(action)
        if num_msg > 0:
            reward = float(1)/num_msg
            self.epi_msg_num += num_msg
        else:
            reward = 0
            self.epi_msg_num += self.l
        self.search_active_nodes()
        self.search_cand_edges()
        if len(self.cand_edges) == 0:
            self.done = True
      
        self.obs_gen()
        return self.observation, reward, self.done


    def reset(self):
        """ Returns initial observations and states"""
        self.done = False
        self.epi_msg_num = 0
        self.spread_restart()
        self.search_active_nodes()
        self.search_cand_edges()

        self.obs_gen()
        return self.observation


    def render(self):
        pass


    def close(self):
        pass


    def seed(self):
        pass


    def save_replay(self):
        pass


    def get_env_info(self):
        env_info = {"obs": self.observation}
        return env_info