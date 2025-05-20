import random
import time
import numpy as np
random.seed(123)
np.random.seed(123)


def msg_p(msg_ps):
    p = 1
    for msg_p in msg_ps:
        p = p*(1 - msg_p)
    ap = 1-p
    return ap
    
    
def jaccard_similarity(set1, set2):  
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def update_r_dict(r_dict, lam):
    for s_node in r_dict.copy():
        for t_node in r_dict[s_node].copy():
            if r_dict[s_node][t_node] < lam:
                del r_dict[s_node][t_node]
    return r_dict


def convert_p_to_lam(p, r_dict, n):
    weights = []
    for s_node in r_dict:
        for t_node in r_dict[s_node]:
            weights.append(r_dict[s_node][t_node])

    total_weights_num = n * n
    zero_weights_num = total_weights_num - len(weights)

    if (1-p) * total_weights_num <= zero_weights_num:
        lam = 0
    else:
        sorted_weights = sorted(weights, reverse=True)
        lam = sorted_weights[int(p * total_weights_num)]
    return lam


def rrs_log(args, time):
    folder_path = args.data_path_prefix.format(args.data) + '/{}/k{}l{}p{}'.format(args.algo, args.t, args.l, args.p)
    if time != None:
        with open(folder_path + '/RRCost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(str(time))


def ReachableRangeSearch(args, g, p):
        start_time = time.time()

        node_list = list(g.nodes())
        n = len(node_list)
        r_dict = dict()

        i = 1
        for s_node in node_list:
            if g.out_degree(s_node) == 0:
                continue
            else:
                r_dict[s_node] = dict()
                for t_node in g.successors(s_node):
                    r_dict[s_node][t_node] = g.edges[s_node, t_node]['pp']

        lam = convert_p_to_lam(p, r_dict, n)
        r_dict = update_r_dict(r_dict, lam)
        
        i += 1
        max_iter = np.ceil(np.log2(n))
        
        while (i <= max_iter+1):
            for s_node in r_dict:
                for t_node in r_dict[s_node].copy():
                    for q_node in g.successors(t_node):
                        pp = r_dict[s_node][t_node] * g.edges[t_node, q_node]['pp']
                        if q_node in r_dict[s_node]:
                            r_dict[s_node][q_node] = max(pp, r_dict[s_node][q_node])
                        else:
                            r_dict[s_node][q_node] = pp
            
            lam = convert_p_to_lam(p, r_dict, n)
            r_dict = update_r_dict(r_dict, lam)
            i += 1
        
        r_size = 0
        r_nodes = set()
        
        best_r = []
        tem_r = set()
        seeds = []
        seed = None
        
        for _ in range(args.t):
            for s_node in r_dict:
                s_node_list = [t_node for t_node in r_dict[s_node]]
                tem_r = r_nodes.union(set(s_node_list))
                if len(tem_r) > r_size:
                    r_size = len(tem_r)
                    best_r = s_node_list
                    seed = s_node
            seeds.append(seed)
            r_nodes = r_nodes.union(set(best_r))

        remain_nodes = list(r_nodes) + seeds
        end_time = time.time()
        cost_time = end_time - start_time

        rr = g.subgraph(remain_nodes)
        rrs_log(args, cost_time)
        return seeds, rr


def SocialUpdate(args, G, Event_tokens_for_update):
        start_time = time.time()
        for edge in G.edges():
            for i in range(len(Event_tokens_for_update)):
                SE_Frame = Event_tokens_for_update[i]
                SE_Frame = set(SE_Frame)
                sim = np.log2(jaccard_similarity(SE_Frame, set(G.edges[edge]['feat']))+1)

                G.edges[edge]['SE_num'] += 1
                if len(G.edges[edge]['sim']) < args.l:
                    G.edges[edge]['sim'].append(sim)
                    if sim < G.edges[edge]['min_sim']:
                        G.edges[edge]['min_sim'] = sim
                else:
                    if sim > G.edges[edge]['min_sim']:
                        min_index = G.edges[edge]['sim'].index(G.edges[edge]['min_sim'])
                        G.edges[edge]['sim'].pop(min_index)
                        G.edges[edge]['sim'].append(sim)
                        G.edges[edge]['min_sim'] = min(G.edges[edge]['sim'])

            ppd = []
            if G.edges[edge]['flag'] == True:
                for x in G.edges[edge]['sim']:
                    if 2 * x >= 1:
                        ppd.append(1)
                    else:
                        ppd.append(2 * x)
            else:
                ppd = G.edges[edge]['sim']
                G.edges[edge]['ppd'] = ppd
                G.edges[edge]['pp'] = msg_p(ppd)

        end_time = time.time()
        update_cost = end_time-start_time
        
        folder_path = args.data_path_prefix.format(args.data) + '/{}/k{}l{}p{}'.format(args.algo, args.t, args.l, args.p)
        with open(folder_path + '/SUCost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(str(update_cost))
        return G