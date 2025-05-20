import networkx as nx
import numpy as np
import random
random.seed(123)


def msg_p(msg_ps):
    p = 1
    for msg_p in msg_ps:
        p = p*(1 - msg_p)
    ap = 1-p
    return ap


def search_cand_nodes(flag, diff_g, rr):
    if flag == 'selection':
        cand_nodes = []
        for node in diff_g.nodes():
            cand_nodes_for_specified_node = [neighbor for neighbor in rr.successors(node) if not diff_g.has_node(neighbor)]
            cand_nodes.extend(cand_nodes_for_specified_node)
        cand_nodes = list(set(cand_nodes))
    else:
        cand_nodes = []
        for node in diff_g.nodes():
            if diff_g.nodes[node]['active'] == True:
                cand_nodes_for_specified_node = [neighbor for neighbor in diff_g.successors(node) if not (diff_g.nodes[neighbor]['active'] or diff_g.nodes[neighbor]['abandoned'])]
                cand_nodes.extend(cand_nodes_for_specified_node)
        cand_nodes = list(set(cand_nodes))
    return cand_nodes
            

def search_cand_edges(flag, cand_nodes, diff_g, rr):
    if flag == 'selection':
        cand_edges = []
        for node in cand_nodes:
            active_neighbors = [neighbor for neighbor in rr.predecessors(node) if diff_g.has_node(neighbor)]
            for active_node in active_neighbors:
                cand_edges.append((active_node, node))
    else:
        cand_edges = []
        for node in cand_nodes:
            active_neighbors = [neighbor for neighbor in diff_g.predecessors(node) if diff_g.nodes[neighbor]['active']]
            for active_node in active_neighbors:
                cand_edges.append((active_node, node))
    return cand_edges


def update_cand_graph(cand_g, g, diff_g, selected_node):
    remove_edge_list = list(cand_g.in_edges(selected_node))
    cand_g.remove_edges_from(remove_edge_list)
    cand_g.remove_node(selected_node)
    
    new_edges = list(g.out_edges(selected_node))
    for edge in new_edges:
        t_node = edge[1]
        if t_node not in diff_g:
            if t_node not in cand_g:
                cand_g.add_edge(selected_node, t_node)
                expected_bene = diff_g.nodes[selected_node]['inf'] * sum(g.edges[edge]['ppd'])
                cand_g.nodes[t_node]['max_bene'] = expected_bene
                cand_g.nodes[t_node]['optimal_neighbor'] = selected_node
            else:
                cand_g.add_edge(selected_node, t_node)
                expected_bene = diff_g.nodes[selected_node]['inf'] * sum(g.edges[edge]['ppd'])
                if expected_bene > cand_g.nodes[t_node]['max_bene']:
                    cand_g.nodes[t_node]['max_bene'] = expected_bene
                    cand_g.nodes[t_node]['optimal_neighbor'] = selected_node

    return cand_g


def EventInfluenceSimulation(args, diff_g, rr, seeds):
        i = 0
        bene = 0
        inf = 0
        msg = 0
        fair_score_list = []

        while(i<args.sim_num):
            max_bene = 0
            min_bene = args.l

            for node in diff_g.nodes():
                diff_g.nodes[node]['active'] = False
                diff_g.nodes[node]['abandoned'] = False
                diff_g.nodes[node]['bene'] = 0
                diff_g.nodes[node]['reached'] = False

            for seed in seeds:
                diff_g.nodes[seed]['active'] = True

            cand_nodes = search_cand_nodes('simulation', diff_g, rr)

            while(len(cand_nodes)):
                g_edges = []
                for node in cand_nodes:
                    g_node = []
                    g_node.append(node)
                    cand_edges = search_cand_edges('simulation', g_node, diff_g, rr)
                    g_edges.append(random.choice(cand_edges))
                
                for edge in g_edges:
                    t_node = edge[1]
                    diff_g.nodes[t_node]['reached'] = True
                    
                    msg_probs = rr.edges[edge]['ppd']
                    success = np.random.uniform(0, 1, args.l) < msg_probs
                    num_act = np.sum(success)
                    bene += num_act
                    msg += args.l

                    if num_act > 0:
                        diff_g.nodes[t_node]['active'] = True
                        inf += 1
                        diff_g.nodes[t_node]['bene'] = num_act
                        max_bene = max(num_act, max_bene)
                        min_bene = min(num_act, min_bene)
                    else:
                        diff_g.nodes[t_node]['abandoned'] = True
                cand_nodes = search_cand_nodes('simulation', diff_g, rr)

            i += 1
            bene_list = []
            for node in diff_g.nodes():
                if diff_g.nodes[node]['reached']:
                    bene_list.append(diff_g.nodes[node]['bene'])
            
            n = len(bene_list)
            if 2 * n * sum(bene_list) > 0:
                disparity_sum = sum(abs(bene_list[i] - bene_list[j]) for i in range(n) for j in range(n))
                fair_score = disparity_sum / (2 * n * sum(bene_list))
                if fair_score > 0:
                    fair_score_list.append(fair_score)

        msg_gap = np.mean(fair_score_list)
        return bene, msg, inf, msg_gap


def FES(rr, seeds):
    diff_g = nx.DiGraph()
    cand_g = nx.DiGraph()
    for seed in seeds:
        diff_g.add_node(seed)
        diff_g.nodes[seed]['inf'] = 1

    cand_nodes = search_cand_nodes('selection', diff_g, rr)
    cand_edges = search_cand_edges('selection', cand_nodes, diff_g, rr)
    for edge in cand_edges:
        cand_g.add_edge(edge[0], edge[1])

    for node in cand_g.nodes():
        if cand_g.in_degree(node)>0:
            cand_g.nodes[node]['max_bene'] = 0
            cand_g.nodes[node]['optimal_neighbor'] = None
            for edge in cand_g.in_edges(node):
                s_node = edge[0]
                expected_bene = diff_g.nodes[s_node]['inf'] * sum(rr.edges[s_node, node]['ppd'])
                if expected_bene > cand_g.nodes[node]['max_bene']:
                    cand_g.nodes[node]['max_bene'] = expected_bene
                    cand_g.nodes[node]['optimal_neighbor'] = s_node

    while(cand_g.number_of_edges()):
        max_margin_score = 0
        g_edge = None
        for node in cand_g.nodes():
            if cand_g.in_degree(node) > 0:
                margin_score = cand_g.nodes[node]['max_bene']
                if margin_score > max_margin_score:
                    g_edge = (cand_g.nodes[node]['optimal_neighbor'], node)
                max_margin_score = max(margin_score, max_margin_score)

        if g_edge == None:
            break
        else:
            diff_g.add_edge(g_edge[0], g_edge[1])
            diff_g.nodes[g_edge[1]]['inf'] = diff_g.nodes[g_edge[0]]['inf'] * msg_p(rr.edges[edge]['ppd'])

        cand_g = update_cand_graph(cand_g, rr, diff_g, g_edge[1])
    return diff_g
