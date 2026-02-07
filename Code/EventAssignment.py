# import networkx as nx
import numpy as np
import random
# import numba
from math import radians, sin, cos, sqrt, atan2
# import os
random.seed(123)


def msg_p(msg_ps):
    p = 1
    for msg_p in msg_ps:
        p = p*(1 - msg_p)
    ap = 1-p
    return ap


def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in km
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Compute the distance
    distance = R * c
    return distance


def search_cand(diff_g, rr):
    """Return (cand_nodes, cand_edges): nodes reachable from active nodes (not active/abandoned)
    and all edges from active nodes to those candidates."""
    cand_nodes = []
    for node in diff_g.nodes():
        if diff_g.nodes[node]['active']:
            for neighbor in diff_g.successors(node):
                if not (diff_g.nodes[neighbor]['active'] or diff_g.nodes[neighbor]['abandoned']):
                    cand_nodes.append(neighbor)
    cand_nodes = list(set(cand_nodes))
    cand_edges = []
    for node in cand_nodes:
        for pred in diff_g.predecessors(node):
            if diff_g.nodes[pred]['active'] and not diff_g.edges[pred, node]['abandoned']:
                cand_edges.append((pred, node))
    return cand_nodes, cand_edges


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


'''@numba.jit(nopython=True)
def process_edges_batch_numba(edges, edge_probs_array, l):
    """
    Numba-accelerated batch processing of edges.
    """
    results = []
    for i in range(len(edges)):
        edge = edges[i]
        prob = edge_probs_array[i]
        successes = np.random.uniform(0, 1, l) < prob
        num_act = np.sum(successes)
        results.append((edge, num_act))
    return results'''


def EventInfluenceSimulation(args, diff_g, rr, seeds):
        bene = 0
        inf = 0
        msg = 0
        fair_score_list = []
        dist_list = []

        if args.data == 'NepalEQuake':
            x = 28.3973623
            y = 84.1257684
        elif args.data == 'TexasFlood':
            x = 31.169621
            y = -99.683617
        else:
            x = -14.2400732
            y = -53.1805017

        for _ in range(args.sim_num):
            max_bene = 0
            min_bene = args.l
            max_dist = 0

            for node in diff_g.nodes():
                diff_g.nodes[node]['active'] = False
                diff_g.nodes[node]['abandoned'] = False
                diff_g.nodes[node]['bene'] = 0
                diff_g.nodes[node]['reached'] = False

            for seed in seeds:
                diff_g.nodes[seed]['active'] = True

            for edge in diff_g.edges():
                diff_g.edges[edge]['abandoned'] = False

            cand_nodes, g_edges = search_cand(diff_g, rr)

            while len(cand_nodes):
                # Group edges by target node so each user gets top-l from all edges from active nodes
                edges_by_target = {}
                for (s_node, t_node) in g_edges:
                    edges_by_target.setdefault(t_node, []).append((s_node, t_node))

                for t_node, in_edges in edges_by_target.items():
                    diff_g.nodes[t_node]['reached'] = True
                    # ppd has length n (one prob per sub-event). Per sub-event, take max prob across
                    # active edges; then propagate only the l distinct sub-events with highest probs.
                    ppd_stack = np.array([rr.edges[e]['ppd'] for e in in_edges])
                    n_subevents = ppd_stack.shape[1]
                    max_per_subevent = np.max(ppd_stack, axis=0)
                    if n_subevents <= args.l:
                        top_l_probs = max_per_subevent
                    else:
                        top_l_idx = np.argsort(max_per_subevent)[-args.l:]
                        top_l_probs = max_per_subevent[top_l_idx]
                    num_act = np.sum(np.random.uniform(0, 1, len(top_l_probs)) < top_l_probs)

                    bene += num_act
                    msg += len(top_l_probs)

                    if num_act > 0:
                        diff_g.nodes[t_node]['active'] = True
                        inf += 1
                        diff_g.nodes[t_node]['bene'] = num_act
                        max_bene = max(num_act, max_bene)
                        min_bene = min(num_act, min_bene)

                        if 'x' in rr.nodes[t_node]:
                            dist = haversine(x, y, rr.nodes[t_node]['x'], rr.nodes[t_node]['y'])
                            max_dist = max(dist, max_dist)
                    else:
                        diff_g.nodes[t_node]['abandoned'] = True

                cand_nodes, g_edges = search_cand(diff_g, rr)

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

            dist_list.append(max_dist)


        msg_gap = np.mean(fair_score_list)
        node_num = rr.number_of_nodes() - len(seeds)
        ave_dist = np.mean(dist_list)
        return bene, msg, inf, msg_gap, node_num, ave_dist



'''def FES(rr, seeds):
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
    return diff_g'''

