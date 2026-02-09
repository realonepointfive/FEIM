import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
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


def search_cand(diff_g, l):
    """
    Return candidate edges from currently active nodes to neighbors that have
    received fewer than l sub-events so far, and whose connecting edge has not
    yet been used to send sub-events.
    """
    cand_edges = []
    for node in diff_g.nodes():
        if diff_g.nodes[node].get('active'):
            for neighbor in diff_g.successors(node):
                if (
                    diff_g.nodes[neighbor].get('msg_num', 0) < l
                    and not diff_g.edges[node, neighbor].get('used', False)
                ):
                    cand_edges.append((node, neighbor))
    return cand_edges


def EventInfluenceSimulation(args, diff_g, seeds):
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
                diff_g.nodes[node]['bene'] = 0
                diff_g.nodes[node]['reached'] = False
                diff_g.nodes[node]['msg_num'] = 0

            for seed in seeds:
                diff_g.nodes[seed]['active'] = True

            # Initialize per-edge usage flag: each edge can be used only once
            for edge in diff_g.edges():
                diff_g.edges[edge]['used'] = False

            g_edges = search_cand(diff_g, args.l)

            while len(g_edges):
                # Group edges by target node so each user gets top-l from all edges from active nodes.
                # Two cases for each t_node:
                #   (1) diff_g is original rr: each edge has n sub-events; we select top-l across them.
                #   (2) diff_g is event-optimized G_prime: each edge has a (possibly different) number
                #       of assigned sub-events and the total across in_edges is <= l; we just use them.
                edges_by_target = {}
                for (s_node, t_node) in g_edges:
                    edges_by_target.setdefault(t_node, []).append((s_node, t_node))

                for t_node, in_edges in edges_by_target.items():
                    diff_g.nodes[t_node]['reached'] = True
                    # Build per-edge sub-event probability lists
                    edge_ppds = [np.asarray(diff_g.edges[e]['ppd']) for e in in_edges]
                    total_len = sum(len(p) for p in edge_ppds)

                    if total_len > args.l:
                        # Case 1: original rr graph where each edge has the same n sub-events.
                        # Combine into a matrix and select top-l sub-events across all in-edges.
                        ppd_stack = np.vstack(edge_ppds)
                        n_subevents = ppd_stack.shape[1]
                        max_per_subevent = np.max(ppd_stack, axis=0)
                        if n_subevents <= args.l:
                            top_l_probs = max_per_subevent
                        else:
                            top_l_idx = np.argsort(max_per_subevent)[-args.l:]
                            top_l_probs = max_per_subevent[top_l_idx]
                    else:
                        # Case 2: event-optimized G_prime where each edge already stores its
                        # assigned sub-events and the total count across all in-edges is <= l.
                        # Just use these directly.
                        if total_len == 0:
                            top_l_probs = np.array([])
                        else:
                            top_l_probs = np.concatenate(edge_ppds)

                    # Mark all incoming edges as used so they won't be selected again
                    for e in in_edges:
                        diff_g.edges[e]['used'] = True

                    # Update how many sub-events this node has received so far
                    diff_g.nodes[t_node]['msg_num'] += len(top_l_probs)

                    num_act = np.sum(np.random.uniform(0, 1, len(top_l_probs)) < top_l_probs)

                    bene += num_act
                    msg += len(top_l_probs)

                    if num_act > 0:
                        diff_g.nodes[t_node]['active'] = True
                        inf += 1
                        diff_g.nodes[t_node]['bene'] = num_act
                        max_bene = max(num_act, max_bene)
                        min_bene = min(num_act, min_bene)

                        if 'x' in diff_g.nodes[t_node]:
                            dist = haversine(x, y, diff_g.nodes[t_node]['x'], diff_g.nodes[t_node]['y'])
                            max_dist = max(dist, max_dist)

                g_edges = search_cand(diff_g, args.l)

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
        node_num = diff_g.number_of_nodes() - len(seeds)
        ave_dist = np.mean(dist_list)
        return bene, msg, inf, msg_gap, node_num, ave_dist


def EventAssignment(args, rr, seeds):
    """
    Event Assignment for Event-Optimized Graph (EASoG).
    Returns event-optimized graph G' with same V, E as G and node attribute 'ap' (activation probability).
    Inf_dist is taken as edge 'ppd'; sub-event assignment selects top-l sub-events by pp_dist per edge.
    """
    # Build G' with same structure as G; use shallow copy if available
    
    G_prime = type(rr)()
    G_prime.add_nodes_from(rr.nodes())
    G_prime.add_edges_from(rr.edges())

    for n in G_prime.nodes():
        # Initialize activation probability
        G_prime.nodes[n]['ap'] = 0.0
        # Copy spatial information from rr if available
        if 'x' in rr.nodes[n]:
            G_prime.nodes[n]['x'] = rr.nodes[n]['x']
        if 'y' in rr.nodes[n]:
            G_prime.nodes[n]['y'] = rr.nodes[n]['y']

    for u, v in G_prime.edges():
        G_prime.edges[u, v]['ppd'] = []

    Vstd = set(seeds)
    for u in seeds:
        G_prime.nodes[u]['ap'] = 1.0

    DeltaV = set()
    for u in seeds:
        for v in rr.successors(u):
            if v not in Vstd:
                DeltaV.add(v)

    while DeltaV:
        for v in list(DeltaV):
            E_vstd = [(u, v) for u in rr.predecessors(v) if u in Vstd]

            # PPdist(u,v) = ap(u) * ppd(u,v); sub-event assignment: top-l by this distribution
            ppd_stack = np.array([
                G_prime.nodes[u]['ap'] * np.asarray(rr.edges[u, v]['ppd'])
                for (u, v) in E_vstd
            ])
            n_subevents = ppd_stack.shape[1]
            max_per_subevent = np.max(ppd_stack, axis=0)
            argmax_per_subevent = np.argmax(ppd_stack, axis=0)
            if n_subevents <= args.l:
                top_l_idx = np.arange(n_subevents)
                top_l_probs = max_per_subevent
            else:
                top_l_idx = np.argsort(max_per_subevent)[-args.l:]
                top_l_probs = max_per_subevent[top_l_idx]
            # Store top-l sub-events on each edge: which sub-events are assigned to this edge and their ppd
            for i, (u, v) in enumerate(E_vstd):
                assigned_j = [int(j) for j in top_l_idx if argmax_per_subevent[j] == i]
                G_prime.edges[u, v]['ppd'] = ppd_stack[i, assigned_j].tolist()
            # Ensemble activation: ap(v) from assigned top-l sub-events
            G_prime.nodes[v]['ap'] = msg_p(top_l_probs)

        Vstd |= DeltaV
        DeltaV_prev = DeltaV
        DeltaV = set()
        for u in DeltaV_prev:
            for v in rr.successors(u):
                if v not in Vstd:
                    DeltaV.add(v)

    return G_prime