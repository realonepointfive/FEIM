import math
import random
import time
import networkx as nx
from collections import deque, defaultdict
import numpy as np
import heapq
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


def rr_generate(G, C, theta_c):
    """
    Algorithm 1: Generate RR sets for fair influence estimation
    
    Args:
        G: NetworkX directed graph with edge probabilities
        C: List of communities (each is a set of nodes)
        theta_c: List of integers - number of RR sets to generate for each community
    
    Returns:
        R: List of RR sets (each is a set of nodes)
        kappa: Dictionary mapping node -> community -> count of covered RR sets
        eta: Dictionary mapping node -> list of RR set indices it covers
        community_map: Dictionary mapping node -> its community index
    """
    # Create community mapping
    community_map = {}
    for c_idx, community in enumerate(C):
        for node in community:
            community_map[node] = c_idx
    
    # Initialize data structures
    R = []  # List of RR sets
    R_info = []
    kappa = defaultdict(lambda: defaultdict(int))  # kappa[v][c] = count
    eta = defaultdict(list)  # eta[v] = list of RR set indices
    
    # Generate RR sets for each community
    for c_idx, community in enumerate(C):
        num_rr_sets = theta_c[c_idx]
        
        for _ in range(num_rr_sets):
            # Select random root node from community c
            root = random.choice(list(community))
            
            # Generate RR set using reverse BFS
            rr_set = set()
            queue = deque([root])
            rr_set.add(root)
            
            while queue:
                current = queue.popleft()
                
                # Traverse incoming edges (reverse direction)
                for predecessor in G.predecessors(current):
                    if predecessor not in rr_set:
                        # Get edge probability
                        p = G[predecessor][current]['pp']
                        
                        # Include with probability p
                        if random.random() < p:
                            rr_set.add(predecessor)
                            queue.append(predecessor)
            
            # Add RR set to collection
            rr_index = len(R)
            R.append(rr_set)
            R_info.append(c_idx)
            eta[root].append(rr_index)
            
            # Update kappa and eta for all nodes in RR set
            for node in rr_set:
                kappa[node][c_idx] += 1
    
    return R, R_info, kappa, eta, community_map


def compute_eta_coefficient(n, alpha):
    """
    Compute η(n, α) coefficient for Taylor expansion
    """
    if n == 1:
        return 1.0
    else:
        product = 1.0
        for i in range(1, n):
            product *= (i - alpha)
        return product / math.factorial(n)
    

def compute_marginal_gain(v, phi, kappa, theta_c, n_c, alpha, Q):
    """
    Compute marginal gain of adding node v to seed set S
    """
    gain = 0.0
    
    for c in range(len(phi)):
        phi_c = phi[c]
        kappa_vc = kappa[v].get(c, 0)
        theta_c_val = theta_c[c]
        n_c_val = n_c[c]
        
        if theta_c_val == 0:
            continue
            
        series_diff = 0.0
        
        for n in range(1, Q + 1):
            if n > theta_c_val:
                break
                
            eta_n = compute_eta_coefficient(n, alpha)
            
            # Product term without v
            product_without = 1.0
            for i in range(n):
                if theta_c_val - i <= 0:
                    product_without = 0.0
                    break
                product_without *= (theta_c_val - phi_c - i) / (theta_c_val - i)
            
            # Product term with v
            product_with = 1.0
            for i in range(n):
                if theta_c_val - i <= 0:
                    product_with = 0.0
                    break
                product_with *= (theta_c_val - phi_c - kappa_vc - i) / (theta_c_val - i)
            
            series_diff += eta_n * (product_without - product_with)
        
        community_gain = alpha * n_c_val * series_diff
        gain += community_gain
    
    return gain


def estimate_theta_alpha_positive(n_G, C, k, epsilon, ell, Q, b0):
    """
    Estimate θ for α > 0 case (Theorem 1).
    """
    # Set δ1 = δ2 = 1/(2n_G)
    delta1 = delta2 = 1 / (2 * n_G)
    
    # Calculate τ1 and τ2
    tau1 = math.sqrt(math.log(C) + ell * math.log(n_G) + math.log(2))
    tau2_sq = tau1**2 + math.log(math.comb(n_G, k))  # n_G choose k
    tau2 = math.sqrt(tau2_sq)
    
    # Calculate ε1
    # Calculate θ1 and θ2 from Lemmas 3 and 4
    epsilon1 = epsilon * (math.e / (math.e - 1)) * (math.sqrt(3) * tau1) / (math.sqrt(3) * tau1 + math.sqrt(2) * tau2)
    theta1 = (12 * Q**2 * math.log(C / delta1)) / (epsilon1**2 * (1 - b0))
    
    # ε2 = (e/(e-1)) * ε - ε1
    epsilon2 = (math.e / (math.e - 1)) * epsilon - epsilon1
    theta2 = (8 * Q**2 * math.log(C * math.comb(n_G, k) / delta2)) / (epsilon2**2 * (1 - b0))
    
    # Total θ needed (Theorem 1)
    theta_total = C * max(theta1, theta2)
    
    # Alternative formula from the text (more precise)
    # θ = ((e-1)/e)^2 * (4CQ^2(√3τ1+√2τ2)^2) / (e^2(1-b0))
    theta_alt = ((math.e - 1) / math.e)**2 * (4 * C * Q**2 * (math.sqrt(3) * tau1 + math.sqrt(2) * tau2)**2) / (math.e**2 * (1 - b0))
    
    # Use the maximum of both estimates for safety
    theta = max(theta_total, theta_alt)
    
    # Distribute θ across communities (proportional to community size or equal)
    # Here we use equal distribution for simplicity
    theta_c = [int(theta / C)] * C
    
    return theta_c


def community_generation(G):
    susceptibility_scores = {}

    for node in G.nodes():
        susceptibility_scores[node] = G.in_degree(node)

    sorted_nodes = sorted(G.nodes(), key=lambda x: susceptibility_scores.get(x, 0))
    
    # Create communities by dividing the sorted list
    num_communities = 10
    communities = []
    n_nodes = len(sorted_nodes)
    nodes_per_community = n_nodes // num_communities
    
    for i in range(num_communities):
        start_idx = i * nodes_per_community
        if i == num_communities - 1:  # Last community gets remaining nodes
            end_idx = n_nodes
        else:
            end_idx = (i + 1) * nodes_per_community
        
        community_nodes = sorted_nodes[start_idx:end_idx]
        if i == num_communities - 1:
            print(sorted_nodes[start_idx])
        communities.append(set(community_nodes))
    return communities


def FairIMM(args, G):
    """
    Algorithm 2: Fair Influence Maximization using RR sets
    
    Args:
        G: NetworkX directed graph
        C: List of communities
        k: Budget (number of seeds to select)
        alpha: Aversion parameter
        Q: Taylor expansion truncation parameter
        theta_c: List of RR set counts per community
    
    Returns:
        S: Selected seed set
    """

    C = community_generation(G)
    theta_c = estimate_theta_alpha_positive(G.number_of_nodes(), C, args.t, 2, 2, 2, 0.9)

    # Step 1: Generate RR sets
    R, R_info, kappa, eta, community_map = rr_generate(G, C, theta_c)
    
    # Initialize variables
    phi = [0] * len(C)  # Coverage counts for each community
    covered = [False] * len(R)  # Whether each RR set is covered
    S = set()  # Seed set
    
    # Precompute n_c (number of nodes in each community)
    n_c = [len(community) for community in C]
    
    # Initialize marginal gains using max-heap
    # Heap elements: (-gain, node, gain) for max-heap behavior
    heap = []
    for node in G.nodes():
        gain = compute_marginal_gain(node, phi, kappa, theta_c, n_c, 0.5, 2)
        heapq.heappush(heap, (-gain, node, gain))
    
    # Lazy greedy selection
    for _ in range(args.t):
        while heap:
            # Get node with maximum marginal gain
            neg_gain, node, old_gain = heapq.heappop(heap)
            
            # Recompute marginal gain if needed
            current_gain = compute_marginal_gain(node, phi, kappa, theta_c, n_c, 0.5, 2)
            
            if current_gain == old_gain:
                # Add node to seed set
                S.add(node)
                
                # Update coverage for communities
                for c in range(len(C)):
                    phi[c] += kappa[node].get(c, 0)
                
                # Update covered RR sets
                for rr_index in eta[node]:
                    if not covered[rr_index]:
                        covered[rr_index] = True
                        rr_set = R[rr_index]
                        c_index = R_info[rr_index]
                        
                        # Find root community
                        # Since we don't store root explicitly, we need to infer it
                        # For simplicity, we'll update all nodes in the RR set
                        # In practice, we should track the root community during RR generation
                        for u in rr_set:
                            if u != node:
                                kappa[u][c_index] -= 1
                                # This is simplified - in actual implementation,
                                # we need to know which community this RR set belongs to
                                # We'll need to modify RR generation to store root information
                                pass
                
                # Recompute gains for remaining nodes
                new_heap = []
                while heap:
                    neg_gain, remaining_node, old_gain = heapq.heappop(heap)
                    new_gain = compute_marginal_gain(remaining_node, phi, kappa, theta_c, n_c, 0.5, 2)
                    heapq.heappush(new_heap, (-new_gain, remaining_node, new_gain))
                heap = new_heap
                break
            else:
                # Push back with updated gain
                heapq.heappush(heap, (-current_gain, node, current_gain))
        else:
            # No more nodes to select
            break
    
    return S
