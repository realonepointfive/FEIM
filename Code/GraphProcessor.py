import random
import time
import os
import numpy as np
from collections import deque
random.seed(123)
np.random.seed(123)
    
    
def jaccard_similarity(set1, set2):  
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def msg_p(msg_ps):
    p = 1
    for msg_p in msg_ps:
        p = p*(1 - msg_p)
    ap = 1-p
    return ap


'''def update_r_dict(r_dict, lam, full_size, iteration):
    kept_num = 0
    if iteration == 1:
        for s_node in r_dict.copy():
            for t_node in r_dict[s_node].copy():
                if r_dict[s_node][t_node] <= lam:
                    del r_dict[s_node][t_node]
            kept_num += len(r_dict[s_node])
        loss = (full_size - kept_num) / full_size
    else:
        removed_num = 0
        for s_node in r_dict.copy():
            kept_num += len(r_dict[s_node])
            for t_node in r_dict[s_node].copy():
                if r_dict[s_node][t_node] <= lam:
                    del r_dict[s_node][t_node]
                    removed_num += 1
        loss = removed_num / kept_num
    return r_dict, loss'''


def update_r_dict(r_dict, lam, full_size, iteration):
    removed_r = 0
    ori_r = 0
    
    for s_node in r_dict.copy():
        for t_node in r_dict[s_node].copy():
            ori_r += r_dict[s_node][t_node]
            if r_dict[s_node][t_node] <= lam:
                removed_r += r_dict[s_node][t_node]
                del r_dict[s_node][t_node]

    loss = removed_r / ori_r
    return r_dict, loss


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


def rrs_log(args, time, rr=None):
    folder_path = args.data_path_prefix.format(args.data) + '/RRSp/l{}p{}q{}eps{}'.format(
        args.l,
        args.p,
        format(getattr(args, "hrq", 0.95), "g"),
        format(getattr(args, "eps", 1e-16), "g"),
    )
    if time != None:
        os.makedirs(folder_path, exist_ok=True)
        with open(folder_path + '/Cost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(f"cost\t{time}\n")
            if rr is not None:
                file.write(f"nodes\t{rr.number_of_nodes()}\n")
                file.write(f"edges\t{rr.number_of_edges()}\n")
                print(
                    f"RRSparse time_signal={getattr(args,'time_signal','NA')} "
                    f"range_nodes={rr.number_of_nodes()} range_edges={rr.number_of_edges()}"
                )
    '''if rr is not None:
        with open(folder_path + '/RR_{}.txt'.format(args.time_signal), 'w', encoding='utf-8') as file:
            for u, v in rr.edges():
                edge_data = rr.edges[u, v]
                sim_idx = edge_data.get('se_idx', edge_data.get('sim_idx', []))
                ppd = edge_data.get('ppd', [])
                file.write(f"{u}\t{v}\t{sim_idx}\t{ppd}\n")'''


def _rrs_folder_path(args):
    return args.data_path_prefix.format(args.data) + '/RRSp/l{}p{}q{}eps{}'.format(
        args.l,
        args.p,
        format(getattr(args, "hrq", 0.95), "g"),
        format(getattr(args, "eps", 1e-16), "g"),
    )


def _dump_rr_and_seeds(args, k, rr, seeds, elapsed_from_rr_start=None):
    folder_path = _rrs_folder_path(args)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, f"Seeds_{k}.txt"), "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(f"{s}\n")
    with open(os.path.join(folder_path, f"RR_{k}.txt"), "w", encoding="utf-8") as f:
        for u, v in rr.edges():
            edge_data = rr.edges[u, v]
            sim_idx = edge_data.get('se_idx', edge_data.get('sim_idx', []))
            ppd = edge_data.get('ppd', [])
            f.write(f"{u}\t{v}\t{sim_idx}\t{ppd}\n")
    if elapsed_from_rr_start is not None:
        with open(os.path.join(folder_path, f"RRSeedStoreCost_{args.time_signal}.txt"), "a", encoding="utf-8") as f:
            f.write(f"k\t{k}\telapsed\t{elapsed_from_rr_start}\n")


def information_loss(g, p):
    node_list = list(g.nodes())
    n = len(node_list)
    r_dict = dict()
    loss_list = []
        
    i = 1
    for s_node in node_list:
        if g.out_degree(s_node) == 0:
            continue
        else:
            r_dict[s_node] = dict()
            for t_node in g.successors(s_node):
                r_dict[s_node][t_node] = msg_p(g.edges[s_node, t_node]['ppd'])
        
    lam = convert_p_to_lam(p, r_dict, n)
    r_dict, loss = update_r_dict(r_dict, lam, n * n, i)
    loss_list.append(loss)
    print(f"InfoLoss iter={i} loss={loss}")

    i += 1
    max_iter = np.ceil(np.log2(n))    
    while (i <= max_iter+1):
        for s_node in r_dict:
            for t_node in r_dict[s_node].copy():
                for q_node in g.successors(t_node):
                    pp = r_dict[s_node][t_node] * msg_p(g.edges[t_node, q_node]['ppd'])
                    if q_node in r_dict[s_node]:
                        r_dict[s_node][q_node] = max(pp, r_dict[s_node][q_node])
                    else:
                        r_dict[s_node][q_node] = pp
            
        lam = convert_p_to_lam(p, r_dict, n)
        r_dict, loss = update_r_dict(r_dict, lam, n * n, i)
        loss_list.append(loss)
        print(f"InfoLoss iter={i} loss={loss}")
        i += 1
    print(f"InfoLoss mean={np.mean(loss_list)}")


def ReachableRangeSearch(args, g):
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
                r_dict[s_node][t_node] = msg_p(g.edges[s_node, t_node]['ppd'])

    lam = convert_p_to_lam(args.p, r_dict, n)
    r_dict, _ = update_r_dict(r_dict, lam, n * n, i)
        
    i += 1
    max_iter = np.ceil(np.log2(n))
    iter_start = time.time()
        
    while (i <= max_iter+1):
        for s_node in r_dict:
            for t_node in r_dict[s_node].copy():
                for q_node in g.successors(t_node):
                    pp = r_dict[s_node][t_node] * msg_p(g.edges[t_node, q_node]['ppd'])
                    if q_node in r_dict[s_node]:
                        r_dict[s_node][q_node] = max(pp, r_dict[s_node][q_node])
                    else:
                        r_dict[s_node][q_node] = pp
            
        lam = convert_p_to_lam(args.p, r_dict, n)
        r_dict, _ = update_r_dict(r_dict, lam, n * n, i)
        done = i - 1
        total = int(max_iter)
        elapsed = max(time.time() - iter_start, 1e-9)
        rate = done / elapsed
        remain = max(0, total - done)
        eta = remain / max(rate, 1e-9)
        pct = 100.0 * done / max(total, 1)
        print(f"ReachableRangeSearch time_signal={getattr(args,'time_signal','NA')} iter: {done}/{total} ({pct:.1f}%), {rate:.2f} iter/s, ETA {eta:.1f}s")
        i += 1
        
    r_size = 0
    r_nodes = set()
        
    seeds = []
    seed_start = time.time()
        
    checkpoints = {5, 10, 15, 20}
    for seed_i in range(args.t):
        best_r = []
        tem_r = set()
        seed = None
        best_size = -1
        for s_node in r_dict:
            if s_node in seeds:
                continue
            s_node_list = [t_node for t_node in r_dict[s_node]]
            tem_r = r_nodes.union(set(s_node_list))
            if len(tem_r) > best_size:
                best_size = len(tem_r)
                best_r = s_node_list
                seed = s_node
        if seed is None:
            break
        seeds.append(seed)
        r_nodes = r_nodes.union(set(best_r))
        r_size = max(r_size, len(r_nodes))
        if (seed_i + 1) in checkpoints:
            remain_nodes = list(r_nodes.union(set(seeds)))
            rr_snapshot = g.subgraph(remain_nodes)
            _dump_rr_and_seeds(args, seed_i + 1, rr_snapshot, seeds, elapsed_from_rr_start=(time.time() - start_time))
        elapsed = max(time.time() - seed_start, 1e-9)
        rate = (seed_i + 1) / elapsed
        remain = args.t - (seed_i + 1)
        eta = remain / max(rate, 1e-9)
        pct = 100.0 * (seed_i + 1) / max(args.t, 1)
        print(f"ReachableRangeSearch time_signal={getattr(args,'time_signal','NA')} seeds: {seed_i + 1}/{args.t} ({pct:.1f}%), {rate:.2f} seeds/s, ETA {eta:.1f}s")

    remain_nodes = list(r_nodes.union(set(seeds)))
    end_time = time.time()
    cost_time = end_time - start_time

    rr = g.subgraph(remain_nodes)
    rrs_log(args, cost_time, rr=rr)
    return seeds, rr


def RRSparse(args, g):
    """
    Sparse-matrix variant of ReachableRangeSearch.
    Keeps the original function untouched and can be called explicitly.
    """
    start_time = time.time()
    try:
        from scipy import sparse as sp
    except Exception:
        return ReachableRangeSearch(args, g)

    node_list = list(g.nodes())
    n = len(node_list)
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    rows = []
    cols = []
    vals = []
    for u, v in g.edges():
        pp = msg_p(g.edges[u, v]['ppd'])
        if pp > 0:
            rows.append(node_to_idx[u])
            cols.append(node_to_idx[v])
            vals.append(pp)

    P = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    cur = P.copy()

    def lam_from_sparse(mat):
        nnz = mat.nnz
        total = n * n
        zero_num = total - nnz
        if nnz == 0:
            return 0.0
        if (1 - args.p) * total <= zero_num:
            return 0.0
        weights = np.sort(mat.data)[::-1]
        idx = int(args.p * total)
        if idx < 0:
            idx = 0
        if idx >= len(weights):
            idx = len(weights) - 1
        return float(weights[idx])

    def prune_sparse(mat, lam):
        if mat.nnz == 0:
            return mat
        coo = mat.tocoo()
        keep = coo.data > lam
        if not np.any(keep):
            return sp.csr_matrix(mat.shape, dtype=float)
        return sp.csr_matrix(
            (coo.data[keep], (coo.row[keep], coo.col[keep])),
            shape=mat.shape,
            dtype=float,
        )

    lam = lam_from_sparse(cur)
    cur = prune_sparse(cur, lam)

    i = 2
    max_iter = int(np.ceil(np.log2(n)))
    iter_start = time.time()
    while i <= max_iter + 1:
        # Sparse propagation via matrix multiplication (sum-product semiring).
        nxt = cur.dot(P)
        # Keep strongest known probabilities found so far.
        cur = cur.maximum(nxt).tocsr()
        lam = lam_from_sparse(cur)
        cur = prune_sparse(cur, lam)
        done = i - 1
        total = max_iter
        elapsed = max(time.time() - iter_start, 1e-9)
        rate = done / elapsed
        remain = max(0, total - done)
        eta = remain / max(rate, 1e-9)
        pct = 100.0 * done / max(total, 1)
        if done == total:
            print(f"RRSparse time_signal={getattr(args,'time_signal','NA')} iter: {done}/{total} ({pct:.1f}%), {rate:.2f} iter/s, ETA {eta:.1f}s")
        i += 1

    covered = np.zeros(n, dtype=bool)
    seeds = []
    seed_idx_set = set()
    seed_start = time.time()

    checkpoints = {5, 10, 15, 20}
    for seed_i in range(args.t):
        best_seed_idx = None
        best_new = -1

        for s_idx, s_node in enumerate(node_list):
            if s_idx in seed_idx_set:
                continue
            row = cur.getrow(s_idx)
            if row.nnz == 0:
                continue
            # Number of currently-uncovered nodes this seed can newly cover.
            new_cov = int(np.count_nonzero(~covered[row.indices]))
            if new_cov > best_new:
                best_new = new_cov
                best_seed_idx = s_idx

        if best_seed_idx is None:
            break

        seeds.append(node_list[best_seed_idx])
        seed_idx_set.add(best_seed_idx)
        best_row = cur.getrow(best_seed_idx)
        if best_row.nnz > 0:
            covered[best_row.indices] = True
        if (seed_i + 1) in checkpoints:
            covered_nodes = [node_list[i] for i in np.where(covered)[0]]
            remain_nodes = list(set(covered_nodes).union(set(seeds)))
            rr_snapshot = g.subgraph(remain_nodes)
            _dump_rr_and_seeds(args, seed_i + 1, rr_snapshot, seeds, elapsed_from_rr_start=(time.time() - start_time))
        elapsed = max(time.time() - seed_start, 1e-9)
        rate = (seed_i + 1) / elapsed
        remain = args.t - (seed_i + 1)
        eta = remain / max(rate, 1e-9)
        pct = 100.0 * (seed_i + 1) / max(args.t, 1)
        print(f"RRSparse time_signal={getattr(args,'time_signal','NA')} seeds: {seed_i + 1}/{args.t} ({pct:.1f}%), {rate:.2f} seeds/s, ETA {eta:.1f}s")

    covered_nodes = [node_list[i] for i in np.where(covered)[0]]
    remain_nodes = list(set(covered_nodes).union(set(seeds)))
    end_time = time.time()
    cost_time = end_time - start_time

    rr = g.subgraph(remain_nodes)
    rrs_log(args, cost_time, rr=rr)
    return seeds, rr


def EventAssignment(args, rr, seeds, assign_l=None, log_prefix='EventAssignment'):
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
        G_prime.edges[u, v]['se_idx'] = []

    for u in seeds:
        G_prime.nodes[u]['ap'] = 1.0

    assign_l = max(int(args.l if assign_l is None else assign_l), 0)

    worklist = deque()
    in_queue = set()
    for u in seeds:
        for v in rr.successors(u):
            if v not in seeds:
                if v not in in_queue:
                    worklist.append(v)
                    in_queue.add(v)

    assign_start = time.time()
    processed = 0
    max_reprocess = max(rr.number_of_nodes() * 50, 10000)
    eps = float(getattr(args, 'eps', 0))

    while worklist:
        v = worklist.popleft()
        in_queue.discard(v)
        processed += 1

        if processed > max_reprocess:
            print(
                f"{log_prefix} time_signal={getattr(args,'time_signal','NA')} "
                f"stopped at max_reprocess={max_reprocess}"
            )
            break

        if v in seeds:
            continue

        in_edges_all = [(u, v) for u in rr.predecessors(v)]
        for e in in_edges_all:
            G_prime.edges[e]['ppd'] = []
            G_prime.edges[e]['se_idx'] = []

        # Only activated predecessors contribute to v.
        E_vstd = [(u, v) for (u, v) in in_edges_all if G_prime.nodes[u].get('ap', 0.0) > 0.0]
        ap_old = G_prime.nodes[v].get('ap', 0.0)
        ap_new = 0.0

        if E_vstd:
            # Unique sub-event assignment by explicit sub-event IDs (se_idx).
            # For each sub-event, keep only the best incoming edge (max PPdist).
            best_by_sub = {}  # sub_id -> (best_dist_prob, raw_prob, best_edge)
            for (u, v_edge) in E_vstd:
                edge_data = rr.edges[u, v_edge]
                probs = edge_data.get('ppd', [])
                idxs = edge_data.get('se_idx', edge_data.get('sim_idx', []))
                scale = G_prime.nodes[u].get('ap', 0.0)
                for sub_id, p_raw in zip(idxs, probs):
                    p_dist = scale * p_raw
                    prev = best_by_sub.get(sub_id)
                    if prev is None or p_dist > prev[0]:
                        best_by_sub[sub_id] = (p_dist, p_raw, (u, v_edge))

            # Sort by p_dist (the diffusion probability), not raw p.
            ranked = sorted(best_by_sub.items(), key=lambda x: x[1][0], reverse=True)
            selected = ranked[:assign_l]
            top_l_probs = [p_dist for _, (p_dist, _, _) in selected]

            assigned = {e: ([], []) for e in E_vstd}  # edge -> (ppd_list, se_idx_list)
            for sub_id, (_, p_raw, owner_edge) in selected:
                assigned[owner_edge][0].append(p_raw)
                assigned[owner_edge][1].append(int(sub_id))

            for e in E_vstd:
                G_prime.edges[e]['ppd'] = assigned[e][0]
                G_prime.edges[e]['se_idx'] = assigned[e][1]

            ap_new = msg_p(top_l_probs)

        G_prime.nodes[v]['ap'] = ap_new

        if abs(ap_new - ap_old) > eps and ap_new > 0.0:
            for w in rr.successors(v):
                if w in seeds:
                    continue
                if w not in in_queue:
                    worklist.append(w)
                    in_queue.add(w)

        if processed % 2000 == 0 or len(worklist) == 0:
            elapsed = max(time.time() - assign_start, 1e-9)
            rate = processed / elapsed
            print(
                f"{log_prefix} time_signal={getattr(args,'time_signal','NA')} "
                f"processed={processed} queue={len(worklist)} rate={rate:.2f} nodes/s"
            )

    return G_prime


