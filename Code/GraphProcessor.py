import random
import time
import os
import heapq
import numpy as np
import math
from collections import deque
from scipy import sparse as sp
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
    return r_dict, loss


def update_r_dict(r_dict, lam, full_size, iteration):
    removed_r = 0
    ori_r = 0
    
    for s_node in r_dict.copy():
        for t_node in r_dict[s_node].copy():
            ori_r += r_dict[s_node][t_node]
            if r_dict[s_node][t_node] < lam: #Here, must be < to avoid a lam removes all the values
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
    return lam'''


def _rrs_folder_path(args):
    return args.data_path_prefix.format(args.data) + '/RRSp/l{}p{}q{}'.format(
        args.l,
        args.p,
        format(getattr(args, "hrq", 0.95), "g"),
    )


def _dump_rr_and_seeds(args, k, rr, seeds, elapsed_from_rr_start=None):
    folder_path = _rrs_folder_path(args)
    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, f"Seeds_{k}_{args.time_signal}.txt"), "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(f"{s}\n")
    with open(os.path.join(folder_path, f"RR_{k}_{args.time_signal}.txt"), "w", encoding="utf-8") as f:
        for u, v in rr.edges():
            edge_data = rr.edges[u, v]
            sim_idx = edge_data.get('se_idx', edge_data.get('sim_idx', []))
            ppd = edge_data.get('ppd', [])
            f.write(f"{u}\t{v}\t{sim_idx}\t{ppd}\n")
    if elapsed_from_rr_start is not None:
        with open(os.path.join(folder_path, f"Cost_{k}_{args.time_signal}.txt"), "a", encoding="utf-8") as f:
            f.write(f"k\t{k}\telapsed\t{elapsed_from_rr_start}\n")


'''def InfoLoss(g, p):
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
    print(f"InfoLoss mean={np.mean(loss_list)}")'''


def InfoLossparse(g, p):
    """
    Accelerated variant of information_loss with aligned semantics.
    Uses indexed adjacency/precomputation, while preserving the original
    max-times propagation and in-loop update behavior.
    """
    node_list = list(g.nodes())
    n = len(node_list)
    if n == 0:
        print("InfoLossSparse mean=0.0")
        return

    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Precompute outgoing adjacency probabilities by indexed node id.
    succ_idx = [[] for _ in range(n)]
    succ_pp = [[] for _ in range(n)]
    # Reachability rows: r_rows[s][t] = probability from s to t.
    r_rows = [dict() for _ in range(n)]

    for u, v in g.edges():
        pp = msg_p(g.edges[u, v]['ppd'])
        if pp <= 0:
            continue
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        succ_idx[u_idx].append(v_idx)
        succ_pp[u_idx].append(pp)
        r_rows[u_idx][v_idx] = pp

    succ_idx_np = [np.asarray(x, dtype=np.int64) for x in succ_idx]
    succ_pp_np = [np.asarray(x, dtype=float) for x in succ_pp]
    loss_list = []

    def lam_from_rows(rows_data):
        weights = []
        for row in rows_data:
            if row:
                weights.extend(row.values())

        if len(weights) == 0:
            return 0.0
        total = n * n
        zero_num = total - len(weights)
        if (1 - p) * total <= zero_num:
            return 0.0
        weights = np.sort(np.asarray(weights, dtype=float))[::-1]
        idx = int(p * total)
        return float(weights[idx])

    def prune_rows_with_loss(rows_data, lam):
        ori_r = 0.0
        removed_r = 0.0

        for row in rows_data:
            if not row:
                continue
            for t_idx in list(row.keys()):
                val = row[t_idx]
                ori_r += val
                if val < lam: #Here, must be < to avoid a lam removes all the values
                    removed_r += val
                    del row[t_idx]

        if ori_r <= 0:
            return 0.0
        return removed_r / ori_r

    i = 1
    lam = lam_from_rows(r_rows)
    loss = prune_rows_with_loss(r_rows, lam)
    loss_list.append(loss)
    print(f"InfoLossSparse iter={i} loss={loss}")

    i += 1
    max_iter = int(np.ceil(np.log2(n)))
    while i <= max_iter + 1:
        # Keep information_loss semantics:
        # iterate over a snapshot of current t-keys, updating each row in place.
        for s_idx in range(n):
            row = r_rows[s_idx]
            if not row:
                continue
            t_snapshot = list(row.keys())
            for t_idx in t_snapshot:
                base = row.get(t_idx, 0.0)
                if base <= 0:
                    continue
                q_idx_arr = succ_idx_np[t_idx]
                if q_idx_arr.size == 0:
                    continue
                q_pp_arr = succ_pp_np[t_idx]
                for k, q_idx in enumerate(q_idx_arr):
                    pp = base * q_pp_arr[k]
                    prev = row.get(int(q_idx))
                    if prev is None or pp > prev:
                        row[int(q_idx)] = pp

        lam = lam_from_rows(r_rows)
        loss = prune_rows_with_loss(r_rows, lam)
        loss_list.append(loss)
        print(f"InfoLossSparse iter={i} loss={loss}")
        i += 1

    remaining_reachability = sum(len(row) for row in r_rows)
    print(f"InfoLossSparse remaining_reachability={remaining_reachability}")
    print(f"InfoLossSparse mean={np.mean(loss_list)}")


'''def ReachableRangeSearch(args, g):
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
    rr = g.subgraph(remain_nodes)
    return seeds, rr'''


def RRSparse(args, g):
    """
    Accelerated ReachableRangeSearch with semantics aligned to the
    original max-times propagation/pruning logic.
    """
    start_time = time.time()

    node_list = list(g.nodes())
    n = len(node_list)
    if n == 0:
        return [], g.subgraph([])

    node_to_idx = {node: i for i, node in enumerate(node_list)}

    # Precompute outgoing adjacency probabilities by indexed node id.
    succ_idx = [[] for _ in range(n)]
    succ_pp = [[] for _ in range(n)]
    # Reachability rows: r_rows[s][t] = probability from s to t.
    r_rows = [dict() for _ in range(n)]

    for u, v in g.edges():
        pp = msg_p(g.edges[u, v]['ppd'])
        if pp <= 0:
            continue
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        succ_idx[u_idx].append(v_idx)
        succ_pp[u_idx].append(pp)
        r_rows[u_idx][v_idx] = pp

    succ_idx_np = [np.asarray(x, dtype=np.int64) for x in succ_idx]
    succ_pp_np = [np.asarray(x, dtype=float) for x in succ_pp]

    def lam_from_rows(rows_data):
        weights = []
        for row in rows_data:
            if row:
                weights.extend(row.values())
        total = n * n
        zero_num = total - len(weights)
        if len(weights) == 0:
            return 0.0
        if (1 - args.p) * total <= zero_num:
            return 0.0
        weights = np.sort(np.asarray(weights, dtype=float))[::-1]
        idx = int(args.p * total)
        return float(weights[idx])

    def prune_rows(rows_data, lam):
        for row in rows_data:
            if not row:
                continue
            for t_idx in list(row.keys()):
                if row[t_idx] < lam:
                    del row[t_idx]

    lam = lam_from_rows(r_rows)
    prune_rows(r_rows, lam)

    i = 2
    max_iter = int(np.ceil(np.log2(n)))
    iter_start = time.time()
    while i <= max_iter + 1:
        # Keep original ReachableRangeSearch semantics:
        # iterate over snapshot of current t-keys, updating row in place.
        for s_idx in range(n):
            row = r_rows[s_idx]
            if not row:
                continue
            t_snapshot = list(row.keys())
            for t_idx in t_snapshot:
                base = row.get(t_idx, 0.0)
                if base <= 0:
                    continue
                q_idx_arr = succ_idx_np[t_idx]
                if q_idx_arr.size == 0:
                    continue
                q_pp_arr = succ_pp_np[t_idx]
                for k, q_idx in enumerate(q_idx_arr):
                    pp = base * q_pp_arr[k]
                    prev = row.get(int(q_idx))
                    if prev is None or pp > prev:
                        row[int(q_idx)] = pp

        lam = lam_from_rows(r_rows)
        prune_rows(r_rows, lam)
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

    remaining_reachability = sum(len(row) for row in r_rows)
    print(f"InfoLossSparse remaining_reachability={remaining_reachability}")

    covered_set = set()
    seeds = []
    seed_idx_set = set()
    seed_start = time.time()

    checkpoints = {5, 10, 15, 20}
    for seed_i in range(args.t):
        best_seed_idx = None
        best_r = None
        best_size = -1

        for s_idx in range(n):
            if s_idx in seed_idx_set:
                continue
            row = r_rows[s_idx]
            if not row:
                continue
            s_reach = set(row.keys())
            tem_r = covered_set.union(s_reach)
            if len(tem_r) > best_size:
                best_size = len(tem_r)
                best_r = s_reach
                best_seed_idx = s_idx

        if best_seed_idx is None:
            break

        seeds.append(node_list[best_seed_idx])
        seed_idx_set.add(best_seed_idx)
        if best_r:
            covered_set = covered_set.union(best_r)
        if (seed_i + 1) in checkpoints:
            covered_nodes = [node_list[idx] for idx in covered_set]
            remain_nodes = list(set(covered_nodes).union(set(seeds)))
            rr_snapshot = g.subgraph(remain_nodes)
            _dump_rr_and_seeds(args, seed_i + 1, rr_snapshot, seeds, elapsed_from_rr_start=(time.time() - start_time))
        elapsed = max(time.time() - seed_start, 1e-9)
        rate = (seed_i + 1) / elapsed
        remain = args.t - (seed_i + 1)
        eta = remain / max(rate, 1e-9)
        pct = 100.0 * (seed_i + 1) / max(args.t, 1)
        print(f"RRSparse time_signal={getattr(args,'time_signal','NA')} seeds: {seed_i + 1}/{args.t} ({pct:.1f}%), {rate:.2f} seeds/s, ETA {eta:.1f}s")

    covered_nodes = [node_list[idx] for idx in covered_set]
    remain_nodes = list(set(covered_nodes).union(set(seeds)))
    rr = g.subgraph(remain_nodes)
    return seeds, rr


def TIM(args, rr, num_rr_sets=10000, rng_seed=12345, log_prefix='TIM'):
    """
    TIM-style seed selection on rr.

    Steps:
    1) Build diff_g where each edge stores pp = average(ppd).
    2) Sample num_rr_sets reverse reachable sets from diff_g.
    3) Greedily select t seeds maximizing RR-set coverage.

    Args:
      rr: directed graph with edge attribute 'ppd' (list-like probabilities)
      t: number of new seeds to select
      num_rr_sets: RR set sample size (default 10000)
      rng_seed: RNG seed for reproducibility

    Returns:
      selected_seeds, diff_g
    """
    rng = random.Random(int(rng_seed))
    t = max(int(args.t), 0)
    num_rr_sets = max(int(num_rr_sets), 0)

    # Build diffusion graph with per-edge probability pp = average(ppd).
    diff_g = type(rr)()
    diff_g.add_nodes_from(rr.nodes(data=True))
    for u, v in rr.edges():
        ed = rr.edges[u, v]
        ppd = ed.get('ppd', [])
        if ppd is None:
            ppd = []
        if len(ppd) > 0:
            pp = float(np.mean(np.asarray(ppd, dtype=float)))
        else:
            pp = 0.0
        pp = float(np.clip(pp, 0.0, 1.0))
        diff_g.add_edge(u, v)
        diff_g.edges[u, v]['pp'] = pp

    nodes = list(diff_g.nodes())
    n = len(nodes)
    if n == 0 or t == 0 or num_rr_sets == 0:
        return [], diff_g

    # Indexed reverse adjacency for faster RR sampling.
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    rev_adj = [[] for _ in range(n)]  # rev_adj[v] = list[(u, p_uv)]
    for u, v in diff_g.edges():
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        p = float(diff_g.edges[u, v].get('pp', 0.0))
        if p > 0.0:
            rev_adj[vi].append((ui, p))

    # Sample RR sets by reverse BFS from random roots.
    # rr_sets[rid] stores node indices in that RR set.
    rr_sets = []
    node_to_rr_ids = [[] for _ in range(n)]
    rr_sample_start = time.time()
    rr_log_every = max(1, num_rr_sets // 10)
    for rid in range(num_rr_sets):
        root_idx = rng.randrange(n)
        reached = {root_idx}
        queue = deque([root_idx])
        rr_nodes = [root_idx]
        while queue:
            cur = queue.popleft()
            for pred_idx, p in rev_adj[cur]:
                if pred_idx in reached:
                    continue
                if rng.random() <= p:
                    reached.add(pred_idx)
                    queue.append(pred_idx)
                    rr_nodes.append(pred_idx)
        rr_sets.append(rr_nodes)
        for nid in rr_nodes:
            node_to_rr_ids[nid].append(rid)

        sampled = rid + 1
        if sampled % rr_log_every == 0 or sampled == num_rr_sets:
            elapsed = max(time.time() - rr_sample_start, 1e-9)
            rate = sampled / elapsed
            remain = num_rr_sets - sampled
            eta = remain / max(rate, 1e-9)
            pct = 100.0 * sampled / max(num_rr_sets, 1)
            print(
                f"{log_prefix} RR sampling: {sampled}/{num_rr_sets} "
                f"({pct:.1f}%), {rate:.1f} sets/s, ETA {eta:.1f}s"
            )

    selected_idx = set()
    selected = []

    covered = [False] * num_rr_sets
    covered_count = 0

    # gain_est[u] = current uncovered RR sets that contain u.
    gain_est = [len(node_to_rr_ids[u]) for u in range(n)]

    # Lazy-heap greedy.
    heap = []
    for u in range(n):
        heapq.heappush(heap, (-gain_est[u], u))

    for i in range(t):
        chosen = None
        chosen_gain = 0

        while heap:
            neg_gain, u = heapq.heappop(heap)
            if u in selected_idx:
                continue
            cur_gain = gain_est[u]
            if -neg_gain != cur_gain:
                heapq.heappush(heap, (-cur_gain, u))
                continue
            chosen = u
            chosen_gain = cur_gain
            break

        if chosen is None or chosen_gain <= 0:
            break

        selected_idx.add(chosen)
        selected.append(nodes[chosen])

        # Cover RR sets hit by chosen node and update gain estimates.
        for rid in node_to_rr_ids[chosen]:
            if covered[rid]:
                continue
            covered[rid] = True
            covered_count += 1
            for nid in rr_sets[rid]:
                gain_est[nid] -= 1

        if (i + 1) % 5 == 0 or (i + 1) == t:
            done = i + 1
            print(
                f"{log_prefix} selected={done}/{t} "
                f"covered_rr={covered_count}/{num_rr_sets}"
            )

    print(
        f"{log_prefix} done: selected={len(selected)} "
        f"num_rr_sets={num_rr_sets} uncovered_rr={num_rr_sets - covered_count}"
    )
    return selected, diff_g


def EventAssignment(args, rr, seeds, l=None, log_prefix='EventAssignment'):
    """
    Event Assignment for Event-Optimized Graph (EASoG).
    Returns event-optimized graph G' with same V, E as G and node attribute 'ap' (activation probability).
    Inf_dist is taken as edge 'ppd'; sub-event assignment selects top-l sub-events by pp_dist per edge.
    """

    # Build G* with same topology as rr.
    G_prime = type(rr)()
    G_prime.add_nodes_from(rr.nodes())
    G_prime.add_edges_from(rr.edges())

    for n in G_prime.nodes():
        G_prime.nodes[n]['ap'] = 0.0
        if 'x' in rr.nodes[n]:
            G_prime.nodes[n]['x'] = rr.nodes[n]['x']
        if 'y' in rr.nodes[n]:
            G_prime.nodes[n]['y'] = rr.nodes[n]['y']

    for u, v in G_prime.edges():
        G_prime.edges[u, v]['ppd'] = []
        G_prime.edges[u, v]['se_idx'] = []

    seed_set = {s for s in seeds if s in rr}
    for s in seed_set:
        G_prime.nodes[s]['ap'] = 1.0

    l = max(int(args.l if l is None else l), 0)
    eps = float(getattr(args, 'eps', 0.0))

    # ap_pre in the pseudo-code.
    ap_pre = {n: 0.0 for n in G_prime.nodes()}

    # Vstd and ?Vstd initialization.
    vstd = set(seed_set)
    delta_vstd = set()
    for u in seed_set:
        for v in rr.successors(u):
            if v not in seed_set:
                delta_vstd.add(v)

    assign_start = time.time()
    processed = 0
    max_reprocess = max(rr.number_of_nodes() * 50, 10000)

    while len(delta_vstd) > 0:
        processed += len(delta_vstd)
        if processed > max_reprocess:
            print(
                f"{log_prefix} time_signal={getattr(args,'time_signal','NA')} "
                f"stopped at max_reprocess={max_reprocess}"
            )
            break

        # Round updates before commit:
        # round_updates[v] = (ap_new, assigned_dict)
        round_updates = {}

        for v in delta_vstd:
            if v in seed_set:
                continue

            in_edges_all = [(u, v) for u in rr.predecessors(v)]
            E_vstd = [(u, v) for (u, v) in in_edges_all if u in vstd]

            ap_new = 0.0
            assigned = {e: ([], []) for e in E_vstd}

            if E_vstd:
                # For each sub-event, keep only the best incoming predecessor edge.
                best_by_sub = {}
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

                ranked = sorted(best_by_sub.items(), key=lambda x: x[1][0], reverse=True)

                # Adaptive assignment budget per node.
                no_msg_prob = 1.0
                for (u, v_edge) in E_vstd:
                    pred_success = G_prime.nodes[u].get('ap', 0.0)
                    pred_success = min(max(pred_success, 0.0), 1.0)
                    no_msg_prob *= (1.0 - pred_success)
                msg_prob = 1.0 - no_msg_prob

                if msg_prob > 0.0 and l > 0:
                    adaptive_l = min(len(ranked), max(1, math.ceil(l / max(msg_prob, 1e-12))))
                    selected = ranked[:adaptive_l]
                else:
                    selected = []
                selected = ranked[:l]

                for sub_id, (_, p_raw, owner_edge) in selected:
                    assigned[owner_edge][0].append(p_raw)
                    assigned[owner_edge][1].append(int(sub_id))

                # Ensemble estimation: ap(v) = 1 - ?_u (1 - ap(u) * p(u->v)).
                pred_fail_prob = 1.0
                for (u, v_edge) in E_vstd:
                    edge_ppd = assigned[(u, v_edge)][0]
                    edge_success = msg_p(edge_ppd) if len(edge_ppd) > 0 else 0.0
                    pred_success = G_prime.nodes[u].get('ap', 0.0) * edge_success
                    pred_success = min(max(pred_success, 0.0), 1.0)
                    pred_fail_prob *= (1.0 - pred_success)
                ap_new = 1.0 - pred_fail_prob

            round_updates[v] = (ap_new, assigned)

        # Build ?V for next round from nodes with AP improvement > eps.
        delta_v = set()
        for v in delta_vstd:
            if v in seed_set:
                continue
            # ap_now = G_prime.nodes[v].get('ap', 0.0)
            ap_now = round_updates[v][0]
            if ap_now - ap_pre.get(v) > eps:
                for w in rr.successors(v):
                    if w not in seed_set:
                        delta_v.add(w)

                for (u, v_edge), (ppd_list, idx_list) in round_updates[v][1].items():
                    G_prime.edges[u, v_edge]['ppd'] = ppd_list
                    G_prime.edges[u, v_edge]['se_idx'] = idx_list

                G_prime.nodes[v]['ap'] = ap_now
                ap_pre[v] = ap_now

        vstd = vstd.union(delta_vstd)
        delta_vstd = delta_v

        if processed % 2000 == 0 or len(delta_vstd) == 0:
            elapsed = max(time.time() - assign_start, 1e-9)
            rate = processed / elapsed
            print(
                f"{log_prefix} time_signal={getattr(args,'time_signal','NA')} "
                f"processed={processed} frontier={len(delta_vstd)} rate={rate:.2f} nodes/s"
            )

    return G_prime


def EdgeSelection(rr, seeds, log_prefix='B-PEI'):
    """
    Construct an l-subnetwork from rr and seeds using the greedy
    optimal-neighbor / optimal-candidate-node process.

    Input:
      rr: directed graph with edge attribute 'ppd' (and optional 'se_idx'/'sim_idx')
      seeds: iterable of seed nodes
      l: post/event limit used in edge benefit computation

    Output:
      G_l: directed subnetwork
    """

    sg = type(rr)()

    seed_set = {s for s in seeds if s in rr}
    selected_edges = set()

    # ap for currently selected/activated nodes.
    ap = {s: 1.0 for s in seed_set}

    # candidate_info[v] = (best_u, best_score), and every v here has an assigned edge (best_u, v).
    candidate_info = {}

    processed = 0
    while True:
        # Selected pool for parent candidates: seeds + already-assigned candidate nodes.
        selected_pool = set(seed_set).union(set(candidate_info.keys()))

        # assignable: nodes that have at least one incoming neighbor in selected_pool,
        # but are not already assigned in candidate_info and not seeds.
        assignable = {}
        for u in selected_pool:
            for v in rr.successors(u):
                if v in seed_set or v in candidate_info:
                    continue
                assignable[v] = None

        # Stop when there is no assignable node.
        if len(assignable) == 0:
            break

        # Determine best parent for all assignable nodes simultaneously.
        for v in list(assignable.keys()):
            best_u = None
            best_score = -1.0
            for u in rr.predecessors(v):
                if u not in selected_pool:
                    continue
                score = ap.get(u, 0.0) * msg_p(rr.edges[u, v]['ppd'])
                if score > best_score:
                    best_score = score
                    best_u = u
            if best_u is not None:
                assignable[v] = (best_u, best_score)
            else:
                del assignable[v]

        if len(assignable) == 0:
            break

        # No single optimal-candidate step: lock selected edges for all nodes
        # in candidate_info simultaneously in this round.
        added_this_round = 0
        for v, info in assignable.items():
            u_star, best_score = info
            if best_score <= 0.0:
                continue
            candidate_info[v] = (u_star, best_score)
            selected_edges.add((u_star, v))
            ap[v] = best_score
            added_this_round += 1

        if added_this_round == 0:
            break
        processed += added_this_round

        if processed % 2000 == 0:
            print(
                f"{log_prefix} processed={processed} "
                f"selected_nodes={len(seed_set) + len(candidate_info)} assignable={len(assignable)}"
            )

    # Materialize subnetwork.
    selected_nodes = set(seed_set).union(set(candidate_info.keys()))
    for n in selected_nodes:
        sg.add_node(n)

    for u, v in selected_edges:
        if u in sg and v in sg and rr.has_edge(u, v):
            sg.add_edge(u, v)
            ed = rr.edges[u, v]
            sg.edges[u, v]['ppd'] = list(ed.get('ppd', []))
            sg.edges[u, v]['se_idx'] = list(ed.get('se_idx', ed.get('sim_idx', [])))

    print(
        f"{log_prefix} done: seeds={len(seed_set)} "
        f"selected_nodes={sg.number_of_nodes()} selected_edges={sg.number_of_edges()}"
    )
    return sg


