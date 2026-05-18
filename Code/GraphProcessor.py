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
    print(f"RRSparse remaining_reachability={remaining_reachability}")

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
