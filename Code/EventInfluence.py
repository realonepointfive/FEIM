import numpy as np
import random
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return 0.0
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
    return max(0.0, sim)


def search_cand(diff_g, l, seed_set=None):
    """
    Return candidate edges from currently active nodes to neighbors that have
    received fewer than l sub-events so far, and whose connecting edge has not
    yet been used to send sub-events.
    """
    cand_edges = []
    seed_set = seed_set or set()
    for node in diff_g.nodes():
        if diff_g.nodes[node].get('active'):
            for neighbor in diff_g.successors(node):
                if (
                    neighbor not in seed_set
                    and
                    diff_g.nodes[neighbor].get('msg_num', 0) < l
                    and not diff_g.edges[node, neighbor].get('used', False)
                ):
                    cand_edges.append((node, neighbor))
    return cand_edges


def _build_static_graph_cache(diff_g, seeds, node_list=None):
    nodes = node_list if node_list is not None else list(diff_g.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    seed_idx = np.asarray([node_to_idx[s] for s in seeds if s in node_to_idx], dtype=np.int32)
    is_seed = np.zeros(n_nodes, dtype=bool)
    if seed_idx.size > 0:
        is_seed[seed_idx] = True

    has_coord = np.zeros(n_nodes, dtype=bool)
    node_x = np.zeros(n_nodes, dtype=float)
    node_y = np.zeros(n_nodes, dtype=float)
    for i, node in enumerate(nodes):
        nd = diff_g.nodes[node]
        if 'x' in nd and 'y' in nd:
            has_coord[i] = True
            node_x[i] = nd['x']
            node_y[i] = nd['y']

    edge_dst = []
    edge_probs = []
    edge_idxs = []
    edge_pairs = []
    edge_id_by_pair = {}
    out_eids = [[] for _ in range(n_nodes)]
    for (u, v) in diff_g.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        ed = diff_g.edges[u, v]
        probs = ed.get('ppd', [])
        idxs = ed.get('se_idx', ed.get('sim_idx'))
        if idxs is None:
            idxs = list(range(len(probs)))
        eid = len(edge_dst)
        edge_dst.append(vi)
        edge_probs.append(probs)
        edge_idxs.append(idxs)
        edge_pairs.append((u, v))
        edge_id_by_pair[(u, v)] = eid
        out_eids[ui].append(eid)

    return {
        'nodes': nodes,
        'node_to_idx': node_to_idx,
        'n_nodes': n_nodes,
        'n_edges': len(edge_dst),
        'seed_idx': seed_idx,
        'is_seed': is_seed,
        'has_coord': has_coord,
        'node_x': node_x,
        'node_y': node_y,
        'edge_dst': edge_dst,
        'edge_probs': edge_probs,
        'edge_idxs': edge_idxs,
        'edge_pairs': edge_pairs,
        'edge_id_by_pair': edge_id_by_pair,
        'out_eids': out_eids,
    }


def _run_chunk(sim_count, l, x0, y0, base_seed, static, rr_static):
    n_nodes = static['n_nodes']
    n_edges = static['n_edges']
    seed_idx = static['seed_idx']
    is_seed = static['is_seed']
    has_coord = static['has_coord']
    node_x = static['node_x']
    node_y = static['node_y']
    edge_dst = static['edge_dst']
    edge_probs = static['edge_probs']
    edge_idxs = static['edge_idxs']
    out_eids = static['out_eids']
    edge_pairs = static['edge_pairs']

    rr_edge_dst = rr_static['edge_dst']
    rr_edge_probs = rr_static['edge_probs']
    rr_edge_idxs = rr_static['edge_idxs']
    rr_out_eids = rr_static['out_eids']
    rr_edge_id_by_pair = rr_static['edge_id_by_pair']
    rr_n_edges = rr_static['n_edges']

    rng = np.random.default_rng(base_seed)

    bene = 0
    msg = 0
    inf = 0
    fair_sum = 0.0
    fair_cnt = 0
    dist_sum = 0.0

    for _ in range(sim_count):
        max_dist = 0.0

        active = np.zeros(n_nodes, dtype=bool)
        reached = np.zeros(n_nodes, dtype=bool)
        msg_num = np.zeros(n_nodes, dtype=np.int32)
        node_bene = np.zeros(n_nodes, dtype=np.int32)
        edge_used = np.zeros(n_edges, dtype=bool)
        rr_edge_used = np.zeros(rr_n_edges, dtype=bool)

        if seed_idx.size > 0:
            active[seed_idx] = True
        active_nodes = list(seed_idx)

        phase = 'diff'
        while True:
            edges_by_target = {}
            if phase == 'diff':
                for ui in active_nodes:
                    for eid in out_eids[ui]:
                        if edge_used[eid]:
                            continue
                        vi = edge_dst[eid]
                        if is_seed[vi] or msg_num[vi] >= l:
                            continue
                        edges_by_target.setdefault(vi, []).append((eid, False))

                if len(edges_by_target) == 0:
                    # Check if rr phase should start: any target with active incoming diff_g edge
                    # but not enough messages, and a usable rr edge exists.
                    rr_needed = False
                    for ui in active_nodes:
                        for deid in out_eids[ui]:
                            vi = edge_dst[deid]
                            if is_seed[vi] or msg_num[vi] >= l:
                                continue
                            pair = edge_pairs[deid]
                            rr_eid = rr_edge_id_by_pair.get(pair)
                            if rr_eid is None or rr_edge_used[rr_eid]:
                                continue
                            rr_needed = True
                            break
                        if rr_needed:
                            break
                    if rr_needed:
                        phase = 'rr'
                        continue
                    break
            else:
                # RR phase: use rr edges corresponding to active incoming diff_g edges.
                for ui in active_nodes:
                    for deid in out_eids[ui]:
                        vi = edge_dst[deid]
                        if is_seed[vi] or msg_num[vi] >= l:
                            continue
                        pair = edge_pairs[deid]
                        rr_eid = rr_edge_id_by_pair.get(pair)
                        if rr_eid is None or rr_edge_used[rr_eid]:
                            continue
                        edges_by_target.setdefault(vi, []).append((rr_eid, True))
                if len(edges_by_target) == 0:
                    break

            for vi, in_eids in edges_by_target.items():
                reached[vi] = True
                subevent_prob = {}
                for eid, is_rr in in_eids:
                    if is_rr:
                        probs = rr_edge_probs[eid]
                        idxs = rr_edge_idxs[eid]
                    else:
                        probs = edge_probs[eid]
                        idxs = edge_idxs[eid]
                    for sub_idx, prob in zip(idxs, probs):
                        prev = subevent_prob.get(sub_idx)
                        if prev is None or prob > prev:
                            subevent_prob[sub_idx] = prob

                remain_cap = int(l - msg_num[vi])
                if remain_cap > 0 and len(subevent_prob) > 0:
                    ranked = sorted(subevent_prob.items(), key=lambda x: x[1], reverse=True)
                    selected_probs = np.asarray([x[1] for x in ranked[:remain_cap]], dtype=float)
                else:
                    selected_probs = np.asarray([], dtype=float)

                for eid, is_rr in in_eids:
                    if is_rr:
                        rr_edge_used[eid] = True
                    else:
                        edge_used[eid] = True

                chosen_num = int(selected_probs.size)
                msg_num[vi] += chosen_num
                msg += chosen_num

                if chosen_num > 0:
                    num_act = int(np.sum(rng.uniform(0, 1, chosen_num) < selected_probs))
                else:
                    num_act = 0
                bene += num_act
                node_bene[vi] += num_act

                if num_act > 0 and not active[vi]:
                    active[vi] = True
                    active_nodes.append(vi)
                    inf += 1
                    if has_coord[vi]:
                        dist = haversine(x0, y0, node_x[vi], node_y[vi])
                        if dist > max_dist:
                            max_dist = dist

        bene_list = node_bene[reached]
        n = int(bene_list.size)
        total_bene = int(np.sum(bene_list))
        if n > 0 and total_bene > 0:
            x_sorted = np.sort(bene_list.astype(float))
            coeff = (2 * np.arange(1, n + 1) - n - 1)
            disparity_sum = float(np.sum(coeff * x_sorted))
            fair_score = disparity_sum / (n * total_bene)
            if fair_score > 0:
                fair_sum += fair_score
                fair_cnt += 1

        dist_sum += max_dist

    return bene, msg, inf, fair_sum, fair_cnt, dist_sum


def EventInfluenceSimulation(args, diff_g, seeds, rr=None):
    if args.data == 'NepalEQuake':
        x = 28.3973623
        y = 84.1257684
    elif args.data == 'TexasFlood':
        x = 31.169621
        y = -99.683617
    else:
        x = -14.2400732
        y = -53.1805017

    static = _build_static_graph_cache(diff_g, seeds)
    rr_graph = rr if rr is not None else diff_g
    rr_static = _build_static_graph_cache(rr_graph, seeds, node_list=static['nodes'])
    workers = max(1, int(getattr(args, 'workers', 1)))
    sim_num = int(args.sim_num)
    report_every = max(1, sim_num // 5)
    process_start = time.time()

    bene = 0
    msg = 0
    inf = 0
    fair_sum = 0.0
    fair_cnt = 0
    dist_sum = 0.0

    if workers == 1 or sim_num <= 1:
        done = 0
        while done < sim_num:
            step = min(report_every, sim_num - done)
            b, m, i, fs, fc, ds = _run_chunk(
                step,
                args.l,
                x,
                y,
                12345 + done + int(getattr(args, 'time_signal', 0)) * 1000003,
                static,
                rr_static,
            )
            bene += b
            msg += m
            inf += i
            fair_sum += fs
            fair_cnt += fc
            dist_sum += ds
            done += step

            elapsed = max(time.time() - process_start, 1e-9)
            rate = done / elapsed
            remain = sim_num - done
            eta = remain / max(rate, 1e-9)
            pct = 100.0 * done / max(sim_num, 1)
            print(
                f"EventInfluenceSimulation time_signal={getattr(args, 'time_signal', 'NA')} "
                f"progress: {done}/{sim_num} ({pct:.1f}%), {rate:.2f} sims/s, ETA {eta:.1f}s"
            )
    else:
        workers = min(workers, sim_num, (os.cpu_count() or workers))
        chunks = max(workers * 4, 1)
        base_chunk = sim_num // chunks
        remainder = sim_num % chunks
        sim_chunks = []
        for ci in range(chunks):
            c = base_chunk + (1 if ci < remainder else 0)
            if c > 0:
                sim_chunks.append(c)

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = []
            future_to_chunk = {}
            for ci, c in enumerate(sim_chunks):
                seed = 12345 + ci + int(getattr(args, 'time_signal', 0)) * 1000003
                fut = ex.submit(_run_chunk, c, args.l, x, y, seed, static, rr_static)
                futures.append(fut)
                future_to_chunk[fut] = c

            done_futures = 0
            finished_sims = 0
            next_report = report_every
            for fut in as_completed(futures):
                b, m, i, fs, fc, ds = fut.result()
                bene += b
                msg += m
                inf += i
                fair_sum += fs
                fair_cnt += fc
                dist_sum += ds
                done_futures += 1
                finished_sims += future_to_chunk[fut]

                if finished_sims >= next_report or done_futures == len(sim_chunks):
                    elapsed = max(time.time() - process_start, 1e-9)
                    rate = finished_sims / elapsed
                    remain = sim_num - finished_sims
                    eta = remain / max(rate, 1e-9)
                    pct = 100.0 * finished_sims / max(sim_num, 1)
                    print(
                        f"EventInfluenceSimulation time_signal={getattr(args, 'time_signal', 'NA')} "
                        f"progress: {finished_sims}/{sim_num} ({pct:.1f}%), {rate:.2f} sims/s, ETA {eta:.1f}s"
                    )
                    while next_report <= finished_sims:
                        next_report += report_every

    msg_gap = (fair_sum / fair_cnt) if fair_cnt > 0 else 0.0
    node_num = diff_g.number_of_nodes() - len(seeds)
    ave_dist = (dist_sum / sim_num) if sim_num > 0 else 0.0
    return bene, msg, inf, msg_gap, node_num, ave_dist


def SocialUpdate(args, G, event_vectors_for_update):
    start_time = time.time()
    edges = list(G.edges())
    num_new = len(event_vectors_for_update)
    top_l = max(int(args.l), 0)
    se_offset = int(G.graph.get('SE_num_total', 0))
    num_total = se_offset + num_new
    hinge_floor = str(getattr(args, 'hinge_floor', 'sigma')).lower()
    if hinge_floor not in ('sigma', 'random'):
        hinge_floor = 'sigma'
    stats_bins = max(128, min(16384, int(getattr(args, 'hist_bins', 2048))))

    def calibrate_cosine_hinge(raw_cos_values, mu_floor, sigma_floor, k_sigma):
        # Sigma-shift hinge: T = mu + k*sigma, then max(0, (S - T) / (1 - T))
        raw = np.clip(np.asarray(raw_cos_values, dtype=float), 0.0, 1.0)
        mu = float(np.clip(mu_floor, 0.0, 1.0))
        sigma = float(max(sigma_floor, 0.0))
        k = float(k_sigma)
        t = float(np.clip(mu + k * sigma, 0.0, 1.0))
        denom = max(1.0 - t, 1e-12)
        sim_vals = np.maximum(0.0, (raw - t) / denom)
        return np.clip(sim_vals, 0.0, 1.0)

    def calibrate_cosine_hinge_t(raw_cos_values, t_floor):
        # Hinge with explicit floor T.
        raw = np.clip(np.asarray(raw_cos_values, dtype=float), 0.0, 1.0)
        t = float(np.clip(t_floor, 0.0, 1.0))
        denom = max(1.0 - t, 1e-12)
        sim_vals = np.maximum(0.0, (raw - t) / denom)
        return np.clip(sim_vals, 0.0, 1.0)

    def init_ppd_stats():
        return {'count': 0, 'sum': 0.0, 'hist': np.zeros(stats_bins, dtype=np.int64)}

    def update_ppd_stats(stats, values):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return
        arr = np.clip(arr, 0.0, 1.0)
        stats['count'] += int(arr.size)
        stats['sum'] += float(np.sum(arr))
        idx = np.minimum((arr * stats_bins).astype(np.int64), stats_bins - 1)
        stats['hist'] += np.bincount(idx, minlength=stats_bins)

    def ppd_stats_percentile(stats, q):
        if stats['count'] <= 0:
            return 0.0
        target = max(1, int(np.ceil(float(q) * stats['count'])))
        cdf = np.cumsum(stats['hist'], dtype=np.int64)
        b = int(np.searchsorted(cdf, target, side='left'))
        b = max(0, min(stats_bins - 1, b))
        return float((b + 0.5) / stats_bins)

    def ppd_stats_summary(stats):
        if stats['count'] <= 0:
            return "count=0 mean=0.0000 median=0.0000 p95=0.0000"
        mean = stats['sum'] / stats['count']
        median = ppd_stats_percentile(stats, 0.5)
        p95 = ppd_stats_percentile(stats, 0.95)
        return f"count={stats['count']} mean={mean:.4f} median={median:.4f} p95={p95:.4f}"

    def ensure_hinge_floor_state():
        prev_floor = G.graph.get('hinge_floor_mode_first')
        if prev_floor not in (None, hinge_floor):
            G.graph.pop('calib_mu_first', None)
            G.graph.pop('calib_sigma_first', None)
            G.graph.pop('calib_hinge_t_first', None)
        G.graph['hinge_floor_mode_first'] = hinge_floor

    def compute_random_pair_floor(edge_unit, edge_pairs, rng, num_pairs, quantile):
        n_edges = int(edge_unit.shape[0])
        if n_edges < 2:
            return 0.0
        src = np.asarray([e[0] for e in edge_pairs])
        dst = np.asarray([e[1] for e in edge_pairs])
        neighbors = {}
        for s, d in edge_pairs:
            if s not in neighbors:
                neighbors[s] = set()
            if d not in neighbors:
                neighbors[d] = set()
            neighbors[s].add(d)
            neighbors[d].add(s)
        sims = []
        attempts = 0
        max_attempts = max(num_pairs * 20, 1000)
        while len(sims) < num_pairs and attempts < max_attempts:
            i = int(rng.integers(0, n_edges))
            j = int(rng.integers(0, n_edges - 1))
            if j >= i:
                j += 1
            attempts += 1
            if src[i] == src[j] or src[i] == dst[j] or dst[i] == src[j] or dst[i] == dst[j]:
                continue
            if neighbors.get(src[i], set()) & neighbors.get(src[j], set()):
                continue
            if neighbors.get(src[i], set()) & neighbors.get(dst[j], set()):
                continue
            if neighbors.get(dst[i], set()) & neighbors.get(src[j], set()):
                continue
            if neighbors.get(dst[i], set()) & neighbors.get(dst[j], set()):
                continue
            sims.append(float(np.dot(edge_unit[i], edge_unit[j])))
        if len(sims) < num_pairs:
            while len(sims) < num_pairs:
                i = int(rng.integers(0, n_edges))
                j = int(rng.integers(0, n_edges - 1))
                if j >= i:
                    j += 1
                sims.append(float(np.dot(edge_unit[i], edge_unit[j])))
        sims = np.clip(np.asarray(sims, dtype=float), 0.0, 1.0)
        if quantile is None or quantile <= 0:
            return float(np.max(sims)) if sims.size > 0 else 0.0
        q = float(np.clip(quantile, 0.0, 1.0))
        return float(np.quantile(sims, q)) if sims.size > 0 else 0.0

    def merge_top_l_by_ppd(old_ppd, old_idx, new_ppd, new_idx, l):
        if l <= 0:
            return [], []

        old_ppd = np.asarray(old_ppd, dtype=float)
        old_idx = np.asarray(old_idx, dtype=int)
        new_ppd = np.asarray(new_ppd, dtype=float)
        new_idx = np.asarray(new_idx, dtype=int)

        if old_ppd.size == 0:
            all_ppd = new_ppd
            all_idx = new_idx
        elif new_ppd.size == 0:
            all_ppd = old_ppd
            all_idx = old_idx
        else:
            all_ppd = np.concatenate((old_ppd, new_ppd))
            all_idx = np.concatenate((old_idx, new_idx))

        if all_ppd.size == 0:
            return [], []

        if all_ppd.size <= l:
            keep = np.argsort(all_ppd)[::-1]
        else:
            keep = np.argpartition(all_ppd, -l)[-l:]
            keep = keep[np.argsort(all_ppd[keep])[::-1]]

        return all_ppd[keep].tolist(), all_idx[keep].tolist()

    # No new sub-events: nothing to update besides count consistency.
    if num_new == 0 or len(edges) == 0:
        G.graph['SE_num_total'] = num_total
        return G

    pre_topl_stats = init_ppd_stats()
    post_topl_stats = init_ppd_stats()
    vectorized_done = False
    try:
        edge_feat_matrix = np.asarray([G.edges[e]['feat'] for e in edges], dtype=float)
        event_matrix_new = np.asarray(event_vectors_for_update, dtype=float)

        if edge_feat_matrix.ndim == 2 and event_matrix_new.ndim == 2 and edge_feat_matrix.shape[1] == event_matrix_new.shape[1]:
            ensure_hinge_floor_state()
            edge_norms = np.linalg.norm(edge_feat_matrix, axis=1, keepdims=True)
            edge_norms[edge_norms == 0] = 1.0
            edge_unit = edge_feat_matrix / edge_norms

            event_norms = np.linalg.norm(event_matrix_new, axis=1, keepdims=True)
            event_norms[event_norms == 0] = 1.0
            event_unit_new = event_matrix_new / event_norms

            total_edges = len(edges)
            batch_size = 4096

            t_floor = G.graph.get('calib_hinge_t_first')
            mu_first = G.graph.get('calib_mu_first')
            sigma_first = G.graph.get('calib_sigma_first')
            if hinge_floor == 'random':
                if t_floor is None:
                    rng = np.random.default_rng(int(getattr(args, 'hinge_random_seed', 12345)))
                    t_floor = compute_random_pair_floor(
                        edge_unit,
                        edges,
                        rng,
                        int(getattr(args, 'hinge_random_pairs', 1000)),
                        float(getattr(args, 'hrq', 0.95)),
                    )
                    G.graph['calib_hinge_t_first'] = float(t_floor)
                    print(
                        f"SocialUpdate time_signal={args.time_signal} "
                        f"floor=random hinge_t={float(t_floor):.6f} "
                        f"pairs={int(getattr(args, 'hinge_random_pairs', 1000))} "
                        f"q={float(getattr(args, 'hrq', 0.95))}"
                    )
            else:
                if mu_first is None or sigma_first is None:
                    cos_sum = 0.0
                    cos_sumsq = 0.0
                    cos_count = 0
                    for left in range(0, total_edges, batch_size):
                        right = min(left + batch_size, total_edges)
                        cos_chunk = np.clip(edge_unit[left:right] @ event_unit_new.T, 0.0, 1.0)
                        cos_sum += float(np.sum(cos_chunk))
                        cos_sumsq += float(np.sum(cos_chunk * cos_chunk))
                        cos_count += int(cos_chunk.size)
                    mu_first = (cos_sum / cos_count) if cos_count > 0 else 0.0
                    var = (cos_sumsq / cos_count) - (mu_first * mu_first) if cos_count > 0 else 0.0
                    sigma_first = float(np.sqrt(max(var, 0.0)))
                    G.graph['calib_mu_first'] = float(mu_first)
                    G.graph['calib_sigma_first'] = float(sigma_first)
                if sigma_first is None:
                    sigma_first = 0.0

            progress_start = time.time()
            log_every = int(getattr(args, 'su_log_every', 0))
            if log_every <= 0:
                log_every = max(1, total_edges // 20)
            new_indices = np.arange(se_offset, se_offset + num_new, dtype=int)

            for left in range(0, total_edges, batch_size):
                right = min(left + batch_size, total_edges)
                cos_chunk = np.clip(edge_unit[left:right] @ event_unit_new.T, 0.0, 1.0)
                if hinge_floor == 'random':
                    sim_cal_chunk = calibrate_cosine_hinge_t(cos_chunk, t_floor or 0.0)
                else:
                    sim_cal_chunk = calibrate_cosine_hinge(
                        cos_chunk,
                        mu_first,
                        sigma_first,
                        getattr(args, 'hinge_k', 1.0),
                    )

                for local_idx, edge in enumerate(edges[left:right]):
                    edge_data = G.edges[edge]
                    
                    sign_r = 1 if edge_data.get('flag', False) else 0
                    scale = 1 + sign_r
                    ppd_new = np.minimum(1.0, scale * np.log2(1.0 + sim_cal_chunk[local_idx]))
                    update_ppd_stats(pre_topl_stats, ppd_new)
            
                    old_ppd = edge_data.get('ppd', [])
                    old_idx = edge_data.get('se_idx', edge_data.get('sim_idx', []))
                    kept_ppd, kept_idx = merge_top_l_by_ppd(old_ppd, old_idx, ppd_new, new_indices, top_l)
                    edge_data['ppd'] = kept_ppd
                    edge_data['se_idx'] = kept_idx
                    update_ppd_stats(post_topl_stats, kept_ppd)

                processed = right
                if processed % log_every == 0 or processed == total_edges:
                    elapsed = max(time.time() - progress_start, 1e-9)
                    rate = processed / elapsed
                    pct = 100.0 * processed / max(total_edges, 1)
                    remain = total_edges - processed
                    eta = remain / max(rate, 1e-9)
                    print(f"SocialUpdate time_signal={args.time_signal} progress: {processed}/{total_edges} ({pct:.2f}%), {rate:.1f} edges/s, ETA {eta:.1f}s")
            vectorized_done = True
    except Exception:
        vectorized_done = False
        pre_topl_stats = init_ppd_stats()
        post_topl_stats = init_ppd_stats()

    if not vectorized_done:
        total_edges = len(edges)
        progress_start = time.time()
        log_every = int(getattr(args, 'su_log_every', 0))
        if log_every <= 0:
            log_every = max(1, total_edges // 20)

        edge_feat_matrix = np.asarray([G.edges[e]['feat'] for e in edges], dtype=float)
        event_matrix_new = np.asarray(event_vectors_for_update, dtype=float)
        ensure_hinge_floor_state()

        t_floor = G.graph.get('calib_hinge_t_first')
        mu_first = G.graph.get('calib_mu_first')
        sigma_first = G.graph.get('calib_sigma_first')
        if hinge_floor == 'random':
            if t_floor is None:
                edge_unit = edge_feat_matrix / np.maximum(np.linalg.norm(edge_feat_matrix, axis=1, keepdims=True), 1e-12)
                rng = np.random.default_rng(int(getattr(args, 'hinge_random_seed', 12345)))
                t_floor = compute_random_pair_floor(
                    edge_unit,
                    edges,
                    rng,
                    int(getattr(args, 'hinge_random_pairs', 1000)),
                    float(getattr(args, 'hrq', 0.95)),
                )
                G.graph['calib_hinge_t_first'] = float(t_floor)
                print(
                    f"SocialUpdate time_signal={args.time_signal} "
                    f"floor=random hinge_t={float(t_floor):.6f} "
                    f"pairs={int(getattr(args, 'hinge_random_pairs', 1000))} "
                    f"q={float(getattr(args, 'hrq', 0.95))}"
                )
        else:
            if mu_first is None or sigma_first is None:
                cos_sum = 0.0
                cos_sumsq = 0.0
                cos_count = 0
                for edge in edges:
                    feat = np.asarray(G.edges[edge]['feat'], dtype=float)
                    for v in event_vectors_for_update:
                        v_arr = np.asarray(v, dtype=float)
                        c = cosine_similarity(v_arr, feat)
                        cos_sum += c
                        cos_sumsq += c * c
                        cos_count += 1
                mu_first = (cos_sum / cos_count) if cos_count > 0 else 0.0
                var = (cos_sumsq / cos_count) - (mu_first * mu_first) if cos_count > 0 else 0.0
                sigma_first = float(np.sqrt(max(var, 0.0)))
                G.graph['calib_mu_first'] = float(mu_first)
                G.graph['calib_sigma_first'] = float(sigma_first)
            if sigma_first is None:
                sigma_first = 0.0

        new_indices = np.arange(se_offset, se_offset + num_new, dtype=int)

        for idx, edge in enumerate(edges, start=1):
            edge_data = G.edges[edge]
            feat = np.asarray(edge_data['feat'], dtype=float)
            cos_new = []
            for v in event_vectors_for_update:
                v_arr = np.asarray(v, dtype=float)
                cos_new.append(cosine_similarity(v_arr, feat))
            cos_new = np.asarray(cos_new, dtype=float)
            if hinge_floor == 'random':
                sim_cal = calibrate_cosine_hinge_t(cos_new, t_floor or 0.0)
            else:
                sim_cal = calibrate_cosine_hinge(
                    cos_new,
                    mu_first,
                    sigma_first,
                    getattr(args, 'hinge_k', 1.0),
                )

            sign_r = 1 if edge_data.get('flag', False) else 0
            scale = 1 + sign_r
            ppd_new = np.minimum(1.0, scale * np.log2(1.0 + sim_cal))
            update_ppd_stats(pre_topl_stats, ppd_new)

            kept_ppd, kept_idx = merge_top_l_by_ppd(edge_data.get('ppd', []), edge_data.get('se_idx', edge_data.get('sim_idx', [])), ppd_new, new_indices, top_l)
            edge_data['ppd'] = kept_ppd
            edge_data['se_idx'] = kept_idx
            update_ppd_stats(post_topl_stats, kept_ppd)

            if idx % log_every == 0 or idx == total_edges:
                elapsed = max(time.time() - progress_start, 1e-9)
                rate = idx / elapsed
                pct = 100.0 * idx / max(total_edges, 1)
                remain = total_edges - idx
                eta = remain / max(rate, 1e-9)
                print(f"SocialUpdate time_signal={args.time_signal} progress: {idx}/{total_edges} ({pct:.2f}%), {rate:.1f} edges/s, ETA {eta:.1f}s")

    print(
        f"SocialUpdate time_signal={args.time_signal} floor={hinge_floor} pre_topl_ppd_stats: "
        f"{ppd_stats_summary(pre_topl_stats)}"
    )
    print(
        f"SocialUpdate time_signal={args.time_signal} floor={hinge_floor} post_topl_ppd_stats: "
        f"{ppd_stats_summary(post_topl_stats)}"
    )

    G.graph['SE_num_total'] = num_total

    end_time = time.time()
    update_cost = end_time - start_time
    return G
