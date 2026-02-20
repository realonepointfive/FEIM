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


def _build_static_graph_cache(diff_g, seeds):
    nodes = list(diff_g.nodes())
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
    out_eids = [[] for _ in range(n_nodes)]
    for (u, v) in diff_g.edges():
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        ed = diff_g.edges[u, v]
        probs = ed.get('ppd', [])
        idxs = ed.get('sim_idx')
        if idxs is None:
            idxs = list(range(len(probs)))
        eid = len(edge_dst)
        edge_dst.append(vi)
        edge_probs.append(probs)
        edge_idxs.append(idxs)
        out_eids[ui].append(eid)

    return {
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
        'out_eids': out_eids,
    }


def _run_chunk(sim_count, l, x0, y0, base_seed, static):
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

        if seed_idx.size > 0:
            active[seed_idx] = True
        active_nodes = list(seed_idx)

        while True:
            edges_by_target = {}
            for ui in active_nodes:
                for eid in out_eids[ui]:
                    if edge_used[eid]:
                        continue
                    vi = edge_dst[eid]
                    if is_seed[vi] or msg_num[vi] >= l:
                        continue
                    edges_by_target.setdefault(vi, []).append(eid)

            if len(edges_by_target) == 0:
                break

            for vi, in_eids in edges_by_target.items():
                reached[vi] = True
                subevent_prob = {}
                for eid in in_eids:
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

                for eid in in_eids:
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


def EventInfluenceSimulation(args, diff_g, seeds):
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
    workers = max(1, int(getattr(args, 'workers', 1)))
    sim_num = int(args.sim_num)
    report_every = max(1, sim_num // 20)
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
                fut = ex.submit(_run_chunk, c, args.l, x, y, seed, static)
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
    num_events = len(event_vectors_for_update)
    top_l = max(int(args.l), 0)
    se_offset = int(G.graph.get('SE_num_total', 0))
    cosine_theta = float(getattr(args, 'cosine_theta', 0.7))
    cosine_gamma = float(getattr(args, 'cosine_gamma', 2.0))
    cosine_theta = min(max(cosine_theta, 0.0), 0.999999)
    cosine_gamma = max(cosine_gamma, 1.0)

    def merge_top_l(old_sims, old_idx, new_sims, new_idx, l):
        if l <= 0:
            return [], []
        if len(old_sims) == 0:
            all_sims = np.asarray(new_sims, dtype=float)
            all_idx = np.asarray(new_idx, dtype=int)
        elif len(new_sims) == 0:
            all_sims = np.asarray(old_sims, dtype=float)
            all_idx = np.asarray(old_idx, dtype=int)
        else:
            all_sims = np.concatenate((np.asarray(old_sims, dtype=float), np.asarray(new_sims, dtype=float)))
            all_idx = np.concatenate((np.asarray(old_idx, dtype=int), np.asarray(new_idx, dtype=int)))

        if all_sims.size <= l:
            order = np.argsort(all_sims)[::-1]
            return all_sims[order].tolist(), all_idx[order].tolist()

        keep = np.argpartition(all_sims, -l)[-l:]
        keep = keep[np.argsort(all_sims[keep])[::-1]]
        return all_sims[keep].tolist(), all_idx[keep].tolist()

    vectorized_done = False
    if num_events > 0 and len(edges) > 0:
        try:
            edge_feat_matrix = np.asarray([G.edges[e]['feat'] for e in edges], dtype=float)
            event_matrix = np.asarray(event_vectors_for_update, dtype=float)

            if edge_feat_matrix.ndim == 2 and event_matrix.ndim == 2 and edge_feat_matrix.shape[1] == event_matrix.shape[1]:
                event_norms = np.linalg.norm(event_matrix, axis=1, keepdims=True)
                event_norms[event_norms == 0] = 1.0
                event_unit = event_matrix / event_norms

                total_edges = len(edges)
                batch_size = 4096
                progress_start = time.time()

                for left in range(0, total_edges, batch_size):
                    right = min(left + batch_size, total_edges)
                    edge_chunk = edge_feat_matrix[left:right]
                    edge_norms = np.linalg.norm(edge_chunk, axis=1, keepdims=True)
                    edge_norms[edge_norms == 0] = 1.0
                    edge_unit = edge_chunk / edge_norms

                    cos_chunk = edge_unit @ event_unit.T
                    cos_chunk = np.clip(cos_chunk, 0.0, 1.0)
                    sim_chunk = np.clip((cos_chunk - cosine_theta) / max(1.0 - cosine_theta, 1e-12), 0.0, 1.0)
                    sim_chunk = np.power(sim_chunk, cosine_gamma)

                    for local_idx, edge in enumerate(edges[left:right]):
                        edge_data = G.edges[edge]
                        new_indices = np.arange(se_offset, se_offset + num_events, dtype=int)
                        old_sims = edge_data.get('sim', [])
                        old_idx = edge_data.get('sim_idx', [])
                        kept_sims, kept_idx = merge_top_l(old_sims, old_idx, sim_chunk[local_idx], new_indices, top_l)
                        edge_data['sim'] = kept_sims
                        edge_data['sim_idx'] = kept_idx

                    processed = right
                    elapsed = max(time.time() - progress_start, 1e-9)
                    rate = processed / elapsed
                    pct = 100.0 * processed / total_edges
                    remain = total_edges - processed
                    eta = remain / max(rate, 1e-9)
                    print(f"SocialUpdate time_signal={args.time_signal} progress: {processed}/{total_edges} ({pct:.2f}%), {rate:.1f} edges/s, ETA {eta:.1f}s")

                vectorized_done = True
        except Exception:
            vectorized_done = False

    # Fallback path for ragged/mismatched vectors.
    if not vectorized_done:
        total_edges = len(edges)
        progress_start = time.time()
        for idx, edge in enumerate(edges, start=1):
            edge_data = G.edges[edge]
            new_sims = []
            new_indices = []
            for i in range(num_events):
                se_vectors = event_vectors_for_update[i]
                c = cosine_similarity(se_vectors, G.edges[edge]['feat'])
                sim = max(0.0, (c - cosine_theta) / max(1.0 - cosine_theta, 1e-12))
                sim = sim ** cosine_gamma
                new_sims.append(sim)
                new_indices.append(se_offset + i)

            old_sims = edge_data.get('sim', [])
            old_idx = edge_data.get('sim_idx', [])
            kept_sims, kept_idx = merge_top_l(old_sims, old_idx, new_sims, new_indices, top_l)
            edge_data['sim'] = kept_sims
            edge_data['sim_idx'] = kept_idx

            if idx % 2000 == 0 or idx == total_edges:
                elapsed = max(time.time() - progress_start, 1e-9)
                rate = idx / elapsed
                pct = 100.0 * idx / max(total_edges, 1)
                remain = total_edges - idx
                eta = remain / max(rate, 1e-9)
                print(f"SocialUpdate time_signal={args.time_signal} progress: {idx}/{total_edges} ({pct:.2f}%), {rate:.1f} edges/s, ETA {eta:.1f}s")

    all_ppd = []
    for edge in edges:
        G.edges[edge]['ppd'] = G.edges[edge]['sim']
        all_ppd.extend(G.edges[edge]['ppd'])

    if len(all_ppd) > 0:
        ppd_arr = np.asarray(all_ppd, dtype=float)
        print(
            f"SocialUpdate time_signal={args.time_signal} ppd_stats: "
            f"mean={float(np.mean(ppd_arr)):.4f} median={float(np.median(ppd_arr)):.4f} "
            f"p95={float(np.percentile(ppd_arr, 95)):.4f}"
        )

    G.graph['SE_num_total'] = se_offset + num_events

    end_time = time.time()
    update_cost = end_time-start_time
    
    '''folder_path = args.data_path_prefix.format(args.data) + '/{}'.format(args.algo)
    os.makedirs(folder_path, exist_ok=True)
    with open(folder_path + '/Cost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
        file.write(str(update_cost))'''
    return G
