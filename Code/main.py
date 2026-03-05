import ast
import argparse
import networkx as nx
import random
import time
import GraphProcessor
import EventInfluence
import Influence
import os
from datetime import datetime
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(description="FEIM")
    parser.add_argument('--data_path_prefix', type=str, default='../Data/{}')
    parser.add_argument('--data_path_suffix', type=str, default='.txt')
    parser.add_argument('--graph', type=str, default='/Msg_concept_vectors')
    parser.add_argument('--SE', type=str, default='/SE_concept_vectors')
    parser.add_argument('--p', type=float, default='5.5e-6',help='the percentage of remained reachability')
    parser.add_argument('--t', type=int, default='10',help='the number of seed users')
    parser.add_argument('--l', type=int, default='20',help='message limit')
    parser.add_argument('--sim_num', type=int, default='10000',help='number of mc simulations')
    parser.add_argument('--workers', type=int, default='8', help='number of worker processes for simulation')
    parser.add_argument('--hinge_k', type=float, default='1.0',
                        help='sigma-shift hinge calibration: T = mu + k*sigma')
    parser.add_argument('--hinge_floor', type=str, default='random', choices=['sigma', 'random'],
                        help='hinge floor type: sigma-shift or random-pair baseline')
    parser.add_argument('--hinge_random_pairs', type=int, default='10000',
                        help='number of random edge pairs for empirical floor')
    parser.add_argument('--hrq', type=float, default='0.99',
                        help='quantile of random-pair similarities used as floor (<=0 means max)')
    parser.add_argument('--hinge_random_seed', type=int, default='12345',
                        help='random seed for random-pair floor sampling')
    parser.add_argument('--eps', type=float, default='0',
                        help='epsilon for EventAssignment update threshold')
    parser.add_argument('--su_log_every', type=int, default='10000',
                        help='SocialUpdate progress log interval in edges (0 means auto)')
    parser.add_argument('--algo', type=str, default='FEIM')
    parser.add_argument('--data', type=str, default='NepalEQuake')
    parser.add_argument('--start_time', type=str, default='auto',
                        help='dataset start time; use "auto" to infer from --data')
    parser.add_argument('--time_signal', type=int, default='1')
    parser.add_argument('--start_time_signal', type=int, default='0',
                        help='begin diffusion optimization at this time_signal (inclusive)')
    parser.add_argument('--start_time_format', type=str, default='%a %B %d %H:%M:%S %Y')
    parser.add_argument('--once_info_loss', action='store_true',
                        help='Run one SocialUpdate batch, then compute information loss and exit.')
    return parser.parse_args()


def resolve_start_time(args):
    if isinstance(args.start_time, str) and args.start_time.lower() != 'auto':
        return args.start_time

    # Match dataset to start time by the reported time windows:
    # NepalEQuake: 25 Apr 2015
    # TexasFlood: 22 May 2015
    # WC2014: 12 Jun 2014
    dataset_start = {
        'nepalequake': 'Sat April 25 00:00:00 2015',
        'texasflood': 'Fri May 22 00:00:00 2015',
        'wc2014': 'Thu June 12 00:00:00 2014',
    }
    return dataset_start.get(args.data.lower(), 'Sat April 25 00:00:00 2015')


def resolve_update_interval_ms(args):
    data_key = str(args.data).lower()
    if data_key == 'wc2014':
        return 1 * 60 * 60 * 1000
    return 12 * 60 * 60 * 1000


def resolve_p(args):
    data_key = str(args.data).lower()
    if data_key == 'nepalequake':
        return 2.2e-6
    if data_key == 'texasflood':
        return 2.5e-8
    if data_key == 'wc2014':
        return 6e-7
    return args.p


def resolve_hrq(args):
    data_key = str(args.data).lower()
    if data_key == 'nepalequake':
        return 0.99
    if data_key == 'texasflood':
        return 0.99
    if data_key == 'wc2014':
        return 0.99
    return args.hrq


def resolve_min_social_update_signal(args):
    data_key = str(args.data).lower()
    if data_key == 'nepalequake':
        return 1
    if data_key == 'texasflood':
        return 13
    if data_key == 'wc2014':
        return 0
    return 0


def build_experiment_folder(args):
    return args.data_path_prefix.format(args.data) + '/{}/k{}l{}p{}q{}eps{}'.format(
        args.algo,
        args.t,
        args.l,
        args.p,
        format(getattr(args, "hrq", 0.95), "g"),
        format(getattr(args, "eps", 1e-16), "g"),
    )


def build_rrsp_folder(args):
    return args.data_path_prefix.format(args.data) + '/RRSp/l{}p{}q{}'.format(
        args.l,
        args.p,
        format(getattr(args, "hrq", 0.95), "g"),
    )


def simandlog(args, folder_path, seeds, diff_g, time_cost, rr=None):
    os.makedirs(folder_path, exist_ok=True)
    with open(folder_path + '/Cost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
        file.write(str(time_cost))
    
    start_time = time.time()
    if args.algo == 'TIM':
        # inf, msg, msg_gap = Influence.ICmc(args, diff_g, seeds)
        # bene = inf
        bene, msg, inf, msg_gap, _, _ = EventInfluence.EventInfluenceSimulation(args, diff_g, seeds, rr=rr)
    else:
        bene, msg, inf, msg_gap, _, _ = EventInfluence.EventInfluenceSimulation(args, diff_g, seeds, rr=rr)
    end_time = time.time()
    diff_cost = (end_time - start_time)/args.sim_num

    output = "epi " + str(args.time_signal) + " bene " + str(bene/args.sim_num) + " msg " + str(msg/args.sim_num) + " inf " + str(inf/args.sim_num) + " msg_gap " + str(msg_gap)
    with open(folder_path + '/Outcome_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(output)
            file.write('\n')
    print(output)
    
    with open(folder_path + '/DiffCost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(str(diff_cost))


def load_rr_and_seeds(folder_path, k, time_signal):
    seeds_path = os.path.join(folder_path, f"Seeds_{k}_{time_signal}.txt")
    rr_path = os.path.join(folder_path, f"RR_{k}_{time_signal}.txt")
    if not (os.path.isfile(seeds_path) and os.path.isfile(rr_path)):
        return None, None

    seeds = []
    with open(seeds_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                seeds.append(ast.literal_eval(s) if s.startswith("(") or s.startswith("[") else s)

    rr = nx.DiGraph()
    with open(rr_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            u = ast.literal_eval(parts[0]) if parts[0].startswith("(") or parts[0].startswith("[") else parts[0]
            v = ast.literal_eval(parts[1]) if parts[1].startswith("(") or parts[1].startswith("[") else parts[1]
            sim_idx = ast.literal_eval(parts[2])
            ppd = ast.literal_eval(parts[3])
            rr.add_edge(u, v)
            rr.edges[u, v]['se_idx'] = sim_idx
            rr.edges[u, v]['ppd'] = ppd
    print(
        f"Loaded RR/Seeds cache at time_signal={time_signal}: "
        f"seeds={len(seeds)} rr_nodes={rr.number_of_nodes()} rr_edges={rr.number_of_edges()}"
    )
    return rr, seeds


def has_rr_seed_cache(folder_path, k, time_signal):
    seeds_new = os.path.join(folder_path, f"Seeds_{k}_{time_signal}.txt")
    rr_new = os.path.join(folder_path, f"RR_{k}_{time_signal}.txt")
    if os.path.isfile(seeds_new) and os.path.isfile(rr_new):
        return True
    else:
        return False


def DiffOptim(args, G):
    start_time = time.time()
    folder_path = build_experiment_folder(args)
    rr = None
    if args.algo == 'TIM':
        # TIM always constructs its own diffusion graph.
        seeds, diff_g = GraphProcessor.TIM(args, G)
    else:
        rr_folder = build_rrsp_folder(args)
        rr, seeds = load_rr_and_seeds(rr_folder, args.t, args.time_signal)
        if rr is None or seeds is None:
            seeds, rr = GraphProcessor.RRSparse(args, G)

        if args.algo == 'FEIM-NoEA':
            diff_g = rr.copy()
        elif args.algo == 'B-PEI':
            diff_g = GraphProcessor.EdgeSelection(rr, seeds)
        elif args.algo == 'FEIM':
            diff_g = GraphProcessor.EventAssignment(args, rr, seeds)

    end_time = time.time()
    time_cost = end_time - start_time
    simandlog(args, folder_path, seeds, diff_g, time_cost, rr=rr)


def main(args):
    print(args.data_path_prefix.format(args.data))
    print(args.algo)
    print("l{} t{}".format(args.l, args.t))
    args.start_time = resolve_start_time(args)
    args.p = resolve_p(args)
    args.hrq = resolve_hrq(args)
    print(f"start_time={args.start_time}")
    print(f"p={args.p}")
    print(f"hrq={args.hrq}")
    folder_path = args.data_path_prefix.format(args.data) + '/{}'.format(args.algo)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    G = nx.DiGraph()
    with open(args.data_path_prefix.format(args.data) + args.graph + args.data_path_suffix, 'r', encoding = 'utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            edge_str, feat_str, time_str = parts
            edge = ast.literal_eval(edge_str)
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                continue
            sender, receiver = edge[0], edge[1]
            # Msg_concept_vectors stores vectors as space-separated numbers.
            # Also supports legacy list format: "[v1, v2, ...]".
            if feat_str.startswith('[') and feat_str.endswith(']'):
                feat = ast.literal_eval(feat_str)
            else:
                feat = [float(x) for x in feat_str.split()]
            msg_time = int(time_str)

            if not G.has_edge(sender, receiver):
                G.add_edge(sender, receiver)
                G.edges[edge]['feat_sum'] = feat[:]
                G.edges[edge]['feat_cnt'] = 1
                G.edges[edge]['time'] = msg_time
                G.edges[edge]['flag'] = False
            else:
                feat_sum = G.edges[edge]['feat_sum']
                if len(feat_sum) == len(feat):
                    for i in range(len(feat)):
                        feat_sum[i] += feat[i]
                    G.edges[edge]['feat_cnt'] += 1
                G.edges[edge]['time'] = max(G.edges[edge]['time'], msg_time)

    # Finalize per-edge feature as the average vector over all observed messages.
    for edge in G.edges():
        feat_sum = G.edges[edge]['feat_sum']
        feat_cnt = G.edges[edge]['feat_cnt']
        G.edges[edge]['feat'] = [x / feat_cnt for x in feat_sum]
        del G.edges[edge]['feat_sum']
        del G.edges[edge]['feat_cnt']
	
    for edge in G.edges():
        if G.out_degree(edge[1]) > 0:
            for inter_edge in G.edges(edge[1]):
                if G.edges[edge]['time'] < G.edges[inter_edge]['time']:
                    G.edges[edge]['flag'] = True
                    break
                else:
                    continue

    start_time = int(datetime.strptime(args.start_time, args.start_time_format).timestamp() * 1000)
    update_interval_ms = resolve_update_interval_ms(args)
    min_social_signal = resolve_min_social_update_signal(args)
    optim_start_signal = max(int(args.start_time_signal), int(min_social_signal))
    event_vectors_for_update = []

    if args.algo == 'InfoLoss':
        print(
            f"InfoLoss mode: applying SocialUpdate cumulatively from signal 0 "
            f"to min_social_signal={min_social_signal}"
        )

        def apply_info_loss_batch(graph, batch_signal, vectors):
            # In InfoLoss mode, build G from signal 0 up to min_social_signal.
            if batch_signal < 0 or batch_signal > min_social_signal:
                return graph, False
            args.time_signal = batch_signal
            print(
                f"Run SocialUpdate at time_signal={args.time_signal}: "
                f"sub_events_used={len(vectors)}"
            )
            return EventInfluence.SocialUpdate(args, graph, vectors), True

        current_info_signal = None
        reached_min_signal = False
        with open(args.data_path_prefix.format(args.data) + args.SE + args.data_path_suffix, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) != 2:
                    continue
                vector_str, time_str = parts
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    se_vectors = ast.literal_eval(vector_str)
                else:
                    se_vectors = [float(x) for x in vector_str.split()]
                time = int(time_str)
                signal = (time - start_time) // update_interval_ms

                if current_info_signal is None:
                    current_info_signal = signal

                # Flush previous signal batch before starting a new one.
                if signal != current_info_signal:
                    if len(event_vectors_for_update) > 0:
                        G, applied = apply_info_loss_batch(G, current_info_signal, event_vectors_for_update)
                        if applied:
                            if current_info_signal == min_social_signal:
                                reached_min_signal = True
                                GraphProcessor.InfoLossparse(G, args.p)
                                return
                    event_vectors_for_update = []
                    current_info_signal = signal

                event_vectors_for_update.append(se_vectors)
                if signal > min_social_signal and reached_min_signal:
                    break

        # Flush trailing batch.
        if current_info_signal is not None and len(event_vectors_for_update) > 0:
            G, applied = apply_info_loss_batch(G, current_info_signal, event_vectors_for_update)
            if applied and current_info_signal == min_social_signal:
                reached_min_signal = True
                GraphProcessor.InfoLossparse(G, args.p)
                return

        raise ValueError(
            f"InfoLoss mode could not reach min_social_signal={min_social_signal} "
            f"from available SE batches."
        )
        return
    
    current_signal = None
    # Batches skipped due to existing RR/Seeds cache.
    # If a later time_signal misses cache, we replay these updates first
    # so G is rebuilt from the initial state up to that missing time_signal.
    deferred_updates = []
    with open(args.data_path_prefix.format(args.data) + args.SE + args.data_path_suffix, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            vector_str, time_str = parts
            if vector_str.startswith('[') and vector_str.endswith(']'):
                se_vectors = ast.literal_eval(vector_str)
            else:
                se_vectors = [float(x) for x in vector_str.split()]
            time = int(time_str)
            signal = (time - start_time) // update_interval_ms

            if current_signal is None:
                current_signal = signal

            # Flush previous signal batch before starting a new one.
            if signal != current_signal:
                social_updated_in_flush = False
                if len(event_vectors_for_update) > 0:
                    args.time_signal = current_signal
                    rr_folder = build_rrsp_folder(args)
                    if args.algo != 'TIM' and has_rr_seed_cache(rr_folder, args.t, current_signal):
                        print(f"Skip SocialUpdate at time_signal={current_signal}: RR/Seeds cache exists.")
                        deferred_updates.append((current_signal, event_vectors_for_update[:]))
                    else:
                        if len(deferred_updates) > 0:
                            print(
                                f"RR/Seeds missing at time_signal={current_signal}. "
                                f"Replaying {len(deferred_updates)} deferred SocialUpdate batches first."
                            )
                            for deferred_signal, deferred_vectors in deferred_updates:
                                args.time_signal = deferred_signal
                                print(
                                    f"Run SocialUpdate at time_signal={args.time_signal}: "
                                    f"sub_events_used={len(deferred_vectors)}"
                                )
                                G = EventInfluence.SocialUpdate(args, G, deferred_vectors)
                                social_updated_in_flush = True
                            deferred_updates = []
                        args.time_signal = current_signal
                        print(
                            f"Run SocialUpdate at time_signal={args.time_signal}: "
                            f"sub_events_used={len(event_vectors_for_update)}"
                        )
                        G = EventInfluence.SocialUpdate(args, G, event_vectors_for_update)
                        social_updated_in_flush = True
                    if current_signal >= optim_start_signal:
                        args.time_signal = current_signal
                        DiffOptim(args, G)
                        if args.algo == 'TIM' and social_updated_in_flush:
                            return
                event_vectors_for_update = []
                current_signal = signal

            # Always keep the current line's sub-event.
            event_vectors_for_update.append(se_vectors)

    # Flush the last signal batch.
    if current_signal is not None and len(event_vectors_for_update) > 0:
        social_updated_in_flush = False
        args.time_signal = current_signal
        rr_folder = build_rrsp_folder(args)
        if args.algo != 'TIM' and has_rr_seed_cache(rr_folder, args.t, current_signal):
            print(f"Skip SocialUpdate at time_signal={current_signal}: RR/Seeds cache exists.")
            deferred_updates.append((current_signal, event_vectors_for_update[:]))
        else:
            if len(deferred_updates) > 0:
                print(
                    f"RR/Seeds missing at time_signal={current_signal}. "
                    f"Replaying {len(deferred_updates)} deferred SocialUpdate batches first."
                )
                for deferred_signal, deferred_vectors in deferred_updates:
                    args.time_signal = deferred_signal
                    print(
                        f"Run SocialUpdate at time_signal={args.time_signal}: "
                        f"sub_events_used={len(deferred_vectors)}"
                    )
                    G = EventInfluence.SocialUpdate(args, G, deferred_vectors)
                    social_updated_in_flush = True
                deferred_updates = []
            args.time_signal = current_signal
            print(
                f"Run SocialUpdate at time_signal={args.time_signal}: "
                f"sub_events_used={len(event_vectors_for_update)}"
            )
            G = EventInfluence.SocialUpdate(args, G, event_vectors_for_update)
            social_updated_in_flush = True
        if current_signal >= optim_start_signal:
            args.time_signal = current_signal
            DiffOptim(args, G)
            if args.algo == 'TIM' and social_updated_in_flush:
                return


if __name__ == "__main__":
	args = parse_args()
	main(args)


