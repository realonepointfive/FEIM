import random
import random
from collections import deque


def _fairness_coefficient_from_binary(benefits):
    n = len(benefits)
    if n <= 0:
        return 0.0
    total_bene = int(sum(benefits))
    if total_bene <= 0:
        return 0.0
    x_sorted = sorted(float(x) for x in benefits)
    disparity_sum = 0.0
    for i, x in enumerate(x_sorted, start=1):
        disparity_sum += (2 * i - n - 1) * x
    return disparity_sum / (n * total_bene)


def independent_cascade_once(graph, seeds, rng=None, return_round_fairness=False):
    """
    Run one Independent Cascade simulation.

    Returns:
      activated_set: all activated nodes (including seeds)
      newly_activated_count: number of non-seed nodes activated
      attempts: number of edge activation attempts made
      round_fairness: fairness coefficient per round (optional)
    """
    if rng is None:
        rng = random.Random()

    activated = set(seeds)
    frontier = deque(activated)
    attempts = 0
    round_fairness = []

    while frontier:
        attempted_nodes_this_round = set()
        u = frontier.popleft()
        current_round = [u]
        while frontier:
            current_round.append(frontier.popleft())

        next_frontier = []
        for src in current_round:
            for v in graph.successors(src):
                if v in activated:
                    continue
                attempted_nodes_this_round.add(v)
                attempts += 1
                p_uv = graph.edges[src, v]['pp']
                if rng.random() < p_uv:
                    activated.add(v)
                    next_frontier.append(v)

        if len(attempted_nodes_this_round) > 0:
            benefits = [1 if node in activated else 0 for node in attempted_nodes_this_round]
            round_fairness.append(_fairness_coefficient_from_binary(benefits))
        else:
            round_fairness.append(0.0)

        frontier = deque(next_frontier)

    newly_activated_count = max(0, len(activated) - len(set(seeds)))
    if return_round_fairness:
        return newly_activated_count, attempts, round_fairness
    return newly_activated_count, attempts


def _build_ic_cache(graph, seeds):
    node_list = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n = len(node_list)

    out_neighbors = [[] for _ in range(n)]
    out_probs = [[] for _ in range(n)]
    for u, v in graph.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        ui = node_to_idx[u]
        vi = node_to_idx[v]
        p = float(graph.edges[u, v].get('pp', 0.0))
        if p <= 0.0:
            continue
        if p > 1.0:
            p = 1.0
        out_neighbors[ui].append(vi)
        out_probs[ui].append(p)

    seed_idx = [node_to_idx[s] for s in seeds if s in node_to_idx]
    seed_set = set(seeds)
    return {
        "node_list": node_list,
        "node_to_idx": node_to_idx,
        "n": n,
        "out_neighbors": out_neighbors,
        "out_probs": out_probs,
        "seed_idx": seed_idx,
        "seed_count": len(seed_set),
    }


def _independent_cascade_once_cached(cache, rng, return_round_fairness=False):
    n = cache["n"]
    out_neighbors = cache["out_neighbors"]
    out_probs = cache["out_probs"]
    seed_idx = cache["seed_idx"]
    seed_count = cache["seed_count"]

    active = [False] * n
    frontier = []
    for s in seed_idx:
        if not active[s]:
            active[s] = True
            frontier.append(s)

    attempts = 0
    round_fairness = []

    while frontier:
        next_frontier = []
        attempted_nodes_this_round = set()

        for u in frontier:
            nbrs = out_neighbors[u]
            probs = out_probs[u]
            for i, v in enumerate(nbrs):
                if active[v]:
                    continue
                attempts += 1
                attempted_nodes_this_round.add(v)
                if rng.random() < probs[i]:
                    active[v] = True
                    next_frontier.append(v)

        if return_round_fairness:
            if len(attempted_nodes_this_round) > 0:
                benefits = [1 if active[v] else 0 for v in attempted_nodes_this_round]
                round_fairness.append(_fairness_coefficient_from_binary(benefits))
            else:
                round_fairness.append(0.0)

        frontier = next_frontier

    total_active = 0
    for flag in active:
        if flag:
            total_active += 1
    newly_activated_count = max(0, total_active - seed_count)

    if return_round_fairness:
        return newly_activated_count, attempts, round_fairness
    return newly_activated_count, attempts


def ICmc(args, graph, seeds, seed=12345):
    """
    Monte Carlo estimate for Independent Cascade.

    Returns:
      total_inf, total_attempts, msg_gap
      where msg_gap is the average of all simulation-round fairness coefficients
    """
    rng = random.Random(seed)
    cache = _build_ic_cache(graph, seeds)
    sim_num = max(1, int(args.sim_num))

    total_inf = 0.0
    total_attempts = 0.0
    fairness_sum = 0.0
    fairness_cnt = 0

    for _ in range(sim_num):
        inf, attempts, round_fairness = _independent_cascade_once_cached(
            cache, rng=rng, return_round_fairness=True
        )
        if len(round_fairness) > 0:
            fairness_sum += float(sum(round_fairness))
            fairness_cnt += len(round_fairness)
        total_inf += inf
        total_attempts += attempts

    msg_gap = (fairness_sum / fairness_cnt) if fairness_cnt > 0 else 0.0
    return total_inf, total_attempts, msg_gap
