import ast
import argparse
import networkx as nx
import random
import time
import GraphProcessor
import EdgeSelection
import os
from datetime import datetime
random.seed(123)


def parse_args():
    parser = argparse.ArgumentParser(description="FEIM")
    parser.add_argument('--data_path_prefix', type=str, default='../{}')
    parser.add_argument('--data_path_suffix', type=str, default='.txt')
    parser.add_argument('--graph', type=str, default='/G_Tokens')
    parser.add_argument('--SE', type=str, default='/SE_Tokens')
    parser.add_argument('--p', type=float, default='5.5e-6',help='the percentage of remained reachability')
    parser.add_argument('--t', type=int, default='10',help='the number of seed users')
    parser.add_argument('--l', type=int, default='20',help='message limit')
    parser.add_argument('--sim_num', type=int, default='1000',help='number of mc simulations')
    parser.add_argument('--algo', type=str, default='FEIM')
    parser.add_argument('--data', type=str, default='NepalEQuake')
    parser.add_argument('--start_time', type=str, default='Sat April 25 00:00:00 2015')
    parser.add_argument('--time_signal', type=int, default='1')
    parser.add_argument('--start_time_format', type=str, default='%a %b %d %H:%M:%S %Y')
    return parser.parse_args()


def simandlog(args, folder_path, rr, targets, diff_g, time_cost):
    with open(folder_path + '/EdgeCost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
        file.write(str(time_cost))
    
    start_time = time.time()
    bene, msg, inf, msg_gap = EdgeSelection.EventInfluenceSimulation(args, diff_g, rr, targets)
    end_time = time.time()
    diff_cost = (end_time - start_time)/args.sim_num

    output = "epi " + str(args.time_signal) + " bene " + str(bene/args.sim_num) + " msg " + str(msg/args.sim_num) + " inf " + str(inf/args.sim_num) + " msg_gap " + str(msg_gap)
    with open(folder_path + '/Outcome_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(output)
            file.write('\n')
    print(output)
    
    with open(folder_path + '/DiffCost_{}.txt'.format(args.time_signal), 'w', encoding = 'utf-8') as file:
            file.write(str(diff_cost))


def diffusion(args, G):
    start_time = time.time()
    folder_path = args.data_path_prefix.format(args.data) + '/{}/k{}l{}p{}'.format(args.algo, args.t, args.l, args.p)
    targets, rr = GraphProcessor.ReachableRangeSearch(args, G)
    diff_g = EdgeSelection.FES(rr, targets)
    end_time = time.time()
    time_cost = end_time - start_time
    simandlog(args, folder_path, rr, targets, diff_g, time_cost)
    

def main(args):
    print(args.data_path_prefix.format(args.data))
    print(args.algo)
    print("l{} t{}".format(args.l, args.t))
    folder_path = args.data_path_prefix.format(args.data) + '/{}/k{}l{}p{}'.format(args.algo, args.t, args.l, args.p)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    G = nx.DiGraph()
    with open(args.data_path_prefix.format(args.data) + args.graph + args.data_path_suffix, 'r', encoding = 'utf-8') as file:
        for line in file:
            edge, tokens, time = line.strip().split('\t')
            edge = ast.literal_eval(edge)
            sender = edge[0]
            receiver = edge[1]
            G.add_edge(sender, receiver)
            G.edges[edge]['feat'] = ast.literal_eval(tokens)
            G.edges[edge]['time'] = int(time)
            G.edges[edge]['sim'] = []
            G.edges[edge]['min_sim'] = 1
            G.edges[edge]['SE_num'] = 0
            G.edges[edge]['flag'] = False
	
    for edge in G.edges():
        if G.out_degree(edge[1]) > 0:
            for inter_edge in G.edges(edge[1]):
                if G.edges[edge]['time'] < G.edges[inter_edge]['time']:
                    G.edges[edge]['flag'] = True
                    break
                else:
                    continue

    start_time = int(datetime.strptime(args.start_time, args.start_time_format).timestamp() * 1000)
    pre_signal = 0
    twelve_hours_in_ms = 12 * 60 * 60 * 1000
    Event_tokens_for_update = []
    
    
    with open(args.data_path_prefix + args.SE + args.data_path_suffix, 'r', encoding = 'utf-8') as file:
        for line in file:
            token_str, time_str = line.strip().split('\t')
            SE_tokens = ast.literal_eval(token_str)
            time = int(time_str)
            args.time_signal = (time - start_time) // twelve_hours_in_ms
                
            if args.time_signal > pre_signal:
                if len(Event_tokens_for_update) > 0:
                    G = GraphProcessor.SocialUpdate(args, G, Event_tokens_for_update)
                    Event_tokens_for_update = []
                    diffusion(args, G)
                pre_signal = args.time_signal
            else:
                Event_tokens_for_update.append(SE_tokens)

    if len(Event_tokens_for_update) > 0:
        args.time_signal += 1
        G = GraphProcessor.SocialUpdate(args, G, Event_tokens_for_update)
        Event_tokens_for_update = []
        diffusion(args, G)


if __name__ == "__main__":
	args = parse_args()
	main(args)
