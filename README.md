This file provides instruction on running the code.

required library:
networkx 3.4.2
numpy 2.2.6
numba 0.62.1

Graph file format: each line consists of an edge with two node IDs in parentheses, followed by a list of this edge's tokenized text content in square brackets, and ending with a timestamp in milliseconds indicating the creation time of this edge, all separated by tabs.
Sub-event file format: each line comprises a list of tokenized text content for a sub-event enclosed in square brackets, followed by a timestamp in milliseconds indicating the sub-event's occurrence time, separated by tabs.

Exmaple of Running command: python main.py --data NepalEQuake --t 5 --l 20

data: the name of input dataset
t: the number of seed nodes
l: the number of messages received per user
p: the percentage of remaining reachability values
