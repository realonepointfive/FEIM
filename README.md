This file provides instruction on running the code.

required library:
networkx 1.11
numpy
numba

Graph file format: each line corresponds to an edge and consists of two integers which represent two node id and seperated by space. 
Sub-event file format: 

Exmaple of Running command: python main.py --data NepalEQuake --t 5 --l 20 --p 5.5e-4

data: the name of input dataset
t: the number of seed nodes
l: the number of messages received per user
p: the percentage of remaining reachability values
