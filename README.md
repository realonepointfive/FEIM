This file provides instruction on running the code.

required library:
networkx 1.11
numpy
numba

Input file format: each line corresponds to an edge and consists of two integers which represent two node id and seperated by space. (See datasets/BK.txt for an example. BK.txt refers to Brightkite in the experiment)

Exmaple of Running command: python main.py --data NepalEQuake --t 5 --l 20

data: the name of input dataset
t: the number of seed nodes
l: the number of messages received per user

Note: all datasets (excluding datasets from Tencent) are public and accessible via http://konect.cc/. Here we only include BK.txt for illustration since Gitfront cannot well handle large-size folders.
