This file provides instruction on running the code.

Exmaple of Running command: 

data preprocessing(download conceptnet from https://conceptnet.io/):

python ConTF-IDF.py --data NepalEQuake

FEIM:

python main.py --data NepalEQuake --t 5 --l 20

data: the name of input dataset
t: the number of seed nodes
l: the number of messages received per user
