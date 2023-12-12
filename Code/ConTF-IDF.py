import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import ast
import random
import re
from sklearn.decomposition import TruncatedSVD

curr_dir = os.path.dirname(__file__)

concept_index = {}
index = 0

with open(curr_dir + '512to516InterMsg.txt', 'r', encoding = 'utf-8') as file:
    for line in file:
        msg_feat = line.split('</msgcontfidf>')[0].split('<msgcontfidf>')[-1].strip()
        concept_value_pairs = msg_feat.split('\t')
        concepts = []
        for concept_value in concept_value_pairs:
            concepts.append(concept_value.split(' ')[0])
        for concept in concepts:
            if concept not in concept_index:
                concept_index[concept] = index
                index += 1
               
with open(curr_dir  + 'SE522to528.txt', 'r', encoding = 'utf-8') as file:
    for line in file:
        msg_feat = line.split('</msgcontfidf>')[0].split('<msgcontfidf>')[-1].strip()
        concept_value_pairs = msg_feat.split('\t')
        concepts = []
        for concept_value in concept_value_pairs:
            concepts.append(concept_value.split(' ')[0])
        for concept in concepts:
            if concept not in concept_index:
                concept_index[concept] = index
                index += 1
                
feat_matrix = np.zeros((296020, len(concept_index)))
row = 0
with open(curr_dir + '512to516InterMsg.txt', 'r', encoding = 'utf-8') as file:
    for line in file:
        msg_feat = line.split('</msgcontfidf>')[0].split('<msgcontfidf>')[-1].strip()
        concept_value_pairs = msg_feat.split('\t')
        concepts = []
        values = []
        for concept_value in concept_value_pairs:
            concepts.append(concept_value.split(' ')[0])
            values.append(concept_value.split(' ')[1])
        for i, concept in enumerate(concepts):
            feat_matrix[row][concept_index[concept]] = float(values[i])
        row += 1
        
with open(curr_dir + 'SE522to528.txt', 'r', encoding = 'utf-8') as file:
    for line in file:
        msg_feat = line.split('</msgcontfidf>')[0].split('<msgcontfidf>')[-1].strip()
        concept_value_pairs = msg_feat.split('\t')
        concepts = []
        values = []
        for concept_value in concept_value_pairs:
            concepts.append(concept_value.split(' ')[0])
            values.append(concept_value.split(' ')[1])
        for i, concept in enumerate(concepts):
            feat_matrix[row][concept_index[concept]] = float(values[i])
        row += 1
        
np.savetxt(curr_dir + 'feat_matrix_512to516.txt', feat_matrix, fmt='%.8f')
'''
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
svd.fit(feat_matrix)
feat_matrix_SVD = svd.transform(feat_matrix)

subevent_matrix = feat_matrix_SVD[-310:]
np.savetxt(dataset_path + 'Flood512to521SG\\SE522to528SVD.txt', subevent_matrix, fmt='%.8f')
msg_matrix = feat_matrix_SVD[:24973]
np.savetxt(dataset_path + 'Flood512to521SG\\MsgSVD.txt', msg_matrix, fmt='%.8f')
'''