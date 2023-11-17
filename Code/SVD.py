import numpy as np
import torch
import os

rela_dir = "../Dataset/Flood_input_data/"
curr_dir = os.path.dirname(__file__)
dataset_path = os.path.normpath(os.path.join(curr_dir, rela_dir))

SE_matrix = np.loadtxt(dataset_path + '522to528Tensor.txt')

U, S, V = torch.linalg.svd(torch(SE_matrix), full_matrices=False)

print(U.shape)