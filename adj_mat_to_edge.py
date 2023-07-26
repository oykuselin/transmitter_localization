import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import torch 

current_path = os.getcwd()
subdir = 'adjacency_matrix.txt'
matrix_file_path = os.path.join(current_path, subdir)
adj_t = np.loadtxt(matrix_file_path)


adj_t = torch.tensor(adj_t)
edge_index = adj_t.nonzero().t().contiguous()

print(edge_index)

np.savetxt('edge_index.txt', edge_index.numpy(), fmt='%d')
torch.save(edge_index, 'edge_index.pt')

