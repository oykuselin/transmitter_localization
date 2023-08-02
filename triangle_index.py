import os
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import torch 

mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
mesh = mesh.as_open3d
Triangle_indices = {}
vertex_indices_triangles = np.asarray(mesh.triangles)

for index, element in enumerate(vertex_indices_triangles):
    Triangle_indices[index] = list(element)

print(Triangle_indices)

    
 