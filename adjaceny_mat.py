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

vertex_indices_triangles = np.asarray(mesh.triangles)
print(vertex_indices_triangles)
print(vertex_indices_triangles[0])
print(vertex_indices_triangles[226])
#print(vertex_indices_triangles)

num_triangles = len(vertex_indices_triangles)
adjacency_matrix = np.zeros((num_triangles, num_triangles), dtype=int)
adjacency_matrix2 = np.zeros((num_triangles, num_triangles), dtype=int)


# Helper function to check if two triangles share a vertex
def share_vertex(triangle1, triangle2):
    for vertex in triangle1:
        if vertex in triangle2:
            return True
    return False

for i in range(num_triangles):
    triangle1 = vertex_indices_triangles[i]
    for j in range(num_triangles):
        triangle2 = vertex_indices_triangles[j]
        if (triangle1[0][0] == triangle2[0][0] and triangle1[0][1] == triangle2[0][1] and triangle1[0][2] == triangle2[0][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[0][0] == triangle2[1][0] and triangle1[0][1] == triangle2[1][1] and triangle1[0][2] == triangle2[1][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[0][0] == triangle2[2][0] and triangle1[0][1] == triangle2[2][1] and triangle1[0][2] == triangle2[2][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[1][0] == triangle2[0][0] and triangle1[1][1] == triangle2[0][1] and triangle1[1][2] == triangle2[0][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[1][0] == triangle2[1][0] and triangle1[1][1] == triangle2[1][1] and triangle1[1][2] == triangle2[1][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[1][0] == triangle2[2][0] and triangle1[1][1] == triangle2[2][1] and triangle1[1][2] == triangle2[2][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[2][0] == triangle2[0][0] and triangle1[2][1] == triangle2[0][1] and triangle1[2][2] == triangle2[0][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[2][0] == triangle2[1][0] and triangle1[2][1] == triangle2[1][1] and triangle1[2][2] == triangle2[1][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if (triangle1[2][0] == triangle2[2][0] and triangle1[2][1] == triangle2[2][1] and triangle1[2][2] == triangle2[2][2]):
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1
        if i == j:
            adjacency_matrix2[i, j]  = adjacency_matrix2[j, i] = 1


for i in range(num_triangles):
    for j in range(i + 1, num_triangles):
        if share_vertex(vertex_indices_triangles[i], vertex_indices_triangles[j]):
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

num_triangles = len(mesh.faces)
print("Number of triangles:", num_triangles)
#print(adjacency_matrix)
#np.savetxt("adjacency_matrix.txt", adjacency_matrix, fmt='%d')
np.savetxt("adjacency_matrix_diagonallar_1.txt", adjacency_matrix2, fmt='%d')