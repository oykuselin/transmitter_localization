import numpy as np
import trimesh

mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
vertex_indices_triangles = np.asarray(mesh.triangles)
num_triangles = len(vertex_indices_triangles)

def share_edge(triangle1, triangle2):
    common_vertices = 0
    for vertex1 in triangle1:
        for vertex2 in triangle2:
            if np.array_equal(vertex1, vertex2):
                common_vertices += 1
    return common_vertices == 2

adjacency_matrix = np.zeros((num_triangles, num_triangles), dtype=int)

for i in range(num_triangles):
    triangle1 = vertex_indices_triangles[i]
    for j in range(i + 1, num_triangles):
        triangle2 = vertex_indices_triangles[j]
        if share_edge(triangle1, triangle2):
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

print("Number of triangles:", num_triangles)
print(mesh.triangles)
np.savetxt("adjacency_matrix.txt", adjacency_matrix, fmt='%d')
