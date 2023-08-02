import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import shapely
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



path = '/home/oyku/yonsei/transmitter_localization/single_data_test.txt'
data = np.loadtxt(path)
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
#mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, 0, 1], [0, 0, 0])
#mesh = trimesh.intersections.slice_mesh_plane(mesh, [0, -1, 0], [0, 0, 0])
mesh = mesh.as_open3d
vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
print(triangles)
# Plot the mesh triangles
tri = Poly3DCollection(vertices[triangles], alpha=0.2, facecolors='yellow')

ax.add_collection(tri)
lines = Line3DCollection(vertices[triangles], colors='k', linewidths=0.5, alpha=1)
ax.add_collection(lines)

ax.scatter(x[x == x[0]], y[x == x[0]], z[x == x[0]], c='red', s=50)
ax.scatter(x[x == x[1]], y[x == x[1]], z[x == x[1]], c='red', s=50)
ax.scatter(x[x == x[2]], y[x == x[2]], z[x == x[2]], c='red', s=50)
ax.scatter(x, y, z, c='blue', label='Transmitter 0', s=10)

sphere_radius = 5  
ax.set_xlim(-sphere_radius, sphere_radius)
ax.set_ylim(-sphere_radius, sphere_radius)
ax.set_zlim(-sphere_radius, sphere_radius)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Add a legend
ax.legend()

# Show the plot

plt.show()
