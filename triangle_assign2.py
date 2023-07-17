import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


class DataPoint:
    def __init__(self, x, y, z, time, t_id):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.time = int(time)
        self.t_id = int(t_id)


class Vertex:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class Triangle:
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

directory_path = '/Users/berkecaliskan/Documents/MultiTX Localization/public-archivedwl-242/test/2tx/results' 

for file_name in os.listdir(directory_path):
    if file_name.endswith('result_2_0.txt'):  
        file_path = os.path.join(directory_path, file_name)
        
        data = np.loadtxt(file_path)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        time = data[:, 3]
        transmitter = data[:, 4]
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
        mesh = mesh.as_open3d

        vertices_array =np.asarray(mesh.vertices)
        vertices = []
        for vertex in vertices_array:
            x1, y1, z1 = vertex
            vertice = Vertex(x1, y1, z1)
            vertices.append(vertice)
        
        vertex_indexes_triangle = np.asarray(mesh.triangles)
        triangles = []
        for vertex_indexes in vertex_indexes_triangle:
            v1_idx, v2_idx, v3_idx = vertex_indexes
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            v3 = vertices[v3_idx]
            triangle = Triangle(v1, v2, v3)
            triangles.append(triangle)

        data_points = []
        array_size = x.shape[0]
        for i in range(array_size):
            data_point = DataPoint(x[i], y[i], z[i], time[i], transmitter[i])
            data_points.append(data_point)
        
        triangle_dict = {}
        for data_point in data_points:
            closest_triangle = None
            closest_distance = np.inf
            
            for triangle in triangles:
                # Calculate distance between data point and vertices of the triangle
                G_x = (triangle.v1.x + triangle.v2.x + triangle.v3.x)/3.0
                G_y = (triangle.v1.y + triangle.v2.y + triangle.v3.y)/3.0
                G_z = (triangle.v1.z + triangle.v2.z + triangle.v3.z)/3.0

                # Update closest triangle if distance is smaller
                dist = np.linalg.norm([data_point.x - G_x, data_point.y - G_y, data_point.z - G_z])
                if dist < closest_distance:
                    closest_distance = dist
                    closest_triangle = triangle
            
            # Assign the closest triangle as the value for the data point key
            if closest_triangle in triangle_dict:
                triangle_dict[closest_triangle].append(data_point)
            else:
                triangle_dict[closest_triangle] = [data_point]
        for triangle, data_points in triangle_dict.items():
            print(f"Triangle: Vertices -")
            print(f"v1: ({triangle.v1.x}, {triangle.v1.y}, {triangle.v1.z})")
            print(f"v2: ({triangle.v2.x}, {triangle.v2.y}, {triangle.v2.z})")
            print(f"v3: ({triangle.v3.x}, {triangle.v3.y}, {triangle.v3.z})")
            print("Associated Data Points:")
            for data_point in data_points:
                print(f"Data Point: ({data_point.x}, {data_point.y}, {data_point.z})")
            print()
