import os
from math import sqrt
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

def heron(a,b,c):  
    s = (a + b + c) / 2   
    area = (s*(s-a) * (s-b)*(s-c)) ** 0.5        
    return area

def distance3d(x1,y1,z1,x2,y2,z2):    
    a=(x1-x2)**2+(y1-y2)**2 + (z1-z2)**2
    d= a ** 0.5  
    return d  

def areatriangle3d(x1,y1,z1,x2,y2,z2,x3,y3,z3):  
    a=distance3d(x1,y1,z1,x2,y2,z2)  
    b=distance3d(x2,y2,z2,x3,y3,z3)  
    c=distance3d(x3,y3,z3,x1,y1,z1)  
    area = heron(a,b,c)
    return area

def cart2sph(x, y, z):
   xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)
   x_2 = x**2
   y_2 = y**2
   z_2 = z**2
   r = np.sqrt(x_2 + y_2 + z_2) # r = sqrt(x² + y² + z²)
   theta = np.arctan2(y, x) 
   phi = np.arctan2(xy, z) 
   return r, theta, phi

directory_path = '/Users/berkecaliskan/Documents/MultiTX Localization/public-archivedwl-242/test/2tx/results' 

nodes = []

for file_name in os.listdir(directory_path):
    if file_name.endswith('result_2_0.txt'): 
        file_path = os.path.join(directory_path, file_name)
        
        data = np.loadtxt(file_path)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        time = data[:, 3]
        transmitter = data[:, 4]
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=5)
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
                vertex1 = triangle.v1
                vertex2 = triangle.v2
                vertex3 = triangle.v3
                coordinates1 = np.array([vertex1.x, vertex1.y, vertex1.z])
                coordinates2 = np.array([vertex2.x, vertex2.y, vertex2.z])
                coordinates3 = np.array([vertex3.x, vertex3.y, vertex3.z])
                point_coordinates = np.array([data_point.x, data_point.y, data_point.z])
                # Update closest triangle if distance is smaller
                dist1 = np.linalg.norm(point_coordinates - coordinates1)
                dist2 = np.linalg.norm(point_coordinates - coordinates2)
                dist3 = np.linalg.norm(point_coordinates - coordinates3)
                total_dist = dist1 + dist2 + dist3
                if total_dist < closest_distance:
                    closest_distance = total_dist
                    closest_triangle = triangle
            
            # Assign the closest triangle as the value for the data point key
            if closest_triangle in triangle_dict:
                triangle_dict[closest_triangle].append(data_point)
            else:
                triangle_dict[closest_triangle] = [data_point]
        i = 0
        for triangle, data_points in triangle_dict.items():
            node = []
            if i == 0:
                print(triangle_dict[triangle])
                i += 1
            v1 = triangle.v1
            v2 = triangle.v2
            v3 = triangle.v3
            mass_center = [(v1.x + v2.x + v3.x)/3, (v1.y + v2.y + v3.y)/3, (v1.z + v2.z + v3.z)/3]
            polar_r, polar_theta, polar_phi = cart2sph(mass_center[0], mass_center[1], mass_center[2])
            n_molecules = len(data_points)
            t_max = 0
            t_min = np.inf
            all_times = []
            for data_point in data_points:
                time = data_point.time
                if time > t_max:
                    t_max = time
                if time < t_min:
                    t_min = time
                all_times.append(time)
            t_avg = np.mean(all_times)
            t_var = np.var(all_times)
            t_std = np.std(all_times)

            area = areatriangle3d(v1.x, v1.y, v1.z,
                                  v2.x, v2.y, v2.z,
                                  v3.x, v3.y, v3.z)

            p1 = np.array([v1.x, v1.y, v1.z])
            p2 = np.array([v2.x, v2.y, v2.z])
            p3 = np.array([v3.x, v3.y, v3.z])
            normal = np.cross(p2-p1, p3-p1)
            normal = normal / normal.sum()

            node.append(mass_center)
            node.append([polar_r, polar_theta, polar_phi])
            node.append(n_molecules)
            node.append(t_max)
            node.append(t_min)
            node.append(t_avg)
            node.append(t_var)
            node.append(t_std)
            node.append(area)
            node.append(normal.tolist())

            nodes.append(node)
#print(nodes)
k = 0
for triangle, data_points in triangle_dict.items():
    if k < 10:
        v1 = triangle.v1
        v2 = triangle.v2
        v3 = triangle.v3
        print( "Triangle {}:".format(k))
        print([v1.x, v1.y, v1.z], "\n" , [v3.x, v2.y, v2.z],  "\n", [v3.x, v3.y, v3.z],  "\n")
        k += 1
for i in range(10):
    print(nodes[i])


