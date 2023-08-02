import os
from math import sqrt
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
import torch
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


current_directory = os.getcwd()
sub_directory = 'test/2tx/results'
directory_path = os.path.join(current_directory, sub_directory)


# create the directory for storing data, if it is not created
if not os.path.exists("gnn_data"):
    os.makedirs("gnn_data")

results_list = os.listdir(directory_path)
for i in range(len(results_list)):
    nodes = []
    count = 0
    file_path = os.path.join(directory_path, results_list[i])
    
    match = re.search(r"_\d+(\.)", results_list[i])
    if match:
        exp_number = match.group(0)[1:-1]

    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    time = data[:, 3]
    transmitter = data[:, 4]
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
    """vertex_indices_triangles = np.asarray(mesh.triangles)
    #print(vertex_indices_triangles)

    num_triangles = len(vertex_indices_triangles)
    adjacency_matrix = np.zeros((num_triangles, num_triangles), dtype=int)

    # Helper function to check if two triangles share a vertex
    def share_vertex(triangle1, triangle2):
        for vertex in triangle1:
            if vertex in triangle2:
                return True
        return False

    for i in range(num_triangles):
        for j in range(i + 1, num_triangles):
            if share_vertex(vertex_indices_triangles[i], vertex_indices_triangles[j]):
                adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

    num_triangles = len(mesh.faces)
    adj_t = torch.tensor(adjacency_matrix)
    edge_index = adj_t.nonzero().t().contiguous()
    edge_index_str = str(edge_index.tolist())"""
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
    for triangle in triangles:
        triangle_dict[triangle] = []
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
        triangle_dict[closest_triangle].append(data_point)
    
    i = 0
    for triangle, data_points in triangle_dict.items():
        node = []
        first_to_all_ratio = 0
        if i == 0:
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
        f_transmitter_points = 0

        for data_point in data_points:
            time = data_point.time
            if time > t_max:
                t_max = time
            if time < t_min:
                t_min = time
            all_times.append(time)
            if data_point.t_id == 0:
                f_transmitter_points += 1
        #first_to_all_ratio = f_transmitter_points/(len(data_points)*1.0)
        
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

        if t_min == np.inf:
            t_min = 0
            t_avg = 0
            t_var = 0
            t_std = 0
        node.append(mass_center[0])
        node.append(mass_center[1])
        node.append(mass_center[2])
        node.append(polar_r)
        node.append(polar_theta)
        node.append(polar_phi)
        node.append(n_molecules) #gaussian normalization
        node.append(t_max) #t'ler de gaus norm olabilir
        node.append(t_min)
        node.append(t_avg)
        node.append(t_var)
        node.append(t_std)
        node.append(area) #gaussian normalization(kesin)
        node.append(normal[0])
        node.append(normal[1])
        node.append(normal[2])

        count_ones = sum(1 for data in data_points if data.t_id == 1)
        count_zeros = sum(1 for data in data_points if data.t_id == 0)
        if count_ones == 0 and count_zeros == 0:
            first_to_all = 0
            second_to_all = 0
        else:
            first_to_all = count_zeros / (count_zeros + count_ones)
            second_to_all = count_ones / (count_zeros + count_ones)
        node.append(first_to_all)
        #node.append(second_to_all)
        
        nodes.append(node)
        if len(data_points) == 0:
            count += 1
    
    np.savetxt('gnn_data_one_label/node_features_{}.txt'.format(exp_number), nodes, delimiter=", ", fmt='%1.5f')
    #with open('gnn_data/edge_index_{}.txt'.format(exp_number), 'w+') as file2:
        #file2.write(edge_index_str)
# k = 0
# print(len(triangle_dict.keys()))
# for triangle, data_points in triangle_dict.items():
#     if k < 10:
#         v1 = triangle.v1
#         v2 = triangle.v2
#         v3 = triangle.v3
#         print( "Triangle {}:".format(k))
#         print([v1.x, v1.y, v1.z], "\n" , [v3.x, v2.y, v2.z],  "\n", [v3.x, v3.y, v3.z],  "\n")
#         k += 1

for i in range(100,110):
    print(nodes[i])
print(count)