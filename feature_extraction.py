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

def first_time_slot(data_points):
    if (len(data_points) == 0):
        n_hit = 0
        t_max = t_min = t_avg = t_var = t_std = x_avg = y_avg = z_avg = -1
        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg
    else:
        first_slot_datapoints = []
        for data_point in data_points:
            if (data_point.time <= 166573):
                first_slot_datapoints.append(data_point)
        data_points = first_slot_datapoints
        if len(data_points) == 0:
            n_hit = 0
            t_max = t_min = t_avg = t_var = t_std = x_avg = y_avg = z_avg = -1
            return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg
        else:
            n_hit = len(data_points)
            if n_hit == 1:
                t_max = data_points[0].time*1e-6
                t_min = data_points[0].time*1e-6
                
                t_avg = data_points[0].time*1e-6
                t_var = 0
                t_std = 0

                x_avg = data_points[0].x
                y_avg = data_points[0].y
                z_avg = data_points[0].z
            else:
                t_max = 0
                t_min = np.inf
                all_times = []
                all_x_coordinates = []
                all_y_coordinates = []
                all_z_coordinates = []
                for data_point in data_points:
                    time = data_point.time*1e-6
                    if time > t_max:
                        t_max = time
                    if time < t_min:
                        t_min = time
                    all_times.append(time)
                    all_x_coordinates.append(data_point.x)
                    all_y_coordinates.append(data_point.y)
                    all_z_coordinates.append(data_point.z)
                
                t_avg = np.mean(all_times)
                t_var = np.var(all_times)
                t_std = np.std(all_times)

                x_avg = np.mean(all_x_coordinates)
                y_avg = np.mean(all_y_coordinates)
                z_avg = np.mean(all_z_coordinates)

        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg

def second_time_slot(data_points):
    if (len(data_points) == 0):
        n_hit = 0
        t_max = t_min = t_avg = t_var = t_std = x_avg = y_avg = z_avg = -1
        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg
    else:
        second_slot_datapoints = []
        for data_point in data_points:
            if (data_point.time <= 333147):
                second_slot_datapoints.append(data_point)
        data_points = second_slot_datapoints
        if(len(data_points) == 0):
            n_hit = 0
            t_max = t_min = t_avg = t_var = t_std = x_avg = y_avg = z_avg = -1
            return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg
        else:
            n_hit = len(data_points)
            if n_hit == 1:
                t_max = data_points[0].time*1e-6
                t_min = data_points[0].time*1e-6
                
                t_avg = data_points[0].time*1e-6
                t_var = 0
                t_std = 0

                x_avg = data_points[0].x
                y_avg = data_points[0].y
                z_avg = data_points[0].z
            else:
                t_max = 0
                t_min = np.inf
                all_times = []
                all_x_coordinates = []
                all_y_coordinates = []
                all_z_coordinates = []
                for data_point in data_points:
                    time = data_point.time*1e-6
                    if time > t_max:
                        t_max = time
                    if time < t_min:
                        t_min = time
                    all_times.append(time)
                    all_x_coordinates.append(data_point.x)
                    all_y_coordinates.append(data_point.y)
                    all_z_coordinates.append(data_point.z)
                
                t_avg = np.mean(all_times)
                t_var = np.var(all_times)
                t_std = np.std(all_times)

                x_avg = np.mean(all_x_coordinates)
                y_avg = np.mean(all_y_coordinates)
                z_avg = np.mean(all_z_coordinates)

        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg

def third_time_slot(data_points):
    if (len(data_points) == 0):
        n_hit = 0
        t_max = t_min = t_avg = t_var = t_std = x_avg = y_avg = z_avg = -1
        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg
    else:
        n_hit = len(data_points)
        if n_hit == 1:
            t_max = data_points[0].time*1e-6
            t_min = data_points[0].time*1e-6
            
            t_avg = data_points[0].time*1e-6
            t_var = 0
            t_std = 0

            x_avg = data_points[0].x
            y_avg = data_points[0].y
            z_avg = data_points[0].z
        else:
            t_max = 0
            t_min = np.inf
            all_times = []
            all_x_coordinates = []
            all_y_coordinates = []
            all_z_coordinates = []
            for data_point in data_points:
                time = data_point.time*1e-6
                if time > t_max:
                    t_max = time
                if time < t_min:
                    t_min = time
                all_times.append(time)
                all_x_coordinates.append(data_point.x)
                all_y_coordinates.append(data_point.y)
                all_z_coordinates.append(data_point.z)
            
            t_avg = np.mean(all_times)
            t_var = np.var(all_times)
            t_std = np.std(all_times)

            x_avg = np.mean(all_x_coordinates)
            y_avg = np.mean(all_y_coordinates)
            z_avg = np.mean(all_z_coordinates)

        return n_hit, t_max, t_min, t_avg, t_var, t_std, x_avg, y_avg, z_avg

def normalize_area(nodes):
    # print(nodes)
    nodes = np.array(nodes)
    area_mean = np.mean(nodes[:, 33], axis=0)
    area_std = np.std(nodes[:, 33], axis=0)
    for node in nodes:
        area = node[33]
        
        # Calculate the Z-score normalized values for the 36th column
        normalized_area = (area - area_mean) / (area_std*1.0)
        
        # Replace the 36th column with the normalized values
        node[33] = normalized_area
    return nodes

current_directory = os.getcwd()
sub_directory = 'all_simulation_data_2tx/results'
directory_path = os.path.join(current_directory, sub_directory)


# create the directory for storing data, if it is not created
if not os.path.exists("gnn_data_son"):
    os.makedirs("gnn_data_son")

results_list = os.listdir(directory_path)
for i in range(len(results_list)):
    nodes = []

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
    
    for triangle, data_points in triangle_dict.items():
        node = []

        v1 = triangle.v1
        v2 = triangle.v2
        v3 = triangle.v3
        mass_center = [(v1.x + v2.x + v3.x)/3, (v1.y + v2.y + v3.y)/3, (v1.z + v2.z + v3.z)/3]
        polar_r, polar_theta, polar_phi = cart2sph(mass_center[0], mass_center[1], mass_center[2])

        area = areatriangle3d(v1.x, v1.y, v1.z,
                                v2.x, v2.y, v2.z,
                                v3.x, v3.y, v3.z)

        p1 = np.array([v1.x, v1.y, v1.z])
        p2 = np.array([v2.x, v2.y, v2.z])
        p3 = np.array([v3.x, v3.y, v3.z])
        normal = np.cross(p2-p1, p3-p1)
        normal = normal / normal.sum()

        n_hit_1, t_max_1, t_min_1, t_avg_1, t_var_1, t_std_1, x_avg_1, y_avg_1, z_avg_1 = first_time_slot(data_points)
        n_hit_2, t_max_2, t_min_2, t_avg_2, t_var_2, t_std_2, x_avg_2, y_avg_2, z_avg_2 = second_time_slot(data_points)
        n_hit_3, t_max_3, t_min_3, t_avg_3, t_var_3, t_std_3, x_avg_3, y_avg_3, z_avg_3 = third_time_slot(data_points)

        node.append(mass_center[0])
        node.append(mass_center[1])
        node.append(mass_center[2])
        node.append(polar_r)
        node.append(polar_theta)
        node.append(polar_phi)
        node.append(n_hit_1) #gaussian normalization
        node.append(t_max_1) #t'ler de gaus norm olabilir
        node.append(t_min_1)
        node.append(t_avg_1)
        node.append(t_var_1)
        node.append(t_std_1)
        node.append(x_avg_1)
        node.append(y_avg_1)
        node.append(z_avg_1)
        node.append(n_hit_2) #gaussian normalization
        node.append(t_max_2) #t'ler de gaus norm olabilir
        node.append(t_min_2)
        node.append(t_avg_2)
        node.append(t_var_2)
        node.append(t_std_2)
        node.append(x_avg_2)
        node.append(y_avg_2)
        node.append(z_avg_2)
        node.append(n_hit_3) #gaussian normalization
        node.append(t_max_3) #t'ler de gaus norm olabilir
        node.append(t_min_3)
        node.append(t_avg_3)
        node.append(t_var_3)
        node.append(t_std_3)
        node.append(x_avg_3)
        node.append(y_avg_3)
        node.append(z_avg_3)
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
        node.append(second_to_all)
        
        nodes.append(node)
    
    nodes = normalize_area(nodes)
    np.savetxt('gnn_data_son/node_features_{}.txt'.format(exp_number), nodes, delimiter=", ", fmt='%1.5f')
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

# for i in range(100,110):
#     print(nodes[i])
