import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

directory_path = '/Users/senamumcu/Desktop/2020_MultiTxLocalization/3tx_1/result1' 

for file_name in os.listdir(directory_path):
    counter =0
    if file_name.endswith('.txt') and counter == 0:  
        file_path = os.path.join(directory_path, file_name)
        
        data = np.loadtxt(file_path)
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        time = data[:, 3]
        transmitter = data[:, 4]
        
        # Define receiver coordinates
        receiver_x = 0  
        receiver_y = 0  
        receiver_z = 0  

        # 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5)
        #print(mesh.vertices)
        mesh = mesh.as_open3d

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        # Plot the mesh triangles
        tri = Poly3DCollection(vertices[triangles], alpha=0.6)
        ax.add_collection(tri)
        lines = Line3DCollection(vertices[triangles], colors='k', linewidths=0.2, alpha=0.5)
        ax.add_collection(lines)
        # molecules hitting the receiver
        ax.scatter(x[transmitter == 0], y[transmitter == 0], z[transmitter == 0], c='blue', label='Transmitter 0')
        ax.scatter(x[transmitter == 1], y[transmitter == 1], z[transmitter == 1], c='red', label='Transmitter 1')
        
        # Plot the receiver
        ax.scatter(receiver_x, receiver_y, receiver_z, c='black', s=100, label='Receiver')
        
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
        counter = 1