import os
from math import sqrt
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import trimesh
#import open3d as o3d
#import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

def find_ones_indexes(matrix, index):
    ones_indexes = []
    for i, value in enumerate(matrix[index]):
        if value == 1:
            ones_indexes.append(i)
    return ones_indexes

current_directory = os.getcwd()
#sub_directory = './cnn_data_new_withoutneig'
gnn_data_dir = "/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/cnn_data_new_withoutneig"
list_of_node_features = os.listdir(gnn_data_dir)

if not os.path.exists("cnn_data_new_neighbors"):
    os.makedirs("cnn_data_new_neighbors")

adjacency_matrix = np.loadtxt("/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/adjacency_matrix.txt", delimiter=" ")

for i in range(len(list_of_node_features)):
    features = np.loadtxt(os.path.join(gnn_data_dir, list_of_node_features[i]), delimiter=',')
    if features.shape[0] != 320:
        continue
    
    match = re.search(r"_\d+(\.)", list_of_node_features[i])
    if match:
        exp_number = match.group(0)[1:-1]

    features_to_add = []

    for index, triangle_row in enumerate(features):
        features_row_to_add = []

        base_triangle = index
        

        neighbours = find_ones_indexes(adjacency_matrix, base_triangle)
        for neighbour in neighbours:
            for num in range(len(features[neighbour])):
                features_row_to_add.append(features[neighbour][num])
        features_to_add.append(list(features_row_to_add))

    #labels_of_base = features[:, -2:]
    #features_without_labels = features[:, :-2]
    cnn_features = np.hstack((features, np.asarray(features_to_add)))
    #cnn_data_with_labels = np.hstack((cnn_features, labels_of_base))
    #print(cnn_data_with_labels.shape)
    np.savetxt('cnn_data_new_neighbors/node_features_{}.txt'.format(exp_number), cnn_features, delimiter=", ", fmt='%1.5f')
