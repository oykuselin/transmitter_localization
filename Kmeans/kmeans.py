import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans

def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
# loaded_data = np.load("all_simulations.npy")

all_sim_data = []

for i in range(1):
    path = "/Users/senamumcu/Desktop/2020_MultiTxLocalization/kerem's data{}/results".format(i)
    for j in range(3): #len(os.listdir(path))):
        file_path = os.path.join(path, "result_2_{}.txt".format(j + i * 5000))
        # print(file_path)
        simulation = np.loadtxt(file_path, delimiter=" ", dtype=float)
        all_sim_data.append(simulation)

# print(len(all_sim_data))
# np.save("all_simulations.npy", all_sim_data)

loaded_data = all_sim_data

for i in range(3):

    data = np.array(loaded_data[i])
    X = data[:, 0:3]
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    true_cluster1 = X[data[:, 4] == 0]
    true_cluster2 = X[data[:, 4] == 1]

    kmeans_cluster1 = X[labels == 0]
    kmeans_cluster2 = X[labels == 1]

    # Create 3D scatter plots for the true clusters and K-means clusters
    fig = plt.figure()

    # 3D scatter plot for the true clusters
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(true_cluster1[:, 0], true_cluster1[:, 1], true_cluster1[:, 2], c='blue', label='True Cluster 1')
    ax1.scatter(true_cluster2[:, 0], true_cluster2[:, 1], true_cluster2[:, 2], c='red', label='True Cluster 2')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_zlabel('Z-axis')
    ax1.set_title('True Clusters')
    ax1.legend()
    ax1.grid(True)

    # 3D scatter plot for the K-means clusters
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(kmeans_cluster1[:, 0], kmeans_cluster1[:, 1], kmeans_cluster1[:, 2], c='blue', label='K-Means Cluster 1')
    ax2.scatter(kmeans_cluster2[:, 0], kmeans_cluster2[:, 1], kmeans_cluster2[:, 2], c='red', label='K-Means Cluster 2')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_zlabel('Z-axis')
    ax2.set_title('K-Means Clusters')
    ax2.legend()
    ax2.grid(True)

    plt.show()