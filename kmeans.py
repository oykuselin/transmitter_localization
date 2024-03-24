import numpy as np
from sklearn.cluster import KMeans

def analyze_molecules(simulation_data):
    """
    Analyzes the molecules in the simulation data by clustering and comparing means.

    :param simulation_data: A numpy array where each row represents a molecule with its x, y, z coordinates,
                            time info, and transmitter origin.
    :return: A dictionary containing the mean coordinates for each transmitter and cluster,
             and the clustered data.
    """
    # Separate data based on transmitter
    transmitter_0_data = simulation_data[simulation_data[:, 4] == 0, :3]
    transmitter_1_data = simulation_data[simulation_data[:, 4] == 1, :3]

    # Calculate means for each transmitter
    mean_transmitter_0 = np.mean(transmitter_0_data, axis=0)
    mean_transmitter_1 = np.mean(transmitter_1_data, axis=0)

    # Perform k-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(simulation_data[:, :3])
    clusters = kmeans.labels_

    # Calculate means for each cluster
    cluster_0_data = simulation_data[clusters == 0, :3]
    cluster_1_data = simulation_data[clusters == 1, :3]
    mean_cluster_0 = np.mean(cluster_0_data, axis=0)
    mean_cluster_1 = np.mean(cluster_1_data, axis=0)

    # Prepare the results
    results = {
        'mean_transmitter_0': mean_transmitter_0,
        'mean_transmitter_1': mean_transmitter_1,
        'mean_cluster_0': mean_cluster_0,
        'mean_cluster_1': mean_cluster_1
    }

    return results

# Path to your simulation data file
file_path = '/home/oyku/yonsei/new_transmitter_localization/transmitter_localization/all_simulation_data_2tx/results/result_2_3.txt'

# Read the simulation data from the file
simulation_data = np.loadtxt(file_path)

# Analyzing the simulation data
analysis_results = analyze_molecules(simulation_data)
print(analysis_results)
