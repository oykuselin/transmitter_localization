import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
import os

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
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(simulation_data[:, :3])
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

def calculate_mae(results):
    """
    Calculate MAE between true and predicted means for transmitters and clusters.
    """
    mae_transmitter = (mean_absolute_error(results['mean_transmitter_0'], results['mean_cluster_0']) +
                       mean_absolute_error(results['mean_transmitter_1'], results['mean_cluster_1'])) / 2
    return mae_transmitter

directory_path = './transmitter_localization/all_simulation_data_2tx/results'
mae_results = []
analysis_results = []

counter = 0
limit = 5  # Change this to the number of simulations you want to process

for filename in os.listdir(directory_path):
    if filename.endswith(".txt") and counter < limit:
        file_path = os.path.join(directory_path, filename)
        simulation_data = np.loadtxt(file_path)
        analysis_result = analyze_molecules(simulation_data)
        analysis_results.append(analysis_result)
        # Calculate MAE
        mae = calculate_mae(analysis_result)  # Passing the same dictionary twice
        mae_results.append(mae)
        
        counter += 1  # Increment the counter after each processed file

# Calculate and print the mean of MAE results
mean_mae = np.mean(mae_results)
print(f"Mean MAE over {counter} simulations: {mean_mae}")
# Example output
for result in mae_results[:5]:  # Just showing the first few results for brevity
    print(result)
for result in analysis_results[:5]:
    print(result)
