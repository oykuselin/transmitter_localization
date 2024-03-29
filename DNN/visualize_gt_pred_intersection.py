import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Load the CSV file
file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/DNN/ground_truths_and_predictions_new_cnn2024_intersection.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Take only the first row
row = data.iloc[102]

# Extract ground truth and predicted intersection points
ground_truths = np.array([row[['GroundTruth_Coord1', 'GroundTruth_Coord2', 'GroundTruth_Coord3']],
                          row[['GroundTruth_Coord4', 'GroundTruth_Coord5', 'GroundTruth_Coord6']]])
predictions = np.array([row[['Predicted_Coord1', 'Predicted_Coord2', 'Predicted_Coord3']],
                        row[['Predicted_Coord4', 'Predicted_Coord5', 'Predicted_Coord6']]])

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot ground truth intersection points
ax.scatter(ground_truths[:, 0], ground_truths[:, 1], ground_truths[:, 2], c='blue', s=50, label='Ground Truth Tx1 & Tx2')

# Plot predicted intersection points
ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2], c='red', s=50, label='Predicted Tx1 & Tx2')

# Setting labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('First Entry: Ground Truth vs Predicted Intersection Points')
ax.legend()

plt.show()