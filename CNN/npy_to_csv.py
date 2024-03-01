import pandas as pd
import numpy as np

file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/CNN/predictions_new_cnn2024.npy'  # Update with the actual path to your .npy file
predictions = np.load(file_path, allow_pickle=True)
# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Coord1', 'Coord2', 'Coord3', 'Coord4', 'Coord5', 'Coord6'])

# Save the DataFrame to a CSV file
csv_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/CNN/predictions.csv'  # Update with the actual path
predictions_df.to_csv(csv_file_path, index=False)