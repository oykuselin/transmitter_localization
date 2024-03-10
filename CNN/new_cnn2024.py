import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels_intersection.npy'  # Update with the actual path to your .npy file
dataset = np.load(file_path, allow_pickle=True)

# Extracting data and labels
data = np.array([item[0] for item in dataset])
labels = np.array([item[1] for item in dataset])

# Reshape data to include a channel dimension
data = data.reshape((-1, 320, 148, 1))

# Split the data into training and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=(320, 148, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='linear'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='linear'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='linear'),
    Dense(128, activation='linear'),
    Dense(64, activation='linear'),
    Dense(32, activation='linear'),
    Dense(8)  # Assuming the task is to predict 6 coordinates
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Assuming a regression problem; adjust as necessary
              metrics=['accuracy'])

# Train the model
model.fit(data_train, labels_train, epochs=10, validation_data=(data_val, labels_val))


# Making predictions on the validation set (or replace with test set or new data as needed)
predictions = model.predict(data_val)

# Save the predictions to a NumPy array file
predictions_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/CNN/predictions_new_cnn2024_intersection.npy'  # Update with the actual path
np.save(predictions_file_path, predictions)

# Optionally, to save as a CSV file:
import pandas as pd

# Convert predictions to a DataFrame
predictions_df = pd.DataFrame(predictions, columns=['Coord1', 'Coord2', 'Coord3', 'Dist1', 'Coord4', 'Coord5', 'Coord6', 'Dist2'])

# Save the DataFrame to a CSV file
csv_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/CNN/predictions_new_cnn2024_intersection.csv'  # Update with the actual path
predictions_df.to_csv(csv_file_path, index=False)

# Create a DataFrame from the ground truths and predictions
ground_truths_and_predictions = pd.DataFrame({
    'GroundTruth_Coord1': labels_val[:, 0],
    'GroundTruth_Coord2': labels_val[:, 1],
    'GroundTruth_Coord3': labels_val[:, 2],
    'GroundTruth_Dist1': labels_val[:, 3],
    'GroundTruth_Coord4': labels_val[:, 4],
    'GroundTruth_Coord5': labels_val[:, 5],
    'GroundTruth_Coord6': labels_val[:, 6],
    'GroundTruth_Dist2': labels_val[:, 7],
    'Predicted_Coord1': predictions[:, 0],
    'Predicted_Coord2': predictions[:, 1],
    'Predicted_Coord3': predictions[:, 2],
    'Predicted_Dist1': predictions[:, 3],
    'Predicted_Coord4': predictions[:, 4],
    'Predicted_Coord5': predictions[:, 5],
    'Predicted_Coord6': predictions[:, 6],
    'Predicted_Dist2': predictions[:, 7],
})

# Save the DataFrame to a CSV file
combined_csv_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/CNN/ground_truths_and_predictions_new_cnn2024_intersection.csv'  # Update with the actual path
ground_truths_and_predictions.to_csv(combined_csv_file_path, index=False)

print(f"Ground truths and predictions saved to {combined_csv_file_path}")

first_10_labels = labels_val[:10]
first_10_predictions = predictions[:10]

# Displaying the ground truth and predicted coordinates
for i in range(10):
    print(f"Ground Truth {i+1}: {first_10_labels[i]}")
    print(f"Prediction {i+1}: {first_10_predictions[i]}\n")