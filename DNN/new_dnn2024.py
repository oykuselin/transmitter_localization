import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD, Adadelta, Nadam

# Load the dataset
file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels_intersection.npy'
dataset = np.load(file_path, allow_pickle=True)

# Extracting data and labels
data = np.array([item[0] for item in dataset])
labels = np.array([item[1] for item in dataset])

# Flatten data from 2D (320, 148) to 1D (45440)
data_flattened = data.reshape((-1, 320 * 148))  # Reshape data

# Split the data into training and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data_flattened, labels, test_size=0.2, random_state=42)

# Define the DNN model
model = Sequential([
   # Dense(47360, activation='linear', input_shape=(320 * 148,)),

    #Dense(23680, activation='linear', input_shape=(320 * 148,)),

    #Dense(11840, activation='linear'),

    #Dense(5920, activation='linear'),

    #Dense(2960, activation='linear'),

    #Dense(1480, activation='linear'),

    #Dense(740, activation='linear'),

    #Dense(370, activation='linear'),

    #Dense(185, activation='linear'),

    #Dense(90, activation='linear'),

    #Dense(45, activation='linear'),

    Dense(512, activation='linear', input_shape=(320 * 148,)),

    Dense(256, activation='linear'),

    Dense(128, activation='linear'),
    Dense(64, activation='linear'),
    Dense(32, activation='linear'),
    Dense(16, activation='linear'),

    Dense(8, activation='linear'),
    
    Dense(8)  # Output layer with 6 units for the 6 coordinates
])

custom_optimizer = Adam(learning_rate=0.0001)  # Change the learning rate as needed
custom_optimizer2 = RMSprop(learning_rate=0.0001)
custom_optimizer3 = SGD(learning_rate=0.0001, momentum=0.9)  # Change the learning rate and momentum as needed
custom_optimizer4 = Adadelta()
custom_optimizer5 = Nadam()


# Compile the model
model.compile(optimizer=custom_optimizer,
              loss='mean_absolute_error',  # Assuming a regression problem; adjust as necessary
              metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
model.fit(data_train, labels_train, epochs=20, validation_data=(data_val, labels_val))

# Making predictions on the validation set
predictions = model.predict(data_val)

# Save the predictions to a NumPy array file
predictions_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/DNN/predictions_dnn.npy'
np.save(predictions_file_path, predictions)

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
combined_csv_file_path = '/Users/berkecaliskan/Documents/new_transmitter_localization/transmitter_localization/DNN/ground_truths_and_predictions_new_cnn2024_intersection.csv'  # Update with the actual path
ground_truths_and_predictions.to_csv(combined_csv_file_path, index=False)

print(f"Ground truths and predictions saved to {combined_csv_file_path}")

first_10_labels = labels_val[:10]
first_10_predictions = predictions[:10]

# Displaying the ground truth and predicted coordinates
for i in range(10):
    print(f"Ground Truth {i+1}: {first_10_labels[i]}")
    print(f"Prediction {i+1}: {first_10_predictions[i]}\n")