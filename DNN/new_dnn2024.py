import numpy as np
import tensorflow as tf
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam, RMSprop, SGD, Adadelta, Nadam

# Load the dataset
file_path = './cnn_data_new/data_with_labels_intersection.npy'
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

    Dense(1480, activation='linear', input_shape=(320 * 148,)),

    Dense(740, activation='linear'),

    #Dense(370, activation='linear'),

    #Dense(185, activation='linear'),

    #Dense(90, activation='linear'),

    #Dense(45, activation='linear'),

    #Dense(512, activation='linear', input_shape=(320 * 148,)),

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

# Angle based cost function
def custom_angle_and_distance_loss(y_true, y_pred):
    # Extracting coordinates
    x1_true, y1_true, z1_true = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    x2_true, y2_true, z2_true = y_true[:, 4], y_true[:, 5], y_true[:, 6]
    
    x1_pred, y1_pred, z1_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    x2_pred, y2_pred, z2_pred = y_pred[:, 4], y_pred[:, 5], y_pred[:, 6]

    add_val1_true, add_val2_true = y_true[:, 3], y_true[:, 7]
    add_val1_pred, add_val2_pred = y_pred[:, 3], y_pred[:, 7]
    
    # Calculate vectors from origin
    v1_true = tf.stack([x1_true, y1_true, z1_true], axis=1)
    v2_true = tf.stack([x2_true, y2_true, z2_true], axis=1)
    
    v1_pred = tf.stack([x1_pred, y1_pred, z1_pred], axis=1)
    v2_pred = tf.stack([x2_pred, y2_pred, z2_pred], axis=1)
    
    # Calculate distances from origin
    distance_v1_true = tf.sqrt(tf.reduce_sum(tf.square(v1_true), axis=1))
    distance_v2_true = tf.sqrt(tf.reduce_sum(tf.square(v2_true), axis=1))
    distance_v1_pred = tf.sqrt(tf.reduce_sum(tf.square(v1_pred), axis=1))
    distance_v2_pred = tf.sqrt(tf.reduce_sum(tf.square(v2_pred), axis=1))
    
    # Calculate distance differences
    distance_diff1 = tf.abs(distance_v1_true - distance_v1_pred)
    distance_diff2 = tf.abs(distance_v2_true - distance_v2_pred)
    
    # Normalize vectors for angle calculation
    v1_true_norm = tf.nn.l2_normalize(v1_true, axis=1)
    v2_true_norm = tf.nn.l2_normalize(v2_true, axis=1)
    v1_pred_norm = tf.nn.l2_normalize(v1_pred, axis=1)
    v2_pred_norm = tf.nn.l2_normalize(v2_pred, axis=1)
    
    # Calculate dot product for angles
    dot_product1 = tf.reduce_sum(tf.multiply(v1_true_norm, v1_pred_norm), axis=1)
    dot_product2 = tf.reduce_sum(tf.multiply(v2_true_norm, v2_pred_norm), axis=1)
    
    # Calculate angles in radians
    angle1 = tf.acos(tf.clip_by_value(dot_product1, -1.0, 1.0))
    angle2 = tf.acos(tf.clip_by_value(dot_product2, -1.0, 1.0))
    
    # Sum of angles as angular error
    total_angle_error = tf.reduce_mean(angle1 + angle2)
    
    # Mean of distance differences as distance error
    total_distance_error = tf.reduce_mean(distance_diff1 + distance_diff2)
    
    add_val_diff1 = tf.abs(add_val1_true - add_val1_pred)
    add_val_diff2 = tf.abs(add_val2_true - add_val2_pred)

    total_add_val_error = tf.reduce_mean(add_val_diff1 + add_val_diff2)

    # Total error as a combination of angle and distance errors
    total_error = total_angle_error + total_distance_error + total_add_val_error
    
    return total_error


# Compile the model
model.compile(optimizer=custom_optimizer,
              loss=custom_angle_and_distance_loss,  # Assuming a regression problem; adjust as necessary
              metrics=['mean_absolute_error', 'mean_squared_error'])

# Train the model
model.fit(data_train, labels_train, epochs=50, validation_data=(data_val, labels_val))

# Making predictions on the validation set
predictions = model.predict(data_val)

# Save the predictions to a NumPy array file
predictions_file_path = './DNN/predictions_dnn.npy'
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
combined_csv_file_path = './DNN/ground_truths_and_predictions_new_cnn2024_intersection.csv'  # Update with the actual path
ground_truths_and_predictions.to_csv(combined_csv_file_path, index=False)

print(f"Ground truths and predictions saved to {combined_csv_file_path}")

first_10_labels = labels_val[:10]
first_10_predictions = predictions[:10]

# Displaying the ground truth and predicted coordinates
for i in range(10):
    print(f"Ground Truth {i+1}: {first_10_labels[i]}")
    print(f"Prediction {i+1}: {first_10_predictions[i]}\n")