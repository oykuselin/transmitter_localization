import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = '/home/oyku/yonsei/new_transmitter_localization/transmitter_localization/cnn_data_new/data_with_labels.npy'  # Update with the actual path to your .npy file
dataset = np.load(file_path, allow_pickle=True)

# Extracting data and labels
data = np.array([item[0] for item in dataset])
labels = np.array([item[1] for item in dataset])

# Reshape data to include a channel dimension
data = data.reshape((-1, 320, 142, 1))

# Split the data into training and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(320, 142, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6)  # Assuming the task is to predict 6 coordinates
])

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Assuming a regression problem; adjust as necessary
              metrics=['accuracy'])

# Train the model
model.fit(data_train, labels_train, epochs=10, validation_data=(data_val, labels_val))
