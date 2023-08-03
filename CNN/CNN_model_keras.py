import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = np.loadtxt("/Users/senamumcu/Desktop/2020_MultiTxLocalization/cnn_data_final/node_features_0.txt", delimiter=",")
inputs = data[:, :-2]  # Extract inputs
labels = data[:, -2:]  # Extract labels

# Split data into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Reshape inputs for 2D CNN
train_inputs = train_inputs.reshape(-1, 320, 148, 1)
test_inputs = test_inputs.reshape(-1, 320, 148, 1)

# Convert the data to TensorFlow tensors
train_inputs = tf.convert_to_tensor(train_inputs, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
test_inputs = tf.convert_to_tensor(test_inputs, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

# Define the CNN model using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(320, 148, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2)  # Output layer with 2 neurons for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
num_epochs = 10
batch_size = 32

model.fit(train_inputs, train_labels, epochs=num_epochs, batch_size=batch_size, verbose=1)

# Evaluate the model on the test set
test_loss = model.evaluate(test_inputs, test_labels, batch_size=batch_size, verbose=0)
print(f"Test Loss: {test_loss}")
