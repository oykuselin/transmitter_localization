import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split


# Load and preprocess data
data = np.loadtxt("/home/oyku/yonsei/transmitter_localization/cnn_data_final/node_features_0.txt", delimiter=",")
inputs = data[:, :-2]  # Extract inputs
labels = data[:, -2:]  # Extract labels

# Check the shape of the data before reshaping
print(inputs.shape)
print(labels.shape)

# Split data into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Reshape the data
train_inputs = train_inputs.reshape(-1, 148, 1)  # Each sample has 148 features
test_inputs = test_inputs.reshape(-1, 148, 1)

# Check the shape after reshaping
print(train_inputs.shape)
print(train_labels.shape)
print(test_inputs.shape)
print(test_labels.shape)

# Define the model
model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(148, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=40, padding='same'))  # Add padding here
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=2, activation='linear'))  # Output layer with 2 neurons for regression
model.summary()

# Compile the model
optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001)
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer, metrics=['mse'])

# Train the model
num_epochs = 10
batch_size = 32

model.fit(train_inputs, train_labels, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_mse = model.evaluate(test_inputs, test_labels)
print("Test MSE: %.4f" % test_mse)

# Make predictions on the test set
predictions = model.predict(test_inputs)

# Print the predictions and ground truth
print("Predictions:")
print(predictions)

print("Ground Truth:")
print(test_labels)