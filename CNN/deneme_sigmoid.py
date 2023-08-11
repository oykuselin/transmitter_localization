import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data = np.load("CNN/Data_3label.npy")
inputs = data[:, :, :, :-3]
labels = data[:, :, :, -3:]

X = np.array(inputs)
y = np.array(labels)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 320, 1, 148)
X_test = X_test.reshape(X_test.shape[0], 320, 1, 148)
y_train = y_train.reshape(y_train.shape[0], 1, 320, 3)  # Correct the shape here
y_test = y_test.reshape(y_test.shape[0], 1, 320, 3)      # Correct the shape here

class SumToOneConstraint(tf.keras.constraints.Constraint):
    def __init__(self, axis=3):
        self.axis = axis

    def __call__(self, w):
        return w / tf.reduce_sum(w, axis=self.axis, keepdims=True)

class RegressionCNN(tf.keras.Model):
    def __init__(self):
        super(RegressionCNN, self).__init__()
        
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 1), padding='same', input_shape=(320, 1, 148)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
            
            tf.keras.layers.Conv2D(128, kernel_size=(3, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(256, kernel_size=(3, 1), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 1))
        ])
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.linear_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(320, activation='relu'),  # Add sigmoid activation here
            tf.keras.layers.Dense(320 * 3, activation='linear', kernel_constraint=SumToOneConstraint())  # No activation for linear output
        ])
    
    def call(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        x = tf.reshape(x, (-1, 1, 320, 3))
        return x


# Create an instance of the model
model = RegressionCNN()

# Compile the model
model.compile(loss='mse', optimizer='adam')


# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Make predictions
y_pred = model.predict(X_test)

# Print ground truth and prediction results for the first sample
sample_index = 0 
ground_truth = y_test[sample_index, 0, :]
prediction = y_pred[sample_index, 0, :]

print("Ground Truth:")
print(ground_truth)

print("Prediction:")
print(prediction)
