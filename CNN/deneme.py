import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load data
data = np.load("CNN/Data_3label.npy")
inputs = data[:, :, :, :-3]
labels = data[:, :, :, -3:]

X = np.array(inputs)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data
X_train = X_train.reshape(X_train.shape[0], 320, 1, 148)
X_test = X_test.reshape(X_test.shape[0], 320, 1, 148)
y_train = y_train.reshape(y_train.shape[0], 1, 320, 3)
y_test = y_test.reshape(y_test.shape[0], 1, 320, 3)

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
            tf.keras.layers.Dense(320 * 3, activation='softmax')
        ])
    
    def call(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.linear_layers(x)
        x = tf.reshape(x, (-1, 320, 3))
        return x

# Create an instance of the model
model = RegressionCNN()

# Define a custom loss function to match your label shape
def custom_loss(y_true, y_pred):
    return tf.keras.losses.categorical_crossentropy(tf.reshape(y_true, (-1, 320, 3)), y_pred)

# Compile the model using the custom loss function
model.compile(loss=custom_loss, optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Make predictions
y_pred = model.predict(X_test)

# Print ground truth and prediction results for the first sample
sample_index = 0
ground_truth = y_test[sample_index, 0, :]
prediction = y_pred[sample_index, :, :]

print("Ground Truth:")
print(ground_truth)

print("Prediction:")
print(prediction)
