import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = np.load("CNN/Data.npy")
data = data.reshape(1000, 320, 151)

inputs = data[:, :, :-3] # Reshape inputs to (num_samples, channels, height, width)
reshaped_inputs = np.transpose(inputs, (0, 2, 1))
labels = data[:, :, -3:] # Extract labels
reshaped_labels = np.transpose(labels, (0, 2, 1))


# Split data into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(reshaped_inputs, reshaped_labels, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
train_inputs = torch.Tensor(train_inputs)
train_labels = torch.Tensor(train_labels)
test_inputs = torch.Tensor(test_inputs)
test_labels = torch.Tensor(test_labels)

# Create DataLoader for train and test sets
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the CNN model
class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
    
        self.conv_layers = nn.Sequential(
            nn.Conv1d(148, 256, kernel_size=3, padding=1),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
    
        self.linear_layers = nn.Sequential(
            nn.Linear(163840, 320*3),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        print(x.shape)
        x = x.view(x.size(0), -1)         # Flatten the features
        x = self.linear_layers(x)
        x = x.view(x.size(0), 3, 320)  # Reshape to the desired output shape
        return x

# Create an instance of the model
model = RegressionCNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.000001)
print(optimizer)
# Training loop
num_epochs = 30

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    
    for batch in train_loader:
        
        inputs, labels = batch 
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)  # Use inputs as targets for autoencoder
        optimizer.zero_grad()
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        total_loss += curr_loss

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

# Validation
model.eval()  # Set the model to evaluation mode
total_validation_loss = 0.0

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_validation_loss += loss.item()

average_validation_loss = total_validation_loss / len(test_loader)
print(f"Validation Loss: {average_validation_loss}")