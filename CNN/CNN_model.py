import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Load and preprocess data
data = np.loadtxt("/Users/senamumcu/Desktop/2020_MultiTxLocalization/cnn_data_final/node_features_0.txt", delimiter=",")
inputs = data[:, :-2]  # Reshape inputs to (num_samples, channels, height, width)
labels = data[:, -2:] # Extract labels

# Split data into train and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
train_len = len(train_inputs)
test_len = len(test_inputs)

train_inputs = train_inputs.reshape(-1, train_len, 148)
test_inputs = test_inputs.reshape(-1, test_len, 148)
train_labels = train_labels.reshape(-1, train_len, 2)
test_labels = test_labels.reshape(-1, test_len, 2)


# Convert the data to PyTorch tensors
train_inputs = torch.Tensor(train_inputs)
train_labels = torch.Tensor(train_labels)
test_inputs = torch.Tensor(test_inputs)
test_labels = torch.Tensor(test_labels)
print(train_inputs.size(), train_labels.size(), test_inputs.size(), test_labels.size())

# Create DataLoader for train and test sets
train_dataset = TensorDataset(train_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = TensorDataset(test_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the CNN model
class RegressionCNN(nn.Module):
    def __init__(self):
        super(RegressionCNN, self).__init__()
    
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
        self.fc_layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 40 * 18, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 2)  # Output layer with 2 neurons for regression
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Create an instance of the model
model = RegressionCNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0
    
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()  # Zero the gradients
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)  # Compare outputs to ground truth labels
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
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