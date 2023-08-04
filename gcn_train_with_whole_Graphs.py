import os
import torch
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy import stats
from scipy.special import boxcox, boxcox1p
from sklearn.preprocessing import StandardScaler
import torch.nn.init as init
from sklearn.metrics import confusion_matrix
import random
from torch.utils.data import DataLoader


os.environ['TORCH'] = torch.__version__
os.environ['PYTHONWARNINGS'] = "ignore"

#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#print(dataset[0].edge_index)


class CustomDataset:
    def __init__(self, data_folder, num_features):
        self.num_features = num_features
        self.data_entries = []
        list_files = os.listdir(data_folder)
        random.shuffle(list_files)
        num_graphs = len(list_files)

        # Use 80% of data_entries for training, 10% for validation, and 10% for testing
        train_indices = np.random.choice(num_graphs, int(0.8 * num_graphs), replace=False)
        val_test_indices = np.setdiff1d(np.arange(num_graphs), train_indices)
        val_indices = np.random.choice(val_test_indices, int(0.5 * len(val_test_indices)), replace=False)
        test_indices = np.setdiff1d(val_test_indices, val_indices)

        for i, filename in enumerate(list_files):
            if filename.endswith(".txt") and filename not in ("edge_index_new.txt"):
                data_entry = self.read_data_entry(data_folder, filename)

                # For training, use all edges (full edge_index)
                if i in train_indices:
                    data_entry['train_mask'] = torch.ones(len(data_entry['x']), dtype=torch.bool)

                # For validation, set labels to zeros since we don't have access to them
                elif i in val_indices:
                    data_entry['train_mask'] = torch.zeros(len(data_entry['x']), dtype=torch.bool)
                    data_entry['y'] = torch.zeros_like(data_entry['y'])

                # For testing, set labels to zeros since we don't have access to them
                elif i in test_indices:
                    data_entry['train_mask'] = torch.zeros(len(data_entry['x']), dtype=torch.bool)
                    data_entry['y'] = torch.zeros_like(data_entry['y'])

                self.data_entries.append(data_entry)


    def read_data_entry(self, data_folder, filename):
        # Read node features and ground truth labels from the x_file
        x_file_path = os.path.join(data_folder, filename)
        x_data = np.loadtxt(x_file_path, delimiter=',')
        
        # Assuming the filename for edge_index file is consistent with x_file_path
        edge_file_path = os.path.join(data_folder, "edge_index_new.txt")
        edge_index = np.loadtxt(edge_file_path, delimiter=' ', dtype=int)

        # Extract node features and ground truth labels
        x = torch.FloatTensor(x_data[:, :-3])
        y = torch.FloatTensor(x_data[:, -3:])
        edge_index = torch.LongTensor(edge_index)

        # Set up masks for unsupervised learning (random split for illustration)
        num_nodes = len(x)
        num_train = int(0.8 * num_nodes)  # 80% for training
        num_val = int(0.1 * num_nodes)    # 10% for validation
        num_test = num_nodes - num_train - num_val  # Remaining 10% for testing

        train_indices = np.random.choice(num_nodes, num_train, replace=False)
        val_indices = np.random.choice(np.setdiff1d(np.arange(num_nodes), train_indices), num_val, replace=False)
        test_indices = np.setdiff1d(np.arange(num_nodes), np.concatenate((train_indices, val_indices)))

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        return {'x': x, 'edge_index': edge_index, 'y': y, 'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}



current_directory = os.getcwd()
sub_directory = 'gnn_data_son'
directory_path = os.path.join(current_directory, sub_directory)
dataset = CustomDataset(directory_path, num_features=37)
#print(f'Dataset: {dataset}:')
#print('======================')
#print(f'Number of graphs: {len(dataset.data_entries)}')
#print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')




class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 3)
        #init.xavier_normal_(self.linear.weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.6, training=self.training)
        x = self.linear(x)
        x = F.softmax(x)
        return x

def visualize(h, color):
    z = TSNE(n_components=1).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 5))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], np.zeros_like(z[:, 0]), s=70, c=color, cmap="Set2")
    #plt.show()
    return z

class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Add a small epsilon to avoid taking the log of zero
        epsilon = 1e-8
        alpha = 1
        pred = torch.clamp(pred, min=epsilon)

        # Calculate the log-scaled error
        log_pred = torch.log(pred + 1e-9)
        log_target = torch.log(target + 1e-9)
        squared_log_error = alpha * F.mse_loss(log_pred, log_target)
        total_loss = F.mse_loss(pred, target) + squared_log_error

        return total_loss
    
model = GCN(hidden_channels=24)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = torch.nn.MSELoss()

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data['x'], data['edge_index'])
    pred_values = out[data['train_mask']]
    true_values = data['y'][data['train_mask']]
    loss = criterion(pred_values, true_values)
    loss.backward()
    optimizer.step()
    return loss

def test(model, data):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[data['test_mask']]
    true_values = data['y'][data['test_mask']]
    return pred_values, true_values

def validate(model, data):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[data['val_mask']]
    true_values = data['y'][data['val_mask']]
    return pred_values, true_values

model = GCN(hidden_channels=24)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = torch.nn.MSELoss()
train_loader = DataLoader(dataset.data_entries, batch_size=32, shuffle=True)

for epoch in range(1, 1001):
    for batch_data in train_loader:
        optimizer.zero_grad()
        for data_entry in batch_data:
            loss = train(model, data_entry, optimizer, criterion)
        optimizer.step()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# Initialize the model, optimizer, and criterion
model = GCN(hidden_channels=24)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
criterion = torch.nn.MSELoss()


# Test the model on the testing data_entries without knowing the labels
test_predictions, test_labels = [], []
for data_entry in dataset.data_entries:
    if data_entry['test_mask'].any():
        test_pred, test_label = test(model, data_entry)
        test_predictions.append(test_pred)
        test_labels.append(test_label)
test_predictions = torch.cat(test_predictions)
test_labels = torch.cat(test_labels)
test_loss = criterion(test_predictions, test_labels)
print(f'Test Loss: {test_loss:.4f}')

# Validate the model on the validation data_entries and calculate loss
val_predictions, val_labels = [], []
for data_entry in dataset.data_entries:
    if data_entry['val_mask'].any():
        val_pred, val_label = validate(model, data_entry)
        val_predictions.append(val_pred)
        val_labels.append(val_label)
val_predictions = torch.cat(val_predictions)
val_labels = torch.cat(val_labels)
val_loss = criterion(val_predictions, val_labels)
print(f'Validation Loss: {val_loss:.4f}')
