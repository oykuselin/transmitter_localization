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
        for i in range(2):
            filename = list_files[i]
            print(filename)
            if filename.endswith(".txt") and filename not in ("edge_index_new.txt"):
                data_entry = self.read_data_entry(data_folder, filename)
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

data = dataset.data_entries[0]  # Get the first graph object.
print(data['x'])
#print(data)
#print(data['edge_index'])
#print(data['x'])
#print(len(data['x']))
#print(len(data['y']))
#print(data['y'])
#print(data['train_mask'].all())


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

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data['x'], data['edge_index'])
    pred_values = out[data['train_mask']]
    loss = criterion(pred_values, data['y'][data['train_mask']])
    loss.backward()
    optimizer.step()
    return loss

def test(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]
    #print(pred_values)
    true_values = data['y'][mask]
    #print(true_values)
    for i in range(len(pred_values)):
        print('Predicted Values:')
        print(pred_values[i])
        print('True Values:')
        print(true_values[i])
        print()
        #if true_values[i][0] == 0 and true_values[i][1] == 0:
         #   pred_values = tensor.detach().numpy()
         #   np.delete(pred_values,i,axis=0)
         #   np.delete(true_values,i,axis=0)
    loss = criterion(pred_values, true_values)
    return loss

def validate(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]  
    np.savetxt('output.txt', pred_values.detach().numpy())
    true_values = data['y'][mask]
    np.savetxt("test_output.txt", true_values.detach().numpy())
    loss = criterion(pred_values, true_values)
    return loss

for epoch in range(1, 1001):
    loss = train()
    #for name, param in model.named_parameters():
        #if param.requires_grad:
            #print(name, param.data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

model.eval()
test_loss = test(data['test_mask'])
print(f'Test Loss: {test_loss:.4f}')

val_loss = validate(data['val_mask'])
print(f'Validation Loss: {val_loss:.4f}')

model.eval()
#print(data['x'][0][10:50])
out = model(data['x'], data['edge_index'])
"""z = visualize(out, color=data['y'])
z = z.astype(float)

z = z.ravel()
min_val = min(z)
max_val = max(z)
print(max_val)
    # Perform Min-Max normalization
normalized_z = [(x - min_val) / (max_val - min_val) for x in z]
normalized_z = np.around(normalized_z, decimals=4)
print(normalized_z)
print(data['y'])

mse_loss = np.mean((data['y'] - normalized_z) ** 2)
print(f'Mean Squared Error (MSE) Loss: {mse_loss:.4f}')"""
def calculate_confusion_matrix(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]
    
    # Compare p_1 and p_2 to determine the predicted class
    pred_labels = torch.where(pred_values[:, 0] >= pred_values[:, 1], 1, 0)
    true_labels = torch.where(data['y'][mask][:, 0] >= data['y'][mask][:, 1], 1, 0)
    print(pred_labels)
    print(true_labels)
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    return cm

test_confusion_matrix = calculate_confusion_matrix(data['test_mask'])
print("Test Confusion Matrix:")
print(test_confusion_matrix)