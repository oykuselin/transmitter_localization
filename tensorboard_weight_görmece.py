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
            if filename.endswith(".txt") and filename not in ("edge_index_diag_1.txt"):
                data_entry = self.read_data_entry(data_folder, filename)
                self.data_entries.append(data_entry)

    def read_data_entry(self, data_folder, filename):
        # Read node features and ground truth labels from the x_file
        x_file_path = os.path.join(data_folder, filename)
        x_data = np.loadtxt(x_file_path, delimiter=',')
        
        # Assuming the filename for edge_index file is consistent with x_file_path
        edge_file_path = os.path.join(data_folder, "edge_index_diag_1.txt")
        edge_index = np.loadtxt(edge_file_path, delimiter=',', dtype=int)

        # Extract node features and ground truth labels
        x = torch.FloatTensor(x_data[:, :-1])
        y = torch.FloatTensor(x_data[:, -1:])
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


def apply_yeojohnson_transform(x_data):
    # Convert tensor array to numpy array
    x_data_np = x_data.numpy()

    # Check for missing values
    if np.isnan(x_data_np).any():
        raise ValueError("Data contains missing values (NaN). Please handle missing values before applying the transformation.")

    # Check for zero variance
    if np.all(np.var(x_data_np, axis=0) == 0):
        raise ValueError("Data has zero variance. Please check for constant columns in the input data.")

    # Scale and apply Yeo-Johnson transformation only to columns 7 to 11
    scaler = StandardScaler()
    x_data_scaled = x_data_np.copy()
    x_data_scaled[:, :] = scaler.fit_transform(x_data_np[:, :])

    # Convert the transformed numpy array back to a torch tensor
    x_transformed = torch.tensor(x_data_scaled, dtype=torch.float32)

    return x_transformed

current_directory = os.getcwd()
sub_directory = 'gnn_data_one_label'
directory_path = os.path.join(current_directory, sub_directory)
dataset = CustomDataset(directory_path, num_features=16)
#print(f'Dataset: {dataset}:')
#print('======================')
#print(f'Number of graphs: {len(dataset.data_entries)}')
#print(f'Number of features: {dataset.num_features}')
#print(f'Number of classes: {dataset.num_classes}')

data = dataset.data_entries[0]  # Get the first graph object.
data['x'] = apply_yeojohnson_transform(data['x'])
np.savetxt("deneme_feature_normalization.txt", data["x"])
print(data['x'])
#print(data)
#print(data['edge_index'])
#print(data['x'])
#print(len(data['x']))
#print(len(data['y']))
#print(data['y'])
#print(data['train_mask'].all())

class CustomGCNConv(GCNConv):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True):
        super().__init__(in_channels, out_channels, improved, cached, bias)

        # Initialize weights using Xavier/Glorot initialization
        init.xavier_normal_(self.lin.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = CustomGCNConv(dataset.num_features, hidden_channels)
        self.conv2 = CustomGCNConv(hidden_channels, 8)
        self.linear = torch.nn.Linear(8, 1)
        init.xavier_normal_(self.linear.weight)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.linear(x)
        x = x.sigmoid() 
        return x

model = GCN(hidden_channels=8)
print(model)

def visualize(h, color):
    z = TSNE(n_components=1).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10, 5))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], np.zeros_like(z[:, 0]), s=70, c=color, cmap="Set2")
    #plt.show()
    return z

model.eval()

out = model(data['x'], data['edge_index'])
#visualize(out, color=data['y'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, weight_decay=1e-5)
class MSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Add a small epsilon to avoid taking the log of zero
        epsilon = 1e-8
        alpha = 1e-2
        pred = torch.clamp(pred, min=epsilon)

        # Calculate the log-scaled error
        log_pred = torch.log(pred + 1e-9)
        log_target = torch.log(target + 1e-9)
        squared_log_error = alpha * F.mse_loss(log_pred, log_target)
        total_loss = F.mse_loss(pred, target) + squared_log_error

        return total_loss
criterion = torch.nn.MSELoss()
def loss_function_1(pred_values, true_values):
    # Calculate the first loss using MSE
    loss = F.mse_loss(pred_values[:, 0], true_values[:, 0])
    return loss

def loss_function_2(pred_values, true_values):
    # Calculate the second loss using MSE
    loss = F.mse_loss(pred_values[:, 1], true_values[:, 1])
    return loss

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data['x'], data['edge_index'])
    pred_values = out[data['train_mask']]
    #normalized_pred = pred_values / pred_values.sum(dim=1, keepdim=True)
    #loss_1 = loss_function_1(pred_values, data['y'][data['train_mask']])
    #loss_2 = loss_function_2(pred_values, data['y'][data['train_mask']])
    #total_loss = loss_1 + loss_2
    #total_loss.backward()
    loss = criterion(pred_values, data['y'][data['train_mask']])
    optimizer.step()
    return loss

def test(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]
    #normalized_pred = pred_values / pred_values.sum(dim=1, keepdim=True)
    print(len(pred_values))
    print(pred_values)
    true_values = data['y'][mask]
    print(len(true_values))
    print(true_values)
    #loss_1 = loss_function_1(pred_values, data['y'][data['test_mask']])
    #loss_2 = loss_function_2(pred_values, data['y'][data['test_mask']])
    #total_loss = loss_1 + loss_2
    loss = criterion(pred_values, true_values)
    return loss

def validate(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]  
    #normalized_pred = pred_values / pred_values.sum(dim=1, keepdim=True)
    np.savetxt('output.txt', pred_values.detach().numpy())
    true_values = data['y'][mask]
    np.savetxt("test_output.txt", true_values.detach().numpy())
    loss = criterion(pred_values, true_values)
    #loss = torch.sqrt(los
    return loss

for epoch in range(1, 101):
    loss = train()
    #for name, param in model.named_parameters():
        #if param.requires_grad:
            #print(name, param.data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

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
"""def calculate_confusion_matrix(mask):
    model.eval()
    out = model(data['x'], data['edge_index'])
    pred_values = out[mask]
    
    # Compare p_1 and p_2 to determine the predicted class
    pred_labels = torch.where(pred_values[:, 0] > pred_values[:, 1], 1, 0)
    true_labels = torch.where(data['y'][mask][:, 0] > data['y'][mask][:, 1], 1, 0)
    print(pred_labels)
    print(true_labels)
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    return cm

test_confusion_matrix = calculate_confusion_matrix(data['test_mask'])
print("Test Confusion Matrix:")
print(test_confusion_matrix)"""