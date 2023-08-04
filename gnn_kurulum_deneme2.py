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


os.environ['TORCH'] = torch.__version__
os.environ['PYTHONWARNINGS'] = "ignore"

#dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
#print(dataset[0].edge_index)


class CustomDataset:
    def __init__(self, data_folder, num_features):
        self.num_features = num_features
        self.data_entries = []
        list_files = os.listdir(data_folder)
        for i in range(2):
            filename = list_files[i]
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
        x = torch.FloatTensor(x_data[:, :-2])
        y = torch.FloatTensor(x_data[:, -2:])
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
        self.linear = torch.nn.Linear(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.relu()
        x = self.linear(x)
        x = x.sigmoid()
        return x

model = GCN(hidden_channels=4)
print(model)

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    print(np.shape(z))
    print(z)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    #plt.plot(z)
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model.eval()

out = model(data['x'], data['edge_index'])
#visualize(out, color=data['y'])

model = GCN(hidden_channels=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, weight_decay=10e-5)
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
    print(len(pred_values))
    print(pred_values)
    true_values = data['y'][mask]
    print(len(true_values))
    print(true_values)
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


for epoch in range(1, 101):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    

test_loss = test(data['test_mask'])
print(f'Test Loss: {test_loss:.4f}')

val_loss = validate(data['val_mask'])
print(f'Validation Loss: {val_loss:.4f}')

model.eval()
out = model(data['x'], data['edge_index'])
#visualize(out, color=data['y'])
