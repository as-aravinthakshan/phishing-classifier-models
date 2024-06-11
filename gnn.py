import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
dataset = pd.read_csv('dataset.csv')

# Drop the index column
dataset = dataset.drop(columns=['index'])

# Convert target labels -1 to 0
dataset['Result'] = dataset['Result'].replace(-1, 0)

# Split the dataset into features and target
X = dataset.drop(columns=['Result'])
y = dataset['Result']
print("Target Variable Fixed !")

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)
print("Normalization Done !")

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Create a fully connected graph
def create_fully_connected_graph(X, y):
    # Number of nodes
    num_nodes = X.shape[0]
    
    # Node features
    x = torch.tensor(X, dtype=torch.float)
    
    # Edge indices (fully connected graph)
    edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes)], dtype=torch.long).t().contiguous()
    
    # Target labels
    y = torch.tensor(y.values, dtype=torch.long)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data

# Create the graph data for training and test sets
train_data = create_fully_connected_graph(X_train, y_train)
test_data = create_fully_connected_graph(X_test, y_test)

print("Connected Graph Created !")

# Define a simple GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Model parameters
num_features = X_train.shape[1]
num_classes = 2  # Phishing or not

# Instantiate the model
model = GCN(num_features, num_classes)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(train_data)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

# Testing the model
def test(data):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred.eq(data.y).sum().item())
    acc = correct / data.num_nodes
    return acc

# Training loop
for epoch in range(1, 201):
    loss = train()
    train_acc = test(train_data)
    test_acc = test(test_data)
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Print final accuracy
print(f'Final Train Accuracy: {train_acc:.4f}')
print(f'Final Test Accuracy: {test_acc:.4f}')
