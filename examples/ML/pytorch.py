import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import time

# Load and prepare data
data = np.load('mnist.npz')

def image_to_vector(X):
    return np.reshape(X, (len(X), -1))

Xtrain = image_to_vector(data['train_x'])
Ytrain = np.zeros((Xtrain.shape[0], 10))
Ytrain[np.arange(Xtrain.shape[0]), data['train_y']] = 1.0

Xtest = image_to_vector(data['test_x'])
Ytest = data['test_y']

# Convert to PyTorch tensors and move to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.FloatTensor(Xtrain).to(device)
Y_train = torch.FloatTensor(Ytrain).to(device)
X_test = torch.FloatTensor(Xtest).to(device)
Y_test = torch.LongTensor(Ytest).to(device)

# Create dataset and dataloader
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Define the model
class DenseMNIST(nn.Module):
    def __init__(self):
        super(DenseMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.reshape([-1, 1, 28, 28])
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = DenseMNIST().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 40
total_iterations = 0
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, torch.max(batch_Y, 1)[1])
        loss.backward()
        optimizer.step()
        
        total_iterations += 1
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == Y_test).float().mean()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}")

end_time = time.time()
training_time = end_time - start_time

# Final test accuracy
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    final_accuracy = (predicted == Y_test).float().mean()
print(f"Final Test Accuracy: {final_accuracy.item():.4f}")

# Calculate and print iterations per second
iterations_per_second = total_iterations / training_time
print(f"Number of optimization iterations per second: {iterations_per_second:.2f}")