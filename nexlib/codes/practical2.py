import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



from torch.utils.data import TensorDataset, DataLoader

# Load CSV files
train_data = pd.read_csv("./Handwritten/mnist_train.csv")
test_data = pd.read_csv("./Handwritten/mnist_test.csv")

# Separate images and labels
X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

X_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.long)

# Create datasets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Dataset Loaded Successfully!")


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = NeuralNetwork()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)


epochs = 5

train_losses = []
train_accuracies = []

for epoch in range(epochs):

    correct = 0
    total = 0
    running_loss = 0

    for images, labels in train_loader:
        #forward pass
        outputs = model(images)

        #calculate loss 
        loss = criterion(outputs, labels)

        # Backward pass and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate correct predictions
        _, predicted = torch.max(outputs, 1)

        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        running_loss += loss.item()

    accuracy = (100 * correct / total)
    avg_loss = (running_loss/len(train_loader))
    
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    # Print loss and accuracy
    print("Epoch: ", epoch+1,
         "Loss: ",round(avg_loss, 4),
         "Accuracy: ", round(accuracy, 2))


#Evaluation mode 
model.eval()

correct = 0
total = 0

# Disable gradient calculation
with torch.no_grad():

    for images, labels in test_loader:
        # Predict output
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        # Count correct predictions
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Calculate accuracy
print("Accuracy: ", round(100*correct/total, 2))


plt.figure(figsize=(12, 4))

# Loss graph
plt.subplot(1,2,1)
plt.plot(train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Accuracy graph
plt.subplot(1,2,2)
plt.plot(train_accuracies, marker='o')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.show()


# Get test images
images, labels = next(iter(test_loader))

# Predict digits
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Display 10 images
plt.figure(figsize=(10,4))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i].reshape(28, 28), cmap='gray')
    plt.title("Pred: " + str(predicted[i].item()))
    plt.axis('off')

plt.show()


