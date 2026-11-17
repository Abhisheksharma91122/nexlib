import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import matplotlib.pyplot as plt 
import time 
from torch.utils.data import DataLoader, TensorDataset, random_split


# Load CSV files
train_data = pd.read_csv("./Handwritten/mnist_train.csv")
test_data = pd.read_csv("./Handwritten/mnist_test.csv")

# seperate images and labels 
X_train = torch.tensor(train_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_train = torch.tensor(train_data.iloc[:, 0].values, dtype=torch.long)

X_test = torch.tensor(test_data.iloc[:, 1:].values, dtype=torch.float32) / 255.0
y_test = torch.tensor(test_data.iloc[:, 0].values, dtype=torch.long)

# create dataset 
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# seperate data as training and validationabs
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size]
)

# create data Loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("MNIST Dataset Loaded Successfully!")


# MLP model with a selectable activation function

class NeuralNetwork(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x



# train and evaluate one model 

def train_model(model, epochs=5):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_accuracies = []
    val_accuracies = []
    epoch_time = []
    gradient_flow = []

    for epoch in range(epochs):

        start_time = time.time()
        model.train()
        
        total = 0
        correct = 0 
        total_loss = 0 
        layer1_grad = 0 
        layer2_grad = 0
        batches = 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Record gradient size for the two hidden layers
            layer1_grad += model.fc1.weight.grad.norm().item()
            layer2_grad += model.fc2.weight.grad.norm().item()
            batches += 1

            optimizer.step()

            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        train_accuracies.append(100 * correct / total)
        gradient_flow.append([
            layer1_grad / batches,
            layer2_grad / batches
        ])


        # validation accuracy 
        model.eval()
        correct = 0 
        total = 0 

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                predicted = outputs.argmax(1)
                total += labels.size(0) 
                correct += (predicted == labels).sum().item()

        val_accuracies.append(100 * correct / total)
        epoch_time.append(time.time() - start_time)

        print(
            "Epoch: ", epoch+1,
            "| Train Accuracy: ", round(train_accuracies[-1], 2), "%",
            "| Validation Accuracy: ", round(val_accuracies[-1], 2), "%"
        )

    return train_accuracies, val_accuracies, epoch_time, gradient_flow


# Train the model with three activation functions 

activations = ["sigmoid", "tanh", "relu"]
models = {}
histories = {}
training_times = {}
test_accuracies = {}

for act in activations:
    print("Training with ", act)

    # Use the same starting weight for fair comparison
    torch.manual_seed(42)
    model = NeuralNetwork(act)

    start = time.time()
    history = train_model(model, epochs=5)
    training_times[act] = time.time() - start 

    models[act] = model 
    histories[act] = history

    # Test accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracies[act] = 100 * correct / total
    print("Test Accuracy:", round(test_accuracies[act], 2), "%")
    print("Total Training Time: ", round(training_times[act], 2), "seconds")



# Compare Training and Validation Accuracy

plt.figure(figsize=(10,5))

for act in activations:
    plt.plot(histories[act][0], marker="o", label= act + "- Training")
    plt.plot(histories[act][1], marker="x", linestyle="--", label= act + "- Validation")

plt.title("Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.show()


# Compare Total Training Time

plt.figure(figsize=(7,4))
plt.bar(activations, [training_times[a] for a in activations])
plt.title("Training Time Comparison")
plt.xlabel("Activation Fuction")
plt.ylabel("Time (seconds)")
plt.show()


# Compare test accuracy

plt.figure(figsize=(7, 4))
plt.bar(activations, [test_accuracies[a] for a in activations])
plt.title("Test Accuracy Comparison")
plt.xlabel("Activation Function")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.show()


# Visualize gradient flow in the hidden layers

plt.figure(figsize=(8, 5))

for act in activations:
    gradients = histories[act][3]
    layer1 = [g[0] for g in gradients]
    layer2 = [g[1] for g in gradients]

    plt.plot(layer1, marker="o", label=act + " - Hidden Layer 1")
    plt.plot(layer2, marker="x", linestyle="--", label=act + " - Hidden Layer 2")

plt.title("Gradient Flow Comparison")
plt.xlabel("Epochs")
plt.ylabel("Average Gradient Norm")
plt.legend()
plt.grid()
plt.show()