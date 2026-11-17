import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, random_split

def load(p):
    d = pd.read_csv(p).values
    return TensorDataset(torch.tensor(d[:, 1:], dtype=torch.float32) / 255, torch.tensor(d[:, 0]))

train, val = random_split(load("./Handwritten/mnist_train.csv"), [0.8, 0.2])
train_loader = DataLoader(train, 64, shuffle=True)
val_loader = DataLoader(val, 64)
test_loader = DataLoader(load("./Handwritten/mnist_test.csv"), 64)


class NeuralNetwork(nn.Module):

    def __init__(self, dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(model, optimizer, epochs=5):

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Training
        model.train()

        total_loss = 0
        correct = 0
        total = 0

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
            predicted = outputs.argmax(1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = (100 * correct / total)
        avg_loss = (total_loss/len(train_loader))

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Validation
        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item()

                predicted = outputs.argmax(1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print("Epoch:", epoch + 1,
              "Train Accuracy:", round(train_accuracies[-1], 2),
              "Validation Accuracy:", round(val_accuracies[-1], 2))

    return train_losses, val_losses, train_accuracies, val_accuracies



model1 = NeuralNetwork(dropout=False)

optimizer1 = optim.Adam(model1.parameters(), lr=0.001)

history1 = train_model(model1, optimizer1)


model2 = NeuralNetwork(dropout=False)

optimizer2 = optim.Adam(
    model2.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

history2 = train_model(model2, optimizer2)


model3 = NeuralNetwork(dropout=True)

optimizer3 = optim.Adam(model3.parameters(), lr=0.001)

history3 = train_model(model3, optimizer3)


plt.figure(figsize=(12, 5))

# Training Loss
plt.subplot(1, 2, 1)

plt.plot(history1[0], label="Without Regularization")
plt.plot(history2[0], label="L2 Regularization")
plt.plot(history3[0], label="Dropout")

plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Validation Loss
plt.subplot(1, 2, 2)

plt.plot(history1[1], label="Without Regularization")
plt.plot(history2[1], label="L2 Regularization")
plt.plot(history3[1], label="Dropout")

plt.title("Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))

# Training Accuracy
plt.subplot(1, 2, 1)

plt.plot(history1[2], label="Without Regularization")
plt.plot(history2[2], label="L2 Regularization")
plt.plot(history3[2], label="Dropout")

plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

# Validation Accuracy
plt.subplot(1, 2, 2)

plt.plot(history1[3], label="Without Regularization")
plt.plot(history2[3], label="L2 Regularization")
plt.plot(history3[3], label="Dropout")

plt.title("Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

