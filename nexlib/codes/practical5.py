import torch
import torch.nn as nn 
import torch.optim as optim 
import pandas as pd 
import matplotlib.pyplot as plt 
import time

from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms


train_data = pd.read_csv("./Handwritten/mnist_train.csv")
test_data = pd.read_csv("./Handwritten/mnist_test.csv")

x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

x_test = train_data.iloc[:, 1:].values
y_test = train_data.iloc[:, 0].values


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].reshape(1, 28, 28) / 255.0

        if self.transform:
            image = self.transform(image)

        return image, self.labels[index]


train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
])

train_dataset = MNISTDataset(x_train, y_train, train_transform) 
test_dataset = MNISTDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_accuracy = []

    for epoch in range(epochs):
        correct = 0
        total = 0

        model.train()

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        accuracy = 100 * correct / total
        train_accuracy.append(accuracy)

        print("Epoch : ", epoch+1,
             "Training Accuracy: ", round(accuracy, 2), " % ")

    return train_accuracy

def test_model(model):
    model.eval()

    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy


models = {
    "MLP" : MLP(),
    "CNN" : CNN()
}

histories = {}
accuracies = {}
training_time = {}

for name, model in models.items():
    print("Training Model: ", name)

    start_time = time.time()

    histories[name] = train_model(model)
    
    training_time[name] = time.time() - start_time

    accuracies[name] = test_model(model)
    
    print("Testing Accuracy: ", round(accuracies[name], 2), " %",
         "Training Time: ", round(training_time[name], 2), "seconds")



for name in histories:
    plt.plot(histories[name], marker="o", label= name)

plt.title("MLP vs CNN Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.show()


plt.bar(accuracies.keys(), accuracies.values())

plt.title("MLP vs CNN Test Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.show()


plt.bar(training_time.keys(), training_time.values())

plt.title("MLP vs CNN Training Time")
plt.xlabel("Model")
plt.ylabel("Time (seconds)")
plt.show()


print("Final Comparison")

for name in models:
    print(
        name,
        "| Accuracy:", round(accuracies[name], 2), "%",
        "| Time:", round(training_time[name], 2), "seconds"
    )