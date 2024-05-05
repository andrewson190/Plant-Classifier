import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#50x50: 40 epochs, batch size = 64, 80% accuracy
#       70 epochs, batch size = 128, 80% accuracy

#one conv, one linear layer, 32 batch 25 sepochs, 16 batch, 50 epochs
#both 72% out, 100% in


# Step 1: Data Preparation
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
])

# Load dataset
train_dataset = datasets.ImageFolder(root="C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/50x50", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

"""class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
                self.fc = nn.Linear(32 * 25 * 25, 100)  # 100 classes

            def forward(self, x):
                x = nn.functional.relu(self.conv1(x))
                x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
                x = x.view(-1, 32 * 25 * 25)
                x = self.fc(x)
                return x"""
# Model Definition
"""class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 12 * 12, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 12 * 12)
        x = self.fc(x)
        return x"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 12 * 12)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 25 * 25, 512)

        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(32 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        #x = nn.functional.relu(self.conv2(x))
        #x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16 * 25 * 25)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x"""
"""class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 512)  # Adjusted input size after the second convolution
        self.fc2 = nn.Linear(512, 256)  # New fully connected layer
        self.fc3 = nn.Linear(256, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 12 * 12)  # Adjusted flatten size after the second convolution
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))  # Added activation after the new fully connected layer
        x = self.fc3(x)
        return x"""
"""class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 12 * 12, 512)  # Adjusted input size after the second convolution
        self.fc2 = nn.Linear(512, 256)  # New fully connected layer
        self.fc3 = nn.Linear(256, 128)  # Additional fully connected layer
        self.fc4 = nn.Linear(128, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 12 * 12)  # Adjusted flatten size after the second convolution
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))  # Added activation after the new fully connected layer
        x = nn.functional.relu(self.fc3(x))  # Added activation after the additional fully connected layer
        x = self.fc4(x)
        return x"""
"""class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 25 * 25, 100)  # 100 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * 25 * 25)
        x = self.fc(x)
        return x"""
#Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

#Testing
# Load test dataset and evaluate the model on it
test_dataset = datasets.ImageFolder(root="C:/Users/sonan/OneDrive/Documents/GitHub/Machine-Learning-for-Plants/50x50_test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {(correct / total) * 100:.2f}%')

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on training set: {(correct / total) * 100:.2f}%')

    