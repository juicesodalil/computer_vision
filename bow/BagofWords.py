import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from torch.utils import data


# Define the Bag-of-Visual-Words model
class BoVW(nn.Module):
    def __init__(self, num_clusters):
        super(BoVW, self).__init__()
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300)
        self.fc = nn.Linear(num_clusters, 10)

    def forward(self, x):
        # Extract features using pre-trained ResNet18
        resnet18 = torchvision.models.resnet18(weights='DEFAULT')
        modules = list(resnet18.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        features = features.view(features.size(0), -1)

        # Cluster the features using KMeans
        features = features.detach().numpy()
        self.kmeans.fit(features)
        cluster_centers = self.kmeans.cluster_centers_

        # Assign the features to the nearest cluster center
        assignments = self.kmeans.predict(features)
        histogram = np.zeros((x.shape[0], self.num_clusters))
        for assignment in assignments:
            for i in range(histogram.shape[0]):
                histogram[i][assignment] += 1

        # Normalize the histogram
        histogram /= np.sum(histogram)

        # Classify the image using a linear classifier
        outputs = self.fc(torch.Tensor(histogram))

        return outputs


# Define the training loop
def train(model, criterion, optimizer, trainloader, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"[{i} / {len(trainloader)}], loss : {loss.item()}")

        print('Epoch [%d], Loss: %.4f' % (epoch + 1, running_loss / len(trainloader)))
        print("save bow model")
        torch.save(model.state_dict(), "./pretrain/bow.pth")


# 超参数
batch_size = 64
# Load the CIFAR10 dataset
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=True, download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='../datasets/cifar10', train=False, download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Train the model
model = BoVW(num_clusters=50)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train(model, criterion, optimizer, trainloader, num_epochs=10)

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000')
