import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.cluster import KMeans
from torch.utils import data


# 定义 Bag-of-Visual-Words model
class BoVW(nn.Module):
    def __init__(self, num_clusters):
        super(BoVW, self).__init__()
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, n_init=10, max_iter=300)
        self.fc = nn.Linear(num_clusters, 10)

    def forward(self, x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 使用预训练好的resnet18抽取特征抽取特征，最后一层全联接不需要
        resnet18 = torchvision.models.resnet18(weights='DEFAULT')
        resnet18.to(device)
        modules = list(resnet18.children())[:-1]
        feature_extractor = nn.Sequential(*modules)
        features = feature_extractor(x)
        features = features.view(features.size(0), -1)

        # 使用KMeans算法进行特征的聚类
        features = features.detach().cpu().numpy()
        self.kmeans.fit(features)
        cluster_centers = self.kmeans.cluster_centers_

        # 将特征向量对应的聚类中心作为视觉词汇，将其出现次数统计为直方图，并对直方图进行归一化
        assignments = self.kmeans.predict(features)
        histogram = np.zeros((x.shape[0], self.num_clusters))
        for assignment in assignments:
            for i in range(histogram.shape[0]):
                histogram[i][assignment] += 1

        # 归一化直方图
        histogram /= np.sum(histogram)
        histogram = torch.tensor(histogram)
        histogram = histogram.to(torch.float32)
        if torch.cuda.is_available():
            histogram = histogram.cuda()
        # 直接过全联接层做分类
        outputs = self.fc(histogram)

        return outputs


def train(model, criterion, optimizer, trainloader, device, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"[{i} / {len(trainloader)}], loss : {loss.item()}")

        print('Epoch [%d], Loss: %.4f' % (epoch + 1, running_loss / len(trainloader)))
    print("save bow model")
    torch.save(model.state_dict(), f"../pretrain/bow-{num_epochs}.pth")


# 超参数
batch_size = 128
num_epochs = 1
# 定义对数据集的预处理
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

# 准备训练的模型优化器和损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BoVW(num_clusters=50)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
is_pretrain = True
if is_pretrain:
    model.load_state_dict(torch.load(f"../pretrain/bow-{num_epochs}.pth"))
    model.to(device)
else:
    train(model, criterion, optimizer, trainloader, device, num_epochs=num_epochs)

# 测试模型
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        if images.shape[0] < 50:
            continue
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        print("correct : %.2f%%" % ((correct / total) * 100))
