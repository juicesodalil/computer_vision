# 导入所需的库
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data

# 定义超参数
batch_size = 128
num_epochs = 10
learning_rate = 0.01

# 定义数据集的转换，将图片转换为张量，并归一化
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 下载并加载训练集和测试集
trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True,
                                        download=True, transform=transform)
trainloader = data.DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../datasets', train=False,
                                       download=True, transform=transform)
testloader = data.DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

# 定义类别标签
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义 bag of words 模型，使用线性层和 softmax 激活函数
class BoWModel(torch.nn.Module):
    def __init__(self, vocab_size, num_classes):
        super(BoWModel, self).__init__()
        self.linear = torch.nn.Linear(vocab_size, num_classes)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        return outputs


# 定义词汇表的大小，即图片的像素数
vocab_size = 3 * 32 * 32  # RGB channels * height * width

# 定义模型，损失函数和优化器
model = BoWModel(vocab_size, 10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入数据和标签
        inputs, labels = data

        # 将输入数据展平为一维向量
        inputs = inputs.view(-1, vocab_size)

        # 将梯度清零
        optimizer.zero_grad()

        # 前向传播，计算损失，反向传播，更新参数
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if i % 200 == 199:  # 每200个批次打印一次平均损失
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# 测试模型在测试集上的准确率
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.view(-1, vocab_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
