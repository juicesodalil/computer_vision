import torch, torchvision
from torchvision import datasets, transforms
from torch.utils import data
from network import ResidualNetwork
from torch import nn
from backbone import train, test

"""
    cifar10 和 cifar100 图像识别任务使用resnet
"""
network_name = "resnet_50"
batch_size = 128
epochs = 40
lr = 0.01
norm_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
norm_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
transform_train = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# train_set_10 = datasets.CIFAR10("./datasets/cifar10", train=True, transform=transform_train, download=True)
# test_set_10 = datasets.CIFAR10("./datasets/cifar10", train=False, transform=transform_test, download=True)

train_set_100 = datasets.CIFAR100("./datasets/cifar100", train=True, transform=transform_train, download=True)
test_set_100 = datasets.CIFAR100("./datasets/cifar100", train=False, transform=transform_test, download=True)

train_loader = data.DataLoader(train_set_100, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_set_100, batch_size, shuffle=True)

device = torch.device("cuda")
# 得到模型
model = ResidualNetwork.ResNet50(100)
# model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

is_test = True

if is_test:
    model.load_state_dict(torch.load(f"pretrain/cifar10-{network_name}-{epochs}.pth", map_location="cuda:0"))
    model.to(device)
    test(model, test_loader, criterion, device, network_name)
else:
    # model.load_state_dict(torch.load(f"pretrain/cifar10-resnet_50-70.pth", map_location="cuda:0"))
    model.to(device)
    train(epochs, model, train_loader, criterion, optimizer, scheduler, device, network_name)
