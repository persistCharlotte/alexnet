import sys
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dropout = 0.5
batch_size = 32
learning_rate = 0.001
num_epoch = 10


transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))] #RGB三个通道都进行张量归一化
)

# 请于别处下载
# train_dataset = CIFAR10('./data', train=True, transform=transform, download=True)
# test_dataset = CIFAR10('./data', train=False, transform=transform)


# 直接使用绝对路径
data_root = r'D:\桌面\CIFAR10\data'

# 确保数据集路径正确
if not os.path.exists(data_root):
    raise RuntimeError(f"Dataset not found at {data_root}. Please ensure the path is correct.")

# 加载训练数据集（不下载）
train_dataset = CIFAR10(root=data_root, train=True, transform=transform, download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 加载测试数据集（不下载）
test_dataset = CIFAR10(root=data_root, train=False, transform=transform, download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class alexnet(nn.Module):
    def __init__(self, num_classes=10, init_weight=False):
        super(alexnet,self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True), #这里是降低内存
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 * 6 * 6,2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )

        #其实这个部分在当前版本是直接被使用的
        if init_weight:
            self._initialize_weight()

    def forward(self,x):
        x = self.feature(x)
        # 展平的索引是1，相当于在channel维度
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

model = alexnet(num_classes=10, init_weight=True)
model.to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(num_epoch):
    # 训练模式，可以改变参数
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        # 让梯度为0
        optimizer.zero_grad()
        # forward+ backward + loss
        outputs = model(images.to(device))
        loss = loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.set_description(f"train epoch[{epoch + 1}/{num_epoch}] loss:{loss.item():.3f}")


    # 验证模式，参数不得改变
    model.eval()
    acc = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_image, test_labels = test_data
            outputs = model(test_image.to(device))
            predict = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict, test_labels.to(device)).sum().item()

        test_acc =  acc / len(test_loader.dataset)
        print(f'[epoch {epoch + 1}] train_loss: {running_loss / len(train_loader):.3f} test_acc: {test_acc:.3f}')

