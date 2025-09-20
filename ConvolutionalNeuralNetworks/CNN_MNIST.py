import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


#定义CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #定义卷积层：输入1通道，输出32通道，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 定义卷积层：输入32通道，输出64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128) #输入大小 = 特征图大小 * 通道数
        self.fc2 = nn.Linear(128, 10) #10个类别

    def forward(self, x):
        x = F.relu(self.conv1(x)) #第一层卷积 + ReLU
        x = F.max_pool2d(x, 2) #最大池化
        x = F.relu(self.conv2(x)) #第二层卷积 + ReLU
        x = F.max_pool2d(x, 2) #最大池化
        x = x.view(-1, 64 * 7 * 7) #展平
        x = F.relu(self.fc1(x)) #全连接层 + ReLU
        x = self.fc2(x) #全连接层输出
        return x

#创建模型实例
model = SimpleCNN()

#定义损失函数与优化器
criterion = nn.CrossEntropyLoss() #多分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

#训练模型
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        #向前传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

#测试模型
model.eval() #设置为评估模式
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100.0 * correct / total
print(f"Accuracy of the network : {accuracy: .2f}%")

#可视化测试
dataiter = iter(test_loader)
images, labels = next(dataiter)
outputs = model(images)
_, predictions = torch.max(outputs, 1)

fig, axes = plt.subplots(1, 6, figsize=(12, 4))
for i in range(6):
    axes[i].imshow(images[i][0], cmap='gray')
    axes[i].set_title(f"Label: {labels[i]}\nPred: {predictions[i]}")
    axes[i].axis('off')
plt.show()