import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


#定义RNN模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        #定义RNN层
        self.nn = nn.RNN(input_size, hidden_size, batch_first=True)
        #定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #x:(batch_size, seq_len, input_size)
        out, _ = self.nn(x)
        #取序列最后一个时间步作为模型的输出
        out = out[:, -1, :] #(barch_size, hidden_size)
        out = self.fc(out)
        return out

#创建训练数据
#生成一群随机序列数据
num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2 #假设二分类问题
#随机生成数据(batch_size, seq_len, input_size)
X = torch.randn(num_samples, seq_len, input_size)
#随机生成目标标签(batch_size, output_size)
Y = torch.randint(0, output_size, (num_samples,))
#创建数据加载器
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size = 32, shuffle=True)

#定义损失函数与优化器
model = SimpleRNN(input_size = input_size, hidden_size = 64, output_size = output_size)
criterion = nn.CrossEntropyLoss() #多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

#训练模型
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        #前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        #计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss:{total_loss / len(train_loader):.4f}, Accuracy:{accuracy:.2f}% ")

#测试模型
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")




