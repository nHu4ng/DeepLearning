import torch
import torch.nn as nn
import torch.optim as optim

#定义一个简单的全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(2,2) #输入层到隐藏层
        self.fc2 = nn.Linear(2,1) #隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #ReLU激活函数
        x = self.fc2(x)
        return x

#创建网络实例
model = SimpleNN()

#打印模型结构
##print(model)

# 训练过程：
# 前向传播（Forward Propagation）： 在前向传播阶段，输入数据通过网络层传递，每层应用权重和激活函数，直到产生输出。
# 计算损失（Calculate Loss）： 根据网络的输出和真实标签，计算损失函数的值。
# 反向传播（Backpropagation）： 反向传播利用自动求导技术计算损失函数关于每个参数的梯度。
# 参数更新（Parameter Update）： 使用优化器根据梯度更新网络的权重和偏置。
# 迭代（Iteration）： 重复上述过程，直到模型在训练数据上的性能达到满意的水平。

#前向传播与损失计算
x = torch.randn(1,2)
##前向传播
output = model(x)
print(f"output = {output}")
##定义损失函数MSE
criterion = nn.MSELoss()
##假设目标值为1
target = torch.randn(1,1)
loss = criterion(output, target)
print(f"loss = {loss}")

#优化器Optimizers
##Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
##训练
optimizer.zero_grad()#清空梯度
loss.backward()#反向传播
optimizer.step()#更新参数
