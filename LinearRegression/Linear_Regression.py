import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


########数据准备
#随机种子
torch.manual_seed(42)

#生成训练数据(带噪声)
X = torch.randn(100, 2) #100个样本，每个样本2个特征
true_w = torch.tensor([2.0, 3.0])
true_b = 4.0
Y = X @ true_w + true_b + torch.randn(100) * 0.1

#打印部分数据
print(X[:5])
print(Y[:5])


#########定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        #定义一个线性层，输入特征为2，输出1个预测值
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

###########定义损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#随机梯度下降法

########训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    predictions = model(X)
    loss = criterion(predictions.squeeze(), Y)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if(epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1} / 1000, Loss {loss.item():.4f}]")

############评估模型
print("Finished Training")
print(f"Predicted weight:{model.linear.weight.data.numpy()}")
print(f"Predicted bias:{model.linear.bias.data.numpy()}")
#在新数据上预测
with torch.no_grad(): #评估时不需要计算梯度，节省内存
    predictions = model(X)

#可视化
plt.scatter(X[:,0], Y, color="blue", label = "True values")
plt.scatter(X[:, 0], predictions, color = 'red', label = "Predicted values")
plt.legend()
plt.show()




