import torch

#创建一个需要计算梯度的张量
x = torch.randn(2, 3, requires_grad=True)
print(x)

#执行操作
y = x + 2
z = y * y * 3
out = z.mean()#将z的所有元素求平均，得到一个标量，反向传播必须从一个标量开始，除非额外指定gradient参数
print(out)

#反向传播，计算梯度
out.backward()#启动反向传播，从out开始沿着计算图一路往回求导，直到把图中所有requires_grad=True的叶子节点（这里就是x）的梯度都算出来。
print(x.grad)
# 根据链式法则，
# out = mean(3·(x+2)²)
# ∂out/∂x = 6(x+2)/6 = x+2
#理论上x.grad=x+2的逐元素值

#使用torch.no_grad()禁用梯度计算
with torch.no_grad():
    y = x * 2