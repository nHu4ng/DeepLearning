import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型移动到设备
model.to(device)

# 将数据移动到设备
X = X.to(device)
Y = Y.to(device)