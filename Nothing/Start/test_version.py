import torch

print(torch.__version__)
# 检查 CUDA 是否可用，即你的系统有 NVIDIA 的 GPU
print(torch.cuda.is_available())
print(torch.version.cuda)  # 查看 PyTorch 编译时用的 CUDA 版本
print(torch.backends.cudnn.version())  # 查看 cuDNN 版本