# PyTorch torchvision 计算机视觉模块
# torchvision 是 PyTorch 生态系统中专门用于计算机视觉任务的扩展库，它提供了以下核心功能：
#
# 预训练模型：包含经典的 CNN 架构实现（如 ResNet、VGG、AlexNet 等）
# 数据集工具：内置常用视觉数据集（如 CIFAR10、MNIST、ImageNet 等）
# 图像变换：提供各种图像预处理和数据增强方法
# 实用工具：包括视频处理、图像操作等辅助功能

#############################
#############################
### 核心组件解析

# 1. torchvision.models
# 提供预训练的计算机视觉模型，可直接用于迁移学习：
#
# 实例
from torchvision import transforms
import torchvision.models as models
from torchvision.models import AlexNet_Weights

# 加载预训练模型
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)   # 或 .DEFAULT
vgg16 = models.vgg16(pretrained=True)
# 常用模型列表：
# 模型名称	    适用场景	    参数量	    Top-1 准确率
# ResNet	    通用图像分类	11M-60M	    69%-80%
# VGG	        特征提取	    138M	    71.3%
# MobileNet	    移动端应用	3.4M	    70.6%
# EfficientNet	高效模型	    5M-66M	    77%-84%


# 2. torchvision.datasets
# 内置常用计算机视觉数据集，简化数据加载流程：
#
# 实例
from torchvision import datasets

# 加载 CIFAR10 数据集
train_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 加载 MNIST 数据集
test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True
)


# 3. torchvision.transforms
# 图像预处理和数据增强的核心工具：
#
# 实例
from torchvision import transforms
# 定义图像变换管道
transform = transforms.Compose([
    transforms.Resize(256),          # 调整大小
    transforms.CenterCrop(224),       # 中心裁剪
    transforms.ToTensor(),           # 转为张量
    transforms.Normalize(             # 标准化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# 常用变换方法分类：
# 类别	    方法示例	                            作用
# 几何变换	RandomRotation, RandomResizedCrop	增加位置不变性
# 颜色变换	ColorJitter, Grayscale	            增强颜色鲁棒性
# 模糊/噪声	GaussianBlur, RandomErasing	        防止过拟合
# 组合变换	RandomApply, RandomChoice	        灵活组合策略
