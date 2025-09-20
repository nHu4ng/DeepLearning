from torchvision import transforms

#####
#基础变换操作
#####
# 1、ToTensor
# 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量。
# 同时将像素值从 [0, 255] 归一化为 [0, 1]。
transform = transforms.ToTensor()

# 2、Normalize
# 对数据进行标准化，使其符合特定的均值和标准差。
# 通常用于图像数据，将其像素值归一化为零均值和单位方差。
transform = transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]

# 3、Resize
# 调整图像的大小。
transform = transforms.Resize((128, 128))  # 将图像调整为 128x128

# 4、CenterCrop
# 从图像中心裁剪指定大小的区域
transform = transforms.CenterCrop(128)  # 裁剪 128x128 的区域


#####
#数据增强操作
#####
# 1、RandomCrop
# 从图像中随机裁剪指定大小。
transform = transforms.RandomCrop(128)

# 2、RandomHorizontalFlip
# 以一定概率水平翻转图像。
transform = transforms.RandomHorizontalFlip(p=0.5)  # 50% 概率翻转

# 3、RandomRotation
# 随机旋转一定角度。
transform = transforms.RandomRotation(degrees=30)  # 随机旋转 -30 到 +30 度

# 4、ColorJitter
# 随机改变图像的亮度、对比度、饱和度或色调。
transform = transforms.ColorJitter(brightness=0.5, contrast=0.5)

#####
#组合变换
#####
#通过 transforms.Compose 将多个变换组合起来。
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#####
#自定义变换
#####
class CustomTransform:
    def __call__(self, x):
        # 这里可以自定义任何变换逻辑
        return x * 2

transform = CustomTransform()