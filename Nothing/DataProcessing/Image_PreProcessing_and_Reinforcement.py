import torchvision.transforms as transforms
from PIL import Image

# 定义数据预处理的流水线
transform_1 = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像调整为 128x128
    transforms.ToTensor(),  # 将图像转换为张量,值会归一化到[0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，通常使用预训练模型时需要进行标准化处理。
])

# 加载图像
image = Image.open('image.jpg')

# 应用预处理
image_tensor = transform_1(image)
print(image_tensor.shape)  # 输出张量的形状


#图像数据增强,这些数据增强方法可以通过 transforms.Compose() 组合使用，保证每个图像在训练时具有不同的变换。
transform_2 = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(30),  # 随机旋转 30 度
    transforms.RandomResizedCrop(128),  # 随机裁剪并调整为 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

