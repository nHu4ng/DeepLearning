######## 基本保存和加载方法
#### 保存整个模型
# 这是最简单的方法，保存模型的架构和参数：
import torch
import torchvision.models as models
# 创建并训练一个模型
model = models.resnet18(pretrained=True)
# ... 训练代码 ...
# 保存整个模型
torch.save(model, 'model.pth')
# 加载整个模型
loaded_model = torch.load('model.pth')
# 优点：
# 代码简单直观
# 保存了完整的模型结构
# 缺点：
# 文件体积较大
# 对模型类的定义有依赖

#### 仅保存模型参数（推荐方式）
#更推荐的方式是只保存模型的状态字典(state_dict)：
# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')
# 加载模型参数
model = models.resnet18()  # 必须先创建相同架构的模型
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # 设置为评估模式
# 优点：
# 文件更小
# 更灵活，可以加载到不同架构中
# 兼容性更好

########保存和加载训练状态
#在实际项目中，我们通常还需要保存优化器状态、epoch等信息：

# 保存检查点
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    # 可以添加其他需要保存的信息
}

torch.save(checkpoint, 'checkpoint.pth')

# 加载检查点
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()  # 或者 model.train() 取决于你的需求



#######跨设备加载模型

####CPU/GPU兼容性处理
# 保存时指定map_location
torch.save(model.state_dict(), 'model_weights.pth')
# 加载到CPU（当模型是在GPU上训练时）
device = torch.device('cpu')
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
# 加载到GPU
device = torch.device('cuda')
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.to(device)
##
####多GPU训练模型加载
# 保存多GPU模型
torch.save(model.module.state_dict(), 'multigpu_model.pth')
# 加载到单GPU
model = ModelClass()
model.load_state_dict(torch.load('multigpu_model.pth'))


#######模型转换与兼容性
####PyTorch版本兼容性
# 保存时指定_use_new_zipfile_serialization=True以获得更好的兼容性
torch.save(model.state_dict(), 'model.pth', _use_new_zipfile_serialization=True)

####转换为TorchScript
# 将模型转换为TorchScript格式
scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_scripted.pt')
# 加载TorchScript模型
loaded_script = torch.jit.load('model_scripted.pt')