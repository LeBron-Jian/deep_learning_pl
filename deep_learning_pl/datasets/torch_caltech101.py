import torchvision.transforms as transforms
from torchvision.datasets import Caltech101
from torch.utils.data import DataLoader

# 设置数据集路径和转换
root = r"D:\workdata\data\caltech-101\caltech101"  # 设置数据集的根目录路径
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

# 创建 Caltech-101 数据集对象
dataset = Caltech101(root=root, transform=transform, download=True)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 使用数据加载器获取数据
images, labels = next(iter(train_loader))

# 打印数据的形状和标签
print(f"图像张量的形状: {images.shape}")
print(f"标签张量的形状: {labels.shape}")
print(f"标签列表: {labels}")

# 显示图像等处理...
