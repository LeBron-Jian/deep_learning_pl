from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

# VOC数据集分类对应颜色标签
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]


# 颜色标签空间转到序号标签空间，就他妈这里浪费巨量的时间,这里还他妈的有问题
def voc_label_indices(colormap, colormap2label):
    """Assign label indices for Pascal VOC2012 Dataset."""
    idx = ((colormap[:, :, 2] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 0])
    # out = np.empty(idx.shape, dtype = np.int64)
    out = colormap2label[idx]
    out = out.astype(np.int64)  # 数据类型转换
    end = time.time()
    return out


class VOCDataset(data.Dataset):  # 创建自定义的数据读取类
    def __init__(self, root, is_train, crop_size=(320, 480)):
        self.rgb_mean = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)
        self.root = root
        self.crop_size = crop_size
        images = []  # 创建空列表存文件名称
        txt_fname = '%s/ImageSets/segmentation/%s' % (root, 'train.txt' if is_train else 'val.txt')
        with open(txt_fname, 'r') as f:
            self.images = f.read().split()
        # 数据名称整理
        self.files = []
        for name in self.images:
            img_file = os.path.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = os.path.join(self.root, "SegmentationClass/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.colormap2label = np.zeros(256 ** 3)
        # 整个循环的意思就是将颜色标签映射为单通道的数组索引
        for i, cm in enumerate(VOC_COLORMAP):
            self.colormap2label[(cm[2] * 256 + cm[1]) * 256 + cm[0]] = i

    # 按照索引读取每个元素的具体内容
    def __getitem__(self, index):

        datafiles = self.files[index]
        name = datafiles["name"]
        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["label"]).convert('RGB')  # 打开的是PNG格式的图片要转到rgb的格式下，不然结果会比较要命
        # 以图像中心为中心截取固定大小图像，小于固定大小的图像则自动填0
        imgCenterCrop = transforms.Compose([
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.rgb_mean, self.rgb_std),  # 图像数据正则化
        ])
        labelCenterCrop = transforms.CenterCrop(self.crop_size)
        cropImage = imgCenterCrop(image)
        croplabel = labelCenterCrop(label)
        croplabel = torch.from_numpy(np.array(croplabel)).long()  # 把标签数据类型转为torch

        # 将颜色标签图转为序号标签图
        mylabel = voc_label_indices(croplabel, self.colormap2label)

        return cropImage, mylabel

    # 返回图像数据长度
    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    # VOC数据集分类对应颜色标签
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]

    root = '../datasets/VOCdevkit/VOC2012'
    train_data = VOCDataset(root, True)
    trainloader = data.DataLoader(train_data, 4)

    # 从数据集中拿出一个批次的数据
    for i, data in enumerate(trainloader):
        getimgs, labels = data
        img = transforms.ToPILImage()(getimgs[0])

        labels = labels.numpy()  # tensor转numpy
        labels = labels[0]  # 获得批次标签集中的一张标签图像
        labels = labels.transpose((1, 0))  # 数组维度切换，将第1维换到第0维，第0维换到第1维

        ##将单通道索引标签图片映射回颜色标签图片
        newIm = Image.new('RGB', (480, 320))  # 创建一张与标签大小相同的图片，用以显示标签所对应的颜色
        for i in range(0, 480):
            for j in range(0, 320):
                sele = labels[i][j]  # 取得坐标点对应像素的值
                newIm.putpixel((i, j),
                               (int(VOC_COLORMAP[sele][0]), int(VOC_COLORMAP[sele][1]), int(VOC_COLORMAP[sele][2])))

        # 显示图像和标签
        plt.figure("image")
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        plt.sca(ax1)
        plt.imshow(img)
        plt.sca(ax2)
        plt.imshow(newIm)
        plt.show()
