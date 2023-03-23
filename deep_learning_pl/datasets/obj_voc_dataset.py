"""python
    PascalVOCDataset具体实现过程
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2  # 使用PIL进行图片处理也可以
import xml.etree.ElementTree as ET  # 用来解析.xml文件

#  存储Voc数据集中的类别标签的字典 没打全
Voc_label = {'aeroplane', 'bycycle', 'bird', 'boat', 'bottle', '...'}

dict_classes = dict(zip(Voc_label, range(len(Voc_label))))


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    #初始化相关变量
    #读取images和objects标注信息
    def __init__(self, root_path):
        super().__init__
        self.root_path = root_path

        self.img_idx = []
        self.anno_idx = []
        self.bbox = []
        self.obj_name = []
        train_txt_path = self.root_path + "/ImageSets/存放用于Training 的图片的名称的txt文件"
        self.img_path = self.root + "/JPEGImage/存放.jpg图片的地址"
        self.anno_path = self.root + "/Annotations/存放annotation标注.xml文件的地址"

        train_txt = open(train_txt_path)
        lines = train_txt.readline()
        for line in lines:
            name = line.strip().split()[0]
            self.img_idx.append(self.img_path + name + '.jpg')
            self.ano_idx.append(self.ano_path + name + '.xml')


    def __getitem__(self, item):
        img = cv2.imread(self.img_idx[item])
        height, width, channels = img.shape
        targets = ET.parse(self.ano_idx[item])
        res = []  # 存储标注信息 即边框左上角和右下角的四个点的坐标信息和目标的类别标签
        for obj in targets.iter('object'):
            name = obj.find('name').text.lower().strip()
            class_idx = dict_classes[name]
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            obj_bbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt))
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height  # 将坐标做一个线性变换
                obj_bbox.append(cur_pt)
            res.append(obj_bbox)
            res.append(class_idx)
        return img, res


    def __length__(self):
        data_length = len(self.img_idx)
        return data_length


if __name__ == "__main__":
    #  开始调用 Read_data类读取数据，并使用Dataloader生成迭代数据为送入模型中做准备
    Voc_data_path = ' Voc数据集地址'
    train_data = PascalVOCDataset(root_path=Voc_data_path)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=False)

    #  可以这样理解：自己定义的Read_data负读取数据，而DataLoader负责按照定义的batch_size指派Read_data去读取
    # 指定数目的数据，然后再进行相应的拼接等其它内部操作。




