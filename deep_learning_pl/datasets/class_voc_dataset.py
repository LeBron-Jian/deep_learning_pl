"""
python
    PascalVOCDataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import transforms
import cv2
import xml.etree.ElementTree as ET
import torchvision.datasets.voc as voc

dict_classes = {'__background__': 0,
                'aeroplane': 1,
                'bicycle': 2,
                'bird': 3,
                'boat': 4,
                'bottle': 5,
                'bus': 6,
                'car': 7,
                'cat': 8,
                'chair': 9,
                'cow': 10,
                'diningtable': 11,
                'dog': 12,
                'horse': 13,
                'motorbike': 14,
                'person': 15,
                'pottedplant': 16,
                'sheep': 17,
                'sofa': 18,
                'train': 19,
                'tvmonitor': 20
                }


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, root_path, transforms=None):
        super().__init__()
        self.root_path = root_path
        self.img_idx = []
        self.anno_idx = []
        self.transforms = transforms
        if self.transforms is not None:
            train_txt_path = self.root_path + "/ImageSets/Layout/" + "train.txt"
        else:
            train_txt_path = self.root_path + "/ImageSets/Layout/" + "val.txt"
        self.img_path = self.root_path + "/JPEGImages/"
        self.anno_path = self.root_path + "/Annotations/"

        train_txt = open(train_txt_path, 'r')
        lines = train_txt.readlines()
        for line in lines:
            name = line.strip().split()[0]
            self.img_idx.append(self.img_path + name + '.jpg')
            self.anno_idx.append(self.anno_path + name + '.xml')

    def __getitem__(self, item):
        img = cv2.imread(self.img_idx[item])
        targets = ET.parse(self.anno_idx[item])
        objs = targets.findall('object')
        boxes_cl = np.zeros((len(objs)), dtype=np.int32)
        for ix, obj in enumerate(targets.iter('object')):
            name = obj.find('name').text.lower().strip()
            boxes_cl[ix] = dict_classes[name]

        if self.transforms is not None:
            img = self.transforms(img)
        labels = []
        lbl = np.zeros(len(dict_classes))
        lbl[boxes_cl] = 1
        labels.append(lbl)
        return img, np.array(labels).astype(np.float32)

    def __len__(self):
        return len(self.img_idx)


if __name__ == "__main__":
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    Voc_data_path = r"D:\workdata\data\VOC\VOC2012"
    train_data = PascalVOCDataset(root_path=Voc_data_path)
    train_loader = DataLoader(train_data, batch_size=5, shuffle=False)

    # select the image index to view
    image_index = 0
    # Obtain images and labels for the specified index
    image, label = train_data[image_index]
    print("Label:", label)

    for image_index in range(2):
        # Obtain images and labels for the specified index
        image, label = train_data[image_index]
        print("Label:", label)

    # 遍历 DataLoader 中的数据
    for batch in train_loader:
        # 在这里可以查看每个批次的数据
        inputs, labels = batch  # 假设每个样本包含输入和标签
        # 打印批次的大小和数据内容
        print("Batch Size:", inputs.size(0))
        print("Inputs:", inputs)
        print("Labels:", labels)
        break
