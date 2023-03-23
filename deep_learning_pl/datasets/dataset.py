import logging
import os
from os import listdir
from pathlib import Path
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from deep_learning_pl.utils import *


class BasicDataset:
    classes = {
        "cat": 0,
        "dog": 1
    }

    def __init__(
            self,
            images_dir,
            image_size=(224, 224),
            transform=None,
    ):
        # initialize file path or list of file names
        assert images_dir is not None

        # 文件目录
        self.images_dir = images_dir
        self.imgs = os.listdir(self.images_dir)
        self.image_size = image_size
        # 变换
        self.transform = transform

    def __getitem__(self, idx):
        # 1,read one datasets from file (e.g using numpy.fromfile, PIL.Image.open)
        # 2, preprocess the datasets (e.g torchvision.Transform)
        # 3, return a datasets pair (e.g image and label)

        # 根据索引index获取该图片地址
        image_index = self.imgs[idx]
        # 获取索引为index的图片的路径名
        if isinstance(image_index, (str, Path)):
            img = os.path.join(self.images_dir, image_index)
            img = open_img(img)
            img = self.preprocess(img)
        if image_index[0].isupper():
            label = self.classes['dog']
        else:
            label = self.classes['cat']
        # 根据图片和标签创建字典
        sample = dict(image=img, label=label)

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        # we need change 0 to the total size of your dataset
        return len(self.imgs)

    def preprocess(cls, pil_img, is_mask=False):
        pil_img = pil_img.resize(cls.image_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)
        if len(img_ndarray.shape) == 2:
            img_ndarray = np.expand_dims(img_ndarray, axis=0)
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))
        if img_ndarray.max() > 1:
            img_ndarray = img_ndarray / 255
        return img_ndarray.astype(np.float32)
