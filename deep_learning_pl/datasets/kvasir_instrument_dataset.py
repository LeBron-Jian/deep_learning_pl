"""
Kvasir-SEG分段息肉数据集
数据集内容：
    结直肠癌是女性中第二大最常见的癌症类型，并且是男性中第三大最常见的癌症。息肉是结肠直肠癌的前体，因此对于早期发现和清除很重要
    息肉的检测已被证明可以降低结直肠癌的风险。因此，早期自动检测更多的息肉对预防和生存大肠癌起着至关重要的作用。
    这是开发息肉分割数据集的主要动机。
数据集数量：
    Kvasir-SEG数据集基于先前的Kvasir数据集，该数据集是第一个用于胃肠道（GI）疾病检测和分类的多类数据集。
    原始的Kvasir数据集包含来自8个类别的8,000个GI道图像，每个类别由1000个图像组成。
    而Kvasir-SEG分段息肉数据集用新图像替换了息肉类的13张图像，以提高数据集的质量。

    kvasir是一个用于胃镜图像分析的开源数据集。


"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Union
from PIL import Image
from torch.utils.data import Dataset


class KvasirDataset(Dataset):
    def __init__(self, images_dir: Union[str, Path], masks_dir: Union[str, Path], size: tuple):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = size

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Reading images and obtaining labels
        img_file = os.path.join(self.images_dir, str(self.ids[idx]) + '.jpg')
        mask_file = os.path.join(self.masks_dir, str(self.ids[idx]) + '.png')

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        # convert images toPytorch Tensor，And normalize the image pixels to the range of [0,1]
        img = self.preprocess(img, self.size, is_mask=False)
        mask = self.preprocess(mask, self.size, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


    @staticmethod
    def preprocess(pil_img, resize, is_mask):
        newW, newH = resize
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255
        else:
            if img_ndarray.shape[-1] == 3:
                img_ndarray = np.mean(img_ndarray, axis=2).astype(np.uint8)

        return img_ndarray


if __name__ == "__main__":
    dir_img = Path(r'D:\Desktop\workdata\public\kvasir-instrument/images/')
    dir_mask = Path(r'D:\Desktop\workdata\public\kvasir-instrument/masks/')
    dataset = KvasirDataset(dir_img, dir_mask, size=(416, 416))
    print(len(dataset))

    print(dataset[0]["image"].size(), dataset[0]["mask"].size())
    img = dataset[0]["image"]
    label = dataset[0]["mask"]
    res1 = img.permute(1, 2, 0)
    print(res1.size())

    # show image
    from deep_learning_pl.utils.images import show_img_from_dataset

    show_img_from_dataset(dataset)



