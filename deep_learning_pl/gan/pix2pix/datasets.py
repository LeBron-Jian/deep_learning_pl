import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, image_postfix='jpg'):
        self.root = root
        self.transform = transforms.Compose(transforms_)

        self.files = glob.glob(os.path.join(root, "*.%s")%image_postfix)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        w, h = img.size
        img_A = img.crop((0, 0, int(w / 2), h))
        img_B = img.crop((int(w / 2), 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


if __name__ == "__main__":
    root_path = r"D:\Desktop\workdata\public\code_paper\Pytorch-UNet-master\data\imgs"
    # Configure dataloaders
    transforms_ = [
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = ImageDataset(root_path, transforms_)
    print(len(dataset))
    # {"A": img_A, "B": img_B}
    print(dataset[0]['A'].size())
