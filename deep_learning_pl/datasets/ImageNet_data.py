from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    def __init__(self, imagenet_dir, transform=None):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        # return 1000
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = {'image': self.dataset[idx][0]}
        return sample


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


def load_infinite(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt

    imagenet_dir = r'J:\dataset\public\ImageNet\train'
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512), ),
        transforms.RandomGrayscale(p=0.1),
        # 6: Convert Idist to gray scale with a probability of 0.1 and 18: Convert Idist to gray scale with a probability of 0.1
        transforms.ToTensor(),
    ])

    imagenet_dataset = ImageNetDataset(imagenet_dir, data_transforms)
    # dataloader = DataLoader(imagenet_dataset, batch_size=8, shuffle=True, num_workers=4,
    #                         pin_memory=True)

    # 显示第一个样本
    print("Total number of samples in the dataset:", len(imagenet_dataset))
    sample_image = imagenet_dataset[0]
    plt.imshow(sample_image.permute(1, 2, 0))  # 将张量转换为图像格式
    plt.title(f"Class: None")
    plt.show()
    # load data

    full_train_set = ImageFolderWithoutTarget(imagenet_dir, data_transforms)
    print("Total number of samples in the full_train_set:", len(full_train_set))
    train_loader = DataLoader(full_train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)

    import imgaug.augmenters as iaa
    import numpy as np
    import cv2

    # 创建仿射变换的增强器
    augmenter = iaa.Affine(rotate=(-45, 45), scale=(0.5, 1.5))