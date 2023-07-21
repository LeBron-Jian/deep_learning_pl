from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):
    def __init__(self, imagenet_dir, transform=None):
        super().__init__()
        self.imagenet_dir = imagenet_dir
        self.transform = transform
        self.dataset = ImageFolder(self.imagenet_dir, transform=self.transform)

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        return self.dataset[idx][0]


def load_infinite(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
