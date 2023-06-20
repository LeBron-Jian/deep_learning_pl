from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import os
from glob import glob
from typing import *
from PIL import Image
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import Caltech101
from sklearn.model_selection import train_test_split


class Caltech101Dataset(VisionDataset):
    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(os.path.join(root, "caltech-101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        if isinstance(target_type, str):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation")) for t in target_type]

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class


        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {
            "Faces": "Faces_2",
            "Faces_easy": "Faces_3",
            "Motorbikes": "Motorbikes_16",
            "airplanes": "Airplanes_Side_2",
        }
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index: List[int] = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(glob(os.path.join(self.root, "101_ObjectCategories", c, "*.jpg")))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        )

        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                print(data["obj_contour"])
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.index)


if __name__ == "__main__":
    root_path = "D:\workdata\data\caltech-101"
    # 创建自定义数据集对象
    dataset = Caltech101Dataset(root_path)
    # 可视化数据集中的第一个样本

    print(len(dataset))  # 8677

    # 划分训练集、验证集和测试集
    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    # print(train_val_indices)

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # 创建训练集、验证集和测试集的Subset对象
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # 创建DataLoader对象
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)