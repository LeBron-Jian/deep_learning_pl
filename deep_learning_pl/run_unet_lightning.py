import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from .unet.model import UNet
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


import random

DEFAULT_VOID_LABELS = (0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1)
DEFAULT_VALID_LABELS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)

# Sweep parameters
hyperparameter_defaults = dict(
    data_path='data_semantics',
    batch_size=2,
    lr=1e-3,
    num_layers=5,
    features_start=64,
    bilinear=False,
    grad_batches=1,
    epochs=20
)

wandb.init(config=hyperparameter_defaults)
# Config parameters are automatically set by W&B sweep agent
config = wandb.config


class KITTI(Dataset):
    '''
    Dataset Class for KITTI Semantic Segmentation Benchmark dataset
    Dataset link - http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

    There are 34 classes in the given labels. However, not all of them are useful for training
    (like railings on highways, road dividers, etc.).
    So, these useless classes (the pixel values of these classes) are stored in the `void_labels`.
    The useful classes are stored in the `valid_labels`.

    The `encode_segmap` function sets all pixels with any of the `void_labels` to `ignore_index`
    (250 by default). It also sets all of the valid pixels to the appropriate value between 0 and
    `len(valid_labels)` (since that is the number of valid classes), so it can be used properly by
    the loss function when comparing with the output.

    The `get_filenames` function retrieves the filenames of all images in the given `path` and
    saves the absolute path in a list.

    In the `get_item` function, images and masks are resized to the given `img_size`, masks are
    encoded using `encode_segmap`, and given `transform` (if any) are applied to the image only
    (mask does not usually require transforms, but they can be implemented in a similar way).
    '''
    IMAGE_PATH = os.path.join('training', 'image_2')
    MASK_PATH = os.path.join('training', 'semantic')

    def __init__(
            self,
            data_path,
            split,
            img_size=(1242, 376),
            void_labels=DEFAULT_VOID_LABELS,
            valid_labels=DEFAULT_VALID_LABELS,
            transform=None
    ):
        self.img_size = img_size
        self.void_labels = void_labels
        self.valid_labels = valid_labels
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(len(self.valid_labels))))
        self.transform = transform

        self.split = split
        self.data_path = data_path
        self.img_path = os.path.join(self.data_path, 'training/image_2')
        self.mask_path = os.path.join(self.data_path, 'training/semantic')
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)

        # Split between train and valid set
        random_inst = random.Random(12345)  # for repeatability
        n_items = len(self.img_list)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        if self.split == 'train': idxs = [idx for idx in range(n_items) if idx not in idxs]
        self.img_list = [self.img_list[i] for i in idxs]
        self.mask_list = [self.mask_list[i] for i in idxs]

    def __len__(self):
        return (len(self.img_list))

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = img.resize(self.img_size)
        img = np.array(img)

        mask = Image.open(self.mask_list[idx]).convert('L')
        mask = mask.resize(self.img_size)
        mask = np.array(mask)
        mask = self.encode_segmap(mask)

        if self.transform:
            img = self.transform(img)

        return img, mask

    def encode_segmap(self, mask):
        '''
        Sets void classes to zero so they won't be considered for training
        '''
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        # remove extra idxs from updated dataset
        mask[mask > 18] = self.ignore_index
        return mask

    def get_filenames(self, path):
        '''
        Returns a list of absolute paths to images inside given `path`
        '''
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list


class SegModel(pl.LightningModule):
    '''
    Semantic Segmentation Module

    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It uses the FCN ResNet50 model as an example.

    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    '''

    def __init__(self, hparams):
        super().__init__()
        self.lr = hparams.lr
        self.net = UNet(num_classes=19, num_layers=hparams.num_layers,
                        features_start=hparams.features_start, bilinear=hparams.bilinear)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log('train_loss', loss_val)  # log training loss
        return loss_val

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        self.log('val_loss', loss_val)  # will be automatically averaged over an epoch

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]


class KittiDataModule(pl.LightningDataModule):
    '''
    Kitti Data Module

    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    '''

    def __init__(self, hparams):
        super().__init__()
        print(hparams)
        self.data_path = hparams.data_path
        self.batch_size = hparams.batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])

    def setup(self, stage=None):
        self.trainset = KITTI(self.data_path, split='train', transform=self.transform)
        self.validset = KITTI(self.data_path, split='valid', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)


def main(config):
    # ------------------------
    # 1 LIGHTNING MODEL
    # ------------------------
    model = SegModel(config)

    # ------------------------
    # 2 DATA PIPELINES
    # ------------------------
    kittiData = KittiDataModule(config)

    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------
    wandb_logger = WandbLogger()

    # optional: log model topology
    wandb_logger.watch(model.net)

    # ------------------------
    # 4 TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=-1,
        logger=wandb_logger,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.grad_batches,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model, kittiData)


if __name__ == '__main__':
    print(f'Starting a run with {config}')
    main(config)
