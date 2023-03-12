import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from data.dataset import BasicDataset
from classification import Classifier

root = r'D:\workdata\data\smp_data\OxfordPet\images'
val_percent = 0.2

# 首先设置随机数种子
pl.seed_everything(1234)

"""
# init train, val, test sets # 1. Create dataset
dataset = BasicDataset(root)


# 2. Split into train / validation partitions
n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))


# 3. Create data loaders
n_cpu = os.cpu_count()
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0) # num_workers=n_cpu
val_dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
# 设置好数据类，就可以使用Dataloader加载数据了
# for i_batch, batch_data in enumerate(train_dataloader):
#     print('打印 batch 编号： ', i_batch)
#     print('打印 batch 图片大小: ', batch_data['image'].size())
#     print('打印 batch 里面图片的标签', batch_data['label'])
"""

from torchvision import transforms
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# prepare transforms standard to MIST
minst_train = MNIST(root=r'D:\workdata\public\deep_learning_pl\tests\data', train=True, download=True,
                    transform=transform)
minst_test = MNIST(root=r'D:\workdata\public\deep_learning_pl\tests\data', train=False, download=True,
                   transform=transform)

train_dataloader = DataLoader(minst_train, batch_size=64)
val_dataloader = DataLoader(minst_test, batch_size=64)

# create mode
model = Classifier(label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
TASK_PATH = r'D:\workdata\data\smp_data\test_model_cache'
callback = pl.callbacks.ModelCheckpoint(monitor="val_accuracy", verbose=True, mode='max', dirpath=TASK_PATH)
trainer = pl.Trainer(
    default_root_dir=TASK_PATH,
    max_epochs=20,
    check_val_every_n_epoch=5,
    accelerator='gpu',
    devices=[1],
    callbacks=[callback])
trainer.fit(model, train_dataloader, val_dataloader)
# save trained model
# torch.save(model.state_dict(), config.MODEL_NAME)
# # test on test data
# trainer.test(model)