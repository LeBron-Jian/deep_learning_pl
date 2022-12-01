import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from data.dataset import BasicDataset
from classification.classification import Classifier

root = r'D:\Desktop\workdata\data\segmentation_dataset\images'
val_percent = 0.2

# 这个是用于固定seed用
pl.seed_everything(1234)

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

# create mode
model = Classifier()

lr_monitor = LearningRateMointor(logging_interval='step')

trainer = pl.Trainer(callbacks=[lr_monitor], max_epochs=10, num_sanity_val_steps=2)
trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, dataloaders=val_dataloader, verbose=False)