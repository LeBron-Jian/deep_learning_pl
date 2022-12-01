'''
Copyright (C) 2022 JABIL

'''
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from pl_wang.classification.classifier import Classifier


def test_instantiate():
    data_info = DatasetInfo(["a", "b", "c"])

    model = Classifier(data_info)


def test_get_from_registry():
    print(Classifier.get_head_from_registry("linear"))
    print(Classifier.get_backbone_from_registry("resnet18"))


def test_available_registry():
    assert len(Classifier.available_backbones()) >= 1
    assert len(Classifier.available_heads()) >= 1
    print(Classifier.available_backbones())
    print(Classifier.available_heads())


def test_optimizers_with_schedulers():

    data_info = DatasetInfo(["a", "b", "c"])
    trainer = pl.Trainer()
    for optim in Classifier.available_optimizers():
        for sch in Classifier.available_lr_schedulers():
            print(optim, sch)
            try:
                model = Classifier(data_info, optimizer=optim,
                                   lr_scheduler=sch,
                                   learning_rate=1e-3)
                model.trainer = trainer
                model.configure_optimizers()
            except Exception as e:
                print(e)



def test_train_and_predict():
    labels = ["a", "b", "c", "d"]
    size = 10
    data_info = DatasetInfo(labels, multi_label=False, img_channel=1)

    model = Classifier(data_info)

    class MockDataset(Dataset):

        def __getitem__(self, idx):
            return torch.randn(1, 32, 32), {"labels": list(self._generate_labels())}

        def _generate_labels(self):
            return np.random.choice(labels, 1)

        def __len__(self):
            return size

    def collate_fn(batch):
        images = [bi[0] for bi in batch]
        labels = [bi[1] for bi in batch]
        return torch.stack(images), labels

    train_dataloader = DataLoader(MockDataset(), batch_size=2, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(MockDataset(), batch_size=2, collate_fn=collate_fn)
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_dataloader, test_dataloader)

    trainer.model.freeze()
    output = trainer.model.predict(torch.randn(1, 1, 32, 32))
    assert output.shape == (1, len(labels))

