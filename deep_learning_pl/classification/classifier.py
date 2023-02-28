from typing import Any, Optional, Union, Tuple, List
from argparse import ArgumentParser
from types import FunctionType

import torch
from torch.nn import functional as F
from torchmetrics import Metric, F1Score, Accuracy
from torch import nn
from torchmetrics.functional.classification.accuracy import accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .backbones.cls_timm import _fn_timm
from .heads.linear import LinearHead


class Classifier(pl.LightningModule):
    def __init__(self,
                 label_list: List,
                 model_name: Union[str, Tuple[nn.Module, int]] = "resnet18",
                 pretrained: Union[bool, str] = True,
                 head: Union[str, FunctionType, nn.Module] = "linear",
                 ):
        """
            Inputs:
                model_name - Name of the model/CNN to run used for creating the model(see function below)

        """
        super().__init__()
        # Exports the huperparameters to a YAML file, and create 'self.hparams' namespace
        self.save_hyperparameters()

        assert label_list is not None, "The `label_list` can not be None"

        self.labelset = label_list
        if len(self.labelset) == 2:
            self.task = 'binary'
        else:
            self.task = 'multiclass'

        num_classes = len(self.labelset)

        # get create_model function
        backbone, num_features = _fn_timm(model_name, pretrained, num_classes)
        self.model = backbone
        # self.head = nn.Linear(num_features, num_classes)
        self.head = LinearHead(num_features, num_classes)

        # example input for visualizing the graph in tensorboard
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x

    def label_convert_onehot(self, y):
        return F.one_hot(y)

    def cross_entropy_loss(self, logits, labels):
        # create loss function module
        # F.nll_loss在函数内部不含有提前使用softmax转化的部分；
        # return F.nll_loss(logits, labels)
        # nn.CrossEntropyLoss内部先将输出使用softmax方式转化为概率的形式，后使用F.nll_loss函数计算交叉熵。
        return nn.CrossEntropyLoss()(logits, labels)

    def training_step(self, train_batch, batch_idx):
        # 单次训练过程， 相当于训练过程中处理一个batch的内容
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=len(self.labelset))
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        # return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser
