from typing import Any, Optional, Union, Tuple
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
                 label_list,
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
        # forward function that is run when visualizing the graph
        # 模型前向传递过程，主要是指val, test， 当然train也可以使用，保持代码统一
        x = self.model(x)
        # probability distribution over labels
        # x = F.log_softmax(x, dim=1)
        # x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def label_convert_onehot(self, y):
        return F.one_hot(y)

    def forward1(self, x):
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)
        return x

    def cross_entropy_loss(self, logits, labels):
        # create loss function module
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        # 单次训练过程， 相当于训练过程中处理一个batch的内容
        # x= train_batch['image']
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        # 单次验证过程，相当于验证过程中处理一个batch的内容
        # x = val_batch['image']
        # y = val_batch['label']
        # y = self.label_convert_onehot(y)
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, num_classes=len(self.labelset))
        # acc = accuracy(preds, y, self.task, num_classes=len(self.labelset))
        # calling self.log will surface up scalars for you in tensorboard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', acc, prog_bar=True)
        # return loss

    def test_step(self, test_batch, batch_idx):
        # x = test_batch['image']
        # y = test_batch['label']
        # y = self.label_convert_onehot(y)
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
