from typing import Any, Optional, Union
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from torch import nn
from torchmetrics.functional.classification.accuracy import accuracy

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import  STEP_OUTPUT

from backbones.cls_timm import _fn_timm


class Classifier(pl.LightningModule):
    def __init__(self,
                 model_name: Union[str, Tuple[nn.Module, int]] = "resnet18",
                 pretrained: Union[bool, str] = True,
                 num_classes: int = 0,  
        ):
        """
            Inputs:
                model_name - Name of the model/CNN to run used for creating the model(see function below)

        """
        super().__init__()
        # Exports the huperparameters to a YAML file, and create 'self.hparams' namespace
        self.save_hyperparameters()

        # get create_model function
        backbone, num_features = _fn_timm()
        self.model = backbone

        # create loss function module
        self.loss_module = nn.CrossEntropyLoss()
        
        # example input for visualizing the graph in tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, x):
        # forward function that is run when visualizing the graph
        # 模型前向传递过程，主要是指val, test， 当然train也可以使用，保持代码统一
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e3)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # 单次训练过程， 相当于训练过程中处理一个batch的内容
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, x)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # 单次验证过程，相当于验证过程中处理一个batch的内容
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # calling self.log will surface up scalars for you in tensorboard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser