from functools import partial
from typing import Tuple

import timm
import torch
from torch import nn


class _TimmBackbone(nn.Module):

    def __init__(self, timm_backbone: nn.Module):
        super().__init__()
        self.backbone = timm_backbone
    
    def forward(self, x):
        return self.backbone.forward_features(x)


def _fn_timm(model_name: str,
             pretrained: bool = True,
             num_classes: int = 0,  
             **kwargs) -> Tuple[nn.Module, int]:
    # 如果不设置num_classes， 则表示使用的是原始的预训练模型的分类层
    # 当搭建迁移学习的时候，需要重新设置num_classes, 表示重新设全连接层
    # features_only=True,表示只输出特征层，但当使用了output_stride或out_indices时，features_true必须为True
    # output_stride 控制最后一个特征层layer的dilated，一些网络只支持output_stride=32
    # out_indices：选定特征层所在的index
    backbone = timm.create_model(model_name, pretrained, num_classes, **kwargs)
    num_features = backbone.num_features
    backbone = _TimmBackbone(backbone)
    return backbone, num_features


if __name__ == "__main__":

    