'''
Copyright (C) 2022 Jian

'''
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
    models_names = timm.list_models(pretrained=True)
    print('支持的预训练模型数量：%s'%len(models_names))
    strs = "*mobile*"
    select_models = timm.list_models(strs)
    print('通过通配符 %s 查询到的可用模型： %s'%(strs, len(select_models)))
    print(select_models)

    # 测试一个预训练模型, 搭建迁移学习模型库
    m = timm.create_model('resnet18', pretrained=True, num_classes=13)
    m.eval()
    o = m(torch.randn(2, 3, 224, 224))
    print('classifcation layer shape is ', o.shape)
    # 输出flatten或者 global_pool层的前一层的数据(flatten, global_pool曾通常接分类层)
    o = m.forward_features(torch.randn(2, 3, 224, 224))
    print('feature shape is ', o.shape)    

    m1 = timm.create_model('resnet18', features_only=True, output_stride=8, out_indices=(0, 1, 2, 3, 4),
        pretrained=True, num_classes=13)
    print(f'Feature channels: {m1.feature_info.channels()}')
    print(f'Feature reduction: {m1.feature_info.reduction()}')
    o1 = m1(torch.randn(2, 3, 224, 224))
    # for x in o:
    #     print(x.shape)
    
    backbone, num_features = _fn_timm(model_name='resnet18', pretrained=True, num_classes=12)
    print('===================')
    backbone.eval()
    res = backbone(torch.randn(2, 3, 224, 224))
    print('classifcation layer shape is ', res.shape)
