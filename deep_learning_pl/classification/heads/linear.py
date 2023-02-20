import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearHead(nn.Module):

    def __init__(self, num_features: int, num_classes: int, pool_type: str = "avg"):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.pool_type = pool_type

        if pool_type == "cat":
            self.num_features = num_features * 2

        self.linear = nn.Linear(self.num_features, num_classes)

    def forward(self, x: Tensor):
        if self.pool_type == "avg":
            x = F.adaptive_avg_pool2d(x, output_size=1)
        elif self.pool_type == "max":
            x = F.adaptive_max_pool2d(x, output_size=1)
        elif self.pool_type == "cat":
            x1 = F.adaptive_avg_pool2d(x, output_size=1)
            x2 = F.adaptive_max_pool2d(x, output_size=1)
            x = torch.cat([x1, x2], dim=1)
        if x.dim() == 4:
            x = x[:, :, 0, 0]
        return self.linear(x)
