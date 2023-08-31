import torch
import torchvision
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models import Wide_ResNet101_2_Weights

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm


class WideResNet(ResNet):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, target_dim=384):
        super(WideResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                         groups, width_per_group, replace_stride_with_dilation,
                                         norm_layer)
        self.target_dim = target_dim

    def _forward_impl(self, x):
        x = imagenet_norm_batch(
            x)  # Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x0 = self.layer1(x)
        x1 = self.layer2(x0)
        x2 = self.layer3(x1)
        # pdb.set_trace()
        ret = self._proj(x1, x2)
        # x3 = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return ret

    def _proj(self, x1, x2):
        # [2, 512, 64, 64]->[2, 512, 64, 64],[2, 1024, 32, 32]->[2, 1024, 64, 64]
        # cat [2, 512, 64, 64],[2, 1024, 64, 64]->[2, 1536, 64, 64]
        # pool [2, 1536, 64, 64]->[2, 384, 32, 32]
        b, c, h, w = x1.shape
        x2 = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        features = torch.cat([x1, x2], dim=1)
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w)
        features = features.transpose(1, 2)
        target_features = F.adaptive_avg_pool1d(features, self.target_dim)
        # pdb.set_trace()
        target_features = target_features.transpose(1, 2)
        target_features = target_features.reshape(b, self.target_dim, h, w)
        return target_features


def _resnet(url, block, layers, pretrained, progress, **kwargs):
    model = WideResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model


def wide_resnet101_2(arch, pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    url = torchvision.models.get_weight(arch).url
    return _resnet(url, Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


if __name__ == "__main__":
    # Wide_ResNet101_2_Weights.IMAGENET1K_V1
    # print(summary(wide_resnet101_2("Wide_ResNet101_2_Weights.IMAGENET1K_V2").cuda(), (3, 512, 512)))
    def load_pretrain(self):
        self.pretrain = wide_resnet101_2(self.wide_resnet_101_arch, pretrained=True)
        # self.pretrain.load_state_dict(torch.load('pretrained_model.pth'))
        self.pretrain.eval()
        self.pretrain = self.pretrain.cuda()
        # print(summary(self.pretrain, (3, 512, 512)))
