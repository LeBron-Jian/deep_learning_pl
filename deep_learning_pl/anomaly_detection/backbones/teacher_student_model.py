from torch import nn
import torch.nn.functional as F


class PDN_S(nn.Module):
    """
    . Patch description network architecture of the teacher network for EfficientAD-S.(small)
    the student network has the same architecture, but 768 kernels instead of 384 in the
    conv-4 layer.

        layerName    stride    kernelSize  numberOfKernels  Padding     Activation
        Conv-1        1*1       4*4          128              3             ReLu
        AvgPool-1     2*2       2*2          128              1              -
        Conv-2        1*1       4*4          256              3             ReLu
        AvgPool-2     2*2       2*2          256              1              -
        Conv-3        1*1       4*4          256              3             ReLu
        Conv-4        1*1       4*4          384              0             -
    """

    def __init__(self, last_kernel_size=384, with_bn=False, *args, **kwargs) -> None:
        super().__init__()
        # FIXME  input 3???  in_channels=3, out_channels=128
        self.with_bn = with_bn
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 384, kernel_size=4, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(128)
            self.bn2 = nn.BatchNorm2d(256)
            self.bn3 = nn.BatchNorm2d(256)
            self.bn4 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        # 当 inplace=True 时，ReLU 函数将直接修改输入张量，将负值元素置为零，并返回修改后的张量。
        # 这意味着原始输入张量的值会被更改，节省了额外的内存开销，但也可能导致梯度计算或回溯方面的问题
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        return x


class PDN_M(nn.Module):
    """
    . Patch description network architecture of the teacher network for EfficientAD-M.(middle)
    the student network has the same architecture, but 768 kernels instead of 384 in the
    conv-5 and conv-6 layer.

        layerName    stride    kernelSize  numberOfKernels  Padding     Activation
        Conv-1        1*1       4*4          256              3             ReLu
        AvgPool-1     2*2       2*2          256              1              -
        Conv-2        1*1       4*4          512              3             ReLu
        AvgPool-2     2*2       2*2          512              1              -
        Conv-3        1*1       1*1          512              0             ReLu
        Conv-4        1*1       3*3          512              1             ReLu
        Conv-5        1*1       4*4          384              0             ReLu
        Conv-6        1*1       1*1          384              0             -
    """

    def __init__(self, last_kernel_size=384, with_bn=False) -> None:
        super().__init__()
        self.with_bn = with_bn
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 384, kernel_size=4, stride=1, padding=0)
        self.conv6 = nn.Conv2d(384, 384, kernel_size=1, stride=1, padding=0)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(256)
            self.bn2 = nn.BatchNorm2d(512)
            self.bn3 = nn.BatchNorm2d(512)
            self.bn4 = nn.BatchNorm2d(512)
            self.bn5 = nn.BatchNorm2d(last_kernel_size)
            self.bn6 = nn.BatchNorm2d(last_kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.with_bn else x
        x = F.relu(x)
        x = self.avgpool2(x)
        x = self.conv3(x)
        x = self.bn3(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn4(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x) if self.with_bn else x
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x) if self.with_bn else x
        return x


class EncConv(nn.Module):
    """
    Network architecture of the autoencoder for EfficientAD-S and EfficientAD-M,
    Layers named "EncConv" and "DecConv" are standard 2D convolutional layers.

        layerName    stride    kernelSize  numberOfKernels  Padding     Activation
        EncConv-1     2×2         4×4          32             1           ReLU
        EncConv-2     2×2         4×4          32             1           ReLU
        EncConv-3     2×2         4×4          64             1           ReLU
        EncConv-4     2×2         4×4          64             1           ReLU
        EncConv-5     2×2         4×4          64             1           ReLU
        EncConv-6     1×1         8×8          64             0           -
    """

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class DecConv(nn.Module):
    """
        Network architecture of the autoencoder for EfficientAD-S and EfficientAD-M,
    Layers named "EncConv" and "DecConv" are standard 2D convolutional layers.

        layerName    stride    kernelSize  numberOfKernels  Padding     Activation
        Bilinear-1    Resizes the 1×1 input features maps to 3×3
        DecConv-1    1×1         4×4            64            2             ReLU
        Dropout-1    Dropout rate = 0.2
        Bilinear-2    Resizes the 4×4 input features maps to 8×8
        DecConv-2    1×1         4×4            64             2            ReLU
        Dropout-2    Dropout rate = 0.2
        Bilinear-3    Resizes the 9×9 input features maps to 15×15
        DecConv-3    1×1         4×4            64            2             ReLU
        Dropout-3    Dropout rate = 0.2
        Bilinear-4    Resizes the 16×16 input features maps to 32×32
        DecConv-4    1×1         4×4            64            2             ReLU
        Dropout-4    Dropout rate = 0.2
        Bilinear-5    Resizes the 33×33 input features maps to 63×63
        DecConv-5    1×1         4×4            64           2              ReLU
        Dropout-5    Dropout rate = 0.2
        Bilinear-6    Resizes the 64×64 input features maps to 127×127
        DecConv-6    1×1         4×4            64           2              ReLU
        Dropout-6    Dropout rate = 0.2
        Bilinear-7    Resizes the 128×128 input features maps to 64×64
        DecConv-7    1×1         3×3            64           1              ReLU
        DecConv-8    1×1        3×3            384           1                -

    """

    def __init__(self, is_bn=False, *args, **kwargs) -> None:
        super().__init__()
        self.is_bn = is_bn
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, 384, kernel_size=3, stride=1, padding=1)
        if self.is_bn:
            self.dropout1 = nn.BatchNorm2d(64)
            self.dropout2 = nn.BatchNorm2d(64)
            self.dropout3 = nn.BatchNorm2d(64)
            self.dropout4 = nn.BatchNorm2d(64)
            self.dropout5 = nn.BatchNorm2d(64)
            self.dropout6 = nn.BatchNorm2d(64)
        else:
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)
            self.dropout4 = nn.Dropout(p=0.2)
            self.dropout5 = nn.Dropout(p=0.2)
            self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=3, mode='bilinear')
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to('cuda')
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to('cuda')
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm


class Teacher(nn.Module):
    def __init__(self, size, with_bn=False, channel_size=384, *args, **kwargs) -> None:
        super().__init__()
        if size == 'M':
            self.pdn = PDN_M(last_kernel_size=channel_size, with_bn=with_bn)
        elif size == 'S':
            self.pdn = PDN_S(last_kernel_size=channel_size, with_bn=with_bn)
        # self.pdn.apply(weights_init)

    def forward(self, x):
        x = imagenet_norm_batch(
            x)  # Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = self.pdn(x)
        return x


class Student(nn.Module):
    def __init__(self, size, with_bn=False, channel_size=768, *args, **kwargs) -> None:
        super().__init__()
        if size == 'M':
            # The student network has the same arch,but 768 kernels instead of 384 in the Conv-5 and Conv-6 layers.
            self.pdn = PDN_M(last_kernel_size=channel_size,
                             with_bn=with_bn)
        elif size == 'S':
            # The student network has the same architecture, but 768 kernels instead of 384 in the Conv-4 layer
            self.pdn = PDN_S(last_kernel_size=channel_size,
                             with_bn=with_bn)
            # self.pdn.apply(weights_init)

    def forward(self, x):
        # Comments on Algorithm 3: We use the image normalization of the pretrained models of torchvision [44].
        x = imagenet_norm_batch(
            x)
        pdn_out = self.pdn(x)
        return pdn_out


class AutoEncoder(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = EncConv()
        self.decoder = DecConv()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    import torch

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoEncoder()
    model = model.to('cuda')
    summary(model, (3, 256, 256))
