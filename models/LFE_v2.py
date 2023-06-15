from torch import nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_cbam=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if use_cbam:
            self.ca = ChannelAttention(out_channels)
            self.sa = SpatialAttention()
        else:
            self.ca = None
            self.sa = None

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.ca and self.sa:
            x = self.ca(x) * x
            x = self.sa(x) * x
        x += self.shortcut(residual)
        x = F.relu(x)
        return x



class LFE(nn.Module):
    """
    Now, the architecture includes group convolutions with increasing numbers of groups for each level, 
    which may also help with model performance while controlling computational costs. This model 
    design maintains a balance between performance improvement and computational efficiency.
    """
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        self.in_channels = in_channels

        # First level convolutions and SEBlock
        self.conv_64_1  = DepthwiseSeparableConv(in_channels=self.in_channels, out_channels=32, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(32)
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64)
        self.conv_64_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=2) # Grouped convolution
        self.bn_64_2    = nn.BatchNorm2d(64)
        self.cbam_64 = CBAM(64)

        # Second level convolutions and SEBlock
        self.conv_128_1 = DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_128_1   = nn.BatchNorm2d(64)
        self.res_block3 = ResidualBlock(64, 128)
        self.res_block4 = ResidualBlock(128, 128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=4) # Grouped convolution
        self.bn_128_2   = nn.BatchNorm2d(128)
        self.cbam_128 = CBAM(128)

        # Third level convolutions and SEBlock
        self.conv_256_1 = DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_256_1   = nn.BatchNorm2d(128)
        self.res_block5 = ResidualBlock(128, 128)
        self.res_block6 = ResidualBlock(128, 256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=8) # Grouped convolution
        self.bn_256_2   = nn.BatchNorm2d(256)
        self.cbam_256 = CBAM(256)

        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        # First level outputs
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.res_block1(out1)
        out1 = self.res_block2(out1)
        out1 = self.bn_64_2(self.conv_64_2(out1))
        out1 = self.cbam_64(out1)

        # Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2 = self.res_block3(out2)
        out2 = self.res_block4(out2)
        out2 = self.bn_128_2(self.conv_128_2(out2))
        out2 = self.cbam_128(out2)

        # Adding skip connection from the first level
        # out2 = out2 + out1_mp

        # Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3 = self.res_block5(out3)
        out3 = self.res_block6(out3)
        out3 = self.bn_256_2(self.conv_256_2(out3))
        out3 = self.cbam_256(out3)

        # Adding skip connection from the second level
        # out3 = out3 + out2_mp

        return out1, out2, out3
