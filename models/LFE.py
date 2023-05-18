from torch import nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class LFE(nn.Module):
    """
    Now, the architecture includes the skip connections between the outputs of the max pooling layers and the next level of output. 
    In addition, I've added group convolutions with increasing numbers of groups for each level, 
    which may also help with model performance while controlling computational costs. This model 
    design maintains a balance between performance improvement and computational efficiency.
    """
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        self.in_channels = in_channels

        # First level convolutions and SEBlock
        self.conv_64_1  = DepthwiseSeparableConv(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(64)
        self.conv_64_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=2) # Grouped convolution
        self.bn_64_2    = nn.BatchNorm2d(64)
        self.se_block_64 = SEBlock(64)

        # Second level convolutions and SEBlock
        self.conv_128_1 = DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1   = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=4) # Grouped convolution
        self.bn_128_2   = nn.BatchNorm2d(128)
        self.se_block_128 = SEBlock(128)

        # Third level convolutions and SEBlock
        self.conv_256_1 = DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1   = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=8) # Grouped convolution
        self.bn_256_2   = nn.BatchNorm2d(256)
        self.se_block_256 = SEBlock(256)

        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        # First level outputs
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1 = self.bn_64_2(self.conv_64_2(out1))
        out1 = self.se_block_64(out1)

        # Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2 = self.bn_128_2(self.conv_128_2(out2))
        out2 = self.se_block_128(out2)

        # Adding skip connection from the first level
        # out2 = out2 + out1_mp

        # Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3 = self.bn_256_2(self.conv_256_2(out3))
        out3 = self.se_block_256(out3)

        # Adding skip connection from the second level
        # out3 = out3 + out2_mp

        return out1, out2, out3
