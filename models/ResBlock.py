from models.conv3x3 import conv3x3
from torch import nn

def depthwise_separable_conv(in_channels, out_channels, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, dilation=dilation, groups=in_channels, padding=dilation, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1, dilation=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale

        self.conv1 = depthwise_separable_conv(in_channels, out_channels, stride=stride, dilation=dilation)
        self.conv2 = depthwise_separable_conv(out_channels, out_channels, dilation=dilation)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out
