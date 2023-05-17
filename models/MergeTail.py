from models.conv3x3 import conv3x3
from models.conv1x1 import conv1x1


import torch
import torch.nn.functional as F
from torch import nn


class MergeTail(nn.Module):
    def __init__(self, n_feats, out_channels):
        super(MergeTail, self).__init__()
        self.conv13 = conv1x1(n_feats, int(n_feats/4))
        self.conv23 = conv1x1(int(n_feats/2), int(n_feats/4))
        self.conv_merge = conv3x3(3*int(n_feats/4), out_channels)
        self.conv_tail1 = conv3x3(out_channels, out_channels)
        self.conv_tail2 = conv1x1(n_feats//2, 3)

    def forward(self, x1, x2, x3):
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x = F.relu(self.conv_merge( torch.cat((x3, x13, x23), dim=1) ))
        x = self.conv_tail1(x)
        #x = self.conv_tail2(x)
        return x