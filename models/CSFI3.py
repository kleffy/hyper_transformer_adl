from models.HyperTransformer_V2 import conv1x1, conv3x3


import torch
import torch.nn.functional as F
from torch import nn


class CSFI3(nn.Module):
    def __init__(self, n_feats):
        super(CSFI3, self).__init__()
        n_feats1 = n_feats
        self.conv12 = conv1x1(n_feats1, n_feats1)
        self.conv13 = conv1x1(n_feats1, n_feats1)

        n_feats2 = int(n_feats/2)
        self.conv21 = conv3x3(n_feats2, n_feats2, 2)
        self.conv23 = conv1x1(n_feats2, n_feats2)

        n_feats3 = int(n_feats/4)
        self.conv31_1 = conv3x3(n_feats3, n_feats3, 2)
        self.conv31_2 = conv3x3(n_feats3, n_feats3, 2)
        self.conv32 = conv3x3(n_feats3, n_feats3, 2)

        self.conv_merge1 = conv3x3(n_feats1*3, n_feats1)
        self.conv_merge2 = conv3x3(n_feats2*3, n_feats2)
        self.conv_merge3 = conv3x3(n_feats3*3, n_feats3)

    def forward(self, x1, x2, x3):
        x12 = F.interpolate(x1, scale_factor=2, mode='bicubic')
        x12 = F.relu(self.conv12(x12))
        x13 = F.interpolate(x1, scale_factor=4, mode='bicubic')
        x13 = F.relu(self.conv13(x13))

        x21 = F.relu(self.conv21(x2))
        x23 = F.interpolate(x2, scale_factor=2, mode='bicubic')
        x23 = F.relu(self.conv23(x23))

        x31 = F.relu(self.conv31_1(x3))
        x31 = F.relu(self.conv31_2(x31))
        x32 = F.relu(self.conv32(x3))

        x1 = F.relu(self.conv_merge1( torch.cat((x1, x21, x31), dim=1) ))
        x2 = F.relu(self.conv_merge2( torch.cat((x2, x12, x32), dim=1) ))
        x3 = F.relu(self.conv_merge3( torch.cat((x3, x13, x23), dim=1) ))

        return x1, x2, x3