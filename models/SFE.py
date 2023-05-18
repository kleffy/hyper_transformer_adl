from models.conv3x3 import conv3x3
from models.ResBlock import ResBlock

import torch.nn as nn

def depthwise_separable_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class SFE(nn.Module):
    def __init__(self, in_feats=224, num_res_blocks=10, n_feats=64, res_scale=1.0, dropout=0.5, dilation=2):
        super(SFE, self).__init__()
        
        self.conv_head = nn.Sequential(
            depthwise_separable_conv(in_feats, n_feats),
            nn.ReLU()
        )

        self.RBs = nn.Sequential(*[
            nn.Sequential(
                ResBlock(n_feats, n_feats, dilation=dilation, res_scale=res_scale),
                nn.BatchNorm2d(n_feats),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_res_blocks)
        ])

        self.conv_tail = depthwise_separable_conv(n_feats, n_feats)

    def forward(self, x):
        x = x
        x1 = self.conv_head(x)
        x = self.RBs(x1)
        x = self.conv_tail(x)
        return x + x1

# from models.conv3x3 import conv3x3
# from models.ResBlock import ResBlock

# import torch.nn.functional as F
# from torch import nn


# class SFE(nn.Module):
#     def __init__(self, in_feats, num_res_blocks, n_feats, res_scale):
#         super(SFE, self).__init__()
#         self.num_res_blocks = num_res_blocks
#         self.conv_head = conv3x3(in_feats, n_feats)

#         self.RBs = nn.ModuleList()
#         for i in range(self.num_res_blocks):
#             self.RBs.append(ResBlock(in_channels=n_feats, out_channels=n_feats,
#                 res_scale=res_scale))

#         self.conv_tail = conv3x3(n_feats, n_feats)

#     def forward(self, x):
#         x = F.relu(self.conv_head(x))
#         x1 = x
#         for i in range(self.num_res_blocks):
#             x = self.RBs[i](x)
#         x = self.conv_tail(x)
#         x = x + x1
#         return x