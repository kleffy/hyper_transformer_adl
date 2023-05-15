from torch import nn

# This function implements the learnable spectral feature extractor (abreviated as LSFE)
# Input:    Hyperspectral or PAN image
# Outputs:  out1 = features at original resolution, out2 = features at original resolution/2, out3 = features at original resolution/4
#This function implements the multi-head attention
class LFE_lvx(nn.Module):
    def __init__(self, in_channels, n_feates, level):
        super(LFE_lvx, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        self.level = level
        lv1_c = int(n_feates)
        lv2_c = int(n_feates/2)
        lv3_c = int(n_feates/4)

        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=lv3_c, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(lv3_c)
        self.conv_64_2  = nn.Conv2d(in_channels=lv3_c, out_channels=lv3_c, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(lv3_c)

        #Second level convolutions
        if self.level == 1 or self.level==2:
            self.conv_128_1 = nn.Conv2d(in_channels=lv3_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_1   = nn.BatchNorm2d(lv2_c)
            self.conv_128_2 = nn.Conv2d(in_channels=lv2_c, out_channels=lv2_c, kernel_size=3, padding=1)
            self.bn_128_2   = nn.BatchNorm2d(lv2_c)

        #Third level convolutions
        if  self.level == 1:
            self.conv_256_1 = nn.Conv2d(in_channels=lv2_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_1   = nn.BatchNorm2d(lv1_c)
            self.conv_256_2 = nn.Conv2d(in_channels=lv1_c, out_channels=lv1_c, kernel_size=3, padding=1)
            self.bn_256_2   = nn.BatchNorm2d(lv1_c)

        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out    = self.bn_64_2(self.conv_64_2(out))

        #Second level outputs
        if self.level == 1 or self.level==2:
            out    = self.MaxPool2x2(self.LeakyReLU(out))
            out    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out)))
            out    = self.bn_128_2(self.conv_128_2(out))

        #Third level outputs
        if  self.level == 1:
            out     = self.MaxPool2x2(self.LeakyReLU(out))
            out     = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out)))
            out     = self.bn_256_2(self.conv_256_2(out))

        return out