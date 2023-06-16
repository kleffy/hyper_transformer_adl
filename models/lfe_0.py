import torch.nn as nn

class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()
        #Define number of input channels
        self.in_channels = in_channels
        
        #First level convolutions
        self.conv_64_1  = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn_64_1    = nn.BatchNorm2d(64)
        self.conv_64_2  = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_2    = nn.BatchNorm2d(64)
        
        #Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1   = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_2   = nn.BatchNorm2d(128)
        
        #Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1   = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_2   = nn.BatchNorm2d(256)
        
        #Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LeakyReLU 
        self.LeakyReLU  = nn.LeakyReLU(negative_slope=0.0)
    def forward(self, x):
        #First level outputs
        out1    = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))
        out1    = self.bn_64_2(self.conv_64_2(out1))

        #Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2    = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2    = self.bn_128_2(self.conv_128_2(out2))

        #Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3    = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3    = self.bn_256_2(self.conv_256_2(out3))

        return out1, out2, out3