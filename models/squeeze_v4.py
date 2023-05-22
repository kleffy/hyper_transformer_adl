import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x

class SpectralAttention(nn.Module):
    def __init__(self):
        super(SpectralAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=(2, 3), keepdim=True).view(b, c)
        y = self.conv1(y.unsqueeze(2)).squeeze(2)
        return self.sigmoid(y).view(b, c, 1, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HyperspectralNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super(HyperspectralNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Spatial and Spectral attention
        self.spatial_attention = SpatialAttention(512)
        self.spectral_attention = SpectralAttention()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.conv2(x)

        # Apply spatial attention
        spatial_attention_map = self.spatial_attention(x)
        x = x * spatial_attention_map

        # Spectral attention
        spectral_attention_map = self.spectral_attention(x)
        x = x * spectral_attention_map

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Initialize the model
    in_channels = 224
    num_classes = 10
    batch_size = 20
    model = HyperspectralNet(in_channels, num_classes)
    sample_input = torch.randn(batch_size, in_channels, 32, 32)
    output = model(sample_input)
    print(output.shape)
