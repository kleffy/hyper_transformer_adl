import torch
from torch import nn
from torchvision import models
from models.MeanShift import MeanShift

class VGG_LFE(torch.nn.Module):
    def __init__(self, in_channels, requires_grad=True, rgb_range=1):
        super(VGG_LFE, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.conv_RGB = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)

        self.slice1 = nn.Sequential(*[vgg_pretrained_features[i] if not isinstance(vgg_pretrained_features[i], nn.ReLU) else nn.ReLU(inplace=False) for i in range(2)])
        self.slice2 = nn.Sequential(*[vgg_pretrained_features[i] if not isinstance(vgg_pretrained_features[i], nn.ReLU) else nn.ReLU(inplace=False) for i in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg_pretrained_features[i] if not isinstance(vgg_pretrained_features[i], nn.ReLU) else nn.ReLU(inplace=False) for i in range(7, 12)])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    def forward(self, x):
        x = self.conv_RGB(x)

        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x

        return x_lv1, x_lv2, x_lv3


# from models.MeanShift import MeanShift


# import torch
# from torch import nn
# from torchvision import models


# class VGG_LFE(torch.nn.Module):
#     def __init__(self, in_channels, requires_grad=True, rgb_range=1):
#         super(VGG_LFE, self).__init__()

#         ### use vgg19 weights to initialize
#         vgg_pretrained_features = models.vgg19(pretrained=True).features

#         #Initial convolutional layer to form RGB image
#         self.conv_RGB = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=3, padding=1)

#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()

#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.slice1.parameters():
#                 param.requires_grad = requires_grad
#             for param in self.slice2.parameters():
#                 param.requires_grad = requires_grad
#             for param in self.slice3.parameters():
#                 param.requires_grad = requires_grad

#         vgg_mean = (0.485, 0.456, 0.406)
#         vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
#         self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

#     def forward(self, x):
#         # Initial convolutional layer to make the RGB image...
#         x = self.conv_RGB(x)

#         # Extracting VGG Features...
#         x = self.sub_mean(x)
#         x = self.slice1(x)
#         x_lv1 = x
#         x = self.slice2(x)
#         x_lv2 = x
#         x = self.slice3(x)
#         x_lv3 = x
#         return x_lv1, x_lv2, x_lv3