import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models
import timm

from models.VGG16 import Vgg16Main
model_path = r"/vol/research/RobotFarming/Projects/tripplet_net/experiments/condor/model_hw160_o005_b10_fext3_man_20221130_175210_2.pth"
class VGG16(nn.Module):
    def __init__(self, 
                load_ckpt_path=r"/vol/research/RobotFarming/Projects/tripplet_net/experiments/condor/model_hw160_o005_b10_fext3_man_20221130_175210_2.pth"
            ):
            
        super(VGG16, self).__init__()
        '''
         use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
        '''
        self.feature_list = [1, 3, 5]
        self.load_ckpt_path = load_ckpt_path
        # vgg16 = Vgg16Main.load_from_checkpoint(load_ckpt_path)
        # vgg16.freeze()

        vgg16 = Vgg16Main()
        vgg16.load_state_dict(torch.load(self.load_ckpt_path))
        vgg16.freeze()

        self.model = torch.nn.Sequential(
            vgg16.block1, 
            vgg16.block2, 
            vgg16.block3, 
            vgg16.block4, 
            vgg16.block5, 
            vgg16.fextractor
        )
        # f = vgg16.features.children()


    def forward(self, x):
        # x = (x-0.5)/0.5
        features = []
        for i, layer in enumerate(list(self.model)):
            # print(i)
            if i == 5:
                x = x.view(x.size(0), -1)
            # print(x.shape)
            x = layer(x)
            if i in self.feature_list:
                features.append(x)
        return features

def HyperVGGPerceptualLoss(fakeIm, realIm, vggnet, backbone=None):
    '''
    use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
    '''
    # fakeIm = expand_dim_with_zero_pad(fakeIm)
    # realIm = expand_dim_with_zero_pad(realIm)

    weights = [1, 0.2, 0.04, 0.008, 0.0016, 0.00032, 0.000064, 0.0000128]
    print(f'fake im shape: {fakeIm.shape}')
    print(f'real im shape: {realIm.shape}')
    features_fake = vggnet(fakeIm)
    features_real = vggnet(realIm.float())
    features_real_no_grad = [f_real.detach() for f_real in features_real]
    # features_real_no_grad = torch.stack(features_real_no_grad)
    mse_loss = nn.MSELoss(reduction='elementwise_mean')

    loss = 0
    for i in range(len(features_real)):
        # loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        # breakpoint()
        if backbone == 'resnet':
            loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
        else:
            loss_i = mse_loss(F.normalize(features_fake[i]), features_real_no_grad[i])
        loss = loss + loss_i * weights[i]

    return loss

def expand_dim_with_zero_pad(img, out_channel=3):
    w = img.shape[-1]
    h = img.shape[-2]
    b = img.shape[0]
    padded_img = torch.zeros((b, out_channel, w, h))

    padded_img[:,0,:,:]=img[:,0,:,:]
    padded_img[:,1,:,:]=img[:,1,:,:]
    padded_img[:,2,:,:]=img[:,2,:,:]

    return padded_img