from models.conv3x3 import conv3x3
from models.LFE import LFE
from models.SFE import SFE
from models.ScaledDotProductAttentionOnly import ScaledDotProductAttentionOnly


import torch
import torch.nn.functional as F
from torch import nn

# We pre-train this model first and then train the above model with pre-trained weights
class HyperTransformerPre(nn.Module):
    def __init__(self, config):
        super(HyperTransformerPre, self).__init__()
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]

        self.num_res_blocks = [16, 1, 1, 1, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        # Scaled dot product attention
        lv1_dim      = config[config["train_dataset"]]["LR_size"]**2
        lv2_dim      = (2*config[config["train_dataset"]]["LR_size"])**2
        lv3_dim      = (4*config[config["train_dataset"]]["LR_size"])**2
        ### Scaled Dot Product Attention ###
        self.TS_lv3     = ScaledDotProductAttentionOnly(temperature=lv1_dim)
        self.TS_lv2     = ScaledDotProductAttentionOnly(temperature=lv2_dim)
        self.TS_lv1     = ScaledDotProductAttentionOnly(temperature=lv3_dim)
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)
        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)

        self.conv22_head    = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)

        self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))

        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        with torch.no_grad():
            if not self.is_DHP_MS:
                X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor,self.factor),mode ='bicubic')

            else:
                X_MS_UP = X_MS

            # Generating PAN, and PAN (UD) images
            X_PAN   = X_PAN.unsqueeze(dim=1)
            PAN_D   = F.interpolate(X_PAN, scale_factor=(1/self.factor, 1/self.factor), mode ='bilinear')
            PAN_UD  = F.interpolate(PAN_D, scale_factor=(self.factor, self.factor), mode ='bilinear')

        #Extracting T and S at multiple-scales
        #lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        V_lv1, V_lv2, V_lv3 = self.LFE_PAN(X_PAN)
        K_lv1, K_lv2, K_lv3 = self.LFE_PAN(PAN_UD)
        Q_lv1, Q_lv2, Q_lv3 = self.LFE_HSI(X_MS_UP)

        T_lv3  = self.TS_lv3(V_lv3, K_lv3, Q_lv3)
        T_lv2  = self.TS_lv2(V_lv2, K_lv2, Q_lv2)
        T_lv1  = self.TS_lv1(V_lv1, K_lv1, Q_lv1)

        #Shallow Feature Extraction (SFE)
        x = self.SFE(X_MS)

        x11 = x
        #HyperTransformer at (L/4, W/4) scale
        x11_res = x11
        x11_res = torch.cat((x11_res, T_lv3), dim=1)
        x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11     = x11 + x11_res

        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        x22_res = torch.cat((x22_res, T_lv2), dim=1)
        x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22     = x22 + x22_res

        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        x33_res = torch.cat((x33_res, T_lv1), dim=1)
        x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33     = x33 + x33_res

        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)

        x      = self.final_conv(xF)

        output = {  "pred": x}

        return output
