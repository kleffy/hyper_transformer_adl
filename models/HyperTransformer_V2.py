import torch
import torch.nn.functional as F
from torch import nn
from scipy.io import savemat

from models.LFE import LFE
from models.MultiHeadAttention import MultiHeadAttention
from models.NoAttention import NoAttention
from models.ResBlock import ResBlock
from models.SFE import SFE
from models.ScaledDotProductAttentionOnly import ScaledDotProductAttentionOnly
from models.VGG_LFE import VGG_LFE

LOSS_TP = nn.L1Loss()

EPS = 1e-10

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=True)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=True)

class HyperTransformer(nn.Module):
    def __init__(self, config):
        super(HyperTransformer, self).__init__()
        # Settings
        self.is_DHP_MS      = config["is_DHP_MS"]
        self.in_channels    = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels   = config[config["train_dataset"]]["spectral_bands"]
        self.factor         = config[config["train_dataset"]]["factor"]
        self.config         = config

        # Parameter setup
        self.num_res_blocks = [16, 4, 4, 4, 4]
        self.n_feats        = 256
        self.res_scale      = 1

        # FE-PAN & FE-HSI
        self.LFE_HSI    = LFE(in_channels=self.in_channels)
        self.LFE_PAN    = LFE(in_channels=1)
        
        # Dimention of each Scaled-Dot-Product-Attention module
        lv1_dim      = config[config["train_dataset"]]["LR_size"]**2
        lv2_dim      = (2*config[config["train_dataset"]]["LR_size"])**2
        lv3_dim      = (4*config[config["train_dataset"]]["LR_size"])**2

        # Number of Heads in Multi-Head Attention Module
        n_head          = config["N_modules"]
        
        # Setting up Multi-Head Attention or Single-Head Attention
        if n_head == 0:
            # No Attention #
            # JUst passing the HR features from PAN image (Values) #
            self.TS_lv3     = NoAttention()
            self.TS_lv2     = NoAttention()
            self.TS_lv1     = NoAttention()
        elif n_head == 1:
            ### Scaled Dot Product Attention ###
            self.TS_lv3     = ScaledDotProductAttentionOnly(temperature=lv1_dim)
            self.TS_lv2     = ScaledDotProductAttentionOnly(temperature=lv2_dim)
            self.TS_lv1     = ScaledDotProductAttentionOnly(temperature=lv3_dim)
        else:   
            ### Multi-Head Attention ###
            lv1_pixels      = config[config["train_dataset"]]["LR_size"]**2
            lv2_pixels      = (2*config[config["train_dataset"]]["LR_size"])**2
            lv3_pixels      = (4*config[config["train_dataset"]]["LR_size"])**2
            self.TS_lv3     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels = int(lv1_pixels), 
                                                    linear_dim = int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features = self.n_feats)
            self.TS_lv2     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels= int(lv2_pixels), 
                                                    linear_dim= int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features=int(self.n_feats/2))
            self.TS_lv1     = MultiHeadAttention(   n_head= int(n_head), 
                                                    in_pixels = int(lv3_pixels), 
                                                    linear_dim = int(config[config["train_dataset"]]["LR_size"]), 
                                                    num_features=int(self.n_feats/4))
        
        self.SFE        = SFE(self.in_channels, self.num_res_blocks[0], self.n_feats, self.res_scale)

        if config[config["train_dataset"]]["feature_sum"]:
            self.conv11_headSUM    = conv3x3(self.n_feats, self.n_feats)
        else:
            self.conv11_head    = conv3x3(2*self.n_feats, self.n_feats)

        self.conv12         = conv3x3(self.n_feats, self.n_feats*2)
        self.ps12           = nn.PixelShuffle(2)
        #Residial blocks
        self.RB11           = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB11.append(ResBlock(in_channels=self.n_feats, out_channels=self.n_feats,
                res_scale=self.res_scale))
        self.conv11_tail = conv3x3(self.n_feats, self.n_feats)

        if config[config["train_dataset"]]["feature_sum"]:
            self.conv22_headSUM = conv3x3(int(self.n_feats/2), int(self.n_feats/2))
        else:
            self.conv22_head = conv3x3(2*int(self.n_feats/2), int(self.n_feats/2))
        self.conv23         = conv3x3(int(self.n_feats/2), self.n_feats)
        self.ps23           = nn.PixelShuffle(2)
        #Residual blocks
        self.RB22 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB22.append(ResBlock(in_channels=int(self.n_feats/2), out_channels=int(self.n_feats/2), res_scale=self.res_scale))
        self.conv22_tail = conv3x3(int(self.n_feats/2), int(self.n_feats/2))

        if config[config["train_dataset"]]["feature_sum"]:
             self.conv33_headSUM   = conv3x3(int(self.n_feats/4), int(self.n_feats/4))
        else:
            self.conv33_head    = conv3x3(2*int(self.n_feats/4), int(self.n_feats/4))
        self.RB33 = nn.ModuleList()
        for i in range(self.num_res_blocks[3]):
            self.RB33.append(ResBlock(in_channels=int(self.n_feats/4), out_channels=int(self.n_feats/4), res_scale=self.res_scale))
        self.conv33_tail = conv3x3(int(self.n_feats/4), int(self.n_feats/4))

        self.final_conv     = nn.Conv2d(in_channels=self.n_feats+int(self.n_feats/2)+int(self.n_feats/4), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.RBF = nn.ModuleList()
        for i in range(self.num_res_blocks[4]):
            self.RBF.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.convF_tail = conv3x3(self.out_channels, self.out_channels)

        self.BN_x11 = nn.BatchNorm2d(self.n_feats)
        self.BN_x22 = nn.BatchNorm2d(int(self.n_feats/2))
        self.BN_x33 = nn.BatchNorm2d(int(self.n_feats/4))

        self.up_conv13 = nn.ConvTranspose2d(in_channels=self.n_feats, out_channels=self.in_channels, kernel_size=3, stride=4, output_padding=1)
        self.up_conv23 = nn.ConvTranspose2d(in_channels=int(self.n_feats/2), out_channels=self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.VGG_LFE_HSI    = VGG_LFE(in_channels=self.in_channels, requires_grad=False)
        self.VGG_LFE_PAN    = VGG_LFE(in_channels=1, requires_grad=False)

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
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x11_res = x11_res + T_lv3
            x11_res = self.conv11_headSUM(x11_res) #F.relu(self.conv11_head(x11_res))
        else:
            x11_res = torch.cat((self.BN_x11(x11_res), T_lv3), dim=1)
            x11_res = self.conv11_head(x11_res) #F.relu(self.conv11_head(x11_res))
        x11     = x11 + x11_res
        #Residial learning
        x11_res = x11
        for i in range(self.num_res_blocks[1]):
            x11_res = self.RB11[i](x11_res)
        x11_res = self.conv11_tail(x11_res)
        x11 = x11 + x11_res

        x22 = self.conv12(x11)
        x22 = F.relu(self.ps12(x22))
        #HyperTransformer at (L/2, W/2) scale
        x22_res = x22
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x22_res = x22_res + T_lv2
            x22_res = self.conv22_headSUM(x22_res) #F.relu(self.conv22_head(x22_res))
        else:
            x22_res = torch.cat((self.BN_x22(x22_res), T_lv2), dim=1)
            x22_res = self.conv22_head(x22_res) #F.relu(self.conv22_head(x22_res))
        x22     = x22 + x22_res
        #Residial learning
        x22_res = x22
        for i in range(self.num_res_blocks[2]):
            x22_res = self.RB22[i](x22_res)
        x22_res = self.conv22_tail(x22_res)
        x22 = x22 + x22_res

        x33 = self.conv23(x22)
        x33 = F.relu(self.ps23(x33))
        #HyperTransformer at (L, W) scale
        x33_res = x33
        if self.config[self.config["train_dataset"]]["feature_sum"]:
            x33_res = x33_res + T_lv1
            x33_res = self.conv33_headSUM(x33_res) #F.relu(self.conv33_head(x33_res))
        else:
            x33_res = torch.cat((self.BN_x33(x33_res), T_lv1), dim=1)
            x33_res = self.conv33_head(x33_res) #F.relu(self.conv33_head(x33_res))
        x33     = x33 + x33_res
        #Residual learning
        x33_res = x33
        for i in range(self.num_res_blocks[3]):
            x33_res = self.RB33[i](x33_res)
        x33_res = self.conv33_tail(x33_res)
        x33 = x33 + x33_res

        x11_up  = F.interpolate(x11, scale_factor=4, mode='bicubic')
        x22_up  = F.interpolate(x22, scale_factor=2, mode='bicubic')
        xF      = torch.cat((x11_up, x22_up, x33), dim=1)
        
        xF      = self.final_conv(xF)
        xF_res  = xF

        #Final resblocks
        for i in range(self.num_res_blocks[4]):
            xF_res = self.RBF[i](xF_res)
        xF_res  = self.convF_tail(xF_res)
        x       = xF + xF_res

        Phi_lv1, Phi_lv2, Phi_lv3   = self.LFE_HSI(x.detach())
        Phi_T_lv3  = self.TS_lv3(V_lv3, K_lv3, Phi_lv3)
        Phi_T_lv2  = self.TS_lv2(V_lv2, K_lv2, Phi_lv2)
        Phi_T_lv1  = self.TS_lv1(V_lv1, K_lv1, Phi_lv1)
        loss_tp                                 = LOSS_TP(Phi_T_lv1, T_lv1)+LOSS_TP(Phi_T_lv2, T_lv2)+LOSS_TP(Phi_T_lv3, T_lv3)

        x13 = self.up_conv13(x11)
        x23 = self.up_conv23(x22)
        output = {  "pred": x,
                    "x13": x13,
                    "x23": x23,
                    "tp_loss": loss_tp}
        return output





