from models.LFE_lvx import LFE_lvx
from models.ScaledDotProductAttentionOnly import ScaledDotProductAttentionOnly


import torch
from torch import nn

# HyperTransformerResBlock Implementation
class HyperTransformerResBlock(nn.Module):
    ''' Hyperspectral Transformer Residual Block '''

    def __init__(self, HSI_in_c, n_feates, lv, temperature):
        super().__init__()
        self.HSI_in_c = HSI_in_c  #Number of input channels = Number of output channels
        self.lv = lv              #Spatial level 1-> hxw, 2-> 2hx2w, 3-> 3hx3w
        if lv == 1:
            out_channels = int(n_feates)
        elif lv ==2:
            out_channels = int(n_feates/2)
        elif lv ==3:
            out_channels = int(n_feates/4)

        #Learnable feature extractors (FE-PAN & FE-HSI)
        self.LFE_HSI    = LFE_lvx(in_channels=self.HSI_in_c, n_feates = n_feates, level=lv)
        self.LFE_PAN    = LFE_lvx(in_channels=1,  n_feates = n_feates, level=lv)

        #Attention
        self.DotProductAttention = ScaledDotProductAttentionOnly(temperature=temperature)

        #Texture & Spectral Mixing
        self.TSmix = nn.Conv2d(in_channels=int(2*out_channels), out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, F, X_PAN, PAN_UD, X_MS_UP):
        # Obtaining Values, Keys, and Queries
        V = self.LFE_PAN(X_PAN)
        K = self.LFE_PAN(PAN_UD)
        Q = self.LFE_HSI(X_MS_UP)

        # Obtaining T (Transfered HR Features)
        T  = self.DotProductAttention(V, K, Q)

        #Concatenating F and T
        FT = torch.cat((F, T), dim=1)

        #Texture spectral mixing
        res = self.TSmix(FT)

        #Output
        output = F + res
        return output