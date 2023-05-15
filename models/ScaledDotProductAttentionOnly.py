import torch
import torch.nn.functional as F
from torch import nn


class ScaledDotProductAttentionOnly(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q, mask=None):
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)

        # Reshaping K,Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        # Reshaping K,Q, and Vs...
        # q = q.view(q.size(0), q.size(1), q.size(2) * q.size(3))
        # k = k.view(k.size(0), k.size(1), k.size(2) * k.size(3))
        # v = v.view(v.size(0), v.size(1), v.size(2) * v.size(3))


        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        # attn = torch.matmul(q / q.size(2), k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        # Normalization (SoftMax)
        attn    = F.softmax(attn, dim=-1)

        # Attention output
        output  = torch.matmul(attn, v)

        #Reshape output to original format
        output  = output.view(b, c, h, w)
        return output