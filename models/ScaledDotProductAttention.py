import torch
import torch.nn.functional as F
from torch import nn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature_init_value, dropout=0.1):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature_init_value]), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        return output
