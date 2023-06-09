import torch
from torch import nn
from models.ScaledDotProductAttention import ScaledDotProductAttention

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()

        self.n_head = n_head
        self.linear_dim = linear_dim

        self.w_qs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.w_ks = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.w_vs = nn.Linear(in_pixels, n_head * linear_dim, bias=False)
        self.fc = nn.Linear(n_head * linear_dim, in_pixels, bias=False)

        self.attention = ScaledDotProductAttention(temperature_init_value=in_pixels ** 0.5)

        self.OutBN = nn.BatchNorm2d(num_features=num_features)
        self.layer_norm = LayerNorm(n_head * linear_dim)
        self.activation = Swish()

    def forward(self, v, k, q, mask=None):
        b, c, h, w = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head = self.n_head
        linear_dim = self.linear_dim

        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        output = v

        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        v_attn = self.attention(v, k, q, mask=mask)

        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(self.layer_norm(self.activation(v_attn)))

        output = output + v_attn

        output = output.view(b, c, h, w)
        output = self.OutBN(output)

        return output
