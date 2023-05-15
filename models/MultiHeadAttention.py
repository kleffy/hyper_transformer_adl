from models.ScaledDotProductAttention import ScaledDotProductAttention


from torch import nn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Hyperspectral Pansharpening (Image Fusion) '''

    def __init__(self, n_head, in_pixels, linear_dim, num_features):
        super().__init__()
        #Parameters
        self.n_head         = n_head        #No of heads
        self.in_pixels      = in_pixels     #No of pixels in the input image
        self.linear_dim     = linear_dim    #Dim of linear-layer (outputs)

        #Linear layers

        self.w_qs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for queries
        self.w_ks   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for keys
        self.w_vs   = nn.Linear(in_pixels, n_head * linear_dim, bias=False) #Linear layer for values
        self.fc     = nn.Linear(n_head * linear_dim, in_pixels, bias=False) #Final fully connected layer

        #Scaled dot product attention
        self.attention = ScaledDotProductAttention(temperature=in_pixels ** 0.5)

        #Batch normalization layer
        self.OutBN = nn.BatchNorm2d(num_features=num_features)

    def forward(self, v, k, q, mask=None):
        # Reshaping matrixes to 2D
        # q = b, c_q, h*w
        # k = b, c_k, h*w
        # v = b, c_v, h*w
        b, c, h, w      = q.size(0), q.size(1), q.size(2), q.size(3)
        n_head          = self.n_head
        linear_dim      = self.linear_dim

        # Reshaping K, Q, and Vs...
        q = q.view(b, c, h*w)
        k = k.view(b, c, h*w)
        v = v.view(b, c, h*w)

        #Save V
        output = v

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(b, c, n_head, linear_dim)
        k = self.w_ks(k).view(b, c, n_head, linear_dim)
        v = self.w_vs(v).view(b, c, n_head, linear_dim)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        # Computing ScaledDotProduct attention for each head
        v_attn = self.attention(v, k, q, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        v_attn = v_attn.transpose(1, 2).contiguous().view(b, c, n_head * linear_dim)
        v_attn = self.fc(v_attn)


        output = output + v_attn
        #output  = v_attn

        #Reshape output to original image format
        output = output.view(b, c, h, w)

        #We can consider batch-normalization here,,,
        #Will complete it later
        output = self.OutBN(output)
        return output