from telnetlib import DO
from turtle import forward
import torch
from torch import inverse, nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.entropy_models import EntropyBottleneck, GaussianConditional
from models.gdn import GDN
from models.ops.ops import ste_round
from models.layers import *


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h*w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))   # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v   # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)    # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, inverse=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            # F.gelu(),
            GDN(dim*mult, inverse=inverse),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            # F.gelu(),
            GDN(dim*mult, inverse=inverse),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            inverse,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim, inverse=inverse))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class TransEncoder(nn.Module):
    def __init__(self, channel_in, stage=4, num_blocks=[2, 4, 4, 6]):
        super().__init__()
        self.stage = stage
        self.embedding = conv3x3(channel_in, channel_in)
        self.encoder_layers = nn.ModuleList([])
        dim_stage = channel_in
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=channel_in, heads=dim_stage // channel_in, inverse=False),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

    def forward(self, x):
        x = self.embedding(x)
        for (MSAB, Downsample) in self.encoder_layers:
            x = MSAB(x)
            x = Downsample(x)

        return x


class TransDecoder(nn.Module):
    def __init__(self, channel_in, dim, stage, num_blocks=[6, 4, 4, 2]):
        super().__init__()
        self.decoder_layers = nn.ModuleList([])
        self.mapping = conv3x3(dim, dim)
        dim_stage = channel_in
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                #nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim, inverse=True),
            ]))
            dim_stage //= 2
    def forward(self, x):
        for (Upsample, MSAB) in self.decoder_layers:
            x = Upsample(x)
            x = MSAB(x)
        x = self.mapping(x)
        return x

class HyperEncoder(nn.Module):

    def __init__(self, channel_in=192, channel_out=192):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_out,
                               kernel_size=3,
                               stride=1,
                               padding=3 // 2)
        self.conv2 = nn.Conv2d(in_channels=channel_out,
                               out_channels=channel_out,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)
        self.conv3 = nn.Conv2d(in_channels=channel_out,
                               out_channels=channel_out,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv3(x)
        return x


class TransformerHyperCompress(nn.Module):
    def __init__(self, channel_in=31, stage=4, num_blocks=[2,4,4,6]):
        super(TransformerHyperCompress, self).__init__()
        self.encoder = TransEncoder(channel_in, stage, num_blocks)
        self.decoder = TransDecoder(
            channel_in*int(pow(2, stage)), channel_in, stage, num_blocks[::-1])

    def forward(self, x):
        y = self.encoder(x)
        print("y: ", y.shape)
        x_hat = self.decoder(y)
        print("x_hat:", x_hat.shape)
        
        return x_hat
