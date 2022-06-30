import math
import torch
from torch import  nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.entropy_models import EntropyBottleneck, GaussianConditional
from models.gdn import GDN
from models.ops.ops import ste_round
from models.layers import *
from compressai.ans import BufferedRansEncoder, RansDecoder

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

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
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
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

    def __init__(self, channel_in=192):
        super(HyperEncoder, self).__init__()
        self.channel_out = channel_in//2
        self.conv1 = nn.Conv2d(channel_in, self.channel_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.channel_out, self.channel_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(self.channel_out, self.channel_out, 3, 2, 1)
        self.conv4 = nn.Conv2d(self.channel_out, self.channel_out, 3, 1, 1)
        self.conv5 = nn.Conv2d(self.channel_out, self.channel_out, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv3(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv4(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv5(x)
        return x


class HyperDecoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=192):
        super(HyperDecoder, self).__init__()
        self.channel_out = channel_in*2
        self.conv1 = nn.Conv2d(channel_in, channel_in, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_in, channel_in * 2**2, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv3 = nn.Conv2d(channel_in, channel_in*3//2, 3, 1, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_in*3//2, channel_in*3//2 * 2**2, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv5 = nn.Conv2d(channel_in*3//2, self.channel_out, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv3(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv4(x)
        x = nn.LeakyReLU(inplace=True)(x)
        x = self.conv5(x)
        return x


class ChannelTrans(nn.Module):
    def __init__(self, channel_in=31, stage=4, num_blocks=[2, 4, 4, 6], num_slices=8):
        super(ChannelTrans, self).__init__()
        self.num_slices = num_slices
        self.channel_N = channel_in*int(pow(2, stage))
        self.max_support_slices =10
        self.s = 4

        self.encoder = TransEncoder(channel_in, stage, num_blocks)
        self.decoder = TransDecoder(
            self.channel_N, channel_in, stage, num_blocks[::-1])
        self.hyper_encoder = HyperEncoder(self.channel_N)
        self.hyper_mean_decoder = HyperDecoder(channel_in=self.hyper_encoder.channel_out)
        self.hyper_scale_decoder = HyperDecoder(channel_in=self.hyper_encoder.channel_out)


        self.channel_p = self.hyper_mean_decoder.channel_out//num_slices
        self.mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*i, self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p*8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p*4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p*2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*i, self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p*8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p*4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p*2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*(1+i), self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p*8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p*4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p*2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(channels=self.hyper_encoder.channel_out)
        self.gaussian_conditional = GaussianConditional(None)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).  
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

    def forward(self, x):
        y = self.encoder(x)
        #print("y: ", y.shape)

        z = self.hyper_encoder(y)
        #print("z: ", z.shape)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        #print("z_hat: ", z_hat.shape)

        latent_scales = self.hyper_scale_decoder(z_hat)
        latent_means = self.hyper_mean_decoder(z_hat)
        #print("latent_scales: ",latent_scales.shape)
        #print("latent_means: ", latent_means.shape)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_id, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices <
                              0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            #print("slice %d mean support: "%slice_id, mean_support.shape)
            mu = self.mean_transforms[slice_id](mean_support)
            #print("slice %d mu: "%slice_id, mu.shape)
            #mu = mu[:, :, :y_shape[0], :y_shape[1]]
            ##print("slice %d mu: "%slice_index, mu.shape)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            #print("slice %d scale support: "%slice_id, scale_support.shape)
            scale = self.scale_transforms[slice_id](scale_support)
            #print("slice %d scale: "%slice_id, scale.shape)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu  #TODO: y_hat is computed by C in tensorflow version 

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        #print("y_hat: ", y_hat.shape)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        #print("y_likelihoods: ", y_likelihoods.shape)

        x_hat = self.decoder(y_hat)
        #print("x_hat: ", x_hat.shape)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},  
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def compress(self, x):
        y = self.encoder(x)
        #print("y: ", y.shape)

        z = self.hyper_encoder(y)
        #print("z: ", z.shape)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        #print("z_hat: ", z_hat.shape)

        latent_scales = self.hyper_scale_decoder(z_hat)
        latent_means = self.hyper_mean_decoder(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_id, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices <
                              0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            #print("slice %d mean support: "%slice_id, mean_support.shape)
            mu = self.mean_transforms[slice_id](mean_support)
            #print("slice %d mu: "%slice_id, mu.shape)
            #mu = mu[:, :, :y_shape[0], :y_shape[1]]
            ##print("slice %d mu: "%slice_index, mu.shape)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            #print("slice %d scale support: "%slice_id, scale_support.shape)
            scale = self.scale_transforms[slice_id](scale_support)
            #print("slice %d scale: "%slice_id, scale.shape)

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)
        # print(y_strings)
        # print(z.size()[-2:])
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.hyper_scale_decoder(z_hat)
        latent_means = self.hyper_mean_decoder(z_hat)

        y_shape = [z_hat.shape[2] * self.s, z_hat.shape[3] * self.s]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_id in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.mean_transforms[slice_id](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.scale_transforms[slice_id](scale_support)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.decoder(y_hat)
        return {"x_hat": x_hat}
