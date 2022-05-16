"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
"""
from calendar import prcal
from typing import ChainMap
from re import S
from builtins import print, super
import torch
from torch import nn

from gdn import GDN
from masked_conv import MaskedConv2d
from entropy_models import EntropyBottleneck, GaussianConditional


class ResidualBlock(nn.Module):
    #Simple residual block with two 3x3 convolutions.
    def __init__(self, channel_in, channel_out):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.gdn = GDN(in_channels=channel_out)
        if channel_in != channel_out:
            self.skip = nn.Conv2d(channel_in, channel_out, 1, 1)
        else:
            self.skip = None

    def forward(self, x):
        y = self.conv1(x)
        y = nn.LeakyReLU(inplace=True)(y)
        y = self.conv2(y)
        y = nn.LeakyReLU(inplace=True)(y)
        if self.skip is not None:
            x = self.skip(x)
        out = x + y
        return out


class ResidualBlockWithStride(nn.Module):
    #Residual block with a stride on the first convolution
    def __init__(self, channel_in, channel_out, stride=2):
        super(ResidualBlockWithStride, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, stride, 1)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.gdn = GDN(in_channels=channel_out, inverse=False)
        self.skip = nn.Conv2d(channel_in, channel_out, 1, stride)

    def forward(self, x):
        y = self.conv1(x)
        y = nn.LeakyReLU(inplace=True)(y)
        y = self.conv2(y)
        y = self.gdn(y)
        x = self.skip(x)
        out = x + y
        return out


class ResidualBlockUpsample(nn.Module):
    #Simple residual block with two 3x3 convolutions.
    def __init__(self, channel_in, channel_out, upscale=2):
        super(ResidualBlockUpsample, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_in, channel_out * upscale**2, 3, padding=1),
            nn.PixelShuffle(upscale))
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.igdn = GDN(in_channels=channel_out, inverse=True)
        self.skip = nn.Sequential(
            nn.Conv2d(channel_in, channel_out * upscale**2, 3, padding=1),
            nn.PixelShuffle(upscale))

    def forward(self, x):
        y = self.conv1(x)
        y = nn.LeakyReLU(inplace=True)(y)
        y = self.conv2(y)
        y = self.igdn(y)
        x = self.skip(x)
        out = x + y
        return out


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(N, N // 2, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N // 2, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(N // 2, N, 1, 1),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                out = self.conv(x)
                out += x
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(),
                                    ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            nn.Conv2d(N, N, 1, 1),
        )

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += x
        return out


class Encoder(nn.Module):

    def __init__(self,
                 channel_in=3,
                 channel_mid=128,
                 channel_out=192,
                 stride=2):
        super(Encoder, self).__init__()

        self.RBS1 = ResidualBlockWithStride(channel_in, channel_mid, stride)
        self.RB1 = ResidualBlock(channel_mid, channel_mid)
        self.RBS2 = ResidualBlockWithStride(channel_mid, channel_mid, stride)
        self.att1 = AttentionBlock(channel_mid)
        self.RB2 = ResidualBlock(channel_mid, channel_mid)
        self.RBS3 = ResidualBlockWithStride(channel_mid, channel_mid, stride)
        self.RB3 = ResidualBlock(channel_mid, channel_mid)
        self.conv = nn.Conv2d(channel_mid, channel_out, 3, 2, 1)
        self.att2 = AttentionBlock(channel_out)

    def forward(self, x):
        x = self.RBS1(x)
        x = self.RB1(x)
        x = self.RBS2(x)
        x = self.att1(x)
        x = self.RB2(x)
        x = self.RBS3(x)
        x = self.RB3(x)
        x = self.conv(x)
        x = self.att2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=128, channel_out=3):
        super(Decoder, self).__init__()

        self.att1 = AttentionBlock(channel_in)
        self.RB1 = ResidualBlock(channel_in, channel_mid)
        self.RBU1 = ResidualBlockUpsample(channel_mid, channel_mid, 2)
        self.RB2 = ResidualBlock(channel_mid, channel_mid)
        self.RBU2 = ResidualBlockUpsample(channel_mid, channel_mid, 2)
        self.att2 = AttentionBlock(channel_mid)
        self.RB3 = ResidualBlock(channel_mid, channel_mid)
        self.RBU3 = ResidualBlockUpsample(channel_mid, channel_mid, 2)
        self.RB4 = ResidualBlock(channel_mid, channel_mid)
        self.conv = nn.Sequential(
            nn.Conv2d(channel_mid, channel_out * 2**2, 3, padding=1),
            nn.PixelShuffle(2))

    def forward(self, x):
        x = self.att1(x)
        x = self.RB1(x)
        x = self.RBU1(x)
        #print("1: ", x.shape)
        x = self.RB2(x)
        x = self.RBU2(x)
        #print("2: ", x.shape)
        x = self.att2(x)
        x = self.RB3(x)
        x = self.RBU3(x)
        #print("3: ", x.shape)
        x = self.RB4(x)
        x = self.conv(x)
        #print("4: ", x.shape)
        return x


class HyperEncoder(nn.Module):

    def __init__(self, channel_in=192, channel_out=192):
        super(HyperEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.conv3 = nn.Conv2d(channel_out, channel_out, 3, 2, 1)
        self.conv4 = nn.Conv2d(channel_out, channel_out, 3, 1, 1)
        self.conv5 = nn.Conv2d(channel_out, channel_out, 3, 2, 1)

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

        self.conv1 = nn.Conv2d(channel_in, channel_mid, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_mid, channel_mid * 2**2, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv3 = nn.Conv2d(channel_mid, channel_mid*3//2, 3, 1, 1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_mid*3//2, channel_mid*3//2 * 2**2, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv5 = nn.Conv2d(channel_mid*3//2, channel_mid*2, 3, 1, 1)
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


class EntropyParameters(nn.Module):

    def __init__(self, channel_in):
        super(EntropyParameters, self).__init__()
        channel_mid = channel_in // 4
        self.conv1 = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_mid * 10 // 3,
                               kernel_size=1,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=channel_mid * 10 // 3,
                               out_channels=channel_mid * 8 // 3,
                               kernel_size=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=channel_mid * 8 // 3,
                               out_channels=channel_mid * 2,
                               kernel_size=1,
                               stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU()(x)
        x = self.conv2(x)
        x = nn.LeakyReLU()(x)
        x = self.conv3(x)
        return x


class ContextPrediction(nn.Module):

    def __init__(self, channel_in=192):
        super(ContextPrediction, self).__init__()
        self.masked = MaskedConv2d("A",
                                   in_channels=channel_in,
                                   out_channels=channel_in * 2,
                                   kernel_size=5,
                                   stride=1,
                                   padding=2)

    def forward(self, x):
        return self.masked(x)


class Cheng2020Attention(nn.Module):

    def __init__(self,
                 channel_in=3,
                 channel_N=128,
                 channel_M=192,
                 channel_out=3):
        super(Cheng2020Attention, self).__init__()
        self.encoder = Encoder(channel_in=channel_in,
                               channel_mid=channel_N,
                               channel_out=channel_M)
        self.decoder = Decoder(channel_in=channel_M, channel_out=channel_out)

        self.hyper_encoder = HyperEncoder(channel_in=channel_M,
                                          channel_out=channel_N)
        self.hyper_decoder = HyperDecoder(channel_in=channel_N,
                                          channel_mid=channel_M)

        self.entropy_parameters = EntropyParameters(channel_in=channel_M * 4)
        self.context = ContextPrediction(channel_in=channel_M)
        self.entropy_bottleneck = EntropyBottleneck(channels=channel_N)
        self.gaussian = GaussianConditional(None)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

    def forward(self, x):
        #print(x.shape)
        y = self.encoder(x)
        #print("y:", y.shape)
        z = self.hyper_encoder(y)
        #print("z:", z.shape)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        #print("z_hat:", z_hat.shape)
        #print("z_lakelihoods:", z_likelihoods.shape)
        psi = self.hyper_decoder(z_hat)
        #print("psi:", psi.shape)

        y_hat = self.gaussian.quantize(
            y, "noise" if self.training else "dequantize")
        #print("y_hat:", y_hat.shape)
        phi = self.context(y_hat)
        #print("phi:", phi.shape)

        gaussian_params = self.entropy_parameters(torch.cat((psi, phi), dim=1))
        #print('gaussian_params', gaussian_params.shape)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #print("scales_hat:", scales_hat.shape, 'means_hat:', means_hat.shape)
        _, y_likelihoods = self.gaussian(y, scales_hat, means=means_hat)
        #print("y_likelihoods:", y_likelihoods.shape)

        x_hat = self.decoder(y_hat)
        #print("x_hat:", x_hat.shape)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            }
        }


import math

if __name__ == "__main__":
    net = Cheng2020Attention()
    x = torch.randn(1, 3, 256, 256)
    out = net(x)
    x_hat = out['x_hat']
    likelihoods = out['likelihoods']
    y_likelihoods = likelihoods['y']
    z_likelihoods = likelihoods['z']
    #print(x_hat)
    #print(y_likelihoods)
    #print(z_likelihoods)
    bpp_loss = sum((torch.log(likelihoods).sum() / (-math.log(2) * 256 * 256))
                   for likelihoods in out["likelihoods"].values())
    print(bpp_loss)
    latent_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * 256 * 256)
    hyper_loss = torch.log(z_likelihoods).sum() / (-math.log(2) * 256 * 256)
    print(latent_loss + hyper_loss)
    loss = nn.MSELoss()
    mse_loss = loss(x_hat, x)
    print(mse_loss)
'''
torch.Size([1, 3, 256, 256])
y: torch.Size([1, 192, 16, 16])
z: torch.Size([1, 192, 4, 4])
z_hat: torch.Size([1, 192, 4, 4])
z_lakelihoods: torch.Size([1, 192, 4, 4])
psi: torch.Size([1, 384, 16, 16])
y_hat: torch.Size([1, 192, 16, 16])
phi: torch.Size([1, 384, 16, 16])
gaussian_params torch.Size([1, 384, 16, 16])
scales_hat: torch.Size([1, 192, 16, 16]) means_hat: torch.Size([1, 192, 16, 16])
y_likelihoods: torch.Size([1, 192, 16, 16])
x_hat: torch.Size([1, 3, 256, 256])
'''