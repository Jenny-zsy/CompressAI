"""
Implementation of the model from the paper

Minnen, David, Johannes Ballé, and George D. Toderici.
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
"""

from builtins import print
from turtle import forward
from urllib.error import ContentTooShortError
import torch
from torch import nn
import numpy as np

from models.gdn import GDN
from models.masked_conv import MaskedConv2d
from models.entropy import EntropyBottleneck, GaussianConditional


class Encoder(nn.Module):

    def __init__(self, channel_in=3, channel_mid=192, channel_out=192):
        super(Encoder, self).__init__()

        self.first_conv = nn.Conv2d(in_channels=channel_in,
                                    out_channels=channel_mid,
                                    kernel_size=5,
                                    stride=2,
                                    padding=5 // 2)
        self.conv1 = nn.Conv2d(in_channels=channel_mid,
                               out_channels=channel_mid,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)
        self.conv2 = nn.Conv2d(in_channels=channel_mid,
                               out_channels=channel_mid,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)
        self.conv3 = nn.Conv2d(in_channels=channel_mid,
                               out_channels=channel_out,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)
        self.gdn = GDN(channel_mid, inverse=False)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.gdn(x)
        x = self.conv1(x)
        x = self.gdn(x)
        x = self.conv2(x)
        x = self.gdn(x)
        x = self.conv3(x)
        return x


class Decoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=192, channel_out=3):
        super(Decoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_in,
                                          out_channels=channel_mid,
                                          kernel_size=5,
                                          stride=2,
                                          output_padding=1,
                                          padding=5 // 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_mid,
                                          out_channels=channel_mid,
                                          kernel_size=5,
                                          stride=2,
                                          output_padding=1,
                                          padding=5 // 2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=channel_mid,
                                          out_channels=channel_mid,
                                          kernel_size=5,
                                          stride=2,
                                          output_padding=1,
                                          padding=5 // 2)
        self.last_deconv = nn.ConvTranspose2d(in_channels=channel_mid,
                                              out_channels=channel_out,
                                              kernel_size=5,
                                              stride=2,
                                              output_padding=1,
                                              padding=5 // 2)
        self.igdn = GDN(channel_mid, inverse=True)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.igdn(x)
        x = self.deconv2(x)
        x = self.igdn(x)
        x = self.deconv3(x)
        x = self.igdn(x)
        x = self.last_deconv(x)
        return x


class HyperEncoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=192):
        super(HyperEncoder, self).__init__()
        channel_out = channel_mid
        self.conv1 = nn.Conv2d(in_channels=channel_in,
                               out_channels=channel_mid,
                               kernel_size=3,
                               stride=1,
                               padding=3 // 2)
        self.conv2 = nn.Conv2d(in_channels=channel_mid,
                               out_channels=channel_mid,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)
        self.conv3 = nn.Conv2d(in_channels=channel_mid,
                               out_channels=channel_out,
                               kernel_size=5,
                               stride=2,
                               padding=5 // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU()(x)
        x = self.conv2(x)
        x = nn.LeakyReLU()(x)
        x = self.conv3(x)
        return x


class HyperDecoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=192):
        super(HyperDecoder, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(in_channels=channel_in,
                                          out_channels=channel_mid,
                                          kernel_size=5,
                                          stride=2,
                                          output_padding=1,
                                          padding=5 // 2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=channel_mid,
                                          out_channels=channel_mid * 3 // 2,
                                          kernel_size=5,
                                          stride=2,
                                          output_padding=1,
                                          padding=5 // 2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=channel_mid * 3 // 2,
                                          out_channels=channel_mid * 2,
                                          kernel_size=3,
                                          stride=1,
                                          output_padding=0,
                                          padding=3 // 2)

    def forward(self, x):
        x = self.deconv1(x)
        x = nn.LeakyReLU()(x)
        x = self.deconv2(x)
        x = nn.LeakyReLU()(x)
        x = self.deconv3(x)
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


class ContextHyperprior(nn.Module):

    def __init__(self,
                 channel_in=3,
                 channel_N=128,
                 channel_M=192,
                 channel_out=3):
        super(ContextHyperprior, self).__init__()
        self.encoder = Encoder(channel_in=channel_in,
                               channel_mid=channel_N,
                               channel_out=channel_M)
        self.decoder = Decoder(channel_in=channel_M, channel_out=channel_out)

        self.hyper_encoder = HyperEncoder(channel_in=channel_M,
                                          channel_mid=channel_N)
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
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
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
    net = ContextHyperprior()
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