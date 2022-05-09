"""
Implementation of the model from the paper

Minnen, David, Johannes Ballé, and George D. Toderici.
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
"""

from turtle import forward
from urllib.error import ContentTooShortError
import torch
from torch import nn
import numpy as np

from models.gdn import GDN
from models.masked_conv import MaskedConv2d
from models.entropy import EntropyBottleneck, GaussianConditional


class Model(nn.Module):

    def __init__(self, device):
        self.device = device
        super(Model, self).__init__()
        self.encoder = Encoder(3, self.device)
        self.decoder = Decoder(192, self.device)
        self.hyper_encoder = HyperEncoder(192)
        self.hyper_decoder = HyperDecoder(192)
        self.entropy = EntropyParameters(768)
        self.context = ContextPrediction(192)

    def quantize(self, x):
        """
		Quantize function:  The use of round function during training will cause the gradient to be 0 and will stop encoder from training.
		Therefore to immitate quantisation we add a uniform noise between -1/2 and 1/2
		:param x: Tensor
		:return: Tensor
		"""
        uniform = -1 * torch.rand(x.shape) + 1 / 2
        return x + uniform.to(self.device)

    def forward(self, x):
        y = self.encoder(x)
        y_hat = self.quantize(y)
        z = self.hyper_encoder(y)
        z_hat = self.quantize(z)
        phi = self.context(y_hat)
        psi = self.hyper_decoder(z_hat)
        phi_psi = torch.cat([phi, psi], dim=1)
        sigma_mu = self.entropy(phi_psi)
        sigma, mu = torch.split(sigma_mu, y_hat.shape[1], dim=1)

        # clip sigma so it's larger than 0 - to make sure it satisfies statistical requirement of sigma >0 and not too close to 0 so it doesn't cause computational issues
        sigma = torch.clamp(sigma, min=1e-6)

        x_hat = self.decoder(y_hat)
        return x_hat, sigma, mu, y_hat, z_hat


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

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).")

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i:i + 1],
                params[i:i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    '''def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
		cdf = self.gaussian_conditional.quantized_cdf.tolist()
		cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
		offsets = self.gaussian_conditional.offset.tolist()

		encoder = BufferedRansEncoder()
		symbols_list = []
		indexes_list = []

        # Warning, this is slow...
		masked_weight = self.context_prediction.weight * self.context_prediction.mask
		for h in range(height):
			for w in range(width):
				y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
				ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
				p = params[:, :, h : h + 1, w : w + 1]
				gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
				gaussian_params = gaussian_params.squeeze(3).squeeze(2)
				scales_hat, means_hat = gaussian_params.chunk(2, 1)

				indexes = self.gaussian_conditional.build_indexes(scales_hat)

				y_crop = y_crop[:, :, padding, padding]
				y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
				y_hat[:, :, h + padding, w + padding] = y_q + means_hat
				symbols_list.extend(y_q.squeeze().tolist())
				indexes_list.extend(indexes.squeeze().tolist())

		encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

		string = encoder.flush()
		return string'''

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU).")

# FIXME: we don't respect the default entropy coder and directly call the
# range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding,
             y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i:i + 1],
                params[i:i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    '''def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
		cdf = self.gaussian_conditional.quantized_cdf.tolist()
		cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
		offsets = self.gaussian_conditional.offset.tolist()

		decoder = RansDecoder()
		decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
		for h in range(height):
			for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
				y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
				ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
				p = params[:, :, h : h + 1, w : w + 1]
				gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
				scales_hat, means_hat = gaussian_params.chunk(2, 1)

				indexes = self.gaussian_conditional.build_indexes(scales_hat)
				rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
				rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
				rv = self.gaussian_conditional.dequantize(rv, means_hat)

				hp = h + padding
				wp = w + padding
				y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv'''

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