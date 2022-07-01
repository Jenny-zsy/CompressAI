"""Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.
"""
import torch
import warnings
import torch.nn.functional as F

from builtins import print, super
from torch import nn

from models.gdn import GDN
from models.masked_conv import MaskedConv2d
from models.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from models.ops.ops import ste_round

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def mu(Q,M):

    return (torch.sum(Q*M)) / M.sum()

def sigma(Q,M):
   
    return torch.sqrt( abs(torch.sum(Q*Q*M-mu(Q,M)) / M.sum()))


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

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
        #print(x)
        y = self.conv1(x)    
        y = self.igdn(y)
        #y = nn.LeakyReLU(inplace=True)(y)    
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


class DecoderwithDeg(nn.Module):

    def __init__(self, channel_in=192, channel_mid=128, channel_out=3):
        super(DecoderwithDeg, self).__init__()

        '''Decoder'''
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
            
        '''DegDecoder'''
        self.conv1 =  nn.Sequential(
            nn.Conv2d(channel_in, channel_mid*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_mid, channel_mid*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_mid, channel_mid*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_mid, channel_out*4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(inplace=True))
        self.sig = nn.Sigmoid()
        self.softmax  = nn.Softmax()
    def Feature_fusion(self, f, mask):
        '''one = torch.ones(mask.shape).cuda()
        f_s = sigma(f, one-mask)*((f*mask-mu(f,mask))/sigma(f,mask)) + mu(f, one-mask)
        #print(f_s)
        return f_s*mask + f*(one-mask)'''
        return 
        

    def forward(self, x, deg):

        deg1 = self.conv1(deg)
        deg2 = self.conv2(deg1)
        deg3 = self.conv3(deg2)
        deg_mask = self.conv4(deg3)
        #print(deg)
        #print(x)
        
        x1 = x-deg
        x1 = self.att1(x1)
        x1 = self.RB1(x1)  
        x1 = self.RBU1(x1)
        #print("1: ", x1)
        
        x2 = x1-deg1
        x2 = self.RB2(x2)
        x2 = self.RBU2(x2)
        #print("2: ", x.shape)

        x3 = x2-deg2
        x3 = self.att2(x3)
        x3 = self.RB3(x3)
        x3 = self.RBU3(x3)
        #print("3: ", x.shape)

        x4 = x3-deg3
        x4 = self.RB4(x4)
        x4 = self.conv(x4)
        #print("4: ", x.shape)

        return x4, deg_mask

class Decoder(nn.Module):

    def __init__(self, channel_in=192, channel_mid=128, channel_out=3):
        super(Decoder, self).__init__()

        '''Decoder'''
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
            
    def forward(self, x, deg):

        deg1 = self.conv1(deg)
        deg2 = self.conv2(deg1)
        deg3 = self.conv2(deg2)
        deg4 = self.conv4(deg3)
        deg_mask = self.conv5(deg4)

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

class DegEncoder(nn.Module):
    def __init__(self, channel_in=31, channel_out=192):
        super(DegEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 2, 1)
        self.conv3 = nn.Conv2d(channel_out, channel_out, 3, 2, 1)
        self.conv4 = nn.Conv2d(channel_out, channel_out, 3, 2, 1)
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

class DegDecoder(nn.Module):
    def __init__(self, channel_in=192, channel_out=31):
        super(DegDecoder, self).__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv2d(channel_in, channel_in*4, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_in, channel_in*4, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_in, channel_in*4, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_in, channel_in*4, 3, padding=1),
            nn.PixelShuffle(2))
        self.conv5 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

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


class Degcompress(nn.Module):

    def __init__(self,
                 channel_in=3,
                 channel_N=128,
                 channel_M=192,
                 channel_out=31):
        super(Degcompress, self).__init__()
        self.encoder = Encoder(channel_in=channel_in,
                               channel_mid=channel_N,
                               channel_out=channel_M)
        self.decoder = DecoderwithDeg(channel_in=channel_M, channel_out=channel_out)

        self.hyper_encoder = HyperEncoder(channel_in=channel_M,
                                          channel_out=channel_N)
        self.hyper_decoder = HyperDecoder(channel_in=channel_N,
                                          channel_mid=channel_M)
        self.deg_encode = DegEncoder(channel_in=channel_in, channel_out=channel_M)
        #self.deg_decoder = DegDecoder(channel_in=channel_M, channel_out=channel_out)
        self.entropy_parameters = EntropyParameters(channel_in=channel_M * 4)
        self.context = ContextPrediction(channel_in=channel_M)
        self.hyper_entropy_bottleneck = EntropyBottleneck(channels=channel_N)
        self.deg_entropy_bottleneck = EntropyBottleneck(channels=channel_M)
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
        z_hat, z_likelihoods = self.hyper_entropy_bottleneck(z)
        #print("z_hat:", z_hat.shape)
        #print("z_lakelihoods:", z_likelihoods.shape)
        psi = self.hyper_decoder(z_hat)
        #print("psi:", psi.shape)

        deg = self.deg_encode(x)
        #print(deg.shape)
        deg_hat, deg_likelihoods = self.deg_entropy_bottleneck(deg)
        #print(deg_hat.shape, deg_likelihoods.shape)

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

        x_hat, deg = self.decoder(y_hat, deg_hat)
        #print("x_hat:", x_hat.shape)
        return {
            "x_hat": x_hat,
            "deg": deg,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
                "deg": deg_likelihoods,
            }
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian.update_scale_table(scale_table, force=force)
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.encoder(x)
        z = self.hyper_encoder(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.hyper_decoder(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian.quantized_cdf.tolist()
        cdf_lengths = self.gaussian.cdf_length.tolist()
        offsets = self.gaussian.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context.masked.weight * self.context.masked.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h : h + kernel_size, w : w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context.masked.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.hyper_decoder(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.context.masked.in_channels, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        #print(y_hat)
        x_hat = self.decoder(y_hat).clamp_(0, 1)
        print(x_hat)
        return {"x_hat": x_hat}

    def _decompress_ar(
        self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian.quantized_cdf.tolist()
        cdf_lengths = self.gaussian.cdf_length.tolist()
        offsets = self.gaussian.offset.tolist()

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
                    self.context.masked.weight,
                    bias=self.context.masked.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h : h + 1, w : w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp : hp + 1, wp : wp + 1] = rv





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