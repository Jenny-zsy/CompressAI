from turtle import forward
import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from math import log, pi
import math
import numpy as np

from utils import *
from models.ops import ste_round
from models.entropy_models import EntropyBottleneck, GaussianConditional
from models.masked_conv import MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
eps = 1e-8
n = 0


def conv(in_channels, out_channels, kernel_size=5, stride=2, bias=False):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        bias=bias,
        stride=stride,
        padding=kernel_size // 2,
    )


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = self.unsqueeze2d(input, self.factor)
        else:
            output = self.squeeze2d(input, self.factor)

        return output

    def squeeze2d(self, input, factor):
        if factor == 1:
            return input

        B, C, H, W = input.size()

        assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)

        return x

    def unsqueeze2d(self, input, factor):
        if factor == 1:
            return input

        factor2 = factor**2

        B, C, H, W = input.size()

        assert C % (factor2) == 0, "C module factor squared is not 0"

        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)

        return x


'''class ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.

    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() - bias)**2,
                              dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(torch.abs(self.scale / (torch.sqrt(vars) + 1e-8)))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):
        #input = input.to(self.logs.device)
        #print(input.device, self.logs.device)
        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        if(torch.isinf(input).sum() > 0):
            print("here!  actnorm input inf")
            exit()
        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)
        else:
            input, logdet = self._scale(input, logdet, reverse)
            if(torch.isinf(input).sum() > 0):
                print("here!  scale inf")
                exit()
            input = self._center(input, reverse)
            if(torch.isinf(input).sum() > 0):
                print(self.bias)
                print(input)
                print("here!  center inf")
                exit()
        return input, logdet'''

logabs = lambda x: torch.log(torch.abs(x))
class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            return self.reverse(input)
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
 
        log_abs = logabs(self.scale)

        logdet = height * width * torch.sum(log_abs)

        return self.scale * (input + self.loc), logdet

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            return F.conv2d(
                input, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
            )
        _, _, height, width = input.shape

        out = F.conv2d(input, self.weight)
        if logdet is not None:
            logdet += (
                height * width *
                torch.slogdet(self.weight.squeeze().double())[1].float()
            )
        logdet = (
            height * width *
            torch.slogdet(self.weight.squeeze().double())[1].float()
        )

        return out, logdet



class Invertible1x1Conv(nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        # initialization done with rotation matrix
        w_init = np.linalg.qr(np.random.randn(
            self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward(self, x, objective=None, reverse=False):
        if reverse:
            return self.reverse_(x, objective)
        dlogdet = torch.det(self.weight.squeeze()).abs(
        ).log() * x.size(-2) * x.size(-1)
        objective += dlogdet
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)

        return output, objective

    def reverse_(self, x, objective):
        dlogdet = torch.det(self.weight.squeeze()).abs(
        ).log() * x.size(-2) * x.size(-1)
        objective -= dlogdet
        weight_inv = torch.inverse(
            self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return output, objective


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]  # 利用QR分解 ，得到正交矩阵

        if not LU_decomposed:
            self.weight = nn.Parameter(torch.Tensor(w_init))
        else:
            p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
            s = torch.diag(upper)
            sign_s = torch.sign(s)
            log_s = torch.log(torch.abs(s))
            upper = torch.triu(upper, 1)
            l_mask = torch.tril(torch.ones(w_shape), -1)
            eye = torch.eye(*w_shape)

            self.register_buffer("p", p)
            self.register_buffer("sign_s", sign_s)
            self.lower = nn.Parameter(lower)
            self.log_s = nn.Parameter(log_s)
            self.upper = nn.Parameter(upper)
            self.l_mask = l_mask
            self.eye = eye

        self.w_shape = w_shape
        self.LU_decomposed = LU_decomposed

    def get_weight(self, input, reverse):
        b, c, h, w = input.shape

        if not self.LU_decomposed:
            dlogdet = torch.slogdet(self.weight)[1] * h * w
            if reverse:
                weight = torch.inverse(self.weight)
            else:
                weight = self.weight
        else:
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)

            lower = self.lower * self.l_mask + self.eye

            u = self.upper * self.l_mask.transpose(0, 1).contiguous()
            u += torch.diag(self.sign_s * torch.exp(self.log_s))

            dlogdet = torch.sum(self.log_s) * h * w

            if reverse:
                # print(u)
                u_inv = torch.inverse(u)
                l_inv = torch.inverse(lower)
                p_inv = torch.inverse(self.p)

                weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
            else:
                weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        if(torch.isnan(input).sum() > 0):
            print(input)
            print("here!  input")
            exit()
        weight, dlogdet = self.get_weight(input, reverse)
        if(torch.isnan(weight).sum() > 0):
            print("here!  weight")
            exit()
        if(torch.isnan(dlogdet).sum() > 0):
            print("here!  dlogdet")
            exit()

        if not reverse:
            z = F.conv2d(input, weight)
            if(torch.isnan(z).sum() > 0):
                print("here!  z")
                exit()
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc,
                               channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        '''if init == 'xavier':
            initialize_weights_xavier(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights(
                [self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)'''

    def forward(self, x):

        x1 = self.conv1(x)
        if(torch.isnan(x1).sum() > 0):
            '''for p in self.conv1.parameters():
                print(p)'''
            print('x1', x1)
            print('x', x)
            print("here!  x1")
            exit()
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        if(torch.isnan(x2).sum() > 0):
            print("here!  x2")
            exit()
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        if(torch.isnan(x3).sum() > 0):
            print("here!  x3")
            exit()
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        if(torch.isnan(x4).sum() > 0):
            print("here!  x4")
            exit()
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if(torch.isnan(x5).sum() > 0):
            print("here!  x5")
            exit()

        return x5


class Permute2d(nn.Module):
    def __init__(self, num_channels, shuffle):
        super().__init__()
        self.num_channels = num_channels
        self.indices = np.arange(self.num_channels - 1, -1, -1).astype(np.long)
        self.indices_inverse = np.zeros((self.num_channels), dtype=np.long)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i
        if shuffle:
            self.reset_indices()

    def reset_indices(self):
        np.random.shuffle(self.indices)
        for i in range(self.num_channels):
            self.indices_inverse[self.indices[i]] = i

    def forward(self, input, reverse=False):
        assert len(input.size()) == 4
        if not reverse:
            return input[:, self.indices, :, :]
        else:
            return input[:, self.indices_inverse, :, :]


class FlowStep(nn.Module):
    def __init__(self, in_channel, flow_permutation='shuffle',
                 flow_coupling="additive",
                 LU_decomposed=False):
        super().__init__()

        self.flow_coupling = flow_coupling

        # Actnorm
        self.actnorm = ActNorm(in_channel)

        # permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channel, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(
                z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channel, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (
                self.shuffle(z, rev),
                logdet,
            )
        else:
            self.reverse = Permute2d(in_channel, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (
                self.reverse(z, rev),
                logdet,
            )

        #self.invconv = Permute2d(in_channel, True)

        # Affine Coulping
        if flow_coupling == "additive":
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2, in_channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel, 1),
                nn.ReLU(inplace=True),
                ZeroConv2d(in_channel, in_channel//2),
            )
        elif flow_coupling == "affine":
            self.net = nn.Sequential(
                nn.Conv2d(in_channel // 2, in_channel, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, in_channel, 1),
                nn.ReLU(inplace=True),
                ZeroConv2d(in_channel, in_channel),
            )
        self.g1 = DenseBlock(in_channel//2, in_channel//2)
        self.g2 = DenseBlock(in_channel//2, in_channel//2)
        self.g3 = DenseBlock(in_channel//2, in_channel//2)

    def forward(self, x, logdet, reverse=False):

        if not reverse:
            return self.normal_flow(x, logdet)
        else:
            return self.reverse_flow(x, logdet)

    def normal_flow(self, input, logdet):
        x = input
        '''global n
        n += 1
        print('-------', n, ' forward-------')'''
        # print('input', x.max())

        # Actnorm
        x, logdet = self.actnorm(x, logdet, False)
        #print('actnorm ', x.max())
        if(torch.isnan(x).sum() > 0):
            print("here!  actnorm")
            exit()
        if(torch.isinf(x).sum() > 0):
            print(x)
            print("here!  actnorm inf")
            exit()

        # Permute
        x, logdet = self.flow_permutation(x, logdet, False)
        '''if(torch.isnan(x).sum()>0):
            print("here!  Invconv")
            exit()'''

        # Coupling
        # Split
        _, C, _, _ = x.shape
        x_a, x_b = split_feature(x, "split")
        '''if(torch.isnan(x_a).sum()>0):
            print("here!  xa")
            exit()
        if(torch.isnan(x_b).sum()>0):
            print("here!  xb")
            exit()'''
        # Coupling
        x_a1 = x_a
        if self.flow_coupling == "additive":
            x_b1 = x_b - self.net(x_a)
        elif self.flow_coupling == "affine":
            h = self.net(x_a)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            x_b1 = x_b + shift
            x_b1 = x_b1 * scale
            logdet = torch.sum(logabs(scale), dim=[1, 2, 3]) + logdet

        #x_a1 = x_a + self.g1(x_b)
        '''if(torch.isinf(x_a1).sum()>0):
            print("here!  xa1")
            exit()
        print('x_a1 ', x_a1.max())'''
        #x_g2 = self.g2(x_a1)
        #x_b1 = x_g2*x_b + self.g3(x_a1)
        '''if(torch.isinf(x_b1).sum()>0):
            #print(x_a)
            print("here!  xb1")
            exit()
        print('x_b1 ', x_b1.max()) '''
        # Concat
        x = torch.cat((x_a1, x_b1), dim=1)
        '''if(torch.isinf(x).sum()>0):
            print("here!  x actnorm")
            exit()'''

        return x, logdet

    def reverse_flow(self, x, logdet):
        # global n

        # print('-------', n, ' backward -------')
        # print('input', x.max())
        # n = n-1
        # Coupling
        # Split
        x_a1, x_b1 = split_feature(x, "split")
        # Coupling
        # (x_a1.shape)
        x_a = x_a1
        if self.flow_coupling == "additive":
            x_b = x_b1 + self.net(x_a1)
        elif self.flow_coupling == "affine":
            h = self.net(x_a1)
            shift, scale = split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            x_b = x_b1 / scale
            x_b = x_b - shift
        # x_b = (x_b1-self.g3(x_a1))/(self.g2(x_a1)+eps)
        # x_a = x_a1 - self.g1(x_b)
        # Concat
        x = torch.cat((x_a, x_b), dim=1)
        #print('Affine', x.max())

        # InvConv
        x, logdet = self.flow_permutation(x, logdet, True)

        # Actnorm
        x = self.actnorm(x, logdet, True)

        return x


class FlowBlock(nn.Module):
    def __init__(self, in_channel, step_num, flow_permutation='shuffle',
                 flow_coupling="additive",
                 LU_decomposed=False):
        super().__init__()
        self.step_num = step_num

        # Squeeze
        self.squeeze = SqueezeLayer(factor=2)

        # step_num FlowSteps
        self.FlowSteps = nn.ModuleList()
        for _ in range(step_num):
            self.FlowSteps.append(
                FlowStep(in_channel, flow_permutation, flow_coupling, LU_decomposed))

    def forward(self, x, logdet=0, reverse=False):

        if reverse:
            return self.reverse_flow(x)
        else:
            return self.normal_flow(x, logdet)

    def normal_flow(self, x, logdet):
        global n
        n = 0
        x = self.squeeze(x, reverse=False)
        if(torch.isinf(x).sum() > 0):
            print("here!  x inf")
            exit()
        for i in range(self.step_num):
            x, logdet = self.FlowSteps[i](x, logdet, reverse=False)
        return x, logdet

    def reverse_flow(self, x):
        global n
        n = 4
        for step in reversed(self.FlowSteps):
            x = step(x, 0, reverse=True)
        x = self.squeeze(x, reverse=True)
        return x


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out


def gaussian_log_p(x, mean, log_sd):

    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / (torch.exp(2 * log_sd)+eps)


def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2.0 + ((x - mean) ** 2) / torch.exp(logs * 2.0) + c)


def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])


class FlowNet(nn.Module):
    def __init__(self, in_channel, block_num=2, step_num=4, patch_size=128,  flow_permutation='shuffle', flow_coupling="additive", LU_decomposed=False):
        super().__init__()

        self.C, self.H, self.W = in_channel, patch_size, patch_size

        self.FlowBlocks = nn.ModuleList()
        for _ in range(block_num):
            in_channel, patch_size = in_channel*4, patch_size//2

            self.FlowBlocks.append(FlowBlock(
                in_channel, step_num, flow_permutation, flow_coupling, LU_decomposed))

        self.prior = ZeroConv2d(in_channel, in_channel * 2)
        '''self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    in_channel * 2,
                    patch_size,
                    patch_size,
                ]
            ),
        )'''

    '''def prior(self, data):
        h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        print(h)
        exit()
        return h'''

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):

        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, x, logdet=0.0):

        x, logdet = uniform_binning_correction(x)
        for block in self.FlowBlocks:

            x, logdet = block(x, logdet, reverse=False)

            #print(x.shape, logdet)

        # mean, logs = self.prior(torch.zeros_like(x)).chunk(2, 1)
        # log_p = gaussian_likelihood( mean, logs,x)
        # #print(log_p.shape, logdet.shape)
        # #log_p = log_p.view(x.shape[0], -1).sum(1)
        # # print('log_p',log_p)
        # # print('logdet', logdet)
        # bpd = torch.mean((-log_p-logdet) / (math.log(2.0) * self.C * self.H * self.W))

        return x, logdet

    def decode(self, x, temperature=None):
        for block in reversed(self.FlowBlocks):
            x = block(x, reverse=True)
        return x


class HyperEncoder(nn.Module):

    def __init__(self, channel_in=192):
        super(HyperEncoder, self).__init__()
        self.channel_out = channel_in
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
class NFC(nn.Module):
    def __init__(self, channel_in=3, block_num=2, step_num=4, num_slices=4, patch_size=256, flow_permutation='shuffle', flow_coupling="additive", LU_decomposed=False):
        super().__init__()
        self.channel_N = channel_in*4**block_num
        
        self.encoder = FlowNet(channel_in, block_num, step_num)
        self.hyper_encoder = HyperEncoder(self.channel_N)
        channel_M = self.hyper_encoder.channel_out
        self.hyper_decoder = HyperDecoder(channel_in=channel_M)
        self.entropy_parameters = EntropyParameters(channel_in=channel_M * 4)
        self.context = ContextPrediction(channel_in=channel_M)
        self.entropy_bottleneck = EntropyBottleneck(channels=self.channel_N)
        self.gaussian = GaussianConditional(None)
    def forward(self, x):
        y, logdet = self.encoder(x)
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

        x_hat = self.encoder(y_hat, reverse=True)

        '''return {
            "y_hat": y_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "logdet": logdet
        }'''
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "logdet": logdet
        }
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).  
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

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
        

        y, logdet = self.encoder(x)
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
        x_hat = self.encoder(y_hat, reverse=True).clamp_(0, 1)
        #print(x_hat)
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
    
''' 
class NFC(nn.Module):
    def __init__(self, channel_in=3, block_num=2, step_num=4, num_slices=4, patch_size=256, flow_permutation='shuffle', flow_coupling="additive", LU_decomposed=False):
        super().__init__()
        self.num_slices = num_slices
        self.channel_N = channel_in*4**block_num
        self.max_support_slices = 4

        self.encoder = FlowNet(channel_in, block_num, step_num)
        self.hyper_encoder = HyperEncoder(self.channel_N)
        self.hyper_mean_decoder = HyperDecoder(
            channel_in=self.hyper_encoder.channel_out)
        self.hyper_scale_decoder = HyperDecoder(
            channel_in=self.hyper_encoder.channel_out)

        self.channel_p = self.hyper_mean_decoder.channel_out//num_slices
        self.mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*min(i,
                     self.max_support_slices), self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p *
                     8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p *
                     4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p *
                     2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*min(i,
                     self.max_support_slices), self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p *
                     8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p *
                     4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p *
                     2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )
        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(self.hyper_mean_decoder.channel_out + self.channel_p*min(i+1,
                     self.max_support_slices+1), self.channel_p*12, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*12, self.channel_p *
                     8, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*8, self.channel_p *
                     4, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*4, self.channel_p *
                     2, stride=1, kernel_size=3),
                nn.GELU(),
                conv(self.channel_p*2, self.channel_p, stride=1, kernel_size=3),
            ) for i in range(num_slices)
        )

        self.entropy_bottleneck = EntropyBottleneck(
            channels=self.hyper_encoder.channel_out)
        self.gaussian_conditional = GaussianConditional(None)

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).  
        """
        aux_loss = sum(m.loss() for m in self.modules()
                       if isinstance(m, EntropyBottleneck))
        return aux_loss

    def forward(self, x):
        y, logdet = self.encoder(x)
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
            #print("slice %d mu: "%slice_index, mu.shape)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            #print("slice %d scale support: "%slice_id, scale_support.shape)
            scale = self.scale_transforms[slice_id](scale_support)
            #print("slice %d scale: "%slice_id, scale.shape)

            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            # TODO: y_hat is computed by C in tensorflow version
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        #print("y_hat: ", y_hat.shape)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        #print("y_likelihoods: ", y_likelihoods.shape)
        #print("x_hat: ", x_hat.shape)
        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "logdet": logdet
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force)
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def compress(self, x):
        y, logdet = self.encoder(x)
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
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(
            -1).int().tolist()
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
            y_q_slice = self.gaussian_conditional.quantize(
                y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets)

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
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(
            -1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_id in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices <
                              0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.mean_transforms[slice_id](mean_support)

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.scale_transforms[slice_id](scale_support)

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(
                index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_id](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.encoder(y_hat, reverse=True)
        return {"x_hat": x_hat}'''
