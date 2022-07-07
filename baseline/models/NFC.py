from turtle import forward
from xml.etree.ElementInclude import include
import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
from scipy import linalg as la


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = self.unsqueeze2d(input, self.factor)
        else:
            output = self.squeeze2d(input, self.factor)

        return output, logdet

    def squeeze2d(input, factor):
        if factor == 1:
            return input

        B, C, H, W = input.size()

        assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)

        return x

    def unsqueeze2d(input, factor):
        if factor == 1:
            return input

        factor2 = factor**2

        B, C, H, W = input.size()

        assert C % (factor2) == 0, "C module factor squared is not 0"

        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)

        return x


class ActNorm(nn.Module):
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
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias)**2,
                              dim=[0, 2, 3],
                              keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias

    def _scale(self, input, logdet=None, reverse=False):

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

        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.qr(torch.randn(*w_shape))[0]

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
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet


class FlowStep(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # Actnorm
        self.actnorm = ActNorm(in_channel)

        # Invconv
        self.invconv = InvertibleConv1x1(in_channel, False)

        # Affine Coulping


    def forward(self, x, logdet, inverse=False):
        if not inverse:
            return self.normal_flow(x, logdet)
        else:
            return self.reverse_flow(x, logdet)
        return

    def normal_flow(self, x, logdet):

        # Actnorm
        x, logdet = self.actnorm(x, logdet, False)

        # InvConv
        x, logdet = self.invconv(x, logdet, False)

        # Affine Coulping
        x1, x2 = x.narrow 
        return

    def reverse_flow(self, x, logdet):
        return


class FlowBlock(nn.Module):
    def __init__(self, in_channel, step_num, patch_size):
        super().__init__()

        # Squeeze
        self.squeeze = SqueezeLayer(factor=2)

        # step_num FlowSteps
        self.FlowSteps = nn.ModuleList()
        for _ in range(step_num):
            self.FlowSteps.append(FlowStep(in_channel))

    def forward(self, x):
        return


class FlowNet(nn.Module):
    def __init__(self, in_channel, block_num=2, step_num=8, patch_size=128):
        super().__init__()

        C, H, W = in_channel, patch_size, patch_size

        self.FlowBlocks = nn.ModuleList()
        for _ in range(block_num):
            in_channel, patch_size = in_channel*4, patch_size//2

            self.FlowBlocks.append(FlowBlock(in_channel, step_num, patch_size))

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, x, logdet=0.0):
        for block in self.FlowBlocks:
            x, logdet = block(x, logdet, reverse=False)
        return x, logdet

    def decode(self, x, temperature=None):
        return x


class NFC(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.encoder = FlowNet(in_channel)

    def forward(self, x):
        x = self.encoder(x)
        return x
