"""
Implementation of the Rate-Distortion Loss
with a simplified hyper-latent rate

Minnen, David, Johannes Ballé, and George D. Toderici.
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
"""

import torch
import torch.nn as nn
import math
import numpy as np


def SAM_GPU(im_true, im_fake):
    C = im_true.size()[0]
    H = im_true.size()[1]
    W = im_true.size()[2]
    esp = 1e-12
    Itrue = im_true.clone()#.resize_(C, H*W)
    Ifake = im_fake.clone()#.resize_(C, H*W)
    nom = torch.mul(Itrue, Ifake).sum(dim=0)#.resize_(H*W)
    denominator = Itrue.norm(p=2, dim=0, keepdim=True).clamp(min=esp) * \
                  Ifake.norm(p=2, dim=0, keepdim=True).clamp(min=esp)
    denominator = denominator.squeeze()
    sam = torch.div(nom, denominator).acos()
    sam[sam != sam] = 0
    sam_sum = torch.sum(sam) / (H * W) / np.pi * 180
    return sam_sum

def Batch_SAM(im_fake, im_true):

    N = im_true.size()[0]
    sam = 0
    for i in range(N):
        sam += SAM_GPU(im_fake[i], im_true[i])

    return sam/N

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        return out

class RateDistortion_SAM_Loss(nn.Module):
    def __init__(self, lmbda=1e-2, beta=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.beta = beta
    
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        SAM_Loss = 0
        for i in range(N):
            SAM_Loss += SAM_GPU(output["x_hat"][i], target[i])
            
        SAM_Loss = SAM_Loss / N
        out["sam_loss"] = SAM_Loss
        
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] + self.beta*SAM_Loss
        return out


class RateDistortion_SAM_Deg_Loss(nn.Module):
    def __init__(self, lmbda=1e-2, alpha=1e-2, beta=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.beta = beta
        self.alpha = alpha
    
    def forward(self, output, target, noise_input):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        SAM_Loss = 0
        for i in range(N):
            SAM_Loss += SAM_GPU(output["x_hat"][i], target[i])
            
        SAM_Loss = SAM_Loss / N
        out["sam_loss"] = SAM_Loss
        
        out["deg_loss"] = self.mse(output["deg"]+target, noise_input)

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] + self.beta*SAM_Loss + self.alpha* 255**2 * out["deg_loss"]
        
        return out

