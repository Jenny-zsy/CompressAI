'''
Implementation of the GDN non-linearity based on the papers:

"Density Modeling of Images using a Generalized Normalization Transformation"

Johannes Ballé, Valero Laparra, Eero P. Simoncelli

https://arxiv.org/abs/1511.06281

 from https://github.com/jorge-pessoa/pytorch-gdn/blob/master/pytorch_gdn/__init__.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from parametrizers import NonNegativeParametrizer

class GDN(nn.Module):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        beta = torch.ones(in_channels)
        beta = self.beta_reparam.init(beta)
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        gamma = gamma_init * torch.eye(in_channels)
        gamma = self.gamma_reparam.init(gamma)
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: Tensor) -> Tensor:
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(x**2, gamma, beta)

        if self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)

        out = x * norm

        return out


'''
class GDN(nn.Module):
	"""Generalized divisive normalization layer.
	y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
	"""
	
	def __init__(self,
	             ch,
	             device,
	             inverse=False,
	             beta_min=1e-6,
	             gamma_init=.1,
	             reparam_offset=2 ** -18):
		super(GDN, self).__init__()
		self.inverse = inverse
		self.beta_min = beta_min
		self.gamma_init = gamma_init
		self.reparam_offset = torch.FloatTensor([reparam_offset])
		
		self.build(ch, torch.device(device))
	
	def build(self, ch, device):
		self.pedestal = self.reparam_offset ** 2
		self.beta_bound = (self.beta_min + self.reparam_offset ** 2) ** .5
		self.gamma_bound = self.reparam_offset
		
		# Create beta param
		beta = torch.sqrt(torch.ones(ch) + self.pedestal)
		self.beta = nn.Parameter(beta.to(device))
		
		# Create gamma param
		eye = torch.eye(ch)
		g = self.gamma_init * eye
		g = g + self.pedestal
		gamma = torch.sqrt(g)
		
		self.gamma = nn.Parameter(gamma.to(device))
		self.pedestal = self.pedestal.to(device)
	
	def forward(self, inputs):
		# Assert internal parameters to same device as input
		self.beta = self.beta.to(inputs.device)
		self.gamma = self.gamma.to(inputs.device)
		self.pedestal = self.pedestal.to(inputs.device)
		
		unfold = False
		if inputs.dim() == 5:
			unfold = True
			bs, ch, d, w, h = inputs.size()
			inputs = inputs.view(bs, ch, d * w, h)
		
		_, ch, _, _ = inputs.size()
		
		# Beta bound and reparam
		beta = LowerBound.apply(self.beta, self.beta_bound)
		beta = beta ** 2 - self.pedestal
		
		# Gamma bound and reparam
		gamma = LowerBound.apply(self.gamma, self.gamma_bound)
		gamma = gamma ** 2 - self.pedestal
		gamma = gamma.view(ch, ch, 1, 1)
		
		# Norm pool calc
		norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
		norm_ = torch.sqrt(norm_)
		
		# Apply norm
		if self.inverse:
			outputs = inputs * norm_
		else:
			outputs = inputs / norm_
		
		if unfold:
			outputs = outputs.view(bs, ch, d, w, h)
		return outputs
		'''