import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.entropy_models import EntropyBottleneck, GaussianConditional
from models.ops.ops import ste_round

class TransEncoder(nn.Module):
    def __init__(self):
        super().__init__()

class TransformerHyperCompress(nn.Module):
    def __init__(self,
                 channel_in=31,
                 channel_N=128,
                 channel_M=192,
                 channel_out=31):
        super(TransformerHyperCompress, self).__init__()
        self.encoder = TransEncoder()