from PIL import Image
import os
import numpy as np
import torch
import scipy.io as sio
import imgvision as iv
from images.plot import imsave, imsave_deg
from models.TransformerHyperCompress import ChannelTrans
from models.transformercompress import SymmetricalTransFormer
from models.cheng2020attention import Cheng2020Attention, Cheng2020channel
from models.CA.hypercompress4 import HyperCompress4
from models.degradation import Degcompress
from models.NFC import NFC


if __name__ == "__main__":
    model = NFC()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)