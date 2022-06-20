import torch
from models.TransformerHyperCompress import TransformerHyperCompress
from models.transformercompress import SymmetricalTransFormer
from models.cheng2020attention import Cheng2020Attention, Cheng2020channel
from models.CA.hypercompress4 import HyperCompress4
from models.degradation import Degcompress


'''if __name__ == "__main__":
    net = Degcompress(channel_in=31)
    x = torch.randn(1, 31, 256, 256)
    out = net(x)
    #print(out.shape)'''

def AGWN_Batch(x, SNR):
    b, h, m, n = x.shape
    snr = 10**(SNR/10.0)
    x_ = []
    for i in range(b):
        img = x[i, :, :, :].unsqueeze(0)
        xpower = torch.sum(img**2)/(h*m*n)
        npower = xpower/snr
        x_.append(img + torch.randn_like(img) * torch.sqrt(npower))
    return torch.cat(x_, 0)

def AGWN_np(x, SNR):
    b,m,n = x.shape
    snr = 10**(SNR/10.0)
    xpower = np.sum(x**2)/(b*m*n)
    npower = xpower/snr
    return  x + np.array(torch.randn_like(torch.from_numpy(x)))*np.sqrt(npower)

import scipy.io as sio
import numpy as np
import os
from PIL import Image

def gray2color(gray_array, color_map):
    
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
 
    for i in range(0, rows):
        for j in range(0, cols):
            #print(gray_array[i][j])
            color_array[i][j] = color_map[gray_array[i][j]]
    
    #color_image = Image.fromarray(color_array)
 
    return color_array

if __name__ == "__main__":
    img = sio.loadmat('/data3/zhaoshuyi/Datasets/CAVE/hsi/test/face_ms.mat')['data']/1.932
    noise_img = AGWN_np(img, 20)

    residual = np.abs(noise_img-img)
    residual /= residual.max()
    residual_image = np.zeros((residual.shape[1], residual.shape[2]))
    residual_image = (np.average(residual, axis=0) * 255).clip(0, 255).astype(int)
    print(residual_image)
    #residual_image = ((residual_image-residual_image.min())*255/(residual_image.max()-residual_image.min())).clip(0,255).astype(int)
    jet_map = np.loadtxt('./images/jet_int.txt', dtype=int)
    color_jet = gray2color(residual_image, jet_map)
    color_jet = Image.fromarray(np.uint8(color_jet))
    #residual_images = concat_images(Image.fromarray(np.uint8(residual_image)), color_jet)
    color_jet.save("residual.png")
