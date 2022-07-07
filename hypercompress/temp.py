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
    x = torch.randn(1, 31, 256, 256)
    out = model(x)
    '''model.update(force=True)
    out_enc = model.compress(x)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])'''
    # print(out.shape)
'''if __name__ == "__main__":
    denoise = sio.loadmat("./photo_and_face_den.mat")
    denoise =denoise['denoised4']

    data = sio.loadmat("/data3/zhaoshuyi/compressresults/cheng2020_CAVE_chN192_chM192_lambda0.01_beta0.1_bs16_lr0.0001/checkpoint1000_0.0001N/chart_and_stuffed_toy_ms.mat")
    inputs = data['inputs']
    ori = data['ori']
    recon =data['RE']
    gt =sio.loadmat("/data3/zhaoshuyi/Datasets/CAVE/hsi/test/chart_and_stuffed_toy_ms.mat")['data']/1.9321
    print(inputs.shape, ori.shape)
    Metric = iv.spectra_metric( recon, ori)

    #imsave(ori, gt.transpose(1,2,0), './', 1)

    PSNR =  Metric.PSNR()
    mse = Metric.MSE()
    SAM = Metric.SAM()
    SSIM = Metric.SSIM()
    print(PSNR)
    print(mse)
    print(SAM)
    print(SSIM)'''


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
    b, m, n = x.shape
    snr = 10**(SNR/10.0)
    xpower = np.sum(x**2)/(b*m*n)
    npower = xpower/snr
    return x + np.array(torch.randn_like(torch.from_numpy(x)))*np.sqrt(npower)


def gray2color(gray_array, color_map):

    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    for i in range(0, rows):
        for j in range(0, cols):
            # print(gray_array[i][j])
            color_array[i][j] = color_map[gray_array[i][j]]

    #color_image = Image.fromarray(color_array)

    return color_array


'''if __name__ == "__main__":
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
    color_jet.save("residual.png")'''
