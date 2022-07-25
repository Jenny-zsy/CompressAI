from PIL import Image
import os
from cv2 import add
import numpy as np
import torch
import scipy.io as sio
import imgvision as iv
from models.NFC import NFC, FlowNet
from torchvision import transforms

def add_noise_Gauss(img,loc=0,scale=0.005):
    """
    添加高斯噪声
    :param img: 输入灰度图像
    :param loc: 高斯均值
    :param scale: 高斯方差
    :param scale**2:高斯标准差
    :return: img：加入高斯噪声后的图像
    """
 
    img = np.float32(img)/255  # (0-255)->(0-1)
    Gauss_noise =np.random.normal(loc,scale,img.shape)
    img = img + Gauss_noise  # 加性噪声
    if img.min()<0:
        low_clip = -1
    else:
        low_clip = 0
    img = np.clip(img,low_clip,1.0)  # 约束结果在(0-1)
    img = np.uint8(img*255)  # (0-1)->(0-255)
   
    return img

def awgn(x, snr, seed=7):
    m,n,c = x.shape
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    x = np.float32(x)/255  # (0-255)->(0-1)

    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / (c*m*n)
    npower = xpower / snr
    print(npower)
    noise = np.random.randn(m,n,c) * np.sqrt(npower)
    img = x+noise
    img = np.uint8(img*255)
    return img

if __name__ == "__main__":
    '''img = Image.open("/data3/zhaoshuyi/Datasets/CUB_200_2011/images/018.Spotted_Catbird/Spotted_Catbird_0024_796791.jpg").convert("RGB")
    t =transforms.ToTensor()
    #img  = t(img)
    image_arr = np.array(img)/255.
    print(image_arr.shape)
    w,h,c = image_arr.shape
    imageNoiseSigma = np.eye(3)*(50/255)**2
    print(imageNoiseSigma)
    noise = np.random.multivariate_normal(np.zeros(3), imageNoiseSigma, w*h)
    
    noise = np.reshape(noise, image_arr.shape)
    #print(noise)
    #print(image_arr.shape)
    #noise = torch.FloatTensor(img.size()).normal_(mean=0, std=20/255.)
    #noisy = add_noise_Gauss(image_arr,0,50/255)
    #print(type(image_arr.shape+ noise.shape))
    img = image_arr+noise
    print(img)
    img = np.clip(np.uint8(img*255),0,255)
    #print(img)
    img =  Image.fromarray(img)
    print(type(img), type(noise))
    tran =  transforms.ToPILImage()
    img = tran(img+noise)
    img.save('temp.png')'''
    '''reconstructed_image = Image.fromarray(np.uint8(img+noise))
    reconstructed_image.save('temp.png')'''
    model = NFC(3)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    y= out['x_hat']

    print(y.shape)