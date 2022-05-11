import os
import h5py
import glob
import numpy as np

from torchvision import transforms
from PIL import Image


def Im2Patch(img, win, stride=1):
    """
    crop images

    return: imgs with shape(channel, patch_size, patch_size, num)
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def gen_data_from_list(path, txt, patch_size, stride):
    global toTensor, h5f

    num = 0
    list = np.loadtxt(txt, dtype=np.str)
    for i in range(len(list)):
        img_path = os.path.join(path, list[i][0])
        img = Image.open(img_path).convert("RGB")
        img = toTensor(img)

        patches = Im2Patch(img, patch_size, stride)
        patch_num = patches.shape[3]
        print('{} has {} samples after patch'.format(list[i][0], patch_num))

        for j in range(patch_num):
            h5f.create_dataset(str(num), data=patches[:,:,:,j], dtype=np.float32)
            num += 1
                
    print('{} samples\n'.format(num))

def gen_data(path, patch_size, stride, mode='train'):
    """
    generate train or test dataset, save in h5 file

    args:
        path: root directory of images
        patch_size: size of image to be crop
        stride: stride when crop image
        mode: "train" or "test"

    -path/
        -train/
            -img01.png
            -img01.png
            ...
        -valid/
            -img01.png
            -img01.png
            ...
    """
    global toTensor, h5f

    path = os.path.join(path, mode)
    file_names = glob.glob(os.path.join(path, '*.png'))
    file_names.sort()
    num = 1
    for i in range(len(file_names)):
        print(file_names[i])
        img = Image.open(file_names[i]).convert("RGB")
        img = toTensor(img)

        patches = Im2Patch(img, patch_size, stride)
        patch_num = patches.shape[3]
        print('{} has {} samples after patch'.format(file_names[i], patch_num))

        for j in range(patch_num):
            h5f.create_dataset(str(num), data=patches[:,:,:,j], dtype=np.float32)
            num += 1
                
    print('{} {} samples\n'.format(num-1, mode))

if __name__ == "__main__":

    toTensor=transforms.ToTensor()# 实例化一个toTensor
    h5f = h5py.File('./data/train.h5', 'w')

    #path = '/data1/zhaoshuyi/Datasets/CLIC2020/'
    #path = '/data1/zhaoshuyi/Datasets/COCO/train2017/'
    path = '/data1/langzhiqiang/ImageNet_ILSVRC2012/train/'
    txt = './data/train.txt'
    gen_data_from_list(path=path, txt=txt, patch_size=256, stride=256)
    h5f.close()