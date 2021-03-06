import os
import h5py
import glob
import numpy as np

from torchvision import transforms
from PIL import Image


def Im2Patch(img, win, stride=1):
    """
    crop images

    return: imgs with shape(patch_size, patch_size, channel, num)
    """
    k = 0
    endw = img.shape[0]
    endh = img.shape[1]
    endc = img.shape[2]

    patch = img[0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride, :]
    TotalPatNum = patch.shape[0] * patch.shape[1]
    Y = np.zeros([win * win, TotalPatNum, endc], np.uint8)
    for i in range(win):
        for j in range(win):
            patch = img[i:endw - win + i + 1:stride, j:endh - win + j + 1:stride, :]
            Y[k, :, :] = np.array(patch[:]).reshape(TotalPatNum, endc)
            k = k + 1
    return Y.reshape([win, win, endc, TotalPatNum])

def gen_data_from_list(path, txt, patch_size, stride):
    global toTensor, save_path

    patch_num = 0
    list = np.loadtxt(txt, dtype=np.str)
    for pic in range(len(list)):

        img_path = os.path.join(path, list[pic][0])
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        img.save(os.path.join(save_path, 'test.png'))
        h, w = img.size
        #img = np.array(img)

        if h < patch_size or w < patch_size:
            continue
        
        for i in range(0, h-stride+1, stride):
            for j in range(0, w-stride+1, stride):
                patch_num += 1
                #print(i, j, i+patch_size, j+patch_size)
                patch = img.crop((i, j, i+patch_size, j+patch_size))
                patch.save(os.path.join(save_path, '{}.png'.format(patch_num)))
                
    print('{} samples\n'.format(patch))

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

    toTensor=transforms.ToTensor()# ???????????????toTensor
    #h5f = h5py.File('./data/train_ImageNet.h5', 'w')

    #path = '/data1/zhaoshuyi/Datasets/CLIC2020/'
    #path = '/data1/zhaoshuyi/Datasets/COCO/train2017/'
    path = '/data1/langzhiqiang/ImageNet_ILSVRC2012/train/'
    save_path = '/data1/zhaoshuyi/Datasets/ImageNet/compress/'
    txt = './data/train.txt'
    gen_data_from_list(path=path, txt=txt, patch_size=256, stride=256)
    #h5f.close()