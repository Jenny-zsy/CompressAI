from distutils import filelist
from distutils.filelist import FileList
import os
from traceback import format_exc
import h5py
import glob
import random
import numpy as np
import torch

from pathlib import Path
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

def gen_data(path, h5f, patch_size, stride, mode='train'):
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
    path = os.path.join(path, mode)
    
    file_names = glob.glob(os.path.join(path, '*.png'))
    file_names.sort()
    toTensor=transforms.ToTensor()# 实例化一个toTensor
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

def gen_filelist(path, num, f):
    Minsize = 0
    Minname = ''
    cnt = 0
    imgsizemap = {}
    folders = glob.glob(os.path.join(path, '*'))
    folders.sort()
    for folder in folders:
        print(folder)
        #floder_name = folder.split('/')[-1]
        imgs = glob.glob(os.path.join(folder, '*.JPEG'))
        imgs.sort()
        for img in imgs:
            cnt += 1
            imgname = os.path.join(img.split('/')[-2], img.split('/')[-1]) #Relative path
            imgsize = os.path.getsize(img)
            #print(imgname, imgsize)

            '''----------Maintain a dictionary with the largest num pictures. ----------'''
            if cnt < num:
                imgsizemap.setdefault(imgname, imgsize)
            elif cnt == num:
                imgsizemap.setdefault(imgname, imgsize)
                imgsort = sorted(imgsizemap.items(), key=lambda d:d[1], reverse=False)
                Minname = imgsort[0][0]
                Minsize = imgsort[0][1]
            else:
                if imgsize > Minsize:
                    del imgsizemap[Minname]
                    imgsizemap.setdefault(imgname, imgsize)
                    imgsort = sorted(imgsizemap.items(), key=lambda d:d[1], reverse=False)
                    Minname = imgsort[0][0]
                    Minsize = imgsort[0][1]
    np.savetxt(f, imgsort, fmt = "%s", delimiter = ' ', newline = '\n')

    '''for parent, dirnames, filenames in os.walk(path):
        #print(dirnames)
        for dirname in dirnames:
            print(dirname)
            '''
        #filenames.sort()
    '''for filename in filenames :
            cnt += 1
            fileDir = os.path.join(parent,filename)
            filesize = os.path.getsize(fileDir)
            filemap.setdefault(filename, filesize)
            #print(fileDir, filesize)
            if cnt > 10:
                break
    fileList = sorted(filemap.items(), key=lambda d:d[1], reverse=True)
    fileList = fileList[:num]
    np.savetxt(f, fileList, fmt = "%s", delimiter = ' ', newline = '\n')
'''
if __name__ == "__main__":

    path1 = '/data1/zhaoshuyi/Datasets/CLIC2020/'
    path2 = '/data1/zhaoshuyi/Datasets/COCO/train2017/'
    path = '/data1/langzhiqiang/ImageNet_ILSVRC2012/train/'
    txt = '/data1/zhaoshuyi/Datasets/COCO/train.txt'
    gen_filelist(path, 8000, txt)