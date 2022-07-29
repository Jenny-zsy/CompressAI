import os
import glob
import numpy as np
import random
from PIL import Image

def gen_filelist(path, num, f):
    Minsize = 256
    Minname = ''
    cnt = 0
    imgsizemap = {}
    imgsort = []
    folders = glob.glob(os.path.join(path, '*'))
    folders.sort()

    for folder in folders:

        #folder_name = folder.split('/')[-1]
        print(folder)
        imgs = glob.glob(os.path.join(folder, '*.JPEG'))
        imgs.sort()
        for img in imgs:
            #print(img)
            im = Image.open(img)
            w,h = im.size

            if w<256 or h<256:
                continue

            cnt += 1
            imgname = os.path.join(img.split('/')[-2], img.split('/')[-1]) #Relative path
            imgsize = w*h

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

if __name__ == "__main__":

    path = '/data4/langzhiqiang/ImageNet_ILSVRC2012/train/'
    txt = './data/train.txt'
    '''valid_list=random.sample(range(1,8500),500)
    valid_list.sort()

    f_path = open('./data/train.txt', 'r')
    f_train = open('./data/train_v.txt', 'w+')
    f_valid = open('./data/valid.txt', 'w+')

    contents = f_path.readlines()
    for  i  in range(len(contents)):
        if i in valid_list:
            f_valid.write(contents[i])
        else:
            f_train.write(contents[i])
    f_path.close()
    f_train.close()
    f_valid.close()'''
    gen_filelist(path, 8500, txt)
