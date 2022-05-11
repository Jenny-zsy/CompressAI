import os
import glob
import numpy as np

def gen_filelist(path, num, f):
    Minsize = 0
    Minname = ''
    cnt = 0
    imgsizemap = {}
    imgsort = []
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

if __name__ == "__main__":

    path = '/data1/langzhiqiang/ImageNet_ILSVRC2012/train/'
    txt = './data/train.txt'
    gen_filelist(path, 8500, txt)
