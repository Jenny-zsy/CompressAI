import os
import h5py
import glob
import random
import numpy as np
import torch

from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

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
    path = os.path.join(path, mode)
    h5f = h5py.File(path+'/{}.h5'.format(mode), 'w')
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
                
    h5f.close()
    print('{} {} samples\n'.format(num-1, mode))

class CLIC_dataset(Dataset):
    """Load database. 

    Args:
        root (string): root directory of the h5 file
        mode: "train" or "valid"
    """

    def __init__(self, root, mode="train"):
        self.mode = mode
        self.h5f = h5py.File(os.path.join(root, '{}.h5'.format(mode)), 'r')
        self.keys = list(self.h5f.keys())
        if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        key = self.keys[index]
        #print(key)
        data = np.array(self.h5f[key])
        img = torch.Tensor(data)
        #print(img.shape)
        return img

    def __len__(self):
        return len(self.keys)

class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob.glob(os.path.join(self.data_dir, "*.*")))

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __len__(self):
        return len(self.image_path)

from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    root = '/data1/zhaoshuyi/Datasets/CLIC2020/'
    gen_data(root,256,256,'train')
    gen_data(root,256,256, 'valid')
    '''
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(256), transforms.ToTensor()]
    )
    train_dataset = ImageFolder(root=root, transform=train_transforms, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True,
    )
    for batch, (inputs) in enumerate(train_dataloader):
        inputs = inputs.cuda()
        print(inputs.shape)'''
