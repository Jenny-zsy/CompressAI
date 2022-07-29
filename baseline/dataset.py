import os
import h5py
import glob
import random
import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class TestDataset(Dataset):
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

from pathlib import Path
class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, patch_size, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        transform = transforms.Compose(
        [transforms.RandomCrop(patch_size),
         transforms.ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class Flickr(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, patch_size, split="train"):
        #splitdir = Path(root) / split
        splitdir = Path(os.path.join(root,'{}/HR'.format(split)))

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        transform = transforms.Compose(
        [transforms.RandomCrop(patch_size),
         transforms.ToTensor()])
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class CUB_data(Dataset):


    def __init__(self, root, transform=None, mode="train"):
        self.root = root
        f_dir = os.path.join(root, '{}.txt'.format(mode))
        f = open(f_dir, 'r')
        self.datalist = f.readlines()
        #self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        img_path = os.path.join(self.root+'images', self.datalist[index][:-1])
        img = Image.open(img_path).convert("RGB")
        noise_path = os.path.join(self.root+'noise', self.datalist[index][:-1])
        noise = img = Image.open(noise_path).convert("RGB")
        if self.transform:
            return self.transform(img), self.transform(noise)
        return img

    def __len__(self):
        return len(self.datalist)
 
class data_list(Dataset):


    def __init__(self, root, patch_size, mode="train"):
        self.root = root
        f_dir = os.path.join('./data', '{}.txt'.format(mode))
        f = open(f_dir, 'r')
        self.datalist = f.readlines()
        #self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transforms.Compose(
        [transforms.RandomCrop(patch_size),
         transforms.ToTensor()])

    def __getitem__(self, index):
        #print(self.datalist[index].split(' ')[0])
        img_path = os.path.join(self.root, self.datalist[index].split(' ')[0])
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.datalist)
 

from torch.utils.data import DataLoader

if __name__ == "__main__":
    #h5path='/data1/zhaoshuyi/AIcompress/baseline/data/train_ImageNet.h5'
    transforms = transforms.Compose(
        [transforms.RandomCrop(128),
         transforms.ToTensor()])
    path = '/data3/zhaoshuyi/Datasets/CUB_200_2011/'
    train_dataset = CUB_data(path, transform=transforms,mode="valid")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=False,
    )
    for batch, (inputs, noise) in enumerate(train_dataloader):
        print(inputs.shape, noise.shape)
        break