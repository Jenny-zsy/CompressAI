import os
import h5py
import glob
import random
import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class h5dataset_train(Dataset):
    """Load database. 

    Args:
        root (string): root directory of the h5 file
        mode: "train" or "valid"
    """

    def __init__(self, mode, h5path):
        self.mode = mode
        self.h5path = h5path
        self.h5f = h5py.File(h5path, 'r')
        self.keys = list(self.h5f.keys())
        '''if self.mode == 'train':
            random.shuffle(self.keys)
        else:
            self.keys.sort()
        '''
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        key = self.keys[index]
        #key = '{}'.format(index)
        print(key)
        data = np.array(self.h5f[key])
        img = torch.Tensor(data)
        #print(img.shape)
        return img

    def __len__(self):
        #return len(self.keys)
        return 332345

class h5dataset(Dataset):
    """Load database. 

    Args:
        root (string): root directory of the h5 file
        mode: "train" or "valid"
    """

    def __init__(self, mode, h5path):

        self.mode = mode
        self.h5f = h5py.File(h5path, 'r')
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
        print(len(self.keys))
        key = self.keys[index]
        #print(key)
        data = np.array(self.h5f[key])
        img = torch.Tensor(data)
        #print(img.shape)
        return img

    def __len__(self):
        return len(self.keys)


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

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

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

from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    #h5path='/data1/zhaoshuyi/AIcompress/baseline/data/train_ImageNet.h5'
   
    train_dataset = h5dataset(mode="test", h5path='/data1/zhaoshuyi/AIcompress/baseline/data/train_CLIC.h5')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=False,
    )
    for batch, (inputs) in enumerate(train_dataloader):
        print(inputs.shape)
        break