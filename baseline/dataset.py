import os
import h5py
import glob
import random
import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

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

from torchvision import transforms
from torch.utils.data import DataLoader


if __name__ == "__main__":

    root = '/data1/zhaoshuyi/Datasets/CLIC2020/'
    
    train_dataset = CLIC_dataset(root=root, mode="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=True,
    )
    for batch, (inputs) in enumerate(train_dataloader):
        inputs = inputs.cuda()
        print(inputs.shape)
