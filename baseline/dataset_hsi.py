import torch
import os
import copy
import glob
import numpy as np
import torch.utils.data as data
import scipy.io as sio


class CAVE_Dataset(data.Dataset):
    def __init__(self, path, patch_size=128, stride=128, data_aug=False, mode='train'):
        super(CAVE_Dataset, self).__init__()

        self.path = os.path.join(path, mode)
        self.filelist = sorted(glob.glob(os.path.join(self.path, "*.mat")))
        self.patch_size = patch_size
        self.stride = stride
        self.data_aug = data_aug
        self.mode = mode
        self.image_size = 512
        self.patch_num = int(self.image_size/self.stride)


    def __getitem__(self, Index):


        patch_size = self.patch_size
        stride = self.stride
        patch_num = self.patch_num

        if self.data_aug:
            Aug = 2
        else:
            Aug = 1


        image_size = self.image_size
        Patches = patch_num**2
        image_index = int(Index/Aug/Patches)
        patch_index = int(Index/Aug%Patches)

        file_name = self.filelist[image_index]
        data = sio.loadmat(file_name)['data']

        #Generalization
        data = data/1.9321

        X = int(patch_index/patch_num) #X,Y is patch index in image
        Y = int(patch_index%patch_num)

        if X*stride+patch_size > image_size and Y*stride+patch_size <= image_size:
            sample = data[:, -patch_size:, Y * stride: Y * stride + patch_size]
        elif X*stride+patch_size <= image_size and Y*stride+patch_size > image_size:
            sample = data[:, X * stride:X * stride + patch_size, -patch_size:]
        elif X*stride+patch_size > image_size and Y*stride+patch_size > image_size:
            sample = data[:, -patch_size: , -patch_size: ]
        else:
            sample = data[:, X * stride:X * stride + patch_size, Y * stride:Y * stride + patch_size]


        # Data augmantation
        if self.data_aug and self.mode=='train':
            if Index%2 == 1:
                a = np.random.randint(0,6,1)
                if a[0] == 0:
                    sample = copy.deepcopy(np.flip(sample, 1))  # flip the array upside down
                elif a[0] == 1:
                    sample = copy.deepcopy(np.flip(sample, 2))  # flip the array left to right
                elif a[0] == 2:
                    sample = copy.deepcopy(np.rot90(sample, 1, [1, 2]))  # Rotate 90 degrees clockwise
                elif a[0] == 3:
                    sample = copy.deepcopy(np.rot90(sample, -1, [1, 2]))  # Rotate 90 degrees counterclockwise
                elif a[0] == 4:
                    sample = copy.deepcopy(np.roll(sample, int(sample.shape[1] / 2), 1))  # Roll the array up
                elif a[0] == 5:
                    sample = np.roll(sample, int(sample.shape[1] / 2), 1)  # Roll the array up & left
                    sample = copy.deepcopy(np.roll(sample, int(sample.shape[2] / 2), 2))
        
        sample = torch.from_numpy(sample).type(torch.FloatTensor)
        print(int(self.patch_num**2*len(self.filelist)*Aug))
        #print(sample.shape,mask.shape)
        return sample


    def __len__(self):

        if self.data_aug:
            Aug = 2
        else:
            Aug = 1

        return int(self.patch_num**2*len(self.filelist)*Aug)

from torch.utils.data import DataLoader
if __name__ == "__main__":
    #h5path='/data1/zhaoshuyi/AIcompress/baseline/data/train_ImageNet.h5'
    path = '/data1/zhaoshuyi/Datasets/CAVE/hsi/'
    train_dataset = CAVE_Dataset(path, stride=64, data_aug=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=False,
    )
    for batch, (inputs) in enumerate(train_dataloader):
        print(inputs.shape)
        break
    '''valid_dataset = CAVE_Dataset(path,mode='valid')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=4,
        num_workers=1,
        shuffle=False,
    )
    for batch, (inputs) in enumerate(valid_dataloader):
        print(inputs.shape)
        break'''