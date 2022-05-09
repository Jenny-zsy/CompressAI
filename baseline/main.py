from ast import arg
from asyncore import write
from cgi import test
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision

from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from models.model import ContextHyperprior
from ratedistortionloss import RateDistortionLoss
from utils import AverageMeter
from dataset import CLIC_dataset


def train_epoch(args, model, criterion, optimizer, train_dataloader, epoch, epochs,
                f):
    """
	Train model for one epoch
	"""
    loss = AverageMeter()
    mse_loss = AverageMeter()
    bpp_loss = AverageMeter()
    model.train()  # Set model to training mode

    for batch, inputs in enumerate(train_dataloader):
        inputs = Variable(inputs.to(args.device))

        optimizer.zero_grad()

        # forward
        out = model(inputs)

        out_criterion = criterion(out, inputs)

        # backward
        out_criterion["loss"].backward()
        optimizer.step()

        # keep track of loss
        loss.update(out_criterion["loss"].item(), inputs.size(0))
        mse_loss.update(out_criterion["mse_loss"].item(), inputs.size(0))
        bpp_loss.update(out_criterion["bpp_loss"].item(), inputs.size(0))

        if loss.avg>10 and epoch>1:
            for name, parms in model.named_parameters():	
                print('-->name:', name, '-->grad_requirs:',parms.requires_grad, ' -->grad_value:',parms.grad)

        # print out loss and visualise results
        if batch % 10 == 0:
            print(
                'Epoch {}/{}:[{}]/[{}]'.format(
                    epoch, epochs, batch, len(train_dataloader)).ljust(30),
                'Loss: %.4f'.ljust(14) % (loss.avg),
                'mse_loss: %.6f'.ljust(19) % (mse_loss.avg),
                'bpp_loss: %.4f'.ljust(17) % (bpp_loss.avg))
            f.write('Epoch {}/{}:[{}]/[{}]'.format(
                epoch, epochs, batch, len(train_dataloader)).ljust(30))
            f.write('Loss: %.4f'.ljust(14) % (loss.avg))
            f.write('mse_loss: %.6f'.ljust(19) % (mse_loss.avg))
            f.write('bpp_loss: %.4f\n'.ljust(17) % (bpp_loss.avg))

    return loss.avg


def test_epoch(args, model, criterion, test_dataloader, epoch, f):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = Variable(inputs.to(args.device))
            out = model(inputs)
            out_criterion = criterion(out, inputs)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

        f.write('Epoch {} valid '.format(epoch).ljust(30))
        f.write('Loss: %.4f'.ljust(14) % (loss.avg))
        f.write('mse_loss: %.6f'.ljust(19) % (mse_loss.avg))
        f.write('bpp_loss: %.4f\n'.ljust(17) % (bpp_loss.avg))

    print('Epoch {} valid '.format(epoch).ljust(30),
          'Loss: %.4f'.ljust(14) % (loss.avg),
          'mse_loss: %.6f'.ljust(19) % (mse_loss.avg),
          'bpp_loss: %.4f'.ljust(17) % (bpp_loss.avg))

    return loss.avg


def plot(y1, y2, label, outf):
    x = np.arange(0, len(y1), 1)
    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='valid')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(label)

    plt.savefig(outf + label + '.jpg')
    plt.cla()
    plt.close("all") 


def train(args):
    gpu_num = len(args.gpus.split(','))
    device_ids = list(range(gpu_num))

    save_path = './results/lr{}_bs{}_lambda{}/'.format(args.lr, args.batch_size, args.lmbda)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size),
         transforms.ToTensor()])
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size),
         transforms.ToTensor()])

    # load dataset
    train_dataset = CLIC_dataset(args.dataset,
                                mode="train")
    test_dataset = CLIC_dataset(args.dataset,
                               mode="valid")

    # create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size*gpu_num,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size*gpu_num,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = ContextHyperprior()

    criterion = RateDistortionLoss(args.lmbda)
    criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5,20,40,80], gamma=0.5)

    # load model and continue training
    if args.continue_training:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint['epoch']
        f = open(save_path + 'loss.txt', 'a+')
    else:
        start_epoch = 0
        f = open(save_path + 'loss.txt', 'w+')

    # move model to gpu and show structure
    if gpu_num > 1:
        model = nn.DataParallel(model,
                                device_ids=device_ids,
                                output_device=device_ids[0])
    model.to(args.device)
    

    # training
    train_loss_sum = []
    test_loss_sum = []
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(args, model, criterion, optimizer, train_dataloader,
                                 epoch, args.epochs, f)
        test_loss = test_epoch(args, model, criterion, test_dataloader, epoch, f)
        lr_scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        train_loss_sum.append(train_loss)
        test_loss_sum.append(test_loss)

        # save the model
        if epoch % 10 == 9:
            state = {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if gpu_num>1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(
                state,
                os.path.join(save_path, "checkpoint_{}.pth".format(epoch + 1)))
        plot(train_loss_sum, test_loss_sum, 'loss', save_path)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-epochs',
                        type=int,
                        default=2000,
                        help='number of epoch for training')
    parser.add_argument('-batch_size',
                        type=int,
                        default=8,
                        help='number of batch_size for training')
    parser.add_argument('-test_batch_size',
                        type=int,
                        default=4,
                        help='number of batch_size for testing')
    parser.add_argument("-n",
                        "--num-workers",
                        type=int,
                        default=1,
                        help="Dataloaders threads (default: %(default)s)")
    parser.add_argument('-continue_training',
						type=bool,
        				default=False,
        				help='whether to use pretrained model from the checkpoint file')
    parser.add_argument('-checkpoint',
                        type=str,
                        default='compression_model.pth',
                        help='path where to save checkpoint during training')
    parser.add_argument('-dataset',
                        type=str,
                        default='/data1/zhaoshuyi/Datasets/CLIC2020/',
                        help='path to the folder with images')
    parser.add_argument("--patch-size",
						type=int,
						nargs=2,
						default=(256, 256),
						help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument('-lr',
                        type=float,
                        default=1e-4,
                        help='path to the folder with grayscale images')
    parser.add_argument("--gpus", type=str, default="0", help='path log files')
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lmbda",
						type=float,
						default=1e-2,
						help="Bit-rate distortion parameter (default: %(default)s)")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    USE_CUDA = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if USE_CUDA else "cpu")

    train(args)
    print("Done.")
