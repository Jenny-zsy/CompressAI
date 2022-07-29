from audioop import reverse
from email.policy import default
from builtins import type
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models.NFC import NFC

from ratedistortionloss import RateDistortionLoss, RateDistortion_Loss
from utils import AverageMeter

from dataset import *

from models.ContextHyperprior import ContextHyperprior
from models.cheng2020attention import Cheng2020Attention
from models.NFC import NFC

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_lr,
    )
    return optimizer, aux_optimizer


def train_epoch(args, model, criterion, optimizer, aux_optimizer,
                train_dataloader, epoch, epochs, f):
    """
        Train model for one epoch
    """

    model.train()  # Set model to training mode

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    logdet = AverageMeter()
    for batch, inputs in enumerate(train_dataloader):
        inputs = Variable(inputs.to(args.device))

        optimizer.zero_grad()

        # forward
        out = model(inputs)
        '''with torch.no_grad():
            out["x_hat"] = model.encoder(out["y_hat"], reverse=True)'''
        out_criterion = criterion(out, inputs)

        # backward
        #print(out_criterion["loss"])
        out_criterion["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        loss.update(out_criterion["loss"])
        bpp_loss.update(out_criterion["bpp_loss"])
        mse_loss.update(out_criterion["mse_loss"])
        logdet.update(out_criterion["logdet"])

        # print out loss and visualise results
        if batch % 10 == 0:
            print(
                'Epoch {}/{}:[{}]/[{}]'.format(
                    epoch, epochs, batch, len(train_dataloader)).ljust(30),
                'Loss: %.4f'.ljust(14) % (out_criterion["loss"]),
                'mse_loss: %.6f'.ljust(19) % (out_criterion["mse_loss"]),
                'logdet: %.4f'.ljust(19) % (out_criterion["logdet"]),
                'bpp_loss: %.4f'.ljust(17) % (out_criterion["bpp_loss"]),
                'aux_loss: %.2f'.ljust(17) % (aux_loss.item()))
            f.write('Epoch: {}/{}:[{}]/[{}]'.format(
                epoch, epochs, batch, len(train_dataloader)).ljust(30))
            f.write('Loss: %.4f'.ljust(14) % (out_criterion["loss"]))
            f.write('mse_loss: %.6f'.ljust(19) % (out_criterion["mse_loss"]))
            f.write('logdet: %.4f'.ljust(19) % (out_criterion["logdet"]))
            f.write('bpp_loss: %.4f'.ljust(17) % (out_criterion["bpp_loss"]))
            f.write('aux_loss: %.2f\n'.ljust(17) % (aux_loss.item()))

    return loss.avg, mse_loss.avg, bpp_loss.avg, logdet.avg


def test_epoch(args, model, criterion, test_dataloader, epoch, f):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = Variable(inputs.to(args.device))
            out = model(inputs)
            #out["x_hat"] = model.encoder(out["y_hat"], reverse=True)
            out_criterion = criterion(out, inputs)

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            loss.update(out_criterion["loss"])

    f.write('Epoch {} valid '.format(epoch).ljust(30))
    f.write('Loss: %.4f'.ljust(14) % (loss.avg))
    f.write('mse_loss: %.6f'.ljust(19) % (mse_loss.avg))
    f.write('bpp_loss: %.4f\n'.ljust(17) % (bpp_loss.avg))

    print('Epoch {} valid '.format(epoch).ljust(30),
          'Loss: %.4f'.ljust(14) % (loss.avg),
          'mse_loss: %.6f'.ljust(19) % (mse_loss.avg),
          'bpp_loss: %.4f'.ljust(17) % (bpp_loss.avg))

    return loss.avg, mse_loss.avg, bpp_loss.avg


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


def main(args):
    gpu_num = len(args.gpus.split(','))
    device_ids = list(range(gpu_num))

    # load dataset
    if args.train_data == 'CLIC':
        path = '/data3/zhaoshuyi/Datasets/CLIC2020/'
        train_dataset = ImageFolder(path, args.patch_size, split='train')
        valid_dataset = ImageFolder(path, args.patch_size, split='valid')
    elif args.train_data == 'ImageNet':
        path = '/data4/langzhiqiang/ImageNet_ILSVRC2012/train/'
        #path = '/data3/zhaoshuyi/Datasets/ImageNet/train/'
        train_dataset = data_list(path, args.patch_size, mode='train')
        valid_dataset = data_list(path, args.patch_size, mode='valid')
    elif args.train_data == 'Flickr':
        path = '/data4/langzhiqiang/sr_data/Flickr1024/'
        train_dataset = Flickr(path, args.patch_size, split='Train')
        valid_dataset = Flickr(path, args.patch_size, split='Validation')

    # create data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * gpu_num,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.test_batch_size * gpu_num,
        num_workers=args.num_workers,
        shuffle=False,
    )
    save_path = '../../compressresults/{}_{}_alpha{}_beta{}_bs{}_ReduceLR{}_block{}_step{}_{}_{}/'.format(
        args.model, args.train_data,
        args.alpha, args.beta, args.batch_size * gpu_num, args.lr, args.block, args.step, args.flow_permutation, args.flow_coupling+args.LU_decomposed if args.flow_coupling == 'affine' else args.flow_coupling)
    model = NFC(block_num=args.block, step_num=args.step,patch_size=args.patch_size, flow_permutation=args.flow_permutation,
                flow_coupling=args.flow_coupling, LU_decomposed=args.LU_decomposed)
    model.to(args.device)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writter = SummaryWriter(os.path.join(
        '../../compressresults/tensorboard', save_path.split('/')[-2]))

    #criterion = RateDistortionLoss(args.lmbda)
    criterion = RateDistortion_Loss(args.alpha, args.beta)
    criterion.cuda()

    optimizer, aux_optimizer = configure_optimizers(model, args)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)

    # load model and continue training
    if args.continue_training:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
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
    valid_loss_sum = []
    for epoch in range(start_epoch, args.epochs):

        writter.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_mse, train_bpp, logdet = train_epoch(args, model, criterion, optimizer,
                                                               aux_optimizer, train_dataloader, epoch,
                                                               args.epochs, f)
        valid_loss, valid_mse, valid_bpp = test_epoch(args, model, criterion, valid_dataloader,
                                                      epoch, f)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        lr_scheduler.step(valid_loss)

        train_loss_sum.append(train_loss.cpu().detach().numpy())
        valid_loss_sum.append(valid_loss.cpu().detach().numpy())

        writter.add_scalars('loss/total', {'train': train_loss.cpu().detach().numpy(),
                                           'valid': valid_loss.cpu().detach().numpy()}, epoch)
        writter.add_scalars('loss/bpp_loss', {'train': train_bpp.cpu().detach().numpy(),
                                              'valid': valid_bpp.cpu().detach().numpy()}, epoch)
        writter.add_scalars('loss/mse_loss', {'train': train_mse.cpu().detach().numpy(),
                                              'valid': valid_mse.cpu().detach().numpy()}, epoch)
        writter.add_scalar('loss/logdet', logdet.cpu().detach().numpy(), epoch)

        # save the model
        state = {
            'epoch': epoch,
            'state_dict': model.module.state_dict() if gpu_num > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(
            state,
            os.path.join(save_path, "lastcheckpoint.pth"))

        if epoch % 50 == 49:
            torch.save(
                state,
                os.path.join(save_path, "checkpoint_{}.pth".format(epoch + 1)))
        '''state = {
            'epoch': epoch,
            'state_dict': model.module.state_dict() if gpu_num > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "aux_optimizer": aux_optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        torch.save(
            state,
            os.path.join(save_path, "checkpoint_{}.pth".format(epoch + 1)))'''
        plot(train_loss_sum, valid_loss_sum, 'loss', save_path)
        writter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_data',
                        type=str,
                        choices=['ImageNet', 'CLIC', 'Flickr'],
                        default='CLIC',
                        help='data for training')
    parser.add_argument('--aug',
                        type=str,
                        default='aug',
                        help='whether to augment data.')
    parser.add_argument("--patch_size",
                        type=int,
                        default=256,
                        help="Size of the patches to be cropped")
    parser.add_argument("--stride",
                        type=int,
                        default=128,
                        help="Stride when crop paches")
    parser.add_argument("-n",
                        "--num-workers",
                        type=int,
                        default=4,
                        help="Dataloaders threads (default: %(default)s)")
    parser.add_argument('--model',
                        type=str,
                        default='NFC',
                        help='Model architecture')
    parser.add_argument('--block', type=int,
                        default=2,
                        help="how many blocks in encoder")
    parser.add_argument('--step', type=int,
                        default=4,
                        help="how many steps in flowblock")
    parser.add_argument('--slice', type=int, default=8)
    parser.add_argument('--flow_permutation', type=str,
                        choices=['shuffle', 'invconv'],
                        default='shuffle',
                        help='channel shuffle')
    parser.add_argument('--flow_coupling', type=str,
                        choices=['additive', 'affine'],
                        default='additive')
    parser.add_argument('--LU_decomposed', type=bool, default=False)
    parser.add_argument('--channel_N', type=int, default=192)
    parser.add_argument('--channel_M', type=int, default=192)
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='number of epoch for training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=8,
                        help='number of batch_size for training')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=4,
                        help='number of batch_size for testing')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='path to the folder with grayscale images')
    parser.add_argument("--aux_lr",
                        default=1e-3,
                        help="Auxiliary loss learning rate.")
    parser.add_argument("--alpha",
                        type=float,
                        default=1e-2,
                        help="Bit-rate distortion parameter")
    parser.add_argument("--beta",
                        type=float,
                        default=1,
                        help="sam loss parameter")
    parser.add_argument('--continue_training',
                        type=bool,
                        default=False,
                        help='whether to use pretrained model from the checkpoint file')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='compression_model.pth',
                        help='path where to save checkpoint during training')

    parser.add_argument("--milestones",
                        type=list,
                        default=[5],
                        help="how many epoch to reduce the lr")
    parser.add_argument("--gamma",
                        type=int,
                        default=0.5,
                        help="how much to reduce the lr each time")
    parser.add_argument("--gpus", type=str, default="0", help='path log files')
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    USE_CUDA = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if USE_CUDA else "cpu")
    print(args)

    main(args)
    print("Done.")
