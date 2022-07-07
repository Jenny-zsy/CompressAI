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

from ratedistortionloss import RateDistortionLoss, RateDistortion_SAM_Loss
from utils import AverageMeter

from dataset_hsi import CAVE_Dataset

from models.ContextHyperprior import ContextHyperprior
from models.cheng2020attention import Cheng2020Attention
from models.transformercompress import SymmetricalTransFormer
from models.TransformerHyperCompress import ChannelTrans

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
    sam_loss = AverageMeter()
    for batch, inputs in enumerate(train_dataloader):
        inputs = Variable(inputs.to(args.device))

        optimizer.zero_grad()

        # forward
        out = model(inputs)

        out_criterion = criterion(out, inputs)

        # backward
        out_criterion["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        loss.update(out_criterion["loss"])
        bpp_loss.update(out_criterion["bpp_loss"])
        mse_loss.update(out_criterion["mse_loss"])
        sam_loss.update(out_criterion["sam_loss"])

        # print out loss and visualise results
        if batch % 10 == 0:
            print(
                'Epoch {}/{}:[{}]/[{}]'.format(
                    epoch, epochs, batch, len(train_dataloader)).ljust(30),
                'Loss: %.4f'.ljust(14) % (out_criterion["loss"]),
                'mse_loss: %.6f'.ljust(19) % (out_criterion["mse_loss"]),
                'sam_loss: %.4f'.ljust(19) % (out_criterion["sam_loss"]),
                'bpp_loss: %.4f'.ljust(17) % (out_criterion["bpp_loss"]),
                'aux_loss: %.2f'.ljust(17) % (aux_loss.item()))
            f.write('Epoch: {}/{}:[{}]/[{}]'.format(
                epoch, epochs, batch, len(train_dataloader)).ljust(30))
            f.write('Loss: %.4f'.ljust(14) % (out_criterion["loss"]))
            f.write('mse_loss: %.6f'.ljust(19) % (out_criterion["mse_loss"]))
            f.write('sam_loss: %.4f'.ljust(19) % (out_criterion["sam_loss"]))
            f.write('bpp_loss: %.4f'.ljust(17) % (out_criterion["bpp_loss"]))
            f.write('aux_loss: %.2f\n'.ljust(17) % (aux_loss.item()))

    return loss.avg, mse_loss.avg, bpp_loss.avg, sam_loss.avg


def test_epoch(args, model, criterion, test_dataloader, epoch, f):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    sam_loss = AverageMeter()

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = Variable(inputs.to(args.device))
            out = model(inputs)
            out_criterion = criterion(out, inputs)

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            sam_loss.update(out_criterion["sam_loss"])
            loss.update(out_criterion["loss"])

    f.write('Epoch {} valid '.format(epoch).ljust(30))
    f.write('Loss: %.4f'.ljust(14) % (loss.avg))
    f.write('mse_loss: %.6f'.ljust(19) % (mse_loss.avg))
    f.write('sam_loss: %.6f'.ljust(19) % (sam_loss.avg))
    f.write('bpp_loss: %.4f\n'.ljust(17) % (bpp_loss.avg))

    print('Epoch {} valid '.format(epoch).ljust(30),
          'Loss: %.4f'.ljust(14) % (loss.avg),
          'mse_loss: %.6f'.ljust(19) % (mse_loss.avg),
          'sam_loss: %.4f'.ljust(19) % (sam_loss.avg),
          'bpp_loss: %.4f'.ljust(17) % (bpp_loss.avg))

    return loss.avg, mse_loss.avg, bpp_loss.avg, sam_loss.avg


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
    if args.train_data == 'CAVE':
        path = '/data3/zhaoshuyi/Datasets/CAVE/hsi/'
        bands = 31
        train_dataset = CAVE_Dataset(path,
                                     args.patch_size,
                                     args.stride,
                                     data_aug=True if args.aug == 'aug' else False,
                                     mode='train')
        valid_dataset = CAVE_Dataset(path, args.patch_size, mode='valid')

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
    save_path = '../../compressresults/{}_{}_lambda{}_beta{}_bs{}_ReduceLR{}/'.format(
        args.model, args.train_data,
        args.lmbda, args.beta, args.batch_size * gpu_num, args.lr)
    if args.model == 'mbt':
        model = ContextHyperprior(channel_in=bands,
                                  channel_N=args.channel_N,
                                  channel_M=args.channel_M,
                                  channel_out=bands)
    elif args.model == 'cheng2020':
        model = Cheng2020Attention(channel_in=bands,
                                   channel_N=args.channel_N,
                                   channel_M=args.channel_M,
                                   channel_out=bands)
    elif args.model == 'transformer':
        model = SymmetricalTransFormer(channel_in=bands)
    elif args.model == 'channelTrans':
        model = ChannelTrans(channel_in=bands)
        save_path = '../../compressresults/{}_{}_block{}_slice{}_lambda{}_beta{}_bs{}_ReduceLR{}/'.format(
        args.model, args.train_data, args.block, args.slice,
        args.lmbda, args.beta, args.batch_size * gpu_num, args.lr)
    model.to(args.device)

    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writter = SummaryWriter(os.path.join(
        '../../compressresults/tensorboard', save_path.split('/')[-2]))

    #criterion = RateDistortionLoss(args.lmbda)
    criterion = RateDistortion_SAM_Loss(args.lmbda, args.beta)
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

        train_loss, train_mse, train_bpp, train_sam = train_epoch(args, model, criterion, optimizer,
                                                                  aux_optimizer, train_dataloader, epoch,
                                                                  args.epochs, f)
        valid_loss, valid_mse, valid_bpp, valid_sam = test_epoch(args, model, criterion, valid_dataloader,
                                                                 epoch, f)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        lr_scheduler.step(valid_loss)

        train_loss_sum.append(train_loss.cpu().detach().numpy())
        valid_loss_sum.append(valid_loss.cpu().detach().numpy())

        writter.add_scalars('loss', {'train': train_loss.cpu().detach().numpy(),
                                     'valid': valid_loss.cpu().detach().numpy()}, epoch)
        writter.add_scalars('bpp_loss', {'train': train_bpp.cpu().detach().numpy(),
                                         'valid': valid_bpp.cpu().detach().numpy()}, epoch)
        writter.add_scalars('mse_loss', {'train': train_mse.cpu().detach().numpy(),
                                         'valid': valid_mse.cpu().detach().numpy()}, epoch)
        writter.add_scalars('sam_loss', {'train': train_sam.cpu().detach().numpy(),
                                         'valid': valid_sam.cpu().detach().numpy()}, epoch)

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
                        choices=['CAVE'],
                        default='CAVE',
                        help='data for training')
    parser.add_argument('--aug',
                        type=str,
                        default='aug',
                        help='whether to augment data.')
    parser.add_argument("--patch-size",
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
                        default='cheng2020',
                        help='Model architecture')
    parser.add_argument('--block', type=list,
                        default=[1,1,1,1],
                        help="how many blocks in encoder")
    parser.add_argument('--slice', type=int, default=8)
    parser.add_argument('--channel_N', type=int, default=192)
    parser.add_argument('--channel_M', type=int, default=192)
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help='number of epoch for training')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='number of batch_size for training')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=4,
                        help='number of batch_size for testing')
    parser.add_argument('-lr',
                        type=float,
                        default=1e-4,
                        help='path to the folder with grayscale images')
    parser.add_argument("--aux_lr",
                        default=1e-3,
                        help="Auxiliary loss learning rate.")
    parser.add_argument("--lmbda",
                        type=float,
                        default=1e-2,
                        help="Bit-rate distortion parameter")
    parser.add_argument("--beta",
                        type=float,
                        default=1e-2,
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
                        default=[10],
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
