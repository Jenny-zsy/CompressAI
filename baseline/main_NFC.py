from ast import arg
import os
import torch
import torch.nn as nn
import argparse

from torchvision import transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import *

from dataset import CUB_data

from models.NFC import FlowNet


try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter


def train_epoch(args, model, criterion, optimizer, train_dataloader, epoch, epochs, f):
    """
        Train model for one epoch
    """

    model.train()  # Set model to training mode

    losses = AverageMeter()
    bpd_losses = AverageMeter()
    recon_losses = AverageMeter()
    
    for batch, (GT, noise) in enumerate(train_dataloader):
        noise = Variable(noise.to(args.device))
        GT = Variable(GT.to(args.device))

        optimizer.zero_grad()
        
        

        # forward
        out, bpd = model(noise)
        with torch.no_grad():
            C = out.shape[1]
            noise, out_new = out.narrow(1, 0, C//4), out.narrow(1, C//4, C//4*3)
            out_new = torch.cat((torch.zeros(noise.shape).cuda(), out_new), 1)
            recon = model.decode(out_new)

        recon_loss = criterion(GT, recon).requires_grad_(True)
        loss = args.lmbda1*bpd + args.lmbda2*recon_loss
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print('Epoch {}/{}:[{}]/[{}]'.format(
                    epoch, epochs, batch, len(train_dataloader)).ljust(30),
                'Loss: %.4f'.ljust(14) % (loss),
                'recon_loss: %.6f'.ljust(19) % (recon_loss),
                'bpd_loss: %.4f'.ljust(19) % (bpd))


        # backward
       

        losses.update(loss.item())
        bpd_losses.update(bpd)
        recon_losses.update(recon_loss)

    return losses.avg, recon_losses.avg, bpd_losses.avg


def test_epoch(args, model, criterion, test_dataloader, epoch, f):
    model.eval()

    losses = AverageMeter()
    bpd_losses = AverageMeter()
    recon_losses = AverageMeter()

    with torch.no_grad():
        for (GT, noise) in test_dataloader:
            GT, noise = GT.to(args.device), noise.to(args.device)
            out, bpd = model(noise)

            C = out.shape[1]
            noise, out_new = out.narrow(1, 0, C//4), out.narrow(1, C//4, C//4*3)
            out_new = torch.cat((torch.zeros(noise.shape).cuda(), out_new), 1)
            recon = model(out_new, reverse=True)


            recon_loss = criterion(GT, recon)
            loss =  args.lmbda1*bpd + args.lmbda2*recon_loss

            losses.update(loss.item())
            bpd_losses.update(bpd)
            recon_losses.update(recon_loss)


    f.write('Epoch {} valid '.format(epoch).ljust(30))
    f.write('Loss: %.4f'.ljust(14) % (losses.avg))
    f.write('recon_loss: %.6f'.ljust(19) % (recon_losses.avg))
    f.write('bpd_loss: %.4f\n'.ljust(17) % (bpd_losses.avg))

    print('Epoch {} valid '.format(epoch).ljust(30),
          'Loss: %.4f'.ljust(14) % (losses.avg),
          'recon_loss: %.6f'.ljust(19) % (recon_losses.avg),
          'bpd_loss: %.4f'.ljust(17) % (bpd_losses.avg))

    return losses.avg, recon_losses.avg, bpd_losses.avg




def main(args):
    gpu_num = len(args.gpus.split(','))
    device_ids = list(range(gpu_num))

    # load dataset
    data_path = '/data3/zhaoshuyi/Datasets/CUB_200_2011/'
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size),
         transforms.ToTensor(), 
         preprocess])

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size),
         transforms.ToTensor(), 
         preprocess])
    train_dataset = CUB_data(data_path,
                                mode="train",
                                transform=train_transforms)
    valid_dataset = CUB_data(data_path,
                                mode="valid",
                                transform=test_transforms)
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
    save_path = '../../denoise_results/Flow_{}_lambda1{}_lambda2{}_bs{}_ReduceLR{}/'.format(args.train_data,
        args.lmbda1, args.lmbda2, args.batch_size * gpu_num, args.lr)
    model = FlowNet(3, patch_size=args.patch_size)
    
    model.to(args.device)

    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    writter = SummaryWriter(os.path.join(
        '../../denoise_results/tensorboard', save_path.split('/')[-2]))

    #criterion = RateDistortionLoss(args.lmbda)
    criterion = nn.L1Loss()

    optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    #lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, gamma=args.gamma)

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
    valid_loss_sum = []
    for epoch in range(start_epoch, args.epochs):

        writter.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss, train_recon, train_bpd = train_epoch(args, model, criterion, optimizer, train_dataloader, epoch,
                                                                  args.epochs, f)
        valid_loss, valid_recon, valid_bpd = test_epoch(args, model, criterion, valid_dataloader,
                                                                 epoch, f)
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        lr_scheduler.step()
        #print(train_loss.device, valid_loss.device)
        train_loss_sum.append(train_loss)
        valid_loss_sum.append(valid_loss)

        writter.add_scalars('loss', {'train': train_loss,
                                     'valid': valid_loss}, epoch)
        writter.add_scalars('recon_loss', {'train': train_recon,
                                         'valid': valid_recon}, epoch)                 
        writter.add_scalars('bpd_loss', {'train': train_bpd,
                                         'valid': valid_bpd}, epoch)

        # save the model
        state = {
            'epoch': epoch,
            'state_dict': model.module.state_dict() if gpu_num > 1 else model.state_dict(),
            'optimizer': optimizer.state_dict(),
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
                        default='CUB',
                        help='data for training')
    parser.add_argument("--patch-size",
                        type=int,
                        default=64,
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
                        default=2e-4,
                        help='path to the folder with grayscale images')
    parser.add_argument("--lmbda1",
                        type=float,
                        default=1,
                        help="Bit-rate distortion parameter")
    parser.add_argument("--lmbda2",
                        type=float,
                        default=1,
                        help="Bit-rate distortion parameter")
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
