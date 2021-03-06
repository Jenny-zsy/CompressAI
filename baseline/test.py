from audioop import reverse
import os
import json
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import math

from torch.utils.data import DataLoader
from torchvision import transforms
from collections import defaultdict

from torch.autograd import Variable

from dataset import  CUB_data,ImageFolder
from models.NFC import FlowNet,NFC
from utils import *
from images.plot import imsave

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def test_checkpoint(epoch, model, test_loader, args):
    num = 0
    with torch.no_grad():
        model.eval()

        sumPsnr = 0
        sumssim = 0

        for GT in test_loader:
            num += 1
            # GT, noise = GT.to(args.device), noise.to(args.device)
            inputs = Variable(GT.to(args.device))
            out = model(inputs)
            recon = model.encoder(out["y_hat"], reverse=True)
            '''out, bpd = model(noise)
            C = out.shape[1]
            noise, out_new = out.narrow(1, 0, C//4), out.narrow(1, C//4, C//4*3)
            out_new = torch.cat((torch.zeros(noise.shape).cuda(), out_new), 1)
            recon = model(out_new, reverse=True)
            #recon = model(out, reverse=True)'''

        
            GT, recon = GT.squeeze(0).cuda()*255, recon.squeeze(0).cuda()*255
            PSNR = calculate_psnr(GT, recon)
            sumPsnr += PSNR
            SSIM = calculate_ssim(GT, recon)
            sumssim += SSIM

            print("img{}/{} psnr:{:.6f} ssim:{:.6f}".format(num, len(test_loader),PSNR, SSIM))
            #f.write("img{} psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(i+1, PSNR, MS_SSIM, bpp.item()))
            '''--------------------
            plot reconstructed_image and residual_image
            --------------------'''
            if epoch==args.endepoch and args.ifplot:
                save_path = os.path.join(
                    model_path, "checkpoint{}".format(args.checkpoint))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                imsave(recon/255.0, GT/255.0, save_path, num)

    print("Average psnr:{:.6f} ms-ssim:{:.6f}".format(
        sumPsnr / len(test_loader), sumssim / len(test_loader)))
    #f.write("Average psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(sumPsnr/len(file_names), sumMsssim/len(file_names), sumBpp/len(file_names)))

    return {
        "psnr": sumPsnr / len(test_loader),
        "ms-ssim": sumssim / len(test_loader),
        }


class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--test_data', default='CUB', help='test dataset')
    parser.add_argument("--patch-size",
                        type=int,
                        default=256,
                        help="Size of the patches to be cropped")
    parser.add_argument('--model_path', default="/data1/zhaoshuyi/AIcompress/baseline/results/lr0.0001_bs32_lambda0.01/", help='checkpoint path')
    parser.add_argument('--endepoch',
                        type=int,
                        default=150,
                        help='number of epoch for eval')
    parser.add_argument('--startepoch',
                        type=int,
                        default=0,
                        help='number of epoch for eval')
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--ifplot', default=False)
    parser.add_argument('--checkpoint', default=False)
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
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    USE_CUDA = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if USE_CUDA else "cpu")
    path = '/data3/zhaoshuyi/Datasets/CLIC2020/'
    test_dataset = ImageFolder(path, args.patch_size, split='test')
    valid_dataset = ImageFolder(path, args.patch_size, split='valid')

    # create data loader
    '''train_dataloader = DataLoader(
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

    img_path = '/data3/zhaoshuyi/Datasets/CUB_200_2011/'
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size),
         transforms.ToTensor(), 
         preprocess])
    test_dataset = CUB_data(img_path, mode="test", transform=test_transforms)'''
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=1,
                             pin_memory=True)

    model_path = args.model_path
    model = NFC(block_num=args.block, step_num=args.step,patch_size=args.patch_size, flow_permutation=args.flow_permutation,
                flow_coupling=args.flow_coupling, LU_decomposed=args.LU_decomposed)
    results = defaultdict(list)
    if args.checkpoint:
        checkpoint = torch.load(
            os.path.join(model_path, 'lastcheckpoint.pth'))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)
        i = checkpoint['epoch']

        metrics = test_checkpoint(i, model, test_loader, args)
        for k, v in metrics.items():
            results[k].append(v)
    else:
        for i in range(args.startepoch, args.endepoch + 1, 50):
            if i == 0:
                continue
            model = NFC(block_num=args.block, step_num=args.step,patch_size=args.patch_size, flow_permutation=args.flow_permutation,
                flow_coupling=args.flow_coupling, LU_decomposed=args.LU_decomposed)
            checkpoint = torch.load(
                os.path.join(model_path, 'checkpoint_{}.pth'.format(i)))
            model.load_state_dict(checkpoint['state_dict'])
            model.to(args.device)
            args.checkpoint = i

            metrics = test_checkpoint(i, model, test_loader, args)
            for k, v in metrics.items():
                results[k].append(v)
            
            torch.cuda.empty_cache()

    description = (args.test_data)
    output = {
        "name": args.model_path,
        "description": f"Inference ({description})",
        "results": results,
    }
    
    with open(os.path.join(model_path, '{}.json'.format(args.test_data)), "w", encoding="utf-8") as json_file:
        json_dict = json.dump(output, json_file)
    print(json.dumps(output, indent=2))