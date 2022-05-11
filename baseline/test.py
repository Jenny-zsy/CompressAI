from ast import arg
import imghdr
from PIL import Image
import os
import json
import argparse

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from pytorch_msssim import ms_ssim
from collections import defaultdict

from images.plot import imsave
from models.model import ContextHyperprior
from dataset import TestDataset

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def test_checkpoint(model, test_loader, args):
    with torch.no_grad():
        model.eval()

        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0

        for i, img in enumerate(test_loader):
            img = img.to(args.device)
            out = model(img)

            num_pixels = img.size(2) * img.size(3)
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out["likelihoods"].values())
            sumBpp += bpp

            x_hat = out["x_hat"]
            PSNR = psnr(img, x_hat)
            MS_SSIM = ms_ssim(img, x_hat, data_range=1.0)
            MS_SSIM_DB = -10 * (torch.log(1-MS_SSIM) / np.log(10))
            sumPsnr += PSNR
            sumMsssim += MS_SSIM.item()
            sumMsssimDB += MS_SSIM_DB.item()

            print("img{} psnr:{:.6f} ms-ssim:{:.6f} ms-ssim-DB:{:.6f} bpp:{:.6f}".format(
                i + 1, PSNR, MS_SSIM, MS_SSIM_DB, bpp.item()))
            #f.write("img{} psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(i+1, PSNR, MS_SSIM, bpp.item()))
            '''--------------------
            plot reconstructed_image and residual_image
            --------------------'''
            if args.ifplot:
                save_path = os.path.join(
                    model_path, "checkpoint{}".format(args.checkpoint))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                imsave(x_hat, img, save_path, i)

    print("Average psnr:{:.6f} ms-ssim:{:.6f} ms-ssim-DB:{:.6f} bpp:{:.6f}".format(
        sumPsnr / len(test_loader), sumMsssim / len(test_loader), sumMsssimDB / len(test_loader),
        sumBpp / len(test_loader)))
    #f.write("Average psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(sumPsnr/len(file_names), sumMsssim/len(file_names), sumBpp/len(file_names)))

    return {
        "psnr": sumPsnr / len(test_loader),
        "ms-ssim": sumMsssim / len(test_loader),
        "ms-ssim-DB": sumMsssimDB / len(test_loader),
        "bpp": sumBpp.item() / len(test_loader),
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
    parser.add_argument('--test_data', default='Kodak', help='test dataset')
    parser.add_argument('--model_path', default="/data1/zhaoshuyi/AIcompress/baseline/results/lr0.0001_bs32_lambda0.01/", help='checkpoint path')
    parser.add_argument('--channel_N', type=int, default=128)
    parser.add_argument('--channel_M', type=int, default=192)
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='number of epoch for eval')
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--ifplot', default=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    USE_CUDA = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if args.test_data == 'Kodak':
        img_path = "/data1/zhaoshuyi/AIcompress/compression_Liu/data/images/"
    test_dataset = TestDataset(data_dir=img_path)
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=1,
                             pin_memory=True)

    model_path = args.model_path
    model = ContextHyperprior(channel_N=args.channel_N, channel_M=args.channel_M)

    results = defaultdict(list)
    for i in range(10, args.epochs + 1, 100):
        if i == 0:
            continue
        checkpoint = torch.load(
            os.path.join(model_path, 'checkpoint_{}.pth'.format(i)))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(args.device)

        args.checkpoint = i

        metrics = test_checkpoint(model, test_loader, args)
        for k, v in metrics.items():
            results[k].append(v)

    description = (args.test_data)
    output = {
        "name": args.model_path,
        "description": f"Inference ({description})",
        "results": results,
    }
    
    with open(os.path.join(model_path, '{}.json'.format(args.test_data)), "w", encoding="utf-8") as json_file:
        json_dict = json.dump(output, json_file)
    print(json.dumps(output, indent=2))