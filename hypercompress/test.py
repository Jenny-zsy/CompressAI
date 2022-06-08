import os
import json
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import math
import imgvision as iv

from torch.utils.data import DataLoader
from collections import defaultdict
from images.plot import imsave

from dataset_hsi import CAVE_Dataset

from models.ContextHyperprior import ContextHyperprior
from models.cheng2020attention import Cheng2020Attention

def test_checkpoint(model, test_loader, args):
    with torch.no_grad():
        model.eval()

        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumSAM = 0
        for i, data in enumerate(test_loader):
            img = data['data'].to(args.device)
            out = model(img)
            name = ''.join(data['name'])
            num_pixels = img.size(2) * img.size(3)
            bpp = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in out["likelihoods"].values())
            sumBpp += bpp

            x_hat = out["x_hat"].squeeze()

            '''--------------------
            plot reconstructed_image and residual_image
            --------------------'''
            if args.ifplot and args.checkpoint == args.epochs:
                save_path = os.path.join(
                    model_path, "checkpoint{}".format(args.checkpoint))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                imsave(x_hat, img.squeeze(), save_path, i)

            '''--------------------
            compute metric: PSNR, MS-SSIM, SAM
            --------------------'''
            x_hat = x_hat.permute(1,2,0).cpu().numpy()
            img = img.squeeze().permute(1,2,0).cpu().numpy()
            #PSNR = psnr(img, x_hat)
            Metric = iv.spectra_metric(img, x_hat)
            PSNR =  Metric.PSNR()
            MS_SSIM =  Metric.SSIM()
            MS_SSIM_DB = -10 * (np.log(1-MS_SSIM) / np.log(10))
            SAM =  Metric.SAM()

            sumPsnr += PSNR
            sumMsssim += MS_SSIM.item()
            sumMsssimDB += MS_SSIM_DB.item()
            sumSAM += SAM

            print("img {} psnr:{:.6f} ms-ssim:{:.6f} ms-ssim-DB:{:.6f} sam:{:.4f} bpp:{:.6f}".format(
                name, PSNR, MS_SSIM, MS_SSIM_DB, SAM, bpp.item()))
            #f.write("img{} psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(i+1, PSNR, MS_SSIM, bpp.item()))
            

    print("Average psnr:{:.6f} ms-ssim:{:.6f} ms-ssim-DB:{:.6f} sam{:.4f} bpp:{:.6f}".format(
        sumPsnr / len(test_loader), sumMsssim / len(test_loader), sumMsssimDB / len(test_loader), sumSAM / len(test_loader), 
        sumBpp / len(test_loader)))
    #f.write("Average psnr:{:.6f} ms-ssim:{:.6f} bpp:{:.6f}\n".format(sumPsnr/len(file_names), sumMsssim/len(file_names), sumBpp/len(file_names)))

    return {
        "psnr": sumPsnr / len(test_loader),
        "ms-ssim": sumMsssim / len(test_loader),
        "ms-ssim-DB": sumMsssimDB / len(test_loader),
        "sam": sumSAM / len(test_loader),
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
    parser.add_argument('--model',
                        type=str,
                        default='mbt',
                        help='Model architecture')
    parser.add_argument('--test_data', default='CAVE', help='test dataset')
    parser.add_argument('--model_path', default="/data1/zhaoshuyi/AIcompress/baseline/results/lr0.0001_bs32_lambda0.01/", help='checkpoint path')
    parser.add_argument('--channel_N', type=int, default=192)
    parser.add_argument('--channel_M', type=int, default=192)
    parser.add_argument('--endepoch',
                        type=int,
                        default=1000)
    parser.add_argument('--startepoch',
                        type=int,
                        default=0)
    parser.add_argument('--epoch_stride', type=int, default=100)
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--ifplot', default=False)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    USE_CUDA = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if args.test_data == 'CAVE':
        bands = 31
        img_path = '/data3/zhaoshuyi/Datasets/CAVE/hsi/'
        test_dataset = CAVE_Dataset(img_path,512, 512, False, 'test')
        #print('test')
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=1,
                             pin_memory=True)

    model_path = args.model_path
    if args.model == 'mbt':
        model = ContextHyperprior(channel_in=bands,
        channel_N=args.channel_N, channel_M=args.channel_M, channel_out=bands)
    elif  args.model == 'cheng2020':
        model = Cheng2020Attention(channel_in=bands,channel_N=args.channel_N, channel_M=args.channel_M, channel_out= bands)
    elif args.model == 'transformer':
        from models.transformercompress import SymmetricalTransFormer
        model = SymmetricalTransFormer(channel_in=bands)

    results = defaultdict(list)
    for i in range(args.startepoch, args.endepoch + 1, args.epoch_stride):
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