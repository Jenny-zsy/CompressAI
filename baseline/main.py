from ast import arg
from cgi import test
from curses import savetty
import imp
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import torchvision

from torch import optim
from torchvision import transforms
from torch.autograd import Variable
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
from models.model import ContextHyperprior
from ratedistortionloss import RateDistortionLoss
from utils import concat_images, AverageMeter
from dataset import ImageFolder


def train_epoch(model, criterion, optimiser, train_dataloader, epoch, epochs, f):
	"""
	Train model for one epoch
	"""
	loss = AverageMeter()
	mse_loss = AverageMeter()
	bpp_loss = AverageMeter()
	model.train()  # Set model to training mode
	
	for batch, (inputs, _) in enumerate(train_dataloader):
		inputs = inputs.cuda()

		optimiser.zero_grad()
		
		# forward
		out = model(inputs)

		out_criterion = criterion(out, inputs)
		
		# backward
		out_criterion["loss"].backward()
		optimiser.step()
		
		# keep track of loss
		loss.update(out_criterion["loss"].item(), inputs.size(0))
		mse_loss.update(out_criterion["mse_loss"].item(), inputs.size(0))
		bpp_loss.update(out_criterion["bpp_loss"].item(), inputs.size(0))
		
		# print out loss and visualise results
		if batch % 10 == 0:
			print('Epoch {}/{}:[{}]/[{}]'.format(epoch, epochs, batch, len(train_dataloader)).ljust(25), 'Loss: %.4f'.ljust(7)%(loss.avg), 'mse_loss: %.4f'.ljust(7)%(mse_loss.avg), 'bpp_loss: %.4f'.ljust(7)%(bpp_loss.avg))
			f.write('Epoch {}/{}:[{}]/[{}]'.format(epoch, epochs, batch, len(train_dataloader)).ljust(25), 'Loss: %.4f'.ljust(7)%(loss.avg), 'mse_loss: %.4f'.ljust(7)%(mse_loss.avg), 'bpp_loss: %.4f\n'.ljust(7)%(bpp_loss.avg))
			'''reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat[0].to('cpu'))
			image = torchvision.transforms.ToPILImage(mode='RGB')(inputs[0].to('cpu'))
			result_image = concat_images(image, reconstructed_image)
			result_image.save("train_images/epoch{}batch{}.png".format(epoch, batch))'''

	return losses.avg

def test_epoch(model, criterion, test_dataloader, epoch, f):
    model.eval()

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for inputs in test_dataloader:
            inputs = inputs.cuda()
            out = model(inputs)
            out_criterion = criterion(out, inputs)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print('Epoch {}'.format(epoch).ljust(7), 'Loss: %.4f'.ljust(7)%(loss.avg), 'mse_loss: %.4f'.ljust(7)%(mse_loss.avg), 'bpp_loss: %.4f'.ljust(7)%(bpp_loss.avg))
    f.write('Epoch {}'.format(epoch).ljust(7), 'Loss: %.4f'.ljust(7)%(loss.avg), 'mse_loss: %.4f'.ljust(7)%(mse_loss.avg), 'bpp_loss: %.4f\r'.ljust(7)%(bpp_loss.avg))
    return loss.avg

def plot(y1, y2, label, outf):
    x = np.arange(0, len(y1), 1)
    plt.plot(x, y1, label='train')
    plt.plot(x, y2, label='valid')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(label)

    plt.savefig(outf+label+'.jpg')

def train(args):
	gpu_num = len(args.gpus.split(','))
	device_ids = list(range(gpu_num))

	save_path = './results/lr{args.lr}_bs{args.batch_size}_lambda{args.lmbda}/'
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
	test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

	# load dataset
	train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
	test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
	
	# create data loader
	train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
	test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
	
	model = ContextHyperprior()
    
	criterion = RateDistortionLoss(args.lmbda)
	criterion.cuda()
	optimiser = optim.Adam(model.parameters(), lr=args.lr)
	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min")

	# load model and continue training
	if args.continue_training:
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['state_dict'])
		optimiser.load_state_dict(checkpoint['optimiser'])
		lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
		start_epoch = checkpoint['epoch']
		f = open(save_path + 'loss.txt', 'a+')
	else:
		start_epoch = 0
		f = open(save_path + 'loss.txt', 'w+')
	
	# move model to gpu and show structure
	if gpu_num > 1:
		model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
	model.cuda()
	summary(model, input_size=train_data[0][0].shape)

	# training
	train_loss_sum = []
	test_loss_sum = []
	for epoch in range(start_epoch, args.epochs):
		train_loss = train_epoch(model, criterion, optimiser, train_dataloader, epoch, args.epochs, f)
		test_loss = test_epoch(model, criterion, test_dataloader, epoch, f)

		train_loss_sum.append(train_loss)
		test_loss_sum.append(test_loss)

		# save the model
		state = {
					'epoch': epoch,
					'state_dict': model.state_dict(),
					'optimizer': optimiser.state_dict(),
					"lr_scheduler": lr_scheduler.state_dict(),
				}
		torch.save(state, args.checkpoint)

	plot(train_loss_sum, test_loss_sum, 'loss', save_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-epochs', type=int, default=1000, help='number of epoch for training')
	parser.add_argument('-batch_size', type=int, default=4, help='number of epoch for training')
	parser.add_argument('-continue_training', type=bool, default=False, help='whether to use pretrained model from the checkpoint file')
	parser.add_argument('-checkpoint', type=str, default='compression_model.pth', help='path where to save checkpoint during training')
	parser.add_argument('-root', type=str, default='data/', help='path to the folder with images')
	parser.add_argument('-lr', type=float, default=1e-4, help='path to the folder with grayscale images')
	parser.add_argument("--gpus", type=str, default="0", help='path log files')
	parser.add_argument("--lmbda", type=float, default=1e-2, help="Bit-rate distortion parameter (default: %(default)s)",)
	args = parser.parse_args()

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

	train(args)
	print("Done.")
