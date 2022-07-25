import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def gray2color(gray_array, color_map):
    
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
 
    for i in range(0, rows):
        for j in range(0, cols):
            #print(gray_array[i][j])
            color_array[i][j] = color_map[gray_array[i][j]]
    
    #color_image = Image.fromarray(color_array)
 
    return color_array


def concat_images(image1, image2):
	"""
	Concatenates two images together
	"""
	result_image = Image.new('RGB', (image1.width + image2.width, image1.height))
	result_image.paste(image1, (0, 0))
	result_image.paste(image2, (image1.width, 0))
	return result_image

def plot_cmap_jet():
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig = plt(gradient, aspect='auto', cmap=plt.get_cmap('jet'))

def imsave(recon, origin, save_path, i):
    jet_map = np.loadtxt('./images/jet_int.txt', dtype=np.int)
    
    reconstructed_image = transforms.ToPILImage(mode='RGB')(recon.squeeze())
    reconstructed_image.save(
        os.path.join(save_path, "{}recon.png".format(i + 1)))

    residual = (origin - recon).abs().squeeze().detach().cpu().numpy()
    residual /= residual.max()
    residual_image = torch.zeros(residual.shape[1], residual.shape[2])
    residual_image = (np.sum(residual, axis=0) * 255).clip(0, 255).astype(int)
    #residual_image = ((residual_image-residual_image.min())*255/(residual_image.max()-residual_image.min())).clip(0,255).astype(int)
    color_jet = gray2color(residual_image, jet_map)
    color_jet = Image.fromarray(np.uint8(color_jet))
    #residual_images = concat_images(Image.fromarray(np.uint8(residual_image)), color_jet)
    color_jet.save(os.path.join(save_path, "{}residual.png".format(i + 1)))