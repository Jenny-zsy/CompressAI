from matplotlib import cm
import numpy as np

def get_jet():
 
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)
 
    for i in range(0, 256, 1):
       colormap_float[i, 0] = cm.jet(i)[0]
       colormap_float[i, 1] = cm.jet(i)[1]
       colormap_float[i, 2] = cm.jet(i)[2]
 
       colormap_int[i, 0] = np.int_(np.round(cm.jet(i)[0] * 255.0))
       colormap_int[i, 1] = np.int_(np.round(cm.jet(i)[1] * 255.0))
       colormap_int[i, 2] = np.int_(np.round(cm.jet(i)[2] * 255.0))
 
    np.savetxt("/data1/zhaoshuyi/AIcompress/baseline/images/jet_float.txt", colormap_float, fmt = "%f", delimiter = ' ', newline = '\n')
    np.savetxt("/data1/zhaoshuyi/AIcompress/baseline/images/jet_int.txt", colormap_int, fmt = "%d", delimiter = ' ', newline = '\n')
 
    print (colormap_int)
 
    return

def get_spectral():
 
    colormap_int = np.zeros((256, 3), np.uint8)
    colormap_float = np.zeros((256, 3), np.float)
 
    for i in range(0, 256, 1):
       colormap_float[i, 0] = cm.spectral(i)[0]
       colormap_float[i, 1] = cm.spectral(i)[1]
       colormap_float[i, 2] = cm.spectral(i)[2]
 
       colormap_int[i, 0] = np.int_(np.round(cm.spectral(i)[0] * 50.0))
       colormap_int[i, 1] = np.int_(np.round(cm.spectral(i)[1] * 50.0))
       colormap_int[i, 2] = np.int_(np.round(cm.spectral(i)[2] * 50.0))
 
    np.savetxt("/data1/zhaoshuyi/AIcompress/baseline/images/spectral_float.txt", colormap_float, fmt = "%f", delimiter = ' ', newline = '\n')
    np.savetxt("/data1/zhaoshuyi/AIcompress/baseline/images/spectral_int.txt", colormap_int, fmt = "%d", delimiter = ' ', newline = '\n')
 
    print (colormap_int)
 
    return
if __name__ == "__main__":
    get_jet()
    #get_spectral()