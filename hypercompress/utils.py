from PIL import Image
import torchvision
import torch
import numpy as np
import torch
import torch.nn as nn 
#from model import Model

'''
def run_on_image(model_path, image_path, device):
	"""
	Run the pretrained model stored at model_path on an image
	:param model_path: path to the model weights
	:param image_path: path to the image
	:return:
	"""
	model = Model(device)
	
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device)
	model.eval()
	
	transform = torchvision.transforms.Compose([torchvision.transforms.Resize((765, 765)), torchvision.transforms.ToTensor()])
	image = Image.open(image_path)
	inputs = transform(image)
	inputs = torch.unsqueeze(inputs, 0)
	inputs = inputs.to(device)
	
	x_hat, _, _, _, _ = model(inputs)
	reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat.squeeze)
	result_image = concat_images(image, reconstructed_image)
	result_image.show()


def concat_images(image1, image2):
	"""
	Concatenates two images together
	"""
	result_image = Image.new('RGB', (image1.width + image2.width, image1.height))
	result_image.paste(image1, (0, 0))
	result_image.paste(image2, (image1.width, 0))
	return result_image
'''

class AverageMeter(object):
	"""Stores current value of statistics and computes average"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)

def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
    module,
    buffer_name,
    state_dict_key,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')

def update_registered_buffers(
    module,
    module_name,
    buffer_names,
    state_dict,
    policy="resize_if_empty",
    dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    if not module:
        return
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",     # 修改了
            state_dict,
            policy,
            dtype,
        )



def gasuss_noise(image, mean=0, var=0.001): 
    '''    添加高斯噪声    
    :param image：原始图像    
    :param mean: 均值    
    :param var: 方差，越大，噪声越大    
    :return:noise-添加的噪声，out-加噪后的图像    
    '''
    noise =torch.from_numpy(np.random.normal(mean, var**0.5, image.shape)).type_as(image).cuda()  #创建一个均值为mean，方差为var呈高斯分布的图像矩阵

    out = image + noise  # 将噪声和原始图像进行相加得到加噪后的图像
    if out.min()<0:        
        low_clip = -1.    
    else:        
        low_clip = 0.    
        out = torch.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间 
    return noise, out

def gasuss_noise_batch(image, var): 
    '''    添加高斯噪声    
    :param image：原始图像    
    :param mean: 均值    
    :param var: 方差，越大，噪声越大    
    :return:noise-添加的噪声，out-加噪后的图像    
    '''
    noise = [0.0001, 0.001, 0.01, 0.1]
    var = noise[np.random.randint(0,4,1)[0]]
    #print(var)

    b,c,h,w = image.shape
    noises=torch.from_numpy(np.zeros(image.shape)).type_as(image).cuda()
    outs=torch.from_numpy(np.zeros(image.shape)).type_as(image).cuda()
    for i in range(b):
        img = image[i,:,:,:].cpu()

        #noise = np.random.normal(mean, var**0.5, [c,h,w])
        noise = np.random.normal(0, var**0.5, [c,h,w])
        noise =torch.from_numpy(noise) #创建一个均值为mean，方差为var呈高斯分布的图像矩阵
        out = img + noise # 将噪声和原始图像进行相加得到加噪后的图像
        if out.min()<0:        
            low_clip = -1.    
        else:        
            low_clip = 0.    
            out = torch.clip(out, low_clip, 1.0)  # clip函数将元素的大小限制在了low_clip和1之间 
        noises[i] = noise
        outs[i] = out
    #print(noises.shape, outs.shape)
    return noises, outs

def AGWN_Batch(x, SNR):
    b, h, m, n = x.shape
    snr = 10**(SNR/10.0)
    x_ = []
    for i in range(b):
        img = x[i, :, :, :].unsqueeze(0)
        xpower = torch.sum(img**2)/(m*n)
        npower = xpower/snr
        x_.append(img + torch.randn_like(img) * torch.sqrt(npower))
    return torch.cat(x_, 0)


def AGWN_np(x, SNR):
    b,m,n = x.shape
    snr = 10**(SNR/10.0)
    xpower = np.sum(x**2)/(b*m*n)
    npower = xpower/snr
    return  x + np.array(torch.randn_like(torch.from_numpy(x)))*np.sqrt(npower)

class Spa_Downs(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        #n_planes:number of channel
        #factor:the times of downsample
		
        super(Spa_Downs, self).__init__()
        
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = kernel_width
            sigma = sigma
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1./np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'
            
            
        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch       

        self.downsampler_ = downsampler

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)
                
            self.padding = nn.ReplicationPad2d(pad)
        
        self.preserve_size = preserve_size
        
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.downsampler_(x)
        
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        #print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos': 
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor  
                    dj = abs(j + 0.5 - center) / factor 
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                
                
                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                
                kernel[i - 1][j - 1] = val
            
        
    else:
        assert False, 'wrong method name'
    
    kernel /= kernel.sum()
    
    return kernel
