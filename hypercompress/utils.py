from PIL import Image
import torchvision
import torch
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

def AGWN_Batch(x, SNR):
    b, h, m, n = x.shape
    snr = 10**(SNR/10.0)
    x_ = []
    for i in range(b):
        img = x[i, :, :, :].unsqueeze(0)
        xpower = torch.sum(img**2)/(h*m*n)
        npower = xpower/snr
        x_.append(img + torch.randn_like(img) * torch.sqrt(npower))
    return torch.cat(x_, 0)