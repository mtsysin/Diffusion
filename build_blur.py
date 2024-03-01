import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from matplotlib import pyplot as plt


def create_sigma_schedule(dist_to_sigma = lambda x: torch.sqrt(x) // 20, size = (400, 600)):
    x, y = size
    X, Y = torch.meshgrid(torch.arange(x), torch.arange(y))
    return dist_to_sigma((X - x//2)**2 + (Y - y//2)**2)

def create_gaussian_kernel(size: int, sigmas: torch.Tensor):
    if size % 2 == 0:
        raise ValueError("Size must be odd.")
    
    x = torch.arange(size).float() - size // 2
    x = x.view(1, 1, size)
    y = x.view(1, size, 1)
    sigmas =  sigmas.view(-1, 1, 1)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigmas**2)) 
    kernel /= kernel.sum((1, 2)).view(-1, 1, 1)
    return kernel

def test_gaussian(size = 15, sigma = 4, index = 4):
    schedule = create_sigma_schedule()
    gauss = create_gaussian_kernel(size, schedule)
    plt.imsave(f"gaussian{size}_{sigma}.png", gauss[index])

def apply_uniform_blur(img: torch.Tensor, gaussian_kernel: torch.Tensor):
    """
    Apply a non-uniform blur as an example
    Should be working both on batched and unbatched inmages
    """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    b, c, m, n = img.size()
    kernel_size = gaussian_kernel.size()[-1]
    gaussian_kernel = gaussian_kernel.view(m * n, kernel_size * kernel_size)
    pad = int((kernel_size-1)/2)
    img_pad = torch.nn.functional.pad(img, pad=(pad,pad,pad,pad), mode='replicate')
    img_unfold = torch.nn.functional.unfold(img_pad, (kernel_size, kernel_size)).view(b,c,-1,m*n)
    print(img_unfold.transpose(2, 3).size())
    print(gaussian_kernel.view(-1, m*n).size())

    filter_unfold = img_unfold.transpose(2, 3).matmul(gaussian_kernel.view())
    print(img_unfold.size())
    print(filter_unfold.size())
    print(gaussian_kernel.size())
    
    
    


# apply spatially varying blur
# Unfold + matmul + fold worked flawlessly.
# https://discuss.pytorch.org/t/2d-convolution-with-different-kernel-per-location/42969  

if __name__ == "__main__":
    test_gaussian(55, 20)
    test_gaussian(105, 20)
    schedule = create_sigma_schedule()
    plt.imsave(f"schedule.png", schedule / 15.0)

    im = Image.open("test.jpg")
    np_image = np.array(im)
    np_image = np_image.transpose((2, 0, 1)) / 255.0
    img = torch.from_numpy(np_image).float()  # Ensure the tensor is of type float
    apply_uniform_blur(img, create_gaussian_kernel(7, create_sigma_schedule()))
