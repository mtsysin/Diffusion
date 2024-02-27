import torch
import numpy as np
from typing import List, Dict


def apply_uniform_blur(img: torch.Tensor, size_increase)



# apply spatially varying blur
# Unfold + matmul + fold worked flawlessly.
# https://discuss.pytorch.org/t/2d-convolution-with-different-kernel-per-location/42969  