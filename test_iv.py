# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:28:01 2018

@author: spl
"""

import numpy as np
import pysmtb.iv
import importlib
import matplotlib
# import opensimplex
import torch

def rand_im(width, height, num_channels, num_ims, scales):
    ims = np.zeros((width, height, num_channels, num_ims))
    tmp = opensimplex.OpenSimplex()
    for x in range(width):
        for y in range(height):
            for c in range(num_channels):
                for i in range(num_ims):
                    ims[x, y, c, i] = tmp.noise4d(x / scales[0], y / scales[1], c / scales[2], i)
    return ims
    
if __name__ == "__main__":
    from pysmtb import iv
    iv = importlib.reload(iv)
    # ims = np.random.rand(10, 10, 3, 4)
    # # ims = rand_im(100, 100, 3, 4, [10, 10, 3])
    # ims_tensors = torch.rand((12, 3, 15, 15))
    # ims_list = [np.random.rand(12, 12) for _ in range(9)]
    # v = iv.iv(ims, ims_tensors, ims_list, collageBorderWidth=2)
    v = iv.iv(np.random.rand(10, 10, 3, 16).astype(np.float16))
    print()
