# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 22:28:01 2018

@author: spl
"""

global iv
import numpy as np
import iv
import importlib
import matplotlib
import opensimplex

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
    iv = importlib.reload(iv)
    #ims = np.random.rand(10, 10, 3, 4)
    ims = rand_im(100, 100, 3, 4, [10, 10, 3])
    iv.iv(ims, borderWidth = 2)