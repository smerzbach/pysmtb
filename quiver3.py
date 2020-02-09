#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:04:07 2020

@author: merzbach
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def quiver3(XYZ, UVW, fig=None, *args, **kwargs):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]
    U = UVW[0, :]
    V = UVW[1, :]
    W = UVW[2, :]
    
    scat = ax.quiver(X, Y, Z, U, V, W, *args, **kwargs)
    
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()
    return scat
    