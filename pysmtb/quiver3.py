#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:04:07 2020

@author: merzbach
"""

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import numpy as np

def quiver3(XYZ, UVW, fig=None, limits=None, *args, **kwargs):
    if fig is None:
        fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]
    U = UVW[0, :]
    V = UVW[1, :]
    W = UVW[2, :]
    
    if not limits is None and len(limits):
        xlim = limits[:2]
        ylim = limits[2:4]
        zlim = limits[4:6]
        X = X.copy()
        Y = Y.copy()
        Z = Z.copy()
        U = U.copy()
        V = V.copy()
        W = W.copy()
        mask = (X < xlim[0]) | (X > xlim[1]) | (Y < ylim[0]) | (Y > ylim[1]) | (Z < zlim[0]) | (Z > zlim[1])
        U[mask] = np.nan
        V[mask] = np.nan
        W[mask] = np.nan
        X[mask] = np.nan
        Y[mask] = np.nan
        Z[mask] = np.nan
    
    scat = ax.quiver(X, Y, Z, U, V, W, *args, **kwargs)
    
    if not limits is None and len(limits):
        max_range = np.max([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        ax.set_xlim(xlim[0], xlim[0] + max_range)
        ax.set_ylim(ylim[0], ylim[0] + max_range)
        ax.set_zlim(zlim[0], zlim[0] + max_range)
    elif limits is None:
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_position(Bbox([[0, 0], [1, 1]]))
    
    plt.show()
    return scat
    