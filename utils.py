#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 23:01:57 2020

@author: spl
"""

import numpy as np
import re

import scipy.io as spio

def loadmat(filename):
    """wrapper around scipy.io.loadmat that avoids conversion of nested matlab structs to np.arrays"""
    mat = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    for key in mat:
        if isinstance(mat[key], spio.matlab.mio5_params.mat_struct):
            mat[key] = to_dict(mat[key])
    return mat

def to_dict(matobj):
    """construct python dictionary from matobject"""
    output = {}
    for fn in matobj._fieldnames:
        val = matobj.__dict__[fn]
        if isinstance(val, spio.matlab.mio5_params.mat_struct):
            output[fn] = toDict(val)
        else:
            output[fn] = val
    return output

def strparse(strings, pattern, numeric=False, *args):
    res = [re.match(pattern, string) for string in strings]
    matching = np.nonzero(np.array([not r is None for r in res]))
    res = np.array(res)[matching]
    res = np.array([r.groups() for r in res])
    if numeric:
        print(len(args), args)
        if len(args) == 1:
            res = res.astype(args[0])
        elif len(args) == res.shape[1]:
            resOut = []
            for ci in range(len(args)):
                resOut.append(res[:, ci].astype(args[ci]))
            res = resOut
        elif len(args) != 0:
            raise Exception('number of type specifiers must equal the number of matching groups in the pattern!')
    return res