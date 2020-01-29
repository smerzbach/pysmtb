#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 23:01:57 2020

@author: spl
"""

import numpy as np
import re

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