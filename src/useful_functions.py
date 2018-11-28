#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:10:59 2018

@author: barbaragris
"""
import numpy as np
from implicitmodules.src import kernels as ker


def my_new_SKS(Mod):
    """ my_SKS(Mod) compute induce metric on the bundle of
    symmetric   matrice and vectors depending on the order of the
    constraint.
    """
    if 'x,R' in Mod:
        sig = Mod['sig']
        (x, R) = Mod['x,R']
        SKS = ker.my_K(x, x, sig, 1)
        if 'nu' in Mod:
            SKS = SKS + Mod['nu'] * np.eye(SKS.shape[0])
    
    if '0' in Mod:
        sig = Mod['sig']
        x = Mod['0']
        SKS = ker.my_K(x, x, sig, 0)
    return SKS
        





