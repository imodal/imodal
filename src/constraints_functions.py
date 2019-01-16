#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:12:42 2018

@author: barbaragris
"""

import numpy as np
from scipy.linalg import solve
from src import functions_eta as fun_eta


def my_Amh(Mod1,h):
    """ Compute the target value for the strain tensor
    """
    (x,R) = Mod1['x,R']
    C = Mod1['C']
    N = x.shape[0]
    eta = fun_eta.my_eta()
    out = np.asarray([np.tensordot(np.dot(R[i],
        np.dot(np.diag(np.dot(C[i],h)),
        R[i].transpose())),eta,axes = 2) for i in range(N)])
    return out


def my_new_AmKiAm(Mod1):
    SKS = Mod1['SKS']
    (x, R) = Mod1['x,R']
    N = x.shape[0]
    C = Mod1['C']
    dimh = C.shape[2]
    lam = np.zeros((dimh, 3 * N))
    Am = np.zeros((3 * N, dimh))
    
    for i in range(dimh):
        h = np.zeros((dimh))
        h[i] = 1.
        Am[:, i] = my_Amh(Mod1, h).flatten()
        lam[i, :] = solve(SKS, Am[:, i], sym_pos=True)
    return (Am, np.dot(lam, Am))