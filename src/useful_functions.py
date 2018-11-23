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
        (x,R) = Mod['x,R']
        N = x.shape[0]
        SKS = np.zeros((3*N,3*N))
        for i in range(N):
            for j in range(i+1,N):
                SKS[3*i:3*(i+1),3*j:3*(j+1)] = ker.my_K(x[i],x[j],sig,1,1)
        SKS += SKS.transpose()
        for i in range(N):
            SKS[3*i:3*(i+1),3*i:3*(i+1)] = ker.my_K(x[i],x[i],sig,1,1)
        if 'nu' in Mod:
            SKS = SKS + Mod['nu']*np.eye(SKS.shape[0])
            
    if '0' in Mod:
        sig = Mod['sig']
        x = Mod['0']
        N = x.shape[0]
        SKS = np.zeros((2*N,2*N))
        for i in range(N):
            for j in range(i+1,N):
                SKS[2*i:2*(i+1),2*j:2*(j+1)] = ker.my_K(x[i],x[j],sig,0,0)
        SKS += SKS.transpose()
        for i in range(N):
            SKS[2*i:2*(i+1),2*i:2*(i+1)] = ker.my_K(x[i],x[i],sig,0,0)
    return SKS
        





