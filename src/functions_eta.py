#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:08:22 2018

@author: barbaragris
"""

import numpy as np

def my_etaK(K): # Now on the left
    return np.tensordot(my_eta().transpose(), K, axes=2)
def my_etaKeta(K): # Now on both sides (sym case)
    return my_etaK(my_Keta(K))
def my_Keta(K): # transformation on the right (sym case)
    return np.tensordot(K, my_eta(), axes=2)
def my_eta(): #symmetric case
    eta = np.zeros((2,2,3))
    c = 1/np.sqrt(2)
    eta[0,0,0], eta[0,1,1], eta[1,0,1], eta[1,1,2]  = 1., c, c, 1.
    return eta
