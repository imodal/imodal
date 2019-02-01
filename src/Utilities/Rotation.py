#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:09:31 2018

@author: barbaragris
"""
import numpy as np

def my_R(th) :
    """ return the 2D matrix of the rotation of angle theta
    """
    R = np.zeros((2,2))
    R[0,0], R[0,1] = np.cos(th), -np.sin(th)
    R[1,0], R[1,1] = np.sin(th), np.cos(th)
    return R
