#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:10:34 2018

@author: barbaragris
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
plt.show()
plt.ion()

#%%
from src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot, \
    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
#%% import data



x = np.array([[0.,0.]])
R = np.array([[0.,-1.],[-1., 0.]])
px = np.array([[0.,0.]])
pR = np.zeros([x.shape[0], 2, 2])

param = ((x,R), (px, pR))
#%%
Cot = {'0':[], 'x,R':[param]}
fields.my_CotToVs(Cot,1)
