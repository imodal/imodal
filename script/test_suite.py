#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:10:34 2018

@author: barbaragris
"""
import matplotlib.pyplot as plt
import numpy as np

plt.show()
plt.ion()

# %%
from src import field_structures as fields

# %% import data


x = np.array([[0., 0.]])
R = np.array([[0., -1.], [-1., 0.]])
px = np.array([[0., 0.]])
pR = np.zeros([x.shape[0], 2, 2])

param = ((x, R), (px, pR))
# %%
Cot = {'0': [], 'x,R': [param]}
fields.my_CotToVs(Cot, 1)
