# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:58:54 2019

@author: gris
"""


import numpy as np
import matplotlib.pyplot as plt

import DeformationModules.ElasticOrder0
import DeformationModules.ElasticOrder1
import DeformationModules.SilentLandmark
import DeformationModules.Combination as comb_mod
import Forward.shooting as shoot
import Backward.Backward as bck
# %%
from src import rotation as rot
from src.visualisation import my_close

import Backward.ScipyOptimise as opti

#%%
Model0 = DeformationModules.ElasticOrder0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = DeformationModules.ElasticOrder0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

#%%
#param = [param_sil, param_00, param_0, param_1]
param = [param_00, param_0]
#param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5

# %%
Modlist = shoot.shooting_traj(Mod_el, N)