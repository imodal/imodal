# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:58:54 2019

@author: gris
"""


import numpy as np
import matplotlib.pyplot as plt

import implicitmodules.DeformationModules.ElasticOrder0
import implicitmodules.DeformationModules.ElasticOrder1
import implicitmodules.DeformationModules.SilentLandmark
import implicitmodules.DeformationModules.Combination as comb_mod_old
import Forward.shooting as shoot
import Backward.Backward as bck
import implicitmodules.Forward.Hamiltonianderivatives as HamDer_old
# %%
from src import rotation as rot
from src.visualisation import my_close

import Backward.ScipyOptimise as opti

#%%
Model0 = implicitmodules.DeformationModules.ElasticOrder0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = implicitmodules.DeformationModules.ElasticOrder0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
Model1 = implicitmodules.DeformationModules.ElasticOrder1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Mod_el_init = comb_mod_old.CompoundModules([Model00, Model0, Model1])
#Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

#%%
#param = [param_sil, param_00, param_0, param_1]
#param = [param_00, param_0, param_1]
#param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5

# %%
Modlist = shoot.shooting_traj(Mod_el, N)