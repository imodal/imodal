#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:49:33 2018

@author: barbaragris
"""
import numpy as np
import GeometricalDescriptors.GeometricalDescriptors as geo_descr
import implicitmodules.field_structures as fields
import StructerdFIelds.StructuredFields as stru_fie
import DeformationModules.DeformationModules as defmod
import DeformationModules.Combination as comb_mod
import Forward.Hamiltonianderivatives as HamDer
import Forward.shooting as shoot
#%%
from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot_old, \
    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
#%%
dim = 2
Sil = defmod.SilentLandmark(xs.shape[0], dim)
Model1 = defmod.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0])

#%% 

Mod_el = comb_mod.CompoundModules([Sil, Model0, Model1])
#%%

param_sil = (xs, ps)
param_0 = (x0, p0)
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_0, param_1]
GD = Mod_el.GD.copy()
Mod_el.GD.fill_cot_from_param(param)

#%%

Mod_el.update()

#%%
Mod_el.GeodesicControls_curr(Mod_el.GD)
#%%
Mod_el.Cont[2] - Mod1['h']

#%%
v = Mod_el.field_generator_curr()
#%%
dxH = HamDer.dxH(Mod_el)
dpH = HamDer.dpH(Mod_el)

#%%
Mod_el.add_cot(dxH)

#%%

N = 3
traj, contlist = shoot.shooting_traj(Mod_el, N)