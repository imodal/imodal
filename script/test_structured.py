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

Mod_el_init = comb_mod.CompoundModules([Sil, Model0, Model1])

#%%

param_sil = (xs, ps)
param_0 = (x0, p0)
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_0, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)

##%%
#
#Mod_el.update()
#
##%%
#Mod_el.GeodesicControls_curr(Mod_el.GD)
##%%
#Mod_el.Cont[1] - Mod0['mom']

##%%
#v = Mod_el.field_generator_curr()
##%%
#dxH = HamDer.dxH(Mod_el)
#dpH = HamDer.dpH(Mod_el)
#
##%%
#Mod_el.add_cot(dxH)

#%%
Mod_el = Mod_el_init.copy_full()
N = 5
traj, contlist = shoot.shooting_traj(Mod_el, N)

#%%
t_old= -1
t = -1

(tMod0, tMod1, tCot) = Traj[t_old]
print(sum(np.abs(tMod0['0'] - traj[t].Cot['0'][1][0])))
print(sum(np.abs(tMod1['x,R'][0] - traj[t].Cot['x,R'][0][0][0])))
print(sum(np.abs(tMod1['x,R'][1] - traj[t].Cot['x,R'][0][0][1])))
print(sum(np.abs(tMod0['0'] - traj[t].Cot['0'][1][0])))
print(sum(np.abs(tCot['0'][1][0] - traj[t].Cot['0'][0][0])))




#%%

Cot['0'][0][1] -Mod_el.GD.Cot['0'][1][1]

Cot['x,R'][0][0][1] -Mod_el.GD.Cot['x,R'][0][0][1]

#%%
Mod_el.update()
Mod_el.GeodesicControls_curr(Mod_el.GD)
#dGD = HamDer.dpH(Mod_el) #tested
dGD = HamDer.dxH(Mod_el) # tested
#%%


der['0'][1][1] +dGD.Cot['0'][0][1]

der['x,R'][0][1][1] + dGD.Cot['x,R'][0][1][1]
#%%





















