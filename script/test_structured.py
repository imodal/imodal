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

import implicitmodules.src.data_attachment.varifold as var
import implicitmodules.Backward.Backward as bckwd

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
#Mod_el = Mod_el_init.copy_full()
###%%
##
##Mod_el.update()
##
###%%
##Mod_el.GeodesicControls_curr(Mod_el.GD)
###%%
##Mod_el.Cont[1] - Mod0['mom']
#
###%%
##v = Mod_el.field_generator_curr()
###%%
##dxH = HamDer.dxH(Mod_el)
##dpH = HamDer.dpH(Mod_el)
##
###%%
##Mod_el.add_cot(dxH)
##%%
#my_eps = 0.001
##%%
#sig_var = 1
#(varcost, dxvarcost) = var.my_dxvar_cost(xs, xs, sig_var)
#%%
dxvarcost = np.ones(xs.shape)
my_eps = 0.01
#%%
grad = {'0': [(np.zeros(x0.shape), np.zeros(x0.shape)),
              (dxvarcost, np.zeros(xs.shape))],
        'x,R': [((np.zeros(x1.shape), np.zeros(R.shape)),
                 (np.zeros(x1.shape), np.zeros(R.shape)))]}

#ngrad = shoot_old.my_sub_bckwd(Mod0, Mod1, Cot, grad, my_eps)
#%%
Mod_el = Mod_el_init.copy_full()
GD_grad_1 = Mod_el.GD.copy()
param0_0 = (np.zeros(x0.shape), np.zeros(x0.shape))
paramxR_0 = ((np.zeros(x1.shape), np.zeros(R.shape)), (np.zeros(x1.shape), np.zeros(R.shape)))
params_0 = (dxvarcost, np.zeros(xs.shape))

GD_grad_1.fill_cot_from_param([params_0, param0_0, paramxR_0])
#%%

#GDgrad = bckwd.backward_step(Mod_el, my_eps, GD_grad_1)

#%%
#
#sum(np.abs(GDgrad.Cot['0'][1][0] - ngrad['0'][0][0]))
#
#
#sum(np.abs(GDgrad.Cot['x,R'][0][1][0] - ngrad['x,R'][0][1][0]))



#
#
#
#
#%%
N = 5
Modlist = shoot.shooting_traj(Mod_el, N)
#%%
#%%

cgrad = bckwd.backward_shoot_rk2(Modlist, GD_grad_1, my_eps)

#%%
cgrad_old = shoot_old.my_bck_shoot(Traj, grad, my_eps)
#%%
print(sum(np.abs(cgrad.Cot['0'][1][1] - cgrad_old['0'][0][1])))


print(sum(np.abs(cgrad.Cot['x,R'][0][0][0] - cgrad_old['x,R'][0][0][0])))




#%%
t_old= 10
t = 10

(tMod0, tMod1, tCot) = Traj[t_old]
print(sum(np.abs(tMod0['0'] - Modlist[t].GD.Cot['0'][1][0])))
print(sum(np.abs(tMod1['x,R'][0] - Modlist[t].GD.Cot['x,R'][0][0][0])))
print(sum(np.abs(tMod1['x,R'][1] - Modlist[t].GD.Cot['x,R'][0][0][1])))
print(sum(np.abs(tMod0['0'] - Modlist[t].GD.Cot['0'][1][0])))
print(sum(np.abs(tCot['0'][1][0] - Modlist[t].GD.Cot['0'][0][0])))




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





















