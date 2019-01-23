# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:58:34 2019

@author: gris
"""


import scipy .optimize
import numpy as np
import matplotlib.pyplot as plt


import src.DeformationModules.SilentLandmark as defmodsil
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.Combination as comb_mod

import src.Forward.shooting as shoot
import src.Backward.Backward as bckwrd
#%%
#from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot_old, \
#    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
from implicitmodules.src.visualisation import my_close
from implicitmodules.src import rotation as rot
import implicitmodules.src.data_attachment.varifold as var

import implicitmodules.src.Optimisation.ScipyOpti as opti

import pickle
#%%  source

height = 38.
heightt = 209.


with open('../data/basi2temp.pkl', 'rb') as f:
    img, lx = pickle.load(f)
    
#nlx = np.asarray(lx).astype(np.float32)
#(lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
#scale = 38./(lmax-lmin)
#
#nlx[:,1]  =  - scale*(nlx[:,1]-lmin) 
#nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))           

Dx = 0
Dy = 0

nlx = np.asarray(lx).astype(np.float64)
nlx[:,1] = - nlx[:,1]
(lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))

scale = height/(lmax-lmin)
nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0])) + Dx
nlx[:,1]  = scale*(nlx[:,1]-lmin) + Dy


x0 = nlx[nlx[:,2]==0,0:2]
x1 = nlx[nlx[:,2]==1,0:2]
xs = nlx[nlx[:,2]==2,0:2]

#%%  target
with open('../data/basi2target.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

#nlxt = np.asarray(lxt).astype(np.float32)
#(lmin, lmax) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
#scale = 209./(lmax-lmin)
#
#nlxt[:,1]  = 90.0 - scale*(nlxt[:,1]-lmin) #+ 400
#nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))            

nlxt = np.asarray(lxt).astype(np.float64)
nlxt[:,1] = -nlxt[:,1]
(lmint, lmaxt) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
scale = heightt/(lmaxt-lmint)


nlxt[:,1]  = scale*(nlxt[:,1] - lmint)
nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))


xst = nlxt[nlxt[:,2]==2,0:2]


#%% parameter for module of order 1
th = 0*np.pi
th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0],2,1))

K = 10

height = 38.
L = height

a = 1/L

b = 3.
Dy = 0.

z = a*(x1[:,1]-Dy)

C[:,1,0] = K*((1-b)*z**2+b*z)

C[:,0,0] = 0.9*C[:,1,0]
#%%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 30
sig00 = 500
sig1 = 50
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
#%% 

Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])


#%%
p00 = np.zeros([1, 2])

p0 = np.zeros(x0.shape)
ps = np.zeros(xs.shape)
(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))

param_sil = (xs, ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), p00)
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_00, param_0, param_1]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)


#%%
Mod_el = Mod_el_init.copy_full()

N=5

#%%
Mod_el_opti = Mod_el_init.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
# %%
lam_var = 15.
sig_var = 10.
N = 5
args = (Mod_el_opti, xst, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
                              options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
                                       'eps': 1e-08, 'maxfun': 100, 'maxiter': 5, 'iprint': -1, 'maxls': 20})
# %%µ






















