#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 10:58:13 2019

@author: barbaragris
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
import implicitmodules.src.Optimisation.ScipyOpti_attach as opti_att

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
#%%


path_data = '../data/'
with open(path_data + 'basi1b.pkl', 'rb') as f:
    img, lx = pickle.load(f)

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = 38. / (lmax - lmin)

nlx[:, 1] = 38.0 - scale * (nlx[:, 1] - lmin)
nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

x0 = nlx[nlx[:, 2] == 0, 0:2]
x1 = nlx[nlx[:, 2] == 1, 0:2]
xs = nlx[nlx[:, 2] == 2, 0:2]

xs = np.delete(xs, 3, axis=0)
# %% target
with open(path_data + 'basi1t.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = 100. / (lmax - lmin)

nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]


#%% parameter for module of order 1
th = 0*np.pi
th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0],2,1))

K = 10
height = 38.
C = np.zeros((x1.shape[0],2,1))
K = 10

L = height
a, b = -2/L**3, 3/L**2
C[:,1,0] = K*(a*(L-x1[:,1]+Dy)**3 + b*(L-x1[:,1]+Dy)**2)
C[:,0,0] = 1.*C[:,1,0]

#%%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 20
sig00 = 200
sig1 = 30
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
#%% 

Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])

Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

#Mod_el_init = comb_mod.CompoundModules([ Sil])


#%%
p00 = np.zeros([1, 2])
p00 = 0.001*np.random.rand(*p00.shape)

p0 = np.zeros(x0.shape)
p0 = 0.00001*np.random.rand(*p0.shape)
ps = np.zeros(xs.shape)
ps = 0.000001*np.random.rand(*ps.shape)
(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
(p1,PR) = (0.000001*np.random.rand(*x1.shape), 0.000001*np.random.rand(x1.shape[0],2,2))
#%%
param_sil = (xs, ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), p00)
param_1 = ((x1, R), (p1, PR))

#%%
#param = [param_sil, param_00, param_0, param_1]
param = [param_sil, param_1]
#param = [param_sil]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)

#%%
Mod_el = Mod_el_init.copy_full()


P0 = opti.fill_Vector_from_GD(Mod_el.GD)
lam_var = 10.
sig_var = 30.
N = 5
#def attach_fun(x,y):
#    return var.my_dxvar_cost(x, y, sig_var)

def attach_fun(x,y):
    return np.sum( (x-y)**2), 2*(x-y)

y = 1.*xs
args = (Mod_el, y, attach_fun, N, 0.001)
#args = (Mod_el, xst, attach_fun, N, 0.001)
#%%
grad = opti_att.jac(P0, *args)
co_init = opti_att.fun(P0, *args)
#%% Compute manual for x1
grad_man_x1= np.zeros(xs.shape)
eps = 1e-6
for i in range(5):
    for j in range(ps.shape[1]):
        Mod_el_cop = Mod_el.copy()
        args_cop = (Mod_el_cop, y, attach_fun, N, 0.001)
        ps_cop = ps.copy()
        ps_cop[i,j] += eps
        #param1_cop = ((x1, R), (p1, PR))
        params_cop = (xs, ps_cop)
#        param_cop = [param_sil, param_00, param_0, param1_cop]
#        param_cop = [param_sil, param1_cop]
        param_cop = [params_cop, param_1]
        Mod_el_cop.GD.fill_cot_from_param(param_cop)
        P0_cop = opti.fill_Vector_from_GD(Mod_el_cop.GD)
        co_init_cop = opti_att.fun(P0_cop, *args_cop)
        grad_man_x1[i,j] = (co_init_cop - co_init)/eps
#%% Compute manual for R
grad_man_x1= np.zeros(R.shape[0])
dir_theta = np.random.rand(R.shape[0])
dir_R_n = np.asarray([ rot.my_R(eps *thetai) for thetai in dir_theta])

eps = 1e-5
for i in range(5):
    #for j in range(xs.shape[1]):
        Mod_el_cop = Mod_el.copy()
        args_cop = (Mod_el_cop, y, attach_fun, N, 0.001)
        R_cop = R.copy()
        R_cop[i,:,:] = dir_R_n[i]
        param1_cop = ((x1, R_cop), (p1, PR))
#        params_cop = (xs_cop, ps)
#        param_cop = [param_sil, param_00, param_0, param1_cop]
        param_cop = [param_sil, param1_cop]
        Mod_el_cop.GD.fill_cot_from_param(param_cop)
        P0_cop = opti.fill_Vector_from_GD(Mod_el_cop.GD)
        co_init_cop = opti_att.fun(P0_cop, *args_cop)
        grad_man_x1[i] = (co_init_cop - co_init)/eps

#%% Compute manual for R
grad_man_x1= np.zeros(R.shape)
dir_theta = np.random.rand(R.shape[0])
dir_R_n = np.asarray([ rot.my_R(eps *thetai) for thetai in dir_theta])

eps = 1e-2
for i in range(5):
    for j in range(R.shape[1]):
      for k in range(R.shape[2]):
        Mod_el_cop = Mod_el.copy()
        args_cop = (Mod_el_cop, y, attach_fun, N, 0.001)
        R_cop = R.copy()
        R_cop[i,j,k]+= eps
        #R_cop[i,:,:] = dir_R_n[i]
        param1_cop = ((x1, R_cop), (p1, PR))
#        params_cop = (xs_cop, ps)
#        param_cop = [param_sil, param_00, param_0, param1_cop]
        param_cop = [param_sil, param1_cop]
        Mod_el_cop.GD.fill_cot_from_param(param_cop)
        P0_cop = opti.fill_Vector_from_GD(Mod_el_cop.GD)
        co_init_cop = opti_att.fun(P0_cop, *args_cop)
        grad_man_x1[i,j,k] = (co_init_cop - co_init)/eps

#%%
#dim0 = 2*(np.prod(xs.shape)) + np.prod(x1.shape)
dim0 = 1*(np.prod(xs.shape)) + np.prod(x1.shape) + np.prod(R.shape)
print(np.reshape(grad_man_x1[:5,:], -1) )
print(grad[dim0:dim0+ 10])
print(np.reshape(grad_man_x1[:5,:], -1) - grad[dim0:dim0+ 10])  

#%%
nR = R.shape[0]
dim0 = 1*(np.prod(xs.shape)) + np.prod(x1.shape)# + np.prod(R.shape)
grad_vec = grad[dim0:dim0+nR*4]
grad_vec = np.reshape(grad_vec, [nR,2,2])
grad_dir = np.array([np.trace(grad_vec[i] @ (dir_R_n[i] - R [i])/eps) for i in range(5)])

#%%
print(grad_dir[:5])
print(grad_man_x1[:5])
#%%
#dim0 = 2*(np.prod(xs.shape)) + np.prod(x1.shape)
dim0 = 1*(np.prod(xs.shape)) + np.prod(x1.shape) + np.prod(R.shape)
print(np.reshape(grad_man_x1[:5,:], -1) )
print(grad[dim0:dim0+ 10])
print(np.reshape(grad_man_x1[:5,:], -1) - grad[dim0:dim0+ 10])  

#%%
import src.Forward.Hamiltonianderivatives as HamDer
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el_init.update()
Mod_el_init.GeodesicControls_curr(Mod_el_init.GD)
dx = HamDer.dxH(Mod_el_init)
dp = HamDer.dpH(Mod_el_init)
#%%
dp_vect = np.reshape(dp.GD_list[1].tan[0][:5,:], -1)
print(np.reshape(grad_man_x1[:5,:], -1) - grad[dim0:dim0+ 10] - dp_vect)  

#%%
print(dp.GD_list[1].tan[0][:5,:]-grad_man_x1[:5,:])
#print(dp.GD_list[0].tan)

#%%



N=5
Modlist = shoot.shooting_traj(Mod_el, N)

#%%
Mod_el_opti = Mod_el_init.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
# %%
lam_var = 10.
sig_var = 30.
N = 5
args = (Mod_el_opti, xst, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
                              options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
                                       'eps': 1e-08, 'maxfun': 100, 'maxiter': 5, 'iprint': -1, 'maxls': 20})
#%%
P1 = res['x']

# %%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)                                       
#%%
Modlist_opti_tot = shoot.shooting_traj(Mod_el_opti, N)

#%% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')

#%% Shooting from controls

Contlist = []
for i in range(len(Modlist_opti_tot)):
    Contlist.append(Modlist_opti_tot[i].Cont)

#%%
Mod_cont_init = Modlist_opti_tot[0].copy_full()
Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, 5)

#%% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_cont[2 * i].GD.GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')









