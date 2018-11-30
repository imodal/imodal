#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:19:10 2018

@author: barbaragris
"""
import sys, os.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..')*2)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
plt.show()
plt.ion()

#%%
from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot, \
    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
#%% import data

import pickle
with open('../data/basi1.pkl', 'rb') as f:
    img, lx = pickle.load(f)
    
    
nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
scale = 38./(lmax-lmin)

nlx[:,1]  = 38.0 - scale*(nlx[:,1]-lmin)
nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))            

(name, dt, coeffs, nu) = ('../results/basi_expf_', 0.1, [5., 0.05], 0.001)

x0 = nlx[nlx[:,2]==0,0:2] # points of order 0
x1 = nlx[nlx[:,2]==1,0:2] # points of order 1
xs = nlx[nlx[:,2]==2,0:2] # silent points

nlxt = np.asarray(lx).astype(np.float32)
(lmint, lmaxt) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
scale = 100./(lmaxt-lmint)

nlxt[:,1]  = 100.0 - scale*(nlxt[:,1]-lmint)-30.
nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))            


# (name, dt, coeffs, nu) = ('basi_expe_', 0.1, [8., 0.05], 0.001)
(name, dt, coeffs, nu) = ('basi_expf_', 0.1, [5., 0.05], 0.001)
    
x0 = nlx[nlx[:,2]==0,0:2]
x1 = nlx[nlx[:,2]==1,0:2]
xs = nlx[nlx[:,2]==2,0:2]
xst = nlxt[nlxt[:,2]==2,0:2]

#%% Define Modules
nx, ny = (5,11)
(a,b,c,d) = (-10., 10., -3., 40.)
[xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
(nx,ny) = xx.shape

nxs = np.asarray([xx.flatten(), xy.flatten()]).transpose()
nps = np.zeros(nxs.shape)
    
sig = (10., 30.) # sigma's for K0 and K1
(sig0, sig1) = sig

th = 0*np.pi
p0 = np.zeros(x0.shape)
Mod0 ={ '0': x0, 'sig':sig[0], 'coeff':coeffs[0]}


th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])
    
Mod1 = {'x,R':(x1,R), 'sig':sig[1], 'coeff' :coeffs[1], 'nu' : nu}
C = np.zeros((x1.shape[0],2,1))
K = 10
C[:,0,0] = K*(38. - x1[:,1])/38.
C[:,1,0] = K*(38. - x1[:,1])/38.

L = 38.
a, b = (2.)/(2*L), -(2.)/(2*L*L)
C[:,0,0] = 0.85*K*(b*(38. - x1[:,1])**2/2 + a*(38. - x1[:,1]))
C[:,1,0] = K*(b*(38. - x1[:,1])**2/2 + a*(38. - x1[:,1]))

L = 38.
a, b = -2/L**3, 3/L**2
C[:,1,0] = K*(a*(38. - x1[:,1])**3 + b*(38. - x1[:,1])**2)
C[:,0,0] = 0.9*C[:,1,0]

Mod1['C'] = C

# Definition of geodesic momenta (put in Cot)
ps = np.zeros(xs.shape)
ps[0:4,1] = 2.
ps[22:26,1] = 2.
    
(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))

#xs, ps = x0, p0

Cot ={ '0':[(x0,p0), (xs,ps)], 'x,R':[((x1,R),(p1,PR))]}

# creates kernel matrix for points in Mod0
Mod0['SKS'] = fun.my_new_SKS(Mod0)

# Computes \xi_m ^\ast (\eta) as an element of V0
# with m the whole GD
Cot0 = fields.my_CotToVs(Cot,sig0)

# returns the element of H_0^\ast equal to 
# \zeta^0_x0 (Cot0)
v0 = fields.my_VsToV(Cot0, x0,0)

#Computes the geodesic control associated to Cot for Mod0
Mod0['mom'] = solve(Mod0['coeff']*Mod0['SKS'],
        v0.flatten(),sym_pos = True).reshape(x0.shape)

# computes cost0
modop.my_mod_update(Mod0) 


# define the operator that takes lagrangian momentum and returns "constraints"
Mod1['SKS'] = fun.my_new_SKS(Mod1)

# Computes \xi_m ^\ast (\eta) as an element of V1
# with m the whole GD
Cot1 = fields.my_CotToVs(Cot,sig1)


# Computes geodesic controls
# Creates S so that ??
dv = fields.my_VsToV(Cot1,x1,1)
S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2, fun_eta.my_eta())
# Computes ??
tlam = solve(Mod1['coeff']*Mod1['SKS'], S.flatten(), sym_pos = True)
(Am, AmKiAm) = con_fun.my_new_AmKiAm(Mod1)
Mod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)


# Computes cost and other variables to store them
modop.my_mod_update(Mod1) # will compute the new lam, mom and cost


#%% Shoot
N =3
Traj = shoot.my_fd_shoot(Mod0, Mod1, Cot,N)

#%% Plot

fig = plt.figure(5)
plt.clf()
for i in range(N+1):
    plt.clf()
    (tMod0, tMod1, tCot) = Traj[2*i]
    visu.my_plot4(tMod0,tMod1,tCot, fig,nx,ny, name, i)

filepng = name + "*.png"
#os.system("convert " + filepng + " -set delay 20 -reverse " + filepng + " -loop 0 " + name + ".gif")  
os.system("convert " + filepng + " -set delay 0 -reverse " + filepng + " -loop 0 " + name + ".gif")  








