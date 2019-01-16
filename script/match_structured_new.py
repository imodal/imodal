
import scipy .optimize
import numpy as np
import matplotlib.pyplot as plt

import src.StructuredFields.StructuredField_0 as stru_fie0
import src.StructuredFields.Sum as stru_fie_sum

import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.Combination as comb_mod

import src.Forward.Hamiltonianderivatives as HamDer
import src.Forward.shooting as shoot
#%%
#from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, rotation as rot, shooting as shoot_old, \
#    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta, visualisation as visu
from implicitmodules.src.visualisation import my_close
from implicitmodules.src import rotation as rot
import implicitmodules.src.data_attachment.varifold as var
import implicitmodules.Backward.Backward as bckwd
import implicitmodules.Backward.ScipyOptimise as opti

#%%
#path_res = "/home/barbaragris/Results/ImplicitModules/"

#%%

import pickle
with open('../data/basi1.pkl', 'rb') as f:
    img, lx = pickle.load(f)
    
nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
scale = 38./(lmax-lmin)

nlx[:,1]  = 38.0 - scale*(nlx[:,1]-lmin)
nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))            

x0 = nlx[nlx[:,2]==0,0:2]
x1 = nlx[nlx[:,2]==1,0:2]
xs = nlx[nlx[:,2]==2,0:2]

#%% target
with open('../data/basi1t.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
scale = 100./(lmax-lmin)

nlxt[:,1]  = 38.0 - scale*(nlxt[:,1]-lmin)
nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))            

xst = nlxt[nlxt[:,2]==2,0:2]





#%% autre source

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

#%% autre target
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
L = 38.
K = 10
a, b = -2/L**3, 3/L**2
def define_C0(x,y):
    return 1
def define_C1(x,y):
    return K*(a*(38. - y)**4 + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2
def define_C1(x,y):
    return K*(a*(38. - y)**3 + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2

#def define_C1(x,y):
#    return K*(a*y + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2

C[:,1,0] = define_C1(x1[:,0], x1[:,1]) * define_C0(x1[:,0], x1[:,1])
C[:,0,0] = 1.*C[:,1,0]
#
#C = np.ones((x1.shape[0],2,1))
#ymin = min(xs[:,1])
#ymax = max(xs[:,1])
#def define_C1(x,y):
#    return (y - ymin)/(ymax - ymin)

#C[:,1,1] = define_C1(x1[:,0], x1[:,1])
#C[:,0,0] = 1.*C[:,1,1]
##

#%% new shapes

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


#%% plot C profile
plt.figure()
X = np.linspace(0,38,100)
#Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
Y = define_C1(0,X)
plt.plot(Y, X, '-')
plt.ylabel('x(2)')
plt.xlabel('C')
#plt.axis('equal')
plt.axis([0,30,0,40])
#plt.savefig(path_res + 'C_profil_match.pdf', format='pdf', bbox_inches = 'tight')
#%%
plt.figure()
X = np.linspace(-10, 10,100)
Z = define_C1(X, 30 + np.zeros(X.shape))
plt.plot(X,Z, '-')
#%%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 30
sig00 = 500
sig1 = 50
nu = 0.001
dim = 2
#Sil = defmod.SilentLandmark(xs.shape[0], dim)
#Model1 = defmod.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
#Model01 = defmod.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
#%% 

#Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])

Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

#Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

#%%
p0 = np.zeros(x0.shape)
ps = np.zeros(xs.shape)
ps[0:4,1] = 2.
ps[22:26,1] = 2.
(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
param_sil = (xs, 0.3*ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), np.zeros([1, 2]))
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_00, param_0, param_1]
#param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()

#%%
Mod_el_init.GD.fill_cot_from_param(param)


#%%
Mod_el = Mod_el_init.copy_full()

N=5

#%%
Modlist = shoot.shooting_traj(Mod_el, N)
#xst = Modlist[-1].GD.Cot['0'][0][0].copy()
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:32:35 2019

@author: gris
"""

