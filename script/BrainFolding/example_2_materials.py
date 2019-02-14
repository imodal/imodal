#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:29:08 2018

@author: barbaragris
"""

import matplotlib.pyplot as plt
import numpy as np
# %%
from old import DeformationModules, DeformationModules as comb_mod
from src.Utilities import Rotation as rot

import old.Forward.shooting as shoot
#
##%%
import os.path
path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'Results' + os.path.sep
os.makedirs(path_res, exist_ok=True)

#%%
# material 0 : elastic
xmin = -10
xmax = 10
ymin = -5
ymax = 5
nx = 15
ny = 5

X0 = np.linspace(xmin, xmax, nx)
Y0 = np.linspace(ymin, ymax, ny)
Z0 = np.meshgrid(X0,Y0)

Z0 = np.reshape(np.swapaxes(Z0, 0,2), [-1, 2])

Z0_c = np.concatenate([np.array([X0, np.zeros([nx])+ymin]).transpose(), np.array([np.zeros([ny])+xmax, Y0]).transpose(), np.array([np.flipud(X0), np.zeros([nx])+ymax]).transpose(),
                      np.array([np.zeros([ny])+xmin, np.flipud(Y0)]).transpose()])


#%%
# material 1 : growth
xmin1 = -10
xmax1 = 10
ymin1 = 5
ymax1 = 7
nx1 = 15
ny1 = 2

X1 = np.linspace(xmin1, xmax1, nx1)
Y1 = np.linspace(ymin1, ymax1, ny1)
Z1 = np.meshgrid(X1,Y1)

Z1 = np.reshape(np.swapaxes(Z1, 0,2), [-1, 2])

Z1_c = np.concatenate([np.array([X1, np.zeros([nx1])+ymin1]).transpose(), np.array([np.zeros([ny1])+xmax1, Y1]).transpose(), np.array([np.flipud(X1), np.zeros([nx1])+ymax1]).transpose(),
                      np.array([np.zeros([ny1])+xmin1, np.flipud(Y1)]).transpose()])


#%%
N0 = Z0.shape[0]
N1 = Z1.shape[0]
Z = np.concatenate((Z0, Z1))

#%%
plt.plot(Z[:,0], Z[:,1], '.')

plt.plot(Z0_c[:,0], Z0_c[:,1],'-')
plt.plot(Z1_c[:,0], Z1_c[:,1],'-')
plt.axis('equal')
#%%
x1 = Z.copy()
xs0 = Z0_c .copy()
xs1 = Z1_c .copy()
xs = np.concatenate((xs0, xs1))
Ns_0 = xs0.shape[0]
Ns_1 = xs1.shape[0]
#%% parameter for module of order 1
th = 0.25*np.pi
th = th*np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for  i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0],2,1))
L = 38.
K = 100
a, b = -2/L**3, 3/L**2
#def define_C0(x,y):
#    return 1
#def define_C1(x,y):
#    return K*(a*(50. - y)**3 + b*(50. - y)**2) - 100# + 10# - K**3 * a*2 *  x**2
#def define_C1(x,y):
#    return K*(b*(50. - y)**2)# + 10# - K**3 * a*2 *  x**2
#def define_C1(x,y):
#    return K*(b*np.abs(5. - x))# + 10# - K**3 * a*2 *  x**2
#
##def define_C1(x,y):
##    return b*np.abs(y-5)
#def define_C1(x,y):
#    return np.ones(y.shape)

## First constraint
### material down 0
C[:N0,1,0] = 1.
C[:N0,0,0] = -1.
### material up 1
C[N0:,1,0] = 1.
C[N0:,0,0] = 1.

#
### Second constraint
#### material down 0
#C[:N0,1,1] = 1.
#C[:N0,0,1] = 1.
#### material up 1
#C[N0:,1,1] = -1.
#C[N0:,0,1] = 0.




#ZX = define_C1(X0, np.zeros([nx]))
#ZY = define_C1(np.zeros([ny]),Y0)
name_exp = 'linear_angle05pi'
##%% plot C profile
#plt.figure()
##X = np.linspace(0,38,100)
##Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
#ZY = define_C1(np.zeros([ny]),Y0)
#plt.plot(ZY, Y0, '-')
#
#plt.figure()
##X = np.linspace(-10, 10,100)
#ZX = define_C1(X0, np.zeros([nx]))
#plt.plot(X0, ZX, '-')
#
##%% plot C profile with shape
#
#xfigmin = -10
#xfigmax = 10
#yfigmin = -5
#yfigmax = 55
#
#
#xfigmin = -10
#xfigmax = 10
#yfigmin = -10
#yfigmax = 10
#
#from matplotlib import gridspec
#gs = gridspec.GridSpec(10, 4, width_ratios=[1, 1, 1, 1]) 
#
#endv = 7
#endh = 2
#
#ax0 = plt.subplot(gs[endv+1:,:endh])
#ax0.plot(X0, ZX, '-')
#plt.yticks([])
#
#ax1 = plt.subplot(gs[:endv, endh+1])
#ax1.plot(ZY, Y0, '-')
#plt.xticks([])
#
#
#ax2 = plt.subplot(gs[:endv, :endh])
#ax2.plot(Z[:,0], Z[:,1], '.b')
#ax2.plot(Z_c[:,0], Z_c[:,1],'-g', linewidth=2)
#ax2.axis([xfigmin, xfigmax, yfigmin, yfigmax])
#
##plt.savefig(path_res + name_exp + 'init.pdf', format='pdf')



#%%
coeffs = [5., 0.05]
sig0 = 10
sig1 = 3
nu = 0.001
dim = 2
Sil = DeformationModules.SilentLandmark(xs.shape[0], dim)
Model1 = DeformationModules.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
#Model01 = defmod.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
#Model0 = defmod.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0])
#Model00 = defmod.ElasticOrderO(100, 1, dim, 0.1)
#%% 

Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

#Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

#%%

xfigmin = -10
xfigmax = 10
yfigmin = -10
yfigmax = 10

#ind0p = [54, 55, 56, 57]
ind0p = [54, 53]
#ind0m = [40, 71, 72, 73]
ind0m = [40, 41]
ps = np.zeros(xs.shape)
#ps[nx +ny:2*nx + ny, 1] = 0.5
#ps[Ns_0:Ns_0+4, 1] = 1.
#ps[-3:-1, 1] = 1.
ps[ind0p, 1] = 1.
ps[ind0m, 1] = 1.
(p1,PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0],2,2)))
param_sil = (xs, 1*ps)
param_1 = ((x1, R), (p1, PR))

plt.plot(x1[:,0], x1[:,1], '.b')
plt.plot(xs[:,0], xs[:,1], '-g', linewidth=2)
plt.quiver(xs[:,0], xs[:,1], ps[:,0], ps[:,1])
plt.axis('equal')
#plt.axis([-10,10,-10,55])
plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
#%%
param = [param_sil, param_1]
#param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()


N=5
height = 55
nxgrid, nygrid = (11,11) # create a grid for visualisation purpose
u = height/38.
Dx = 0.
Dy = 0.
(a,b,c,d) = (-10.*u + Dx, 10.*u + Dx, -3.*u + Dy, 40.*u + Dy)
(a,b,c,d) = (xfigmin, xfigmax, yfigmin, yfigmax)
[xx, xy] = np.meshgrid(np.linspace(a,b,nxgrid), np.linspace(c,d,nygrid))
(nxgrid,nygrid) = xx.shape
grid_points= np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = DeformationModules.SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)


Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el])
#Mod_tot
#%%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)
#%% Plot with grid

for i in range(N+1):
    plt.figure()
    xgrid = Modlist_opti_tot[2*i].GD.Cot['0'][0][0]
    xsx = xgrid[:,0].reshape((nxgrid,nygrid))
    xsy = xgrid[:,1].reshape((nxgrid,nygrid))
    plt.plot(xsx, xsy, color = 'lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color = 'lightblue')
    xs_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][0]
    x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
    #xs_ic = my_close(xs_i)
    #plt.plot(xs[:,0], xs[:,1], '-b', linewidth=1)
    plt.plot(x1_i[:,0], x1_i[:,1], '.b')
    plt.plot(xs_i[:,0], xs_i[:,1], '-g', linewidth=2)
    plt.axis('equal')
    #plt.axis([-10,10,-10,55])
    plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
    #plt.axis('off')
#    plt.savefig(path_res + name_exp + '_t_' + str(i) + '.pdf', format='pdf', bbox_inches = 'tight')

#%% plot mom at t = i

i=0
plt.figure()
xgrid = Modlist_opti_tot[2*i].GD.Cot['0'][0][0]
xsx = xgrid[:,0].reshape((nxgrid,nygrid))
xsy = xgrid[:,1].reshape((nxgrid,nygrid))
plt.plot(xsx, xsy, color = 'lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color = 'lightblue')
xs_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][0]
ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
#xs_ic = my_close(xs_i)
#plt.plot(xs[:,0], xs[:,1], '-b', linewidth=1)
plt.plot(x1_i[:,0], x1_i[:,1], '.b')
plt.plot(xs_i[:,0], xs_i[:,1], '-g', linewidth=2)
plt.quiver(xs_i[:,0], xs_i[:,1], 0.1*ps_i[:,0], 0.1*ps_i[:,1], scale=1, color='r', linewidth=2)
plt.axis('equal')
#plt.axis([-10,10,-10,55])
plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.axis('off')
plt.savefig(path_res + name_exp + 'mom_t_' + str(i) + '.pdf', format='pdf', bbox_inches = 'tight')






# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:07:58 2019

@author: gris
"""

