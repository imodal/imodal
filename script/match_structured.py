
import scipy .optimize
import numpy as np
import matplotlib.pyplot as plt

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
import implicitmodules.Backward.ScipyOptimise as opti
#%%
sig0 = 30
dim = 2
Sil = defmod.SilentLandmark(xs.shape[0], dim)
Model1 = defmod.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model01 = defmod.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0])
Model00 = defmod.ElasticOrderO(100, 1, dim, 0.1)
#%% 

Mod_el_init = comb_mod.CompoundModules([Sil, Model0, Model1])

#%%

param_sil = (xs, 0.5*ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), np.zeros([1, 2]))
param_1 = ((x1, R), (p1, PR))

#%%
param = [param_sil, param_0, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N=5
#%%
Modlist = shoot.shooting_traj(Mod_el, N)
xst = Modlist[-1].GD.Cot['0'][0][0]
#%%

plt.plot(xs[:,0], xs[:,1], '-b')
plt.plot(xst[:,0], xst[:,1], '-r')
plt.axis('equal')
#%%

xst += 20.
#%%

param_sil = (xs, np.zeros(xs.shape))
param_0 = (x0, np.zeros(x0.shape))
param_01 = (xs, np.zeros(xs.shape))
param_1 = ((x1, R), (np.zeros(x1.shape), np.zeros(R.shape)))

#%%
Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
param = [param_sil, param_0, param_0, param_1]

Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model1])
param = [param_sil, param_1]

Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model00, Model1])
param = [param_sil, param_00, param_1]
#Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model0])
#param = [param_sil, param_0]

GD = Mod_el_init.GD.copy()
Mod_el_init_opti.GD.fill_cot_from_param(param)
Mod_el_opti = Mod_el_init_opti.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
#%%
lam_var = 1.
sig_var = 20.
N=5
args = (Mod_el_opti, xst, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P0,
    args = args,
    method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
    'eps': 1e-08, 'maxfun': 100, 'maxiter': 10, 'iprint': -1, 'maxls': 20})
#%%
P1 = res['x']

#%%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)
#%%
Modlist_opti = shoot.shooting_traj(Mod_el_opti, N)
#%%
xsf = Modlist_opti[-1].GD.Cot['0'][0][0]
#%%
i=5
xs_i = Modlist_opti[2*i].GD.Cot['0'][0][0]
plt.plot(xs[:,0], xs[:,1], '-b')
plt.plot(xst[:,0], xst[:,1], '-r')
plt.plot(xs_i[:,0], xs_i[:,1], '-g')
plt.axis('equal')
#plt.plot(xsf[:,0], xsf[:,1], '-g')

#%% Plot with grid
height = 40
nx, ny = (5,11) # create a grid for visualisation purpose
u = height/38.
Dx = 0.
Dy = 0.
(a,b,c,d) = (-10.*u + Dx, 10.*u + Dx, -3.*u + Dy, 40.*u + Dy)
[xx, xy] = np.meshgrid(np.linspace(a,b,nx), np.linspace(c,d,ny))
(nx,ny) = xx.shape
grid_points= np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = defmod.SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
#%%
xgrid = grid_points.copy()
xsx = xgrid[:,0].reshape((nx,ny))
xsy = xgrid[:,1].reshape((nx,ny))
plt.plot(xsx, xsy, color = 'lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color = 'lightblue')
plt.plot(xs[:,0], xs[:,1], '-b')
plt.axis('equal')
#%%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)
Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el_opti])
#Mod_tot
#%%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)

#%%
i=0
xgrid = Modlist_opti_tot[2*i].GD.Cot['0'][0][0]
xs_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][0]
xsx = xgrid[:,0].reshape((nx,ny))
xsy = xgrid[:,1].reshape((nx,ny))
plt.plot(xsx, xsy, color = 'lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color = 'lightblue')
plt.plot(xs[:,0], xs[:,1], '-b')
plt.plot(xst[:,0], xst[:,1], '-r')
plt.plot(xs_i[:,0], xs_i[:,1], '-g')
plt.axis('equal')
#plt.plot(xsf[:,0], xsf[:,1], '-g')


