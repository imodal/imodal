
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

param_sil = (xs, np.zeros(xs.shape))
param_0 = (x0, np.zeros(x0.shape))
param_1 = ((x1, R), (np.zeros(x1.shape), np.zeros(R.shape)))

#%%
Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model0, Model1])

param = [param_sil, param_0, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init_opti.GD.fill_cot_from_param(param)
Mod_el_opti = Mod_el_init_opti.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
#%%
lam_var = 1.
sig_var = 10.
N=5
args = (Mod_el, xst, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P0,
    args = args,
    method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
    options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
    'eps': 1e-08, 'maxfun': 100, 'maxiter': 5, 'iprint': -1, 'maxls': 20})
#%%
P1 = res['x']


#%%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)
#%%
Modlist_opti = shoot.shooting_traj(Mod_el_opti, N)
#%%
xsf = Modlist_opti[-1].GD.Cot['0'][0][0]
#%%

plt.plot(xs[:,0], xs[:,1], '-b')
plt.plot(xst[:,0], xst[:,1], '-r')
plt.plot(xsf[:,0], xsf[:,1], '-g')
plt.axis('equal')
