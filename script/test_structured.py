import numpy as np
import scipy.optimize

import old.Backward.ScipyOptimise as opti
import old.Forward.shooting as shoot
from old import DeformationModules as comb_mod, shooting as shoot_old
from old.Backward import Backward as bckwd

# %%

# %%
dim = 2
Sil = DeformationModules.SilentLandmark.SilentLandmark(xs.shape[0], dim)
Model1 = DeformationModules.ElasticOrder1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = old.DeformationModules.ElasticOrder0.ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0])

# %%

Mod_el_init = comb_mod.CompoundModules([Sil, Model0, Model1])

# %%

param_sil = (xs, ps)
param_0 = (x0, p0)
param_1 = ((x1, R), (p1, PR))

# %%
param = [param_sil, param_0, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()
# %%
#
# Mod_el.update()
#
##%%
# Mod_el.GeodesicControls_curr(Mod_el.GD)
##%%
# Mod_el.Cont[1] - Mod0['mom']
#
##%%
# v = Mod_el.field_generator_curr()
##%%
# dxH = HamDer.dxH(Mod_el)
# dpH = HamDer.dpH(Mod_el)
#
##%%
# Mod_el.add_cot(dxH)
##%%
# my_eps = 0.001
##%%
# sig_var = 1
# (varcost, dxvarcost) = var.my_dxvar_cost(xs, xs, sig_var)
##%%
# dxvarcost = np.ones(xs.shape)
# my_eps = 0.01
##%%
# grad = {'0': [(np.zeros(x0.shape), np.zeros(x0.shape)),
#              (dxvarcost, np.zeros(xs.shape))],
#        'x,R': [((np.zeros(x1.shape), np.zeros(R.shape)),
#                 (np.zeros(x1.shape), np.zeros(R.shape)))]}
#
# ngrad = shoot_old.my_sub_bckwd(Mod0, Mod1, Cot, grad, my_eps)
##%%
# Mod_el = Mod_el_init.copy_full()
# GD_grad_1 = Mod_el.GD.copy()
# param0_0 = (np.zeros(x0.shape), np.zeros(x0.shape))
# paramxR_0 = ((np.zeros(x1.shape), np.zeros(R.shape)), (np.zeros(x1.shape), np.zeros(R.shape)))
# params_0 = (dxvarcost, np.zeros(xs.shape))
#
# GD_grad_1.fill_cot_from_param([params_0, param0_0, paramxR_0])
##%%
#
# GDgrad = bckwd.backward_step(Mod_el, my_eps, GD_grad_1)
#
##%%
#
# print(sum(np.abs(GDgrad.Cot['0'][1][0] - ngrad['0'][0][0])))
#
#
# print(sum(np.abs(GDgrad.Cot['x,R'][0][1][0] - ngrad['x,R'][0][1][0])))


# %%
#
#
#
#
# %%
N = 5
Modlist = shoot.shooting_traj(Mod_el, N)

# %%

cgrad = bckwd.backward_shoot_rk2(Modlist, GD_grad_1, my_eps)

# %%
cgrad_old = shoot_old.my_bck_shoot(Traj, grad, my_eps)
# %%
print(sum(np.abs(cgrad.Cot['0'][1][1] - cgrad_old['0'][0][1])))

print(sum(np.abs(cgrad.Cot['x,R'][0][0][0] - cgrad_old['x,R'][0][0][0])))

# %%
#
##%%
# t_old= 10
# t = 10
#
# (tMod0, tMod1, tCot) = Traj[t_old]
# print(sum(np.abs(tMod0['0'] - Modlist[t].GD.Cot['0'][1][0])))
# print(sum(np.abs(tMod1['x,R'][0] - Modlist[t].GD.Cot['x,R'][0][0][0])))
# print(sum(np.abs(tMod1['x,R'][1] - Modlist[t].GD.Cot['x,R'][0][0][1])))
# print(sum(np.abs(tMod0['0'] - Modlist[t].GD.Cot['0'][1][0])))
# print(sum(np.abs(tCot['0'][1][0] - Modlist[t].GD.Cot['0'][0][0])))
#
#
#
#
##%%
#
# Cot['0'][0][1] -Mod_el.GD.Cot['0'][1][1]
#
# Cot['x,R'][0][0][1] -Mod_el.GD.Cot['x,R'][0][0][1]
#
##%%
# Mod_el.update()
# Mod_el.GeodesicControls_curr(Mod_el.GD)
##dGD = HamDer.dpH(Mod_el) #tested
# dGD = HamDer.dxH(Mod_el) # tested
##%%
#
#
# der['0'][1][1] +dGD.Cot['0'][0][1]
#
# der['x,R'][0][1][1] + dGD.Cot['x,R'][0][1][1]
##%%
#
#
# %%
Mod_tmp = Mod_el.copy_full()
# %%
P = opti.fill_Vector_from_GD(Mod_el.GD)

opti.fill_Mod_from_Vector(P, Mod_el)

# %%
i = 0
j = 1

Mod_tmp.GD.Cot['0'][i][j] - Mod_el.GD.Cot['0'][i][j]
# %%
i = 1
j = 1

Mod_tmp.GD.Cot['x,R'][0][i][j] - Mod_el.GD.Cot['x,R'][0][i][j]

# %%
dimP = P.shape[0]
dimP = int(0.5 * dimP)
PX = P[:dimP]

# %%
Mod_el.GD.Cot['0'][i][j] - PX[:52].reshape([26, 2])
# %%
Mod_tmp.GD.Cot['0'][0][0] - P[:52].reshape([26, 2])

# %%

args = (Mod_el, xs, 1, 1, N, my_eps)
opti.fun(P, *args)
# %%
lam_var = 1.
sig_var = 1.
N = 5
args = (Mod_el, xs, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P,
                              args=args,
                              method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
                              options={'disp': True, 'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
                                       'eps': 1e-08, 'maxfun': 100, 'maxiter': 5, 'iprint': -1, 'maxls': 20})

# %%
