import pickle

import numpy as np
import numpy.Forward.shooting as shoot
import numpy.data_attachment.varifold as var

import implicitmodules.numpy.DeformationModules.Combination as comb_mod
import implicitmodules.numpy.DeformationModules.ElasticOrder0 as defmod0
import implicitmodules.numpy.DeformationModules.ElasticOrder1 as defmod1
import implicitmodules.numpy.DeformationModules.SilentLandmark as defmodsil
from implicitmodules.numpy.Utilities import Rotation as rot

# %%
# path_res = "/home/barbaragris/Results/ImplicitModules/"
# %%

with open('../data/basi1.pkl', 'rb') as f:
    img, lx = pickle.load(f)

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = 38. / (lmax - lmin)

nlx[:, 1] = 38.0 - scale * (nlx[:, 1] - lmin)
nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

x0 = nlx[nlx[:, 2] == 0, 0:2]
x1 = nlx[nlx[:, 2] == 1, 0:2]
xs = nlx[nlx[:, 2] == 2, 0:2]

# %% target
with open('../data/basi1t.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = 100. / (lmax - lmin)

nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]

# %% autre source

height = 38.
heightt = 209.

with open('../data/basi2temp.pkl', 'rb') as f:
    img, lx = pickle.load(f)

# nlx = np.asarray(lx).astype(np.float32)
# (lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
# scale = 38./(lmax-lmin)
#
# nlx[:,1]  =  - scale*(nlx[:,1]-lmin)
# nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))

Dx = 0
Dy = 0

nlx = np.asarray(lx).astype(np.float64)
nlx[:, 1] = - nlx[:, 1]
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))

scale = height / (lmax - lmin)
nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0])) + Dx
nlx[:, 1] = scale * (nlx[:, 1] - lmin) + Dy

x0 = nlx[nlx[:, 2] == 0, 0:2]
x1 = nlx[nlx[:, 2] == 1, 0:2]
xs = nlx[nlx[:, 2] == 2, 0:2]

# %% autre target
with open('../data/basi2target.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

# nlxt = np.asarray(lxt).astype(np.float32)
# (lmin, lmax) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
# scale = 209./(lmax-lmin)
#
# nlxt[:,1]  = 90.0 - scale*(nlxt[:,1]-lmin) #+ 400
# nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))

nlxt = np.asarray(lxt).astype(np.float64)
nlxt[:, 1] = -nlxt[:, 1]
(lmint, lmaxt) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = heightt / (lmaxt - lmint)

nlxt[:, 1] = scale * (nlxt[:, 1] - lmint)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]

# %% parameter for module of order 1
th = 0 * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0], 2, 1))
L = 38.
K = 10
a, b = -2 / L ** 3, 3 / L ** 2


def define_C0(x, y):
    return 1


def define_C1(x, y):
    return K * (a * (38. - y) ** 4 + b * (38. - y) ** 2) + 10  # - K**3 * a*2 *  x**2


def define_C1(x, y):
    return K * (a * (38. - y) ** 3 + b * (38. - y) ** 2) + 10  # - K**3 * a*2 *  x**2


# def define_C1(x,y):
#    return K*(a*y + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2

C[:, 1, 0] = define_C1(x1[:, 0], x1[:, 1]) * define_C0(x1[:, 0], x1[:, 1])
C[:, 0, 0] = 1. * C[:, 1, 0]
#
# C = np.ones((x1.shape[0],2,1))
# ymin = min(xs[:,1])
# ymax = max(xs[:,1])
# def define_C1(x,y):
#    return (y - ymin)/(ymax - ymin)

# C[:,1,1] = define_C1(x1[:,0], x1[:,1])
# C[:,0,0] = 1.*C[:,1,1]
##

# %% new shapes

th = 0 * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0], 2, 1))

K = 10

height = 38.
L = height

a = 1 / L

b = 3.
Dy = 0.

z = a * (x1[:, 1] - Dy)

C[:, 1, 0] = K * ((1 - b) * z ** 2 + b * z)

C[:, 0, 0] = 0.9 * C[:, 1, 0]
#
#
##%% plot C profile
# plt.figure()
# X = np.linspace(0,38,100)
##Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
# Y = define_C1(0,X)
# plt.plot(Y, X, '-')
# plt.ylabel('x(2)')
# plt.xlabel('C')
##plt.axis('equal')
# plt.axis([0,30,0,40])
# plt.savefig(path_res + 'C_profil_match.pdf', format='pdf', bbox_inches = 'tight')
##%%
# plt.figure()
# X = np.linspace(-10, 10,100)
# Z = define_C1(X, 30 + np.zeros(X.shape))
# plt.plot(X,Z, '-')
# %%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 30
sig00 = 500
sig1 = 50
nu = 0.001
dim = 2
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
# Model01 = defmod.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, 0.1, nu)
# %%

Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])

# Mod_el_init = comb_mod.CompoundModules([Model00, Model0, Model1])
# Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

# Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

# %%
p00 = np.zeros([1, 2])
p00 = np.random.rand(*p00.shape)

p0 = np.zeros(x0.shape)
p0 = np.random.rand(*p0.shape)
ps = np.random.rand(*xs.shape)
ps[0:4, 1] = 2.
ps[22:26, 1] = 2.
ps = np.zeros(xs.shape)
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
(p1, PR) = (np.random.rand(*x1.shape), np.random.rand(x1.shape[0], 2, 2))

param_sil = (xs, 0.3 * ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), p00)
param_1 = ((x1, R), (p1, PR))

# %%
param = [param_sil, param_00, param_0, param_1]
# param = [param_00, param_0]
# param = [param_00, param_0, param_1]
# param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()

# %%
Mod_el_init.GD.fill_cot_from_param(param)

# %%
Mod_el = Mod_el_init.copy_full()

N = 5
lam_var = 15.
sig_var = 10.
N = 5
args = (Mod_el, xst, lam_var, sig_var, N, 0.001)

jac = opti.jac
fun = opti.fun
# %%

P0 = opti.fill_Vector_from_GD(Mod_el.GD)

# %%
dp = jac(P0, *args)

# %%

min(dp - dP_old)
# %%


ModTraj = shoot.shooting_traj(Mod_el, N)

ModTraj_old = shoot_old.shooting_traj(Mod_el_old, N)
# %%
xsf = ModTraj[-1].ModList[0].GD.GD
(varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
dxvarcost = lam_var * dxvarcost

xsf_old = ModTraj_old[-1].ModList[0].GD.Cot['0'][0][0]
(varcost_old, dxvarcost_old) = var.my_dxvar_cost(xsf_old, xst, sig_var)
dxvarcost_old = lam_var * dxvarcost_old

# %%
eps = 0.001
grad_1 = Mod_el.GD.copy()
grad_1.fill_zero()
grad_1.GD_list[0].tan = dxvarcost
# grad_1.fill_cot_from_GD()

cgrad = bckwd.backward_shoot_rk2(ModTraj, grad_1, eps)

grad_1_old = Mod_el_old.GD.copy()
grad_1_old.fill_zero()
grad_1_old.GD_list[0].fill_GDpts(dxvarcost)
grad_1_old.fill_cot_from_GD()

cgrad_old = bck.backward_shoot_rk2(ModTraj_old, grad_1_old, eps)
# %%
cgrad.GD_list[0].tan - cgrad_old.Cot['0'][0][0]
# %%
grad_1.GD_list[0].cotan - grad_1_old.Cot['0'][0][0]
# %%


# %%
#
##%%
# Modlist_save_new = shoot.shooting_traj(Mod_el, N)
#
#
##%%
# t= -1
# i = 2
# print(Modlist_save_new[t].GD.GD_list[i].GD - Modlist[t].GD.GD_list[i].Cot['0'][0][0])
# print(Modlist_save_new[t].GD.GD_list[i].cotan- Modlist[t].GD.GD_list[i].Cot['0'][0][1])
##print(Modlist[t].GD.GD_list[0].Cot['0'][i])
##%%
# t=-1
# print(Modlist_save_new[t].GD.GD_list[3].GD[0] - Modlist[t].GD.GD_list[3].Cot['x,R'][0][0][0])
# print(Modlist_save_new[t].GD.GD_list[3].GD[1] - Modlist[t].GD.GD_list[3].Cot['x,R'][0][0][1])
##print(Modlist_save_new[t].GD.GD_list[2].cotan[0]- Modlist[t].GD.GD_list[2].Cot['x,R'][0][1][0])
##print(Modlist_save_new[t].GD.GD_list[2].cotan[1]- Modlist[t].GD.GD_list[2].Cot['x,R'][0][1][1])
#
##%%
# t=0
# i=2
# print(Modlist_save_new[t].Cont[i] - Modlist[t].Cont[i])
##%%
# t=0
# v_new = Modlist_save_new[t].field_generator_curr()
# dGD_new = Modlist_save_new[t].GD.dCotDotV(v_new)
# dGD_new = Modlist_save_new[t].GD.action(v_new)
##dGD_new = HamDer.dpH(Modlist_save_new[t])
##%%
# v = Modlist[t].field_generator_curr()
# dGD = Modlist[t].GD.dCotDotV(v)
# dGD = Modlist[t].GD.action(v)
##dGD = HamDer_old.dpH(Modlist[t])
##%%
##
# i=1
##dGD.Cot['0'][i][0]- dGD_new.GD_list[i].tan
# dGD.Cot['x,R'][0][0][1] - dGD_new.GD_list[2].tan[1]
##%%
# t=0
# Modlist[t].ModList[2].Mom -Modlist_save_new[t].ModList[2].Mom
##Modlist[t].ModList[2].Cont -Modlist_save_new[t].ModList[2].Cont
#
##%%
# grad_1 = Modlist_save_new[-1].GD.copy_full()
# grad_1.fill_zero_tan()
# grad_1.fill_zero_cotan()
##
##grad_1.GD_list[0].cotan = 0.2*grad_1.GD_list[0].GD.copy()
##grad_1.GD_list[1].cotan = 0.2*grad_1.GD_list[1].GD.copy()
##grad_1.GD_list[2].cotan = (0.2*grad_1.GD_list[2].GD[0].copy(), 0.2*grad_1.GD_list[2].GD[1].copy())
# grad_1.GD_list[0].tan = 1*np.random.rand(*grad_1.GD_list[0].GD.shape)
##grad_1.GD_list[1].tan = 0.01*np.random.rand(*grad_1.GD_list[1].GD.shape)
##grad_1.GD_list[2].tan = (0.01*np.random.rand(*grad_1.GD_list[2].GD[0].shape),0.01*np.random.rand(*grad_1.GD_list[2].GD[1].shape))
# grad_1.GD_list[0].cotan = 1*np.random.rand(*grad_1.GD_list[0].GD.shape)
##grad_1.GD_list[1].cotan = 0.01*np.random.rand(*grad_1.GD_list[1].GD.shape)
##grad_1.GD_list[2].cotan = (0.01*np.random.rand(*grad_1.GD_list[2].GD[0].shape),0.01*np.random.rand(*grad_1.GD_list[2].GD[1].shape))
#
##%%
# eps = 1e-6
# cgrad_new = bckwrd.backward_shoot_rk2(Modlist_save_new, grad_1, eps)
##out_new = bckwrd.backward_step(Modlist_save_new[-1], eps, grad_1)
##%%
#
# grad_1_o =  Modlist[-1].GD.copy()
# grad_1_o.GD_list[0].Cot['0'].append((grad_1.GD_list[0].tan.copy(), grad_1.GD_list[0].cotan.copy()))
# grad_1_o.GD_list[1].Cot['0'].append((grad_1.GD_list[1].tan.copy(), grad_1.GD_list[1].cotan.copy()))
# grad_1_o.GD_list[2].Cot['0'].append((grad_1.GD_list[2].tan.copy(), grad_1.GD_list[2].cotan.copy()))
# grad_1_o.GD_list[3].Cot['x,R'].append(((grad_1.GD_list[3].tan[0].copy(), grad_1.GD_list[3].tan[1].copy()), (grad_1.GD_list[3].cotan[0].copy(), grad_1.GD_list[3].cotan[1].copy() )))
# grad_1_o.fill_cot_init()
##%%
# cgrad = bck.backward_shoot_rk2(Modlist, grad_1_o, eps)
##out= bck.backward_step(Modlist[-1], eps, grad_1_o)
##%%
# print(out_new.GD_list[0].tan)
# print(out_new.GD_list[0].cotan)
# print(out.Cot['0'][0])
#
#
##%%
# i=2
#
# print(cgrad_new.GD_list[i].tan - cgrad.Cot['0'][i][0])
# print(cgrad_new.GD_list[i].cotan - cgrad.Cot['0'][i][1])
#
##%%
# i=3
# print(cgrad_new.GD_list[i].tan[0] - cgrad.Cot['x,R'][0][0][0])
# print(cgrad_new.GD_list[i].tan[1] - cgrad.Cot['x,R'][0][0][1])
# print(cgrad_new.GD_list[i].cotan[0] - cgrad.Cot['x,R'][0][1][0])
# print(cgrad_new.GD_list[i].cotan[1] - cgrad.Cot['x,R'][0][1][1])
#
#
#
#
#
##%%
##t = -1
##x00_f = Modlist[t].GD.Cot['0'][0][0]
##x00_f_n = Modlist_save_new[t].GD.GD_list[0].GD
##x00_f_nbis = Modlist_save_new[t].ModList[0].GD.GD
###print(x00_f_n -x00_f_nbis )
###print(x00_f -x00_f_nbis )
##print(x00_f)
###print(x00_f_nbis)
##
###%%
##print(Modlist[2].GD.Cot['0'][0][1] - Modlist[0].GD.Cot['0'][0][1])
##print(Modlist_save_new[2].ModList[0].GD.cotan - Modlist_save_new[0].ModList[0].GD.cotan)
###%%
##t = 0
##cont = Modlist[t].Cont
##cont_n = Modlist_save_new[t].Cont
##print(cont[1] - cont_n[1])
##%%
# t=2
# a = Modlist[t].cot_to_innerprod_curr(Modlist[0].GD, 1)
# b = Modlist_save_new[t].cot_to_innerprod_curr(Modlist[0].GD, 1)
#
##a = Modlist[t].DerCost_curr()
##b = Modlist_save_new[t].DerCost_curr()
#
# print(a.GD_list[0].Cot['0'][0][0] - b.GD_list[0].cotan)
#
##%%
# t=2
# i=1
# v = Modlist[t].field_generator_curr()
# v_n = Modlist_save_new[t].field_generator_curr()
# der =  Modlist[t].ModList[i].GD.dCotDotV(v)
# der_n =  Modlist_save_new[t].ModList[i].GD.dCotDotV(v_n)
##%%
# print(der_n.cotan - der.Cot['0'][0][0] )
