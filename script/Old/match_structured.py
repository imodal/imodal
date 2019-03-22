# %%
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import old.Backward.ScipyOptimise as opti
import old.Forward.shooting as shoot
from old import DeformationModules as comb_mod
from old.DeformationModules.ElasticOrder0 import ElasticOrderO
from old.DeformationModules.ElasticOrder1 import ElasticOrder1
from old.DeformationModules.SilentLandmark import SilentLandmark
from src.Utilities import Rotation as rot
# %%
from src.Utilities.Visualisation import my_close

path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'Results' + os.path.sep
os.makedirs(path_res, exist_ok=True)
path_data = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '..' + os.path.sep + 'data' + os.path.sep

# %%
import pickle

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

# %% target
with open(path_data + 'basi1t.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = 100. / (lmax - lmin)

nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]
#
## %% autre source
#
# import pickle
# path_data = '../data/'
# height = 38.
# heightt = 209.
#
# with open(path_data + 'basi2temp.pkl', 'rb') as f:
#    img, lx = pickle.load(f)
#
## nlx = np.asarray(lx).astype(np.float32)
## (lmin, lmax) = (np.min(nlx[:,1]),np.max(nlx[:,1]))
## scale = 38./(lmax-lmin)
##
## nlx[:,1]  =  - scale*(nlx[:,1]-lmin)
## nlx[:,0]  = scale*(nlx[:,0]-np.mean(nlx[:,0]))
#
Dx = 0
Dy = 0
#
# nlx = np.asarray(lx).astype(np.float64)
# nlx[:, 1] = - nlx[:, 1]
# (lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
#
# scale = height / (lmax - lmin)
# nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0])) + Dx
# nlx[:, 1] = scale * (nlx[:, 1] - lmin) + Dy
#
# x0 = nlx[nlx[:, 2] == 0, 0:2]
# x1 = nlx[nlx[:, 2] == 1, 0:2]
# xs = nlx[nlx[:, 2] == 2, 0:2]
#
## %% autre target
# with open('../data/basi2target.pkl', 'rb') as f:
#    imgt, lxt = pickle.load(f)
#
## nlxt = np.asarray(lxt).astype(np.float32)
## (lmin, lmax) = (np.min(nlxt[:,1]),np.max(nlxt[:,1]))
## scale = 209./(lmax-lmin)
##
## nlxt[:,1]  = 90.0 - scale*(nlxt[:,1]-lmin) #+ 400
## nlxt[:,0]  = scale*(nlxt[:,0]-np.mean(nlxt[:,0]))
#
# nlxt = np.asarray(lxt).astype(np.float64)
# nlxt[:, 1] = -nlxt[:, 1]
# (lmint, lmaxt) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
# scale = heightt / (lmaxt - lmint)
#
# nlxt[:, 1] = scale * (nlxt[:, 1] - lmint)
# nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))
#
# xst = nlxt[nlxt[:, 2] == 2, 0:2]

# %% parameter for module of order 1
th = 0 * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

# C = np.zeros((x1.shape[0], 2, 1))
# L = 38.
# K = 10
# a, b = -2 / L ** 3, 3 / L ** 2
#
#
# def define_C0(x, y):
#    return 1
#
# def define_C1(x, y):
#    return K * (a * (38. - y) ** 3 + b * (38. - y) ** 2) + 10  # - K**3 * a*2 *  x**2
#
#
# def define_C1(x,y):
#    return K*(a*y + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2
#
# C[:, 1, 0] = define_C1(x1[:, 0], x1[:, 1]) * define_C0(x1[:, 0], x1[:, 1])
# C[:, 0, 0] = 1. * C[:, 1, 0]
##
## C = np.ones((x1.shape[0],2,1))
## ymin = min(xs[:,1])
## ymax = max(xs[:,1])
## def define_C1(x,y):
##    return (y - ymin)/(ymax - ymin)
#
# C[:,1,0] = define_C1(x1[:,0], x1[:,1])
# C[:,0,0] = 1.*C[:,1,0]

height = 38.
C = np.zeros((x1.shape[0], 2, 1))
K = 10

L = height
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = K * (a * (L - x1[:, 1] + Dy) ** 3 + b * (L - x1[:, 1] + Dy) ** 2)
C[:, 0, 0] = 1. * C[:, 1, 0]

##
#
## %% new shapes
#
# th = 0 * np.pi
# th = th * np.ones(x1.shape[0])
# R = np.asarray([rot.my_R(cth) for cth in th])
# for i in range(x1.shape[0]):
#    R[i] = rot.my_R(th[i])
#
# C = np.zeros((x1.shape[0], 2, 1))
#
# K = 10
#
# height = 38.
# L = height
#
# a = 1 / L
#
# b = 3.
# Dy = 0.
#
# z = a * (x1[:, 1] - Dy)
#
# C[:, 1, 0] = K * ((1 - b) * z ** 2 + b * z)
#
# C[:, 0, 0] = 0.9 * C[:, 1, 0]

# %% plot C profile
# plt.figure()
# X = np.linspace(0, 38, 100)
## Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
# Y = define_C1(0, X)
# plt.plot(Y, X, '-')
# plt.ylabel('x(2)')
# plt.xlabel('C')
##plt.axis('equal')
# plt.axis([0,30,0,40])
##plt.savefig(path_res + 'C_profil_match.pdf', format='pdf', bbox_inches = 'tight')

#
#
#
##%%
# plt.figure()
# X = np.linspace(-10, 10, 100)
# Z = define_C1(X, 30 + np.zeros(X.shape))
# plt.plot(X, Z, '-')
# %%
x00 = np.array([[0., 0.]])
coeffs = [1., 0.01]
sig0 = 20
sig00 = 200
sig1 = 30
nu = 0.001
dim = 2
Sil = SilentLandmark(xs.shape[0], dim)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
# Model01 = DeformationModules.ElasticOrder1.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = ElasticOrderO(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
# %%


# Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
Mod_el_init = comb_mod.CompoundModules([Sil, Model00, Model1])
# Mod_el_init = comb_mod.CompoundModules([Model00, Model0])

# Mod_el_init = comb_mod.CompoundModules([Sil, Model1])

# %%
p0 = np.zeros(x0.shape)
ps = np.zeros(xs.shape)
# ps[0:4, 1] = 2.
# ps[22:26, 1] = 2.
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_sil = (xs, 0.3 * ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), np.zeros([1, 2]))
param_1 = ((x1, R), (p1, PR))

# %%
# param = [param_sil, param_00, param_0, param_1]
param = [param_sil, param_00, param_1]
# param = [param_00, param_0]
# param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5
#
## %%
# Modlist = shoot.shooting_traj(Mod_el, N)
## xst = Modlist[-1].GD.Cot['0'][0][0].copy()
## %%
## xst[:,0] += 30
## xst[:,1] +=50
#
## %%
#
# plt.plot(xs[:, 0], xs[:, 1])
# plt.plot(xst[:, 0], xst[:, 1])
# plt.axis('equal')
# %%

param_sil = (xs, np.zeros(xs.shape))
param_0 = (x0, np.zeros(x0.shape))
param_00 = (x00, np.zeros(x00.shape))
param_01 = (xs, np.zeros(xs.shape))
param_1 = ((x1, R), (np.zeros(x1.shape), np.zeros(R.shape)))

# %%
# Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
# param = [param_sil, param_00, param_0, param_1]

# Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model1])
# param = [param_sil, param_1]
##
Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model00, Model1])
param = [param_sil, param_00, param_1]

# Mod_el_init_opti = comb_mod.CompoundModules([Sil, Model0])
# param = [param_sil, param_0]

GD = Mod_el_init.GD.copy()
Mod_el_init_opti.GD.fill_cot_from_param(param)
Mod_el_opti = Mod_el_init_opti.copy_full()
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
# %%µ
P1 = res['x']

# %%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)
# %%
Modlist_opti = shoot.shooting_traj(Mod_el_opti, N)
# %%
xsf = Modlist_opti[-1].GD.Cot['0'][0][0]
# %%

xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti[2 * i].GD.Cot['0'][0][0]
    xs_ic = my_close(xs_i)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-b', linewidth=1)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    x1_i = Modlist_opti[2 * i].GD.Cot['x,R'][0][0][0]
    plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
    # x00_i = Modlist_opti[2 * i].GD.Cot['0'][2][0]
    # plt.plot(x00_i[:, 0], x00_i[:, 1], '.r')
    # x00_i = Modlist_opti[2 * i].GD.Cot['0'][3][0]
    # plt.plot(x00_i[:, 0], x00_i[:, 1], 'xr')  # , markersize=1)
    plt.axis('equal')
##%%
# i=5
# xs_i = Modlist_opti[2*i].GD.Cot['0'][0][0]
# plt.plot(xs[:,0], xs[:,1], '-+b')
# plt.plot(xst[:,0], xst[:,1], '-r')
# plt.plot(xs_i[:,0], xs_i[:,1], '-xg')
# plt.axis('equal')
##plt.plot(xsf[:,0], xsf[:,1], '-g')
#
## %% Plot with grid
## height = 400
## height = 40
## nx, ny = (11,21) # create a grid for visualisation purpose
## u = height/38.
## Dx = 0.
## Dy = 0.
## (a,b,c,d) = (-10.*u + Dx, 10.*u + Dx, -3.*u + Dy, 40.*u + Dy)
#
## new grid
# nx, ny = (5, 11)  # create a grid for visualisation purpose
# nx, ny = (11, 21)  # create a grid for visualisation purpose
# u = height / 38.
# (a, b, c, d) = (-10. * u + Dx, 10. * u + Dx, -3. * u + Dy, 40. * u + Dy)
#
# [xx, xy] = np.meshgrid(np.linspace(a, b, nx), np.linspace(c, d, ny))
# (nx, ny) = xx.shape
# grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()
#
# Sil_grid = DeformationModules.SilentLandmark.SilentLandmark(grid_points.shape[0], dim)
# param_grid = (grid_points, np.zeros(grid_points.shape))
# Sil_grid.GD.fill_cot_from_param(param_grid)
## %%
# xgrid = grid_points.copy()
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
# plt.plot(xs[:, 0], xs[:, 1], '-b')
# plt.plot(xst[:, 0], xst[:, 1], '-k')
# plt.axis('equal')
## %%
## opti.fill_Mod_from_Vector(P1, Mod_el_opti)
# Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el_opti])
## Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el])
## Mod_tot
## %%
# Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)
## %% Plot with grid
# name_ex = '2order0_1order1_grid'
# name_ex = 'LDDMM_grid'
# name_ex = '1order0_1order1_grid'
# name_ex = '1order0_1order1_grid_bis1'
# name_ex = '2order0_1order1_grid_bis1'
# xst_c = my_close(xst)
# xs_c = my_close(xs)
# for i in range(N + 1):
#    plt.figure()
#    xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
#    xsx = xgrid[:, 0].reshape((nx, ny))
#    xsy = xgrid[:, 1].reshape((nx, ny))
#    plt.plot(xsx, xsy, color='lightblue')
#    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
#    xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
#    xs_ic = my_close(xs_i)
#    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
#    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
#    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
#    x1_i = Modlist_opti_tot[2 * i].GD.Cot['x,R'][0][0][0]
#    plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
#    x00_i = Modlist_opti_tot[2 * i].GD.Cot['0'][2][0]
#    plt.plot(x00_i[:, 0], x00_i[:, 1], '.r')
#    x00_i = Modlist_opti_tot[2 * i].GD.Cot['0'][3][0]
#    plt.plot(x00_i[:, 0], x00_i[:, 1], 'xr')  # , markersize=1)
#    plt.axis('equal')
#    # plt.axis([-20,50,-10,105])
#    # plt.axis([-40,30,-70,45])
#    # plt.axis([-10,10,-10,55])
#    # plt.axis([-10,60,-10,70])
#    # plt.axis('off')
#    plt.savefig(path_res + name_ex + '_t_' + str(i) + '.pdf', format='pdf', bbox_inches='tight')
#
## %% Plot without grid
# name_ex = 'LDDMM'
# xst_c = my_close(xst)
# xs_c = my_close(xs)
# for i in range(N + 1):
#    plt.figure()
#    xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
#    xs_ic = my_close(xs_i)
#    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
#    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
#    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
#    plt.axis('equal')
#    plt.axis([-10, 10, -10, 55])
#    plt.axis('off')
#    plt.savefig(path_res + name_ex + '_t_' + str(i) + '.pdf', format='pdf')
#
## %%
# i = 0
# xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
# xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
# xs_ic = my_close(xs_i)
## ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## x1_str_i = x1_i[np.where(x1[:,1]>30)]
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
# plt.plot(xs[:, 0], xs[:, 1], '-b')
## plt.plot(x1_i[:,0], x1_i[:,1], 'xb')
## plt.plot(x1_str_i[:,0], x1_str_i[:,1], 'xr')
# plt.plot(xst[:, 0], xst[:, 1], '-r')
# plt.plot(xs_i[:, 0], xs_i[:, 1], '-g')
## plt.quiver(xs_i[:,0], xs_i[:,1], ps_i[:,0], ps_i[:,1])
# plt.axis('equal')
## plt.plot(xsf[:,0], xsf[:,1], '-g')
#
#
## %% initialisation without GD
# i = 0
# xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
# xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
# xs_ic = my_close(xs_i)
## ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## x1_str_i = x1_i[np.where(x1[:,1]>30)]
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
#
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## plt.plot(x1[:,0], x1[:,1], '.b')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][2][0]
## plt.plot(x00[:,0], x00[:,1], 'xr')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][3][0]
## plt.plot(x0[:,0], x0[:,1], '.r')
# plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
# plt.axis('equal')
# plt.axis([-30, 20, 0, 40])
## plt.plot(xsf[:,0], xsf[:,1], '-g')
# plt.savefig(path_res + 'matching_source.pdf', format='pdf', bbox_inches='tight')
#
## %% initialisation without GD order 1
# i = 0
# xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
# xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
# xs_ic = my_close(xs_i)
## ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## x1_str_i = x1_i[np.where(x1[:,1]>30)]
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
#
# x1_i = Modlist_opti_tot[2 * i].GD.Cot['x,R'][0][0][0]
# plt.plot(x1[:, 0], x1[:, 1], '.b')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][2][0]
## plt.plot(x00[:,0], x00[:,1], 'xr')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][3][0]
## plt.plot(x0[:,0], x0[:,1], '.r')
# plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
# plt.axis('equal')
# plt.axis([-30, 20, 0, 40])
## plt.plot(xsf[:,0], xsf[:,1], '-g')
# plt.savefig(path_res + 'matching_source_GDorder1.pdf', format='pdf', bbox_inches='tight')
#
## %% initialisatio
# i = 0
# xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
# xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
# xs_ic = my_close(xs_i)
## ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## x1_str_i = x1_i[np.where(x1[:,1]>30)]
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
#
# x1_i = Modlist_opti_tot[2 * i].GD.Cot['x,R'][0][0][0]
# plt.plot(x1[:, 0], x1[:, 1], '.b')
# x00_i = Modlist_opti_tot[2 * i].GD.Cot['0'][2][0]
# plt.plot(x00[:, 0], x00[:, 1], 'xr')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][3][0]
## plt.plot(x0[:,0], x0[:,1], '.r')
# plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
# plt.axis('equal')
## plt.plot(xsf[:,0], xsf[:,1], '-g')
#
#
## %% optimisation
# i = 0
# xgrid = Modlist_opti_tot[2 * i].GD.Cot['0'][0][0]
# xs_i = Modlist_opti_tot[2 * i].GD.Cot['0'][1][0]
# xs_ic = my_close(xs_i)
## ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
## x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
## x1_str_i = x1_i[np.where(x1[:,1]>30)]
# xsx = xgrid[:, 0].reshape((nx, ny))
# xsy = xgrid[:, 1].reshape((nx, ny))
# plt.plot(xsx, xsy, color='lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
#
# plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
# plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
# plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
# x1_i = Modlist_opti_tot[2 * i].GD.Cot['x,R'][0][0][0]
# plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
# x00_i = Modlist_opti_tot[2 * i].GD.Cot['0'][2][0]
## plt.plot(x00_i[:,0], x00_i[:,1], 'xr')
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][3][0]
## plt.plot(x00_i[:,0], x00_i[:,1], '.r')
# plt.axis('equal')
## plt.plot(xsf[:,0], xsf[:,1], '-g')
# plt.axis([-40, 30, -70, 45])
# plt.axis('off')
# plt.savefig(path_res + name_ex + '_GDorder1.pdf', format='pdf', bbox_inches='tight')
#
## %% Source et target
#
#
## %% initialisatio
# i = 0
# xs_ = xs.copy()
# xs_c = my_close(xs)
#
# xt = xst.copy()
# xt_c = my_close(xt)
## x00_i = Modlist_opti_tot[2*i].GD.Cot['0'][3][0]
## plt.plot(x0[:,0], x0[:,1], '.r')
# plt.plot(xs_c[:, 0], xs_c[:, 1], '-b')
# plt.plot(xt_c[:, 0], xt_c[:, 1], '-k')
# plt.axis('equal')
## plt.plot(xsf[:,0], xsf[:,1], '-g')
# plt.axis('off')
# plt.savefig(path_res + 'leaf_source_target.pdf', format='pdf',
#            bbox_inches='tight')
