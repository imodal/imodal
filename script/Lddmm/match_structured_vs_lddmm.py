import os.path

path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'results' + os.path.sep
os.makedirs(path_res, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
# %%
import implicitmodules.numpy.Optimisation.ScipyOpti as opti
import implicitmodules.numpy.Utilities.Rotation as rot
from implicitmodules.numpy.Utilities.Visualisation import my_close

from implicitmodules.numpy.DeformationModules.Combination import CompoundModules
from implicitmodules.numpy.DeformationModules.ElasticOrder0 import ElasticOrder0
from implicitmodules.numpy.DeformationModules.ElasticOrder1 import ElasticOrder1
from implicitmodules.numpy.DeformationModules.SilentLandmark import SilentLandmark

import implicitmodules.numpy.HamiltonianDynamic.Forward as shoot


# %%
import pickle

with open('./data/basi1.pkl', 'rb') as f:
    img, lx = pickle.load(f)

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = 38. / (lmax - lmin)

nlx[:, 1] = 38.0 - scale * (nlx[:, 1] - lmin)
nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

x0 = nlx[nlx[:, 2] == 0, 0:2]
x1 = nlx[nlx[:, 2] == 1, 0:2]
xs = nlx[nlx[:, 2] == 2, 0:2]

#  %% target
with open('./data/basi1t.pkl', 'rb') as f:
    imgt, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = 100. / (lmax - lmin)
nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))
xst = nlxt[nlxt[:, 2] == 2, 0:2]


# %%
plt.figure()
plt.plot(xs[:, 0], xs[:, 1], '-b')
plt.plot(xst[:, 0], xst[:, 1], '-xr')
plt.axis('equal')
plt.title("Data")
plt.show()

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


# def define_C1(x,y):
#    return K*(a*y + b*(38. - y)**2) + 10# - K**3 * a*2 *  x**2

C[:, 1, 0] = define_C1(x1[:, 0], x1[:, 1]) * define_C0(x1[:, 0], x1[:, 1])
C[:, 0, 0] = 0.5 * C[:, 1, 0]
#
# C = np.ones((x1.shape[0],2,1))
# ymin = min(xs[:,1])
# ymax = max(xs[:,1])
# def define_C1(x,y):
#    return (y - ymin)/(ymax - ymin)

# C[:,1,1] = define_C1(x1[:,0], x1[:,1])
# C[:,0,0] = 1.*C[:,1,1]
##
#
##%% plot C profile
# plt.figure()
# X = np.linspace(0,38,100)
##Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
# Y = define_C1(0,X)
# plt.plot(X, Y, '-')
# plt.figure()
# X = np.linspace(-10, 10,100)
# Z = define_C1(X, 30)
# plt.plot(X,Z, '-')
# %%
x00 = np.array([[0., 0.]])
coeffs = [5., 0.05]
sig0 = 10
sig00 = 1000
sig1 = 30
nu = 0.001
dim = 2
Sil = SilentLandmark(xs.shape[0], dim)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
Model01 = ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
Model0 = ElasticOrder0(sig0, x0.shape[0], dim, coeffs[0], nu)
Model00 = ElasticOrder0(sig00, x00.shape[0], dim, 0.1, nu)
# %%

# Mod_el_init = CompoundModules([Sil, Model00, Model1])
Mod_el_init = CompoundModules([Sil, Model1])

# %%
p0 = np.zeros(x0.shape)
ps = np.zeros(xs.shape)
ps[0:4, 1] = 2.
ps[22:26, 1] = 2.
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_sil = (xs, 0.8 * ps)
param_0 = (x0, p0)
param_00 = (np.zeros([1, 2]), np.zeros([1, 2]))
param_1 = ((x1, R), (p1, PR))

# %%
# param = [param_sil, param_0, param_1]
param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5

# %%
Modlist = shoot.shooting_traj(Mod_el, N)
xst = Modlist[-1].GD.GD_list[0].cotan.copy()

# %%
# param = [param_sil, param_0, param_1]
param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

# %% Plot with grid
height = 40
nx, ny = (11, 21)  # create a grid for visualisation purpose
u = height / 38.
Dx = 0.
Dy = 0.
(a, b, c, d) = (-10. * u + Dx, 10. * u + Dx, -3. * u + Dy, 40. * u + Dy)
[xx, xy] = np.meshgrid(np.linspace(a, b, nx), np.linspace(c, d, ny))
(nx, ny) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
##%%
# xgrid = grid_points.copy()
# xsx = xgrid[:,0].reshape((nx,ny))
# xsy = xgrid[:,1].reshape((nx,ny))
# plt.plot(xsx, xsy, color = 'lightblue')
# plt.plot(xsx.transpose(), xsy.transpose(), color = 'lightblue')
# plt.plot(xs[:,0], xs[:,1], '-b')
# plt.axis('equal')

# %%
# opti.fill_Mod_from_Vector(P1, Mod_el_opti)
# Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el_opti])
Mod_tot = CompoundModules([Sil_grid, Mod_el])
# Mod_tot
# %%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)
# %% Plot with grid
name_ex = 'Elastic_shoot_ex_grid'
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xsx = xgrid[:, 0].reshape((nx, ny))
    xsy = xgrid[:, 1].reshape((nx, ny))
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    plt.axis([-10, 10, -10, 55])
    plt.title(name_ex)
    plt.show()
    # plt.axis('off')
    # plt.savefig(path_res + name_ex + '_t_' + str(i) + '.pdf', format='pdf')

# %%
xs_c = my_close(xs)
xst_c = my_close(xst)
plt.plot(xs_c[:, 0], xs_c[:, 1], 'x-b')
# plt.plot(xs[:4,0], xs[:4,1], 'xb')
# plt.plot(xs[22:26,0], xs[22:26,1], 'xb')
plt.plot(xst_c[:, 0], xst_c[:, 1], 'x-r')
plt.axis('equal')
plt.show()
# %%

# xst[:,0] += 40.
# xst += 2*np.random.rand(*xst.shape)
# %%

param_sil = (xs, np.zeros(xs.shape))
param_0 = (x0, np.zeros(x0.shape))
param_00 = (x00, np.zeros(x00.shape))
param_01 = (xs, np.zeros(xs.shape))
param_1 = ((x1, R), (np.zeros(x1.shape), np.zeros(R.shape)))

# %%
Mod_el_init_opti = CompoundModules([Sil, Model00, Model0, Model1])
param = [param_sil, param_00, param_0, param_1]

# Mod_el_init_opti = CompoundModules([Sil, Model1])
# param = [param_sil, param_1]
#
Mod_el_init_opti = CompoundModules([Sil, Model00, Model1])
param = [param_sil, param_00, param_1]

Mod_el_init_opti = CompoundModules([Sil, Model0])
param = [param_sil, param_0]

GD = Mod_el_init.GD.copy()
Mod_el_init_opti.GD.fill_cot_from_param(param)
Mod_el_opti = Mod_el_init_opti.copy_full()
P0 = opti.fill_Vector_from_GD(Mod_el_opti.GD)
# %%
lam_var = 10.
sig_var = 10.
N = 10
args = (Mod_el_opti, xst, lam_var, sig_var, N, 0.001)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B', jac=opti.jac, bounds=None, tol=None, callback=None,
                              options={'maxcor': 10, 'ftol': 1.e-09, 'gtol': 1e-03,
                                       'eps': 1e-08, 'maxfun': 100, 'maxiter': 25, 'iprint': -1, 'maxls': 20})
# %%µ
P1 = res['x']

# %%
opti.fill_Mod_from_Vector(P1, Mod_el_opti)
##%%
# Modlist_opti = shoot.shooting_traj(Mod_el_opti, N)
##%%
# xsf = Modlist_opti[-1].GD.Cot['0'][0][0]

##%%
# i=5
# xs_i = Modlist_opti[2*i].GD.Cot['0'][0][0]
# plt.plot(xs[:,0], xs[:,1], '-+b')
# plt.plot(xst[:,0], xst[:,1], '-r')
# plt.plot(xs_i[:,0], xs_i[:,1], '-xg')
# plt.axis('equal')
##plt.plot(xsf[:,0], xsf[:,1], '-g')

# %% Plot with grid
height = 40
nx, ny = (11, 21)  # create a grid for visualisation purpose
u = height / 38.
Dx = 0.
Dy = 0.
(a, b, c, d) = (-10. * u + Dx, 10. * u + Dx, -3. * u + Dy, 40. * u + Dy)
[xx, xy] = np.meshgrid(np.linspace(a, b, nx), np.linspace(c, d, ny))
(nx, ny) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
# %%
xgrid = grid_points.copy()
xsx = xgrid[:, 0].reshape((nx, ny))
xsy = xgrid[:, 1].reshape((nx, ny))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
plt.plot(xs[:, 0], xs[:, 1], '-b')
plt.plot(xst[:, 0], xst[:, 1], '-k')
plt.axis('equal')
plt.show()
# %%
# opti.fill_Mod_from_Vector(P1, Mod_el_opti)
Mod_tot = CompoundModules([Sil_grid, Mod_el_opti])
# Mod_tot = comb_mod.CompoundModules([Sil_grid, Mod_el])
# Mod_tot
# %%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)

# %% Plot with grid
name_ex = 'LDDMM_grid'
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xsx = xgrid[:, 0].reshape((nx, ny))
    xsy = xgrid[:, 1].reshape((nx, ny))
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    plt.axis([-10, 10, -10, 55])
    plt.title(name_ex)
    plt.show()
    #  plt.axis('off')
    # plt.savefig(path_res + name_ex + '_t_' + str(i) + '.pdf', format='pdf')

# %% Plot without grid
name_ex = 'LDDMM'
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    plt.axis([-10, 10, -10, 55])
    plt.title(name_ex)
    plt.show()
    #  plt.axis('off')
    # plt.savefig(path_res + name_ex + '_t_' + str(i) + '.pdf', format='pdf')

# %%
i = 0
xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
# ps_i = Modlist_opti_tot[2*i].GD.Cot['0'][1][1]
# x1_i = Modlist_opti_tot[2*i].GD.Cot['x,R'][0][0][0]
# x1_str_i = x1_i[np.where(x1[:,1]>30)]
xsx = xgrid[:, 0].reshape((nx, ny))
xsy = xgrid[:, 1].reshape((nx, ny))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
plt.plot(xs[:, 0], xs[:, 1], '-b')
# plt.plot(x1_i[:,0], x1_i[:,1], 'xb')
# plt.plot(x1_str_i[:,0], x1_str_i[:,1], 'xr')
plt.plot(xst[:, 0], xst[:, 1], '-r')
plt.plot(xs_i[:, 0], xs_i[:, 1], '-g')
# plt.quiver(xs_i[:,0], xs_i[:,1], ps_i[:,0], ps_i[:,1])
plt.axis('equal')
plt.show()
# plt.plot(xsf[:,0], xsf[:,1], '-g')
