"""
Shooting 
--------
"""

####################################################################
# Setup
# ^^^^^

import os.path

path_res = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'results' + os.path.sep
os.makedirs(path_res, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from src.DeformationModules.Combination import CompoundModules
from src.DeformationModules.ElasticOrder1 import ElasticOrder1
from src.DeformationModules.SilentLandmark import SilentLandmark
import src.Forward.Shooting as shoot
import src.Utilities.Rotation as rot

#################################################################
# create the data set

xmin, xmax = -5, 5
ymin, ymax = -15, 15
nx, ny = 10, 30

X0 = np.linspace(xmin, xmax, nx)
Y0 = np.linspace(ymin, ymax, ny)
Z0 = np.meshgrid(X0, Y0)

Z = np.reshape(np.swapaxes(Z0, 0, 2), [-1, 2])

Z_c = np.concatenate([np.array([X0, np.zeros([nx]) + ymin]).transpose(),
                      np.array([np.zeros([ny]) + xmax, Y0]).transpose(),
                      np.array([np.flip(X0), np.zeros([nx]) + ymax]).transpose(),
                      np.array([np.zeros([ny]) + xmin, np.flip(Y0)]).transpose()])

plt.plot(Z[:, 0], Z[:, 1], '.')
plt.plot(Z_c[:, 0], Z_c[:, 1], '-')
plt.axis('equal')
plt.show()

####################################################################
# Modules
# ^^^^^^^

x1 = Z.copy()
xs = Z_c.copy()

####################################################################
# parameter for module of order 1
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

th = 0. * np.pi
th = th * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
for i in range(x1.shape[0]):
    R[i] = rot.my_R(th[i])

C = np.zeros((x1.shape[0], 2, 1))
L = 38.
K = 100
a, b = -2 / L ** 3, 3 / L ** 2


def define_C0(x, y):
    return 1


def define_C1(x, y, j=0):
    if j == 0:
        return K * (a * (50. - y) ** 3 + b * (50. - y) ** 2) - 100
    elif j == 1:
        return K * (b * np.abs(5. - x))
    elif j == 2:
        return b * np.abs(y - 5)
    elif j == 3:
        return np.ones(y.shape)
    elif j == 4:
        return K * (b * (50. - y) ** 2)
    return -1


C[:, 1, 0] = define_C1(x1[:, 0], x1[:, 1]) * define_C0(x1[:, 0], x1[:, 1])
C[:, 0, 0] = 0. * C[:, 1, 0]

# sinusoidal
C[:, 1, 0] = np.cos(x1[:, 1] * np.pi / 7.5) * x1[:, 0] / 5
C[:, 0, 0] = -  np.log(2) * np.cos(x1[:, 1] * np.pi / 30)

ZX = define_C1(X0, np.zeros([nx]))
ZY = define_C1(np.zeros([ny]), Y0)
name_exp = 'linear_angle05pi'
##%% plot C profile
# plt.figure()
##X = np.linspace(0,38,100)
##Y = K*(a*(38. - X)**3 + b*(38. - X)**2)
# ZY = define_C1(np.zeros([ny]),Y0)
# plt.plot(ZY, Y0, '-')
#
# plt.figure()
##X = np.linspace(-10, 10,100)
# ZX = define_C1(X0, np.zeros([nx]))
# plt.plot(X0, ZX, '-')

# %% plot C profile with shape

xfigmin = -10
xfigmax = 10
yfigmin = -5
yfigmax = 55

xfigmin = -10
xfigmax = 10
yfigmin = -20
yfigmax = 20

from matplotlib import gridspec

gs = gridspec.GridSpec(10, 4, width_ratios=[1, 1, 1, 1])

endv = 7
endh = 2

# %%
coeffs = [5., 0.05]
sig0 = 10
sig1 = 5
nu = 0.001
dim = 2

Sil = SilentLandmark(xs.shape[0], dim)
Model1 = ElasticOrder1(sig1, x1.shape[0], dim, coeffs[1], C, nu)
# Model01 = defmod.ElasticOrder1(sig0, x1.shape[0], dim, coeffs[1], C, nu)
# Model0 = defmod.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[0])
# Model00 = defmod.ElasticOrder0(100, 1, dim, 0.1)
# %%

Mod_el_init = CompoundModules([Sil, Model1])

# %%
ps = np.zeros(xs.shape)
ps[nx + ny:2 * nx + ny, 1] = 0.5
ps[:10, :] = 10.
# 1ps[:,:,] = 1.
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_sil = (xs, 1 * ps)
param_1 = ((x1, R), (p1, PR))

# %%
param = [param_sil, param_1]
# param = [param_sil, param_1]
GD = Mod_el_init.GD.copy()
Mod_el_init.GD.fill_cot_from_param(param)
Mod_el = Mod_el_init.copy_full()

N = 5
height = 55
nxgrid, nygrid = (41, 81)  # create a grid for visualisation purpose
u = height / 38.

(a, b, c, d) = (xfigmin, xfigmax, yfigmin, yfigmax)
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))
(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = SilentLandmark(grid_points.shape[0], dim)

param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)

Mod_tot = CompoundModules([Sil_grid, Mod_el])

# %%
Modlist_opti_tot = shoot.shooting_traj(Mod_tot, N)

# %% Plot with grid
for i in range(N + 1):
    plt.figure()
    xgrid = Modlist_opti_tot[2 * i].GD.GD_list[0].GD
    xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
    xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
    xs_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[0].GD
    x1_i = Modlist_opti_tot[2 * i].GD.GD_list[1].GD_list[1].GD[0]
    plt.plot(x1_i[:, 0], x1_i[:, 1], '.b')
    plt.plot(xs_i[:, 0], xs_i[:, 1], '-g', linewidth=2)
    plt.axis('equal')
    plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
    plt.show()
