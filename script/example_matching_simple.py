import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import src.DeformationModules.Combination as comb_mod
import src.DeformationModules.ElasticOrder0 as defmod0
import src.DeformationModules.ElasticOrder1 as defmod1
import src.DeformationModules.SilentLandmark as defmodsil
import src.Forward.Shooting as shoot
import src.Optimisation.ScipyOpti as opti
from src.Utilities import Rotation as rot
from src.Utilities.visualisation import my_close


# helper function
def my_plot(x, title="", col='*b'):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], col)
    plt.title(title)
    plt.axis('equal')
    plt.show()


# Source
path_data = '../data/'
with open(path_data + 'basi1b.pkl', 'rb') as f:
    _, lx = pickle.load(f)

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = 38. / (lmax - lmin)

nlx[:, 1] = 38.0 - scale * (nlx[:, 1] - lmin)
nlx[:, 0] = scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

# %% target
with open(path_data + 'basi1t.pkl', 'rb') as f:
    _, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = 100. / (lmax - lmin)

nlxt[:, 1] = 38.0 - scale * (nlxt[:, 1] - lmin)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]

#  common options
nu = 0.001
dim = 2

# %% Silent Module
xs = nlx[nlx[:, 2] == 2, 0:2]
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
ps = np.zeros(xs.shape)
param_sil = (xs, ps)

my_plot(xs, "Silent Module", '*b')

# %% Modules of Order 0
sig0 = 6
x0 = nlx[nlx[:, 2] == 1, 0:2]
Model0 = defmod0.ElasticOrderO(sig0, x0.shape[0], dim, 1., nu)
p0 = np.zeros(x0.shape)
param_0 = (x0, p0)

my_plot(x0, "Module order 0", 'or')

# %% Modules of Order 0
sig00 = 200
x00 = np.array([[0., 0.]])
Model00 = defmod0.ElasticOrderO(sig00, x00.shape[0], dim, 0.1, nu)
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

my_plot(x00, "Module order 00", '+r')

# %% Modules of Order 1
sig1 = 30
x1 = nlx[nlx[:, 2] == 1, 0:2]
C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, 38
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = K * (a * (L - x1[:, 1]) ** 3 + b * (L - x1[:, 1]) ** 2)
C[:, 0, 0] = 1. * C[:, 1, 0]
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, 0.01, C, nu)

th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

my_plot(x1, "Module order 1", 'og')

# %% Full model
Mod_el = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
Mod_el.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
P0 = opti.fill_Vector_from_GD(Mod_el.GD)



# %%
lam_var = 10.
sig_var = 30.
N = 10
args = (Mod_el, xst, lam_var, sig_var, N, 1e-7)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B',
                              jac=opti.jac,
                              bounds=None,
                              tol=None,
                              callback=None,
                              options={
                                  'maxcor': 10,
                                  'ftol': 1.e-09,
                                  'gtol': 1e-03,
                                  'eps': 1e-08,
                                  'maxfun': 100,
                                  'maxiter': 15,
                                  'iprint': 1,
                                  'maxls': 25
                              })

opti.fill_Mod_from_Vector(res['x'], Mod_el)
Modlist_opti = shoot.shooting_traj(Mod_el, N)

# %% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modlist_opti[2 * i].GD.GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)

    x0_i = Modlist_opti[2 * i].GD.GD_list[1].GD
    plt.plot(x0_i[:, 0], x0_i[:, 1], '*r', linewidth=2)

    x00_i = Modlist_opti[2 * i].GD.GD_list[2].GD
    plt.plot(x00_i[:, 0], x00_i[:, 1], 'or', linewidth=2)

    # x1_i = Modlist_opti[2 * i].GD.GD_list[3].GD
    # plt.plot(x1_i[:, 0], x1_i[:, 1], 'og', linewidth=2)

    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    plt.show()
