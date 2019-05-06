"""
Basipetal growth
----------------

This example shows how to perform a registration using a geodesic shooting with the implicit modules. The method allows us to define a non-uniform cost to model the growth of a leaf starting from the base.

To see another type of growth: :ref:`acropetal <sphx_glr__auto_examples_Leaf_plot_matching_acro.py>` or  :ref:`diffuse <sphx_glr__auto_examples_Leaf_plot_matching_diffuse.py>`
"""

################################################################################
# Setup
# ^^^^^

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import implicitmodules.numpy.DataAttachment.Varifold as var
import implicitmodules.numpy.DeformationModules.Combination as comb_mod
import implicitmodules.numpy.DeformationModules.ElasticOrder0 as defmod0
import implicitmodules.numpy.DeformationModules.ElasticOrder1 as defmod1
import implicitmodules.numpy.DeformationModules.SilentLandmark as defmodsil
import implicitmodules.numpy.DeformationModules.GlobalTranslation as globtrans
import implicitmodules.numpy.Forward.Shooting as shoot
import implicitmodules.numpy.Optimisation.ScipyOpti_attach as opti
from implicitmodules.numpy.Utilities import Rotation as rot
from implicitmodules.numpy.Utilities.Visualisation import my_close, my_plot

name_exp = 'basi_semi_parametric'
flag_show = True

#  common options
nu = 0.001
dim = 2
N = 10
maxiter = 500

lam_var = 2.
sig_var = [50., 10.]


#########################################################################################
# Let us define the data attachment term with a varifold like cost function.

def attach_fun(xsf, xst):
    (varcost0, dxvarcost0) = var.my_dxvar_cost(xsf, xst, sig_var[0])
    (varcost1, dxvarcost1) = var.my_dxvar_cost(xsf, xst, sig_var[1])
    costvar = varcost0 + varcost1
    dcostvar = dxvarcost0 + dxvarcost1
    return (lam_var * costvar, lam_var * dcostvar)


###################################################################################
# Load data
# ~~~~~~~~~
# The source shape is a young leaf. It is segmented from the following image
with open('./data/basi1b.pkl', 'rb') as f:
    img, lx = pickle.load(f)

# sphinx_gallery_thumbnail_number = 1
plt.imshow(img)
plt.show()

Dx = 0.
Dy = 0.
height_source = 38.
height_target = 100.

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = height_source / (lmax - lmin)

nlx[:, 1] = Dy - scale * (nlx[:, 1] - lmax)
nlx[:, 0] = Dx + scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

##################################################################################
# The target shape is the same leaf after a grown period.

with open('./data/basi1t.pkl', 'rb') as f:
    _, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = height_target / (lmax - lmin)
nlxt[:, 1] = - scale * (nlxt[:, 1] - lmax)
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0]))

xst = nlxt[nlxt[:, 2] == 2, 0:2]

####################################################################################
# Modules definitions
# ^^^^^^^^^^^^^^^^^^^


#########################################################################################
# Let us define the coefficients for modules. The higher the coefficient is,
# the more penalized are the vector fields generated by the corresponding 
# module.

coeffs = [0.01, 5, 0.01]

####################################################################################
# Silent Module
# ~~~~~~~~~~~~~
# This module allows to keep track of the shape to be transported. 
# It does not generate any vector field.

xs = nlx[nlx[:, 2] == 2, 0:2]
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
ps = np.zeros(xs.shape)
param_sil = (xs, ps)
if (flag_show):
    my_plot(xs, title="Silent Module", col='*b')

#####################################################################################
# Modules of Order 0
# ~~~~~~~~~~~~~~~~~~
# These modules create vector fields which are sums of local translations: 
# there is no strong geometric prior on these fields.

# The first module of order 0 generates local translations at a very small
# scale. It allows small variations with respect to the model.

sig0_s = 2.
x0_s = nlx[nlx[:, 2] == 1, 0:2]
x0_s = np.concatenate((x0_s, xs))
Model0_s = defmod0.ElasticOrder0(sig0_s, x0_s.shape[0], dim, coeffs[1], nu)
p0_s = np.zeros(x0_s.shape)
param_0_s = (x0_s, p0_s)

if (flag_show):
    my_plot(x0_s, title="Module order 0, small scale", col='or')

######################################################################################

# The first module of order 0 generates local translations at a medium scale.
# It allows larger variations with respect to the model.

sig0_m = 20.
x0_m = nlx[nlx[:, 2] == 1, 0:2]
Model0_m = defmod0.ElasticOrder0(sig0_m, x0_m.shape[0], dim, coeffs[1], nu)
p0_m = np.zeros(x0_m.shape)
param_0_m = (x0_m, p0_m)

if (flag_show):
    my_plot(x0_m, title="Module order 0, medium scale", col='or')

######################################################################################
# The second modules generates one global translation
# It allows to change the position of the object.

x0_t = np.array([[0., 0.]])
Model0_t =  globtrans.GlobalTranslation(dim, coeffs[0])
p0_t = np.zeros([1, 2])
param_0_t = (x0_t, p0_t)

if (flag_show):
    my_plot(x0_t, title="Module global translation", col='+r')


#######################################################################################
# Modules of order 1
# ~~~~~~~~~~~~~~~~~~
# The module of order 1 incorporates the priors in the deformation model.
# It generates vector fields whose infinitesimal strain tensors are constrained
# at points x1 via tensors C and R (see below). 

sig1 = 30.
x1 = nlx[nlx[:, 2] == 1, 0:2]
x11 = x1.copy()
x11[:, 0] *= -1
x1 = np.concatenate((x1, x11))
C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, height_source
a, b = -2 / L ** 3, 3 / L ** 2

#######################################################################################
# The tensor C below stores the eigen values for the constraints on the 
# infinitesimal strain tensor. It does not evolve during the integration of 
# the flow of diffeomorphism.
C[:, 1, 0] = (K * (a * (L - x1[:, 1] + Dy) ** 3 + b * (L - x1[:, 1] + Dy) ** 2))
C[:, 0, 0] = 1. * C[:, 1, 0]
Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)
#######################################################################################
# The tensor th below stores the eigen vectors for the constraints on the 
# infinitesimal strain tensor. It defines the tensor of rotations R which is
# a component of the geometrical descriptors and evolves during the integration
# of the flow of diffeomorphism.
th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])
(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

if (flag_show):
    my_plot(x1, ellipse=C, angles=th, title="Module order 1", col='og')

######################################################################################
# The full model possibilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We gather here the modules defined above in a compound one. There are
# several possibilities detailed below.

######################################################################################
# Non parametric module: no prior on the deformation
if name_exp == 'basi_pure_nonparametric':
    Module = comb_mod.CompoundModules([Sil, Model0_m])
    Module.GD.fill_cot_from_param([param_sil, param_0_m])
######################################################################################
# Pure parametric module: prior on the deformation via Model1, changes in the 
# global position allowed via Model0_l
if name_exp == 'basi_pure_parametric':
    Module = comb_mod.CompoundModules([Sil, Model0_t, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_0_t, param_1])
######################################################################################
# Semi parametric module: prior on the deformation via Model1, changes in the 
# global position allowed via Model0_l
if name_exp == 'basi_semi_parametric':
    Module = comb_mod.CompoundModules([Sil, Model0_t, Model0_m, Model0_s, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_0_t, param_0_m, param_0_s, param_1])
else:
    print('unknown experiment type')

######################################################################################
# First experiment: full parametric
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

name_exp = 'basi_pure_parametric'

Module = comb_mod.CompoundModules([Sil, Model0_t, Model1])
Module.GD.fill_cot_from_param([param_sil, param_0_t, param_1])

P0 = opti.fill_Vector_from_GD(Module.GD)

######################################################################################
# Optimization process
# ~~~~~~~~~~~~~~~~~~~~

args = (Module, xst, attach_fun, N, 1e-7)

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
                                  'maxfun': 500,
                                  'maxiter': maxiter,
                                  'iprint': 1,
                                  'maxls': 25
                              })

P1 = res['x']
opti.fill_Mod_from_Vector(P1, Module)
Module_optimized = Module.copy_full()
Modules_list = shoot.shooting_traj(Module, N)

######################################################################################
# Results
# ~~~~~~~~~~~~~~~~~~~~
#
# A first visualization
# ~~~~~~~~~~~~~~~~~~~~~~
xst_c = my_close(xst)
xs_c = my_close(xs)
if (flag_show):
    for i in range(N + 1):
        plt.figure()
        xs_i = Modules_list[2 * i].GD.GD_list[0].GD
        xs_ic = my_close(xs_i)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        
        x0_l_i = Modules_list[2 * i].GD.GD_list[1].GD
        plt.plot(x0_l_i[:, 0], x0_l_i[:, 1], '+r', linewidth=2)
        
        x1_i = Modules_list[2 * i].GD.GD_list[2].GD[0]
        plt.plot(x1_i[:, 0], x1_i[:, 1], 'og', linewidth=2)
        
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        plt.show()

#########################################################################################
# Plot the deformation grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Create the grid ...

hxgrid = 9
hsl = 1.2 * height_source / 2
a, b, c, d = (Dx - hsl / 2, Dx + hsl / 2, Dy, Dy + 2 * hsl)
hygrid = np.round(hxgrid * (d - c) / (b - a))
nxgrid, nygrid = (2 * hxgrid + 1, 2 * hygrid + 1)  # create a grid for visualisation purpose
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))

(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

#########################################################################################
# ... add it as a silent module to flow it
Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized])

#######################################################################################
# ... and perform the shooting

Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)

########################################################################################
# Plot with grid to show the deformation

xs_c = my_close(xs)
xst_c = my_close(xst)
if (flag_show):
    for i in range(N + 1):
        plt.figure()
        xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
        xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
        xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
        plt.plot(xsx, xsy, color='lightblue')
        plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
        xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
        xs_ic = my_close(xs_i)

        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')

        plt.show()

########################################################################################
# Plot final step with grid deformation
i = N
plt.figure()
xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)

plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')

plt.show()

######################################################################################
# Second experiment: semi parametric
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

name_exp = 'basi_semi_parametric'

Module = comb_mod.CompoundModules([Sil, Model0_t, Model0_m, Model0_s, Model1])
Module.GD.fill_cot_from_param([param_sil, param_0_t, param_0_m, param_0_s, param_1])

P0 = opti.fill_Vector_from_GD(Module.GD)

######################################################################################
# Optimization process
# ~~~~~~~~~~~~~~~~~~~~

args = (Module, xst, attach_fun, N, 1e-7)

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
                                  'maxiter': maxiter,
                                  'iprint': 1,
                                  'maxls': 25
                              })

P1 = res['x']
opti.fill_Mod_from_Vector(P1, Module)
Module_optimized = Module.copy_full()
Modules_list = shoot.shooting_traj(Module, N)

######################################################################################
# Results
# ~~~~~~~~~~~~~~~~~~~~
#
# A first visualization
# ~~~~~~~~~~~~~~~~~~~~~~
xst_c = my_close(xst)
xs_c = my_close(xs)
if (flag_show):
    for i in range(N + 1):
        plt.figure()
        xs_i = Modules_list[2 * i].GD.GD_list[0].GD
        xs_ic = my_close(xs_i)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        
        x0_l_i = Modules_list[2 * i].GD.GD_list[1].GD
        plt.plot(x0_l_i[:, 0], x0_l_i[:, 1], '+r', linewidth=2)
        
        x0_m_i = Modules_list[2 * i].GD.GD_list[2].GD
        plt.plot(x0_m_i[:, 0], x0_m_i[:, 1], 'or', linewidth=2)
        
        x0_s_i = Modules_list[2 * i].GD.GD_list[3].GD
        plt.plot(x0_s_i[:, 0], x0_s_i[:, 1], 'or', linewidth=2)
        
        x1_i = Modules_list[2 * i].GD.GD_list[4].GD[0]
        plt.plot(x1_i[:, 0], x1_i[:, 1], 'og', linewidth=2)
        
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        plt.show()

#########################################################################################
# Plot the deformation grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Create the grid ...

hxgrid = 9
hsl = 1.2 * height_source / 2
a, b, c, d = (Dx - hsl / 2, Dx + hsl / 2, Dy, Dy + 2 * hsl)
hygrid = np.round(hxgrid * (d - c) / (b - a))
nxgrid, nygrid = (2 * hxgrid + 1, 2 * hygrid + 1)  # create a grid for visualisation purpose
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))

(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

#########################################################################################
# ... add it as a silent module to flow it
Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized])

#######################################################################################
# ... and perform the shooting

Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)

########################################################################################
# Plot with grid to show the deformation

xs_c = my_close(xs)
xst_c = my_close(xst)
if (flag_show):
    for i in range(N + 1):
        plt.figure()
        xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
        xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
        xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
        plt.plot(xsx, xsy, color='lightblue')
        plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
        xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
        xs_ic = my_close(xs_i)

        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')

        plt.show()

########################################################################################
# Plot final step with grid deformation
i = N
plt.figure()
xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)

plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')

plt.show()

#########################################################################################
# Follow deformation generated by chosen module
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#########################################################################################

#########################################################################################
# Follow Model_0_l (large translation) and Model_1 (constrained deformation)
indi_modules = [0, 1, 4]

# Store controls
Contlist = []
for i in range(len(Modules_list)):
    # First 0 for the SilentGrid module
    Contlist.append([0, [Modules_list[i].Cont[j] for j in indi_modules]])
Mod_cont_init = comb_mod.CompoundModules([Modlist_opti_tot_grid[0].ModList[0].copy_full(), comb_mod.CompoundModules(
    [Modules_list[0].ModList[j].copy_full() for j in indi_modules])])

# Integrate the corresponding deformation
Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, N)

# Plot final step 
xst_c = my_close(xst)
xs_c = my_close(xs)
i = N
plt.figure()
xgrid = Modlist_cont[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')

xs_i = Modlist_cont[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')

#########################################################################################
# Follow Model_0_m and Model_0_s (variation with respect to the model)
indi_modules = [0, 2, 3]

# Store controls
Contlist = []
for i in range(len(Modules_list)):
    # First 0 for the SilentGrid module
    Contlist.append([0, [Modules_list[i].Cont[j] for j in indi_modules]])
Mod_cont_init = comb_mod.CompoundModules([Modlist_opti_tot_grid[0].ModList[0].copy_full(), comb_mod.CompoundModules(
    [Modules_list[0].ModList[j].copy_full() for j in indi_modules])])

# Integrate the corresponding deformation
Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, N)

# Plot final step 
xst_c = my_close(xst)
xs_c = my_close(xs)
i = N
plt.figure()
xgrid = Modlist_cont[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')

xs_i = Modlist_cont[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')
