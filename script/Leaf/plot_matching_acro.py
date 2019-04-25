"""
Acropetal growth
----------------

This example shows how to perform a registration using a geodesic shooting with the implicit modules. The method allows us to define a non-uniform cost to model the growth of a leaf starting from the upper part of the leaf.

To see another type of growth: :ref:`basipetal <sphx_glr__auto_examples_Leaf_plot_matching_basi.py>` or  :ref:`diffuse <sphx_glr__auto_examples_Leaf_plot_matching_diffuse.py>`
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
import implicitmodules.numpy.Forward.Shooting as shoot
import implicitmodules.numpy.Optimisation.ScipyOpti_attach as opti
from implicitmodules.numpy.Utilities import Rotation as rot
from implicitmodules.numpy.Utilities.Visualisation import my_close, my_plot

name_exp = 'acro'
flag_show = True

# common options
nu = 0.001
dim = 2
N=10
maxiter = 2
 
lam_var = 40.
sig_var = [50., 10.]

########################################################################################
# Let us define the data attachment term with a varifold like cost function.

def attach_fun(xsf, xst):
    (varcost0, dxvarcost0) = var.my_dxvar_cost(xsf, xst, sig_var[0])
    (varcost1, dxvarcost1) = var.my_dxvar_cost(xsf, xst, sig_var[1])
    costvar = varcost0 + varcost1
    dcostvar = dxvarcost0 + dxvarcost1
    return (lam_var * costvar, lam_var * dcostvar )
                   

coeffs =[0.01, 100, 0.01]

###################################################################################
# Load data
# ^^^^^^^^^
# The source shape is segmented from the following image
#
with open('./data/acro.pkl', 'rb') as f:

    img, lx = pickle.load(f)
plt.imshow(img)
plt.show()

Dx = 0.
Dy = 0.
height_source = 90.
height_target = 495.

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = height_source / (lmax - lmin)

nlx[:, 1] = Dy-scale * (nlx[:, 1] - lmax)
nlx[:, 0] = Dx+scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

##################################################################################
# The target shape is ... blah blah

with open('./data/acrot.pkl', 'rb') as f:
    img, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = height_target / (lmax - lmin)
nlxt[:, 1] = - scale * (nlxt[:, 1] - lmax) 
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0])) 

xst = nlxt[nlxt[:, 2] == 2, 0:2]


####################################################################################
# Modules definitions
# ^^^^^^^^^^^^^^^^^^^

####################################################################################
# Silent Module
# ~~~~~~~~~~~~~
# This module is the shape to be transported
#

xs = nlx[nlx[:, 2] == 2, 0:2]
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
ps = np.zeros(xs.shape)
param_sil = (xs, ps)
if(flag_show):
    my_plot(xs, title="Silent Module", col='*b')

#####################################################################################
# Modules of Order 0
# ~~~~~~~~~~~~~~~~~~
# The first module of order 0 corresponds to ...

sig0 = 10.
x0 = nlx[nlx[:, 2] == 1, 0:2]
Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[1], nu)
p0 = np.zeros(x0.shape)
param_0 = (x0, p0)

if(flag_show):
    my_plot(x0, title="Module order 0", col='or')

######################################################################################
# The second modules of order 0 ...
#

sig00 = 800.
x00 = np.array([[0., 0.]])
Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, coeffs[0], nu)
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

if(flag_show):
    my_plot(x00, title="Module order 00", col='+r')

#######################################################################################
# Modules of order 1
# ~~~~~~~~~~~~~~~~~~
# The module of order 1 ... blah blah

sig1 = 60.
x1 = nlx[nlx[:, 2] == 1, 0:2]
C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, height_source
a,b = 1/L, 3.
z = a*(x1[:,1]-Dy)

#######################################################################################
# The matrix C here stores the eigen values of 2-tensors ... blah blah
C[:,1,0] = K*((1-b)*z**2+b*z)
C[:,0,0] = 0.8*C[:,1,0]

Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)

th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

if(flag_show):
    my_plot(x1, ellipse=C ,title="Module order 1", col='og')

######################################################################################
# The full model definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# We gather here the modules defined above. Blah blah...

Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
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
# ^^^^^^^
#

xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xs_i = Modules_list[2 * i].GD.GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)

    x0_i = Modules_list[2 * i].GD.GD_list[1].GD
    plt.plot(x0_i[:, 0], x0_i[:, 1], '*r', linewidth=2)

    x00_i = Modules_list[2 * i].GD.GD_list[2].GD
    plt.plot(x00_i[:, 0], x00_i[:, 1], 'or', linewidth=2)

    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    if(flag_show):
        plt.show()

#########################################################################################
# Plot the deformation grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Create the grid ...

hxgrid = 9
hsl = 1.2*height_source/2
a, b, c, d = (Dx-hsl/2, Dx+hsl/2, Dy, Dy+2*hsl) 
hygrid = np.round(hxgrid*(d-c)/(b-a))
nxgrid, nygrid = (2*hxgrid+1, 2*hygrid+1) # create a grid for visualisation purpose
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))

(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

########################################################################################
# ... add it as a silent module to flow it ...

Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)
param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)
Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized])


#########################################################################################
# ... and perform the shooting.
Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)

#########################################################################################
# Plot with grid to show the deformation


xs_c = my_close(xs)
xst_c = my_close(xst)
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
    if(flag_show):
        plt.show()
