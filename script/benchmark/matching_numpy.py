"""
Acropetal growth
----------------

This script compare the performance of the numpy and pytorch code
"""

import pickle
################################################################################
# Setup
# ^^^^^
import time

import numpy as np

import implicitmodules.numpy.DataAttachment.Varifold as var
import implicitmodules.numpy.DeformationModules.Combination as comb_mod
import implicitmodules.numpy.DeformationModules.ElasticOrder0 as defmod0
import implicitmodules.numpy.DeformationModules.ElasticOrder1 as defmod1
import implicitmodules.numpy.DeformationModules.SilentLandmark as defmodsil
import implicitmodules.numpy.Optimisation.ScipyOpti_attach as opti
from implicitmodules.numpy.Utilities import Rotation as rot

name_exp = 'acro'

# common options
nu = 0.001
dim = 2
N = 10
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
    return (lam_var * costvar, lam_var * dcostvar)


coeffs = [0.01, 100, 0.01]

###################################################################################
# Load data
# ^^^^^^^^^
# The source shape is segmented from the following image
#
with open('../Leaf/data/acro.pkl', 'rb') as f:
    img, lx = pickle.load(f)

Dx = 0.
Dy = 0.
height_source = 90.
height_target = 495.

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = height_source / (lmax - lmin)

nlx[:, 1] = Dy - scale * (nlx[:, 1] - lmax)
nlx[:, 0] = Dx + scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

##################################################################################
# The target shape is ... blah blah

with open('../Leaf/data/acrot.pkl', 'rb') as f:
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

#####################################################################################
# Modules of Order 0
# ~~~~~~~~~~~~~~~~~~
# The first module of order 0 corresponds to ...

sig0 = 10.
x0 = nlx[nlx[:, 2] == 1, 0:2]
Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[1], nu)
p0 = np.zeros(x0.shape)
param_0 = (x0, p0)

######################################################################################
# The second modules of order 0 ...
#

sig00 = 800.
x00 = np.array([[0., 0.]])
Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, coeffs[0], nu)
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

#######################################################################################
# Modules of order 1
# ~~~~~~~~~~~~~~~~~~
# The module of order 1 ... blah blah

sig1 = 60.
x1 = nlx[nlx[:, 2] == 1, 0:2]
C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, height_source
a, b = 1 / L, 3.
z = a * (x1[:, 1] - Dy)

#######################################################################################
# The matrix C here stores the eigen values of 2-tensors ... blah blah
C[:, 1, 0] = K * ((1 - b) * z ** 2 + b * z)
C[:, 0, 0] = 0.8 * C[:, 1, 0]

Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)

th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

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

t_0 = time.perf_counter()
opti.fun(P0, *args)
elapsed = time.perf_counter() - t_0
print("Energy evaluations: {:3.6f}s".format(elapsed))

t_0 = time.perf_counter()
opti.jac(P0, *args)
elapsed = time.perf_counter() - t_0
print("Gradient evaluations: {:3.6f}s".format(elapsed))
