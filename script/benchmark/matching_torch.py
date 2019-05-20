"""
Acropetal growth
----------------

This script compare the performance of the numpy and pytorch code
"""

################################################################################
# Setup
# ^^^^^

import pickle
import time

import torch

import implicitmodules.torch as im

###################################################################################
# Load data
# ^^^^^^^^^
# The source shape is segmented from the following image
#
# WIP
# On va merge les deux .pkl en un seul dictionnary
data_source = pickle.load(open("../Leaf/data/acro.pkl", 'rb'))
data_target = pickle.load(open("../Leaf/data/acrot.pkl", 'rb'))

Dx = 0.
Dy = 0.
height_source = 90.
height_target = 495.

source = torch.tensor(data_source[1]).type(torch.get_default_dtype())
target = torch.tensor(data_target[1]).type(torch.get_default_dtype())

smin, smax = torch.min(source[:, 1]), torch.max(source[:, 1])
sscale = height_source / (smax - smin)
source[:, 1] = Dy - sscale * (source[:, 1] - smax)
source[:, 0] = Dx + sscale * (source[:, 0] - torch.mean(source[:, 0]))

tmin, tmax = torch.min(target[:, 1]), torch.max(target[:, 1])
tscale = height_target / (tmax - tmin)
target[:, 1] = - tscale * (target[:, 1] - tmax)
target[:, 0] = tscale * (target[:, 0] - torch.mean(target[:, 0]))

pos_source = source[source[:, 2] == 2, 0:2]

pos_implicit1 = source[source[:, 2] == 1, 0:2]
pos_target = target[target[:, 2] == 2, 0:2]

##############################################################################
# Local translation module
sigma0 = 10.
nu0 = 0.001
coeff0 = 100.
pos_implicit0 = source[source[:, 2] == 1, 0:2]
implicit0 = im.implicitmodules.ImplicitModule0(
    im.Manifolds.Landmarks(2, pos_implicit0.shape[0], gd=pos_implicit0.view(-1).requires_grad_()), sigma0, nu0, coeff0)

###############################################################################
# Global translation module
sigma00 = 800.
nu00 = 0.001
coeff00 = 0.01
implicit00 = im.implicitmodules.ImplicitModule0(
    im.Manifolds.Landmarks(2, 1, gd=torch.tensor([0., 0.], requires_grad=True)), sigma00, nu00, coeff00)

###############################################################################
# Elastic modules

sigma1 = 30.
nu1 = 0.001
coeff1 = 0.01
C = torch.zeros(pos_implicit1.shape[0], 2, 1)
K, L = 10, height_source
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = (K * (a * (L - pos_implicit1[:, 1] + Dy) ** 3 + b * (L - pos_implicit1[:, 1] + Dy) ** 2))
C[:, 0, 0] = 1. * C[:, 1, 0]
th = 0. * torch.ones(pos_implicit1.shape[0])
R = torch.stack([im.usefulfunctions.rot2d(t) for t in th])

implicit1 = im.implicitmodules.ImplicitModule1(
    im.Manifolds.Stiefel(2, pos_implicit1.shape[0],
                         gd=(pos_implicit1.view(-1).requires_grad_(), R.view(-1).requires_grad_())),
    C,
    sigma1,
    nu1,
    coeff1
)

#############################################################################
# Model fit
# ^^^^^^^^^^
#
# Setting up the model and start the fitting loop
#

model = im.models.ModelCompoundWithPointsRegistration(
    (pos_source, torch.ones(pos_source.shape[0])),
    [implicit0, implicit00, implicit1],
    [True, True, True],
    im.attachement.VarifoldAttachement([10., 50.])
)

# costs = model.fit((pos_target, torch.ones(pos_target.shape[0])), max_iter=40, l=1., lr=5e-1, log_interval=1)
t_0 = time.perf_counter()
model.compute((pos_target, torch.ones(pos_target.shape[0])))
cost = model.attach + model.deformation_cost
elapsed = time.perf_counter() - t_0
print("Energy evaluations: {:3.6f}s".format(elapsed))

print(cost)

t_0 = time.perf_counter()
cost.backward(retain_graph=True)
elapsed = time.perf_counter() - t_0
print("Gradient evaluations: {:3.6f}s".format(elapsed))
