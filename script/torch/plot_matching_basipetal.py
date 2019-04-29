import math
import pickle
# The deformation module library is not automatically installed yet, we need to add its path manually
import sys

sys.path.append("../../")

import numpy as np
import matplotlib.pyplot as plt
import torch

import implicitmodules.torch as im

torch.set_default_tensor_type(torch.DoubleTensor)

##################################################################################
# Data
# ----

# WIP
# On va merge les deux .pkl en un seul dictionnary
data_source = pickle.load(open("../Leaf/data/basi1b.pkl", 'rb'))
data_target = pickle.load(open("../Leaf/data/basi1t.pkl", 'rb'))

Dx = 0.
Dy = 0.
height_source = 38.
height_target = 100.

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
pos_source = torch.tensor(np.delete(pos_source.numpy(), 3, axis=0))
pos_implicit0 = source[source[:, 2] == 1, 0:2]
pos_implicit1 = source[source[:, 2] == 1, 0:2]
pos_target = target[target[:, 2] == 2, 0:2]

aabb = im.usefulfunctions.AABB.build_from_points(pos_target)
aabb.squared()

##################################################################################
#  Plots

fig, axs = plt.subplots(2, 2)

axs[0, 0].set_aspect('equal')
axs[0, 0].axis(aabb.get_list())
axs[0, 0].set_title('Source')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')
axs[0, 0].plot(pos_source[:, 0].numpy(), pos_source[:, 1].numpy(), '-')
axs[0, 0].plot(pos_implicit1[:, 0].numpy(), pos_implicit1[:, 1].numpy(), '.')
axs[0, 0].plot(pos_implicit0[:, 0].numpy(), pos_implicit0[:, 1].numpy(), 'x')

axs[0, 1].set_aspect('equal')
axs[0, 1].axis(aabb.get_list())
axs[0, 1].set_title('Target')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')
axs[0, 1].plot(pos_target[:, 0].numpy(), pos_target[:, 1].numpy(), '-')

axs[1, 0].imshow(data_source[0])

axs[1, 1].imshow(data_target[0])

plt.show()

#############################################################################
# Setting up the modules
# ----------------------

# Local translation module
sigma0 = 15.
nu0 = 0.001
coeff0 = 100.
implicit0 = im.implicitmodules.ImplicitModule0(
    im.manifold.Landmarks(2, pos_implicit0.shape[0], gd=pos_implicit0.view(-1).requires_grad_()), sigma0, nu0, coeff0)

# Global translation module
sigma00 = 400.
nu00 = 0.001
coeff00 = 0.01
implicit00 = im.implicitmodules.ImplicitModule0(
    im.manifold.Landmarks(2, 1, gd=torch.tensor([0., 0.], requires_grad=True)), sigma00, nu00, coeff00)

# Elastic modules
sigma1 = 30.
nu1 = 0.001
coeff1 = 0.01
C = torch.zeros(pos_implicit1.shape[0], 2, 1)
K, L = 10, height_source
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = (K * (a * (L - pos_implicit1[:, 1] + Dy) ** 3 + b * (L - pos_implicit1[:, 1] + Dy) ** 2))
C[:, 0, 0] = 1. * C[:, 1, 0]
th = 0. * math.pi * torch.ones(pos_implicit1.shape[0])
R = torch.stack([im.usefulfunctions.rot2d(t) for t in th])

implicit1 = im.implicitmodules.ImplicitModule1(im.manifold.Stiefel(2, pos_implicit1.shape[0], gd=(
pos_implicit1.view(-1).requires_grad_(), R.view(-1).requires_grad_())), C, sigma1, nu1, coeff1)

#############################################################################
# Setting up the model and start the fitting loop

model = im.models.ModelCompoundWithPointsRegistration(
    (pos_source, torch.ones(pos_source.shape[0])),
    [implicit0, implicit00, implicit1],
    [True, True, True],
    im.attachement.VarifoldAttachement([10., 50.])
)

costs = model.fit((pos_target, torch.ones(pos_target.shape[0])), max_iter=40, l=1., lr=5e-1, log_interval=1)

#############################################################################
# Results
# -------

fig, axs = plt.subplots(1, 3)

axs[0].set_aspect('equal')
axs[0].axis(aabb.get_list())
axs[0].plot(pos_source[:, 0].numpy(), pos_source[:, 1].numpy(), '-')
axs[0].plot(pos_implicit1[:, 0].numpy(), pos_implicit1[:, 1].numpy(), '.')
axs[0].plot(pos_implicit0[:, 0].numpy(), pos_implicit0[:, 1].numpy(), 'x')

axs[1].set_aspect('equal')
axs[1].axis(aabb.get_list())
out = model.shot_manifold[0].gd.view(-1, 2).detach().numpy()
shot_implicit0 = model.shot_manifold[1].gd.view(-1, 2).detach().numpy()
shot_implicit00 = model.shot_manifold[2].gd.view(-1, 2).detach().numpy()
shot_implicit1 = model.shot_manifold[3].gd[0].view(-1, 2).detach().numpy()
axs[1].plot(out[:, 0], out[:, 1], '-')
axs[1].plot(shot_implicit0[:, 0], shot_implicit0[:, 1], 'x')
axs[1].plot(shot_implicit00[:, 0], shot_implicit00[:, 1], 'o')
axs[1].plot(shot_implicit1[:, 0], shot_implicit1[:, 1], '.')

axs[2].set_aspect('equal')
axs[2].axis(aabb.get_list())
axs[2].plot(pos_target[:, 0].numpy(), pos_target[:, 1].numpy(), '-')
axs[2].plot(out[:, 0], out[:, 1], '-')

fig.tight_layout()

plt.show()

#############################################################################
# Evolution of the cost with iterations
plt.title("Cost")
plt.xlabel("Iteration(s)")
plt.ylabel("Cost")
plt.plot(range(len(costs)), costs)
plt.show()
