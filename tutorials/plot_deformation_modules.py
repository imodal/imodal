"""
Deformation module
==================

In this tutorial we will see how to create and use deformation modules.
"""

###############################################################################
# We first need to import
#

import sys
sys.path.append("../")

import math

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm


###############################################################################
# First, we create some synthetic data.
# Lets displace a line using some local translation deformation module.

nb_points_line = 50
line = torch.stack([torch.linspace(-1., 1., nb_points_line),
                    torch.zeros(nb_points_line)], axis=1)

nb_points_translation = 2
translation_points = torch.tensor([[-0.9, -0.1], [0.9, 0.1]])
mom_translation = torch.tensor([[0., -0.5], [0., 0.5]])

###############################################################################
# plot the synthetic data

plt.plot(line[:, 0], line[:, 1], color='blue')
plt.plot(translation_points[:, 0], translation_points[:, 1], 'x', color='black')
plt.quiver(translation_points[:, 0], translation_points[:, 1],
           mom_translation[:, 0], mom_translation[:, 1])
plt.axis('equal')
plt.show()

###############################################################################
# We now create the silent module representing the points that will get
# transported and the local translation module.

silent = dm.DeformationModules.SilentLandmarks(
    2, nb_points_line, gd=line.clone())

silent.manifold.fill_cotan_zeros(False)

translation = dm.DeformationModules.Translations(
    2, nb_points_translation, 0.3,
    gd=translation_points, cotan=mom_translation)

###############################################################################
# Shooting

solver = 'rk4'
it = 5

intermediates = {}
dm.HamiltonianDynamic.shoot(
    dm.HamiltonianDynamic.Hamiltonian([silent, translation]),
    solver, it, intermediates=intermediates)

###############################################################################
# Plotting the result

intermediate_states = intermediates['states']

aabb = dm.Utilities.AABB(-1.1, 1.1, 1., 1.)
plt.rcParams['figure.figsize'] = (4, it*4)
for i, state in zip(range(it), intermediate_states):
    deformed_line = state[0].gd.detach()
    deformed_translation = state[1].gd.detach()

    plt.subplot(it, 1, i+1)
    plt.title("t = {t}".format(t=i*0.25))
    plt.plot(deformed_line[:, 0], deformed_line[:, 1], color='blue')
    plt.plot(deformed_translation[:, 0], deformed_translation[:, 1], 'x', color='black')
    plt.axis('equal')

plt.show()

