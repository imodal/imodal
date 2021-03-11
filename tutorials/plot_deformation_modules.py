"""
Deformation modules
===================

In this tutorial we will see how to create and use deformation modules.

Using a local translation module, we will displace a line represented by a silent module.
"""

###############################################################################
# Import relevant modules.
#

import sys
sys.path.append("../")

import torch
import matplotlib.pyplot as plt

import imodal


###############################################################################
# We create the synthetic data.
# First a line that will get displaced, then the position and moments of the
# local translation module.
#

nb_points_line = 50
line = torch.stack([torch.linspace(-1., 1., nb_points_line),
                    torch.zeros(nb_points_line)], axis=1)

nb_points_translation = 2
translation_points = torch.tensor([[-0.9, -0.1], [0.9, 0.1]])
mom_translation = torch.tensor([[0., -0.5], [0., 0.5]])


###############################################################################
# Plot the synthetic data.
#

plt.plot(line[:, 0], line[:, 1], color='blue')
plt.plot(translation_points[:, 0], translation_points[:, 1], 'x', color='black')
plt.quiver(translation_points[:, 0], translation_points[:, 1],
           mom_translation[:, 0], mom_translation[:, 1])
plt.axis('equal')
plt.show()


###############################################################################
# We now create the silent module representing the points that will get
# displaced.
#

silent = imodal.DeformationModules.SilentLandmarks(
    2, nb_points_line, gd=line.clone())

silent.manifold.fill_cotan_zeros(False)


###############################################################################
# We now create the local translation module that will deform the ambiant space.
#

translation = imodal.DeformationModules.Translations(
    2, nb_points_translation, 0.3,
    gd=translation_points, cotan=mom_translation)


###############################################################################
# Shooting. Solves the shooting ODE using 5 steps of RK4.
#

solver = 'rk4'
it = 5

intermediates = {}
imodal.HamiltonianDynamic.shoot(
    imodal.HamiltonianDynamic.Hamiltonian([silent, translation]),
    solver, it, intermediates=intermediates)


###############################################################################
# Plot each step of the deformation. We see how both points of the local
# translations transport its local ambiant space and thus the silent points.
#

intermediate_states = intermediates['states']

aabb = imodal.Utilities.AABB(-1.1, 1.1, 1., 1.)
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

