"""
Implicit deformation modules
============================

In this tutorial we will see how to use implicit modules of order 1 and how
growth constants influence the deformation pattern.
"""

###############################################################################
# We first need to import the good stuff.
#

import sys
sys.path.append("../")

import math

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm


###############################################################################
# We now define some helper functions.
#
# First our deformation routine.
# 

def deform_rod(positions, moments, sigma, C, method, it):
    rot = torch.stack([dm.Utilities.rot2d(0.)]*positions.shape[0], axis=0)

    rod = dm.DeformationModules.ImplicitModule1(2, positions.shape[0], sigma, C, 0.01, gd=(positions, rot), cotan=(moments, torch.zeros_like(rot)))

    deformation_grid = dm.DeformationModules.DeformationGrid(
        dm.Utilities.AABB(-5., 5., -5., 5.),
        [32, 32])

    dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian([rod, deformation_grid]), it, method)

    return rod.manifold.gd[0].detach(), deformation_grid.togrid()


###############################################################################
# Generates the rod we will deform.
# 

def generate_rod(size, n, f_mom, f_C):
    aabb = dm.Utilities.AABB(-size[0]/2., size[0]/2., -size[1]/2., size[1]/2.)
    x, y = dm.Utilities.generate_mesh_grid(aabb, n)
    positions = dm.Utilities.grid2vec(x, y)

    moments = torch.empty(n[0]*n[1], 2)
    moments[:, 0] = torch.cat([f_mom[0](position).view(1) for position in positions])
    moments[:, 1] = torch.cat([f_mom[1](position).view(1) for position in positions])

    C = torch.empty(n[0]*n[1], 2, 1)
    C[:, 0, 0] = torch.cat([f_C[0](position).view(1) for position in positions])
    C[:, 1, 0] = torch.cat([f_C[1](position).view(1) for position in positions])

    return positions, moments, C


###############################################################################
# Plot the rod, moments and growth constants.
# 

def plot_rod(positions, moments, C):
    plt.subplot(1, 2, 1)
    plt.quiver(positions[:, 0].numpy(), positions[:, 1].numpy(),
               moments[:, 0].numpy(), moments[:, 1].numpy())
    plt.axis('equal')

    ax = plt.subplot(1, 2, 2)
    dm.Utilities.plot_C_arrow(ax, positions, C, color='black', scale=0.3, mutation_scale=8.)
    plt.axis('equal')

    plt.show()


###############################################################################
# Define our rod size and scale
# 

size = [1., 5.]
n = [5, 25]
sigma = 1.


###############################################################################
# First experiment. Constant area growth can be achieved by having sum of growth
# constants equal zero.
#

pos, mom, C = generate_rod(size, n,
                           [lambda x: torch.zeros(1), lambda x: -0.02*x[1]**3],
                           [lambda x: -torch.ones(1), lambda x: torch.ones(1)])

print("Surface before compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(pos).area))

plot_rod(pos, mom, C)


###############################################################################
# Compress the rod from each extremity inwards. We see that the surface stays
# roughly constant by flattening itselft.
#

deformed, grid = deform_rod(pos, mom, sigma, C, 'rk4', 10)

print("Surface after compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(deformed).area))

ax = plt.subplot()
plt.plot(deformed[:, 0].numpy(), deformed[:, 1].numpy(), '.')
dm.Utilities.plot_grid(ax, grid[0], grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')
plt.show()

###############################################################################
# Second experiment. Same growth constants.
#

pos, mom, C = generate_rod(size, n,
                           [lambda x: torch.zeros(1), lambda x: 0.02*x[1]**3],
                           [lambda x: -torch.ones(1), lambda x: torch.ones(1)])

print("Surface before compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(pos).area))

plot_rod(pos, mom, C)


###############################################################################
# Now compress the rod from each extremity outwards. 0. We see that the area
# stays roughly constant by elongating itself.
#

deformed, grid = deform_rod(pos, mom, sigma, C, 'rk4', 10)

print("Surface after compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(deformed).area))

ax = plt.subplot()
plt.plot(deformed[:, 0].numpy(), deformed[:, 1].numpy(), '.')
dm.Utilities.plot_grid(ax, grid[0], grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')
plt.show()


###############################################################################
# Third experiment. We now achieve bending by setting growth constants as a
# linear function of the abscisse. We compress the rod inwards, but only on the
# positive abscisse.
# 

def step(x):
    if x >= 0.:
        return 1.
    else:
        return 0.

pos, mom, C = generate_rod(size, n,
                           [lambda x: torch.zeros(1),
                            lambda x: -0.05*x[1]**5*step(x[0])],
                           [lambda x: torch.zeros(1), lambda x: x[0]])

print("Surface before compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(pos).area))

plot_rod(pos, mom, C)


###############################################################################
# Bending !
#

deformed, grid = deform_rod(pos, mom, sigma, C, 'rk4', 10)

print("Surface after compression: {surface}".format(
    surface=dm.Utilities.AABB.build_from_points(deformed).area))

ax = plt.subplot()
plt.plot(deformed[:, 0].numpy(), deformed[:, 1].numpy(), '.')
dm.Utilities.plot_grid(ax, grid[0], grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')
plt.show()


