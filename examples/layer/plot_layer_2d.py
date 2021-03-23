"""
Layered growth model in 2D
==========================

Example of layered growth model using implicit modules.
"""

###############################################################################
# Import all relevant Python modules.
#

import sys
import math
import time

sys.path.append("../../")

import torch
import matplotlib.pyplot as plt

import imodal

###############################################################################
# Generate the layers.
#

layer_length = 40.
layer_growth_thickness = 2.
layer_rigid_thickness = 12.

aabb_growth = imodal.Utilities.AABB(-layer_length/2., layer_length/2.,
                                    0, layer_growth_thickness)
aabb_rigid = imodal.Utilities.AABB(-layer_length/2., layer_length/2.,
                                   -layer_rigid_thickness, 0.)

points_density = 0.5
points_layer_density = 2.

points_growth = aabb_growth.fill_uniform_density(points_density)
points_rigid = aabb_rigid.fill_uniform_density(points_density)

points_layer_growth = imodal.Utilities.generate_rectangle(imodal.Utilities.AABB.build_from_points(points_growth), points_layer_density)
points_layer_rigid = imodal.Utilities.generate_rectangle(imodal.Utilities.AABB.build_from_points(points_rigid), points_layer_density)


###############################################################################
# Plot everything.
#

plt.plot(points_layer_rigid[:, 0].numpy(), points_layer_rigid[:, 1].numpy(), color='xkcd:red')
plt.plot(points_layer_growth[:, 0].numpy(), points_layer_growth[:, 1].numpy(), color='xkcd:blue')
plt.plot(points_rigid[:, 0].numpy(), points_rigid[:, 1].numpy(), '.', color='xkcd:red')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.', color='xkcd:blue')
plt.axis('equal')

plt.show()


###############################################################################
# Generate points used by the growth module.
#
# Growth constants in the growing layer follow a sum of cosine law,
# parametrized by :math:`k` periods. For the rigid layer, growth constants
# are set to zero. For real models, growth constants would idealy be learned
# from data.
#
# For the initial moments, we will only pull outward the extreme points of the
# growing layer.
#

points = torch.cat([points_rigid, points_growth], dim=0)
R = torch.cat([imodal.Utilities.rot2d(0.).unsqueeze(0)]*points.shape[0])


# Define some math related helper functions.
def step(x):
    if x >= 0.:
        return 1.
    else:
        return 0.


def sign(x):
    if x >= 0.:
        return 1.
    else:
        return -1.


# Moment field generator.
def f_mom(point):
    return torch.tensor([sign(point[0])*step(abs(point[0]) - 19.)*step(point[1]-0.0), 0.])


A_mom = 3000.
moments = A_mom*torch.cat([f_mom(point) for point in points]).view(-1, 2)
moments_R = torch.zeros_like(R)


# Growth constants field generator.
def f_C(point, k):
    if point[1] < 0.:
        return torch.tensor([0., 0.])
    else:
        return torch.tensor([math.cos(2.*math.pi*point[0]/layer_length*k), 0])


k = 3 # Number of periods.
C = torch.cat([f_C(point, k) for point in points]).reshape(-1, 2, 1)


###############################################################################
# Plot initial moments.
#

plt.quiver(points[:, 0].numpy(), points[:, 1], moments[:, 0].numpy(), moments[:, 1].numpy(), scale=1.)
plt.axis('equal')
plt.show()


###############################################################################
# Plot growth constants.
#

ax = plt.subplot()
plt.plot(points_layer_rigid[:, 0].numpy(), points_layer_rigid[:, 1].numpy(), color='xkcd:red', lw=0.5)
plt.plot(points_layer_growth[:, 0].numpy(), points_layer_growth[:, 1].numpy(), color='xkcd:blue', lw=0.5)
imodal.Utilities.plot_C_arrows(ax, points, C, scale=2., color='blue', mutation_scale=2.)
plt.axis('equal')
plt.show()


###############################################################################
# Initialization of the deformation modules. First the growth
# module as an implicit deformation module and then two silent modules
# representing the both layers, for better visualisation.
#

sigma = 5.
growth = imodal.DeformationModules.ImplicitModule1(2, points.shape[0], sigma, C, nu=0.01, gd=(points, R), cotan=(moments, moments_R))

layer_rigid = imodal.DeformationModules.SilentLandmarks(2, points_layer_rigid.shape[0], gd=points_layer_rigid)

layer_growth = imodal.DeformationModules.SilentLandmarks(2, points_layer_growth.shape[0], gd=points_layer_growth)


###############################################################################
# Shooting.
#

start = time.perf_counter()
with torch.autograd.no_grad():
    imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([growth, layer_rigid, layer_growth]), 'midpoint', 10)
print("Elapsed time={elapsed}".format(elapsed=time.perf_counter()-start))

###############################################################################
# Extracting deformed points.
#

deformed_growth = growth.manifold.gd[0].detach()
deformed_layer_rigid = layer_rigid.manifold.gd.detach()
deformed_layer_growth = layer_growth.manifold.gd.detach()


###############################################################################
# Plot the result.
#
# The rigid layer opposes some resistance to the non rigid deformation.
#

plt.plot(deformed_growth[:, 0].numpy(), deformed_growth[:, 1].numpy(), '.', color='xkcd:blue')
plt.plot(deformed_layer_rigid[:, 0].numpy(), deformed_layer_rigid[:, 1].numpy(), '-', color='xkcd:red')
plt.plot(deformed_layer_growth[:, 0].numpy(), deformed_layer_growth[:, 1].numpy(), '-', color='xkcd:blue')
plt.axis('equal')

plt.show()

