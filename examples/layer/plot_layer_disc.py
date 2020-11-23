""" 
Layered growth model in 3D
==========================

Example of a disc shaped layered growth model in 3D.
"""

###############################################################################
# Import relevant modules.
#

import os
import sys
import copy
import math
import pickle
import time

sys.path.append("../../")

import plotly
import plotly.graph_objs as go
import plotly.io as pio
pio.orca.config.use_xvfb = True

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pymesh

import imodal

###############################################################################
# Generation of the disc and control points
# -----------------------------------------
#
# We first define the disc and growth module parameters.
#

# Parameters for the rigid and growth discs
radius_inner_rigid = 2.
radius_outer_rigid = 5.2
radius_inner_growth = 5.2
radius_outer_growth = 6.
thickness = 1.
density = 2.
angular_resolution_rigid = math.ceil(2.*math.pi*radius_outer_rigid*density)
angular_resolution_growth = math.ceil(2.*math.pi*radius_outer_growth*density)

# Parameters for the growth module
control_radius_inner = radius_inner_rigid
control_radius_outer = radius_outer_growth
control_thickness = 1.
control_density = 1.


###############################################################################
# Generate control points of the growth module.
#

# Lower disc
control_points_lower = imodal.Utilities.generate_disc_density(
    control_density, control_radius_outer, control_radius_inner)
control_points_lower = torch.cat(
    [control_points_lower, -control_thickness/2.*torch.ones(control_points_lower.shape[0]).unsqueeze(1)], dim=1)

# Middle disc
control_points_middle = imodal.Utilities.generate_disc_density(
    control_density, control_radius_outer, control_radius_inner)
control_points_middle = torch.cat(
    [control_points_middle, torch.zeros(control_points_middle.shape[0]).unsqueeze(1)], dim=1)

# Top disc
control_points_upper = imodal.Utilities.generate_disc_density(
    control_density, control_radius_outer, control_radius_inner)
control_points_upper = torch.cat(
    [control_points_upper, control_thickness/2.*torch.ones(control_points_upper.shape[0]).unsqueeze(1)], dim=1)

# Concatenate everything together
control_points = torch.cat([control_points_lower, control_points_middle, control_points_upper])

print("Control points count={count}".format(count=control_points.shape[0]))

###############################################################################
# Plot control points.
#

ax = plt.subplot(projection='3d')
plt.plot(control_points[:, 0].numpy(), control_points[:, 1].numpy(), control_points[:, 2].numpy(), '.')
imodal.Utilities.set_aspect_equal_3d(ax)
plt.show()

###############################################################################
# Generate disc shapes that will get deformed
#

# Generate initial meshes.
rigid_mesh = pymesh.generate_tube(
    [0., 0., -thickness/2.], [0., 0., thickness/2.],
    radius_inner_rigid, radius_inner_rigid,
    radius_outer_rigid, radius_outer_rigid,
    num_segments=angular_resolution_rigid)

growth_mesh = pymesh.generate_tube(
    [0., 0., -thickness/2.], [0., 0., thickness/2.],
    radius_inner_growth, radius_inner_growth,
    radius_outer_growth, radius_outer_growth,
    num_segments=angular_resolution_growth)

# Tessellate the mesh a bit more in order to get smoother deformations
rigid_mesh = pymesh.tetrahedralize(rigid_mesh, 1./density, radius_edge_ratio=3., engine='cgal')
growth_mesh = pymesh.tetrahedralize(growth_mesh, 1./density, engine='cgal')

rigid_points = torch.tensor(rigid_mesh.vertices, dtype=torch.get_default_dtype())
growth_points = torch.tensor(growth_mesh.vertices, dtype=torch.get_default_dtype())
rigid_faces = rigid_mesh.faces
growth_faces = growth_mesh.faces

print("Rigid points count={count}".format(count=len(rigid_mesh.vertices)))
print("Growth points count={count}".format(count=len(growth_mesh.vertices)))


###############################################################################
# Plot the disc that will get deformed.
#

ax = plt.subplot(projection='3d')
ax.plot_trisurf(growth_points[:, 0].numpy(), growth_points[:, 1].numpy(), growth_points[:, 2].numpy(), triangles=growth_faces, linewidth=0.2, zsort='max', color=(0., 0., 0., 0.), edgecolor=(0., 0., 1., 1))
ax.plot_trisurf(rigid_points[:, 0].numpy(), rigid_points[:, 1].numpy(), rigid_points[:, 2].numpy(), triangles=rigid_faces, linewidth=0.2, zsort='max', color=(0., 0., 0., 0.), edgecolor=(1., 0., 0., 1))
imodal.Utilities.set_aspect_equal_3d(ax)
plt.show()


###############################################################################
# Filling model parameters
# ------------------------
# Fill in initial moments, initial local frame matrix and growth constants.
#

k = 4
eps = 1e-2

def f_C(point):
    r = math.sqrt(point[0]**2 + point[1]**2)

    if r <= radius_outer_rigid - eps:
        return torch.zeros(3)

    theta = math.atan2(point[1], point[0])
    z = point[2]/control_thickness/2.

    tangential_growth = math.cos(k*theta)
    
    return torch.tensor([0., 1. + z*tangential_growth, 0.])

def f_R(point):
    theta = math.atan2(point[1], point[0])

    return torch.tensor([[math.cos(theta), -math.sin(theta), 0.],
                         [math.sin(theta), math.cos(theta), 0.],
                         [0., 0., 1.]])

A_mom = 3000.
def f_mom(point):
    r = math.sqrt(point[0]**2 + point[1]**2)
    theta = math.atan2(point[1], point[0])

    if r <= radius_inner_growth - eps:
        return torch.zeros(3)

    if abs(theta) >= 0.4:
        return torch.zeros(3)

    return A_mom*torch.tensor([0., 0., 1.])


C = torch.stack([f_C(point).view(-1, 1) for point in control_points], dim=0)
R = torch.stack([f_R(point) for point in control_points], dim=0)
moments = torch.stack([f_mom(point) for point in control_points], dim=0)
moments_R = torch.zeros_like(R)

###############################################################################
# Plot initial moments.
#

ax = plt.subplot(projection='3d')
plt.plot(control_points[:, 0].numpy(), control_points[:, 1].numpy(), control_points[:, 2].numpy(), '.') 
ax.quiver(control_points[:, 0].numpy(), control_points[:, 1].numpy(), control_points[:, 2].numpy(), moments[:, 0].numpy(), moments[:, 1].numpy(), moments[:, 2].numpy(), length=-5., normalize=True)
imodal.Utilities.set_aspect_equal_3d(ax)
plt.show()

###############################################################################
# Plot growth constants arrows for the lower disc.
#

off = control_points_lower.shape[0]
ax = plt.subplot()
imodal.Utilities.plot_C_arrows(ax, control_points_lower[:, 0:2], C[:off, 0:2, :], R=R[:off, 0:2, 0:2], color='xkcd:light blue')
plt.plot(control_points_lower[:, 0].numpy(), control_points_lower[:, 1].numpy(), '.')
plt.axis('equal')
plt.show()


###############################################################################
# Plot growth constants for the middle disc.
#

ax = plt.subplot()
imodal.Utilities.plot_C_arrows(ax, control_points_middle[:, 0:2], C[off:2*off, 0:2, :], R=R[off:2*off, 0:2, 0:2], color='xkcd:light blue')
plt.plot(control_points_lower[:, 0].numpy(), control_points_lower[:, 1].numpy(), '.')
plt.axis('equal')
plt.show()


###############################################################################
# Plot growth constants for the upper disc.
#

ax = plt.subplot()
imodal.Utilities.plot_C_arrows(ax, control_points_upper[:, 0:2], C[2*off:, 0:2, :], R=R[2*off:, 0:2, 0:2], color='xkcd:light blue')
plt.plot(control_points_lower[:, 0].numpy(), control_points_lower[:, 1].numpy(), '.')
plt.axis('equal')
plt.show()


###############################################################################
# Initialising model
# ------------------
#
# We create an implicit module of order 1 to generate the deformation and
# two silent modules.
#


sigma = 2.

growth = imodal.DeformationModules.ImplicitModule1(3, control_points.shape[0], sigma, C, nu=0.01, gd=(control_points.clone().requires_grad_(), R.clone().requires_grad_()), cotan=(moments.clone().requires_grad_(), moments_R.clone().requires_grad_()))

layer_rigid = imodal.DeformationModules.SilentLanimodalarks(3, rigid_points.shape[0], gd=rigid_points.clone().requires_grad_())

layer_growth = imodal.DeformationModules.SilentLanimodalarks(3, growth_points.shape[0], gd=growth_points.clone().requires_grad_())


###############################################################################
# Shooting
# --------
#

start = time.perf_counter()
with torch.autograd.no_grad():
    imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([growth, layer_rigid, layer_growth]), 'euler', 10)
print("Elapsed time={elapsed}".format(elapsed=time.perf_counter()-start))


###############################################################################
# Results
# -------
#

deformed_rigid_points = layer_rigid.manifold.gd.detach()
deformed_growth_points = layer_growth.manifold.gd.detach()


ax = plt.subplot(projection='3d')
ax.plot_trisurf(deformed_growth_points[:, 0].numpy(), deformed_growth_points[:, 1].numpy(), deformed_growth_points[:, 2].numpy(), triangles=growth_faces, linewidth=0.2, zsort='max', color=(0., 0., 0., 0.), edgecolor=(0., 0., 1., 1))
ax.plot_trisurf(deformed_rigid_points[:, 0].numpy(), deformed_rigid_points[:, 1].numpy(), deformed_rigid_points[:, 2].numpy(), triangles=rigid_faces, linewidth=0.2, zsort='max', color=(0., 0., 0., 0.), edgecolor=(1., 0., 0., 1))
imodal.Utilities.set_aspect_equal_3d(ax)
plt.show()



