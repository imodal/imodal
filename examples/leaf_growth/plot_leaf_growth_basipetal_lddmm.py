"""
Basipetal Leaf Growth Model using LDDMM
=======================================

"""

###############################################################################
# Python module import.
#

import sys
sys.path.append("../")
import math
import copy
import pickle

import torch

import matplotlib.pyplot as plt

import imodal

###############################################################################
# We load the data (shape of the source and target leaves).
#
with open("data/basipetal.pickle", 'rb') as f:
    data = pickle.load(f)

shape_source = torch.tensor(data['shape_source']).type(torch.get_default_dtype())
shape_target = torch.tensor(data['shape_target']).type(torch.get_default_dtype())

aabb_source = imodal.Utilities.AABB.build_from_points(shape_target)
aabb_target = imodal.Utilities.AABB.build_from_points(shape_target)


###############################################################################
# Plot source and target.
#

plt.subplot(1, 2, 1)
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='blue')
plt.axis(aabb_target.squared().totuple())
plt.subplot(1, 2, 2)
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='blue')
plt.axis(aabb_target.squared().totuple())
plt.show()


###############################################################################
# We now sample the points that will be used by the implicit deformation
# module of order 0 (LDDMM module).
#

# Build AABB (Axis Aligned Bounding Box) around the source shape and uniformly
# sample points for the growth module.
points_density = 0.1

points_lddmm = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_shape, aabb_source.scale(1.3), points_density, shape=1.3*shape_source)


###############################################################################
# Plot points of the LDDMM module.
#

plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_lddmm[:, 0].numpy(), points_lddmm[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')
plt.show()


###############################################################################
# Create the deformation model which only consists of one implicit module of
# order 0.
#


###############################################################################
# Create and initialize local translations module.
#

nu = 0.1
scale_lddmm = 5./points_density**(1/2)
lddmm = imodal.DeformationModules.ImplicitModule0(2, points_lddmm.shape[0], scale_lddmm, nu=nu, gd=points_lddmm)


###############################################################################
# Define deformables used by the registration model.
#
deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)


###############################################################################
# Define the registration model.
#

model = imodal.Models.RegistrationModel(
    [deformable_shape_source],
    [lddmm],
    [imodal.Attachment.VarifoldAttachment(2, [20, 120.])],
    lam=10.)


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target], 100, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Compute optimized deformation trajectory.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
    deformed_shape = deformed[0][0]

lddmm_controls = [control[1] for control in intermediates['controls']]


###############################################################################
# Plot results.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.axis(aabb_target.totuple())
plt.axis('equal')
plt.show()


###############################################################################
# Recompute the learned deformation trajectory this time with the grid
# deformation to visualize growth.
#

# We extract the modules of the models and fill the right manifolds.
modules = imodal.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone())
silent_shape = copy.copy(modules[0])
lddmm = copy.copy(modules[1])

# Define the deformation grid.
square_size = 2.5
lddmm_grid_resolution = [math.floor(aabb_source.width/square_size),
                         math.floor(aabb_source.height/square_size)]
deformation_grid = imodal.DeformationModules.DeformationGrid(aabb_source, lddmm_grid_resolution)

# We construct the controls we will give while shooting.
controls = [[torch.tensor([]), torch.tensor([]), lddmm_control] for lddmm_control in lddmm_controls]

# Reshoot.
intermediates_lddmm = {}
with torch.autograd.no_grad():
    imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([silent_shape, deformation_grid, lddmm]), shoot_solver, shoot_it, controls=controls, intermediates=intermediates_lddmm)

# Store final deformation.
shoot_deformed_shape = silent_shape.manifold.gd.detach()
shoot_deformed_grid = deformation_grid.togrid()


###############################################################################
# Plot the deformation grid.
#

ax = plt.subplot()
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '.-', color='red')
plt.plot(shoot_deformed_shape[:, 0].numpy(), shoot_deformed_shape[:, 1].numpy())
imodal.Utilities.plot_grid(ax, shoot_deformed_grid[0], shoot_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

plt.show()

