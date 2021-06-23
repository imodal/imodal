"""
Unstructured Basipetal Leaf Growth Model
========================================

"""

###############################################################################
# Import relevant Python modules.
#

import sys
sys.path.append("../../")
import math
import copy
import pickle

import torch

import matplotlib.pyplot as plt

import imodal

imodal.Utilities.set_compute_backend('torch')
torch.set_default_dtype(torch.float32)

###############################################################################
# We load the data
#

with open("../../data/basipetal.pickle", 'rb') as f:
    data = pickle.load(f)

shape_source = torch.tensor(data['shape_source']).type(torch.get_default_dtype())
shape_target = torch.tensor(data['shape_target']).type(torch.get_default_dtype())
dots_source = torch.tensor(data['dots_source'], dtype=torch.get_default_dtype())
dots_target = torch.tensor(data['dots_target'], dtype=torch.get_default_dtype())


aabb_source = imodal.Utilities.AABB.build_from_points(shape_source)
aabb_target = imodal.Utilities.AABB.build_from_points(shape_target)


###############################################################################
# Unstructured growth using the curve information
# -----------------------------------------------
#
# Intitialization
# ^^^^^^^^^^^^^^^
#
# Plot source and target.
#

plt.title("Source and target")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='red')

plt.axis('equal')
plt.show()


###############################################################################
# We now sample the points that will be used by the local translations module for unstructured growth.
#

# Build AABB (Axis Aligned Bounding Box) around the source shape and uniformly
# sample points for the growth module.


points_density = 0.05

points_translations = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_shape, aabb_source.scale(1.3), points_density, shape=1.3*shape_source)


###############################################################################
# Plot points of the local translations module.
#

plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_translations[:, 0].numpy(), points_translations[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')
plt.show()


###############################################################################
# Create the deformation model which only consists of one Translations deformation module.
#


###############################################################################
# Create and initialize local translations module.
#


sigma = 3./points_density**(1/2)
translations = imodal.DeformationModules.Translations(2, points_translations.shape[0], sigma, gd=points_translations)


###############################################################################
# Define deformables used by the registration model.
#
deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)


###############################################################################
# Registration
# ^^^^^^^^^^^^
#
# Define the registration model.
#

model = imodal.Models.RegistrationModel(
    [deformable_shape_source],
    [translations],
    [imodal.Attachment.VarifoldAttachment(2, [20., 120.])],
    lam=10.)


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target], 2, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Results visualization
# ^^^^^^^^^^^^^^^^^^^^^
#
# Compute optimized deformation trajectory.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
    deformed_shape = deformed[0][0]

translations_controls = [control[1] for control in intermediates['controls']]


###############################################################################
# Plot results.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-', color='black')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-', color='blue')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-', color='red')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-', color='blue')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.tight_layout()
plt.show()


###############################################################################
# Recompute the learned deformation trajectory this time with the grid
# deformation to visualize growth.
#

# Reset the local translations module with the learned initialization manifold.
translations.manifold.fill(model.init_manifold[1])

aabb_source.scale_(1.2)
# Define the deformation grid.
square_size = 1.
grid_resolution = [math.floor(aabb_source.width/square_size),
                   math.floor(aabb_source.height/square_size)]

deformable_source = imodal.Models.DeformablePoints(shape_source)
deformable_grid = imodal.Models.DeformableGrid(aabb_source, grid_resolution)
deformable_source.silent_module.manifold.fill_cotan(model.init_manifold[0].cotan)

controls = [[control[1]] for control in intermediates['controls']]

# Shoot.
intermediates = {}
with torch.autograd.no_grad():
    imodal.Models.deformables_compute_deformed([deformable_source, deformable_grid], [translations], shoot_solver, shoot_it, controls=controls, intermediates=intermediates)


###############################################################################
# Plot the growth trajectory.
#
indices = [0, 3, 7, 10]

plt.figure(figsize=[10.*len(indices), 10.])
for i, index in enumerate(indices):
    state = intermediates['states'][index]
    ax = plt.subplot(1, len(indices), i + 1)
    deformable_grid.silent_module.manifold.fill_gd(state[1].gd)
    grid_x, grid_y = deformable_grid.silent_module.togrid()
    imodal.Utilities.plot_grid(ax, grid_x, grid_y, color='xkcd:light blue', lw=0.4)

    plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
    plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='red')
    plt.plot(state[0].gd[:, 0].numpy(), state[0].gd[:, 1].numpy())

    plt.axis('equal')
    plt.axis('off')

plt.tight_layout()
plt.show()


###############################################################################
# Unstructured growth model using curve and dots informations
# -----------------------------------------------------------
#
# Initialization
# ^^^^^^^^^^^^^^
#
# Plot source and target.
#

plt.title("Source and target")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='black')
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='red')
plt.plot(dots_target[:, 0].numpy(), dots_target[:, 1].numpy(), '.', color='red')

plt.axis('equal')
plt.show()



###############################################################################
# We now sample the points that will be used by the local translations module.
#

# Build AABB (Axis Aligned Bounding Box) around the source shape and uniformly
# sample points for the growth module.
points_density = 0.05

aabb_source = imodal.Utilities.AABB.build_from_points(shape_source)

points_translations = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_shape, aabb_source.scale(1.3), points_density, shape=1.3*shape_source)


###############################################################################
# Plot geometrical descriptor of the local translations module.
#

plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_translations[:, 0].numpy(), points_translations[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')
plt.show()


###############################################################################
# Create and initialize local translations module.
#


sigma = 3./points_density**(1/2)
translations = imodal.DeformationModules.Translations(2, points_translations.shape[0], sigma, gd=points_translations)


###############################################################################
# Define deformables used by the registration model.
#

deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)
deformable_dots_source = imodal.Models.DeformablePoints(dots_source)
deformable_dots_target = imodal.Models.DeformablePoints(dots_target)


###############################################################################
# Registration
# ^^^^^^^^^^^^
#
# Define the registration model.
#

model = imodal.Models.RegistrationModel(
    [deformable_shape_source, deformable_dots_source],
    [translations],
    [imodal.Attachment.VarifoldAttachment(2, [20, 120.]),
     imodal.Attachment.EuclideanPointwiseDistanceAttachment(10.)],
    lam=10.)


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target, deformable_dots_target], 50, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Result visualization
# ^^^^^^^^^^^^^^^^^^^^
#
# Compute optimized deformation trajectory.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
    deformed_shape = deformed[0][0]
    deformed_dots = deformed[1][0]


###############################################################################
# Plot results.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-', color='black')
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='black')
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-', color='blue')
plt.plot(deformed_dots[:, 0], deformed_dots[:, 1], '.', color='blue')
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-', color='red')
plt.plot(dots_target[:, 0].numpy(), dots_target[:, 1].numpy(), '.', color='red')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-', color='blue')
plt.plot(deformed_dots[:, 0], deformed_dots[:, 1], '.', color='blue')
plt.axis('equal')

plt.tight_layout()
plt.show()


###############################################################################
# Recompute the learned deformation trajectory this time with the grid
# deformation to visualize growth.
#

# Reset the local translations module with the learned initialization manifold.
translations.manifold.fill(model.init_manifold[2])

aabb_source.scale_(1.2)
# Define the deformation grid.
square_size = 1.
grid_resolution = [math.floor(aabb_source.width/square_size),
                   math.floor(aabb_source.height/square_size)]

deformable_source = imodal.Models.DeformablePoints(shape_source)
deformable_dots_source = imodal.Models.DeformablePoints(dots_source)
deformable_grid = imodal.Models.DeformableGrid(aabb_source, grid_resolution)

deformable_source.silent_module.manifold.fill_cotan(model.init_manifold[0].cotan)
deformable_dots_source.silent_module.manifold.fill_cotan(model.init_manifold[1].cotan)

controls = [[control[2]] for control in intermediates['controls']]

# Shoot.
intermediates = {}
with torch.autograd.no_grad():
    imodal.Models.deformables_compute_deformed([deformable_source, deformable_dots_source, deformable_grid], [translations], shoot_solver, shoot_it, controls=controls, intermediates=intermediates)


###############################################################################
# Plot the growth trajectory.
#

indices = [0, 3, 7, 10]

plt.figure(figsize=[10.*len(indices), 10.])
for i, index in enumerate(indices):
    state = intermediates['states'][index]
    ax = plt.subplot(1, len(indices), i + 1)

    deformable_grid.silent_module.manifold.fill_gd(state[2].gd)
    grid_x, grid_y = deformable_grid.silent_module.togrid()
    imodal.Utilities.plot_grid(ax, grid_x, grid_y, color='xkcd:light blue', lw=0.4)

    plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
    plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='red')
    plt.plot(state[0].gd[:, 0].numpy(), state[0].gd[:, 1].numpy(), color='blue')
    plt.plot(state[1].gd[:, 0].numpy(), state[1].gd[:, 1].numpy(), '.', color='blue')

    plt.axis('equal')
    plt.axis('off')

plt.tight_layout()
plt.show()


