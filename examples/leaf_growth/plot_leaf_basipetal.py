"""
Basipetal growth model of a leaf
================================

In this example we will model basipetal growth of a leaf.

Basipetal growth is characterised by a development of the leaf from the apex
downward toward the basis.

For this, we use an implicit module of order 1 with growth constants following
some quadratic law.

"""

###############################################################################
# We first import what we need.
#

import sys

sys.path.append("../../")

import math
import pickle
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm

###############################################################################
# We load the data, rescale it and zero it.
#

data_source = pickle.load(open("../../data/leafbasi.pkl", 'rb'))
data_target = pickle.load(open("../../data/leafbasit.pkl", 'rb'))

height_source = 38.
height_target = 100.

source = torch.tensor(data_source[1]).type(torch.get_default_dtype())
target = torch.tensor(data_target[1]).type(torch.get_default_dtype())

smin, smax = torch.min(source[:, 1]), torch.max(source[:, 1])
sscale = height_source / (smax - smin)
source[:, 1] = - sscale * (source[:, 1] - smax)
source[:, 0] = sscale * (source[:, 0] - torch.mean(source[:, 0]))

tmin, tmax = torch.min(target[:, 1]), torch.max(target[:, 1])
tscale = height_target / (tmax - tmin)
target[:, 1] = - tscale * (target[:, 1] - tmax)
target[:, 0] = tscale * (target[:, 0] - torch.mean(target[:, 0]))

# Extract shape informations from the dataset
shape_source = source[source[:, 2] == 2, 0:2]
shape_target = target[target[:, 2] == 2, 0:2]

###############################################################################
# We now sample the points that will be used by the deformation modules.
#

# Points for our contour
points_small = shape_source.clone()

# Build AABB around the source shape and uniformly sample points for the implicit
# module of order 1
aabb_source = dm.Utilities.AABB.build_from_points(shape_source)
points_growth = dm.Utilities.fill_area_uniform_density(dm.Utilities.area_shape, aabb_source, 0.25, shape=shape_source)

rot_growth = torch.stack([dm.Utilities.rot2d(0.)]*points_growth.shape[0], axis=0)

###############################################################################
# Plot everything.
#

plt.subplot(1, 2, 1)
plt.title("Source leaf")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_small[:, 0].numpy(), points_small[:, 1].numpy(), 'x', color='red')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Target leaf")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='black')
plt.axis('equal')

plt.show()


###############################################################################
# Define and plot the cubic model of the growth constants.
#

C = torch.zeros(points_growth.shape[0], 2, 1)
K, L = 10, height_source
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = (K * (a * (L - points_growth[:, 1]) ** 3  + b * (L - points_growth[:, 1]) ** 2))
C[:, 0, 0] = 1. * C[:, 1, 0]

ax = plt.subplot()
dm.Utilities.plot_C_ellipses(ax, points_growth, C, color='green', scale=0.2)
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.axis('equal')
plt.show()


###############################################################################
# We now build the deformation modules that will be used by the model.
#

nu = 0.01
coeff_small = 10.
coeff_growth = 0.01
scale_small = 3.
scale_growth = 15.

global_translation = dm.DeformationModules.GlobalTranslation(2)

small_scale_translation = dm.DeformationModules.ImplicitModule0(
    2, points_small.shape[0], scale_small, coeff=coeff_small, nu=0.01,
    gd=points_small)

growth = dm.DeformationModules.ImplicitModule1(
    2, points_growth.shape[0], scale_growth, C, coeff=coeff_growth, nu=nu,
    gd=(points_growth, rot_growth))


###############################################################################
# We now define the model.
#

deformable_shape_source = dm.Models.DeformablePoints(shape_source)
deformable_shape_target = dm.Models.DeformablePoints(shape_target)

model = dm.Models.RegistrationModel([deformable_shape_source],
            [global_translation, small_scale_translation, growth],
            [dm.Attachment.VarifoldAttachment(2, [20., 60.])], lam=100.)


###############################################################################
# Fitting.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = dm.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target], 500, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Plot results. Matching is very good.
#

intermediates = {}
model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
intermediate_states = intermediates['states']

deformed_source = intermediate_states[-1][0].gd
deformed_small = intermediate_states[-1][2].gd
deformed_growth = intermediate_states[-1][3].gd[0]

aabb_target = dm.Utilities.AABB.build_from_points(shape_target).scale(1.2)

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_small[:, 0].numpy(), points_small[:, 1].numpy(), 'x')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.plot(deformed_small[:, 0], deformed_small[:, 1], 'x')
plt.plot(deformed_growth[:, 0], deformed_growth[:, 1], '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.axis(aabb_target.totuple())

plt.show()


###############################################################################
# We now compute the influence on the deformation of the small scale module
# and of the growth module.
#
# We first need to compute the controls of each modules.
#

modules = dm.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone(), copy=True)

intermediates = {}
dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian(modules), shoot_solver, shoot_it, intermediates=intermediates)

intermediate_states = intermediates['states']
intermediate_controls = intermediates['controls']

ss_controls = [control[2] for control in intermediate_controls]
growth_controls = [control[3] for control in intermediate_controls]

###############################################################################
# We know compute the deformation grid of the small scale module.
#


# We extract the modules of the models and fill the right manifolds.
modules = dm.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone(), copy=True)
silent = copy.copy(modules[0])
deformation_grid = dm.DeformationModules.DeformationGrid(dm.Utilities.AABB.build_from_points(shape_source).scale(1.2), [32, 32])
small_scale = copy.copy(modules[2])

# We construct the controls list we will give will shooting
controls = [[torch.tensor([]), torch.tensor([]), control] for control in ss_controls]

dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian([silent, deformation_grid, small_scale]), shoot_solver, shoot_it, controls=controls)

ss_deformed_source = silent.manifold.gd.detach()
ss_deformed_grid = deformation_grid.togrid()


###############################################################################
# We know compute the deformation grid of the growth module.
#

# We extract the modules of the models and fill the right manifolds.
modules = dm.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone(), copy=True)
silent = copy.copy(modules[0])
deformation_grid = dm.DeformationModules.DeformationGrid(dm.Utilities.AABB.build_from_points(shape_source).scale(1.2), [32, 32])
growth = copy.copy(modules[3])

# We construct the controls list we will give will shooting
controls = [[torch.tensor([]), torch.tensor([]), control] for control in growth_controls]

dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian([silent, deformation_grid, growth]), shoot_solver, shoot_it, controls=controls)

growth_deformed_source = silent.manifold.gd.detach()
growth_deformed_grid = deformation_grid.togrid()

###############################################################################
# We now plot both results. We see that most of the deformation comes from the
# growth module, as expected.
#

ax = plt.subplot(1, 2, 1)
plt.title("Growth module")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(growth_deformed_source[:, 0].numpy(), growth_deformed_source[:, 1].numpy())
dm.Utilities.plot_grid(ax, growth_deformed_grid[0], growth_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

ax = plt.subplot(1, 2, 2)
plt.title("Small scale module")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(ss_deformed_source[:, 0].numpy(), ss_deformed_source[:, 1].numpy())
dm.Utilities.plot_grid(ax, ss_deformed_grid[0], ss_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

plt.show()

