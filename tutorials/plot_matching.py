"""
Simple matching
===============

In this tutorial we will introduce matching of shapes.

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
# First, we need to generate our source and target.
# We will match a sphere onto a square.

nb_points_source = 50
radius = 1.
source = radius*dm.Utilities.generate_unit_circle(nb_points_source)[:-1]

nb_points_square_side = 4
target = dm.Utilities.generate_unit_square(nb_points_square_side)
target = dm.Utilities.linear_transform(target, dm.Utilities.rot2d(math.pi/18.))

nb_points_target = target.shape[0]
nb_points_source = source.shape[0]


###############################################################################
# Plotting

plt.subplot(1, 2, 1)
plt.title("Source")
plt.plot(source[:, 0], source[:, 1], '-')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Target")
plt.plot(target[:, 0], target[:, 1], '-')
plt.axis('equal')

plt.show()


###############################################################################
# Deformation modules

sigma_translation = 0.1
translation = dm.DeformationModules.Translations(2, nb_points_source, sigma_translation, gd=source.clone().requires_grad_(), backend='torch')


###############################################################################
# Creating the model

sigma_varifold = 0.5
model = dm.Models.ModelPointsRegistration([source.clone()], [translation], [dm.Attachment.VarifoldAttachment(2, [sigma_varifold])], lam=100.)


###############################################################################
# Fitting
#

fitter = dm.Models.ModelFittingScipy(model)

fitter.fit([target.clone()], 15, log_interval=1)


###############################################################################
# Computing deformed source

modules = dm.DeformationModules.CompoundModule(model.modules)
modules.manifold.fill(model.init_manifold)

dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian(modules), 10, 'euler')

deformed = modules[0].manifold.gd.detach()


###############################################################################
# Displaying result

plt.plot(source[:, 0], source[:, 1], '--', color='grey')
plt.plot(target[:, 0], target[:, 1], '-', color='black')
plt.plot(deformed[:, 0], deformed[:, 1], '-', color='red')
plt.axis('equal')
plt.show()


