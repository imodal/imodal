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

source_deformable = dm.Models.DeformablePoints(source)
target_deformable = dm.Models.DeformablePoints(target)

sigma_varifold = [0.1, 0.5, 1.]
model = dm.Models.RegistrationModel(source_deformable, translation, [dm.Attachment.VarifoldAttachment(2, sigma_varifold)], lam=100.)


###############################################################################
# Fitting. Optimizer can be manually selected (if none is provided, a default optimizer will be choosen). Here, we select Pytorch's LBFGS algorithm with strong Wolfe termination conditions. We also show how optimization can also be easily stopped and resumed.
#

shoot_solver = 'euler'
shoot_it = 10
costs = {}
fitter = dm.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit(target_deformable, 10, costs=costs, options={'line_search_fn': 'strong_wolfe', 'shoot_solver': shoot_solver, 'shoot_it': shoot_it})


###############################################################################
# Plot total cost evolution
#

plt.title("Total cost evolution")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.plot(range(len(costs['total'])), costs['total'], color='black', lw=0.7)
plt.show()


###############################################################################
# Computing deformed source

with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it)[0]


###############################################################################
# Displaying result

plt.plot(source[:, 0], source[:, 1], '--', color='grey')
plt.plot(target[:, 0], target[:, 1], '-', color='black')
plt.plot(deformed[:, 0], deformed[:, 1], '-', color='red')
plt.axis('equal')
plt.show()


