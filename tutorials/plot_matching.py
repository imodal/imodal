"""
Simple matching
===============

In this tutorial we will introduce matching of shapes. We will register a circle onto a square using LDDMM deformation.
"""

###############################################################################
# Import relevant modules.
#

import sys
sys.path.append("../")

import math

import torch
import matplotlib.pyplot as plt

import imodal


###############################################################################
# First, we need to generate our source (ciclre) and target (square).
#

nb_points_source = 50
radius = 1.
source = radius*imodal.Utilities.generate_unit_circle(nb_points_source)

nb_points_square_side = 4
target = imodal.Utilities.generate_unit_square(nb_points_square_side)
target = imodal.Utilities.linear_transform(target, imodal.Utilities.rot2d(math.pi/18.))

nb_points_target = target.shape[0]
nb_points_source = source.shape[0]


###############################################################################
# Plot of the source and target.
#

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
# Deformation module we use for our deformation model. We use a local
# translation module for LDDMM deformation.
#

sigma_translation = 0.1
translation = imodal.DeformationModules.Translations(2, nb_points_source, sigma_translation, gd=source)


###############################################################################
# Creating the model.

source_deformable = imodal.Models.DeformablePoints(source)
target_deformable = imodal.Models.DeformablePoints(target)

sigma_varifold = [0.5]
model = imodal.Models.RegistrationModel(source_deformable, translation, imodal.Attachment.VarifoldAttachment(2, sigma_varifold, backend='torch'), lam=100.)


###############################################################################
# Fitting. Optimizer can be manually selected (if none is provided, a default optimizer will be choosen). Here, we select Pytorch's LBFGS algorithm with strong Wolfe termination conditions.
#

shoot_solver = 'euler'
shoot_it = 10
costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit(target_deformable, 2, costs=costs, options={'line_search_fn': 'strong_wolfe', 'shoot_solver': shoot_solver, 'shoot_it': shoot_it})


###############################################################################
# Compute the final deformed source.
#

with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it)[0][0]


###############################################################################
# Display result. Matching is perfect.
#

plt.plot(source[:, 0], source[:, 1], '--', color='grey')
plt.plot(target[:, 0], target[:, 1], '-', color='black')
plt.plot(deformed[:, 0], deformed[:, 1], '-', color='red')
plt.axis('equal')
plt.show()


