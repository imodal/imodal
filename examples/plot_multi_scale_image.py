"""
Multi-scale deformation on images
=================================

"""

###############################################################################
# We first need to import
#

import sys
sys.path.append("../")

import math

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm

###############################################################################
# We load the data and plot them.
#

source_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/heart_a.png")
target_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/heart_b.png")

plt.subplot(1, 2, 1)
plt.title("Source image")
plt.imshow(source_image)

plt.subplot(1, 2, 2)
plt.title("Target image")
plt.imshow(target_image)

plt.show()


###############################################################################
# Multi scale local points generation
#

aabb = dm.Utilities.AABB(0, source_image.shape[0], 0, source_image.shape[1])
small_scale_density = 0.2
large_scale_density = 0.02

small_scale_points = aabb.fill_uniform_density(small_scale_density)
large_scale_points = aabb.fill_uniform_density(large_scale_density)

small_scale_sigma = 1.5/small_scale_density
large_scale_sigma = 1.5/large_scale_density

small_scale_translations = dm.DeformationModules.ImplicitModule0(2, small_scale_points.shape[0], small_scale_sigma, nu=0.1, gd=small_scale_points.clone().requires_grad_())
large_scale_translations = dm.DeformationModules.ImplicitModule0(2, large_scale_points.shape[0], large_scale_sigma, nu=0.1, gd=large_scale_points.clone().requires_grad_())

###############################################################################
# Plot translations points
#

plt.imshow(source_image)
plt.plot(small_scale_points[:, 0].numpy(), small_scale_points[:, 1].numpy(), '.')
plt.plot(large_scale_points[:, 0].numpy(), large_scale_points[:, 1].numpy(), '.')
plt.show()


###############################################################################
# Model
#

model = dm.Models.ModelImageRegistration(source_image, [small_scale_translations, large_scale_translations], dm.Attachment.EuclideanPointwiseDistanceAttachment(), lam=100.)


###############################################################################
# We fit the model.
#

shoot_solver='euler'
shoot_it = 10

costs = {}
fitter = dm.Models.Fitter(model, optimizer='scipy_l-bfgs-b')
fitter.fit(target_image.clone(), 500, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# We compute the deformed source and plot it.
#

with torch.autograd.no_grad():
    deformed_image = model.compute_deformed(shoot_solver, shoot_it)

fitted_center = model.init_manifold[1].gd.detach()

plt.subplot(1, 3, 1)
plt.title("Source image")
plt.imshow(source_image)

plt.subplot(1, 3, 2)
plt.title("Fitted image")
plt.imshow(deformed_image)

plt.subplot(1, 3, 3)
plt.title("target image")
plt.imshow(target_image)

plt.show()




