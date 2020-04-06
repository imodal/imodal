"""
Fitting some crosses
====================

In this example, we will fit a cross onto the same cross, but rotated. We will
take advantage of this knowledge and use a rotation deformation module. We will
also add some noise on the initial center guess to show how to fit the geometrical
descriptors.

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

source_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/cross_+.png")
target_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/cross_x.png")

plt.subplot(1, 2, 1)
plt.title("Source image")
plt.imshow(source_image)

plt.subplot(1, 2, 2)
plt.title("Target image")
plt.imshow(target_image)

plt.show()

###############################################################################
# We know that the target cross is the result of some rotation at its origin,
# so we use a local rotation deformation module, with an imprecise center
# position to simulate data aquisition noise.
#
# Since we will optimize the rotation center, we flag it as requiring gradient
# computations using `requires_grad_()`.
#

center = torch.tensor([[10., 10.]])

rotation = dm.DeformationModules.LocalRotation(2, 25., gd=center.clone().requires_grad_())


###############################################################################
# We create the model. We set `True` for `fit_gd` so that it also optimize the
# rotation center.
#

model = dm.Models.ModelImageRegistration(source_image, [rotation], dm.Attachment.EuclideanPointwiseDistanceAttachment(), fit_gd=[True], lam=100.)

###############################################################################
# We fit the model.
#

shoot_method='rk4'
shoot_it = 10

fitter = dm.Models.ModelFittingScipy(model)
costs = fitter.fit([target_image], 100, log_interval=10, options={'shoot_method': shoot_method, 'shoot_it': shoot_it})


###############################################################################
# We compute the deformed source and plot it.
#

with torch.autograd.no_grad():
    deformed_image = model.compute_deformed(shoot_method, shoot_it)[0]

fitted_center = model.init_manifold[1].gd.detach()

plt.subplot(1, 3, 1)
plt.title("Source image")
plt.imshow(source_image)
plt.plot(center.numpy()[0, 0], center.numpy()[0, 1], 'X')

plt.subplot(1, 3, 2)
plt.title("Fitted image")
plt.imshow(deformed_image)
plt.plot(fitted_center.numpy()[0, 0], fitted_center.numpy()[0, 1], 'X')

plt.subplot(1, 3, 3)
plt.title("target image")
plt.imshow(target_image)

plt.show()

