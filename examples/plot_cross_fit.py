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

source_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/cross_+.png", origin='lower')
target_image = dm.Utilities.load_greyscale_image("/home/leander/diffeo/implicitmodules/data/images/cross_x.png", origin='lower')

extent_length = 1.
extent = dm.Utilities.AABB(0., extent_length, 0., extent_length)

dots = torch.tensor([[0., 0.5],
                     [0.5, 0.],
                     [0., -0.5],
                     [-0.5, 0.]])

source_dots = 0.8*extent_length*dots + extent_length*torch.tensor([0.5, 0.5])

target_dots = 0.8*extent_length*dm.Utilities.linear_transform(dots, dm.Utilities.rot2d(math.pi/4)) + extent_length*torch.tensor([0.5, 0.5])

plt.subplot(1, 2, 1)
plt.title("Source image")
plt.imshow(source_image, origin='lower', extent=extent.totuple())
plt.plot(source_dots.numpy()[:, 0], source_dots.numpy()[:, 1], '.')

plt.subplot(1, 2, 2)
plt.title("Target image")
plt.imshow(target_image, origin='lower', extent=extent.totuple())
plt.plot(target_dots.numpy()[:, 0], target_dots.numpy()[:, 1], '.')

plt.show()

###############################################################################
# We know that the target cross is the result of some rotation at its origin,
# so we use a local rotation deformation module, with an imprecise center
# position to simulate data aquisition noise.
#
# Since we will optimize the rotation center, we flag it as requiring gradient
# computations using `requires_grad_()`.
#

center = torch.tensor([[0.4, 0.5]])

rotation = dm.DeformationModules.LocalRotation(2, 1.2, gd=center.clone().requires_grad_())


###############################################################################
# We create the model. We set `True` for `fit_gd` so that it also optimize the
# rotation center.
#

source_deformable = dm.Models.DeformableImage(source_image, output='points',
                                              extent=None)
target_deformable = dm.Models.DeformableImage(target_image, output='points', 
                                              extent=None)

source_dots_deformable = dm.Models.DeformablePoints(source_dots)
target_dots_deformable = dm.Models.DeformablePoints(target_dots)

model = dm.Models.RegistrationModel([source_deformable, source_dots_deformable], [rotation], [dm.Attachment.GeomlossAttachment(blur=0.01, scaling=0.9), dm.Attachment.EuclideanPointwiseDistanceAttachment()], fit_gd=[True], lam=100.)


###############################################################################
# We fit the model.
#

shoot_solver='rk4'
shoot_it = 10
max_it = 4

costs = {}
fitter = dm.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([target_deformable, target_dots_deformable], max_it, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Plot total cost evolution
#

total_costs = [sum(cost) for cost in list(map(list, zip(*costs.values())))]

plt.title("Total cost evolution")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True)
plt.plot(range(len(total_costs)), total_costs, color='black', lw=0.7)
plt.show()


###############################################################################
# we compute the deformed source and plot it.
#

with torch.autograd.no_grad():
    model.deformables[0].output = 'bitmap'
    deformed = model.compute_deformed(shoot_solver, shoot_it)

    deformed_image = deformed[0][0]
    deformed_dots = deformed[1][0]

fitted_center = model.init_manifold[2].gd.detach()

print("Fitted rotatation center: {center}".format(center=fitted_center.detach().tolist()))

plt.subplot(1, 3, 1)
plt.title("Source image")
plt.imshow(source_image.numpy(), origin='lower', extent=extent.totuple())
plt.plot(source_dots.numpy()[:, 0], source_dots.numpy()[:, 1], '.')
plt.plot(center.numpy()[0, 0], center.numpy()[0, 1], 'X')

plt.subplot(1, 3, 2)
plt.title("Fitted image")
plt.imshow(deformed_image.numpy(), origin='lower', extent=extent.totuple())
plt.plot(fitted_center.numpy()[0, 0], fitted_center.numpy()[0, 1], 'X')
plt.plot(deformed_dots.numpy()[:, 0], deformed_dots.numpy()[:, 1], '.')

plt.subplot(1, 3, 3)
plt.title("target image")
plt.imshow(target_image.numpy(), origin='lower', extent=extent.totuple())
plt.plot(target_dots.numpy()[:, 0], target_dots.numpy()[:, 1], '.')

plt.show()

