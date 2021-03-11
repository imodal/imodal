"""
Multi-scales LDDMM deformation model on images
==============================================

In this example we will explore multi-scales LDDMM deformation using the IMODAL framework.

"""

###############################################################################
# Import relevant Python modules.
#

import copy
import sys
sys.path.append("../")

import torch
import matplotlib.pyplot as plt

import imodal

imodal.Utilities.set_compute_backend('keops')

# device = 'cpu'
device = 'cuda:2'

###############################################################################
# Load source and target images.
#

source_image = imodal.Utilities.interpolate_image(imodal.Utilities.load_greyscale_image("../data/images/key_a.png", origin='lower'), scale_factor=0.25)
target_image = imodal.Utilities.interpolate_image(imodal.Utilities.load_greyscale_image("../data/images/key_b.png", origin='lower'), scale_factor=0.25)

aabb = imodal.Utilities.AABB(0, source_image.shape[1]-1, 0, source_image.shape[0]-1)


###############################################################################
# Plot both images.
#

plt.subplot(1, 2, 1)
plt.title("Source image")
plt.imshow(source_image, origin='lower')

plt.subplot(1, 2, 2)
plt.title("Target image")
plt.imshow(target_image, origin='lower')

plt.show()


###############################################################################
# Define our model that takes into input the multiple scales.
#

def model(scales, it=1):
    scale_points = [aabb.fill_uniform_density(scale) for scale in scales]
    scale_sigmas = [1./scale for scale in scales]
    scale_translations = [imodal.DeformationModules.ImplicitModule0(2, scale_point.shape[0], scale_sigma, nu=0.1, gd=scale_point) for scale_point, scale_sigma in zip(scale_points, scale_sigmas)]

    source_deformable = imodal.Models.DeformableImage(copy.deepcopy(source_image), extent='match')
    target_deformable = imodal.Models.DeformableImage(copy.deepcopy(target_image), extent='match')

    source_deformable.to_device(device)
    target_deformable.to_device(device)

    model = imodal.Models.RegistrationModel(source_deformable, scale_translations, imodal.Attachment.EuclideanPointwiseDistanceAttachment(), lam=100.)

    model.to_device(device)

    shoot_solver = 'euler'
    shoot_it = 10

    costs = {}
    fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
    fitter.fit(target_deformable, it, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})

    with torch.autograd.no_grad():
        deformed_image = model.compute_deformed(shoot_solver, shoot_it)[0][0]

    return scale_points, scale_sigmas, deformed_image.cpu()


###############################################################################
# Plot function to show points placement given the scale.
#

def plot_points(points, sigmas):
    for point, sigma, i in zip(points, sigmas, range(len(sigmas))):
        plt.subplot(1, len(sigmas), i+1)
        plt.imshow(source_image, cmap='gray', origin='lower')
        plt.title("$\sigma={}$".format(sigma))
        plt.plot(point[:, 0].numpy(), point[:, 1].numpy(), '.')

    plt.show()


###############################################################################
# Small scale
# -----------
#
# Try to register using a small scale
#

scales = [0.02]

scale_points, scale_sigmas, deformed_image = model(scales, 20)


###############################################################################
# Plot of the control points.
#

plot_points(scale_points, scale_sigmas)


###############################################################################
# Resulting deformed image.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.imshow(source_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 2)
plt.title("Fitted")
plt.imshow(deformed_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 3)
plt.title("Target")
plt.imshow(target_image, cmap='gray', origin='lower')
plt.show()


###############################################################################
# Large scale
# -----------
#
# Try to register using a large scale
#

scales = [0.2]

scale_points, scale_sigmas, deformed_image = model(scales, 20)


###############################################################################
# Plot of the control points.
#

plot_points(scale_points, scale_sigmas)


###############################################################################
# Resulting deformed image.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.imshow(source_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 2)
plt.title("Fitted")
plt.imshow(deformed_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 3)
plt.title("Target")
plt.imshow(target_image, cmap='gray', origin='lower')
plt.show()


###############################################################################
# Multi-scales
# ------------
#
# Try to register using both small and large scales.
#

scales = [0.2, 0.02]

scale_points, scale_sigmas, deformed_image = model(scales, 20)


###############################################################################
# Plot of the control points.
#

plot_points(scale_points, scale_sigmas)


###############################################################################
# Resulting deformed image.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.imshow(source_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 2)
plt.title("Fitted")
plt.imshow(deformed_image, cmap='gray', origin='lower')
plt.subplot(1, 3, 3)
plt.title("Target")
plt.imshow(target_image, cmap='gray', origin='lower')
plt.show()

