"""
Tree Growth using Implicit Modules
==================================

Image registration with an implicit module of order 1. Segmentations given by the data are used to initialize its points.

"""

###############################################################################
# Import relevant modules.
#


import time
import pickle
import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import torch

import imodal

# device = 'cuda:2'
device = 'cpu'
torch.set_default_dtype(torch.float64)

imodal.Utilities.set_compute_backend('keops')

###############################################################################
# Load source and target images, along with the source curve.
#

with open("../data/tree_growth.pickle", 'rb') as f:
    data = pickle.load(f)

source_shape = data['source_shape'].to(torch.get_default_dtype())
source_image = data['source_image'].to(torch.get_default_dtype())
target_image = data['target_image'].to(torch.get_default_dtype())

# Segmentations as Axis Aligned Bounding Boxes (AABB)
aabb_trunk = data['aabb_trunk']
aabb_crown = data['aabb_crown']
extent = data['extent']


###############################################################################
# Display source and target images, along with the segmented source curve (orange
# for the trunk, green for the crown).
#

shape_is_trunk = aabb_trunk.is_inside(source_shape)
shape_is_crown = aabb_crown.is_inside(source_shape)

plt.subplot(1, 2, 1)
plt.imshow(source_image, cmap='gray', origin='lower', extent=extent.totuple())
plt.plot(source_shape[shape_is_trunk, 0].numpy(), source_shape[shape_is_trunk, 1].numpy(), lw=2., color='orange')
plt.plot(source_shape[shape_is_crown, 0].numpy(), source_shape[shape_is_crown, 1].numpy(), lw=2., color='green')
plt.subplot(1, 2, 2)
plt.imshow(target_image, cmap='gray', origin='lower', extent=extent.totuple())
plt.show()


###############################################################################
# Generating implicit modules of order 1 points and growth factors
#

implicit1_density = 500.

# Lambda function defining the area in and around the tree shape
area = lambda x, **kwargs: imodal.Utilities.area_shape(x, **kwargs) | imodal.Utilities.area_polyline_outline(x, **kwargs)
polyline_width = 0.07

# Generation of the points
implicit1_points = imodal.Utilities.fill_area_uniform_density(area, imodal.Utilities.AABB(xmin=0., xmax=1., ymin=0., ymax=1.), implicit1_density, shape=source_shape, polyline=source_shape, width=polyline_width)

# Masks that flag points into either the trunk or the crown
implicit1_trunk_points = aabb_trunk.is_inside(implicit1_points)
implicit1_crown_points = aabb_crown.is_inside(implicit1_points)

implicit1_points = implicit1_points[implicit1_trunk_points | implicit1_crown_points]
implicit1_trunk_points = aabb_implicit1_trunk.is_inside(implicit1_points)
implicit1_crown_points = aabb_implicit1_crown.is_inside(implicit1_points)

assert implicit1_points[implicit1_trunk_points].shape[0] + implicit1_points[implicit1_crown_points].shape[0] == implicit1_points.shape[0]

# Initial normal frames
implicit1_r = torch.eye(2).repeat(implicit1_points.shape[0], 1, 1)

# Growth factor
implicit1_c = torch.zeros(implicit1_points.shape[0], 2, 4)

# Horizontal stretching for the trunk
implicit1_c[implicit1_trunk_points, 0, 0] = 1.
# Vertical stretching for the trunk
implicit1_c[implicit1_trunk_points, 1, 1] = 1.
# Horizontal stretching for the crown
implicit1_c[implicit1_crown_points, 0, 2] = 1.
# Vertical stretching for the crown
implicit1_c[implicit1_crown_points, 1, 3] = 1.


###############################################################################
# Display growth points.
#

plt.imshow(source_image, cmap='gray', origin='lower', extent=extent.totuple())
plt.plot(source_shape[:, 0].numpy(), source_shape[:, 1].numpy(), lw=2.)
plt.plot(implicit1_points[implicit1_trunk_points, 0].numpy(), implicit1_points[implicit1_trunk_points, 1], '.')
plt.plot(implicit1_points[implicit1_crown_points, 0].numpy(), implicit1_points[implicit1_crown_points, 1], '.')
plt.show()


###############################################################################
# Create the deformation model with a combination of 3 modules : implicit module
# of order 1 (growth model), implicit module of order 0 (small corrections) and
# a global translation.
#


###############################################################################
# Create and initialize the global translation module.
#

global_translation = imodal.DeformationModules.GlobalTranslation(2, coeff=1.)


###############################################################################
# Create and initialize the growth module.
#

sigma1 = 2./implicit1_density**(1/2)
implicit1 = imodal.DeformationModules.ImplicitModule1(2, implicit1_points.shape[0], sigma1, implicit1_c, nu=100., gd=(implicit1_points, implicit1_r), coeff=0.1)


###############################################################################
# Create and initialize local translations module.
#

sigma0 = 0.1
translations = imodal.DeformationModules.ImplicitModule0(2, source_shape.shape[0], sigma0, nu=0.1, gd=source_shape, coeff=500.)


###############################################################################
# Define deformables used by the registration model.
#

source_image_deformable = imodal.Models.DeformableImage(source_image.to(device=device), output='bitmap', extent=extent)

target_image_deformable = imodal.Models.DeformableImage(target_image.to(device=device), output='bitmap', extent=extent)


###############################################################################
# Move the deformation modules on the right device (e.g. GPU) if necessary.
#

source_image_deformable.silent_module.to_(device)
target_image_deformable.silent_module.to_(device)
global_translation.to_(device)
translations.to_(device)
implicit1.to_(device)
if str(device) is not 'cpu':
    translations._ImplicitModule0_KeOps__keops_backend = 'GPU'
    implicit1._ImplicitModule1_KeOps__keops_backend = 'GPU'


###############################################################################
# Define the registration model.
#

attachment_image = imodal.Attachment.L2NormAttachment(weight=1e0)

model = imodal.Models.RegistrationModel([source_image_deformable], [implicit1, global_translation], [attachment_image], lam=1.)


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([target_image_deformable], 200, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe', 'history_size': 200})


###############################################################################
# Compute optimized deformation trajectory.
#

intermediates = {}
start = time.perf_counter()
with torch.autograd.no_grad():
    deformed_image = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)[0][0].detach()
print("Elapsed={elapsed}".format(elapsed=time.perf_counter()-start))
pts = model.init_manifold[1].gd[0].detach()


###############################################################################
# Display deformed source image against the target.
#

plt.subplot(1, 3, 1)
plt.imshow(source_image, extent=extent.totuple(), origin='lower')

plt.subplot(1, 3, 2)
plt.imshow(deformed_image.cpu(), extent=extent.totuple(), origin='lower')

plt.subplot(1, 3, 3)
plt.imshow(target_image, extent=extent.totuple(), origin='lower')
plt.show()

