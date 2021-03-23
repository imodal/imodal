"""
Building more Complex Deformation Modules
=========================================

In this tutorial, we present the construction of the main deformation modules and display examples of vector fields they can generate.
"""

###############################################################################
# Initialization
# --------------
#
# Import relevant Python modules.
#

import sys
sys.path.append("../")

import torch
import matplotlib.pyplot as plt

import imodal

torch.manual_seed(1337)


###############################################################################
# We define a grid on which we compute the generated vector fields.
#

grid_points = imodal.Utilities.grid2vec(*torch.meshgrid([torch.linspace(-2., 2., 20), torch.linspace(-2., 2., 20)]))


###############################################################################
# Sum of Local Translations
# -------------------------
# We define the parameters, the scale of gaussian kernel **sigma**, **d** the dimension of the ambiant space and **p** the number of local translations (points in the geometrical descriptor).
#

sigma = 0.5
d = 2
p = 5


###############################################################################
# There are two ways to build a sum of local translations module, either with **Translations** (explicit formulation) or with **ImplicitModule0** (implicit formulation).
# The implicit formulation is a regularised version of the explicit one and depends on a regularization parameter **nu**.
#

explicit_translation = imodal.DeformationModules.Translations(d, p, sigma)

nu = 0.1
implicit_translation = imodal.DeformationModules.ImplicitModule0(d, p, sigma, nu=nu)


###############################################################################
# We choose the geometrical descriptor i.e. the centers **gd** carrying the local translations.
# Then, we choose the controls **controls** i.e. the translation vectors.
#

gd = 0.8*torch.randn(p, d)
controls = torch.rand(p, d) - 0.5

explicit_translation.manifold.fill_gd(gd)
implicit_translation.manifold.fill_gd(gd)

explicit_translation.fill_controls(controls)
implicit_translation.fill_controls(controls)


###############################################################################
# We compute and display the generated vector fields
#

explicit_field = explicit_translation(grid_points)
implicit_field = implicit_translation(grid_points)

plt.figure(figsize=[8., 4.])
plt.subplot(1, 2, 1)
plt.title("Explicit")
plt.quiver(grid_points[:, 0], grid_points[:, 1], explicit_field[:, 0], explicit_field[:, 1])
plt.plot(gd[:, 0], gd[:, 1], 'x', color='blue')
plt.quiver(gd[:, 0], gd[:, 1], controls[:, 0], controls[:, 1], scale=5., color='red', lw=1.5)
plt.axis('equal')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Implicit")
plt.quiver(grid_points[:, 0], grid_points[:, 1], implicit_field[:, 0], implicit_field[:, 1])
plt.plot(gd[:, 0], gd[:, 1], 'x', color='blue')
plt.quiver(gd[:, 0], gd[:, 1], controls[:, 0], controls[:, 1], scale=5., color='red', lw=1.5)
plt.axis('equal')
plt.axis('off')

plt.show()


###############################################################################
# Local Constrained Translations
# ------------------------------
#
#

###############################################################################
#
#


local_constrained = imodal.DeformationModules.LocalConstrainedTranslations(d, p, sigma)
local_scaling = imodal.DeformationModules.LocalScaling(d, sigma)
local_rotation = imodal.DeformationModules.LocalRotation(d, sigma)


###############################################################################
# We choose the geometrical descriptor i.e. the centers **gd** carrying the local translations.
# Then, we choose the controls **controls** i.e. the translation vectors.
#

gd = 0.8*torch.randn(p, d)
controls = torch.rand(p, d) - 0.5

explicit_translation.manifold.fill_gd(gd)
implicit_translation.manifold.fill_gd(gd)

explicit_translation.fill_controls(controls)
implicit_translation.fill_controls(controls)


###############################################################################
# We compute and display the generated vector fields
#

explicit_field = explicit_translation(grid_points)
implicit_field = implicit_translation(grid_points)

plt.figure(figsize=[8., 4.])
plt.subplot(1, 2, 1)
plt.title("Explicit")
plt.quiver(grid_points[:, 0], grid_points[:, 1], explicit_field[:, 0], explicit_field[:, 1])
plt.plot(gd[:, 0], gd[:, 1], 'x', color='blue')
plt.quiver(gd[:, 0], gd[:, 1], controls[:, 0], controls[:, 1], scale=5., color='red', lw=1.5)
plt.axis('equal')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Implicit")
plt.quiver(grid_points[:, 0], grid_points[:, 1], implicit_field[:, 0], implicit_field[:, 1])
plt.plot(gd[:, 0], gd[:, 1], 'x', color='blue')
plt.quiver(gd[:, 0], gd[:, 1], controls[:, 0], controls[:, 1], scale=5., color='red', lw=1.5)
plt.axis('equal')
plt.axis('off')

plt.show()


###############################################################################
# Implicit Deformation Module of Order 1
# --------------------------------------
#
#

###############################################################################
# Compound Deformation Module
# ---------------------------
#
#


