"""
Atlas of peanuts
================

In this example we will show how to build an atlas of a population and to
optimise model parameters.

The dataset consists of peanuts that haven't been built using diffeomorphisms.
"""

###############################################################################
# Import relevant modules.
#

import sys
import copy
import math
import pickle
import time

sys.path.append("../")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pymesh

import implicitmodules.torch as dm

###############################################################################
# Load the dataset, extract the template peanut and the target peanuts.
#


data = pickle.load(open("../data/peanuts.pickle", 'rb'))

peanuts_count = 4
#template = torch.tensor(data[0][0][:-1], dtype=torch.get_default_dtype())
peanuts = [torch.tensor(peanut[:-1], dtype=torch.get_default_dtype()) for peanut in data[0][1:peanuts_count+1]]

template = dm.Utilities.generate_unit_circle(200)
template = dm.Utilities.linear_transform(template, torch.tensor([[1.3, 0.], [0., 0.5]]))

point_left_scale = torch.tensor([[-1., 0.]])
point_right_scale = torch.tensor([[1., 0.]])


###############################################################################
# Plot everything.
#

plt.plot(template[:, 0].numpy(), template[:, 1].numpy(), '--', color='xkcd:blue')
plt.plot(point_left_scale[0, 0].numpy(), point_left_scale[0, 1].numpy(), 'x', color='xkcd:blue')
plt.plot(point_right_scale[0, 0].numpy(), point_right_scale[0, 1].numpy(), 'x', color='xkcd:blue')
for peanut in peanuts:
    plt.plot(peanut[:, 0].numpy(), peanut[:, 1].numpy(), lw=0.4, color='xkcd:light blue')

plt.axis('equal')
plt.show()


###############################################################################
# Initialise deformation modules. Two local scaling modules, one for each bulb
# and a global translation module.
#

sigma_scale = 3.
left_scale = dm.DeformationModules.LocalScaling(2, sigma_scale, gd=point_left_scale.clone().requires_grad_(), coeff=0.1)
right_scale = dm.DeformationModules.LocalScaling(2, sigma_scale, gd=point_right_scale.clone().requires_grad_(), coeff=0.1)

global_translation = dm.DeformationModules.GlobalTranslation(2)


###############################################################################
# Initialise the model.
#
# We set the `fit_gd` flags to `True` for the scaling modules in order to optimise
# their positions.
#

sigmas_varifold = [0.5, 2., 5., 0.1]
atlas = dm.Models.Atlas(template.clone(), [global_translation, left_scale, right_scale], [dm.Attachment.VarifoldAttachment(2, sigmas_varifold)], len(peanuts), lam=100., optimise_template=True, ht_sigma=0.5, ht_it=10, ht_coeff=1., ht_nu=0.5, fit_gd=[False, True, True])


###############################################################################
# Fit the atlas.
#

shoot_solver = 'euler'
shoot_it = 10

fitter = dm.Models.ModelFittingScipy(atlas)
fitter.fit(peanuts, 250, log_interval=1, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it})


###############################################################################
# Extract and plot optimised positions.
#

optimised_left = atlas.models[0].init_manifold[2].gd.detach().view(2)
optimised_right = atlas.models[0].init_manifold[3].gd.detach().view(2)
ht = atlas.compute_template()

print("Optimised left scaling position={pos}".format(pos=optimised_left.tolist()))
print("Optimised right scaling position={pos}".format(pos=optimised_right.tolist()))

plt.plot(template[:, 0].numpy(), template[:, 1].numpy(), '--')
plt.plot(ht[:, 0].numpy(), ht[:, 1].numpy())
plt.plot(optimised_left[0].numpy(), optimised_right[1].numpy(), 'x', color='xkcd:black')
plt.plot(optimised_right[0].numpy(), optimised_right[1].numpy(), 'x', color='xkcd:black')

plt.axis('equal')
plt.show()


###############################################################################
# Display the atlas.
#

with torch.autograd.no_grad():
    deformed_templates = atlas.compute_deformed(shoot_solver, shoot_it)

row_count = math.ceil(math.sqrt(len(peanuts)))

for i, deformed, peanut in zip(range(len(peanuts)), deformed_templates, peanuts):
    plt.subplot(row_count, row_count, 1 + i)
    plt.plot(deformed[:, 0].numpy(), deformed[:, 1].numpy())
    plt.plot(peanut[:, 0].numpy(), peanut[:, 1].numpy())
    plt.axis('equal')

plt.show()



