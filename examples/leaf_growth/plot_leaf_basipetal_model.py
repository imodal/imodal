"""
Learning growth constants model
===============================

In this example we will learn a growth constants model from a leaf that follows a
basipetal growth pattern.

"""

###############################################################################
# We first import what we need.
#

import sys
sys.path.append("../../")

import math
import pickle
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm

###############################################################################
# We load the data, rescale it and zero it.
#
# In order to get more information, we also load the points.
#

data = pickle.load(open("../../data/basipetal.pkl", 'rb'))

height_source = 38.
height_target = 100.

dots_source = torch.tensor(data['source_d'], dtype=torch.get_default_dtype())
dots_target = torch.tensor(data['target_d'], dtype=torch.get_default_dtype())
shape_source = torch.tensor(data['source_c'], dtype=torch.get_default_dtype())
shape_target = torch.tensor(data['target_c'], dtype=torch.get_default_dtype())

smin, smax = torch.min(shape_source[:, 1]), torch.max(shape_source[:, 1])
sscale = height_source / (smax - smin)
dots_source[:, 1] = - sscale * (dots_source[:, 1] - smax)
dots_source[:, 0] = sscale * (dots_source[:, 0] - torch.mean(shape_source[:, 0]))
shape_source[:, 1] = - sscale * (shape_source[:, 1] - smax)
shape_source[:, 0] = sscale * (shape_source[:, 0] - torch.mean(shape_source[:, 0]))

tmin, tmax = torch.min(shape_target[:, 1]), torch.max(shape_target[:, 1])
tscale = height_target / (tmax - tmin)
dots_target[:, 1] = - tscale * (dots_target[:, 1] - tmax)
dots_target[:, 0] = tscale * (dots_target[:, 0] - torch.mean(shape_target[:, 0]))
shape_target[:, 1] = - tscale * (shape_target[:, 1] - tmax)
shape_target[:, 0] = tscale * (shape_target[:, 0] - torch.mean(shape_target[:, 0]))

# We only have dot data for one side of the leaf
shape_source = shape_source[shape_source[:, 0] <= 0]
shape_target = shape_target[shape_target[:, 0] <= 0]


###############################################################################
# We now sample the points that will be used by the deformation modules.
#

# Build AABB around the source shape and uniformly sample points for the implicit
# module of order 1
aabb_source = dm.Utilities.AABB.build_from_points(shape_source)
points_growth = dm.Utilities.fill_area_uniform_density(dm.Utilities.area_shape, aabb_source, 0.25, shape=shape_source)

rot_growth = torch.stack([dm.Utilities.rot2d(0.)]*points_growth.shape[0], axis=0)

###############################################################################
# Plot everything.
#

plt.subplot(1, 2, 1)
plt.title("Source leaf")
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='black')
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.title("Target leaf")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='black')
plt.plot(dots_target[:, 0].numpy(), dots_target[:, 1].numpy(), '.', color='black')
plt.axis('equal')

plt.show()


###############################################################################
# We now build the implicit module that will model the growth.
#
# Since we will be learning its growth constants, we need to set its
# `requires_grad` flag to `True`.
#

nu = 0.001
coeff_growth = 1.
scale_growth = 25.

global_translation = dm.DeformationModules.GlobalTranslation(2)

C = torch.ones(points_growth.shape[0], 2, 1)

growth = dm.DeformationModules.ImplicitModule1(
    2, points_growth.shape[0], scale_growth, C, coeff=coeff_growth, nu=nu,
    gd=(points_growth.clone().requires_grad_(),
        rot_growth.clone().requires_grad_()))


###############################################################################
# Quadratic model for the growing constants.
#

# Initial parameters of our model
abc = torch.zeros(6, 2)
abc[0] = torch.ones(2)
abc.requires_grad_()

# The polynomial model we will try to fit on our deformation constants
def pol_order_2(pos, a, b, c, d, e, f):
    return a + b*pos[:, 0] + c*pos[:, 1] + d*pos[:, 0]**2 + e*pos[:, 1]**2 + f*pos[:, 0]*pos[:, 1]

# Callback called when evaluating the model.
# Serves as glue for our model of deformation constants.
def callback_compute_c(init_manifold, modules, parameters):
    abc = parameters['abc'][0]
    a = abc[0].unsqueeze(1)
    b = abc[1].unsqueeze(1)
    c = abc[2].unsqueeze(1)
    d = abc[3].unsqueeze(1)
    e = abc[4].unsqueeze(1)
    f = abc[5].unsqueeze(1)
    modules[3]._ImplicitModule1Base__C = pol_order_2(
        init_manifold[3].gd[0], a, b, c, d, e, f).transpose(0, 1).unsqueeze(2)


###############################################################################
# We now define the model. We set the model parameters as an other parameter
# so that it also get learned.
#

model = dm.Models.ModelPointsRegistration([shape_source, dots_source],
            [global_translation, growth],
            [dm.Attachment.VarifoldAttachment(2, [10., 50.]),
            dm.Attachment.EuclideanPointwiseDistanceAttachment(50.)],
            lam=100., other_parameters=[('abc', [abc])],
            precompute_callback=callback_compute_c)


###############################################################################
# Fitting.
#

shoot_method = 'euler'
shoot_it = 10

fitter = dm.Models.ModelFittingScipy(model)
costs = fitter.fit([shape_target, dots_target], 500, log_interval=25,
                   options={'shoot_method': shoot_method, 'shoot_it': shoot_it})


###############################################################################
# Plot matching results. 
#

intermediate_states, _ = model.compute_deformed(shoot_method, shoot_it, intermediates=True)

deformed_source = intermediate_states[-1][0].gd
deformed_growth = intermediate_states[-1][3].gd[0]
deformed_growth_rot = intermediate_states[-1][3].gd[1]


aabb_target = dm.Utilities.AABB.build_from_points(shape_target).squared().scale(1.1)

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.plot(deformed_growth[:, 0], deformed_growth[:, 1], '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_source[:, 0], deformed_source[:, 1], '-')
plt.axis(aabb_target.totuple())

plt.show()


###############################################################################
# Evaluate and plot learned growth constants. We see more growth at the basis
# than at the apex: this is basipetal growth.
#

learned_abc = abc.detach()
learned_C = pol_order_2(model.init_manifold[3].gd[0].detach(),
                        learned_abc[0].unsqueeze(1),
                        learned_abc[1].unsqueeze(1),
                        learned_abc[2].unsqueeze(1),
                        learned_abc[3].unsqueeze(1),
                        learned_abc[4].unsqueeze(1),
                        learned_abc[5].unsqueeze(1)).transpose(0, 1).unsqueeze(2)

print("Learned growth constants model parameters:")
print(learned_abc)

ax = plt.subplot()
dm.Utilities.plot_C_arrow(ax, points_growth, learned_C, R=deformed_growth_rot, scale=0.0035, mutation_scale=8.)
plt.axis(aabb_source.squared().totuple())
plt.show()

