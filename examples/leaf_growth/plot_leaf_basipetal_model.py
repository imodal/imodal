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
import copy
import pickle

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as dm

torch.set_default_dtype(torch.float64)

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
shape_source = dm.Utilities.close_shape(torch.tensor(data['source_c'], dtype=torch.get_default_dtype()))
shape_target = dm.Utilities.close_shape(torch.tensor(data['target_c'], dtype=torch.get_default_dtype()))

smin, smax = torch.min(shape_source[:, 1]), torch.max(shape_source[:, 1])
sscale = height_source / (smax - smin)
dots_source[:, 1] = sscale * (dots_source[:, 1] - smax)
dots_source[:, 0] = sscale * (dots_source[:, 0] - torch.mean(shape_source[:, 0]))
shape_source[:, 1] = sscale * (shape_source[:, 1] - smax)
shape_source[:, 0] = sscale * (shape_source[:, 0] - torch.mean(shape_source[:, 0]))

tmin, tmax = torch.min(shape_target[:, 1]), torch.max(shape_target[:, 1])
tscale = height_target / (tmax - tmin)
dots_target[:, 1] = tscale * (dots_target[:, 1] - tmax)
dots_target[:, 0] = tscale * (dots_target[:, 0] - torch.mean(shape_target[:, 0]))
shape_target[:, 1] = tscale * (shape_target[:, 1] - tmax)
shape_target[:, 0] = tscale * (shape_target[:, 0] - torch.mean(shape_target[:, 0]))

offset_source = torch.mean(shape_source, dim=0)
offset_target = torch.mean(shape_target, dim=0)
shape_source = shape_source - offset_source
dots_source = dots_source - offset_source
shape_target = shape_target - offset_target
dots_target = dots_target - offset_target


###############################################################################
# We now sample the points that will be used by the deformation modules.
#

# Build AABB around the source shape and uniformly sample points for the implicit
# module of order 1
aabb_source = dm.Utilities.AABB.build_from_points(1.2*shape_source)
# aabb_source = dm.Utilities.AABB.build_from_points(1.*shape_source)
points_growth_density = 0.25
# points_growth = dm.Utilities.fill_area_uniform_density(dm.Utilities.area_shape, aabb_source, points_growth_density, shape=shape_source)
points_growth = dm.Utilities.fill_area_uniform_density(dm.Utilities.area_shape, aabb_source, points_growth_density, shape=1.2*shape_source)

rot_growth = torch.stack([dm.Utilities.rot2d(0.)]*points_growth.shape[0], axis=0)

###############################################################################
# Plot everything.
#

plt.subplot(1, 2, 1)
plt.title("Source leaf")
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='black')
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), 'x', color='black')
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
coeff_growth = 0.005
# scale_growth = 5./math.sqrt(points_growth_density)
scale_growth = 15
scale_translations = height_source/10.

global_translation = dm.DeformationModules.GlobalTranslation(2)

C = torch.empty(points_growth.shape[0], 2, 1)

growth = dm.DeformationModules.ImplicitModule1(
    2, points_growth.shape[0], scale_growth, C, coeff=coeff_growth, nu=nu,
    gd=(points_growth.clone().requires_grad_(),
        rot_growth.clone().requires_grad_()))

translations = dm.DeformationModules.ImplicitModule0(
    2, shape_source.shape[0], scale_translations, nu=nu, gd=shape_source, coeff=10.)

print("Growth sigma={}".format(scale_growth))

###############################################################################
# Quadratic model for the growing constants.
#

# Initial parameters of our model
abc = torch.zeros(6, 2)
abc[0] = torch.ones(2)
abc.requires_grad_()


# The polynomial model we will try to fit on our deformation constants
def pol_order_3_asymetric(pos, a, b, c, d, e, f):
    return a + b*pos[:, 1] + c*pos[:, 0]**2 + d*pos[:, 1]**2 + e*pos[:, 0]**2*pos[:, 1] + f*pos[:, 1]**3
    # return a + b*pos[:, 0] + c*pos[:, 1] + d*pos[:, 0]**2 + e*pos[:, 1]**2 + f*pos[:, 0]*pos[:, 1]


# Callback called when evaluating the model.
# Serves as glue for our model of deformation constants.
def callback_compute_c(init_manifold, modules, parameters):
    abc = parameters['abc']['params'][0]
    a = abc[0].unsqueeze(1)
    b = abc[1].unsqueeze(1)
    c = abc[2].unsqueeze(1)
    d = abc[3].unsqueeze(1)
    e = abc[4].unsqueeze(1)
    f = abc[5].unsqueeze(1)
    modules[3]._ImplicitModule1Base__C = pol_order_3_asymetric(
        init_manifold[3].gd[0], a, b, c, d, e, f).transpose(0, 1).unsqueeze(2)


###############################################################################
# We now define the model. We set the model parameters as an other parameter
# so that it also get learned.
#

deformable_shape_source = dm.Models.DeformablePoints(shape_source)
deformable_shape_target = dm.Models.DeformablePoints(shape_target)
deformable_dots_source = dm.Models.DeformablePoints(dots_source)
deformable_dots_target = dm.Models.DeformablePoints(dots_target)

model = dm.Models.RegistrationModel([deformable_shape_source, deformable_dots_source], [global_translation, growth], [dm.Attachment.VarifoldAttachment(2, [20., 120.]), dm.Attachment.EuclideanPointwiseDistanceAttachment(1000.)], lam=10., other_parameters={'abc': {'params': [abc]}}, precompute_callback=callback_compute_c)


###############################################################################
# Fitting.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = dm.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target, deformable_dots_target], 100, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})

print(abc)
###############################################################################
# Plot matching results.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
    deformed_shape = deformed[0][0]
    deformed_dots = deformed[1][0]

deformed_growth = intermediates['states'][-1][3].gd[0]
deformed_growth_rot = intermediates['states'][-1][3].gd[1]

global_translation_controls = [control[2] for control in intermediates['controls']]
growth_controls = [control[3] for control in intermediates['controls']]

aabb_target = dm.Utilities.AABB.build_from_points(shape_target).squared().scale(1.1)

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.plot(deformed_dots[:, 0], deformed_dots[:, 1], '.')
plt.axis(aabb_target.totuple())

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.axis(aabb_target.totuple())

plt.show()


###############################################################################
# Evaluate and plot learned growth constants. We see more growth at the basis
# than at the apex: this is basipetal growth.
#

learned_abc = abc.detach()
learned_C = pol_order_3_asymetric(model.init_manifold[3].gd[0].detach(),
                                  learned_abc[0].unsqueeze(1),
                                  learned_abc[1].unsqueeze(1),
                                  learned_abc[2].unsqueeze(1),
                                  learned_abc[3].unsqueeze(1),
                                  learned_abc[4].unsqueeze(1),
                                  learned_abc[5].unsqueeze(1)).transpose(0, 1).unsqueeze(2)

print("Learned growth constants model parameters:")
print(learned_abc)

ax = plt.subplot()
dm.Utilities.plot_C_ellipses(ax, points_growth, learned_C, R=deformed_growth_rot, scale=0.00035)
# plt.axis(aabb_source.squared().totuple())
plt.axis('equal')
plt.show()

# We extract the modules of the models and fill the right manifolds.
modules = dm.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone(), copy=True)
silent_shape = copy.copy(modules[0])
silent_dots = copy.copy(modules[1])
global_translation = copy.copy(modules[2])
square_size = 1.
growth_grid_resolution = [math.floor(aabb_source.width/square_size),
                          math.floor(aabb_source.height/square_size)]
deformation_grid = dm.DeformationModules.DeformationGrid(dm.Utilities.AABB.build_from_points(shape_source), growth_grid_resolution)
growth = copy.copy(modules[3])

# We construct the controls list we will give will shooting
controls = [[torch.tensor([]), torch.tensor([]), torch.tensor([]), global_translation_control, growth_control] for growth_control, global_translation_control in zip(growth_controls, global_translation_controls)]

intermediates_growth = {}
with torch.autograd.no_grad():
    dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian([silent_shape, silent_dots, deformation_grid, global_translation, growth]), shoot_solver, shoot_it, controls=controls, intermediates=intermediates_growth)

lddmm_deformed_shape = silent_shape.manifold.gd.detach()
lddmm_deformed_dots = silent_dots.manifold.gd.detach()
lddmm_deformed_grid = deformation_grid.togrid()


###############################################################################
# We now plot the deformation grid.
#
ax = plt.subplot()
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '--', color='red')
plt.plot(lddmm_deformed_shape[:, 0].numpy(), lddmm_deformed_shape[:, 1].numpy())
plt.plot(lddmm_deformed_dots[:, 0].numpy(), lddmm_deformed_dots[:, 1].numpy(), '.')
dm.Utilities.plot_grid(ax, lddmm_deformed_grid[0], lddmm_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

plt.show()


results = {'experience_name': sys.argv[0].split(".")[0], 'source_shape': shape_source, 'target_shape': shape_target, 'source_dots': dots_source, 'target_dots': dots_target, 'deformed_shape': deformed_shape, 'deformed_dots': deformed_dots, 'intermediates': intermediates, 'intermediates_growth': intermediates_growth, 'growth_grid_resolution': growth_grid_resolution, 'abc': abc.detach()}

with open("results_leaf_basipetal_model/results.pt", 'wb') as f:
    torch.save(results, f)

with open("results_leaf_basipetal_model/model.txt", 'w') as f:
    f.write(str(model))

