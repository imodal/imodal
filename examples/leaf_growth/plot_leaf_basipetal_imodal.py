"""
Basipetal Leaf Growth Model using Implicit Modules
==================================================

1.) Curve and dots registration using implicit modules of order 1, learning the growth factor.
2.) Curve registration using implicit modules of order with learned growth factor.
"""


###############################################################################
# Import relevant Python modules.
#

import sys
sys.path.append("../../")
import math
import copy
import pickle

import torch

import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

import imodal


###############################################################################
# Learning the 
# --------------
#
# We load the data (shape and dots of the source and target leaves), rescale it and center it.
#

with open("../../data/basipetal.pickle", 'rb') as f:
    data = pickle.load(f)

dots_source = torch.tensor(data['dots_source'], dtype=torch.get_default_dtype())
dots_target = torch.tensor(data['dots_target'], dtype=torch.get_default_dtype())
shape_source = imodal.Utilities.close_shape(torch.tensor(data['shape_source']).type(torch.get_default_dtype()))
shape_target = imodal.Utilities.close_shape(torch.tensor(data['shape_target']).type(torch.get_default_dtype()))

aabb_source = imodal.Utilities.AABB.build_from_points(shape_source)
aabb_target = imodal.Utilities.AABB.build_from_points(shape_target)


###############################################################################
# Plot source and target.
#

plt.subplot(1, 2, 1)
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='blue')
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='blue')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='blue')
plt.plot(dots_target[:, 0].numpy(), dots_target[:, 1].numpy(), '.', color='blue')
plt.axis('equal')
plt.show()


###############################################################################
# We now sample the points that will be used by the implicit deformation
# module of order 1 (growth module).
#

# Build AABB (Axis Aligned Bounding Box) around the source shape and uniformly
# sample points for the growth module.
growth_scale = 30.
points_density = 0.25

aabb_source = imodal.Utilities.AABB.build_from_points(shape_source)

points_growth = imodal.Utilities.fill_area_uniform_density(imodal.Utilities.area_shape, aabb_source, points_density, shape=shape_source)

# Initial normal frames for the growth module.
rot_growth = torch.stack([imodal.Utilities.rot2d(0.)]*points_growth.shape[0], axis=0)


###############################################################################
# Plot points of the growth module.
#

plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), 'o', color='blue')
plt.axis('equal')
plt.show()


###############################################################################
# Create the deformation model with a combination of 3 modules : implicit module
# of order 1 (growth model), implicit module of order 0 (small corrections) and
# a global translation.
#


###############################################################################
# Create and initialize the global translation module.
#

global_translation = imodal.DeformationModules.GlobalTranslation(2)


###############################################################################
# Create and initialize the growth module.
#

nu = 0.001
coeff_growth = 0.001
scale_growth = 30.

C = torch.empty(points_growth.shape[0], 2, 1)

growth = imodal.DeformationModules.ImplicitModule1(2, points_growth.shape[0], scale_growth, C, coeff=coeff_growth, nu=nu, gd=(points_growth, rot_growth))


###############################################################################
# Create and initialize local translations module.
#

coeff_small = 1.
scale_small = 5.
points_small = shape_source.clone()
small_scale_translations = imodal.DeformationModules.ImplicitModule0(2, points_small.shape[0], scale_small, coeff=coeff_small, nu=nu, gd=points_small)


###############################################################################
# Define our growth factor model.
#

# The polynomial model for our growth factor.
def pol(pos, a, b, c, d):
    return a + b*pos[:, 1]  + c*pos[:, 1]**2 + d*pos[:, 1]**3

# Callback called when evaluating the model to compute the growth factor from parameters.
def callback_compute_c(init_manifold, modules, parameters, deformables):
    abcd = parameters['abcd']['params'][0]
    a = abcd[0].unsqueeze(1)
    b = abcd[1].unsqueeze(1)
    c = abcd[2].unsqueeze(1)
    d = abcd[3].unsqueeze(1)
    modules[3].C = pol(init_manifold[3].gd[0], a, b, c, d).transpose(0, 1).unsqueeze(2)

# Initial parameters of our growth factor model.
abcd = torch.zeros(4, 2)
abcd[0] = 0.1 * torch.ones(2)
abcd.requires_grad_()


###############################################################################
# Define deformables used by the registration model.
#

deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)
deformable_dots_source = imodal.Models.DeformablePoints(dots_source)
deformable_dots_target = imodal.Models.DeformablePoints(dots_target)


###############################################################################
# Registration
# ------------
# Define the registration model.
#

model = imodal.Models.RegistrationModel(
    [deformable_shape_source, deformable_dots_source],
    [global_translation, growth, small_scale_translations],
    [imodal.Attachment.VarifoldAttachment(2, [20., 120.], backend='torch'),
     imodal.Attachment.EuclideanPointwiseDistanceAttachment(10.)],
    lam=10., other_parameters={'abcd': {'params': [abcd]}},
    precompute_callback=callback_compute_c)

###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target, deformable_dots_target], 1, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Compute optimized deformation trajectory.
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


###############################################################################
# Plot results.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.plot(deformed_dots[:, 0], deformed_dots[:, 1], '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.plot(deformed_growth[:, 0], deformed_growth[:, 1], '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')
plt.show()


###############################################################################
# Evaluate learned growth factor.
#

learned_abcd = abcd.detach()
learned_C = pol(model.init_manifold[3].gd[0].detach(),
                learned_abcd[0].unsqueeze(1),
                learned_abcd[1].unsqueeze(1),
                learned_abcd[2].unsqueeze(1),
                learned_abcd[3].unsqueeze(1)).transpose(0, 1).unsqueeze(2).detach()
print("Learned growth constants model parameters:\n {}".format(learned_abcd))


###############################################################################
# Plot learned growth factor.
#

ax = plt.subplot()
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
imodal.Utilities.plot_C_ellipses(ax, points_growth, learned_C, R=deformed_growth_rot, scale=1.)
plt.axis(aabb_source.squared().totuple())
plt.axis('equal')
plt.show()


###############################################################################
# Recompute the learned deformation trajectory this time with the grid
# deformation to visualize growth.
#

# We extract the modules of the models and fill the right manifolds.
modules = imodal.DeformationModules.CompoundModule(copy.copy(model.modules))
modules.manifold.fill(model.init_manifold.clone())
silent_shape = copy.copy(modules[0])
silent_dots = copy.copy(modules[1])
global_translation = copy.copy(modules[2])
growth = copy.copy(modules[3])

# Define the deformation grid.
square_size = 1.
growth_grid_resolution = [math.floor(aabb_source.width/square_size),
                          math.floor(aabb_source.height/square_size)]
deformation_grid = imodal.DeformationModules.DeformationGrid(aabb_source, growth_grid_resolution)

# We construct the controls we will give will shooting.
controls = [[torch.tensor([]), torch.tensor([]), torch.tensor([]), global_translation_control, growth_control] for growth_control, global_translation_control in zip(growth_controls, global_translation_controls)]

# Reshoot.
intermediates_growth = {}
with torch.autograd.no_grad():
    imodal.HamiltonianDynamic.shoot(imodal.HamiltonianDynamic.Hamiltonian([silent_shape, silent_dots, deformation_grid, global_translation, growth]), shoot_solver, shoot_it, controls=controls, intermediates=intermediates_growth)

# Store final deformation.
shoot_deformed_shape = silent_shape.manifold.gd.detach()
shoot_deformed_dots = silent_dots.manifold.gd.detach()
shoot_deformed_grid = deformation_grid.togrid()


###############################################################################
# Plot the deformation grid.
#

ax = plt.subplot()
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '--', color='black')
plt.plot(dots_source[:, 0].numpy(), dots_source[:, 1].numpy(), '.', color='black')
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '.-', color='red')
plt.plot(dots_target[:, 0].numpy(), dots_target[:, 1].numpy(), '.', color='black')
plt.plot(shoot_deformed_shape[:, 0].numpy(), shoot_deformed_shape[:, 1].numpy())
plt.plot(shoot_deformed_dots[:, 0].numpy(), shoot_deformed_dots[:, 1].numpy(), '.')
imodal.Utilities.plot_grid(ax, shoot_deformed_grid[0], shoot_deformed_grid[1], color='xkcd:light blue', lw=0.4)
plt.axis('equal')

plt.show()


###############################################################################
# Perform curve registration using the previously learned growth factor
# ---------------------------------------------------------------------
#

###############################################################################
# Redefine deformation modules.
#

global_translation = imodal.DeformationModules.GlobalTranslation(2)

growth = imodal.DeformationModules.ImplicitModule1(2, points_growth.shape[0], scale_growth, learned_C, coeff=coeff_growth, nu=nu, gd=(points_growth, rot_growth))

small_scale_translation = imodal.DeformationModules.ImplicitModule0(2, shape_source.shape[0], scale_small, coeff=coeff_small, nu=nu, gd=shape_source)


###############################################################################
# Redefine deformables and registration model.
#

deformable_shape_source = imodal.Models.DeformablePoints(shape_source)
deformable_shape_target = imodal.Models.DeformablePoints(shape_target)

refit_model = imodal.Models.RegistrationModel([deformable_shape_source],
                [global_translation, growth, small_scale_translation],
                [imodal.Attachment.VarifoldAttachment(2, [20., 120.], backend='torch')],
                lam=10.)


###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(refit_model, optimizer='torch_lbfgs')
fitter.fit([deformable_shape_target], 1, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Compute optimized deformation trajectory.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = refit_model.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)
    deformed_shape = deformed[0][0]
deformed_growth = intermediates['states'][-1][2].gd[0]
deformed_growth_rot = intermediates['states'][-1][2].gd[1]


###############################################################################
# Plot results.
#

plt.subplot(1, 3, 1)
plt.title("Source")
plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), '-')
plt.plot(points_growth[:, 0].numpy(), points_growth[:, 1].numpy(), '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 2)
plt.title("Deformed source")
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.axis(aabb_target.totuple())
plt.axis('equal')

plt.subplot(1, 3, 3)
plt.title("Deformed source and target")
plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), '-')
plt.plot(deformed_shape[:, 0], deformed_shape[:, 1], '-')
plt.plot(deformed_growth[:, 0], deformed_growth[:, 1], '.')
plt.axis(aabb_target.totuple())
plt.axis('equal')
plt.show()


###############################################################################
# Recompute the learned deformation trajectory this time with the grid
# deformation to visualize growth.
#

modules = imodal.DeformationModules.CompoundModule(copy.copy(refit_model.modules))
modules.manifold.fill(refit_model.init_manifold)

square_size = 1.
grid_resolution = [math.floor(aabb_source.width/square_size),
                   math.floor(aabb_source.height/square_size)]
deformation_grid = imodal.DeformationModules.DeformationGrid(aabb_source, growth_grid_resolution)

controls = [control[1:] for control in intermediates['controls']]
print(len(controls[0]))

deformable_shape = imodal.Models.DeformablePoints(shape_source)
deformable_shape.silent_module.manifold.cotan = refit_model.init_manifold[0].cotan
deformable_grid = imodal.Models.DeformableGrid(aabb_source, grid_resolution)

intermediates = {}
with torch.autograd.no_grad():
    imodal.Models.deformables_compute_deformed([deformable_shape, deformable_grid], modules, shoot_solver, shoot_it, intermediates=intermediates, controls=controls)


###############################################################################
# Plot the growth trajectory.
#

indices = [1, 3, 7, 10]

plt.figure(figsize=[5.*len(indices), 5.])
for i, index in enumerate(indices):
    state = intermediates['states'][index]

    ax = plt.subplot(1, len(indices), i + 1)
    plt.plot(shape_source[:, 0].numpy(), shape_source[:, 1].numpy(), color='black')
    plt.plot(shape_target[:, 0].numpy(), shape_target[:, 1].numpy(), color='red')
    plt.plot(state[0].gd[:, 0].numpy(), state[0].gd[:, 1].numpy())

    deformable_grid.silent_module.manifold.fill_gd(state[1].gd)
    grid_x, grid_y = deformable_grid.silent_module.togrid()
    imodal.Utilities.plot_grid(ax, grid_x, grid_y, color='xkcd:light blue', lw=0.4)
    plt.axis('equal')
    plt.axis('off')

plt.show()

