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
import math
import pickle

import torch
import matplotlib.pyplot as plt

sys.path.append("../")

import imodal

###############################################################################
# Load the dataset, extract the template peanut and the target peanuts.
#

torch.set_default_dtype(torch.float32)

data = pickle.load(open("../data/peanuts.pickle", 'rb'))

peanuts_count = 4
peanuts = [torch.tensor(peanut[:-1], dtype=torch.get_default_dtype()) for peanut in data[0][1:peanuts_count+1]]

template = imodal.Utilities.generate_unit_circle(200)
template = imodal.Utilities.linear_transform(template, torch.tensor([[1.3, 0.], [0., 0.5]]))
template = imodal.Utilities.close_shape(template)

deformable_template = imodal.Models.DeformablePoints(template.clone().requires_grad_(False))
deformable_peanuts = [imodal.Models.DeformablePoints(peanut) for peanut in peanuts]

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

sigma_scale = 1.
sigma_local = 0.03

left_scale = imodal.DeformationModules.LocalScaling(2, sigma_scale, gd=point_left_scale, coeff=0.1)
right_scale = imodal.DeformationModules.LocalScaling(2, sigma_scale, gd=point_right_scale, coeff=0.1)
local_translation = imodal.DeformationModules.ImplicitModule0(2, deformable_template.silent_module.manifold.gd.shape[0], sigma_local, gd=deformable_template.silent_module.manifold.gd.clone(), coeff=1., nu=0.1)

global_translation = imodal.DeformationModules.GlobalTranslation(2)


###############################################################################
# Initialise the model.
#
# We set the `fit_gd` flags to `True` for the scaling modules in order to optimise
# their positions.
#

sigmas_varifold = [0.4, 2.5]
attachment = imodal.Attachment.VarifoldAttachment(2, sigmas_varifold)

atlas = imodal.Models.AtlasModel(deformable_template, [global_translation, left_scale, right_scale], [attachment], len(peanuts), lam=100., optimise_template=True, ht_sigma=0.4, ht_it=10, ht_coeff=.5, ht_nu=0.5, fit_gd=[False, True, True, False])


###############################################################################
# Fit the atlas.
#

shoot_solver = 'euler'
shoot_it = 10

costs = {}
fitter = imodal.Models.Fitter(atlas, optimizer='torch_lbfgs')

# with torch.autograd.detect_anomaly():
fitter.fit(deformable_peanuts, 20, costs=costs, options={'shoot_solver': shoot_solver, 'shoot_it': shoot_it, 'line_search_fn': 'strong_wolfe'})


###############################################################################
# Extract and plot optimised positions.
#

optimised_left = atlas.registration_models[0].init_manifold[2].gd.detach().view(2)
optimised_right = atlas.registration_models[0].init_manifold[3].gd.detach().view(2)
ht = atlas.compute_template().detach()

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

intermediates = {}
with torch.autograd.no_grad():
    deformed_templates = atlas.compute_deformed(shoot_solver, shoot_it, intermediates=intermediates)

row_count = math.ceil(math.sqrt(len(peanuts)))

for i, deformed, peanut in zip(range(len(peanuts)), deformed_templates, peanuts):
    plt.subplot(row_count, row_count, 1 + i)
    plt.plot(deformed[0].detach()[:, 0].numpy(), deformed[0].detach()[:, 1].numpy())
    plt.plot(peanut[:, 0].numpy(), peanut[:, 1].numpy())
    plt.axis('equal')

plt.show()


###############################################################################
# Display shooting steps for each pair.
#

it_per_snapshot = 1
snapshots = int(shoot_it/it_per_snapshot)

for i, deformed_states, peanut in zip(range(len(peanuts)), intermediates['states'], peanuts):
    for j in range(snapshots):
        plt.subplot(peanuts_count, snapshots+1, i*(snapshots+1) + j + 1)
        plt.plot(deformed_states[j*it_per_snapshot].gd[0][:, 0].numpy(), deformed_states[j*it_per_snapshot].gd[0][:, 1].numpy())
        plt.plot(peanut[:, 0], peanut[:, 1], lw=0.5)
        plt.axis('equal')

    plt.subplot(peanuts_count, snapshots+1, i*(snapshots+1)+snapshots+1)
    plt.plot(deformed_states[-1].gd[0][:, 0].numpy(), deformed_states[-1].gd[0][:, 1].numpy())
    plt.plot(peanut[:, 0], peanut[:, 1], lw=0.5)
    plt.axis('equal')

plt.show()

