import sys

import meshio

sys.path.append('../')
import imodal
import torch
import numpy as np
device = 'cuda:0'
torch.set_default_device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)
backend = 'keops'
imodal.Utilities.set_compute_backend(backend)

path_data = '../data/'
path_results = '../Results/'

sigma_fac = 2.
nu_fac = 1e-7
maxiter = 50
solver = 'midpoint'
stop_after_fit = 'False'
method = 'moments'
lam = 1e10
arap_coeff = 1.
arap_cost = 1e-4


#load data and initial control points
source_mesh = meshio.read(path_data + 'ARAP_source.ply')
target_mesh = meshio.read(path_data + 'ARAP_target.ply')

source_points = torch.tensor(
    source_mesh.points, dtype=torch.get_default_dtype())
target_points = torch.tensor(
    target_mesh.points, dtype=torch.get_default_dtype())
source_triangles = torch.tensor(
    source_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)
target_triangles = torch.tensor(
    target_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)


source_grid = meshio.read(path_data + 'ARAP_CP.ply')
control_points = torch.tensor(source_grid.points)

###############################################################################
# Create and initialize the ARAP module.
#
# Compute the minimal distance between two distinct control points
cp_dist = (torch.cdist(control_points, control_points) + torch.eye(control_points.shape[0]) * 1e9).min()
sigma = sigma_fac * cp_dist
nu = nu_fac
arap_module = imodal.DeformationModules.ImplicitARAP(
    3, [control_points.shape[0], control_points.shape[0]],
    sigma=sigma,
    coeff_arap=arap_cost,
    coeff=arap_coeff,
    nu=nu,
    gd=torch.cat([control_points.clone(), control_points.clone()], dim=0)
)


###############################################################################
# Define deformables used by the registration model.
#

deformable_source = imodal.Models.DeformableMesh(
    source_points, source_triangles)
deformable_target = imodal.Models.DeformableMesh(
    target_points, target_triangles)

deformable_source.to_device(device)
deformable_target.to_device(device)


###############################################################################
# Define the registration model.
#
sigmas_varifold = [1. * cp_dist, 2. * cp_dist, 4. * cp_dist,10. * cp_dist]
attachment = imodal.Attachment.VarifoldAttachment(3, sigmas_varifold)

#attachment = imodal.Attachment.EuclideanPointwiseDistanceAttachment()

model = imodal.Models.RegistrationModel(
    deformable_source,
    [arap_module],
    [attachment],
    fit_controls=False,
    init_controls_ones=False,
    it=10, solver=solver,
    lam=lam, reduce='sum'
)


model.to_device(device)

###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = solver
shoot_it = 10
costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')
lamc_costs = fitter.fit(deformable_target, maxiter, miniter=3,
           costs=costs, options={
           'shoot_solver': shoot_solver,
           'shoot_it': shoot_it,
           'line_search_fn': 'strong_wolfe',
           'history_size': 500})


###############################################################################
# Compute optimized deformation trajectory.
#

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(
        shoot_solver, shoot_it, intermediates=intermediates)[0][0].detach()

torch.save(
    intermediates,
    path_results+'ARAP_intermediates.pth'
)

