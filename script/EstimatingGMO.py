
import sys

import meshio

sys.path.append('../')
import imodal
import torch
import numpy as np
device = 'cuda:2'
torch.set_default_device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)
backend = 'torch'
imodal.Utilities.set_compute_backend(backend)

path_data = '../data/'
path_results = '../Results/'

sigma_fac = 4.
nu_fac = 1e-5
maxiter =50 
solver = 'midpoint'
stop_after_fit = 'False'
method = 'moments'
lam = 1e7
gt_coeff = 100
imp1_coeff = 10
imp1_sigma_fac = 4
varifold_facs = [1.0, 2.0, 4.0]
solver_it = 8
#load data and initial control points
source_mesh = meshio.read(path_data + 'EstimatingGMO_source.ply')
target_mesh = meshio.read(path_data + 'EstimatingGMO_target.ply')

source_points = torch.tensor(
    source_mesh.points, dtype=torch.get_default_dtype())
target_points = torch.tensor(
    target_mesh.points, dtype=torch.get_default_dtype())
source_triangles = torch.tensor(
    source_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)
target_triangles = torch.tensor(
    target_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)


source_grid = meshio.read(path_data + 'EstimatingGMO_CP.ply')
control_points = torch.tensor(source_grid.points, dtype=torch.get_default_dtype())

# Compute the minimal distance between two distinct control points
cp_dist = (torch.cdist(control_points, control_points) + torch.eye(control_points.shape[0]) * 1e9).min()
# Compute sigma for the implicit module
sigma = sigma_fac * cp_dist

nu = nu_fac

# params = [A, B]
# C defined by A x + B parameters
# compute C from points and parameter
def linear_C(pts, params):
    z = pts[:, 2]
    a, b = params
    c = a * z.unsqueeze(-1).repeat(1, 3).unsqueeze(-1) + b
    return c

# initialisation of growth model operator: C uniform
params = torch.tensor([0.0, 1.0], dtype=torch.get_default_dtype(), requires_grad=True)
C = linear_C(control_points, params)
R = torch.eye(3, dtype=torch.get_default_dtype()).repeat(control_points.shape[0], 1, 1)
# building the implicit deformation module of order 1
imp1_module = imodal.DeformationModules.ImplicitModule1(
    3, control_points.shape[0],
    sigma, C.clone(),
    coeff=imp1_coeff,
    nu=nu,
    gd=(control_points.clone(), R.clone())
)
#

gt_module = imodal.DeformationModules.GlobalTranslation(3)


# defining a callback function to optimise GMO
def precompute(init_manifold, modules, parameters, deformables):
    cfs = parameters['coeffs']['params'][0]
    print(f'params: {cfs}')
    c = linear_C(init_manifold[1].gd[0], cfs)
    modules[1].C = c

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
sigmas_varifold = [fac * cp_dist for fac in varifold_facs]
attachment = imodal.Attachment.VarifoldAttachment(3, sigmas_varifold)

model = imodal.Models.RegistrationModel(
    deformable_source,
    [imp1_module, gt_module],
    [attachment],
    fit_controls=(False if method == 'moments' else True),
    it=solver_it, solver=solver,
    lam=lam, reduce='sum',
    precompute_callback=precompute,
    other_parameters={'coeffs': {'params': [params]}}
)


model.to_device(device)
###############################################################################
# Fitting using Torch LBFGS optimizer.
#

shoot_solver = solver
shoot_it = solver_it
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
    path_results+'Estimating_GMO_intermediates.pth'
)

torch.save(
    params,
    path_results+'Estimating_GMO_learned_params'
)
