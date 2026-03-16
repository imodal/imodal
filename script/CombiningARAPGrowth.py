import sys
sys.path.append('../')
import pyvista as pv
import time
import imodal
import meshio
import torch
import numpy as np
import os

path_data = '../data/'
path_results = '../Results/'


device = 'cuda:2'

lam = 10000000.0

nu = 1e-05

gt_coeff = 100.0

imp1_coeff = 10.0
imp1_sigma_fac = 4.0


arap_coeff = 2000.0
arap_cost = 0.001
arap_sigma_fac = 2.0

maxiter = 150


solver = 'midpoint'
solver_it = 8

varifold_facs =  [1.0, 2.0, 4.0]

torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
imodal.Utilities.set_compute_backend('torch')


data_folder = f'../Results/CombiningARAPGrowth/'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)




print('Reading meshes ...')

source_mesh = meshio.read(path_data + 'CombiningARAPGrowth_source.ply')
target_mesh = meshio.read(path_data + 'CombiningARAPGrowth_target.ply')

source_points = torch.tensor(
    source_mesh.points, dtype=torch.get_default_dtype())
target_points = torch.tensor(
    target_mesh.points, dtype=torch.get_default_dtype())
source_triangles = torch.tensor(
    source_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)
target_triangles = torch.tensor(
    target_mesh.cells_dict['triangle'].astype(np.int32), dtype=torch.int)

print(f'Source points: {source_points.shape}')
print(f'Target points: {target_points.shape}')
print(f'Source triangles: {source_triangles.shape}')
print(f'Target triangles: {target_triangles.shape}')

imodal.Utilities.export_mesh(
    data_folder+'source.ply', source_points.cpu(), source_triangles.cpu())
imodal.Utilities.export_mesh(
    data_folder+'target.ply', target_points.cpu(), target_triangles.cpu())

xmin, ymin, zmin = source_points.min(0).values
xmax, ymax, zmax = source_points.max(0).values

print(' > Source points')
print(f' > xmin={xmin}, xmax={xmax}')
print(f' > ymin={ymin}, ymax={ymax}')
print(f' > zmin={zmin}, zmax={zmax}')

print('Reading ImplicitModule1 control points ...')


cp_imp1_mesh = meshio.read(path_data + 'CombiningARAPGrowth_CP_im1.ply')

control_points_imp1 = torch.tensor(cp_imp1_mesh.points, dtype=torch.get_default_dtype())

xmin, ymin, zmin = control_points_imp1.min(0).values
xmax, ymax, zmax = control_points_imp1.max(0).values

print(f' > Control points shape for ImplicitModule1: {control_points_imp1.shape}')
print(f' > xmin={xmin}, xmax={xmax}')
print(f' > ymin={ymin}, ymax={ymax}')
print(f' > zmin={zmin}, zmax={zmax}')

cp_dist_imp1 = (torch.cdist(control_points_imp1, control_points_imp1) + torch.eye(control_points_imp1.shape[0]) * 1e9).min()

print(f' > Control points distance: {cp_dist_imp1}')

print('Reading ARAP control points ...')

cp_arap_mesh = meshio.read(path_data + 'CombiningARAPGrowth_CP_arap.ply')

control_points_arap = torch.tensor(cp_arap_mesh.points, dtype=torch.get_default_dtype())

xmin, ymin, zmin = control_points_arap.min(0).values
xmax, ymax, zmax = control_points_arap.max(0).values

print(f' > Control points shape for ARAP: {control_points_arap.shape}')
print(f' > xmin={xmin}, xmax={xmax}')
print(f' > ymin={ymin}, ymax={ymax}')
print(f' > zmin={zmin}, zmax={zmax}')

cp_dist_arap = (torch.cdist(control_points_arap, control_points_arap) + torch.eye(control_points_arap.shape[0]) * 1e9).min()

print(f' > Control points distance: {cp_dist_arap}')

###############################################################################
sigma_imp1 = cp_dist_imp1 * imp1_sigma_fac

C = torch.ones((control_points_imp1.shape[0], 3, 1))
R = torch.eye(3).repeat(control_points_imp1.shape[0], 1, 1)

def linear_C(pts, params):
    z = pts[:, 2]
    a, b = params
    c = a * z.unsqueeze(-1).repeat(1, 3).unsqueeze(-1) + b
    return c

# initialisation of growth model operator: C uniform
params = torch.load('../Results/Estimating_GMO_learned_params', map_location=device).clone().detach()
C = linear_C(control_points_imp1, params)

imp1_module = imodal.DeformationModules.ImplicitModule1(
    3, control_points_imp1.shape[0],
    sigma_imp1, C.clone(),
    coeff=imp1_coeff,
    nu=nu,
    gd=(control_points_imp1.clone(), R.clone())
)

gt_module = imodal.DeformationModules.GlobalTranslation(3, coeff=gt_coeff)


arap_module = imodal.DeformationModules.ImplicitARAP(
    3, [control_points_arap.shape[0], control_points_arap.shape[0]],
    sigma=cp_dist_arap * arap_sigma_fac,
    coeff_arap=arap_cost,
    coeff=arap_coeff,
    nu=nu,
    gd=torch.cat([control_points_arap.clone(), control_points_arap.clone()], dim=0)
)


deformable_source = imodal.Models.DeformableMesh(
    source_points, source_triangles)
deformable_target = imodal.Models.DeformableMesh(
    target_points, target_triangles)

deformable_source.to_device(device)
deformable_target.to_device(device)

sigmas_varifold = [fac * cp_dist_arap for fac in varifold_facs]
attachment = imodal.Attachment.VarifoldAttachment(3, sigmas_varifold)


model = imodal.Models.RegistrationModel(
    deformable_source,
    [imp1_module, arap_module, gt_module],
    [attachment],
    fit_controls=False,
    it=solver_it, solver=solver,
    lam=lam, reduce='sum'
)


model.to_device(device)

###############################################################################
# Fitting using Torch LBFGS optimizer.
#
shoot_solver = solver
shoot_it = solver_it
costs = {}
fitter = imodal.Models.Fitter(model, optimizer='torch_lbfgs')

fitter.fit(
        deformable_target, 
        maxiter,
        costs=costs, 
        options={
            'shoot_solver': shoot_solver,
            'shoot_it': shoot_it,
            'line_search_fn': 'strong_wolfe', 
            'history_size': 20
        }
    )



print('Fitting done ')

print('Cost ratio:', costs['attach'][-1] / costs['attach'][0])

intermediates = {}
with torch.autograd.no_grad():
    deformed = model.compute_deformed(
        shoot_solver, shoot_it, intermediates=intermediates)[0][0].detach()

torch.save(
    intermediates,
    data_folder+'intermediates.pth'
)

for i, inter in enumerate(intermediates['states']):
    imodal.Utilities.export_mesh(
        data_folder+f'matching_{i}.ply', inter[0].gd.cpu(), source_triangles.cpu())
    imodal.Utilities.export_mesh(
        data_folder+f'control_points_{i}.ply', inter[2].gd.cpu(), torch.tensor([[0, 1, 2]]).cpu())


imp1_module = imodal.DeformationModules.ImplicitModule1(
    3, control_points_imp1.shape[0],
    sigma_imp1, C.clone(),
    coeff=imp1_coeff,
    nu=nu,
    gd=(control_points_imp1.clone(), R.clone())
)
arap_module = imodal.DeformationModules.ImplicitARAP(
    3, [control_points_arap.shape[0], control_points_arap.shape[0]],
    sigma=cp_dist_arap * arap_sigma_fac,
    coeff_arap=arap_cost,
    coeff=arap_coeff,
    nu=nu,
    gd=torch.cat([control_points_arap.clone(), control_points_arap.clone()], dim=0)
)

# distance between ARAP control points (i,j)
arap_dist_mat = torch.cdist(control_points_arap, control_points_arap) + torch.eye(control_points_arap.shape[0]) * 1e9

# for each ARAP control point, find the index of the closest control point (j)
closest = arap_dist_mat.argmin(1)

# for each ARAP control point, compute the distance to its respective closest control point (j)
dist_to_closest = arap_dist_mat.min(1).values

# array to store the new ARAP control points
new_arap_controls = []
for i, conts in enumerate(intermediates['controls']):
    # get the ARAP control points for the current time step
    # divide by 2 because controls come from a midpoint shooting
    arap_cp_i = intermediates['states'][i//2][2].gd[:control_points_arap.shape[0]]

    # compute the distance to the closest control point (j) for each ARAP control point (i)
    # at current time step
    dist_to_closest_i = torch.linalg.vector_norm(arap_cp_i - arap_cp_i[closest], dim=1)

    # compute the ratio between the distance to the closest control point (j) at current time step
    # and the distance to the closest control point (j) at the initial time step
    ratio = dist_to_closest_i / dist_to_closest

    # get the ARAP controls at the current timestep
    arap_cont = conts[2].detach().clone().reshape(-1, 3)

    # divide the ARAP controls by the ratio
    # we want to reduce the magnitude of the ARAP controls in later time steps
    # because the distance is increased by ImplicitModule1
    new_arap_cont = arap_cont / ratio.unsqueeze(-1)

    # store the new ARAP controls
    new_arap_controls.append([new_arap_cont.reshape(-1)])


intermediates_imp1 = {}
costs_imp1 = {}
with torch.autograd.no_grad():
    deformed = imodal.Models.deformables_compute_deformed(
        [imodal.Models.DeformableMesh(source_points, source_triangles)],
        [imp1_module],
        shoot_solver, shoot_it,
        controls=[[cont[1]] for cont in intermediates['controls']],
        intermediates=intermediates_imp1,
        costs=costs_imp1)[0]
    
intermediates_arap = {}
costs_arap = {}
with torch.autograd.no_grad():
    deformed = imodal.Models.deformables_compute_deformed(
        [imodal.Models.DeformableMesh(source_points, source_triangles)],
        [arap_module],
        shoot_solver, shoot_it,
        #controls=[[cont[2]] for cont in intermediates['controls']],
        controls=new_arap_controls,
        intermediates=intermediates_arap, costs=costs_arap)[0]

for i, (inter_imp1, inter_arap) in enumerate(zip(intermediates_imp1['states'], intermediates_arap['states'])):
    imodal.Utilities.export_mesh(
        data_folder+f'imp1_matching_{i}.ply', inter_imp1[0].gd.cpu(), source_triangles.cpu())
    imodal.Utilities.export_mesh(
        data_folder+f'arap_matching_{i}.ply', inter_arap[0].gd.cpu(), source_triangles.cpu())
    imodal.Utilities.export_mesh(
        data_folder+f'imp1_control_points_{i}.ply', inter_imp1[1].gd[0].cpu(), torch.tensor([[0, 1, 2]]).cpu())
    imodal.Utilities.export_mesh(
        data_folder+f'arap_control_points_{i}.ply', inter_arap[1].gd.cpu(), torch.tensor([[0, 1, 2]]).cpu())   

# report is a dict contains:
# arguments of the script and their values 
report = {
    'device': device,
    'lam': lam,
    'nu': nu,
    'gt_coeff': gt_coeff,
    'imp1_coeff': imp1_coeff,
    'imp1_sigma_fac': imp1_sigma_fac,
    'arap_coeff': arap_coeff,
    'arap_cost': arap_cost,
    'arap_sigma_fac': arap_sigma_fac,
    'maxiter': maxiter,
    'solver': solver,
    'solver_it': solver_it,
    'varifold_facs': varifold_facs,
    'n_iter': len(costs['attach']),
    'costs': costs,
    'costs_arap': costs_arap,
    'costs_imp1': costs_imp1
}

torch.save(
    report,
    data_folder+'report.pth'
)

text_report = [
    'device',
    'lam', 'nu',
    'gt_coeff',
    'imp1_coeff', 'imp1_sigma_fac',
    'arap_coeff', 'arap_cost', 'arap_sigma_fac',
    'n_iter', 
    'maxiter', 'solver', 'solver_it',
    'varifold_facs'
]

text_report = {key: report[key] for key in text_report}

text_report['attach_ratio'] = costs['attach'][-1] / costs['attach'][0]
text_report['final_attach'] = costs['attach'][-1]
text_report['final_deform'] = costs['deformation'][-1]

text_report['final_deform_arap'] = costs_arap['deformation']
text_report['final_deform_imp1'] = costs_imp1['deformation']

with open(data_folder+'report.txt', 'w') as file:
    for key, value in text_report.items():
        file.write(f'{key}: {value}\n')

    file.write('\n\n')
    file.write('Implicit module controls:\n')
    for i, conts in enumerate(intermediates['controls']):
        imp1_cont = conts[1]
        file.write(f'{i}: {imp1_cont.item()}\n')
