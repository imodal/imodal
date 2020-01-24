import math
import sys
import pickle

import torch
import matplotlib.pyplot as plt

sys.path.append("../../")
import implicitmodules.torch as dm


points_per_side = 40

silent_points = dm.Utilities.generate_unit_square(points_per_side)
silent_points = silent_points*torch.tensor([1., 4.])
aabb = dm.Utilities.AABB.build_from_points(silent_points).scale(0.9)
implicit1_points = aabb.fill_uniform_density(3.)
implicit1_points = implicit1_points - torch.mean(implicit1_points, dim=0)
implicit1_R = torch.eye(2).repeat(implicit1_points.shape[0], 1, 1)

C = torch.zeros(implicit1_points.shape[0], 2, 1)
C[:, 1, 0] = 10.*torch.abs(implicit1_points[:, 0] - torch.mean(implicit1_points, dim=0)[0] + 0.5*aabb.width)/aabb.width



def generate_bending():
    def _generate():
        silent_cotan = torch.randn_like(silent_points)
        implicit1_cotan = torch.zeros_like(implicit1_points)
        #val = 1000.*torch.rand(1) - 500.
        # val = 100.*torch.randn(1)

        # implicit1_cotan[0::7, 0] = val
        # implicit1_cotan[6::7, 0] = val
        #implicit1_cotan[6, 0] = val
        #implicit1_cotan[-1, 0] = 
        implicit1_cotan = 15.*torch.randn_like(implicit1_points)
        implicit1_cotan_R = torch.zeros_like(implicit1_R)

        # plt.plot(silent_points[:, 0].numpy(), silent_points[:, 1].numpy())
        # plt.quiver(implicit1_points[:, 0].numpy(), implicit1_points[:, 1].numpy(), implicit1_cotan[:, 0].numpy(), implicit1_cotan[:, 1].numpy())
        # plt.axis('equal')
        # plt.show()

        # ax = plt.subplot()
        # plt.plot(silent_points[:, 0].numpy(), silent_points[:, 1].numpy())
        # dm.Utilities.plot_C_arrow(ax, implicit1_points, C, scale=0.1, mutation_scale=10.)
        # plt.axis('equal')
        # plt.show()

        silent = dm.DeformationModules.SilentLandmarks(2, silent_points.shape[0], gd=silent_points.clone().requires_grad_(), cotan=silent_cotan.clone().requires_grad_())

        implicit1 = dm.DeformationModules.ImplicitModule1(2, implicit1_points.shape[0], 1.5, C, nu=0.1, gd=(implicit1_points.clone().requires_grad_(), implicit1_R.clone().requires_grad_()), cotan=(implicit1_cotan.clone().requires_grad_(), implicit1_cotan_R.clone().requires_grad_()))

        with torch.autograd.no_grad():
            dm.HamiltonianDynamic.shoot(dm.HamiltonianDynamic.Hamiltonian([silent, implicit1]), 10, 'rk4')

        return silent.manifold.gd.detach(), implicit1.manifold.gd[0].detach()

    while True:
        bending = _generate()
        #print(1.5*torch.norm(silent_points), " ", torch.norm(bending[0]), " ", 4.*torch.norm(silent_points))
        if (not torch.any(torch.isnan(bending[0]))) and (torch.norm(bending[0]) >= 0.*torch.norm(silent_points)) and (torch.norm(bending[0]) <= 10000.*torch.norm(silent_points)):
            print("New data element added. Frobenius norm: {norm}".format(norm=torch.norm(bending[0])))
            return bending


out_filename = sys.argv[1]
N_bending = int(sys.argv[2])
bendings = [(silent_points, implicit1_points, C)]

for i in range(N_bending):
    bendings.append(generate_bending())

for bending in bendings:
    plt.plot(bending[0][:, 0].numpy(), bending[0][:, 1].numpy())

plt.axis('equal')
plt.show()

with open(out_filename, 'wb') as f:
    pickle.dump(bendings, f)

