import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import time
import torch

import implicitmodules.torch as im


def simple_shooting(method, it):
    dim = 2
    nb_pts_silent = 150
    nb_pts_order0 = 100
    nb_pts_order1 = 2000

    pts_silent = torch.rand(nb_pts_silent, dim)
    pts_order0 = torch.rand(nb_pts_order0, dim)
    pts_order1 = torch.rand(nb_pts_order1, dim)

    nu = 0.001
    coeff = 1.
    sigma = 1.5

    C = torch.rand(nb_pts_order1, 2, 1)
    R = torch.rand(nb_pts_order1, 2, 2)

    silent = im.DeformationModules.SilentLandmarks(
        im.Manifolds.Landmarks(dim, nb_pts_silent, gd=pts_silent.view(-1).requires_grad_()))
    order0 = im.DeformationModules.ImplicitModule0(
        im.Manifolds.Landmarks(dim, nb_pts_order0, gd=pts_order0.view(-1).requires_grad_()), sigma, nu, coeff)
    order1 = im.DeformationModules.ImplicitModule1(
        im.Manifolds.Stiefel(dim, nb_pts_order1,
                             gd=(pts_order1.view(-1).requires_grad_(), R.view(-1).requires_grad_())),
        C, sigma, nu, coeff)

    im.shooting.shoot(im.HamiltonianDynamic.Hamiltonian([order1]), it=it, method=method)

    return [silent.manifold.gd, order0.manifold.gd, order1.manifold.gd[0]]


def test_method(method, it, loops):
    time_shooting = []
    time_back = []
    for i in range(loops):
        start = time.time()
        out = simple_shooting(method, it)
        time_shooting.append(time.time() - start)

        scalar = torch.sum(torch.cat([out[0], out[1], out[2]]))

        start = time.time()
        torch.autograd.backward(scalar)
        time_back.append(time.time() - start)

    return sum(time_shooting)/loops, sum(time_back)/loops

def method_summary(method, it, loops):
    avg_shoot, avg_back = test_method(method, it, loops)

    print("For method %s, average shooting time: %5.4f s, average backpropagating time: %5.4f s." % (method, avg_shoot, avg_back))

torch.set_printoptions(precision=4)
method_summary("euler", 10, 1)
method_summary("torch_euler", 10, 1)
for i in range(10):
    method_summary("rk4", i+1, 1)

