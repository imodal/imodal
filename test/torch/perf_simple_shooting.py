#!/usr/bin/env python3

import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import time
import torch

import implicitmodules.torch as im

im.Utilities.set_compute_backend('keops')


def simple_shooting(method, it, device):
    dim = 2
    nb_pts_silent = 500
    nb_pts_order0 = 200
    nb_pts_order1 = 100

    pts_silent = 100.*torch.rand(nb_pts_silent, dim, device=device)
    pts_order0 = 100.*torch.rand(nb_pts_order0, dim, device=device)
    pts_order1 = 100.*torch.rand(nb_pts_order1, dim, device=device)
    p_pts_order1 = torch.rand(nb_pts_order1, dim, device=device)

    nu = 0.001
    coeff = 1.
    sigma = 1.5

    C = torch.rand(nb_pts_order1, 2, 1, device=device)
    R = torch.rand(nb_pts_order1, 2, 2, device=device)
    p_R = torch.rand(nb_pts_order1, 2, 2, device=device)

    silent = im.DeformationModules.SilentLandmarks(im.Manifolds.Landmarks(dim, nb_pts_silent, gd=pts_silent.view(-1).requires_grad_()))
    order0 = im.DeformationModules.ImplicitModule0(im.Manifolds.Landmarks(dim, nb_pts_order0, gd=pts_order0.view(-1).requires_grad_()), sigma, nu, coeff)
    order1 = im.DeformationModules.ImplicitModule1(im.Manifolds.Stiefel(dim, nb_pts_order1, gd=(pts_order1.view(-1).requires_grad_(), R.view(-1).requires_grad_())), C, sigma, nu, coeff)

    #with torch.autograd.no_grad():
    im.HamiltonianDynamic.shooting.shoot(im.HamiltonianDynamic.Hamiltonian([silent, order0, order1]), it=it, method=method)

    return [silent.manifold.gd, order0.manifold.gd, order1.manifold.gd[0]]


def test_method(method, it, loops, device):
    time_shooting = []
    time_back = []
    for i in range(loops):
        # print("Warmup...")
        # simple_shooting(method, it, device)
        # print("Done.")
        start = time.time()
        out = simple_shooting(method, it, device)
        time_shooting.append(time.time() - start)

        scalar = torch.sum(torch.cat([out[0], out[1], out[2]]))

        start = time.time()
        torch.autograd.backward(scalar)
        time_back.append(time.time() - start)

    return sum(time_shooting)/loops, sum(time_back)/loops


def method_summary(method, it, loops, device):
    avg_shoot, avg_back = test_method(method, it, loops, device)

    print("On device %s, for method %s, average shooting time: %5.4f s, average backpropagating time: %5.4f s." % (device, method, avg_shoot, avg_back))


torch.set_printoptions(precision=4)
method_summary('euler', 1, 1, 'cuda')

