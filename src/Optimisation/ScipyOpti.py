# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:31:40 2019

@author: gris
"""

import numpy as np
import src.data_attachment.varifold as var

import src.Forward.shooting as shoot
import src.Backward.Backward as bckwd
import src.Forward.Hamiltonianderivatives as HamDer


def fill_Vector_from_GD(GD):  # 
    PX = GD.get_GDinVector()
    PMom = GD.get_cotaninVector()

    return np.concatenate([PX.copy(), PMom.copy()])


def fill_Mod_from_Vector(P, Mod):  # tested
    """
    Supposes that Mod has a Cot already filled (and that needs to be changed)
    """
    dimP = P.shape[0]
    dimP = int(0.5 * dimP)
    PX = P[:dimP]
    PMom = P[dimP:]
    GD = Mod.GD.copy()
    GD.fill_from_vec(PX, PMom)

    #GD.updatefromCot()
    Mod.fill_GD(GD)


def jac(P0, *args):
    (Mod, xst, lam_var, sig_var, N, eps) = args
    fill_Mod_from_Vector(P0, Mod)
    ModTraj = shoot.shooting_traj(Mod, N)
    xsf = ModTraj[-1].ModList[0].GD.GD
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    dxvarcost = lam_var * dxvarcost
    
    grad_1 = Mod.GD.copy()
    grad_1.fill_zero()
    grad_1.GD_list[0].cotan = dxvarcost
    #grad_1.fill_cot_from_GD()
    
    cgrad = bckwd.backward_shoot_rk2(ModTraj, grad_1, eps)
    
    dP = fill_Vector_from_GD(cgrad)
    n = dP.shape[0]
    n = int(0.5 * n)
    # n = np.prod(xst.shape)
    dP[:n] = 0.
    return dP


def fun(P0, *args):
    (Mod, xst, lam_var, sig_var, N, eps) = args
    fill_Mod_from_Vector(P0, Mod)
    ModTraj = shoot.shooting_traj(Mod, N)
    xsf = ModTraj[-1].ModList[0].GD.GD
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    varcost = lam_var * varcost
    hamval = HamDer.Ham(ModTraj[0])
    
    print("ham     = {0:10.3e}".format(hamval))
    print("varcost = {0:10.3e}".format(varcost))
    print("totener = {0:10.3e}".format(hamval + varcost))
    return hamval + varcost
