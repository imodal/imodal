import numpy as np
import src.DataAttachment.Varifold as var

import old.Forward.shooting as shoot
from old.Backward import Backward as bckwd
import old.Forward.Hamiltonianderivatives as HamDer


def fill_Vector_from_GD(GD):  # tested
    Cot = GD.Cot
    PX = []
    PMom = []
    if '0' in Cot:
        for (x, p) in Cot['0']:
            PX.append(x.flatten())
            PMom.append(p.flatten())
    
    if 'x,R' in Cot:
        for ((x, R), (px, pR)) in Cot['x,R']:
            PX.append(x.flatten())
            PX.append(R.flatten())
            PMom.append(px.flatten())
            PMom.append(pR.flatten())
    
    PX = np.concatenate(PX)
    PMom = np.concatenate(PMom)
    return np.concatenate([PX.copy(), PMom.copy()])


def fill_Mod_from_Vector(P, Mod):  # tested
    """
    Supposes that Mod has a Cot already filled (and that needs to be changed)
    """
    dimP = P.shape[0]
    dimP = int(0.5 * dimP)
    PX = P[:dimP]
    PMom = P[dimP:]
    count = 0
    GD = Mod.GD.copy()
    if '0' in Mod.GD.Cot:
        for (x, p) in Mod.GD.Cot['0']:
            n, d = x.shape
            nx = PX[count: count + n * d].reshape([n, d])
            np = PMom[count: count + n * d].reshape([n, d])
            GD.Cot['0'].append((nx, np))
            count += n * d
    
    if 'x,R' in Mod.GD.Cot:
        for ((x, R), (px, pR)) in Mod.GD.Cot['x,R']:
            n, d = x.shape
            nx = PX[count: count + n * d].reshape([n, d])
            npx = PMom[count: count + n * d].reshape([n, d])
            count += n * d
            nR = PX[count: count + n * d * d].reshape([n, d, d])
            npR = PMom[count: count + n * d * d].reshape([n, d, d])
            GD.Cot['x,R'].append(((nx, nR), (npx, npR)))
            count += n * d * d
    
    GD.updatefromCot()
    Mod.fill_GD(GD)


def jac(P0, *args):
    (Mod, xst, lam_var, sig_var, N, eps) = args
    fill_Mod_from_Vector(P0, Mod)
    ModTraj = shoot.shooting_traj(Mod, N)
    xsf = ModTraj[-1].ModList[0].GD.Cot['0'][0][0]
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    dxvarcost = lam_var * dxvarcost
    
    grad_1 = Mod.GD.copy()
    grad_1.fill_zero()
    grad_1.GD_list[0].fill_GDpts(dxvarcost)
    grad_1.fill_cot_from_GD()
    
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
    xsf = ModTraj[-1].ModList[0].GD.Cot['0'][0][0]
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    varcost = lam_var * varcost
    hamval = HamDer.Ham(ModTraj[0])
    
    print("ham     = {0:10.3e}".format(hamval))
    print("varcost = {0:10.3e}".format(varcost))
    print("totener = {0:10.3e}".format(hamval + varcost))
    return hamval + varcost
