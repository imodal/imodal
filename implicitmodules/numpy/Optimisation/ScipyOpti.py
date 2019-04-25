import numpy as np

import implicitmodules.numpy.Backward.Backward as bckwd
import implicitmodules.numpy.DataAttachment.Varifold as var
import implicitmodules.numpy.Forward.Hamiltonianderivatives as HamDer
import implicitmodules.numpy.Forward.Shooting as shoot


def fill_Vector_from_GD(GD):  # 
    PX = GD.get_GDinVector()
    PMom = GD.get_cotaninVector()
    return np.concatenate([PX.copy(), PMom.copy()])


def fill_Vector_from_tancotan(GD):  # 
    PX = GD.get_taninVector()
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
    
    # GD.updatefromCot()
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
    grad_1.GD_list[0].tan = dxvarcost
    # grad_1.fill_cot_from_GD()
    
    cgrad = bckwd.backward_shoot_rk2(ModTraj, grad_1, eps)
    
    # add gradient of the cost ie of the hamiltonian here:
    fill_Mod_from_Vector(P0, Mod)
    Mod.update()
    Mod.GeodesicControls_curr(Mod.GD)
    # dxH is -derivtes wrt x, it is put in the cotan element
    dx = HamDer.dxH(Mod)
    dx.exchange_tan_cotan()
    dx.mult_tan_scal(-1.)
    dP_dx = fill_Vector_from_tancotan(dx)
    
    # dxH derivtes wrt p, it is a speed of GD, it is put in the tan element
    dp = HamDer.dpH(Mod)
    dp.exchange_tan_cotan()
    dP_dp = fill_Vector_from_tancotan(dp)
    
    dP = fill_Vector_from_tancotan(cgrad)
    dP += dP_dx + dP_dp
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
    
    print("Energy = {0:10.3e}".format(hamval + varcost), end=' ; ')
    print("Hamiltonian = {0:10.3e}".format(hamval), end=' ; ')
    print("Data Attachment = {0:10.3e}".format(varcost), end='\n')
    return hamval + varcost
