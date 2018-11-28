#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:06:44 2018

@author: barbaragris
"""

import numpy as np
from implicitmodules.src import hamiltonian_derivatives as ham, modules_operations as modop, useful_functions as utils

import implicitmodules.src.data_attachment.varifold as var

def my_fun(P0, *args):
    """ compute the shooting and return the total cost
    ie H(X0,P0)+ lam_var \| X1-Xtarg\|^2_var
    args = (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N)
    """
    
    (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N) = args
    (x0, xs, (x1, R)) = utils.my_splitX(X, nX)
    Mod0 = {'0': x0, 'sig': sig0, 'coeff': coeff0}
    Mod1 = {'x,R': (x1, R), 'sig': sig1, 'C': C, 'coeff': coeff1, 'nu': nu}
    Cot = utils.my_CotFromXP(X, P0, nX)
    Mod0 = modop.my_mod_init_from_Cot(Mod0, Cot)
    Mod1 = modop.my_mod_init_from_Cot(Mod1, Cot)
    Step = (Mod0, Mod1, Cot)
    h = 1. / N
    Traj = [Step]
    # rk2 steps
    for i in range(N):
        print(i)
        dStep = my_forward(Step[0], Step[1], Step[2], h / 2.)
        Traj.append(dStep)
        Step = my_nforward(Step[2], dStep[0], dStep[1], dStep[2], h)
        Traj.append(Step)
    cCot = Traj[-1][2]
    xsf = cCot['0'][1][0]
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    varcost = lam_var * varcost
    hamval = ham.my_new_ham(Step[0], Step[1], Step[2])
    print("ham     = {0:10.3e}".format(hamval))
    print("varcost = {0:10.3e}".format(varcost))
    print("totener = {0:10.3e}".format(hamval + varcost))
    return hamval + varcost


def my_fun_Traj(P0, *args):
    """ compute the shooting and return the total cost
    ie H(X0,P0)+ lam_var \| X1-Xtarg\|^2_var
    args = (X, nX, sig0, sig1, coeff0, coeff1, C, nun, xst, lam_var, sig_var, N)
    """
    
    (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N) = args
    (x0, xs, (x1, R)) = utils.my_splitX(X, nX)
    Mod0 = {'0': x0, 'sig': sig0, 'coeff': coeff0}
    Mod1 = {'x,R': (x1, R), 'sig': sig1, 'C': C, 'coeff': coeff1, 'nu': nu}
    Cot = utils.my_CotFromXP(X, P0, nX)
    Mod0 = modop.my_mod_init_from_Cot(Mod0, Cot)
    Mod1 = modop.my_mod_init_from_Cot(Mod1, Cot)
    Step = (Mod0, Mod1, Cot)
    h = 1. / N
    Traj = [Step]
    # rk2 steps
    for i in range(N):
        print(i)
        dStep = my_forward(Step[0], Step[1], Step[2], h / 2.)
        Traj.append(dStep)
        Step = my_nforward(Step[2], dStep[0], dStep[1], dStep[2], h)
        Traj.append(Step)
    cCot = Traj[-1][2]
    xsf = cCot['0'][1][0]
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    varcost = lam_var * varcost
    hamval = ham.my_new_ham(Step[0], Step[1], Step[2])
    print("ham     = {0:10.3e}".format(hamval))
    print("varcost = {0:10.3e}".format(varcost))
    print("totener = {0:10.3e}".format(hamval + varcost))
    return Traj


def my_fd_shoot(Mod0,Mod1,Cot,N):
    h = 1./N
    Step = (Mod0, Mod1, Cot)
    Traj =[Step]
    # rk2 steps
    for i in range(N):
        print(i)
        dStep = my_forward(Step[0], Step[1], Step[2], h/2.)
        Traj.append(dStep)
        Step =  my_nforward(Step[2], dStep[0], dStep[1], dStep[2],h)
        Traj.append(Step)        
    return Traj


def my_nforward(Cot, dMod0, dMod1, dCot, Dt):
    """ similar than my_forward but able to compute z + Dt X_H(z') where z is
    determined by Cot and z' by dMod0, dMod1, dCot. Useful for RK2 shooting
    """
    [(x0, p0), (xs, ps)] = Cot['0']
    [((x1, R), (p1, PR))] = Cot['x,R']
    
    # updating the co-state
    derx = ham.my_dxH(dMod0, dMod1, dCot)
    np0 = p0 - Dt * derx['0'][0][1]
    nps = ps - Dt * derx['0'][1][1]
    np1 = p1 - Dt * derx['x,R'][0][1][0]
    nPR = PR - Dt * derx['x,R'][0][1][1]
    
    # updating the state
    derp = ham.my_dpH(dMod0, dMod1, dCot)
    nx0 = x0 + Dt * derp['0'][0][1]
    nxs = xs + Dt * derp['0'][1][1]
    nx1 = x1 + Dt * derp['x,R'][0][1][0]
    nR = R + Dt * derp['x,R'][0][1][1]
    
    # updating Cot
    nCot = {'0': [(nx0, np0), (nxs, nps)], 'x,R': [((nx1, nR), (np1, nPR))]}
    
    # updating Mod0 (especially h0)
    nMod0 = modop.my_mod_init_from_Cot(dMod0, nCot)
    
    # updating Mod1 (espacially h1)
    nMod1 = modop.my_mod_init_from_Cot(dMod1, nCot)
    
    return (nMod0, nMod1, nCot)


def my_forward(Mod0, Mod1, Cot, Dt):
    """ Compute z + Dt X_H(z) where z is defined through Mod0, Mod1 and Cot """
    [(x0, p0), (xs, ps)] = Cot['0']
    [((x1, R), (p1, PR))] = Cot['x,R']
    
    # updating the co-state
    derx = ham.my_dxH(Mod0, Mod1, Cot)
    np0 = p0 - Dt * derx['0'][0][1]
    nps = ps - Dt * derx['0'][1][1]
    np1 = p1 - Dt * derx['x,R'][0][1][0]
    nPR = PR - Dt * derx['x,R'][0][1][1]
    
    # updating the state
    derp = ham.my_dpH(Mod0, Mod1, Cot)
    nx0 = x0 + Dt * derp['0'][0][1]
    nxs = xs + Dt * derp['0'][1][1]
    nx1 = x1 + Dt * derp['x,R'][0][1][0]
    nR = R + Dt * derp['x,R'][0][1][1]
    
    # updating Cot
    nCot = {'0': [(nx0, np0), (nxs, nps)], 'x,R': [((nx1, nR), (np1, nPR))]}
    
    # updating Mod0 (especially h0)
    nMod0 = modop.my_mod_init_from_Cot(Mod0, nCot)
    
    # updating Mod1 (espacially h1)
    nMod1 = modop.my_mod_init_from_Cot(Mod1, nCot)
    
    return (nMod0, nMod1, nCot)


def my_sub_bckwd(Mod0, Mod1, Cot, grad, my_eps):
    """ my_sub_bckwd compute an elementary backward step associated with the
    hamiltonian flow. grad is a dictionary with the following form
    grad = {'0':[(dx0G, dp0G),(dxsG, dpsG)], 'x,R':[((dx1G, dRG), (dp1G, dpRG))]}
    """
    
    # der ={'0':[(x0,dx0H), (xs,dxsH)], 'x,R':[((x1,R),(dx1H,dRH))]}
    
    [(x0, p0), (xs, ps)] = Cot['0']
    [((x1, R), (p1, PR))] = Cot['x,R']
    
    # computing x - eps \nabla_pG
    nx0 = x0 - my_eps * grad['0'][0][1]
    nxs = xs - my_eps * grad['0'][1][1]
    nx1 = x1 - my_eps * grad['x,R'][0][1][0]
    nR = R - my_eps * grad['x,R'][0][1][1]
    
    # updating p + eps\nabla_xG
    np0 = p0 + my_eps * grad['0'][0][0]
    nps = ps + my_eps * grad['0'][1][0]
    np1 = p1 + my_eps * grad['x,R'][0][0][0]
    nPR = PR + my_eps * grad['x,R'][0][0][1]
    
    nCot = {'0': [(nx0, np0), (nxs, nps)], 'x,R': [((nx1, nR), (np1, nPR))]}
    
    # creating the new mod
    nMod0 = modop.my_mod_init_from_Cot(Mod0, nCot)
    nMod1 = modop.my_mod_init_from_Cot(Mod1, nCot)
    
    # computing x + eps \nabla_pG
    bx0 = x0 + my_eps * grad['0'][0][1]
    bxs = xs + my_eps * grad['0'][1][1]
    bx1 = x1 + my_eps * grad['x,R'][0][1][0]
    bR = R + my_eps * grad['x,R'][0][1][1]
    
    # updating p - eps\nabla_xG
    bp0 = p0 - my_eps * grad['0'][0][0]
    bps = ps - my_eps * grad['0'][1][0]
    bp1 = p1 - my_eps * grad['x,R'][0][0][0]
    bPR = PR - my_eps * grad['x,R'][0][0][1]
    
    nCot = {'0': [(nx0, np0), (nxs, nps)], 'x,R': [((nx1, nR), (np1, nPR))]}
    bCot = {'0': [(bx0, bp0), (bxs, bps)], 'x,R': [((bx1, bR), (bp1, bPR))]}
    
    # creating the new mod
    nMod0 = modop.my_mod_init_from_Cot(Mod0, nCot)
    nMod1 = modop.my_mod_init_from_Cot(Mod1, nCot)
    bMod0 = modop.my_mod_init_from_Cot(Mod0, bCot)
    bMod1 = modop.my_mod_init_from_Cot(Mod1, bCot)
    
    # Computing dF^*(\nabla G) for F the hamiltonian flow
    ngrad = dict(grad)
    
    # Computing (\nabla xH(z+eps J grad) - \nabla_x H(z-eps))/(2*eps)
    derx, nderx = ham.my_dxH(Mod0, Mod1, Cot), ham.my_dxH(nMod0, nMod1, nCot)
    bderx = ham.my_dxH(bMod0, bMod1, bCot)
    
    my_eps = 2 * my_eps
    dx0G = (nderx['0'][0][1] - bderx['0'][0][1]) / my_eps
    dxsG = (nderx['0'][1][1] - bderx['0'][1][1]) / my_eps
    dx1G = (nderx['x,R'][0][1][0] - bderx['x,R'][0][1][0]) / my_eps
    dRG = (nderx['x,R'][0][1][1] - bderx['x,R'][0][1][1]) / my_eps
    
    # Computing (\nabla pH(z+eps J grad) - \nabla_p H(z))/eps
    derp, nderp = ham.my_dpH(Mod0, Mod1, Cot), ham.my_dpH(nMod0, nMod1, nCot)
    bderp = ham.my_dpH(bMod0, bMod1, bCot)
    
    dp0G = (nderp['0'][0][1] - bderp['0'][0][1]) / my_eps
    dpsG = (nderp['0'][1][1] - bderp['0'][1][1]) / my_eps
    dp1G = (nderp['x,R'][0][1][0] - bderp['x,R'][0][1][0]) / my_eps
    dpRG = (nderp['x,R'][0][1][1] - bderp['x,R'][0][1][1]) / my_eps
    
    ngrad = {'0': [(dx0G, dp0G), (dxsG, dpsG)],
             'x,R': [((dx1G, dRG), (dp1G, dpRG))]}
    
    return ngrad


def my_add_grad(grada, gradb):
    ngrad = dict(grada)
    [(dx0Ga, dp0Ga), (dxsGa, dpsGa)] = grada['0']
    [((dx1Ga, dRGa), (dp1Ga, dpRGa))] = grada['x,R']
    [(dx0Gb, dp0Gb), (dxsGb, dpsGb)] = gradb['0']
    [((dx1Gb, dRGb), (dp1Gb, dpRGb))] = gradb['x,R']
    
    (ndx0G, ndp0G, ndxsG, ndpsG) = (dx0Ga + dx0Gb, dp0Ga + dp0Gb,
                                    dxsGa + dxsGb, dpsGa + dpsGb)
    (ndx1G, ndRG, ndp1G, ndpRG) = (dx1Ga + dx1Gb, dRGa + dRGb,
                                   dp1Ga + dp1Gb, dpRGa + dpRGb)
    ngrad = {'0': [(ndx0G, ndp0G), (ndxsG, ndpsG)],
             'x,R': [((ndx1G, ndRG), (ndp1G, ndpRG))]}
    return ngrad


def my_bck_shoot(Traj, grad, my_eps):
    Traj.reverse()
    N = int((len(Traj) - 1) / 2)
    h = 1. / N
    count, cgrad = 0, grad.copy()
    for i in range(N):
        # print(i)
        count += 1
        dStep = Traj[count]
        rnp = my_sub_bckwd(dStep[0], dStep[1], dStep[2], cgrad, my_eps)
        rnp = utils.my_mult_grad(rnp, h)
        count += 1
        Step = Traj[count]
        rn = my_sub_bckwd(Step[0], Step[1], Step[2], rnp, my_eps)
        rn = utils.my_mult_grad(rn, h / 2)
        cgrad = my_add_grad(my_add_grad(cgrad, rnp), rn)
    Traj.reverse()
    return cgrad


def my_jac(P0, *args):
    """ jacobian associated with my_fun
    """
    (X, nX, sig0, sig1, coeff0, coeff1, C, nu, xst, lam_var, sig_var, N) = args
    (x0, xs, (x1, R)) = utils.my_splitX(X, nX)
    Mod0 = {'0': x0, 'sig': sig0, 'coeff': coeff0}
    Mod1 = {'x,R': (x1, R), 'sig': sig1, 'C': C, 'coeff': coeff1, 'nu': nu}
    Cot = utils.my_CotFromXP(X, P0, nX)
    Mod0 = modop.my_mod_init_from_Cot(Mod0, Cot)
    Mod1 = modop.my_mod_init_from_Cot(Mod1, Cot)
    Step = (Mod0, Mod1, Cot)
    h = 1. / N
    Traj = [Step]
    # rk2 steps
    for i in range(N):
        # print(i)
        dStep = my_forward(Step[0], Step[1], Step[2], h / 2.)
        Traj.append(dStep)
        Step = my_nforward(Step[2], dStep[0], dStep[1], dStep[2], h)
        Traj.append(Step)
    cCot = Traj[-1][2]
    xsf = cCot['0'][1][0]
    (varcost, dxvarcost) = var.my_dxvar_cost(xsf, xst, sig_var)
    # varcost = lam_var * varcost
    dxvarcost = lam_var * dxvarcost
    # hamval = ham.my_new_ham(Step[0], Step[1], Step[2])
    # print("hamval     = {0:10.3e}".format(hamval))
    # print("varcost = {0:10.3e}".format(varcost))
    # print("totener = {0:10.3e}".format(hamval+varcost))
    grad = {'0': [(np.zeros(x0.shape), np.zeros(x0.shape)),
                  (dxvarcost, np.zeros(xs.shape))],
            'x,R': [((np.zeros(x1.shape), np.zeros(R.shape)),
                     (np.zeros(x1.shape), np.zeros(R.shape)))]}
    ngrad = my_bck_shoot(Traj, grad, 0.00001)
    derp = ham.my_dpH(Mod0, Mod1, Cot)
    # derp = {'0':[(x0,vp0), (xs,vps)], 'x,R':[((x1,R), (vp1, vpR))]}
    nhgrad = {'0': [(np.zeros(x0.shape), derp['0'][0][1]),
                    (np.zeros(xs.shape), derp['0'][1][1])],
              'x,R': [((np.zeros(x1.shape), np.zeros(R.shape)),
                       (derp['x,R'][0][1][0], derp['x,R'][0][1][1]))]}
    totgrad = my_add_grad(ngrad, nhgrad)
    dP0J = np.concatenate([totgrad['0'][0][1].flatten(),
                           totgrad['0'][1][1].flatten(), totgrad['x,R'][0][1][0].flatten(),
                           totgrad['x,R'][0][1][1].flatten()])
    return dP0J

