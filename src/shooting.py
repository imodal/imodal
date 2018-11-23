#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:06:44 2018

@author: barbaragris
"""

import numpy as np
from scipy.linalg import solve
from implicitmodules.src import constraints_functions as con_fun, field_structures as fields, hamiltonian_derivatives as ham, \
    useful_functions as fun, modules_operations as modop, functions_eta as fun_eta


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




def my_nforward(Cot, dMod0,dMod1,dCot,Dt):
    """ similar than my_forward but able to compute z + Dt X_H(z') where z is
    determined by Cot and z' by dMod0, dMod1, dCot. Useful for RK2 shooting
    """
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = dMod0['sig'], dMod1['sig']
    
    # updating the co-state
    derx = ham.my_dxH(dMod0, dMod1, dCot)
    np0 = p0 - Dt*derx['0'][0][1]
    nps = ps - Dt*derx['0'][1][1]
    np1 = p1 - Dt*derx['x,R'][0][1][0]
    nPR = PR - Dt*derx['x,R'][0][1][1]
    
    # updating the state
    derp = ham.my_dpH(dMod0, dMod1, dCot)
    nx0 = x0 + Dt*derp['0'][0][1]
    nxs = xs + Dt*derp['0'][1][1]
    nx1 = x1 + Dt*derp['x,R'][0][1][0]
    nR  = R  + Dt*derp['x,R'][0][1][1]
            
    # updating Cot
    nCot = {'0':[(nx0,np0), (nxs,nps)], 'x,R':[((nx1,nR),(np1, nPR))]}
    
    # updating h0
    nMod0 = modop.my_init_from_mod(dMod0)
    nMod0['0'] = nx0    
    nMod0['SKS'] = fun.my_new_SKS(nMod0)
    nMod0['mom'] = solve(nMod0['coeff']*nMod0['SKS'],
       fields.my_VsToV(fields.my_CotToVs(nCot,sig0),nx0,0).flatten(),
        sym_pos = True).reshape(x0.shape)
    modop.my_mod_update(nMod0) # compute cost0
    
    # updating h1
    nMod1 = modop.my_init_from_mod(dMod1)
    nMod1['x,R'] = (nx1, nR)
    nMod1['SKS'] = fun.my_new_SKS(nMod1)
    dv = fields.my_VsToV(fields.my_CotToVs(nCot,sig1),nx1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,fun_eta.my_eta())
    tlam = solve(nMod1['coeff']*nMod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = con_fun.my_new_AmKiAm(nMod1)
    nMod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    modop.my_mod_update(nMod1) # will compute the new lam, mom and cost
    
    return (nMod0, nMod1, nCot)
    
def my_forward(Mod0,Mod1,Cot,Dt):        
    
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    
    # updating the co-state
    derx = ham.my_dxH(Mod0, Mod1, Cot)
    np0 = p0 - Dt*derx['0'][0][1]
    nps = ps - Dt*derx['0'][1][1]
    np1 = p1 - Dt*derx['x,R'][0][1][0]
    nPR = PR - Dt*derx['x,R'][0][1][1]
    
    # updating the state
    derp = ham.my_dpH(Mod0, Mod1, Cot)
    nx0 = x0 + Dt*derp['0'][0][1]
    nxs = xs + Dt*derp['0'][1][1]
    nx1 = x1 + Dt*derp['x,R'][0][1][0]
    nR  = R  + Dt*derp['x,R'][0][1][1]
            
    # updating Cot
    nCot = {'0':[(nx0,np0), (nxs,nps)], 'x,R':[((nx1,nR),(np1, nPR))]}
    
    # updating h0
    nMod0 = modop.my_init_from_mod(Mod0)
    nMod0['0'] = nx0    
    nMod0['SKS'] = fun.my_new_SKS(nMod0)
    nMod0['mom'] = solve(nMod0['coeff']*nMod0['SKS'],
        fields.my_VsToV(fields.my_CotToVs(nCot,sig0),nx0,0).flatten(),
        sym_pos = True).reshape(x0.shape)
    modop.my_mod_update(nMod0) # compute cost0
    
    # updating h1
    nMod1 = modop.my_init_from_mod(Mod1)
    nMod1['x,R'] = (nx1, nR)
    nMod1['SKS'] = fun.my_new_SKS(nMod1)
    dv = fields.my_VsToV(fields.my_CotToVs(nCot,sig1),nx1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,fun_eta.my_eta())
    tlam = solve(nMod1['coeff']*nMod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = con_fun.my_new_AmKiAm(nMod1)
    nMod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    modop.my_mod_update(nMod1) # will compute the new lam, mom and cost
    
    return (nMod0, nMod1, nCot)


  
    
