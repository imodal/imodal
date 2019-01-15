#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:05:41 2018

@author: barbaragris
"""
import numpy as np
from scipy.linalg import solve
from src import field_structures as fields, pairing_structures as pair, functions_eta as fun_eta, modules_operations as mod


def my_add_der(der0, der1):
    """ add der
    """
    der = dict()
    if '0' in der0:
        der['0']=[]
        for i in range(len(der0['0'])):
            (x,dxe0), (x,dxe1) = der0['0'][i], der1['0'][i]
            der['0'].append((x,dxe0+dxe1))
    if 'x,R' in der0:
        der['x,R']=[]
        for i in range(len(der0['x,R'])):
            ((x,R),(dxe0,dRe0)),((x,R),(dxe1,dRe1)) \
                = der0['x,R'][i], der1['x,R'][i]
            der['x,R'].append(((x,R),(dxe0+dxe1,dRe0+dRe1)))
    return der


def my_new_ham(Mod0,Mod1,Cot):
    ham = 0.
    mod.my_mod_update(Mod0), mod.my_mod_update(Mod1)
    Vs0 = {'0':[(Mod0['0'],Mod0['mom'])], 'sig':Mod0['sig']}
    (x,R) = Mod1['x,R']
    Vs1 = {'p':[(x,Mod1['mom'])], 'sig':Mod1['sig']}
    ham = pair.my_CotDotV(Cot,Vs0) + pair.my_CotDotV(Cot,Vs1) - Mod0['cost'] - Mod1['cost']
    return ham

  
def my_dxH(Mod0, Mod1, Cot):
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    Vs0 = {'0':[(Mod0['0'],Mod0['mom'])], 'sig':sig0}
    (x,R), P = Mod1['x,R'], Mod1['mom']
    Vs1 = {'p':[(x,P)], 'sig':sig1}  
    
    # derivatives with respect to the end conditions (der of \xi_m)
    cder = my_add_der(pair.my_dCotDotV(Cot,Vs0),pair.my_dCotDotV(Cot,Vs1))
    
    
    # derivatives with respect to the initial conditions (KS_m^\ast)
    Vsr1 = fields.my_CotToVs(Cot, sig1)
    der = pair.my_pSmV(Vs1,Vsr1,1)
    dx1H = der['p'][0][1]
    # 
    
    # derivation of the \zeta_x0 part
    Vsr0 = fields.my_CotToVs(Cot, sig0)
    der = pair.my_pSmV(Vs0,Vsr0,1)
    dx0H = der['0'][0][1]
    
    # derivative of cost0, use the symmetric properties with S_m0
    der = pair.my_pSmV(Vs0,Vs0,1) # to take into acc. the cost var.
    dx0H += -Mod0['coeff']*der['0'][0][1]
    
    
    # derivatives with respect to the operator \lambda(m,h)
    ## Sm1 vsr1
    dv = fields.my_VsToV(Vsr1,x,1)
    dv_sym = (dv + np.swapaxes(dv,1,2))/2
    S = np.tensordot(dv_sym,fun_eta.my_eta())
    
    ## (S1m1Sm1^\ast)^{-1} Sm1 K1 \xi^\ast_m p
    tlam = solve(Mod1['SKS'], S.flatten(), sym_pos = True)
    ## reshape so that it is a symmetric matrix
    tP = np.tensordot(tlam.reshape(S.shape),fun_eta.my_eta().transpose(), axes = 1) 
    
    ## tlam and Sm seen as dual of vector field
    tVs = {'p':[(x,tP)], 'sig':sig1}
    
    ## computes derivative wrt m in (Sm1 K1 Sm1^ast) ^{-1}
    der = pair.my_pSmV(tVs,Vs1,1)
    dx1H += - der['p'][0][1]
    der = pair.my_pSmV(Vs1,tVs,1)
    dx1H += - der['p'][0][1]
    
    # C'était ici !!!!!
    h, C = Mod1['h'], Mod1['C']
    # Amh = np.tensordot(Mod1['Amh'].reshape(S.shape),my_eta().transpose(), axes = 1)
    
    Ptmp = (tP - Mod1['coeff']*P) # takes into acc. the cost variation in x1
    dRH = 2*np.asarray([np.dot(np.dot(Ptmp[i],R[i]),np.diag(np.dot(C[i],h)))
        for i in range(x.shape[0])])
    
    der = pair. my_pSmV(Vs1,Vs1,1)
    dx1H += Mod1['coeff']*der['p'][0][1]
    
    
    # put everything in cder
    (x,dxe) = cder['0'][0]
    cder['0'][0]=(x,dxe+dx0H)
    ((x,R),(dxe,dRe)) = cder['x,R'][0]
    cder['x,R'][0] = ((x,R),(dxe+dx1H, dRe+dRH))
    
    return cder

def my_dpH(Mod0, Mod1, Cot):
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    
    Vs0 = {'0':[(x0,Mod0['mom'])], 'sig':sig0}
    
    P = Mod1['mom']
    Vs1 = {'p':[(x1,P)], 'sig':sig1}
    
    vx0 = fields.my_VsToV(Vs0,x0,0)+ fields.my_VsToV(Vs1,x0,0)
    vx1 = fields.my_VsToV(Vs0,x1,0)+ fields.my_VsToV(Vs1,x1,0)
    vxs = fields.my_VsToV(Vs0,xs,0)+ fields.my_VsToV(Vs1,xs,0)
    dv  = fields.my_VsToV(Vs0,x1,1)+ fields.my_VsToV(Vs1,x1,1)
    S  = (dv - np.swapaxes(dv,1,2))/2
    vR = np.asarray([ np.dot(S[i],R[i]) for i in range(x1.shape[0])]) 
    
    derp = {'0':[(x0,vx0), (xs,vxs)], 'x,R':[((x1,R), (vx1, vR))]}
    return derp
