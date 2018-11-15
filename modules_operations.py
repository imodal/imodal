#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:11:47 2018

@author: barbaragris
"""

import numpy as np
from scipy.linalg import solve
import functions_eta as fun_eta
import useful_functions as fun
import constraints_functions as con_fun

def my_mod_update(Mod): 
    if not 'SKS' in Mod:
        Mod['SKS'] = fun.my_new_SKS(Mod)
        
    if '0' in Mod:        
        (x,p) = (Mod['0'], Mod['mom'].flatten())
        Mod['cost'] = Mod['coeff']*np.dot(p,np.dot(Mod['SKS'],p))/2
    
    if 'x,R' in Mod:
        (x,R) = Mod['x,R']
        N = x.shape[0]
        Mod['Amh'] = con_fun.my_Amh(Mod,Mod['h']).flatten()
        Mod['lam'] = solve(Mod['SKS'], Mod['Amh'], sym_pos = True)
        Mod['mom'] = np.tensordot(Mod['lam'].reshape(N,3),
            fun_eta.my_eta().transpose(), axes =1)
        Mod['cost'] = Mod['coeff']*np.dot(Mod['Amh'],Mod['lam'])/2
    return
    


def my_init_from_mod(Mod):
    if '0' in Mod:
        nMod = {'sig': Mod['sig'], 'coeff': Mod['coeff']}
    
    if 'x,R' in Mod:
        nMod = {'sig':Mod['sig'], 'C':Mod['C'], 'coeff': Mod['coeff']}
        if 'nu' in Mod:
            nMod['nu'] = Mod['nu']
    return nMod
