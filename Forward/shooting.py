#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:17:05 2018

@author: barbaragris
"""

import Forward.Hamiltonianderivatives as HamDer

def forward_step(Mod, step):
    """
    Supposes that Geodesic controls are computed
    """
    dxH = HamDer.dxH(Mod)
    dpH = HamDer.dpH(Mod)
    dxH.mult_Cot_scal(step)
    dpH.mult_Cot_scal(step)
    Mod.add_cot(dxH)
    Mod.add_cot(dpH)


def forward_step_rk2(Mod, step):
    Mod1 = Mod.copy_full()
    forward_step(Mod1, 0.5*step)
    
    Mod1.update()
    Mod1.GeodesicControls_curr(Mod1.GD)
    
    dxH = HamDer.dxH(Mod1)
    dpH = HamDer.dpH(Mod1)
    
    dxH.mult_Cot_scal(step)
    dpH.mult_Cot_scal(step)
    Mod.add_cot(dxH)
    Mod.add_cot(dpH)

def shooting(Mod, N_int):
    """
    Supposes that Cot is filled
    
    """
    
    step = 1./N_int
    
    for i in range(N_int):
        Mod.update()
        Mod.GeodesicControls_curr(Mod.GD)
        forward_step_rk2(Mod, step)
        
    return Mod

def shooting_traj(Mod, N_int):
    """
    Supposes that Cot is filled
    
    """
    
    step = 1./N_int
    GD_list = [Mod.GD.copy_full()]
    Cont_list = []
    for i in range(N_int):
        Mod.update()
        Mod.GeodesicControls_curr(Mod.GD)
        forward_step_rk2(Mod, step)
        GD_list.append(Mod.GD.copy_full())
        Cont_list.append(Mod.Cont)
        
    return GD_list, Cont_list

