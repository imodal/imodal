# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:07:02 2019

@author: gris
"""

import src.Forward.Hamiltonianderivatives as HamDer


def forward_step(Mod, step):
    """
    Supposes that Geodesic controls are computed
    """
    # dxH derivtes wrt x, it is put in the cotan element
    dxH = HamDer.dxH(Mod)
    
    # dxH derivtes wrt p, it is a speed of GD, it is put in the tan element
    dpH = HamDer.dpH(Mod)
    
    dxH.mult_cotan_scal(step)
    #dxH.mult_GD_scal(step)
    dpH.mult_tan_scal(step)
    #dpH.mult_tan_scal(step)
    
    Mod.add_cotan(dxH)
    Mod.add_speedGD(dpH)
    #Mod.add_GD(dxH)
    #Mod.add_GD(dpH)


def forward_step_rk2(Mod, step):
    Mod1 = Mod.copy_full()
    forward_step(Mod1, 0.5*step)
    
    Mod1.update()
    Mod1.GeodesicControls_curr(Mod1.GD)
    
    dxH = HamDer.dxH(Mod1)
    dpH = HamDer.dpH(Mod1)
    
    dxH.mult_cotan_scal(step)
    #dxH.mult_GD_scal(step)
    dpH.mult_tan_scal(step)
    #dpH.mult_tan_scal(step)
    
    Mod.add_cotan(dxH)
    Mod.add_speedGD(dpH)
    
    return Mod1



def shooting(Mod, N_int):
    """
    Supposes that Cot is filled
    
    """
    
    step = 1./N_int
    
    for i in range(N_int):
        Mod.update()
        Mod.GeodesicControls_curr(Mod.GD)
        Modtmp = forward_step_rk2(Mod, step)
        
    return Mod
    


def shooting_traj(Mod, N_int):
    """
    Supposes that Cot is filled
    
    """
    
    step = 1./N_int
    Mod.update()
    Mod.GeodesicControls_curr(Mod.GD)
    Mod_list = [Mod.copy_full()]
    #GD_list = [Mod.GD.copy_full()]
    #Cont_list = []
    for i in range(N_int):
        Mod.update()
        Mod.GeodesicControls_curr(Mod.GD)
        Modtmp = forward_step_rk2(Mod, step)
        Mod_list.append(Modtmp.copy_full())
        Mod_list.append(Mod.copy_full())
        
    return Mod_list






