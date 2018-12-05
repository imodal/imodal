#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 20:02:02 2018

@author: barbaragris
"""

def dxH(Mod):
    """
    Supposes that Mod is updated, in particluar Mod.Cot and 
    Mod.h (with geodesic control) is filled
    """
    
    v = Mod.field_generator_curr()
    
    ## derivation wrt GD in Xi_GD
    der =  Mod.GD.dCotDotV(v)
    
    ## derivation wrt GD in Zeta_GD
    der_fieldgen = Mod.cot_to_innerprod_curr(Mod.GD, 1)
    der.add_cot(der_fieldgen.Cot)
    
    ## derivation of the cost
    der_cost = Mod.DerCost_curr()
    der.add_cot(der_cost.Cot)
    
    out = der.copy()
    # need to exchange values of GD and mom(=0) in cot
    if '0' in der.Cot:
        for (x,p) in der.Cot['0']:
            out.Cot['0'].append( (p, -x) )
    if 'x,R' in der.Cot:
        for ((x, R), (p, P)) in der.Cot['x,R']:
            out.Cot['x,R'].append( ( (p,P), (-x,-R) ) )
    
    out.updatefromCot()
    return out

def dpH(Mod):
    """
    Supposes that Mod is updated, in particluar Mod.Cot and 
    Mod.h (with geodesic control) is filled
    The derivation wrt p is the application of the field to GD
    """
    
    v = Mod.field_generator_curr()
    appli = Mod.GD.Ximv(v)
    appli.updatefromCot()          
    return appli
    
    
def Ham(Mod):
    """
    Supposes that Geodesic controls have been computed and Mod updated
    """
    
    v = Mod.field_generator_curr()
    Mod.Cost_curr()
    return Mod.GD.inner_prod_v(v) - Mod.cost






  