#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 14:47:03 2018

@author: barbaragris
"""

from src import field_structures as fields
from src import pairing_structures as pair
import StructerdFIelds.StructuredFields as stru_fie


def CotToVs_class(GD, sig):
    dic = fields.my_CotToVs(GD.Cot, sig)
    vs_list = []
    if '0' in dic:
        for (x,p) in dic['0']:
            vtmp = stru_fie.StructuredField_0(sig, GD.dim)
            vtmp.fill_fieldparam((x,p)) 
            vs_list.append(vtmp.copy_full())
            
    if 'm' in dic:
        for (x,P) in dic['m']:
            vtmp = stru_fie.StructuredField_m(sig, GD.dim)
            vtmp.fill_fieldparam((x,P)) 
            vs_list.append(vtmp.copy_full())
            
    v_sum = stru_fie.sum_structured_fields(vs_list) 
    v_sum.dic['sig'] = sig
    return v_sum


def VsToV_class(v, pts, j):
    return fields.my_VsToV(v.dic, pts, j)


def dCotdotV_class(Cot, vs):
    return pair.my_dCotDotV(Cot, vs.dic)



