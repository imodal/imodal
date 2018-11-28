#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:39:57 2018

@author: barbaragris
"""

import field_structures as fields
import pairing_structures as pair

class GeometricalDescriptors(object):
    
    """
    Abstract class for geometrica descriptors, needs to have the 
      following methods:
          
          -- Cot_to_Vs (m, p, s) with m value of GD, p a cotangent element and
                s a scale (parametrizing a RKHS). It returns a structured dual
                field
         
          -- Ximv (m, v) with m value of GD, v a structured field.
                Returns the application of m on v
            
          -- dCotDotV(m, p, vs) with m value of GD, p a cotangent element and
                v a structured field. 
                Returns derivative of (p Ximv(m,v)) wrt m
    
    """
    def __init__(self):
        pass
    
       
        
class GD_landmark(GeometricalDescriptors):
    def __init(self):
        self.Cot = {'0':[]}
        
    
    def Cot_to_Vs(self, m, p, s):
        self.Cot['0'] = [(m,p)]
        return fields.my_CotToVs(self.Cot, s)
    
    def Ximv(self, m, v):
        return fields.my_VsToV(v, m, 0)
    
    def dCotDotV(self, m, p, vs):
        self.Cot['0'] = [(m,p)]
        return pair.my_dCotDotV(self.Cot, vs)