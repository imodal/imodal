# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 13:26:02 2019

@author: gris
"""

class GeometricalDescriptors(object):
    
    """
    Abstract class for geometrical descriptors, needs to have the 
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
    