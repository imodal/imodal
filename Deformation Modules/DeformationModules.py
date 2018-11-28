#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:03:15 2018

@author: barbaragris
"""

import numpy as np
import GeometricalDescriptors.GeometricalDescriptors as GeoDescr


class DeformationModule(object):
    """
    Abstract class for deformation module
    
    
    Attributes: type of GD
                dimension of controls ?
                
                
    Methods: 
        -- GeodesicControls (v_s, m) where v_s is in V^\ast, m a GD,
              returns the corresponding geodesic control
        -- FieldGenerator (m, h) where m is a GD, h a control. It returns
              a structured field in V
        -- Cost (m, h) where m is a GD, h a control. It returns the cost
              (scalar)
        -- DerCost (m, h) where m is a GD, h a control. It returns the
              derivative of the cost wrt m
    
    """
    
    
    


class ElasticOrderO(DeformationModule):
    """
     Elastic module of order 0
    """
    
    
    def __init__(self, sigma):
        self.sigma = sigma
        self.GD = GeoDescr.GD_landmark()
        self.SKS = np.empty([0])
        
    
    
    def GeodesicControls(vs, m):
        