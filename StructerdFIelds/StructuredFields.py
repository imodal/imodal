#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 06:29:43 2018

@author: barbaragris
"""




class StructuredField(object):
            
    """
    Abstract class for structured field
     type '0', 'p' or 'm'.
    Methods : 
        
          -- p_Ximv (vsl, vsr, j) with vsl and vsr 2 structured field, j an
               integer. If j=0 it returns (p | Xi_m (vsr)) where vsl is 
               parametrized by (m,p). If j=1 it returns the derivative wrt m
               (Ou Cot à la place de vsl ??)
                   
       """