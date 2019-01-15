#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:15:18 2018

@author: barbaragris
"""

import numpy as np
from src import field_structures as fields


def my_CotDotV(Cot, Vs):
    """  This function computes product betwwen a covector h  of the and a v
    field  such as (h|v.x) or (h[v.(x,R))
    """
    out = 0.
    
    if '0' in Cot:
        for (x, p) in Cot['0']:  # Landmark point
            v = fields.my_VsToV(Vs, x, 0)
            out += np.sum([np.dot(p[i], v[i]) for i in range(x.shape[0])])
    
    if 'x,R' in Cot:
        for ((x, R), (p, P)) in Cot['x,R']:
            v, dv = fields.my_VsToV(Vs, x, 0), fields.my_VsToV(Vs, x, 1)
            skew_dv = (dv - np.swapaxes(dv, 1, 2)) / 2
            out += np.sum([np.dot(p[i], v[i]) +
                           np.tensordot(P[i], np.dot(skew_dv[i], R[i]))
                           for i in range(x.shape[0])])
    return out


def my_dCotDotV(Cot, Vs):
    """  This function computes the derivative with respect to the parameter
    of product between a covector h a v field  such as (h|v.x) or
    (h|v.(x,R))
    """
    der = dict()
    
    if '0' in Cot:
        der['0'] = []
        for (x, p) in Cot['0']:  # Landmark point
            dv = fields.my_VsToV(Vs, x, 1)
            der['0'].append((x, np.asarray([np.dot(p[i], dv[i])
                                            for i in range(x.shape[0])])))
    
    if 'x,R' in Cot:
        der['x,R'] = []
        for ((x, R), (p, P)) in Cot['x,R']:
            dv, ddv = fields.my_VsToV(Vs, x, 1), fields.my_VsToV(Vs, x, 2)
            
            skew_dv = (dv - np.swapaxes(dv, 1, 2)) / 2
            skew_ddv = (ddv - np.swapaxes(ddv, 1, 2)) / 2
            
            dedx = np.asarray([np.dot(p[i], dv[i]) + np.tensordot(P[i],
                                                                  np.swapaxes(
                                                                      np.tensordot(R[i], skew_ddv[i], axes=([0], [1])),
                                                                      0, 1))
                               for i in range(x.shape[0])])
            
            dedR = np.asarray([np.dot(-skew_dv[i], P[i])
                               for i in range(x.shape[0])])
            
            der['x,R'].append(((x, R), (dedx, dedR)))
    return der
    
def my_pSmV(Vsl,Vsr,j):
    """ 
    Compute product (p|Sm(v)) (j=0) (= inner product of Vsl and Vsr) and 
    the gradient in m (j=1) (Geometrical suport of Vsl) coding
    the the linear form in V^* v->(p|Sm(v)) as dictionary Vsl (l for left)
    having only '0' and 'p' types and the fixed v in the formula m->(p|Sm(v)) 
    as Vsr (r for right).
    
    attention: Vs does not necessary come from a cot_to_vs
    Supposes that they are in the same rkhs and that vsl has only '0' and 'p'
    """
    
    if j == 1:
        out = dict()
        
        if '0' in Vsl:
            out['0']=[]
            for (x,p) in Vsl['0']:
                dv = fields.my_VsToV(Vsr,x,j)
                der = np.asarray([np.dot(p[i],dv[i]) for i in range(x.shape[0])])
                out['0'].append((x,der))
                
        if 'p' in Vsl:
            out['p']=[]
            for (x,P) in Vsl['p']:
                ddv = fields.my_VsToV(Vsr,x,j+1)
                ddv = (ddv + np.swapaxes(ddv,1,2))/2
                der = np.asarray([np.tensordot(P[i],ddv[i]) 
                    for i in range(x.shape[0])])
                out['p'].append((x,der))
    elif j == 0:
        out = 0.
        
        if '0' in Vsl:
            for (x,p) in Vsl['0']:
                v = fields.my_VsToV(Vsr,x,j)
                out += np.sum(np.asarray([np.dot(p[i],v[i]) 
                    for i in range(x.shape[0])]))
                
        if 'p' in Vsl:
            for (x,P) in Vsl['p']:
                dv = fields.my_VsToV(Vsr,x,j+1)
                dv = (dv + np.swapaxes(dv,1,2))/2
                out += np.sum(np.asarray([np.tensordot(P[i],dv[i]) 
                    for i in range(x.shape[0])]))
                
    return out

