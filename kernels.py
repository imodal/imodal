#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 19:07:37 2018

@author: barbaragris
"""

import numpy as np
import functions_eta as fun_eta


def my_K(x,y,sig,k,l):
    """ my_K(x,y,sig,k,l) return the dot productuct between the 
        dual of canonical one Oth form, one form and two forms 
        ie (x,a_0)-> (v(x)|a_0) order 0
        (x,a_1) -> (dv(x),a_1) order 1
        (x,a_2) ->(d2v(x),a_2)
        where a_0, a_1, a_2 are respectivement a 1th order, second order 
        and    third order tensor.
        
        my_K is may be abusively using tensordot to keep compact expressions.
        sig is the kernel size and
        k, l and are the two orders. k should be greater or equal to 0
    """
    if (k==0)&(l==0):
        K = my_nker(x-y,0,sig)*np.eye(2)
    elif (k==1)&(l==0):
        K = fun_eta.my_etaK(np.tensordot(my_nker(x-y,1,sig),np.eye(2),axes=0))
    elif (k==1)&(l==1):
        t = np.tensordot(-my_nker(x-y,2,sig), np.eye(2), axes=0)
        K = fun_eta.my_etaKeta(np.swapaxes(t,1,2))
    return K


def my_nker(x,k,sig) : # tested
    """ Gaussian radial function and its derivatives. 
    x in a matrix of positions
    k is the order (0 gives the function at locations x, k=1 its 
    first derivatives and k=2 its hessian
    sig is the gaussian size.
    """

    x = x/sig
    h = np.exp(-np.sum(x**2/2))
    r = h # order 0
    if k==1:
        r = -h*x/sig
    elif k==2:
        r = h*(-np.eye(2)+np.tensordot(x, x, axes=0))/sig**2
    elif k==3:
        r = h*(-np.eye(2)+np.tensordot(x, x, axes=0))
        r = -np.tensordot(r,x,axes = 0) +\
            h*(np.swapaxes(np.tensordot(np.eye(2),x,axes = 0),1,2)
            +np.tensordot(x,np.eye(2),axes = 0))
        r = r/sig**3
    return r
