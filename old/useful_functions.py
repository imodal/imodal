#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:10:59 2018

@author: barbaragris
"""
import numpy as np
from old import kernels as ker


def my_new_SKS(Mod):
    """ my_SKS(Mod) compute induce metric on the bundle of
    symmetric   matrice and vectors depending on the order of the
    constraint.
    """
    if 'x,R' in Mod:
        sig = Mod['sig']
        (x, R) = Mod['x,R']
        SKS = ker.my_K(x, x, sig, 1)
        if 'nu' in Mod:
            SKS = SKS + Mod['nu'] * np.eye(SKS.shape[0])
    
    if '0' in Mod:
        sig = Mod['sig']
        x = Mod['0']
        SKS = ker.my_K(x, x, sig, 0)
    return SKS


def my_X(x0, xs, x1, R):
    X = np.concatenate([x0.flatten(), xs.flatten(), x1.flatten(), R.flatten()])
    return X


def my_P(p0, ps, p1, pR):
    P = np.concatenate([p0.flatten(), ps.flatten(), p1.flatten(), pR.flatten()])
    return P


def my_splitX(X,nX):
    (n0, ns, n1) = nX
    count = 0
    x0 = X[count:count+2*n0].reshape(n0,2)
    count += 2*n0
    xs = X[count:count+2*ns].reshape(ns,2)
    count += 2*ns
    x1 = X[count:count+2*n1].reshape(n1,2)
    count += 2*n1
    R = X[count:count + 4*n1].reshape(n1,2,2)
    return x0,xs,(x1,R)


def my_splitP(P, nX):
    (n0, ns, n1) = nX
    count = 0
    p0 = P[count:count + 2 * n0].reshape(n0, 2)
    count += 2 * n0
    ps = P[count:count + 2 * ns].reshape(ns, 2)
    count += 2 * ns
    p1 = P[count:count + 2 * n1].reshape(n1, 2)
    count += 2 * n1
    pR = P[count:count + 4 * n1].reshape(n1, 2, 2)
    return p0, ps, (p1, pR)


def my_CotFromXP(X, P, nX):
    """ Compute Cot representation from 1d representation X, P. nX = (n0,ns,n1)
    gives the numbers of points in x0, xs, and (x,R)
    return Cot
    """
    
    (x0, xs, (x1, R)) = my_splitX(X, nX)
    (n0, ns, n1) = nX
    
    count = 0
    p0 = P[count:count + 2 * n0].reshape(n0, 2)
    count = count + 2 * n0
    ps = P[count:count + 2 * ns].reshape(ns, 2)
    count += 2 * ns
    p1 = P[count:count + 2 * n1].reshape(n1, 2)
    count += 2 * n1
    pR = P[count:count + 4 * n1].reshape(n1, 2, 2)
    nCot = {'0': [(x0, p0), (xs, ps)], 'x,R': [((x1, R), (p1, pR))]}
    return nCot


def my_mult_grad(grad, a):
    ngrad = dict(grad)
    [(dx0G, dp0G), (dxsG, dpsG)] = grad['0']
    [((dx1G, dRG),(dp1G, dpRG))] = grad['x,R']
    ndx0G, ndp0G, ndxsG, ndpsG = a*dx0G, a*dp0G, a*dxsG, a*dpsG
    ndx1G, ndRG, ndp1G, ndpRG  = a*dx1G, a*dRG,  a*dp1G, a*dpRG
    ngrad ={'0':[(ndx0G, ndp0G),(ndxsG, ndpsG)],
        'x,R':[((ndx1G, ndRG),(ndp1G, ndpRG))]}
    return ngrad