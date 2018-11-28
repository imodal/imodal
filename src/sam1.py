import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from random import random,randint
import numpy.random as rd
from numpy.linalg import eigh,inv
from scipy.linalg import solve
plt.show()
plt.ion()

# Some stuff to pullback everything into 1D rep (symmetric and skew-symmetric mat)


# Core kernels functions
    
    
def my_new_exp0(dom, n0,  n1, th, sig):
    N1 = n1[0]*n1[1]
    (sig0,sig1) = sig
    
    # Creating m^{(1)}
    xvm, yvm = np.meshgrid(np.linspace(dom[0], dom[1], 
        n1[0]),  np.linspace(dom[2], dom[3], n1[1]))
    x = np.zeros((N1,2))
    x[:,0]=xvm.reshape((N1))
    x[:,1]=yvm.reshape((N1))
    th = th*np.ones((N1))*np.pi
    
    R = np.asarray([my_R(cth) for cth in th])
    
    # Creating m^{(0)}
    x0 = my_open_border(dom,n0)
    Mod0 = {'0':x0, 'sig': sig0}
    
    C = np.zeros((N1,2,2))
    c = np.zeros((N1,2))
    for k in range(n1[0]):
        c[x[:,0]==x[k,0],:] \
            = (n1[0]-k)*np.array([1., 1.])/n1[0]
        # c[x[:,0]==x[k,0],:] \
        #     = np.array([1., 1.])
    tmp = np.zeros((2,2,2))
    tmp[0,0,0]=1.
    tmp[1,1,1]=1.
    C = np.tensordot(c,tmp,axes=1)
    
    Mod1 = {'x,R':(x,R), 'C':C, 'sig':sig1}
    return Mod0, Mod1
    
    
    
def my_exp(dom, n0, n1, th, sig, coeff_mixed):
    (sig0, sig1) = sig
    
    # Creating m^{(1)}
    xvm, yvm = np.meshgrid(np.linspace(dom[0], dom[1], 
        n1[0]),  np.linspace(dom[2], dom[3], n1[1]))
    x = np.zeros((N1,2))
    x[:,0]=xvm.reshape((N1))
    x[:,1]=yvm.reshape((N1))
    
    # Creating m^{(0)}
    x0 = my_open_border(dom,n0)
    N0 = x0.shape[0]
    
    # Creating an outline for the object
    outline = my_border(dom, (3,3))
    
    # Creating a slinet module on the boundary
    Ms = my_mods(outline)
    
    # Creating the associated frames
    th = th*np.ones((N1))*np.pi
    # R = np.zeros((N1,2,2))
    # for i in range(N1):
    #     R[i] = my_R(th[i])
        
    # Creating momenta
    M1 = my_mod1(x,th,sig1)
    M0 = my_mod0(x0,sig0)
    
    Exp = {'M0':M0, 'M1':M1, 'Ms':Ms, 'outline':outline, 'coeff_mixed':coeff_mixed}
    return Exp
