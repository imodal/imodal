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

def my_eta(): #symmetric case
    eta = np.zeros((2,2,3))
    c = 1/np.sqrt(2)
    eta[0,0,0], eta[0,1,1], eta[1,0,1], eta[1,1,2]  = 1., c, c, 1.
    return eta

def my_skew_eta(): # skew-symmetric case
    skew_eta = np.zeros((2,2,1))
    c = 1/np.sqrt(2)
    skew_eta[0,1,0], skew_eta[1,0,0] = -c, +c
    return skew_eta

def my_skew_eta(): # skew-symmetric case
    skew_eta = np.zeros((2,2,1))
    c = 1/np.sqrt(2)
    skew_eta[0,1,0], skew_eta[1,0,0] = -c, +c
    return skew_eta

def my_Keta(K): # transformation on the right (sym case)
    return np.tensordot(K, my_eta(), axes=2)
    
def my_Kskew_eta(K):# same (skew-sym case)
    return np.tensordot(K, my_skew_eta(), axes=2)

def my_etaK(K): # Now on the left
    return np.tensordot(my_eta().transpose(), K, axes=2)

def my_skew_etaK(K): # Same for skew-sym case
    return np.tensordot(my_skew_eta().transpose(), K, axes=2)
    
def my_etaKeta(K): # Now on both sides (sym case)
    return my_etaK(my_Keta(K))
    
def my_skew_etaKeta(K): # Same mixed case
    return my_skew_etaK(my_Keta(K))
    
def my_dot_varifold(x,y, sig):
    cx, cy = my_close(x), my_close(y)
    nx, ny = x.shape[0], y.shape[0]
    
    vx, vy = cx[1:nx+1,:]-x, cy[1:ny+1,:]-y
    mx, my = (cx[1:nx+1,:]+x)/2, (cy[1:ny+1,:]+y)/2
    
    xy = np.tensordot(np.swapaxes(np.tensordot(mx,my,axes =0),1,2),np.eye(2))
    
    d2 = np.tile(np.sum(mx*mx,axis = 1).reshape(nx,1),(1,ny))\
        + np.tile(np.sum(my*my,axis = 1),(nx,1)) -2*xy
        
    kxy = np.exp(-d2/(2*sig**2))
    
    vxvy = np.tensordot(np.swapaxes(np.tensordot(vx,vy,axes =0),1,2)
            ,np.eye(2))**2
    
    nvx = np.sqrt(np.sum(vx*vx, axis = 1))
    nvy = np.sqrt(np.sum(vy*vy, axis = 1))
    
    mask = vxvy > 0
    
    cost = np.sum(kxy[mask]*vxvy[mask]/(np.tensordot(nvx,nvy, axes =0)[mask]))
    return cost
    
    
def my_dxdot_varifold(x, y,sig):
    cx, cy = my_close(x), my_close(y)
    nx, ny = x.shape[0], y.shape[0]
    
    vx, vy = cx[1:nx+1,:]-x, cy[1:ny+1,:]-y
    mx, my = (cx[1:nx+1,:]+x)/2, (cy[1:ny+1,:]+y)/2
    
    xy = np.tensordot(np.swapaxes(np.tensordot(mx,my,axes =0),1,2),np.eye(2))
    
    d2 = np.tile(np.sum(mx*mx,axis = 1).reshape(nx,1),(1,ny))\
        + np.tile(np.sum(my*my,axis = 1),(nx,1)) -2*xy
        
    kxy = np.exp(-d2/(2*sig**2))
    
    vxvy = np.tensordot(np.swapaxes(np.tensordot(vx,vy,axes =0),1,2)
            ,np.eye(2))**2
    
    nvx = np.sqrt(np.sum(vx*vx, axis = 1))
    nvy = np.sqrt(np.sum(vy*vy, axis = 1))
    
    mask = vxvy > 0
    
    u = np.zeros(vxvy.shape)
    u[mask] = vxvy[mask]/np.tensordot(nvx,nvy, axes =0)[mask]
    cost = np.sum(kxy[mask]*u[mask])
    
    dcost = np.zeros(x.shape)
    
    dcost1 = (-u*kxy)/(2*sig**2)
    dcost1 = 2*(np.tile(np.sum(dcost1,axis = 1).reshape(nx,1),(1,2))*mx
            -np.tensordot(np.swapaxes(np.tensordot(dcost1,np.eye(2),axes = 0),1,2),my))
    
    dcost += dcost1/2
    dcost[1:] += dcost1[0:-1]/2
    dcost[0] += dcost1[-1]/2
    
    u[mask] = kxy[mask]/np.tensordot(nvx,nvy, axes =0)[mask]
    dcost2 = 2*u*np.tensordot(np.swapaxes(np.tensordot(vx,vy,axes =0),1,2)
            ,np.eye(2))
    dcost2 = np.tensordot(np.swapaxes(np.tensordot(dcost2, np.eye(2), axes =0), 1,2),vy)
    dcost += -dcost2
    dcost[1:] += dcost2[0:-1]
    dcost[0] += dcost2[-1]
    
    dcost3 = np.zeros(kxy.shape)
    u[mask] = kxy[mask]*vxvy[mask]/(np.tensordot(nvx,nvy, axes =0)[mask]**2)
    dcost3[mask]  = -u[mask]
    dcost3 = np.dot(dcost3,nvy)
    tmp = np.zeros(nvx.shape)
    tmp[nvx>0] = dcost3[nvx>0]/(2*nvx[nvx>0])
    dcost3 =2* np.tile(tmp.reshape(vx.shape[0],1),(1,vx.shape[1]))*vx
    dcost += - dcost3
    dcost[1:] += dcost3[0:-1]
    dcost[0] += dcost3[-1]
    
    return cost, dcost

def my_dxvar_cost(x,y,sig):
    (cost1, dxcost1)  = my_dxdot_varifold(x,x,sig)
    (cost2, dxcost2)= my_dxdot_varifold(y,y,sig)
    (cost3, dxcost3)= my_dxdot_varifold(x,y,sig)
    return (cost1+cost2-2*cost3, 2*dxcost1-2*dxcost3) 
    
# Core kernels functions
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
    
    # Main kernel dot product 
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
        K = my_etaK(np.tensordot(my_nker(x-y,1,sig),np.eye(2),axes=0))
    elif (k==1)&(l==1):
        t = np.tensordot(-my_nker(x-y,2,sig), np.eye(2), axes=0)
        K = my_etaKeta(np.swapaxes(t,1,2))
    return K

    
def my_R(th) :
    """ return the 2D matrix of the rotation of angle theta
    """
    R = np.zeros((2,2))
    R[0,0], R[0,1] = np.cos(th), -np.sin(th)
    R[1,0], R[1,1] = np.sin(th), np.cos(th)
    return R

def my_Amh(Mod1,h):
    """ Compute the target value for the strain tensor
    """
    (x,R) = Mod1['x,R']
    C = Mod1['C']
    N = x.shape[0]
    eta = my_eta()
    out = np.asarray([np.tensordot(np.dot(R[i],
        np.dot(np.diag(np.dot(C[i],h)),
        R[i].transpose())),eta,axes = 2) for i in range(N)])
    return out
     
 
    # Model constructors
def my_mod1(x1, th, sig): # first order points
    N = x1.shape[0]
    R = np.asarray([my_R(th[i]) for i in range(N)])
    M = {'N':N, 'x':x1, 'th':th, 'R':R, 'sig':sig}
    return M
    
def my_mod0(x0, sig): # zeros'th order points
    N = x0.shape[0]
    M = {'N':N, 'x':x0, 'sig':sig}
    return M
    
def my_mods(xs): # silent module (observable points)
    N = xs.shape[0]
    M = {'N':N, 'x':xs}
    return M
    
def my_border(dom, n):
    (ax, bx, ay, by) = dom
    (Nx, Ny) = n
    x0 = np.zeros((2*(Nx+Ny)+1,2))
    x0[0:Nx,0], x0[0:Nx,1] = np.linspace(ax,bx,Nx,endpoint = False), ay
    x0[Nx:Nx+Ny,0], x0[Nx:Nx+Ny,1] = bx, np.linspace(ay,by,Ny,endpoint = False)
    x0[Nx+Ny:2*Nx+Ny,0], x0[Nx+Ny:2*Nx+Ny,1] \
        = np.linspace(bx,ax,Nx,endpoint = False), by
    x0[2*Nx+Ny:2*Nx+2*Ny+1,0], x0[2*Nx+Ny:2*Nx+2*Ny+1,1] \
        = ax, np.linspace(by,ay,Ny+1,endpoint = True)
    return x0
    
def my_open_border(dom, n):
    (ax, bx, ay, by) = dom
    Nx, Ny = n[0]-1, n[1]-1
    x0 = np.zeros((2*(Nx+Ny),2))
    x0[0:Nx,0], x0[0:Nx,1] = np.linspace(ax,bx,Nx,endpoint = False), ay
    x0[Nx:Nx+Ny,0], x0[Nx:Nx+Ny,1] = bx, np.linspace(ay,by,Ny,endpoint = False)
    x0[Nx+Ny:2*Nx+Ny,0], x0[Nx+Ny:2*Nx+Ny,1] \
        = np.linspace(bx,ax,Nx,endpoint = False), by
    x0[2*Nx+Ny:2*Nx+2*Ny,0], x0[2*Nx+Ny:2*Nx+2*Ny,1] \
        = ax, np.linspace(by,ay,Ny,endpoint = False)
    return x0


def my_new_SKS(Mod):
    """ my_SKS(Mod) compute induce metric on the bundle of 
    symmetric   matrice and vectors depending on the order of the
    constraint.
    """
    if 'x,R' in Mod:
        sig = Mod['sig']
        (x,R) = Mod['x,R']
        N = x.shape[0]
        SKS = np.zeros((3*N,3*N))
        for i in range(N):
            for j in range(i+1,N):
                SKS[3*i:3*(i+1),3*j:3*(j+1)] = my_K(x[i],x[j],sig,1,1)
        SKS += SKS.transpose()
        for i in range(N):
            SKS[3*i:3*(i+1),3*i:3*(i+1)] = my_K(x[i],x[i],sig,1,1)
        if 'nu' in Mod:
            SKS = SKS + Mod['nu']*np.eye(SKS.shape[0])
            
    if '0' in Mod:
        sig = Mod['sig']
        x = Mod['0']
        N = x.shape[0]
        SKS = np.zeros((2*N,2*N))
        for i in range(N):
            for j in range(i+1,N):
                SKS[2*i:2*(i+1),2*j:2*(j+1)] = my_K(x[i],x[j],sig,0,0)
        SKS += SKS.transpose()
        for i in range(N):
            SKS[2*i:2*(i+1),2*i:2*(i+1)] = my_K(x[i],x[i],sig,0,0)
    return SKS
    
def my_new_AmKiAm(Mod1):
    SKS = Mod1['SKS']
    (x,R) = Mod1['x,R']
    N = x.shape[0]
    C = Mod1['C']
    dimh = C.shape[2]
    lam = np.zeros((dimh,3*N))
    Am = np.zeros((3*N,dimh))
    
    for i in range(dimh):
        h = np.zeros((dimh))
        h[i] = 1.
        Am[:,i] = my_Amh(Mod1,h).flatten()
        lam[i,:] = solve(SKS, Am[:,i], sym_pos = True)
    return (Am,np.dot(lam,Am))
    
def my_CotToVs(Cot,sig):
    Vs = {'0':[], 'p':[], 'm':[]}
    [Vs['0'].append(s) for s in Cot['0']]
    
    if 'x,R' in Cot:
        [Vs['0'].append((s[0][0],s[1][0])) for s in Cot['x,R']]
        for ((x,R),(p,P)) in Cot['x,R']:
            Vs['m'].append((x,np.asarray([np.dot(P[i],R[i].transpose()) 
                for i in range(x.shape[0])])))
        #[Vs['m'].append((s[0][0],[np.dot(s[0][1].transpose(),s[1][1]))) 
       
    Vs['sig'] = sig
    return Vs

def my_CotDotV(Cot,Vs):
    """  This function computes product betwwen a covector h  of the and a v 
    field  such as (h|v.x) or (h[v.(x,R))
    """
    out = 0.

    if '0' in Cot:
        for (x,p) in Cot['0']: # Landmark point
            v = my_VsToV(Vs,x,0)
            out += np.sum([np.dot(p[i],v[i]) for i in range(x.shape[0])])
    
    if 'x,R' in Cot:
        for ((x,R),(p,P)) in Cot['x,R']:
            v, dv = my_VsToV(Vs,x,0), my_VsToV(Vs,x,1)
            skew_dv = (dv - np.swapaxes(dv, 1, 2))/2
            out += np.sum([np.dot(p[i],v[i])+
                np.tensordot(P[i],np.dot(skew_dv[i],R[i])) 
                for i in range(x.shape[0])])
    return out
    
def my_dCotDotV(Cot,Vs):
    """  This function computes the derivative with respect to the parameter
    of product between a covector h a v field  such as (h|v.x) or 
    (h|v.(x,R))
    """
    der = dict()

    if '0' in Cot:
        der['0'] = []
        for (x,p) in Cot['0']: # Landmark point
            dv = my_VsToV(Vs,x,1)
            der['0'].append((x,np.asarray([np.dot(p[i],dv[i]) 
                for i in range(x.shape[0])])))
    
    if 'x,R' in Cot:
        der['x,R']=[]
        for ((x,R),(p,P)) in Cot['x,R']:
            dv, ddv = my_VsToV(Vs,x,1), my_VsToV(Vs,x,2)
            
            skew_dv = (dv - np.swapaxes(dv, 1, 2))/2
            skew_ddv = (ddv - np.swapaxes(ddv, 1, 2))/2
            
            dedx = np.asarray([np.dot(p[i],dv[i])+ np.tensordot(P[i],
            np.swapaxes(np.tensordot(R[i],skew_ddv[i], axes =([0],[1])),0,1)) 
            for i in range(x.shape[0])])
            
            dedR = np.asarray([np.dot(-skew_dv[i], P[i]) 
                for i in range(x.shape[0])])
            
            der['x,R'].append(((x,R),(dedx,dedR)))
    return der
    
def my_add_der(der0, der1):
    """ add der
    """
    der = dict()
    if '0' in der0:
        der['0']=[]
        for i in range(len(der0['0'])):
            (x,dxe0), (x,dxe1) = der0['0'][i], der1['0'][i]
            der['0'].append((x,dxe0+dxe1))
    if 'x,R' in der0:
        der['x,R']=[]
        for i in range(len(der0['x,R'])):
            ((x,R),(dxe0,dRe0)),((x,R),(dxe1,dRe1)) \
                = der0['x,R'][i], der1['x,R'][i]
            der['x,R'].append(((x,R),(dxe0+dxe1,dRe0+dRe1)))
    return der


def my_VsToV(Par, z, j): # generic vector field (tested)
    """ This is the main function to conpute the derivative of order 0 to 2 
    for vector fields generated by simple dirac of order 0 (landmarks) and 
    first order dirivatives of dirac: 'p' points corresponds to the symmetric
    of the derivatives and needs a couple (x,p) where x is (N,2) and p is (N,2,2) 
    and 'm' points correspont to skew symmetric part of the derivative and 
    are given by a pair (x,p) where x is (N,2) and p is (N,2,2). You can mixed 
    different kind of points ie '0', 'm' and 'p' points (no nneed to have all
    tyoes present) should be transmited in a dictionary as 
    Par={'0':[(x,p)], 'p':[(x,p]), 'm':[(x,p)], 'sig':sig} 
    z is the location where the values are computed
    
    The output is a list ordered according the the input '0', 'p' and 'm'
    """
    Nz = z.shape[0]
    sig = Par['sig']
    lsize = ((Nz,2),(Nz,2,2),(Nz,2,2,2))
    djv = np.zeros(lsize[j])
    
    if '0' in Par:
        for (x,p) in Par['0']:
            N = x.shape[0]
            for i in range(Nz):
                C = (-1)**j
                cz = z[i].reshape(1,2)
                ker_vec = C*np.asarray([my_nker(s,j,sig) for s in x-cz])
                djv[i] += np.tensordot(np.tensordot(np.eye(2),ker_vec, axes=0),
                    p, axes = ([1,2],[1,0]))
        
    if 'p' in Par:
        for (x,P) in Par['p']:
            P = (P + np.swapaxes(P,1,2))/2
            C = (-1)**j
            for i in range(Nz):
                cz = z[i].reshape(1,2)
                ker_vec = C*np.asarray([my_nker(s,j+1,sig) for s in x-cz]) #(N,2)
                djv[i] += np.tensordot(np.tensordot(np.eye(2),ker_vec,axes=0),
                    P, axes = ([1,2,3],[1,0,2])) # (2,2,N,2)  (2,N)
    
    if 'm' in Par:
        for (x,P) in Par['m']:
            P = (P - np.swapaxes(P,1,2))/2
            C = (-1)**j
            for i in range(Nz):
                cz = z[i].reshape(1,2)
                ker_vec = C*np.asarray([my_nker(s,j+1,sig) for s in x-cz]) #(N,2)
                djv[i] += np.tensordot(
                    np.tensordot(np.eye(2),ker_vec,axes=0),
                    P, axes =([1,2,3],[1, 0, 2])) # (2,2,N,2)  (2,N)
            
    return djv
    
def my_pSmV(Vsl,Vsr,j):
    """ Compute product (p|Sm(v)) (j=0) and the gradient in m (j=1) coding
    the the linear form in V^* v->(p|Sm(v)) as dictionary Vsl (l for left)
    having only '0' and 'p' types and the fixed v in the formula m->(p|Sm(v)) 
    as Vsr (r for right).
    """
    
    if j == 1:
        out = dict()
        
        if '0' in Vsl:
            out['0']=[]
            for (x,p) in Vsl['0']:
                dv = my_VsToV(Vsr,x,j)
                der = np.asarray([np.dot(p[i],dv[i]) for i in range(x.shape[0])])
                out['0'].append((x,der))
                
        if 'p' in Vsl:
            out['p']=[]
            for (x,P) in Vsl['p']:
                ddv = my_VsToV(Vsr,x,j+1)
                ddv = (ddv + np.swapaxes(ddv,1,2))/2
                der = np.asarray([np.tensordot(P[i],ddv[i]) 
                    for i in range(x.shape[0])])
                out['p'].append((x,der))
    elif j == 0:
        out = 0.
        
        if '0' in Vsl:
            for (x,p) in Vsl['0']:
                v = my_VsToV(Vsr,x,j)
                out += np.sum(np.asarray([np.dot(p[i],v[i]) 
                    for i in range(x.shape[0])]))
                
        if 'p' in Vsl:
            for (x,P) in Vsl['p']:
                dv = my_VsToV(Vsr,x,j+1)
                dv = (dv + np.swapaxes(dv,1,2))/2
                out += np.sum(np.asarray([np.tensordot(P[i],dv[i]) 
                    for i in range(x.shape[0])]))
                
    return out
    

def my_hToP(Mod1,h):
    (x,R) = Mod1['x,R']
    N = x.shape[0]
    SKS = my_new_SKS(Mod1)
    lam = solve(SKS, my_Amh(Mod1,h).reshape(3*N), sym_pos = True)
    return (x,np.tensordot(lam.reshape(N,3),my_eta().transpose(), axes =1))

def my_init_from_mod(Mod):
    if '0' in Mod:
        nMod = {'sig': Mod['sig'], 'coeff': Mod['coeff']}
    
    if 'x,R' in Mod:
        nMod = {'sig':Mod['sig'], 'C':Mod['C'], 'coeff': Mod['coeff']}
        if 'nu' in Mod:
            nMod['nu'] = Mod['nu']
    return nMod
    
def my_mod_update(Mod): 
    if not 'SKS' in Mod:
        Mod['SKS'] = my_new_SKS(Mod)
        
    if '0' in Mod:        
        (x,p) = (Mod['0'], Mod['mom'].flatten())
        Mod['cost'] = Mod['coeff']*np.dot(p,np.dot(Mod['SKS'],p))/2
    
    if 'x,R' in Mod:
        (x,R) = Mod['x,R']
        N = x.shape[0]
        Mod['Amh'] = my_Amh(Mod,Mod['h']).flatten()
        Mod['lam'] = solve(Mod['SKS'], Mod['Amh'], sym_pos = True)
        Mod['mom'] = np.tensordot(Mod['lam'].reshape(N,3),
            my_eta().transpose(), axes =1)
        Mod['cost'] = Mod['coeff']*np.dot(Mod['Amh'],Mod['lam'])/2
    return
    
def my_mod_init_from_Cot(Mod,nCot):
    if '0' in Mod:
        nMod = my_init_from_mod(Mod)
        nx0 = nCot['0'][0][0]    
        nMod['0'] = nx0
        nMod['SKS'] = my_new_SKS(nMod)
        nMod['mom'] = solve(nMod['coeff']*nMod['SKS'],
            my_VsToV(my_CotToVs(Cot,Mod['sig']),nx0,0).flatten(),
            sym_pos = True).reshape(nx0.shape)
        my_mod_update(nMod) # compute cost0
    
    # updating h1
    if 'x,R' in Mod:
        nMod = my_init_from_mod(Mod)
        (nx1, nR) = nCot['x,R'][0][0]
        nMod['x,R'] = (nx1, nR)
        nMod['SKS'] = my_new_SKS(nMod)
        dv = my_VsToV(my_CotToVs(nCot,Mod['sig']),nx1,1)
        S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
        tlam = solve(nMod['coeff']*nMod['SKS'], S.flatten(), sym_pos = True)
        (Am, AmKiAm) = my_new_AmKiAm(nMod)
        nMod['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
        my_mod_update(nMod) # will compute the new lam, Amh, mom and cost
    
    return nMod

def my_new_ham(Mod0,Mod1,Cot):
    ham = 0.
    my_mod_update(Mod0), my_mod_update(Mod1)
    Vs0 = {'0':[(Mod0['0'],Mod0['mom'])], 'sig':Mod0['sig']}
    (x,R) = Mod1['x,R']
    Vs1 = {'p':[(x,Mod1['mom'])], 'sig':Mod1['sig']}    
    ham = my_CotDotV(Cot,Vs0) + my_CotDotV(Cot,Vs1) - Mod0['cost'] - Mod1['cost']
    return ham
    
def my_dxH(Mod0, Mod1, Cot):
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    Vs0 = {'0':[(Mod0['0'],Mod0['mom'])], 'sig':sig0}
    (x,R), P = Mod1['x,R'], Mod1['mom']
    Vs1 = {'p':[(x,P)], 'sig':sig1}  
    
    # derivatives with respect to the end conditions
    
    cder = my_add_der(my_dCotDotV(Cot,Vs0),my_dCotDotV(Cot,Vs1))
    
    # derivatives with respect to the initial conditions
    Vsr1 = my_CotToVs(Cot, sig1)
    der = my_pSmV(Vs1,Vsr1,1)
    dx1H = der['p'][0][1]
    # 
    Vsr0 = my_CotToVs(Cot, sig0)
    der = my_pSmV(Vs0,Vsr0,1)
    dx0H = der['0'][0][1]
    
    der = my_pSmV(Vs0,Vs0,1) # to take into acc. the cost var.
    dx0H += -Mod0['coeff']*der['0'][0][1]
    
    # derivatives with respect to the operator
    dv = my_VsToV(Vsr1,x,1)
    S = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
    tlam = solve(Mod1['SKS'], S.flatten(), sym_pos = True)
    tP = np.tensordot(tlam.reshape(S.shape),my_eta().transpose(), axes = 1) 
    tVs = {'p':[(x,tP)], 'sig':sig1}
    
    der = my_pSmV(tVs,Vs1,1)
    dx1H += - der['p'][0][1]
    der = my_pSmV(Vs1,tVs,1)
    dx1H += - der['p'][0][1]
    
    Amh = np.tensordot(Mod1['Amh'].reshape(S.shape),my_eta().transpose(), axes = 1)
    Ptmp = (tP - Mod1['coeff']*P) # takes into acc. the cost variation in x1
    dRH = 2*np.asarray([np.dot(np.dot(Ptmp[i], Amh[i]),R[i]) 
        for i in range(x1.shape[0])])
    
    der = my_pSmV(Vs1,Vs1,1)
    dx1H += Mod1['coeff']*der['p'][0][1]
    # put everything in cder
    (x,dxe) = cder['0'][0]
    cder['0'][0]=(x,dxe+dx0H)
    ((x,R),(dxe,dRe)) = cder['x,R'][0]
    cder['x,R'][0] = ((x,R),(dxe+dx1H, dRe+dRH))
    
    return cder

def my_dpH(Mod0, Mod1, Cot):
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    
    Vs0 = {'0':[(x0,Mod0['mom'])], 'sig':sig0}
    
    P = Mod1['mom']
    Vs1 = {'p':[(x1,P)], 'sig':sig1}
    
    vx0 = my_VsToV(Vs0,x0,0)+ my_VsToV(Vs1,x0,0)
    vx1 = my_VsToV(Vs0,x1,0)+ my_VsToV(Vs1,x1,0)
    vxs = my_VsToV(Vs0,xs,0)+ my_VsToV(Vs1,xs,0)
    dv  = my_VsToV(Vs0,x1,1)+ my_VsToV(Vs1,x1,1)
    S  = (dv - np.swapaxes(dv,1,2))/2
    vR = np.asarray([ np.dot(S[i],R[i]) for i in range(x1.shape[0])]) 
    
    derp = {'0':[(x0,vx0), (xs,vxs)], 'x,R':[((x1,R), (vx1, vR))]}
    return derp
    

def my_nforward(Cot, dMod0,dMod1,dCot,Dt):
    """ similar than my_forward but able to compute z + Dt X_H(z') where z is
    determined by Cot and z' by dMod0, dMod1, dCot. Useful for RK2 shooting
    """
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = dMod0['sig'], dMod1['sig']
    
    # updating the co-state
    derx = my_dxH(dMod0, dMod1, dCot)
    np0 = p0 - Dt*derx['0'][0][1]
    nps = ps - Dt*derx['0'][1][1]
    np1 = p1 - Dt*derx['x,R'][0][1][0]
    nPR = PR - Dt*derx['x,R'][0][1][1]
    
    # updating the state
    derp = my_dpH(dMod0, dMod1, dCot)
    nx0 = x0 + Dt*derp['0'][0][1]
    nxs = xs + Dt*derp['0'][1][1]
    nx1 = x1 + Dt*derp['x,R'][0][1][0]
    nR  = R  + Dt*derp['x,R'][0][1][1]
            
    # updating Cot
    nCot = {'0':[(nx0,np0), (nxs,nps)], 'x,R':[((nx1,nR),(np1, nPR))]}
    
    # updating h0
    nMod0 = my_init_from_mod(dMod0)
    nMod0['0'] = nx0    
    nMod0['SKS'] = my_new_SKS(nMod0)
    nMod0['mom'] = solve(nMod0['coeff']*nMod0['SKS'],
        my_VsToV(my_CotToVs(nCot,sig0),nx0,0).flatten(),
        sym_pos = True).reshape(x0.shape)
    my_mod_update(nMod0) # compute cost0
    
    # updating h1
    nMod1 = my_init_from_mod(dMod1)
    nMod1['x,R'] = (nx1, nR)
    nMod1['SKS'] = my_new_SKS(nMod1)
    dv = my_VsToV(my_CotToVs(nCot,sig1),nx1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
    tlam = solve(nMod1['coeff']*nMod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = my_new_AmKiAm(nMod1)
    nMod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    my_mod_update(nMod1) # will compute the new lam, mom and cost
    
    return (nMod0, nMod1, nCot)
    
def my_forward(Mod0,Mod1,Cot,Dt):        
    
    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    
    # updating the co-state
    derx = my_dxH(Mod0, Mod1, Cot)
    np0 = p0 - Dt*derx['0'][0][1]
    nps = ps - Dt*derx['0'][1][1]
    np1 = p1 - Dt*derx['x,R'][0][1][0]
    nPR = PR - Dt*derx['x,R'][0][1][1]
    
    # updating the state
    derp = my_dpH(Mod0, Mod1, Cot)
    nx0 = x0 + Dt*derp['0'][0][1]
    nxs = xs + Dt*derp['0'][1][1]
    nx1 = x1 + Dt*derp['x,R'][0][1][0]
    nR  = R  + Dt*derp['x,R'][0][1][1]
            
    # updating Cot
    nCot = {'0':[(nx0,np0), (nxs,nps)], 'x,R':[((nx1,nR),(np1, nPR))]}
    
    # updating h0
    nMod0 = my_init_from_mod(Mod0)
    nMod0['0'] = nx0    
    nMod0['SKS'] = my_new_SKS(nMod0)
    nMod0['mom'] = solve(nMod0['coeff']*nMod0['SKS'],
        my_VsToV(my_CotToVs(nCot,sig0),nx0,0).flatten(),
        sym_pos = True).reshape(x0.shape)
    my_mod_update(nMod0) # compute cost0
    
    # updating h1
    nMod1 = my_init_from_mod(Mod1)
    nMod1['x,R'] = (nx1, nR)
    nMod1['SKS'] = my_new_SKS(nMod1)
    dv = my_VsToV(my_CotToVs(nCot,sig1),nx1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
    tlam = solve(nMod1['coeff']*nMod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = my_new_AmKiAm(nMod1)
    nMod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    my_mod_update(nMod1) # will compute the new lam, mom and cost
    
    return (nMod0, nMod1, nCot)

def my_sub_bckwd(Mod0, Mod1, Cot, grad, my_eps):
    """ my_sub_bckwd compute an elementary backward step associated with the 
    hamiltonian flow. grad is a dictionary with the following form
    grad = {'0':[(dx0G, dp0G),(dxsG, dpsG)], 'x,R':[((dx1G, dRG), (dp1G, dpRG))]}
    """
    
    # der ={'0':[(x0,dx0H), (xs,dxsH)], 'x,R':[((x1,R),(dx1H,dRH))]}
    

    [(x0,p0),(xs,ps)]= Cot['0']
    [((x1,R), (p1,PR))]= Cot['x,R']
    sig0, sig1 = Mod0['sig'], Mod1['sig']
    

    # computing x - eps \nabla_pG
    nx0 = x0 - my_eps*grad['0'][0][1]
    nxs = xs - my_eps*grad['0'][1][1]
    nx1 = x1 - my_eps*grad['x,R'][0][1][0]
    nR  =  R - my_eps*grad['x,R'][0][1][1]
    
    # updating p + eps\nabla_xG
    np0 = p0 + my_eps*grad['0'][0][0]
    nps = ps + my_eps*grad['0'][1][0]
    np1 = p1 + my_eps*grad['x,R'][0][0][0]
    nPR = PR + my_eps*grad['x,R'][0][0][1]
    
    
    nCot ={ '0':[(nx0,np0), (nxs,nps)], 'x,R':[((nx1,nR),(np1,nPR))]}
    
    # updating h0
    nMod0 = my_init_from_mod(Mod0)
    nMod0['0'] = nx0    
    nMod0['SKS'] = my_new_SKS(nMod0)
    nMod0['mom'] = solve(nMod0['coeff']*nMod0['SKS'],
        my_VsToV(my_CotToVs(nCot,sig0),nx0,0)
        .flatten(),
        sym_pos = True).reshape(x0.shape)
    my_mod_update(nMod0) # compute cost0
    
    # updating h1
    nMod1 = my_init_from_mod(Mod1)
    nMod1['x,R'] = (nx1, nR)
    nMod1['SKS'] = my_new_SKS(nMod1)
    dv = my_VsToV(my_CotToVs(nCot,sig1),nx1,1)
    S  = np.tensordot((dv + np.swapaxes(dv,1,2))/2,my_eta())
    tlam = solve(nMod1['coeff']*nMod1['SKS'], S.flatten(), sym_pos = True)
    (Am, AmKiAm) = my_new_AmKiAm(nMod1)
    nMod1['h'] = solve(AmKiAm, np.dot(tlam,Am), sym_pos = True)
    my_mod_update(nMod1) # will compute the new lam, mom and cost
    
    # Computing dF^*(\nabla G) for F the hamiltonian flow    
    ngrad = dict(grad)
    
    # Computing (\nabla xH(z+eps J grad) - \nabla_x H(z))/eps 
    derx, nderx = my_dxH(Mod0, Mod1, Cot), my_dxH(nMod0, nMod1, nCot)
    
    dx0G  = (nderx['0'][0][1]-derx['0'][0][1])/my_eps
    dxsG = (nderx['0'][1][1]-derx['0'][1][1])/my_eps
    dx1G  = (nderx['x,R'][0][1][0]-derx['x,R'][0][1][0])/my_eps
    dRG   = (nderx['x,R'][0][1][1]-derx['x,R'][0][1][1])/my_eps
    
    # Computing (\nabla pH(z+eps J grad) - \nabla_p H(z))/eps 
    derp, nderp  = my_dpH(Mod0, Mod1, Cot), my_dpH(nMod0, nMod1, nCot)
    
    dp0G  = (nderp['0'][0][1]-derp['0'][0][1])/my_eps
    dpsG  = (nderp['0'][1][1]-derp['0'][1][1])/my_eps
    dp1G  = (nderp['x,R'][0][1][0]-derp['x,R'][0][1][0])/my_eps
    dpRG  = (nderp['x,R'][0][1][1]-derp['x,R'][0][1][1])/my_eps    
    
    ngrad ={'0':[(dx0G, dp0G),(dxsG, dpsG)], 
        'x,R':[((dx1G, dRG),(dp1G, dpRG))]}
    
    return ngrad

    
def my_fd_shoot(Mod0,Mod1,Cot,N):
    h = 1./N
    Step = (Mod0, Mod1, Cot)
    Traj =[Step]
    # rk2 steps
    for i in range(N):
        print(i)
        dStep = my_forward(Step[0], Step[1], Step[2], h/2.)
        Traj.append(dStep)
        Step =  my_nforward(Step[2], dStep[0], dStep[1], dStep[2],h)
        Traj.append(Step)        
    return Traj

def my_mult_grad(grad, a):
    ngrad = dict(grad)
    [(dx0G, dp0G), (dxsG, dpsG)] = grad['0']
    [((dx1G, dRG),(dp1G, dpRG))] = grad['x,R']
    ndx0G, ndp0G, ndxsG, ndpsG = a*dx0G, a*dp0G, a*dxsG, a*dpsG
    ndx1G, ndRG, ndp1G, ndpRG  = a*dx1G, a*dRG,  a*dp1G, a*dpRG
    ngrad ={'0':[(ndx0G, ndp0G),(ndxsG, ndpsG)], 
        'x,R':[((ndx1G, ndRG),(ndp1G, ndpRG))]}
    return ngrad
    
def my_add_grad(grada, gradb):
    ngrad = dict(grad)
    [(dx0Ga, dp0Ga), (dxsGa, dpsGa)] = grada['0']
    [((dx1Ga, dRGa),(dp1Ga, dpRGa))] = grada['x,R']
    [(dx0Gb, dp0Gb), (dxsGb, dpsGb)] = gradb['0']
    [((dx1Gb, dRGb),(dp1Gb, dpRGb))] = gradb['x,R']
    
    (ndx0G, ndp0G, ndxsG, ndpsG) = (dx0Ga+dx0Gb, dp0Ga+ dp0Gb, 
        dxsGa+dxsGb, dpsGa+ dpsGb)
    (ndx1G, ndRG, ndp1G, ndpRG)  = (dx1Ga+dx1Gb, dRGa+dRGb, 
        dp1Ga+ dp1Gb, dpRGa+ dpRGb)
    ngrad ={'0':[(ndx0G, ndp0G),(ndxsG, ndpsG)], 
        'x,R':[((ndx1G, ndRG),(ndp1G, ndpRG))]}
    return ngrad
        
    
    
def my_bck_shoot(Traj, grad, my_eps):
    
    Traj.reverse()
    N = int((len(Traj)-1)/2)
    h = 1./N
    count, cgrad = 0, grad.copy()
    for i in range(N):
        print(i)
        count +=1
        dStep = Traj[count]
        rnp = my_sub_bckwd(dStep[0], dStep[1], dStep[2], cgrad, my_eps)
        nnp = my_mult_grad(rnp, h)
        count +=1
        Step = Traj[count]
        rn = my_sub_bckwd(Step[0], Step[1], Step[2], rnp, my_eps)
        rn = my_mult_grad(rn, h/2)
        cgrad = my_add_grad(my_add_grad(cgrad, rnp), rn)
    Traj.reverse()
    return cgrad
    
    
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
## Unitary test (passed)
# my_nker order 3
    x = np.array([1.,0.5])
    xp = x+0.001* rd.normal(0,1,(2))
    sig = 0.3
    Dk = my_nker(xp,2,sig)-my_nker(x,2,sig)
    Dx = xp -x
    dk = my_nker(x,3,sig)
    print(np.dot(dk,Dx))
    print(Dk)
    
## my_dv0 (passed)
    M0 = Exp0['M0']
    h = rd.normal(0,1,(M0['N'],2))
    M0['h'] = h
    zp = z+0.00001*rd.normal(0,1,z.shape)
    Dz = zp-z
    Dv0 = my_v0(M0,zp,h)-my_v0(M0,z,h)
    dv0 = my_dv0(M0,z,h)
    print(np.dot(dv0[0],Dz[0]))
    print(Dv0[0])

## my_dv0 an other test (passed)
    Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
    sig = 0.1
    x = rd.normal(0,1,(Nx,2))
    z = rd.normal(0,1,(Nz,2))
    zp = z + 0.000001*rd.normal(0,1,(Nz,2))
    px = rd.normal(0,1,(Nx,2))
    
    Dz = zp -z
    
    M0['x'] = x
    M0['sig'] = sig
    M0['N'] = Nx
    M0['h'] = px
    Dv0 = my_v0(M0,zp,px)-my_v0(M0,z,px)
    dv0 = my_dv0(M0,z,px)
    print(np.dot(dv0[0],Dz[0]))
    print(Dv0[0])

## Test my_VsToV (passed)
    x, p = rd.normal(0,1,(13,2)), rd.normal(0,1,(13,2))
    sig = 0.4
    Par ={'0': [(x,p)], 'p':[], 'm':[], 'sig':sig}
    
    z = rd.normal(0,1,(20,2))
    zp = z+ 0.000001*rd.normal(0,1,z.shape)
    Dz = zp -z
    
    j=0
    Djv = np.asarray(my_VsToV(Par,zp,j))-np.asarray(my_VsToV(Par,z,j))
    djv = my_VsToV(Par,z,j+1)
    k=10
    print(np.dot(djv[k],Dz[k]))
    print(Djv[k])
    
    j=1
    Djv = np.asarray(my_VsToV(Par,zp,j))-np.asarray(my_VsToV(Par,z,j))
    djv = my_VsToV(Par,z,j+1)
    k=10
    print(np.dot(djv[k],Dz[k]))
    print(Djv[k])
    


    x, p = rd.normal(0,1,(13,2)), rd.normal(0,1,(13,2,2))
    Par ={'0':[], 'p': [(x,p)], 'm':[], 'sig':sig}
    z = rd.normal(0,1,(20,2))
    zp = z+ 0.00001*rd.normal(0,1,z.shape)
    Dz = zp -z
    
    j=0
    Djv = np.asarray(my_VsToV(Par,zp,j))-np.asarray(my_VsToV(Par,z,j))
    djv = my_VsToV(Par,z,j+1)
    k=10
    print(np.dot(djv[k],Dz[k]))
    print(Djv[k])
    
    j=1
    Djv = np.asarray(my_VsToV(Par,zp,j))-np.asarray(my_VsToV(Par,z,j))
    djv = my_VsToV(Par,z,j+1)
    k=10
    print(np.dot(djv[k],Dz[k]))
    print(Djv[k])
    
## my_CotDotV, 
   

## Comparing my_SKS with my_new_SKS (passed)

    x = rd.normal(0,1,(12,2))
    th = np.zeros(13)
    R = np.asarray([my_R(cth) for cth in th])
    sig =0.4
    M1 = {'x':x, 'N':x.shape[0], 'sig':sig}
    Mod1 = {'x,R':(x,R), 'sig':sig}
    print(my_SKS_M1(M1) - my_new_SKS(Mod1))

## Checking my_Am

    n0  = (2, 2)   # number of order 0
    n1  = (4,5)   # number of point of order 1
    dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
    sig = (0.25, 0.7) # sigma's for K0 and K1
    th = 0*np.pi
    h =[0.,0.5]
    Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
    (x,P) = my_hToP(Mod1,h)
    
    Vs = {'p':[(x,P)], 'sig' : Mod1['sig']}
    v = my_VsToV(Vs, x, 0)
    
    fig=plt.figure(1)
    fig.clf()
            
    plt.axis('equal')
    Q = plt.quiver(x[:,0], x[:,1],v[:,0], v[:,1],
        units='xy',scale = 1.,zorder=0,headwidth=5.,
        width=0.005,headlength=5.,color='blue')
        
    xb = my_border(dom, n1)
    vxb = my_VsToV(Vs, xb, 0)
    
    plt.plot(xb[:,0]+vxb[:,0],xb[:,1]+vxb[:,1],color='red')
    
    
    plt.ylim(-1,3)
    plt.show()  
    
            #     
            # QT = plt.quiver(x0[:,0], x0[:,1], v[:,0], v[:,1],
            #     units='xy',scale = 1.,zorder=0,headwidth=5.,
            #     width=0.01,headlength=5.,color='red')
            # 
            # outline = Exp['outline']
            # plt.plot(outline[:,0], outline[:,1],'b')
            # 
            # 
            
            # plt.savefig('plotd' + str(i) +'.png')
## Comparing genv with v
    Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
    sig = 0.1
    x = rd.normal(0,1,(Nx,2))
    xp = rd.normal(0,1,(Nxp,2))
    xm = rd.normal(0,1,(Nxm,2))
    z = rd.normal(0,1,(Nz,2))
    zp = z + 0.000001*rd.normal(0,1,(Nz,2))
    px = rd.normal(0,1,(Nx,2))
    pxm = rd.normal(0,1,(Nxm,1))
    pxp = rd.normal(0,1,(Nxp,3))
    
    Dz = zp -z
    
    M0['x'] = x
    M0['sig'] = sig
    M0['N'] = Nx
    M1['x'] = xp
    M1['sig'] = sig
    M1['N'] =Nxp
    vx, vxp, vxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, z)
    print(vx - my_v0(M0,z,px))
    print(vxp - my_v1(M1,z,pxp.reshape(3*Nxp)))
    
    dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
    print(dvx -my_dv0(M0,z,px))
    print(dvxp -my_dv1(M1,z,pxp.reshape(3*Nxp)))

## my_gen_dv (passed)
    Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
    sig = 0.1
    x = rd.normal(0,1,(Nx,2))
    xp = rd.normal(0,1,(Nxp,2))
    xm = rd.normal(0,1,(Nxm,2))
    z = rd.normal(0,1,(Nz,2))
    zp = z + 0.0001*rd.normal(0,1,(Nz,2))
    Dz = zp -z
    
    px = rd.normal(0,1,(Nx,2))
    pxm = rd.normal(0,1,(Nxm,1))
    pxp = rd.normal(0,1,(Nxp,3))
    
    vx, vxp, vxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, z)
    nvx, nvxp, nvxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, zp)
    dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
    Dvx = nvx -vx
    Dvxp = nvxp -vxp
    Dvxm = nvxm - vxm
    print(np.dot(dvx[0],Dz[0]))
    print(Dvx[0])
    print(np.dot(dvxp[0],Dz[0]))
    print(Dvxp[0])
    print(np.dot(dvxm[0],Dz[0]))
    print(Dvxm[0])
    
## my_gen_ddv (passed)
    Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
    sig = 0.1
    x = rd.normal(0,1,(Nx,2))
    xp = rd.normal(0,1,(Nxp,2))
    xm = rd.normal(0,1,(Nxm,2))
    z = rd.normal(0,1,(Nz,2))
    zp = z + 0.00001*rd.normal(0,1,(Nz,2))
    Dz = zp -z
    
    px = rd.normal(0,1,(Nx,2))
    pxm = rd.normal(0,1,(Nxm,1))
    pxp = rd.normal(0,1,(Nxp,3))
    
    dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
    ndvx, ndvxp, ndvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, zp)
    ddvx, ddvxp, ddvxm = my_gen_ddv(x, xp, xm, px, pxp, pxm, sig, z)
    Ddvx = ndvx -dvx
    Ddvxp = ndvxp -dvxp
    Ddvxm = ndvxm - dvxm
    print(np.dot(ddvx[0],Dz[0]))
    print(Ddvx[0])
    print(np.dot(ddvxp[0],Dz[0]))
    print(Ddvxp[0])
    print(np.dot(ddvxm[0],Dz[0]))
    print(Ddvxm[0])

##  my_gen_ximv1, my_gen_pximv1, my_gen_dmpximv1 (passed)

    Nx, Nxp, Ny, Nyp, Nym = 10, 11, 12, 13, 14
    sig = 0.1
    x, px = rd.normal(0,1,(Nx,2)), rd.normal(0,1,(Nx,2))
    xp, pxp = rd.normal(0,1,(Nxp,2)), rd.normal(0,1,(Nxp,3))
    y, py = rd.normal(0,1,(Ny,2)), rd.normal(0,1,(Ny,2))
    yp, pyp = rd.normal(0,1,(Nyp,2)), rd.normal(0,1,(Nyp,3))
    ym, pym = rd.normal(0,1,(Nym,2)), rd.normal(0,1,(Nym,1))
    
    C=0.0001
    ny = y + C*rd.normal(0,1,(Ny,2))
    nyp = yp + C*rd.normal(0,1,(Nyp,2))
    nym = ym + C*rd.normal(0,1,(Nym,2))
    
    
    e = my_gen_pximv1(x, xp, px, pxp, sig, y, yp, ym, py, pyp, pym)
    ne = my_gen_pximv1(x, xp, px, pxp, sig, ny, nyp, nym, py, pyp, pym)
    De = ne -e
    Dy, Dyp, Dym = ny - y, nyp - yp, nym - ym
    dpy, dpyp, dpym = my_gen_dmpximv1( x, xp, px, pxp, sig, y, yp, ym, py, pyp, pym)
    print(np.tensordot(dpy, Dy)+np.tensordot(dpyp, Dyp)+np.tensordot(dpym, Dym))
    print(De)
    

    
## my dxpv0(M0,z,alpha,pz) (passed)
    
    M0 = Exp0['M0']
    h = rd.normal(0,1,(M0['N'],2))
    pz = rd.normal(0,1,z.shape)
    M0['h'] = h
    M0p = dict(M0)
    M0p['x'] = M0['x'] + 0.00001*rd.normal(0,1,M0['x'].shape)
    
    x, xp = M0['x'], M0p['x']
    M0['v'] = my_v0(M0,z,h)
    M0p['v'] = my_v0(M0p,z,h)
    
    Dx = xp - x
    Dpv0 = np.tensordot(pz,M0p['v']-M0['v'], axes = 2)
    dxpv0 = my_dxpv0(M0,z,h,pz)
    dpv0 = 0.
    for i in range(M0['N']):
        dpv0 += np.dot(dxpv0[i],Dx[i])
    print(Dpv0)
    print(dpv0)
    
## my_dzdv1 (passed)
    h = np.array([0.0, -0.1])
    lam = my_lam_M1(M1,h)
    zp = z+0.0001*rd.normal(0,1,z.shape)
    Dz = zp-z
    
    Dzdv1 = my_dv1(M1,zp,lam)-my_dv1(M1,z,lam)
    dzdv1 = my_dzdv1(M1,z,lam)
    print(np.dot(dzdv1[10],Dz[10]))
    print(Dzdv1[10])
    
## Miscellenea
    u = rd.normal(0,1,(2,2))
    s = np.tensordot(u,my_eta(),axes = 2)
    print(u)
    print(np.dot(my_eta(),s))
    
## Consistence test between Am(h) and (dv1+dv1t)/2 (passed)
    M1 = Exp0['M1']
    h = np.array([0.0, 0.2])
    lam = my_lam_M1(M1,h)
    M1['lam'] = lam
    M1['h'] = h
    my_Am(M1)
    print(M1['Am'])
    dv = my_dv1(M1, M1['x'],lam)
    sdv = np.tensordot((dv + np.swapaxes(dv,1, 2))/2,my_eta(),axes =2)
    print(M1['Am']-sdv)
    
## my_dener1 (gradient in x)  (passed)
    M1 = Exp0['M1']
    h = np.array([0.0, -0.1])
    lam = my_lam_M1(M1,h)
    M1['lam'] = lam
    M1['h'] = h
    my_Am(M1)
    
    M1p = dict(M1)
    M1p['x'] = M1['x'] + 0.00001*rd.normal(0,1,M1['x'].shape)
    lamp = my_lam_M1(M1p,h)
    M1p['lam'] = lamp
    my_Am(M1p)
    
    x, xp = M1['x'], M1p['x']
    Dx = xp -x
    De = my_ener1(M1p)-my_ener1(M1)
    my_dener1(M1)
    
    de = 0
    dxe = M1['dener']['x']
    for i in range(M1['N']):
        de += np.dot(dxe[i],Dx[i])
    print(De)
    print(de)
    
## my_dener1 (gradient in th)  (passed)
    M1 = Exp0['M1']
    h = np.array([0.0, -0.1])
    lam = my_lam_M1(M1,h)
    M1['lam'] = lam
    M1['h'] = h
    my_Am(M1)
    
    M1p = dict(M1)
    M1p['th'] = M1['th'] + 0.000001*rd.normal(0,1,M1['th'].shape)
    M1p['R'] = np.asarray([my_R(M1p['th'][i]) for i in range(M1p['N'])])
    lamp = my_lam_M1(M1p,h)
    M1p['lam'] = lamp
    my_Am(M1p)

    
    th, thp = M1['th'], M1p['th']
    Dth = thp-th
    De = my_ener1(M1p)-my_ener1(M1)
    my_dener1(M1)
    
    de = 0
    dthe = M1['dener']['th']
    for i in range(M1['N']):
        de += np.dot(dthe[i],Dth[i])
    print(De)
    print(np.sqrt(2)*de)

## my_dener0 (gradient in x) (passed)

    M0 = Exp0['M0']
    h = rd.normal(0,1,(M0['N'],2))
    M0['h'] = h
    M0p = dict(M0)
    M0p['x'] = M0['x'] + 0.00001*rd.normal(0,1,M0['x'].shape)
    
    x, xp = M0['x'], M0p['x']
    M0['v'] = my_v0(M0,x,h)
    M0p['v'] = my_v0(M0p,xp,h)
    
    Dx = xp -x
    De = my_ener0(M0p)-my_ener0(M0)
    my_dener0(M0)
    
    de = 0
    dxe = M0['dener']['x']
    for i in range(M0['N']):
        de += np.dot(dxe[i],Dx[i])
    print(De)
    print(de)
##

    N = M0['N']
    zp = z+0.0001*rd.normal(0,1,z.shape)
    alpha = 0.1*rd.normal(0,1,(N,2))
    v, vp = my_v0(M0,z,alpha), my_v0(M0,zp,alpha)
    Dz, Dv = zp-z, vp -v
    dv = my_dv0(M0,z,alpha)
    test = [np.dot(dv[s],Dz[s])-Dv[s] for s in range(z.shape[0])]
    
## my_dv1

    zp = z+0.0001*rd.normal(0,1,z.shape)
    v, vp = my_v1(M1,z,lam), my_v1(M1,zp,lam)
    Dz, Dv = zp-z, vp -v
    dv = my_dv1(M1,z,lam)
    test = [np.dot(dv[s],Dz[s])-Dv[s] for s in range(z.shape[0])]
    
##


    N = M0['N']
    zp = z+0.0001*rd.normal(0,1,z.shape)
    alpha = 0.1*rd.normal(0,1,(N,2))
    M0['h'] = alpha
    v, vp = my_v(M0,M1,z), my_v(M0,M1,zp)
    Dz, Dv = zp-z, vp -v
    dv = my_dv(M0,M1,z)
    test = [np.dot(dv[s],Dz[s])-Dv[s] for s in range(z.shape[0])]
    
##
    
    
    M0, M1 = Exp0['M0'], Exp0['M1']
    N = M0['N']
    alpha = 0.01*rd.normal(0,1,(N,2))
    M0['h'] = alpha
    h = np.array([0.0, -0.1])
    lam = my_lam_M1(M1,h)
    M1['lam']=lam
    M1['h']=h
    Exp = Exp0
    
    x0, x1 = M0['x'], M1['x']
    my_rho(Exp)
    my_Am(M1)
    
    # Energy computation
    e1 = my_ener1(M1)
    e0 = my_ener0(M0)
    print(e1+e0)
    
    
    
    om, vx1, vx0 = M1['om'], M1['v'], M0['v']
    
    fig=plt.figure(1)
    fig.clf()
    
    
    
    plt.axis('equal')
    Q = plt.quiver(x1[:,0],x1[:,1],vx1[:,0], vx1[:,1],
        units='xy',scale = 1.,zorder=0,headwidth=5.,
        width=0.005,headlength=5.,color='blue')
        
    # QR = plt.quiver(x1[:,0],x1[:,1],nR[:,0,0], nR[:,0,1],
    #     units='xy',scale = 1.,zorder=0,headwidth=5.,
    #     width=0.005,headlength=5.,color='black')
        
    QT = plt.quiver(x0[:,0], x0[:,1], vx0[:,0], vx0[:,1],
        units='xy',scale = 1.,zorder=0,headwidth=5.,
        width=0.01,headlength=5.,color='red')
    plt.show()

##
    outline = Exp['outline']
    plt.plot(outline[:,0], outline[:,1],'b')
    
    
    plt.plot(xv+vz[0:nx*ny,0].reshape((ny,nx)), yv+vz[0:nx*ny,1].reshape((ny,nx)),color='green')
    plt.plot(np.transpose(xv+vz[0:nx*ny,0].reshape((ny,nx))), np.transpose(yv+vz[0:nx*ny,1].reshape((ny,nx))),color='green')
    
    plt.ylim(-1,3)
    plt.show()  
    
    test = [np.dot(dv[s],Dz[s])-Dv[s] for s in range(z.shape[0])]
## Test 4  using ptov


    
    n0  = (2, 2)   # number of order 0
    n1  = (4,5)   # number of point of order 1
    dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
    sig = (0.25, 0.7) # sigma's for K0 and K1
    th = np.pi/4
    
    coeff_mixed = np.array([0.1, 10]) # Metric weight order 0 and 1
    Exp0 = my_exp(dom,  n0, n1, th, sig, coeff_mixed)
    
    coeff_mixed = np.array([100, 0.1]) # Metric weight order 0 and 1
    Exp1 = my_exp(dom,  n0, n1, th, sig, coeff_mixed)
    
    M = Exp0['M1']
    C = np.zeros((M['N'],2,2))
    c = np.zeros((M['N'],2))
    x = M['x']
    for k in range(n1[0]):
        c[x[:,0]==x[k,0],:] \
            = (n1[0]-k)*np.array([1., 1.])/n1[0]
    tmp = np.zeros((2,2,2))
    tmp[0,0,0]=1.
    tmp[1,1,1]=1.
    C = np.tensordot(c,tmp,axes=1)
    M.setdefault('C', C)
    Exp1['M1'].setdefault('C', C)
    
    
##
    ptov = my_ptov(Exp0['M0'],Exp0['M1'],Exp0['coeff_mixed'])
    Exp0.setdefault('ptov',ptov)
    
    ptov = my_ptov(Exp1['M0'],Exp1['M1'],Exp1['coeff_mixed'])
    Exp1.setdefault('ptov',ptov)

##
    # construction of the visualisation grid
    nx, ny = (6,12)
    (ax, bx, ay, by) =dom
    xv, yv = np.meshgrid(np.linspace(ax,bx,nx),np.linspace(ay,by,ny))
    z = np.zeros((nx*ny,2))
    z[:,0]=xv.reshape((nx*ny))
    z[:,1]=yv.reshape((nx*ny))
    z = np.append(z,Exp0['outline'], axis=0)
  
##
        
    (Nx, Ny) = n0
    
    for i in range(1):
        Exp = Exp1
        M0, M1, coeff_mixed = Exp['M0'], Exp['M1'], Exp['coeff_mixed']
        x0 = M0['x']
        ptov = Exp['ptov']
        
        
        #h = [0.0, 0.5]
        h = 0.3*rd.normal(0,1,(2))
        # lam = my_lam_M1(M1,h)
        v = my_KS_M1(M1,x0, my_lam_M1(M1,h))
        
        for k in range(2):
            
            if k==0:
                Exp = Exp0
            elif k==1:
                Exp = Exp1
                
            p = solve(Exp['ptov'], v.reshape(4*(Nx+Ny)-8))
            lam = my_lam_M1(M1,my_ptoh(M0,M1,p))
            
            coeff_mixed = Exp['coeff_mixed']
            
            vz = my_KS_M1(M1,z,lam/coeff_mixed[1])\
                +my_KS_M0(M0,z,p/coeff_mixed[0])
                
            nv = my_KS_M1(M1,x0,lam/coeff_mixed[1])\
            +my_KS_M0(M0,x0,p/coeff_mixed[0])
                
    
            fig=plt.figure(0)
            
            
            
            
            # Plot of the result
            
            
            if k==0:
                fig.clf()
                plt.subplot(121)
            else:
                plt.subplot(122)
            
            plt.axis('equal')
            Q = plt.quiver(xv.reshape((nx*ny)), 
                yv.reshape((nx*ny)), vz[0:nx*ny,0], vz[0:nx*ny,1],
                units='xy',scale = 1.,zorder=0,headwidth=5.,
                width=0.005,headlength=5.,color='blue')
                
            QT = plt.quiver(x0[:,0], x0[:,1], v[:,0], v[:,1],
                units='xy',scale = 1.,zorder=0,headwidth=5.,
                width=0.01,headlength=5.,color='red')
            
            outline = Exp['outline']
            plt.plot(outline[:,0], outline[:,1],'b')
            
            
            plt.plot(xv+vz[0:nx*ny,0].reshape((ny,nx)), yv+vz[0:nx*ny,1].reshape((ny,nx)),color='green')
            plt.plot(np.transpose(xv+vz[0:nx*ny,0].reshape((ny,nx))), np.transpose(yv+vz[0:nx*ny,1].reshape((ny,nx))),color='green')
            
            plt.ylim(-1,3)
            plt.show()  
            plt.savefig('plotd' + str(i) +'.png')
        