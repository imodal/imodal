#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 17:07:08 2018

@author: barbaragris
"""
import numpy as np

from old import GeometricalDescriptors, DeformationModules as comb_mod, field_structures as fields

import old.Forward.Hamiltonianderivatives as hamder
import old.Forward.shooting as shoot
#%%
import old.StructuredFields.SummedFields
import old.StructuredFields.ZeroFields

#%%

N_pts = 3
dim = 2

#%%

GD = old.GeometricalDescriptors.GD_landmark.GD_landmark(N_pts, dim)

Pts = np.random.rand(N_pts, dim)
mom = np.random.rand(N_pts, dim)

param = (Pts, mom)

GD.fill_cot_from_param(param)

#%%

GD.mult_Cot_scal(-2)
GD1 = GD.copy_full()
GD1.mult_Cot_scal(-2)
#%%
print(GD.Cot)
GD.add_cot(GD.Cot)
print(GD.Cot)
print(GD1.Cot)
GD1.add_cot(GD.Cot)
print(GD.Cot)

#%%

Cot = {'0':[param], 'x,R':[]}
GD1.add_cot(Cot)
print(GD1.Cot)



#%%
sigma = 1
dim = 2
v = old.StructuredFields.StructuredFields_0.StructuredField_0(sigma, dim)

N_pts_f = 5
Pts = np.random.rand(N_pts_f, dim)
mom = np.random.rand(N_pts_f, dim)

param = [Pts, mom]

v.fill_fieldparam(param)


#%%

print(GD1.Ximv(v).Cot['0'][0][0])
print(fields.my_VsToV(v.dic, GD1.get_points(), 0))





#%%

v = GD.Cot_to_Vs(2)

pts1 = np.array([[0,0]])
mom1 = np.array([[1,0]])

param1 = [pts1, mom1]
GD2 = old.GeometricalDescriptors.GD_landmark.GD_landmark(1, 2)
GD2.fill_cot_from_param(param1)

print(GD1.Ximv(v).Cot['0'][0][0])
#print(fields.my_VsToV(v.dic, GD1.Cot['0'][0][0], 0))

#%%

GD.inner_prod_v(v)

#%%

v = GD.Cot_to_Vs(2)
GD3 = GD.Ximv(v)

print(GD3.Cot['0'][0][0])
print(GD.Cot['0'][0][1])

#%%

Pts = np.random.rand(N_pts, dim)
mom = np.random.rand(N_pts, dim)

param = (Pts, mom)
#%%

GD1.fill_cot_from_param(param)

dGD1 = GD1.dCotDotV(v)

inn = GD1.inner_prod_v(v)
GD_eps = GD1.copy()
x, p = GD1.Cot['0'][0]
zero_arr = np.zeros([N_pts, dim])
i=0
j=0
eps = 0.0001

der = np.zeros([N_pts, dim])

for i in range(N_pts):
    for j in range(dim):
        eps_arr = zero_arr.copy()
        eps_arr[i,j] = eps
        GD_eps.fill_cot_from_param((x+eps_arr, p))
        inn_eps = GD_eps.inner_prod_v(v)
        der[i,j] = (inn_eps - inn)/eps



print(dGD1.Cot['0'][0][0] -der )
print(der)

#%% 
#print(pair.my_CotDotV(GD1.Cot, v.dic))
print(GD1.inner_prod_v(v))



#%%
print(GD1.inner_prod_v(v))

#%%

GD_comb = old.GeometricalDescriptors.Combine_GD.Combine_GD([GD2, GD1])
print(GD_comb.Cot)
#%%
GD_comb1 = GD_comb.copy()
GD_comb1.Cot
#%%
param = [GD2.Cot['0'][0], GD1.Cot['0'][0]]
print(GD_comb1.Cot)
GD_comb1.fill_cot_from_param(param)
print(GD_comb1.Cot)


#%%
#print(GD_comb.Cot)
GD_comb1 = GD_comb.copy_full()
GD_comb.mult_Cot_scal(2)
print(GD_comb.Cot)
print(GD_comb1.Cot)

#%%_

#PROBLEME

print(GD_comb1.Cot)
GD_comb1.add_cot(GD_comb1.Cot)
print(GD_comb.Cot)
print(GD_comb1.Cot)

#%%
print(GD_comb.Ximv(v).Cot['0'])

print(GD_comb.GD_list[0].Ximv(v).Cot['0'])
print( GD_comb.GD_list[1].Ximv(v).Cot['0'] )


#%%

print(GD_comb.dCotDotV(v).Cot['0'])
print(GD_comb.GD_list[0].dCotDotV(v).Cot['0'])
print(GD_comb.GD_list[1].dCotDotV(v).Cot['0'])


#%%
print(GD_comb.inner_prod_v(v))
print(GD_comb.GD_list[0].inner_prod_v(v) + GD_comb.GD_list[1].inner_prod_v(v))

#%% 
print(GD_comb.Cot)
#print(GD_comb.Cot_to_Vs)
#%%
print(GD.Cot_to_Vs)


#%%
v1 = v.copy()
v2 = old.StructuredFields.SummedFields.sum_structured_fields([v, v1])
#%%

z = np.array([[0,0], [1,2]])
print(v2.Apply(z,1))
print(v.Apply(z,1) + v1.Apply(z,1))


#%%
sig = 1
N_pts = 3
dim=2
coeff=2
Mod = old.DeformationModules.ElasticOrder0.ElasticOrderO(sig, N_pts, dim, coeff)

#%%

Mod1 = Mod.copy_full()

#%%
Mod.fill_GD(GD)
#Mod.Compute_SKS_curr()
Mod.update()
print(Mod.SKS)
#%%

print(Mod.GeodesicControls_curr(GD))

print(Mod.Mom)
print(Mod.GeodesicControls(GD,GD))
#%%

print(Mod.GD)


#%%

v = Mod.field_generator(GD, Mod.Mom)
v = Mod.field_generator_curr()

#%%

Mod.Cost_curr()
print(Mod.cost)

print(Mod.Cost(GD, Mod.Mom))

#%%
dGD = GD.copy()
der = np.zeros([N_pts, dim])
pts = GD.get_points()
mom = Mod.Mom
zeroep = np.zeros([N_pts, dim])
co = Mod.Cost(GD, mom)

eps = 0.001

for i in range(N_pts):
    for j in range(dim):
       print(i,j)
       zeroep = np.zeros([N_pts, dim])
       dGD = GD.copy()
       zeroep[i,j]=eps
       dGD.fill_cot_from_param((pts+zeroep, mom))
       der[i,j]=(Mod.Cost(dGD, mom) - co)/eps
    
print(der)
print(Mod.DerCost(Mod.GD, Mod.Mom).Cot)

print(Mod.DerCost_curr().Cot)

#%%
vs  = Mod.field_generator(GD, mom)
dertmp = vs.p_Ximv(vs, 1)
print(dertmp)

#%%
der = Mod.cot_to_innerprod_curr(GD,1)
der.Cot

#%%


dGD = GD.copy()
der = np.zeros([N_pts, dim])
pts = GD.get_points()
mom = GD.get_mom()
zeroep = np.zeros([N_pts, dim])
co = Mod.cot_to_innerprod_curr(GD, 0)

eps = 0.001

for i in range(N_pts):
    for j in range(dim):
       print(i,j)
       zeroep = np.zeros([N_pts, dim])
       dGD = GD.copy()
       zeroep[i,j]=eps
       dGD.fill_cot_from_param((pts+zeroep, mom))
       der[i,j]=(Mod.cot_to_innerprod_curr(dGD, 0) - co)/eps
    
print(der)

print(Mod.cot_to_innerprod_curr(GD,1).Cot)

#%%
dim=2
v0 = old.StructuredFields.ZeroFields.ZeroField(dim)
v00 = v0.copy_full()
v0.fill_fieldparam([])
#%%
v_sum = old.StructuredFields.SummedFields.sum_structured_fields([v, v0])
#%%
print(v_sum.Apply(Pts, 0))
print(v.Apply(Pts,0))

#%%

Sil = DeformationModules.SilentLandmark.SilentLandmark(N_pts, dim)

#%%
Sil.GD.fill_GD(Pts)
Sil.GD.fill_cot_from_param([Pts, mom])
#%%
v00 = Sil.field_generator(Sil.GD, Sil.Cont)

#%%
Sil.DerCost(Sil.GD, Sil.Cont).Cot
#%%
Sil.cot_to_innerprod_curr(GD,1).Cot

#%%
Mod_c = comb_mod.CompoundModules([Sil, Mod])
#%%
Mod_c2 = Mod_c.copy_full()

#%% 
GDc = Mod_c.GD
co = GDc.Cot
GDc.mult_Cot_scal(2)
#%%
Mod_c2.fill_GD(GDc)

#%%
Mod.update()
Mod.GD.mult_Cot_scal(0.01)

#%%
Mod.SKS

#%%

Mod_c.update()
#%%
Mod_c.ModList[1].SKS

Mod_c.GD.Cot

#%%
print(Mod_c.GD.Cot)
Mod_c.GD.mult_Cot_scal(0.01)
print(Mod_c.GD.Cot)

#%%

Mod0 = old.DeformationModules.ElasticOrder0.ElasticOrderO(10, 1, 2, 1)
Mod01 = old.DeformationModules.ElasticOrder0.ElasticOrderO(1, 1, 2, 1)
ModS = DeformationModules.SilentLandmark.SilentLandmark(2, 2)

Modc = comb_mod.CompoundModules([ModS, Mod0, Mod01])

#print(Mod0.Cont)
#print(Modc.Cont)

Pts0 = np.array([[1., 0.]])
Pts01 = np.array([[-1., 0.]])
PtsS = np.array([[1., 0.], [-1., 0.]])
Mom0 = np.array([[0.,0.]])
Mom01 = np.array([[0.,0.]])
MomS = 1*np.array([[5.,0.],[-5.,0.]])
param0 = [Pts0, Mom0]
param01 = [Pts01, Mom01]
paramS = [PtsS, MomS]
param = [paramS, param0, param01]

Modc.GD.fill_cot_from_param(param)
Modc.update()
Modc.GeodesicControls_curr(Modc.GD)
print(Modc.Cont)

#%%

vc = Modc.field_generator_curr()
vc = Modc.field_generator(Modc.GD, Modc.Cont)
#%%
vc.Apply(PtsS, 1)
#%%
print(Modc.cost)
Modc.Cost_curr()
print(Modc.cost)
print(2*Modc.Cost(Modc.GD, Modc.Cont))


#%%
dGD = Modc.DerCost_curr()
dGD.Cot
#%%
Modc.Cont

#%%
GD = Modc.GD
GD0 = GD.GD_list[0]
Modc.cot_to_innerprod_curr(GD0, 1).Cot

#%%
Modc.update()
Modc.GeodesicControls_curr(Modc.GD)
dMom = hamder.dxH(Modc)
dGD  = hamder.dpH(Modc)
#%%
hamder.Ham(Modc)
#%%
#for j in dMom.indi_0[0]

#%%
#print(Modc.GD.Cot)
print(dMom.Cot)
#%%
print(Modc.GD.Cot)
Modc.add_cot(dGD)
print(Modc.GD.Cot)
#%%
print(Modc.GD.Cot)
print(Modc.GD.GD_list[0].Cot)
print(Modc.ModList[0].GD.Cot)
#%%

dGD.mult_Cot_scal(0.1)

#%%


traj, cont_list = shoot.shooting_traj(Modc, 10)
#%%

import matplotlib.pyplot as plt
i = 0
GDS = traj[i].GD_list[0].Cot['0'][0][0]
GD0 = traj[i].GD_list[1].Cot['0'][0][0]
GD01 = traj[i].GD_list[2].Cot['0'][0][0]
MomS = traj[i].GD_list[0].Cot['0'][0][1]
Mom0 = traj[i].GD_list[1].Cot['0'][0][1]
Mom01 = traj[i].GD_list[2].Cot['0'][0][1]

plt.plot(GDS[:,0], GDS[:,1], 'xb')
plt.plot(GD0[:,0], GD0[:,1], 'xr')
plt.plot(GD01[:,0], GD01[:,1], 'xr')
plt.quiver(GDS[:,0], GDS[:,1],MomS[:,0], MomS[:,1])
plt.quiver(GD0[:,0], GD0[:,1],Mom0[:,0], Mom0[:,1])
plt.quiver(GD01[:,0], GD01[:,1],Mom01[:,0], Mom01[:,1])


i = 10
GDS = traj[i].GD_list[0].Cot['0'][0][0]
GD0 = traj[i].GD_list[1].Cot['0'][0][0]
GD01 = traj[i].GD_list[2].Cot['0'][0][0]
MomS = traj[i].GD_list[0].Cot['0'][0][1]
Mom0 = traj[i].GD_list[1].Cot['0'][0][1]
Mom01 = traj[i].GD_list[2].Cot['0'][0][1]

plt.plot(GDS[:,0], GDS[:,1], 'ob')
plt.plot(GD0[:,0], GD0[:,1], 'or')
plt.plot(GD01[:,0], GD01[:,1],'or')
plt.quiver(GDS[:,0], GDS[:,1],MomS[:,0], MomS[:,1])
plt.quiver(GD0[:,0], GD0[:,1],Mom0[:,0], Mom0[:,1])
plt.quiver(GD01[:,0], GD01[:,1],Mom01[:,0], Mom01[:,1])


plt.axis([-5,5,-5,5])
plt.axis('equal')


#%%
N_pts = 1
dim = 2
C = np.array([[[1],[1]]])
GD = old.GeometricalDescriptors.GD_xR.GD_xR(N_pts, dim, C)
#%%
GD1 = GD.copy()

#%%
x = np.array([[0.,0.]])
R = np.array([[[0.,-1.],[1., 0.]]])
px = np.random.rand(*x.shape)
pR = np.random.rand(x.shape[0], 2, 2)

param = ((x,R), (px, pR))
GD.fill_cot_from_param(param)

Cot_old = {'x,R':param}

#%%
GD1 = GD.copy_full()
#%%
print(GD.get_points())
print(GD.get_R())
print(GD.get_mom())

#%%
GD.mult_Cot_scal(10)
print(GD.Cot)

#%%
print(GD.Cot)
GD.add_cot(GD.Cot)
print(GD.Cot)

#%%
v = GD.Cot_to_Vs(1)
print(v.dic)
#%%
#v0 = Modc.field_generator_curr()

#%%
#GD.Ximv(v0)
#GD.dCotDotV(v0)
##%%
#dicv_old = fields.my_CotToVs(Cot_old, 1)
v1 = GD.Cot_to_Vs(1)
v0 = Mod.field_generator_curr()

v = old.StructuredFields.SummedFields.sum_structured_fields([v0, v1])
#%%
dGD = GD.Ximv(v0)
#%%
#fields.my_VsToV(dicv_old, x, 0)

#%%
GD.inner_prod_v(v)


#%%
import copy
dGD = GD.dCotDotV(v)
param = GD.Cot['x,R'][0]
eps = 1e-7
xz = np.array([[0.,0.]])
Rz = np.array([[[0.,0.],[0., 0.]]])
pxz = np.zeros(x.shape)
pRz = np.zeros([x.shape[0], 2, 2])

paramz = ((xz,Rz), (pxz, pRz))

der = (np.array([[0.,0.]]), np.array([[[0.,0.],[0., 0.]]]))

GD1 = GD.copy_full()

inn = GD.inner_prod_v(v)
for i in range(dim):
    parameps = copy.deepcopy(param)
    parameps[0][0][0,i] += eps
    GD1.fill_cot_from_param(parameps)
    
    d_inn = GD1.inner_prod_v(v)
    
    der[0][0,i] = (d_inn - inn)/eps

for i in range(dim):
    for j in range(dim):
        parameps = copy.deepcopy(param)
        parameps[0][1][0][i,j] += eps
        GD1.fill_cot_from_param(parameps)
        
        d_inn = GD1.inner_prod_v(v)
        
        der[1][0][i,j] = (d_inn - inn)/eps
        
print(der)
print(dGD.Cot)

#%%
sig = 1
N_pts = 2
dim = 2
coeff =1
C = np.array([[[1],[1]]])
C = np.ones([N_pts, dim, 1])
Mod1 = DeformationModules.ElasticOrder1.ElasticOrder1(sig, N_pts, dim, coeff, C, 0.001)
#%%
x = np.array([[0.,0.], [1.5,1.]])
R = np.array([[[0.,-1.],[1., 0.]], [[0.,-1.],[1., 0.]]])
px = np.random.rand(*x.shape)
pR = np.random.rand(x.shape[0], 2, 2)

param = ((x,R), (px, pR))
Mod1.GD.fill_cot_from_param(param)

#%%

Mod11 = Mod1.copy()

#%%
Mod1.fill_GD(GD)

#%%

Mod1.add_cot(GD)
#%%
Mod1.Compute_SKS_curr()

#%%
Mod1.update()
#%%
Mod1.AmKiAm_curr()
#%%
Mod1.compute_mom_from_cont_curr()


#%%
Mod1.GeodesicControls_curr(Modc.GD)

#%%
v = Mod1.field_generator_curr()
#%%
Mod1.Cost_curr()
#%%
(Am, AmKiAm) = Mod1.AmKiAm_curr()


#%%
Mod1.update()
Mod1.GeodesicControls_curr(Modc.GD)
dGD = Mod1.DerCost_curr() 
Mod1.Cost_curr()
co = Mod1.cost

import copy
#param = GD.Cot['x,R'][0]
eps = 1e-6

#der = (np.array([[0.,0.]]), np.array([[[0.,0.],[0., 0.]]]))
der = (np.zeros([Mod1.N_pts, dim]), np.zeros([Mod1.N_pts, dim, dim]))
Mod11 = Mod1.copy_full()
Mod11.Cont = Mod1.Cont.copy()
for i in range(dim):
    for j in range(Mod1.N_pts):
        parameps = copy.deepcopy(param)
        parameps[0][0][j,i] += eps
        Mod11.GD.fill_cot_from_param(parameps)
        Mod11.update()
        Mod11.compute_mom_from_cont_curr()
        Mod11.Cost_curr()
        d_co = Mod11.cost
        
        der[0][j,i] = (d_co - co)/eps

for i in range(dim):
    for j in range(dim):
        for k in range(Mod1.N_pts):
            parameps = copy.deepcopy(param)
            parameps[0][1][k][i,j] += eps
            Mod11.GD.fill_cot_from_param(parameps)
            Mod11.update()
            Mod11.compute_mom_from_cont_curr()
            Mod11.Cost_curr()
            d_co = Mod11.cost
            
            der[1][k][i,j] = (d_co - co)/eps
        
print(der)
print(dGD.Cot)
#%%
Mod1.update()
Mod1.GeodesicControls_curr(Modc.GD)
dGD = Mod1.cot_to_innerprod_curr(Modc.GD, 1) 
co = Mod1.cot_to_innerprod_curr(Modc.GD, 0)

import copy
eps = 1e-6

der = (np.zeros([Mod1.N_pts, dim]), np.zeros([Mod1.N_pts, dim, dim]))

Mod11 = Mod1.copy_full()
Mod11.Cont = Mod1.Cont.copy()
for i in range(dim):
    for j in range(Mod1.N_pts):
        parameps = copy.deepcopy(param)
        parameps[0][0][j,i] += eps
        Mod11.GD.fill_cot_from_param(parameps)
        Mod11.update()
        Mod11.compute_mom_from_cont_curr()
        #Mod11.Cost_curr()
        d_co = Mod11.cot_to_innerprod_curr(Modc.GD, 0)
        
        der[0][j,i] = (d_co - co)/eps

for i in range(dim):
    for j in range(dim):
        for k in range(Mod1.N_pts):
            parameps = copy.deepcopy(param)
            parameps[0][1][k][i,j] += eps
            Mod11.GD.fill_cot_from_param(parameps)
            Mod11.update()
            Mod11.compute_mom_from_cont_curr()
            #Mod11.Cost_curr()
            d_co = Mod11.cot_to_innerprod_curr(Modc.GD, 0)
            
            der[1][k][i,j] = (d_co - co)/eps
        
print(der)
print(dGD.Cot)



