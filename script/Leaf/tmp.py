"""
This example is a matching!
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

import implicitmodules.numpy.DataAttachment.Varifold as var
import implicitmodules.numpy.DeformationModules.Combination as comb_mod
import implicitmodules.numpy.DeformationModules.ElasticOrder0 as defmod0
import implicitmodules.numpy.DeformationModules.ElasticOrder1 as defmod1
import implicitmodules.numpy.DeformationModules.GlobalRotation as globrot
import implicitmodules.numpy.DeformationModules.GlobalTranslation as globtrans
import implicitmodules.numpy.DeformationModules.SilentLandmark as defmodsil
import implicitmodules.numpy.HamiltonianDynamic.Forward as shoot
import implicitmodules.numpy.Optimisation.ScipyOpti_attach as opti
from implicitmodules.numpy.Utilities import Rotation as rot
from implicitmodules.numpy.Utilities.Visualisation import my_close


# helper function
def my_plot(x, title="", col='*b'):
    plt.figure()
    plt.plot(x[:, 0], x[:, 1], col)
    plt.title(title)
    plt.axis('equal')
    plt.show()
    
name_exp = 'basi_pure_parametric'
#name_exp = 'basi_pure_nonparametric'
#name_exp = 'basi_semi_parametric'

generate_variation_target = False
use_target_variation = False

flag_show = False
flag_save = True
#  common options
nu = 0.001
dim = 2
N=10
maxiter = 300
 
lam_var = 40.
sig_var = [50., 10.]

# define attachment_function
#def attach_fun(xsf, xst):
#    return var.my_dxvar_cost(xsf, xst, sig_var)
                   
def attach_fun(xsf, xst):
    (varcost0, dxvarcost0) = var.my_dxvar_cost(xsf, xst, sig_var[0])
    (varcost1, dxvarcost1) = var.my_dxvar_cost(xsf, xst, sig_var[1])
    costvar = varcost0 + varcost1
    dcostvar = dxvarcost0 + dxvarcost1
    return (lam_var * costvar, lam_var * dcostvar )
                   

coeffs =[0.01, 10, 0.01]
coeffs_str = '0_01__1__0_01'
#coeffs =[100, 0.1,100]
#coeffs_str = '100__0_1__100'

# Source
path_data = '../data/'
with open(path_data + 'basi1b.pkl', 'rb') as f:
    _, lx = pickle.load(f)
    
Dx = 0.
Dy = 0.
height_source = 38.
height_target = 100.

nlx = np.asarray(lx).astype(np.float32)
(lmin, lmax) = (np.min(nlx[:, 1]), np.max(nlx[:, 1]))
scale = height_source / (lmax - lmin)

nlx[:, 1] = Dy-scale * (nlx[:, 1] - lmax)
nlx[:, 0] = Dx+scale * (nlx[:, 0] - np.mean(nlx[:, 0]))

# %% target
with open(path_data + 'basi1t.pkl', 'rb') as f:
    _, lxt = pickle.load(f)

nlxt = np.asarray(lxt).astype(np.float32)
(lmin, lmax) = (np.min(nlxt[:, 1]), np.max(nlxt[:, 1]))
scale = height_target / (lmax - lmin)
nlxt[:, 1] = - scale * (nlxt[:, 1] - lmax) 
nlxt[:, 0] = scale * (nlxt[:, 0] - np.mean(nlxt[:, 0])) 

if use_target_variation:
    xst = np.array(np.loadtxt('../data/basi_target_variation'))
else:
    xst = nlxt[nlxt[:, 2] == 2, 0:2]

    
if(flag_show):
    my_plot(xst, "Target", '-b')






# %% Silent Module
xs = nlx[nlx[:, 2] == 2, 0:2]
#xs = np.delete(xs, 3, axis=0)
Sil = defmodsil.SilentLandmark(xs.shape[0], dim)
ps = np.zeros(xs.shape)
param_sil = (xs, ps)
if(flag_show):
    my_plot(xs, "Silent Module", '*b')

# %% Modules of Order 0
sig0 = 20.
x0 = nlx[nlx[:, 2] == 1, 0:2]
#x0 = np.concatenate((x0, xs))
Model0 = defmod0.ElasticOrder0(sig0, x0.shape[0], dim, coeffs[1], nu)
p0 = np.zeros(x0.shape)
param_0 = (x0, p0)

if(flag_show):
    my_plot(x0, "Module order 0", 'or')

# %% Modules of Order 0
sig01 = 2.
x01 = xs.copy()
#x0 = np.concatenate((x0, xs))
Model01 = defmod0.ElasticOrder0(sig01, x01.shape[0], dim, coeffs[1], nu)
p01 = np.zeros(xs.shape)
param_01 = (x01, p01)

if(flag_show):
    my_plot(x01, "Module order 0", 'or')

# %% Modules of Order 0
sig00 = 2000.
x00 = np.array([[0., 0.]])
Model00 = defmod0.ElasticOrder0(sig00, x00.shape[0], dim, coeffs[0], nu)
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

if(flag_show):
    my_plot(x00, "Module order 00", '+r')

# %% Global translation
x00 = np.array([[0., 0.]])
Model00_g = globtrans.GlobalTranslation(dim, coeffs[0])
p00 = np.zeros([1, 2])
param_00 = (x00, p00)

if(flag_show):
    my_plot(x00, "Module order 00", '+r')

# %% Global rotation
x00 = np.array([[0., 0.]])
Model00_rot = globrot.GlobalRotation(dim, 10.)
p00_r = np.zeros([1, 2])
param_00_r = (x00, p00_r)

if(flag_show):
    my_plot(x00, "Module order 00", '+r')

# %% Modules of Order 1
sig1 = 30.

x1 = nlx[nlx[:, 2] == 1, 0:2]
x11 = x1.copy()
x11[:,0] *= -1
x1 = np.concatenate((x1, x11))




C = np.zeros((x1.shape[0], 2, 1))
K, L = 10, height_source
a, b = -2 / L ** 3, 3 / L ** 2
C[:, 1, 0] = (K * (a * (L - x1[:, 1]+Dy) ** 3 
     + b * (L - x1[:, 1]+Dy) ** 2))
C[:, 0, 0] = 1. * C[:, 1, 0]



Model1 = defmod1.ElasticOrder1(sig1, x1.shape[0], dim, coeffs[2], C, nu)

th = 0 * np.pi * np.ones(x1.shape[0])
R = np.asarray([rot.my_R(cth) for cth in th])

(p1, PR) = (np.zeros(x1.shape), np.zeros((x1.shape[0], 2, 2)))
param_1 = ((x1, R), (p1, PR))

if(flag_show):
    my_plot(x1, "Module order 1", 'og')

#%%
GD_r = Model00_rot.GD
GD_r.GD = np.array([[0., 0.]])
GD_t = Model00_g.GD

GD_t.GD = np.array([[5., 1.]])
GD_t.cotan = 0.001*np.array([[1., -7.]])
Model00_rot.GeodesicControls_curr(GD_t)
print(Model00_rot.Cont)
print(GD_t.infinitesimal_action(Model00_rot.field_generator_curr()).tan)
#%%
v0 = Model00_rot.cot_to_innerprod_curr(GD_t, 0)
v1 = Model00_rot.cot_to_innerprod_curr(GD_t, 1)
eps = 0.01
GD_r.GD = np.array([[eps, 0.]])
v00 = Model00_rot.cot_to_innerprod_curr(GD_t, 0)
GD_r.GD = np.array([[0., eps]])
v01 = Model00_rot.cot_to_innerprod_curr(GD_t, 0)

diff0 = (v00 - v0) / eps
diff1 = (v01 - v0) / eps
print(v1.cotan)
print(diff0, diff1)
#%%
GD_r.GD = np.array([[0., 0.]])
GD_1 = Model1.GD
GD_1.cotan = [np.random.rand(*x1.shape), np.random.rand(x1.shape[0], 2, 2)]
vrot = Model00_rot.field_generator_curr()

s1 = GD_1.infinitesimal_action(vrot)

v0 = Model00_rot.cot_to_innerprod_curr(GD_1, 0)
v1 = Model00_rot.cot_to_innerprod_curr(GD_1, 1)
eps = 0.01
GD_r.GD = np.array([[eps, 0.]])
v00 = Model00_rot.cot_to_innerprod_curr(GD_1, 0)
GD_r.GD = np.array([[0., eps]])
v01 = Model00_rot.cot_to_innerprod_curr(GD_1, 0)

diff0 = (v00 - v0) / eps
diff1 = (v01 - v0) / eps
print(v1.cotan)
print(diff0, diff1)
#%%
x0 = np.array([[1., 0.]])
speed0 = vrot.Apply(x0, 0)
eps = 0.01
diff = np.random.rand(1,2)
x1 = x0 + eps * diff
speed1 = vrot.Apply(x1, 0)

speeddiff = (speed1 - speed0)/eps

speedder = vrot.Apply(np.array([[1., 0.]]), 1)


print(speeddiff)
print(speedder[0] @ np.transpose(diff))
# %% Full model

if name_exp == 'basi_pure_nonparametric':
    Module = comb_mod.CompoundModules([Sil, Model0])
    Module.GD.fill_cot_from_param([param_sil, param_0])  
elif name_exp == 'basi_pure_parametric':
    Module = comb_mod.CompoundModules([Sil, Model00_g, Model00_rot, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_00_r, param_1])
elif name_exp == 'basi_semi_parametric':
    Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model1])
    Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_1])
    #Module = comb_mod.CompoundModules([Sil, Model00, Model0, Model01, Model1])
    #Module.GD.fill_cot_from_param([param_sil, param_00, param_0, param_01, param_1])
else:
    print('unknown experiment type')

P0 = opti.fill_Vector_from_GD(Module.GD)



# %%

args = (Module, xst, attach_fun, N, 1e-7)

res = scipy.optimize.minimize(opti.fun, P0,
                              args=args,
                              method='L-BFGS-B',
                              jac=opti.jac,
                              bounds=None,
                              tol=None,
                              callback=None,
                              options={
                                  'maxcor': 10,
                                  'ftol': 1.e-09,
                                  'gtol': 1e-03,
                                  'eps': 1e-08,
                                  'maxfun': 500,
                                  'maxiter': maxiter,
                                  'iprint': 1,
                                  'maxls': 25
                              })

P1 = res['x']
opti.fill_Mod_from_Vector(P1, Module)
Module_optimized = Module.copy_full()
Modules_list = shoot.shooting_traj(Module, N)

# %% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
if(flag_show):
    for i in range(N + 1):
        plt.figure()
        xs_i = Modules_list[2 * i].GD.GD_list[0].GD
        xs_ic = my_close(xs_i)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    
        #x0_i = Modules_list[2 * i].GD.GD_list[1].GD
       # plt.plot(x0_i[:, 0], x0_i[:, 1], '*r', linewidth=2)
    
        x00_i = Modules_list[2 * i].GD.GD_list[1].GD
        plt.plot(x00_i[:, 0], x00_i[:, 1], 'or', linewidth=2)
    
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        plt.show()

# %% With grid
#nxgrid, nygrid = (21, 21)  # create a grid for visualisation purpose
#xfigmin, xfigmax, yfigmin, yfigmax = -20, 20, 0, 40
#(a, b, c, d) = (xfigmin, xfigmax, yfigmin, yfigmax)
#[xx, xy] = np.meshgrid(np.linspace(xfigmin, xfigmax, nxgrid), np.linspace(yfigmin, yfigmax, nygrid))

hxgrid = 9
hsl = 1.2*height_source/2
a, b, c, d = (Dx-hsl/2, Dx+hsl/2, Dy, Dy+2*hsl) 
hygrid = np.round(hxgrid*(d-c)/(b-a))
nxgrid, nygrid = (2*hxgrid+1, 2*hygrid+1) # create a grid for visualisation purpose
[xx, xy] = np.meshgrid(np.linspace(a, b, nxgrid), np.linspace(c, d, nygrid))



(nxgrid, nygrid) = xx.shape
grid_points = np.asarray([xx.flatten(), xy.flatten()]).transpose()

Sil_grid = defmodsil.SilentLandmark(grid_points.shape[0], dim)

param_grid = (grid_points, np.zeros(grid_points.shape))
Sil_grid.GD.fill_cot_from_param(param_grid)

Mod_tot = comb_mod.CompoundModules([Sil_grid, Module_optimized])

path_fig = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/Implicit/'
# %%
Modlist_opti_tot_grid = shoot.shooting_traj(Mod_tot, N)
# %% Plot with grid
xs_c = my_close(xs)
xst_c = my_close(xst)
if(flag_save):
    for i in range(N + 1):
        plt.figure()
        xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
        xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
        xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
        plt.plot(xsx, xsy, color='lightblue')
        plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
        xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
        xs_ic = my_close(xs_i)
        # plt.plot(xs[:,0], xs[:,1], '-b', linewidth=1)
        plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
        plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
        plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
        plt.axis('equal')
        plt.axis([-60,60,-10,110])
        #plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
        plt.axis('off')
        #plt.show()
        plt.savefig(path_fig + name_exp + '_t_' + str(i) + '.png', format='png', bbox_inches='tight')
#%% save figure for last time
print(aa)
i=N
xlength = 8
ylength = 13
xmin = -40
xmax = 40
ymin = -10
ymax = 120
plt.figure(figsize=(xlength,ylength))
xgrid = Modlist_opti_tot_grid[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')
xs_i = Modlist_opti_tot_grid[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
xs_c = my_close(xs)
plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
plt.axis('equal')
#plt.axis('off')
#plt.axis([-50,50,-10,120])
plt.axis([xmin, xmax, ymin, ymax])
#plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.axis('off')
plt.tight_layout()
if(flag_show):
    plt.show()
path_fig = '/Network/Servers/ldap.ann.jussieu.fr/Volumes/DATA/users/thesards/gris/Results/DeformationModule/Implicit/'
name_tot = name_exp
if use_target_variation:
    name_tot += '_variation'

plt.savefig(path_fig + name_tot + coeffs_str + '.pdf', bbox_inches = 'tight')
print(coeffs)
print(coeffs_str)
plt.close()

#%% Generate variation of target
if generate_variation_target:
    Sil_variation = defmodsil.SilentLandmark(xst.shape[0], dim)
    ps = np.zeros(xst.shape)
    param_sil_variation = (xst, ps)
    if(flag_show):
        my_plot(xst, "Silent Module", '*b')

    sig0 = 30.
    x0_variation = np.array([[0., 90.], [20., 60.]])
    Model0_variation  = defmod0.ElasticOrderO(sig0, x0_variation.shape[0], dim, coeffs[1], nu)
    p0 = 10*np.array([[2., 0.], [-1., 0.]])
    param_0_variation = (x0_variation, p0)
    
    if(flag_show):
        my_plot(x0_variation, "Module order 0", 'or')

    Module_variation = comb_mod.CompoundModules([Sil_variation, Model0_variation])
    Module_variation.GD.fill_cot_from_param([param_sil_variation, param_0_variation])  

    Modlist_variation= shoot.shooting_traj(Module_variation, N)

    xst_variation = Modlist_variation[-1].GD.GD_list[0].GD
    xst_variation_c = my_close(xst_variation)
    
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xst_variation_c[:, 0], xst_variation_c[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    np.savetxt('../data/basi_target_variation', xst_variation)


# %% Shooting from controls
indi_modules = [0, 3]

Contlist = []
for i in range(len(Modules_list)):
    # First 0 for the SilentGrid module
    Contlist.append([0,[Modules_list[i].Cont[j] for j in indi_modules ]])

# %%
#Mod_cont_init = Modules_list[0].copy_full()
Mod_cont_init = comb_mod.CompoundModules([Modlist_opti_tot_grid[0].ModList[0].copy_full(), comb_mod.CompoundModules([Modules_list[0].ModList[j].copy_full() for j in indi_modules])])
#Mod_cont_init.GD.fill_cot_from_param([param_sil, param_1])
Modlist_cont = shoot.shooting_from_cont_traj(Mod_cont_init, Contlist, N)

# %% Visualisation and save for last figure
xst_c = my_close(xst)
xs_c = my_close(xs)
i=N
plt.figure(figsize=(xlength,ylength))
xgrid = Modlist_cont[2 * i].GD.GD_list[0].GD
xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
plt.plot(xsx, xsy, color='lightblue')
plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')

xs_i = Modlist_cont[2 * i].GD.GD_list[1].GD_list[0].GD
xs_ic = my_close(xs_i)
plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
#x00_i = Modlist_cont[2 * i].GD.GD_list[1].GD_list[1].GD
#plt.plot(x00_i[:, 0], x00_i[:, 1], 'xr')
plt.axis('equal')
#plt.axis('off')
#plt.axis([-50,50,-10,120])
plt.axis([xmin, xmax, ymin-40, ymax])
#plt.axis([xfigmin, xfigmax, yfigmin, yfigmax])
plt.axis('off')
plt.tight_layout()
plt.show()
#plt.savefig(path_fig + name_tot + 'follow_controls' + '.pdf', bbox_inches = 'tight')

# %% Visualisation
xst_c = my_close(xst)
xs_c = my_close(xs)
for i in range(N + 1):
    plt.figure()
    xgrid = Modlist_cont[2 * i].GD.GD_list[0].GD
    xsx = xgrid[:, 0].reshape((nxgrid, nygrid))
    xsy = xgrid[:, 1].reshape((nxgrid, nygrid))
    plt.plot(xsx, xsy, color='lightblue')
    plt.plot(xsx.transpose(), xsy.transpose(), color='lightblue')

    xs_i = Modlist_cont[2 * i].GD.GD_list[1].GD_list[0].GD
    xs_ic = my_close(xs_i)
    plt.plot(xst_c[:, 0], xst_c[:, 1], '-k', linewidth=1)
    plt.plot(xs_ic[:, 0], xs_ic[:, 1], '-g', linewidth=2)
    plt.plot(xs_c[:, 0], xs_c[:, 1], '-b', linewidth=1)
    plt.axis('equal')
    plt.show()

#%%

