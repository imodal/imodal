import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rd
from scipy.linalg import solve
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd
from scipy.linalg import solve

plt.show()
plt.ion()


## Unitary test (passed)
## my_dv0 (passed)
M0 = Exp0['M0']
h = rd.normal(0, 1, (M0['N'], 2))
M0['h'] = h
zp = z + 0.00001 * rd.normal(0, 1, z.shape)
Dz = zp - z
Dv0 = my_v0(M0, zp, h) - my_v0(M0, z, h)
dv0 = my_dv0(M0, z, h)
print(np.dot(dv0[0], Dz[0]))
print(Dv0[0])

## my_dv0 an other test (passed)
Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
sig = 0.1
x = rd.normal(0, 1, (Nx, 2))
z = rd.normal(0, 1, (Nz, 2))
zp = z + 0.000001 * rd.normal(0, 1, (Nz, 2))
px = rd.normal(0, 1, (Nx, 2))

Dz = zp - z

M0['x'] = x
M0['sig'] = sig
M0['N'] = Nx
M0['h'] = px
Dv0 = my_v0(M0, zp, px) - my_v0(M0, z, px)
dv0 = my_dv0(M0, z, px)
print(np.dot(dv0[0], Dz[0]))
print(Dv0[0])



## my_CotDotV,


## Comparing my_SKS with my_new_SKS (passed)

x = rd.normal(0, 1, (12, 2))
th = np.zeros(13)
R = np.asarray([my_R(cth) for cth in th])
sig = 0.4
M1 = {'x': x, 'N': x.shape[0], 'sig': sig}
Mod1 = {'x,R': (x, R), 'sig': sig}
print(my_SKS_M1(M1) - my_new_SKS(Mod1))

## Checking my_Am

n0 = (2, 2)  # number of order 0
n1 = (4, 5)  # number of point of order 1
dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
sig = (0.25, 0.7)  # sigma's for K0 and K1
th = 0 * np.pi
h = [0., 0.5]
Mod0, Mod1 = my_new_exp0(dom, n0, n1, th, sig)
(x, P) = my_hToP(Mod1, h)

Vs = {'p': [(x, P)], 'sig': Mod1['sig']}
v = my_VsToV(Vs, x, 0)

fig = plt.figure(1)
fig.clf()

plt.axis('equal')
Q = plt.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1],
               units='xy', scale=1., zorder=0, headwidth=5.,
               width=0.005, headlength=5., color='blue')

xb = my_border(dom, n1)
vxb = my_VsToV(Vs, xb, 0)

plt.plot(xb[:, 0] + vxb[:, 0], xb[:, 1] + vxb[:, 1], color='red')

plt.ylim(-1, 3)
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
x = rd.normal(0, 1, (Nx, 2))
xp = rd.normal(0, 1, (Nxp, 2))
xm = rd.normal(0, 1, (Nxm, 2))
z = rd.normal(0, 1, (Nz, 2))
zp = z + 0.000001 * rd.normal(0, 1, (Nz, 2))
px = rd.normal(0, 1, (Nx, 2))
pxm = rd.normal(0, 1, (Nxm, 1))
pxp = rd.normal(0, 1, (Nxp, 3))

Dz = zp - z

M0['x'] = x
M0['sig'] = sig
M0['N'] = Nx
M1['x'] = xp
M1['sig'] = sig
M1['N'] = Nxp
vx, vxp, vxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, z)
print(vx - my_v0(M0, z, px))
print(vxp - my_v1(M1, z, pxp.reshape(3 * Nxp)))

dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
print(dvx - my_dv0(M0, z, px))
print(dvxp - my_dv1(M1, z, pxp.reshape(3 * Nxp)))

## my_gen_dv (passed)
Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
sig = 0.1
x = rd.normal(0, 1, (Nx, 2))
xp = rd.normal(0, 1, (Nxp, 2))
xm = rd.normal(0, 1, (Nxm, 2))
z = rd.normal(0, 1, (Nz, 2))
zp = z + 0.0001 * rd.normal(0, 1, (Nz, 2))
Dz = zp - z

px = rd.normal(0, 1, (Nx, 2))
pxm = rd.normal(0, 1, (Nxm, 1))
pxp = rd.normal(0, 1, (Nxp, 3))

vx, vxp, vxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, z)
nvx, nvxp, nvxm = my_gen_v(x, xp, xm, px, pxp, pxm, sig, zp)
dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
Dvx = nvx - vx
Dvxp = nvxp - vxp
Dvxm = nvxm - vxm
print(np.dot(dvx[0], Dz[0]))
print(Dvx[0])
print(np.dot(dvxp[0], Dz[0]))
print(Dvxp[0])
print(np.dot(dvxm[0], Dz[0]))
print(Dvxm[0])

## my_gen_ddv (passed)
Nx, Nxp, Nxm, Nz = 10, 11, 12, 13
sig = 0.1
x = rd.normal(0, 1, (Nx, 2))
xp = rd.normal(0, 1, (Nxp, 2))
xm = rd.normal(0, 1, (Nxm, 2))
z = rd.normal(0, 1, (Nz, 2))
zp = z + 0.00001 * rd.normal(0, 1, (Nz, 2))
Dz = zp - z

px = rd.normal(0, 1, (Nx, 2))
pxm = rd.normal(0, 1, (Nxm, 1))
pxp = rd.normal(0, 1, (Nxp, 3))

dvx, dvxp, dvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, z)
ndvx, ndvxp, ndvxm = my_gen_dv(x, xp, xm, px, pxp, pxm, sig, zp)
ddvx, ddvxp, ddvxm = my_gen_ddv(x, xp, xm, px, pxp, pxm, sig, z)
Ddvx = ndvx - dvx
Ddvxp = ndvxp - dvxp
Ddvxm = ndvxm - dvxm
print(np.dot(ddvx[0], Dz[0]))
print(Ddvx[0])
print(np.dot(ddvxp[0], Dz[0]))
print(Ddvxp[0])
print(np.dot(ddvxm[0], Dz[0]))
print(Ddvxm[0])

##  my_gen_ximv1, my_gen_pximv1, my_gen_dmpximv1 (passed)

Nx, Nxp, Ny, Nyp, Nym = 10, 11, 12, 13, 14
sig = 0.1
x, px = rd.normal(0, 1, (Nx, 2)), rd.normal(0, 1, (Nx, 2))
xp, pxp = rd.normal(0, 1, (Nxp, 2)), rd.normal(0, 1, (Nxp, 3))
y, py = rd.normal(0, 1, (Ny, 2)), rd.normal(0, 1, (Ny, 2))
yp, pyp = rd.normal(0, 1, (Nyp, 2)), rd.normal(0, 1, (Nyp, 3))
ym, pym = rd.normal(0, 1, (Nym, 2)), rd.normal(0, 1, (Nym, 1))

C = 0.0001
ny = y + C * rd.normal(0, 1, (Ny, 2))
nyp = yp + C * rd.normal(0, 1, (Nyp, 2))
nym = ym + C * rd.normal(0, 1, (Nym, 2))

e = my_gen_pximv1(x, xp, px, pxp, sig, y, yp, ym, py, pyp, pym)
ne = my_gen_pximv1(x, xp, px, pxp, sig, ny, nyp, nym, py, pyp, pym)
De = ne - e
Dy, Dyp, Dym = ny - y, nyp - yp, nym - ym
dpy, dpyp, dpym = my_gen_dmpximv1(x, xp, px, pxp, sig, y, yp, ym, py, pyp, pym)
print(np.tensordot(dpy, Dy) + np.tensordot(dpyp, Dyp) + np.tensordot(dpym, Dym))
print(De)

## my dxpv0(M0,z,alpha,pz) (passed)

M0 = Exp0['M0']
h = rd.normal(0, 1, (M0['N'], 2))
pz = rd.normal(0, 1, z.shape)
M0['h'] = h
M0p = dict(M0)
M0p['x'] = M0['x'] + 0.00001 * rd.normal(0, 1, M0['x'].shape)

x, xp = M0['x'], M0p['x']
M0['v'] = my_v0(M0, z, h)
M0p['v'] = my_v0(M0p, z, h)

Dx = xp - x
Dpv0 = np.tensordot(pz, M0p['v'] - M0['v'], axes=2)
dxpv0 = my_dxpv0(M0, z, h, pz)
dpv0 = 0.
for i in range(M0['N']):
    dpv0 += np.dot(dxpv0[i], Dx[i])
print(Dpv0)
print(dpv0)

## my_dzdv1 (passed)
h = np.array([0.0, -0.1])
lam = my_lam_M1(M1, h)
zp = z + 0.0001 * rd.normal(0, 1, z.shape)
Dz = zp - z

Dzdv1 = my_dv1(M1, zp, lam) - my_dv1(M1, z, lam)
dzdv1 = my_dzdv1(M1, z, lam)
print(np.dot(dzdv1[10], Dz[10]))
print(Dzdv1[10])

## Miscellenea
u = rd.normal(0, 1, (2, 2))
s = np.tensordot(u, my_eta(), axes=2)
print(u)
print(np.dot(my_eta(), s))

## Consistence test between Am(h) and (dv1+dv1t)/2 (passed)
M1 = Exp0['M1']
h = np.array([0.0, 0.2])
lam = my_lam_M1(M1, h)
M1['lam'] = lam
M1['h'] = h
my_Am(M1)
print(M1['Am'])
dv = my_dv1(M1, M1['x'], lam)
sdv = np.tensordot((dv + np.swapaxes(dv, 1, 2)) / 2, my_eta(), axes=2)
print(M1['Am'] - sdv)

## my_dener1 (gradient in x)  (passed)
M1 = Exp0['M1']
h = np.array([0.0, -0.1])
lam = my_lam_M1(M1, h)
M1['lam'] = lam
M1['h'] = h
my_Am(M1)

M1p = dict(M1)
M1p['x'] = M1['x'] + 0.00001 * rd.normal(0, 1, M1['x'].shape)
lamp = my_lam_M1(M1p, h)
M1p['lam'] = lamp
my_Am(M1p)

x, xp = M1['x'], M1p['x']
Dx = xp - x
De = my_ener1(M1p) - my_ener1(M1)
my_dener1(M1)

de = 0
dxe = M1['dener']['x']
for i in range(M1['N']):
    de += np.dot(dxe[i], Dx[i])
print(De)
print(de)

## my_dener1 (gradient in th)  (passed)
M1 = Exp0['M1']
h = np.array([0.0, -0.1])
lam = my_lam_M1(M1, h)
M1['lam'] = lam
M1['h'] = h
my_Am(M1)

M1p = dict(M1)
M1p['th'] = M1['th'] + 0.000001 * rd.normal(0, 1, M1['th'].shape)
M1p['R'] = np.asarray([my_R(M1p['th'][i]) for i in range(M1p['N'])])
lamp = my_lam_M1(M1p, h)
M1p['lam'] = lamp
my_Am(M1p)

th, thp = M1['th'], M1p['th']
Dth = thp - th
De = my_ener1(M1p) - my_ener1(M1)
my_dener1(M1)

de = 0
dthe = M1['dener']['th']
for i in range(M1['N']):
    de += np.dot(dthe[i], Dth[i])
print(De)
print(np.sqrt(2) * de)

## my_dener0 (gradient in x) (passed)

M0 = Exp0['M0']
h = rd.normal(0, 1, (M0['N'], 2))
M0['h'] = h
M0p = dict(M0)
M0p['x'] = M0['x'] + 0.00001 * rd.normal(0, 1, M0['x'].shape)

x, xp = M0['x'], M0p['x']
M0['v'] = my_v0(M0, x, h)
M0p['v'] = my_v0(M0p, xp, h)

Dx = xp - x
De = my_ener0(M0p) - my_ener0(M0)
my_dener0(M0)

de = 0
dxe = M0['dener']['x']
for i in range(M0['N']):
    de += np.dot(dxe[i], Dx[i])
print(De)
print(de)
##

N = M0['N']
zp = z + 0.0001 * rd.normal(0, 1, z.shape)
alpha = 0.1 * rd.normal(0, 1, (N, 2))
v, vp = my_v0(M0, z, alpha), my_v0(M0, zp, alpha)
Dz, Dv = zp - z, vp - v
dv = my_dv0(M0, z, alpha)
test = [np.dot(dv[s], Dz[s]) - Dv[s] for s in range(z.shape[0])]

## my_dv1

zp = z + 0.0001 * rd.normal(0, 1, z.shape)
v, vp = my_v1(M1, z, lam), my_v1(M1, zp, lam)
Dz, Dv = zp - z, vp - v
dv = my_dv1(M1, z, lam)
test = [np.dot(dv[s], Dz[s]) - Dv[s] for s in range(z.shape[0])]

##


N = M0['N']
zp = z + 0.0001 * rd.normal(0, 1, z.shape)
alpha = 0.1 * rd.normal(0, 1, (N, 2))
M0['h'] = alpha
v, vp = my_v(M0, M1, z), my_v(M0, M1, zp)
Dz, Dv = zp - z, vp - v
dv = my_dv(M0, M1, z)
test = [np.dot(dv[s], Dz[s]) - Dv[s] for s in range(z.shape[0])]

##


M0, M1 = Exp0['M0'], Exp0['M1']
N = M0['N']
alpha = 0.01 * rd.normal(0, 1, (N, 2))
M0['h'] = alpha
h = np.array([0.0, -0.1])
lam = my_lam_M1(M1, h)
M1['lam'] = lam
M1['h'] = h
Exp = Exp0

x0, x1 = M0['x'], M1['x']
my_rho(Exp)
my_Am(M1)

# Energy computation
e1 = my_ener1(M1)
e0 = my_ener0(M0)
print(e1 + e0)

om, vx1, vx0 = M1['om'], M1['v'], M0['v']

fig = plt.figure(1)
fig.clf()

plt.axis('equal')
Q = plt.quiver(x1[:, 0], x1[:, 1], vx1[:, 0], vx1[:, 1],
               units='xy', scale=1., zorder=0, headwidth=5.,
               width=0.005, headlength=5., color='blue')

# QR = plt.quiver(x1[:,0],x1[:,1],nR[:,0,0], nR[:,0,1],
#     units='xy',scale = 1.,zorder=0,headwidth=5.,
#     width=0.005,headlength=5.,color='black')

QT = plt.quiver(x0[:, 0], x0[:, 1], vx0[:, 0], vx0[:, 1],
                units='xy', scale=1., zorder=0, headwidth=5.,
                width=0.01, headlength=5., color='red')
plt.show()

##
outline = Exp['outline']
plt.plot(outline[:, 0], outline[:, 1], 'b')

plt.plot(xv + vz[0:nx * ny, 0].reshape((ny, nx)), yv + vz[0:nx * ny, 1].reshape((ny, nx)), color='green')
plt.plot(np.transpose(xv + vz[0:nx * ny, 0].reshape((ny, nx))), np.transpose(yv + vz[0:nx * ny, 1].reshape((ny, nx))),
         color='green')

plt.ylim(-1, 3)
plt.show()

test = [np.dot(dv[s], Dz[s]) - Dv[s] for s in range(z.shape[0])]
## Test 4  using ptov


n0 = (2, 2)  # number of order 0
n1 = (4, 5)  # number of point of order 1
dom = (-0.5, 0.5, 0., 2)  # domain to build the grid
sig = (0.25, 0.7)  # sigma's for K0 and K1
th = np.pi / 4

coeff_mixed = np.array([0.1, 10])  # Metric weight order 0 and 1
Exp0 = my_exp(dom, n0, n1, th, sig, coeff_mixed)

coeff_mixed = np.array([100, 0.1])  # Metric weight order 0 and 1
Exp1 = my_exp(dom, n0, n1, th, sig, coeff_mixed)

M = Exp0['M1']
C = np.zeros((M['N'], 2, 2))
c = np.zeros((M['N'], 2))
x = M['x']
for k in range(n1[0]):
    c[x[:, 0] == x[k, 0], :] \
        = (n1[0] - k) * np.array([1., 1.]) / n1[0]
tmp = np.zeros((2, 2, 2))
tmp[0, 0, 0] = 1.
tmp[1, 1, 1] = 1.
C = np.tensordot(c, tmp, axes=1)
M.setdefault('C', C)
Exp1['M1'].setdefault('C', C)

##
ptov = my_ptov(Exp0['M0'], Exp0['M1'], Exp0['coeff_mixed'])
Exp0.setdefault('ptov', ptov)

ptov = my_ptov(Exp1['M0'], Exp1['M1'], Exp1['coeff_mixed'])
Exp1.setdefault('ptov', ptov)

##
# construction of the visualisation grid
nx, ny = (6, 12)
(ax, bx, ay, by) = dom
xv, yv = np.meshgrid(np.linspace(ax, bx, nx), np.linspace(ay, by, ny))
z = np.zeros((nx * ny, 2))
z[:, 0] = xv.reshape((nx * ny))
z[:, 1] = yv.reshape((nx * ny))
z = np.append(z, Exp0['outline'], axis=0)

##

(Nx, Ny) = n0

for i in range(1):
    Exp = Exp1
    M0, M1, coeff_mixed = Exp['M0'], Exp['M1'], Exp['coeff_mixed']
    x0 = M0['x']
    ptov = Exp['ptov']
    
    # h = [0.0, 0.5]
    h = 0.3 * rd.normal(0, 1, (2))
    # lam = my_lam_M1(M1,h)
    v = my_KS_M1(M1, x0, my_lam_M1(M1, h))
    
    for k in range(2):
        
        if k == 0:
            Exp = Exp0
        elif k == 1:
            Exp = Exp1
        
        p = solve(Exp['ptov'], v.reshape(4 * (Nx + Ny) - 8))
        lam = my_lam_M1(M1, my_ptoh(M0, M1, p))
        
        coeff_mixed = Exp['coeff_mixed']
        
        vz = my_KS_M1(M1, z, lam / coeff_mixed[1]) \
             + my_KS_M0(M0, z, p / coeff_mixed[0])
        
        nv = my_KS_M1(M1, x0, lam / coeff_mixed[1]) \
             + my_KS_M0(M0, x0, p / coeff_mixed[0])
        
        fig = plt.figure(0)
        
        # Plot of the result
        
        if k == 0:
            fig.clf()
            plt.subplot(121)
        else:
            plt.subplot(122)
        
        plt.axis('equal')
        Q = plt.quiver(xv.reshape((nx * ny)),
                       yv.reshape((nx * ny)), vz[0:nx * ny, 0], vz[0:nx * ny, 1],
                       units='xy', scale=1., zorder=0, headwidth=5.,
                       width=0.005, headlength=5., color='blue')
        
        QT = plt.quiver(x0[:, 0], x0[:, 1], v[:, 0], v[:, 1],
                        units='xy', scale=1., zorder=0, headwidth=5.,
                        width=0.01, headlength=5., color='red')
        
        outline = Exp['outline']
        plt.plot(outline[:, 0], outline[:, 1], 'b')
        
        plt.plot(xv + vz[0:nx * ny, 0].reshape((ny, nx)), yv + vz[0:nx * ny, 1].reshape((ny, nx)), color='green')
        plt.plot(np.transpose(xv + vz[0:nx * ny, 0].reshape((ny, nx))),
                 np.transpose(yv + vz[0:nx * ny, 1].reshape((ny, nx))), color='green')
        
        plt.ylim(-1, 3)
        plt.show()
        plt.savefig('plotd' + str(i) + '.png')
