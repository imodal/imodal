import numpy as np

from src.Utilities.visualisation import my_close



def my_dot_varifold(x, y, sig):
    cx, cy = my_close(x), my_close(y)
    nx, ny = x.shape[0], y.shape[0]
    
    vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
    mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2
    
    xy = np.tensordot(np.swapaxes(np.tensordot(mx, my, axes=0), 1, 2), np.eye(2))
    
    d2 = np.tile(np.sum(mx * mx, axis=1).reshape(nx, 1), (1, ny)) \
         + np.tile(np.sum(my * my, axis=1), (nx, 1)) - 2 * xy
    
    kxy = np.exp(-d2 / (2 * sig ** 2))
    
    vxvy = np.tensordot(np.swapaxes(np.tensordot(vx, vy, axes=0), 1, 2)
                        , np.eye(2)) ** 2
    
    nvx = np.sqrt(np.sum(vx * vx, axis=1))
    nvy = np.sqrt(np.sum(vy * vy, axis=1))
    
    mask = vxvy > 0
    
    cost = np.sum(kxy[mask] * vxvy[mask] / (np.tensordot(nvx, nvy, axes=0)[mask]))
    return cost


def my_dxdot_varifold(x, y, sig):
    cx, cy = my_close(x), my_close(y)
    nx, ny = x.shape[0], y.shape[0]
    
    vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
    mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2
    
    xy = np.tensordot(np.swapaxes(np.tensordot(mx, my, axes=0), 1, 2), np.eye(2))
    
    d2 = np.tile(np.sum(mx * mx, axis=1).reshape(nx, 1), (1, ny)) \
         + np.tile(np.sum(my * my, axis=1), (nx, 1)) - 2 * xy
    
    kxy = np.exp(-d2 / (2 * sig ** 2))
    
    vxvy = np.tensordot(np.swapaxes(np.tensordot(vx, vy, axes=0), 1, 2)
                        , np.eye(2)) ** 2
    
    nvx = np.sqrt(np.sum(vx * vx, axis=1))
    nvy = np.sqrt(np.sum(vy * vy, axis=1))
    
    mask = vxvy > 0
    
    u = np.zeros(vxvy.shape)
    u[mask] = vxvy[mask] / np.tensordot(nvx, nvy, axes=0)[mask]
    cost = np.sum(kxy[mask] * u[mask])
    
    dcost = np.zeros(x.shape)
    
    dcost1 = (-u * kxy) / (2 * sig ** 2)
    dcost1 = 2 * (np.tile(np.sum(dcost1, axis=1).reshape(nx, 1), (1, 2)) * mx
                  - np.tensordot(np.swapaxes(np.tensordot(dcost1, np.eye(2), axes=0), 1, 2), my))
    
    dcost += dcost1 / 2
    dcost[1:] += dcost1[0:-1] / 2
    dcost[0] += dcost1[-1] / 2
    
    u[mask] = kxy[mask] / np.tensordot(nvx, nvy, axes=0)[mask]
    dcost2 = 2 * u * np.tensordot(np.swapaxes(np.tensordot(vx, vy, axes=0), 1, 2)
                                  , np.eye(2))
    dcost2 = np.tensordot(np.swapaxes(np.tensordot(dcost2, np.eye(2), axes=0), 1, 2), vy)
    dcost += -dcost2
    dcost[1:] += dcost2[0:-1]
    dcost[0] += dcost2[-1]
    
    dcost3 = np.zeros(kxy.shape)
    u[mask] = kxy[mask] * vxvy[mask] / (np.tensordot(nvx, nvy, axes=0)[mask] ** 2)
    dcost3[mask] = -u[mask]
    dcost3 = np.dot(dcost3, nvy)
    tmp = np.zeros(nvx.shape)
    tmp[nvx > 0] = dcost3[nvx > 0] / (2 * nvx[nvx > 0])
    dcost3 = 2 * np.tile(tmp.reshape(vx.shape[0], 1), (1, vx.shape[1])) * vx
    dcost += - dcost3
    dcost[1:] += dcost3[0:-1]
    dcost[0] += dcost3[-1]
    
    return cost, dcost


def my_dxvar_cost(x, y, sig):
    """
    :param x:
    :param y:
    :param sig:
    :return: varifold cost and its derivative
    """
    (cost1, dxcost1) = my_dxdot_varifold(x, x, sig)
    (cost2, dxcost2) = my_dxdot_varifold(y, y, sig)
    (cost3, dxcost3) = my_dxdot_varifold(x, y, sig)
    return (cost1 + cost2 - 2 * cost3, 2 * dxcost1 - 2 * dxcost3)
