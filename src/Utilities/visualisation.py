import matplotlib.pyplot as plt
import numpy as np


def my_close(x):
    N = x.shape[0]
    z = np.zeros((N + 1, 2))
    z[0:N, :] = x
    z[N, :] = x[0, :]
    return z


def my_plot4(Mod0, Mod1, Cot, fig, nx, ny, name, i):
    x0 = Mod0['0']
    plt.axis('equal')
    x0 = my_close(x0)
    
    # xs = Cot['0'][1][0][0:nx*ny,:]
    # xsx = xs[:,0].reshape((nx,ny))
    # xsy = xs[:,1].reshape((nx,ny))
    # plt.plot(xsx, xsy,color = 'lightblue')
    # plt.plot(xsx.transpose(), xsy.transpose(),color = 'lightblue')
    (x1, R) = Mod1['x,R']
    plt.plot(x1[:, 0], x1[:, 1], 'b.')
    plt.plot(x0[:, 0], x0[:, 1], 'r.')
    
    xs = Cot['0'][1][0]  # [nx*ny:,:]
    xs = my_close(xs)
    plt.plot(xs[:, 0], xs[:, 1], 'g')
    
    plt.ylim(-40, 80)
    # plt.show()
    plt.savefig(name + '{:02d}'.format(i) + '.png')
    return
