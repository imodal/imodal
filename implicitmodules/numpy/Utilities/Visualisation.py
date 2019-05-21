import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


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


# helper function
def my_plot(x, ellipse=[], angles=[], title="", col='*b'):
    _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    aabb = implicitmodules.torch.Utilities.usefulfunctions.AABB.build_from_points(x)
    aabb.squared()
    
    ax.set_aspect('equal')
    ax.axis(aabb.get_list())
    
    
    if ellipse == []:
        ax.plot(x[:, 0], x[:, 1], col)
    
    else:
        if angles == []:
            angles = np.zeros([x.shape[0], 1])
            
        ells = [Ellipse(xy=x[i, :],
                        width=ellipse[i, 0, 0] * .3, height=ellipse[i, 1, 0] * .3,
                        angle=np.rad2deg(angles[i]))
                for i in range(x.shape[0])]
        
        for e in ells:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(1)
            e.set_facecolor('g')
    
    # ax.set_xlim(m[0] - (M[0] - m[0]) * .2, M[0] + (M[0] - m[0]) * .2)
    # ax.set_ylim(m[1] - (M[1] - m[1]) * .2, M[1] + (M[1] - m[1]) * .2)
    
    
    
    plt.title(title)
    plt.show()