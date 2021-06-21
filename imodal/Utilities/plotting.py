import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path

from imodal.Utilities import close_shape


def plot_closed_shape(shape, **kwargs):
    closed_shape = close_shape(shape)
    plt.plot(closed_shape[:, 0], closed_shape[:, 1], **kwargs)


def plot_grid(ax, gridx, gridy, **kwargs):
    """ Plot grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the grid will be drawn.
    gridx :
        Abscisse component of the grid that will be drawn.
    gridy :
        Ordinate component of the grid that will be drawn.
    kwargs : dict
        Keyword arguments that gets passed to the plot() functions.
    """
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i, :], gridy[i, :], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:, i], gridy[:, i], **kwargs)


def plot_C_arrows(ax, pos, C, R=None, c_index=0, scale=1., **kwargs):
    """ Plot growth constants as arrows.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the arrows will be drawn.
    pos : torch.Tensor
        Positions of the growth constants arrows.
    C : torch.Tensor
        Growth constants.
    R : torch.Tensor, default=None
        Local frame of each positions. If none, will assume idendity.
    c_index : int, default=0
        The dimension of the growth constants that will get drawn.
    scale : float, default=1.
        Scale applied to the arrow lengths.
    kwargs : dict
        Keyword arguments that gets passed to the underlying matplotlib plot
        functions.
    """
    for i in range(pos.shape[0]):
        C_i = scale*C[i, :, c_index]

        arrowstyle_x = "<->"
        arrowstyle_y = "<->"
        if C_i[0] <= 0.:
            arrowstyle_x += ",head_length=-0.4"
        if C_i[1] <= 0.:
            arrowstyle_y += ",head_length=-0.4"

        if R is not None:
            rotmat = R[i].numpy()
        else:
            rotmat = np.eye(2)

        top_pos = np.dot(rotmat, np.array([0., C_i[1]/2.])) + pos[i].numpy()
        bot_pos = np.dot(rotmat, -np.array([0., C_i[1]/2])) + pos[i].numpy()
        left_pos = np.dot(rotmat, -np.array([C_i[0]/2, 0.])) + pos[i].numpy()
        right_pos = np.dot(rotmat, np.array([C_i[0]/2, 0.])) + pos[i].numpy()

        ax.add_patch(FancyArrowPatch(left_pos, right_pos, arrowstyle=arrowstyle_x, **kwargs))
        ax.add_patch(FancyArrowPatch(bot_pos, top_pos, arrowstyle=arrowstyle_y, **kwargs))


def plot_C_ellipses(ax, pos, C, R=None, c_index=0, scale=1., **kwargs):
    """ Plot growth constants as ellipses.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which the ellipses will be drawn.
    pos : torch.Tensor
        Positions of the growth constants ellipses.
    C : torch.Tensor
        Growth constants.
    R : torch.Tensor, default=None
        Local frame of each positions. If none, will assume idendity.
    c_index : int, default=0
        The dimension of the growth constants that will get drawn.
    scale : float, default=1.
        Scale applied to the ellipses.
    kwargs : dict
        Keyword arguments that gets passed to the underlying matplotlib plot
        functions.
    """
    if R is not None:
        angle = torch.atan2(R[:, 1, 0], R[:, 0, 0])/math.pi*180.
    else:
        angle = torch.zeros(C.shape[0])

    for i in range(pos.shape[0]):
        C_i = scale*C[i, :, c_index]
        e = Ellipse(xy=pos[i], width=abs(C_i[0].item()), height=abs(C_i[1].item()), angle=angle[i].item(), **kwargs)
        a = 0.5*(1+torch.sign(C_i[0])).item()
        b = 0.5*(1+torch.sign(C_i[1])).item()
        # e.set_facecolor((0.5-0.25*(a+b), 0, 0.5+0.25*(a+b)))
        ax.add_artist(e)
        # ax.add_artist(Ellipse(xy=pos[i], width=abs(C_i[0].item()), height=abs(C_i[1].item()), angle=angle[i].item(), **kwargs))


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots. Equivalent to `plt.axis('equal')` in 3D."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def plot_aabb(ax, aabb, **kwargs):
    ax.add_artist(Rectangle((aabb.ymin, aabb.xmin), aabb.width, aabb.height, fill=False, **kwargs))


def plot_polyline(ax, polyline, close=False, **kwargs):
    codes = [Path.MOVETO]
    codes.extend([Path.LINETO]*(len(polyline)-1))

    polyline = polyline.tolist()

    if close:
        codes.append(Path.CLOSEPOLY)
        polyline.append((0., 0.))

    path = Path(polyline, codes)
    ax.add_artist(PathPatch(path, **kwargs))


def plot_basis3d(points, basis, length=0.1, **kwargs):
    plt.quiver(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), basis[:, 0, 0].numpy(), basis[:, 1, 0].numpy(), basis[:, 2, 0].numpy(), length=length, color='red', **kwargs)
    plt.quiver(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), basis[:, 0, 1].numpy(), basis[:, 1, 1].numpy(), basis[:, 2, 1].numpy(), length=length, color='green', **kwargs)
    plt.quiver(points[:, 0].numpy(), points[:, 1].numpy(), points[:, 2].numpy(), basis[:, 0, 2].numpy(), basis[:, 1, 2].numpy(), basis[:, 2, 2].numpy(), length=length, color='blue', **kwargs)


