import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path
import math


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
        ax.add_artist(Ellipse(xy=pos[i], width=abs(C_i[0].item()), height=abs(C_i[1].item()), angle=angle[i].item(), **kwargs))


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

