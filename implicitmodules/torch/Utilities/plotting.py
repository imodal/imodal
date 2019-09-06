import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

def plot_tensor_scatter(x, alpha=1., scale=64.):
    """Scatter plot points in the format: ([x, y], scale) or ([x, y]) (in that case you can specify scale)"""
    if (isinstance(x, tuple) or isinstance(x, list)):
        plt.scatter(x[0].detach().numpy()[:, 1], x[0].detach().numpy()[:, 0],
                    s=50. * x[1].shape[0] * x[1].detach().numpy(), marker='o', alpha=alpha)
        plt.scatter(x[0].detach().numpy()[:, 1], x[0].detach().numpy()[:, 0], s=64. * x[1].shape[0] * x[1], marker='o',
                    alpha=alpha)
    else:
        plt.scatter(x.detach().numpy()[:, 1], x.detach().numpy()[:, 0], s=scale, marker='o', alpha=alpha)


def plot_grid(ax, gridx, gridy, **kwargs):
    for i in range(gridx.shape[0]):
        ax.plot(gridx[i, :], gridy[i, :], **kwargs)
    for i in range(gridx.shape[1]):
        ax.plot(gridx[:, i], gridy[:, i], **kwargs)


def plot_C_arrow(ax, pos, C, R=None, c_index=0, scale=1., **kwargs):
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


def plot_C_ellipse(ax, pos, C, R=None, c_index=0, scale=1., **kwargs):
    if R is not None:
        angle = torch.atan2(R[:, 1, 0], R[:, 0, 0])/math.pi*180.
    else:
        angle = torch.zeros(C.shape[0])

    for i in range(pos.shape[0]):
        C_i = scale*C[i, :, c_index]
        ax.add_artist(Ellipse(xy=pos[i], width=abs(C_i[0].item()), height=abs(C_i[1].item()), angle=angle[i].item(), **kwargs))

