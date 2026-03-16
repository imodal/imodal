import torch
import numpy as np
from scipy.interpolate import splprep, splev

main_points_source = torch.tensor([
    [0., 0.],
    [.5, .1],
    [1., 0.3],
    [1.5, .3],
    [2., .3],
    [2., .5],
    [2., .7],
    [1.5, .7],
    [1., .7],
    [.5, .9],
    [0., 1.],
    [0., .5],
    [0., 0.]
])

main_points_target = torch.tensor([
    [0., 0.3],
    [.5, .3],
    [1., 0.4],
    [1.5, .2],
    [2.2, .1],
    [2.1, .5],
    [2., .9],
    [1.5, .8],
    [1., .6],
    [.5, .7],
    [0., .7],
    [0., .5],
    [0., 0.3]
])

def bspline_smoothing(pts, s=1, density=100):
    x, y = pts[:, 0], pts[:, 1]
    t = np.arange(len(x))
    tck, u = splprep([x, y], s=s, per=True, k=5)
    u_fine = np.linspace(0, 1, density)
    x_smooth, y_smooth = splev(u_fine, tck)
    return np.stack([x_smooth, y_smooth], axis=-1)
        

def generate_divfree_data(device, dtype):
    smoothed_points_source =  torch.tensor(bspline_smoothing(main_points_source.cpu().numpy(), s=1/10000, density=200), device=device, dtype=dtype)
    smoothed_points_target =  torch.tensor(bspline_smoothing(main_points_target.cpu().numpy(), s=1/10000, density=200), device=device, dtype=dtype)
    me = torch.mean(smoothed_points_source, 0, keepdim=True)
    return smoothed_points_source - me, smoothed_points_target - me #- torch.mean(smoothed_points_target, 0, keepdim=True)
