import torch


def linear_transform(points, A):
    return torch.bmm(A.repeat(points.shape[0], 1, 1), points.unsqueeze(1)).view_as(points)


def generate_unit_square(points_per_side, dtype=None):
    interval = 1./points_per_side
    points = torch.zeros(4*points_per_side, 2, dtype=dtype)
    points[0:points_per_side, 0] = torch.linspace(0., 1., points_per_side, dtype=dtype)
    points[points_per_side:2*points_per_side, 1] = torch.linspace(0., 1., points_per_side, dtype=dtype)    
    points[points_per_side-1:2*points_per_side, 0] = 1.
    points[2*points_per_side:3*points_per_side, 0] = torch.linspace(1., 0., points_per_side, dtype=dtype)
    points[2*points_per_side:3*points_per_side, 1] = 1.
    points[3*points_per_side:4*points_per_side, 1] = torch.linspace(1., 0., points_per_side, dtype=dtype)
    # interval = 1./points_per_side
    # points = torch.zeros(4*points_per_side, 2, dtype=dtype)
    # points[0:points_per_side, 0] = torch.linspace(0., 1.-interval, points_per_side, dtype=dtype)
    # points[points_per_side:2*points_per_side, 1] = torch.linspace(0., 1.-interval, points_per_side, dtype=dtype)    
    # points[points_per_side-1:2*points_per_side, 0] = 1.
    # points[2*points_per_side:3*points_per_side, 0] = torch.linspace(1.-interval, 0., points_per_side, dtype=dtype)
    # points[2*points_per_side:3*points_per_side, 1] = 1.
    # points[3*points_per_side:4*points_per_side, 1] = torch.linspace(1.-interval, 0., points_per_side, dtype=dtype)

    # interval = 1./points_per_side
    # points = torch.zeros(4*points_per_side, 2, dtype=dtype)
    # #points[0:points_per_side, 0] = torch.linspace(0., 1.-interval, points_per_side-1, dtype=dtype)
    # points[0:points_per_side, 0] = torch.arange(0., 1.-interval, interval, dtype=dtype)
    # #points[points_per_side:2*points_per_side, 1] = torch.linspace(0., 1.-interval, points_per_side-1, dtype=dtype)
    
    # points[points_per_side-1:2*points_per_side, 0] = 1.
    # points[2*points_per_side:3*points_per_side, 0] = torch.linspace(1.-interval, 0., points_per_side-1, dtype=dtype)
    # points[2*points_per_side:3*points_per_side, 1] = 1.
    # points[3*points_per_side:4*points_per_side, 1] = torch.linspace(1.-interval, 0., points_per_side-1, dtype=dtype)


    return points - torch.tensor([0.5, 0.5])

