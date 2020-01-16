import math
import sys
import pickle

import torch
import matplotlib.pyplot as plt

N_bunny = 1

def generate_bunny(head_radius, ear_widths, ear_lengths, ear_angles, nb_pts_head, nb_pts_ear):
    if head_radius <= 0. or ear_widths[0] <= 0. or ear_widths[1] <= 0. or ear_lengths[0] <= 0. or ear_lengths[1] <= 0.:
        return None
    if ear_widths[0]/head_radius >= 1. or ear_widths[1]/head_radius >= 1.:
        return None

    def generate_ear(ear_pos, angle, ear_a, ear_b, n_points):
        ear = torch.stack([torch.cos(torch.linspace(0., math.pi, n_points)),
                           torch.sin(torch.linspace(0., math.pi, n_points))], dim=1)
        ear = ear*torch.tensor([ear_b, ear_a]).repeat(n_points, 1)
        angle = angle - math.pi/2.
        rot_mat = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        ear = torch.bmm(rot_mat.repeat(n_points, 1, 1), ear.unsqueeze(1).transpose(1, 2)).view(-1, 2)
        ear = ear + ear_pos.repeat(n_points, 1)

        return ear

    ear_angle_intervals = [math.asin(ear_widths[0]/head_radius), math.asin(ear_widths[1]/head_radius)]
    d_alpha = math.pi/nb_pts_head
    alpha = 0.

    ear_generated = 0
    bunny = head_radius*torch.tensor([[1., 0.]])
    ear_tips = []
    alpha = d_alpha
    while alpha < 2.*math.pi-d_alpha:
        for ear_angle, ear_angle_interval, ear_length, ear_width in zip(ear_angles, ear_angle_intervals, ear_lengths, ear_widths):
            if alpha >= ear_angle - ear_angle_interval/2. and alpha < ear_angle + ear_angle_interval/2.:
                ear_pos = head_radius*torch.tensor([math.cos(alpha+ear_angle_interval/2.),
                                                    math.sin(alpha+ear_angle_interval/2.)])
                #ear_pos = ear_pos - ear_width*ear_width*0.005*ear_pos/torch.norm(ear_pos)
                ear = generate_ear(ear_pos, alpha + ear_angle_interval/2., ear_length, ear_width/2., nb_pts_head)
                bunny = torch.cat([bunny, ear])
                ear_generated = ear_generated + 1
                ear_tips.append(ear_pos+0.8*ear_length*ear_pos/torch.norm(ear_pos))
                alpha = alpha + ear_angle_interval
            else:
                bunny = torch.cat([bunny, head_radius*torch.tensor([[math.cos(alpha), math.sin(alpha)]])])
                alpha = alpha + d_alpha

    if ear_generated == len(ear_angles):
        return bunny, ear_tips

sigma_head_radius = 0.
sigma_ear_width = 0.
sigma_ear_length = 0.5
sigma_ear_angle = 0.

mu_head_radius = 0.8
mu_ear_width = 0.5
mu_ear_length = 1.8
mu_ear_angles = [2.*math.pi/3., math.pi/3.]

mu_bunny_pos = torch.tensor([0., 0.])

N_bunny = int(sys.argv[2])
herd = [generate_bunny(mu_head_radius, [mu_ear_width, mu_ear_width], [1., 1.], mu_ear_angles, 200, 100)]
for i in range(N_bunny):
    bunny = None
    while bunny is None:
        head_radius = sigma_head_radius*torch.randn(1).item()+mu_head_radius
        ear_widths = (sigma_ear_width*torch.randn(2)+mu_ear_width).tolist()
        ear_lengths = (sigma_ear_length*torch.randn(2)+mu_ear_length).tolist()
        ear_angles = (sigma_ear_angle*torch.randn(2)+torch.tensor(mu_ear_angles)).tolist()

        bunny, ear_tips = generate_bunny(head_radius, ear_widths, ear_lengths, ear_angles, 200, 100)

    angle = torch.rand(1).item()*math.pi*2.
    #angle = 0.
    trans = 0.1*torch.randn(2)
    rot_mat = torch.tensor([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])

    bunny = torch.bmm(rot_mat.repeat(bunny.shape[0], 1, 1), bunny.unsqueeze(1).transpose(1, 2)).view(-1, 2) + trans.repeat(bunny.shape[0], 1)
    
    herd.append((bunny, ear_tips))

pickle.dump(herd, open(sys.argv[1], 'wb'))

