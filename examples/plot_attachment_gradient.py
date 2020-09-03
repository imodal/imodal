"""
Attachment gradient
===================

"""

###############################################################################
# Import relevant modules.
#

import pickle
import sys

import torch
import matplotlib.pyplot as plt
import scipy.interpolate

sys.path.append("../")

import implicitmodules.torch as dm


###############################################################################
# Function that plots our gradients
#

def plot_attachment_gradient(source, target, attachment):
    gradients = dm.Attachment.compute_attachment_gradient(source, target, attachment)

    plt.plot(source[:, 0].numpy(), source[:, 1].numpy(), color='black')
    plt.plot(source[:, 0].numpy(), source[:, 1].numpy(), 'x', color='black', )
    plt.plot(target[:, 0].numpy(), target[:, 1].numpy(), color='blue')
    plt.quiver(source[:, 0].numpy(), source[:, 1].numpy(), -gradients[:, 0].numpy(), -gradients[:, 1].numpy())


###############################################################################
# Load dataset
#

data = pickle.load(open("../data/peanuts.pickle", 'rb'))

peanuts = [torch.tensor(peanut[:-1], dtype=torch.get_default_dtype()) for peanut in data[0]]

template = dm.Utilities.generate_unit_circle(200)
template = dm.Utilities.linear_transform(template, torch.tensor([[1.3, 0.], [0., 0.5]]))
template = dm.Utilities.close_shape(template)

source = peanuts[4]
target = peanuts[2]

# # resampled = dm.Utilities.resample_curve(source, 1.)
# # resampled = scipy.interpolate.splprep([source[:, 0].numpy(), source[:, 1].numpy()], k=1)
# resampled = scipy.interpolate.SmoothBivariateSpline(source[:, 0].numpy(), source[:, 1].numpy(), kx=1, ky=1)
# print(resampled)
# exit()

###############################################################################
# Varifold attachment
#

sigmas = [[1.], [10.], [0.5, 5.], [0.1, 0.5, 2., 6.]]

for i, sigma in enumerate(sigmas):
    plt.subplot(2, 2, i + 1)
    plt.title("Varifold attachment with scales set to {sigma}".format(sigma=sigma))
    plot_attachment_gradient(source, target, dm.Attachment.VarifoldAttachment(2, sigma))
plt.show()


###############################################################################
# Energy attachment
#

plt.title("Energy attachment")
plot_attachment_gradient(source, target, dm.Attachment.EnergyAttachment())
plt.show()

