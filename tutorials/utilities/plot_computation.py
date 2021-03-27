"""
Computations on IMODAL
======================

In this tutorial we show how to perform computations on GPU with KeOps.

KeOps, which stands for KErnelOPerationS, allow us to perform kernel matrix reductions.
"""

###############################################################################
# Import relevant Python modules.
#

import sys
sys.path.append("../")
import time

import torch

import imodal


###############################################################################
# Each deformation modules that needs kernel matrix reduction at some point has a plain Pytorch and KeOps implementation.
#
# In order to select once and for all the computation backend one can simply call
#

imodal.Utilities.set_compute_backend('keops')


###############################################################################
# Available compute backend are thus **keops** and **torch**. All subsequently deformation modules will use the specified computation backend.
#

d = 2
N = 10000
sigma = 1e-3

translations = imodal.DeformationModules.Translations(d, N, sigma)

print(translations.backend)


###############################################################################
# .. warning::
#   Changing computation backend after initializing modules will not affects settings for already created modules and will have to be initialized again.
#


###############################################################################
# It is also possible to explicitely set the compute backend for a deformation module using the **backend** keyword.
#

keops_translations = imodal.DeformationModules.Translations(d, N, sigma, backend='keops')

torch_translations = imodal.DeformationModules.Translations(d, N, sigma, backend='torch')


###############################################################################
#
#

positions = torch.randn(N, d)
keops_translations.manifold.fill_gd(positions)
keops_translations.to_(device='cuda')

torch_translations = imodal.DeformationModules.Translations(d, N, sigma, positions.to(device='cuda'), backend='torch')

print(keops_translations.backend)
print(torch_translations.backend)


###############################################################################
# .. note::
#   As IMODAL is built on top of Pytorch, the device parameter can either be a string or a torch.device object.
#   It is thus possible to select the GPU on which to perform computations by putting **device='cuda:x'** where **x** specify the GPU index (such as given by the **nvidia-smi** command.
#
# .. warning::
#   For now, when choosing the second method to select the computing device, susequent manifold filling with a different device will lead in an degenerate states which will ultimately fail. It is thus best to use the **to_()** method.
#
# .. note::
#  The **to_()** method can also be used to change the dtype of the module tensors, in exacly the same way as with Pytorch.
#


###############################################################################
# We now compare performance for 
#

points = torch.randn(N, d, device='cuda')

start = time.perf_counter()
keops_translations(points)
print("KeOps backend on GPU, elapsed timed={}".format(time.perf_counter() - start))

start = time.perf_counter()
torch_translations(points)
print("Pytorch backend on GPU, elapsed timed={}".format(time.perf_counter() - start))

