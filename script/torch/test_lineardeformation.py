{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "autoscroll": false,
    "collapsed": false,
    "ein.hycell": false,
    "ein.tags": "worksheet-0",
    "slideshow": {
     "slide_type": "-"
    }
   },
   "outputs": [],
   "source": [
    "%load_ext autoreload\n",
    "%autoreload 2\n",
    "\n",
    "import pickle\n",
    "import math\n",
    "\n",
    "# The deformation module library is not automatically installed yet, we need to add its path manually\n",
    "import sys\n",
    "sys.path.append(\"../../\")\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import torch\n",
    "\n",
    "import implicitmodules.torch as dm\n",
    "\n",
    "torch.set_default_tensor_type(torch.FloatTensor)\n",
    "dm.Utilities.set_compute_backend('torch')\n",
    "\n",
    "torch.manual_seed(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "autoscroll": false,
    "collapsed": false,
    "ein.hycell": false,
    "ein.tags": "worksheet-0",
    "slideshow": {
     "slide_type": "-"
    }
   },
   "outputs": [],
   "source": [
    "A = -torch.tensor([[0., -1.], [1., 0.]])\n",
    "#A = torch.randn(2, 2)\n",
    "gd = torch.randn(1, 2)\n",
    "cotan = torch.randn(1, 2)\n",
    "linear = dm.DeformationModules.LinearDeformation.build(A, gd=gd, cotan=cotan)\n",
    "linear.fill_controls(torch.tensor(4.))\n",
    "points = torch.randn(5, 2)\n",
    "print(linear(points))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "autoscroll": false,
    "collapsed": false,
    "ein.hycell": false,
    "ein.tags": "worksheet-0",
    "slideshow": {
     "slide_type": "-"
    }
   },
   "outputs": [],
   "source": [
    "aabb = dm.Utilities.AABB(-2., 2., -2., 2.)\n",
    "print(\"{width}:{height}\".format(width=aabb.width, height=aabb.height))\n",
    "\n",
    "width = 32\n",
    "height = 32\n",
    "gd_grid = aabb.fill([width, height])\n",
    "\n",
    "vector_field = linear(gd_grid).detach()\n",
    "\n",
    "print(vector_field)\n",
    "\n",
    "%matplotlib qt5\n",
    "plt.plot(gd[:, 0].numpy(), gd[:, 1].numpy(), 'o')\n",
    "plt.quiver(gd_grid.detach().numpy()[:, 0], gd_grid.detach().numpy()[:, 1], vector_field.numpy()[:, 0], vector_field.numpy()[:, 1], scale=1000.)\n",
    "plt.axis('equal')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "autoscroll": false,
    "collapsed": false,
    "ein.hycell": false,
    "ein.tags": "worksheet-0",
    "slideshow": {
     "slide_type": "-"
    }
   },
   "outputs": [],
   "source": [
    "linear.compute_geodesic_control(linear.manifold)\n",
    "print(linear.cost())\n",
    "print(linear.controls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "autoscroll": false,
    "collapsed": false,
    "ein.hycell": false,
    "ein.tags": "worksheet-0",
    "slideshow": {
     "slide_type": "-"
    }
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "name": "Untitled1.ipynb"
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
