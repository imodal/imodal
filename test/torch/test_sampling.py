import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch
import numpy as np

import implicitmodules.torch as im

class TestSampling(unittest.TestCase):
    def test_load_greyscale(self):
        img_8 = im.Utilities.sampling.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png")
        self.assertIsInstance(img_8, torch.Tensor)
        self.assertEqual(img_8.shape, torch.Size([16, 16]))

        # load_greyscale_image should be able to handle multichannel images
        # (and simply take the red channel)
        img_24 = im.Utilities.sampling.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_24.png")
        self.assertEqual(img_24.shape, torch.Size([16, 16]))

    def test_sample_from_greyscale(self):
        img = im.Utilities.sampling.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png")
    
        points = im.Utilities.sampling.sample_from_greyscale(img, 0.)
        self.assertIsInstance(points, tuple)
        self.assertEqual(len(points), 2)
        self.assertIsInstance(points[0], torch.Tensor)
        self.assertIsInstance(points[1], torch.Tensor)
        self.assertEqual(points[0].shape, torch.Size([16*16, 2]))
        self.assertEqual(points[1].shape, torch.Size([16*16]))
    
        points = im.Utilities.sampling.sample_from_greyscale(img, 1.1)
        self.assertEqual(points[0].shape, torch.Size([0, 2]))
    
        points = im.Utilities.sampling.sample_from_greyscale(img, 0.2, centered=True)
        self.assertTrue(torch.allclose(torch.sum(points[0], dim=0), torch.zeros(1, 2)))
    
        points = im.Utilities.sampling.sample_from_greyscale(img, 0.2, normalise_weights=True)
        self.assertTrue(torch.allclose(torch.sum(points[1]), torch.tensor(1.)))

    def test_load_and_sample_greyscale(self):
        points = im.Utilities.sampling.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png", threshold=0.)
        self.assertIsInstance(points, tuple)
        self.assertEqual(len(points), 2)
        self.assertIsInstance(points[0], torch.Tensor)
        self.assertIsInstance(points[1], torch.Tensor)
        self.assertEqual(points[0].shape, torch.Size([16*16, 2]))
        self.assertEqual(points[1].shape, torch.Size([16*16]))
    
        points = im.Utilities.sampling.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png", threshold=1.1)
        self.assertEqual(points[0].shape, torch.Size([0, 2]))
    
        points = im.Utilities.sampling.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png",
            0.2, centered=True)
        self.assertTrue(torch.allclose(torch.sum(points[0], dim=0), torch.zeros(1, 2)))
    
        points = im.Utilities.sampling.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png",
            0.2, normalise_weights=True)
        self.assertTrue(torch.allclose(torch.sum(points[1]), torch.tensor(1.)))

    def test_sample_from_points_2d(self):
        dim = 2
        square_size = 32
        m = 100
        points = float(square_size)*torch.rand(m, dim)
        alpha = torch.rand(m)
        frame_res = torch.Size([square_size, square_size])

        frame_out = im.Utilities.sampling.sample_image_from_points((points, alpha), frame_res)

        self.assertIsInstance(frame_out, torch.Tensor)
        self.assertEqual(frame_out.shape, frame_res)
        self.assertTrue(torch.allclose(torch.sum(frame_out), torch.sum(alpha)))

    def test_kernel_smoother(self):
        # We compare a fast implementation of the kernel smoother with a naive,
        # tested implementation
        dim = 2
        m = 100
        sigma = 1.
        square_size = 32
        frame_res = torch.Size([square_size, square_size])
        points = square_size*torch.rand(m, dim), torch.rand(m)

        x, y = torch.meshgrid([torch.arange(0., square_size, step=1.),
                               torch.arange(0., square_size, step=1.)])

        pos = im.Utilities.usefulfunctions.grid2vec(x, y)

        def kernel(x, sigma):
            return np.exp(-0.5*x**2/sigma**2)

        naive = torch.zeros(pos.shape[0])
        for i in range(0, pos.shape[0]):
            naive[i] = float(np.sum(kernel(np.linalg.norm((pos[i,:].expand_as(points[0]) - points[0]).numpy(), axis=1), sigma)*points[1].numpy()))

        implementation = im.Utilities.sampling.kernel_smoother(pos, points, sigma=sigma)

        self.assertIsInstance(implementation, torch.Tensor)
        self.assertTrue(torch.allclose(naive, implementation))

    def test_sample_from_smoothed_points_2d(self):
        dim = 2
        m = 100
        sigma = 1.
        square_size = 32
        frame_res = torch.Size([square_size, square_size])
        points = square_size*torch.rand(m, dim), torch.rand(m)

        frame = im.Utilities.sampling.sample_from_smoothed_points(points, frame_res, sigma=sigma)

        self.assertIsInstance(frame, torch.Tensor)
        self.assertTrue(frame.shape, frame_res)

    def test_fill_area_uniform(self):
        # Defines a square [0, 1]x[0, 1]
        def area(pos):
            return torch.where((pos[:, 0] >= 0.) & (pos[:, 0] <= 1.) &
                               (pos[:, 1] >= 0.) & (pos[:, 1] <= 1.),
                               torch.tensor([1.]), torch.tensor([0.])).byte()

        def area_circle(pos):
            return torch.where((torch.sqrt(pos[:, 0]**2 + pos[:, 1]**2) <= 0.5),
                               torch.tensor([1.]), torch.tensor([0.])).byte()

        # AABB made bigger on purpose in order to test the rejection
        aabb = im.Utilities.usefulfunctions.AABB(-1., 2., -1., 2.)

        filled = im.Utilities.sampling.fill_area_uniform(area, aabb, 0.2)
        
        self.assertIsInstance(filled, torch.Tensor)
        self.assertEqual(filled.shape[0], 36)
        self.assertTrue(torch.all(area(filled)))

        filled_circle = im.Utilities.sampling.fill_area_uniform(area_circle, aabb, 0.2)
        self.assertTrue(torch.all(area_circle(filled_circle)))

    def test_fill_area_random(self):
        # Defines a square [0, 1]x[0, 1]
        def area(pos):
            return torch.where((pos[:, 0] >= -0.5) & (pos[:, 0] <= 1.) &
                               (pos[:, 1] >= 0.) & (pos[:, 1] <= 1.),
                               torch.tensor([1.]), torch.tensor([0.])).byte()

        def area_circle(pos):
            return torch.where((torch.sqrt(pos[:, 0]**2 + pos[:, 1]**2) <= 0.5),
                               torch.tensor([1.]), torch.tensor([0.])).byte()
        
        # AABB made bigger on purpose in order to test the rejection
        aabb = im.Utilities.usefulfunctions.AABB(-1., 2., -1., 2.)

        filled = im.Utilities.sampling.fill_area_random(area, aabb, 20)

        self.assertIsInstance(filled, torch.Tensor)
        self.assertTrue(torch.all(area(filled)))

        filled_circle = im.Utilities.sampling.fill_area_random(area_circle, aabb, 100)

        self.assertTrue(torch.all(area_circle(filled_circle)))


if __name__ == '__main__':
    unittest.main()
