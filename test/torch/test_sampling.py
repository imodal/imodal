import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch
import numpy as np

import implicitmodules.torch as im

class TestSampling(unittest.TestCase):
    def test_load_greyscale(self):
        img_8 = im.Utilities.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png")
        self.assertIsInstance(img_8, torch.Tensor)
        self.assertEqual(img_8.shape, torch.Size([16, 16]))

        # load_greyscale_image should be able to handle multichannel images
        # (and simply take the red channel)
        img_24 = im.Utilities.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_24.png")
        self.assertEqual(img_24.shape, torch.Size([16, 16]))

    def test_sample_from_greyscale(self):
        img = im.Utilities.load_greyscale_image(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png")
    
        points = im.Utilities.sample_from_greyscale(img, 0.)
        self.assertIsInstance(points, tuple)
        self.assertEqual(len(points), 2)
        self.assertIsInstance(points[0], torch.Tensor)
        self.assertIsInstance(points[1], torch.Tensor)
        self.assertEqual(points[0].shape, torch.Size([16*16, 2]))
        self.assertEqual(points[1].shape, torch.Size([16*16]))
    
        points = im.Utilities.sample_from_greyscale(img, 1.1)
        self.assertEqual(points[0].shape, torch.Size([0, 2]))
    
        points = im.Utilities.sample_from_greyscale(img, 0.2, centered=True)
        self.assertTrue(torch.allclose(torch.sum(points[0], dim=0), torch.zeros(1, 2)))
    
        points = im.Utilities.sample_from_greyscale(img, 0.2, normalise_weights=True)
        self.assertTrue(torch.allclose(torch.sum(points[1]), torch.tensor(1.)))

    def test_load_and_sample_greyscale(self):
        points = im.Utilities.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png", threshold=0.)
        self.assertIsInstance(points, tuple)
        self.assertEqual(len(points), 2)
        self.assertIsInstance(points[0], torch.Tensor)
        self.assertIsInstance(points[1], torch.Tensor)
        self.assertEqual(points[0].shape, torch.Size([16*16, 2]))
        self.assertEqual(points[1].shape, torch.Size([16*16]))
    
        points = im.Utilities.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png", threshold=1.1)
        self.assertEqual(points[0].shape, torch.Size([0, 2]))
    
        points = im.Utilities.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png",
            0.2, centered=True)
        self.assertTrue(torch.allclose(torch.sum(points[0], dim=0), torch.zeros(1, 2)))
    
        points = im.Utilities.load_and_sample_greyscale(
            os.path.dirname(os.path.abspath(__file__)) + "/data/16x16_greyscale_8.png",
            0.2, normalise_weights=True)
        self.assertTrue(torch.allclose(torch.sum(points[1]), torch.tensor(1.)))


if __name__ == '__main__':
    unittest.main()
