import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch
import matplotlib.pyplot as plt

import implicitmodules.torch as im


# TODO: Add test_winding_order(), even if we know it works and wont be changed in the future.
class TestMeshutils(unittest.TestCase):
    def setUp(self):
        self.test_points = torch.tensor([[0., 0.],
                                         [1., 2.],
                                         [1., 1.],
                                         [2., 1.],
                                         [2., 2.],
                                         [-1., 2.],
                                         [-2., 1.],
                                         [-1., -1.]])

    def test_area_side_direction(self):
        """Test area_side() by specifying the side by origin and direction."""
        origin = torch.tensor([1., 1.])
        direction = torch.tensor([1., -1.])

        expected_left = torch.BoolTensor([False, True, False, True, True, False, False, False])
        expected_intersect_left = torch.BoolTensor([False, True, True, True, True, False, False, False])
        expected_right = torch.BoolTensor([True, False, False, False, False, True, True, True])
        expected_intersect_right = torch.BoolTensor([True, False, True, False, False, True, True, True])

        result_left = im.Utilities.area_side(self.test_points, origin=origin, direction=direction, side=1, intersect=False)
        result_intersect_left = im.Utilities.area_side(self.test_points, origin=origin, direction=direction, side=1, intersect=True)
        result_right = im.Utilities.area_side(self.test_points, origin=origin, direction=direction, side=-1, intersect=False)
        result_intersect_right = im.Utilities.area_side(self.test_points, origin=origin, direction=direction, side=-1, intersect=True)

        self.assertTrue(torch.all(torch.eq(result_left, expected_left)))
        self.assertTrue(torch.all(torch.eq(result_intersect_left, expected_intersect_left)))
        self.assertTrue(torch.all(torch.eq(result_right, expected_right)))
        self.assertTrue(torch.all(torch.eq(result_intersect_right, expected_intersect_right)))

    def test_area_side_position(self):
        """Test area_side() by specifying the side by two points."""
        p0 = torch.tensor([0., 2.])
        p1 = torch.tensor([2., 0.])

        expected_left = torch.BoolTensor([False, True, False, True, True, False, False, False])
        expected_intersect_left = torch.BoolTensor([False, True, True, True, True, False, False, False])
        expected_right = torch.BoolTensor([True, False, False, False, False, True, True, True])
        expected_intersect_right = torch.BoolTensor([True, False, True, False, False, True, True, True])

        result_left = im.Utilities.area_side(self.test_points, p0=p0, p1=p1, side=1, intersect=False)
        result_intersect_left = im.Utilities.area_side(self.test_points, p0=p0, p1=p1, side=1, intersect=True)
        result_right = im.Utilities.area_side(self.test_points, p0=p0, p1=p1, side=-1, intersect=False)
        result_intersect_right = im.Utilities.area_side(self.test_points, p0=p0, p1=p1, side=-1, intersect=True)

        self.assertTrue(torch.all(torch.eq(result_left, expected_left)))
        self.assertTrue(torch.all(torch.eq(result_intersect_left, expected_intersect_left)))
        self.assertTrue(torch.all(torch.eq(result_right, expected_right)))
        self.assertTrue(torch.all(torch.eq(result_intersect_right, expected_intersect_right)))

    def test_area_convex_hull(self):
        # Defines a small house shaped convex shape with some more points inside
        scatter = torch.tensor([[-1., 1.],
                                [0., 2.],
                                [1., 1.],
                                [0., 0.],
                                [1., 0.],
                                [0., 1.],
                                [1., -1.],
                                [-1., -1.]])

        expected = torch.BoolTensor([True, False, False, False, False, False, False, False])
        expected_intersect = torch.BoolTensor([True, False, True, False, False, False, False, True])

        result = im.Utilities.area_convex_hull(self.test_points, scatter=scatter)
        result_intersect = im.Utilities.area_convex_hull(self.test_points, scatter=scatter, intersect=True)

        self.assertTrue(torch.all(torch.eq(result, expected)))
        self.assertTrue(torch.all(torch.eq(result_intersect, expected_intersect)))

    def test_area_convex_shape_cw(self):
        # Defines a small house shaped convex CW shape
        shape = torch.tensor([[-1., 1.],
                                [0., 2.],
                                [1., 1.],
                                [1., -1.],
                                [-1., -1.]])

        expected = torch.BoolTensor([True, False, False, False, False, False, False, False])
        expected_intersect = torch.BoolTensor([True, False, True, False, False, False, False, True])

        result = im.Utilities.area_convex_shape(self.test_points, shape=shape, intersect=False)
        result_intersect = im.Utilities.area_convex_shape(self.test_points, shape=shape, intersect=True)
        
        self.assertTrue(torch.all(torch.eq(result, expected)))
        self.assertTrue(torch.all(torch.eq(result_intersect, expected_intersect)))

    def test_area_convex_shape_ccw(self):
        # Defines a small house shaped convex CCW shape
        shape = torch.tensor([[0., 2.],
                                [-1., 1.],
                                [-1., -1.],
                                [1., -1.],
                                [1., 1.]])

        expected = torch.BoolTensor([True, False, False, False, False, False, False, False])
        expected_intersect = torch.BoolTensor([True, False, True, False, False, False, False, True])

        result = im.Utilities.area_convex_shape(self.test_points, shape=shape, intersect=False, side=-1)

        result_intersect = im.Utilities.area_convex_shape(self.test_points, shape=shape, intersect=True, side=-1)

        self.assertTrue(torch.all(torch.eq(result, expected)))
        self.assertTrue(torch.all(torch.eq(result_intersect, expected_intersect)))

    def test_area_shape(self):
        shape_ccw = torch.tensor([[0., 1.],
                                  [-1., 1.],
                                  [-1., -1.],
                                  [1., -1.],
                                  [1., 2.]])

        shape_cw = torch.tensor([[-1., 1.],
                                 [0., 1.],
                                 [1., 2.],
                                 [1., -1.],
                                 [-1., -1.]])

        expected_ccw = torch.BoolTensor([True, False, False, False, False, False, False, True])
        expected_cw = torch.BoolTensor([True, False, True, False, False, False, False, False])

        result_ccw = im.Utilities.area_shape(self.test_points, shape=shape_ccw)
        result_cw = im.Utilities.area_shape(self.test_points, shape=shape_cw, side=-1)

        self.assertTrue(torch.all(torch.eq(result_ccw, expected_ccw)))
        self.assertTrue(torch.all(torch.eq(result_cw, expected_cw)))

    def test_area_polyline_outline(self):
        polyline = torch.tensor([[-1., 1.],
                                 [0., 2.],
                                 [1., 1.],
                                 [1., -1.],
                                 [-1., -1.]])

        width = 0.5

        expected = torch.BoolTensor([False, False, True, False, False, False, False, True])
        result = im.Utilities.area_polyline_outline(self.test_points, polyline=polyline, width=width)

        self.assertTrue(torch.all(torch.eq(result, expected)))


    def test_area_disc(self):
        radius = 1.5
        center = torch.tensor([1., 1.])

        expected = torch.BoolTensor([True, True, True, True, True, False, False, False])
        result = im.Utilities.area_disc(self.test_points, center=center, radius=radius)

        self.assertTrue(torch.all(torch.eq(result, expected)))

    def test_area_AABB(self):
        aabb = im.Utilities.AABB(-1., 1., -1., 2.)

        expected = torch.BoolTensor([True, True, True, False, False, True, False, True])
        result = im.Utilities.area_AABB(self.test_points, aabb=aabb)

        self.assertTrue(torch.all(torch.eq(result, expected)))

    def test_area_segment(self):        
        # An horizontal segment going from (-1, 0) to (1, 0)
        segment0_p0 = torch.tensor([-1., 0.])
        segment0_p1 = torch.tensor([1., 0.])

        # A vertical segment going from (0, 2) to (0, -1)
        segment1_p0 = torch.tensor([0., 2.])
        segment1_p1 = torch.tensor([0., -1.])

        # A segment going from (0, 2) to (2, 0)
        segment2_p0 = torch.tensor([0., 2.])
        segment2_p1 = torch.tensor([2., 0.])

        #  A degenerate segment that forms a point
        segment3_p0 = torch.tensor([0., 0.])
        segment3_p1 = torch.tensor([0., 0.])

        width = 1.

        expected0 = torch.BoolTensor([True, False, True, False, False, False, False, True])
        expected1 = torch.BoolTensor([True, True, True, False, False, True, False, True])
        expected2 = torch.BoolTensor([False, True, True, True, False, True, False, False])
        expected3 = torch.BoolTensor([True, False, False, False, False, False, False, False])

        result0 = im.Utilities.area_segment(self.test_points, p0=segment0_p0, p1=segment0_p1, width=width)
        result1 = im.Utilities.area_segment(self.test_points, p0=segment1_p0, p1=segment1_p1, width=width)
        result2 = im.Utilities.area_segment(self.test_points, p0=segment2_p0, p1=segment2_p1, width=width)
        result3 = im.Utilities.area_segment(self.test_points, p0=segment3_p0, p1=segment3_p1, width=width)

        self.assertTrue(torch.all(torch.eq(expected0, result0)))
        self.assertTrue(torch.all(torch.eq(expected1, result1)))
        self.assertTrue(torch.all(torch.eq(expected2, result2)))
        self.assertTrue(torch.all(torch.eq(expected3, result3)))

    def test_convex_hull(self):
        convex_hull = im.Utilities.extract_convex_hull(self.test_points)

        self.assertEqual(len(convex_hull), 5)

        expected = torch.BoolTensor([False, True, False, True, True, True, True, True])
        result = im.Utilities.area_polyline_outline(self.test_points, polyline=convex_hull)

        self.assertTrue(torch.all(torch.eq(result, expected)))

    def test_close_shape(self):
        shape = torch.tensor([[-1., 1.],
                              [0., 2.],
                              [1., 1.],
                              [1., -1.],
                              [-1., -1.]])

        self.assertFalse(im.Utilities.is_shape_closed(shape))
        shape = im.Utilities.close_shape(shape)
        self.assertTrue(im.Utilities.is_shape_closed(shape))

    def test_point_side(self):
        points = torch.tensor([[1., 1.],
                               [0., 0.],
                               [-1., -1]])
        p0 = torch.tensor([0.5, 0.])
        p1 = torch.tensor([1.5, 3.])

        self.assertEqual(im.Utilities.point_side(points[0], p0, p1), -1) 
        self.assertEqual(im.Utilities.point_side(points[1], p0, p1), 1)
        self.assertEqual(im.Utilities.point_side(points[2], p0, p1), 1)

    def test_fill_area_uniform(self):        
        # AABB made bigger on purpose in order to test the rejection
        aabb = im.Utilities.AABB(-1., 2., -1., 2.)

        aabb_area = im.Utilities.AABB(-0.5, 0.5, -0.5, 0.5)

        filled = im.Utilities.fill_area_uniform(im.Utilities.area_AABB, aabb, 0.2, aabb=aabb_area)
        
        self.assertIsInstance(filled, torch.Tensor)
        self.assertEqual(filled.shape[0], 25)
        self.assertTrue(all(im.Utilities.area_AABB(filled, aabb=aabb_area)))

        filled_circle = im.Utilities.fill_area_uniform(im.Utilities.area_disc, aabb, 0.2, center=torch.tensor([0., 0.]), radius=0.5)
        self.assertTrue(all(im.Utilities.area_disc(filled_circle, center=torch.tensor([0., 0.]), radius=0.5)))

    def test_fill_area_random(self):
        # Defines a square [0, 1]x[0, 1]
        def area(pos, **kwargs):
            return torch.where((pos[:, 0] >= -0.5) & (pos[:, 0] <= 1.) &
                               (pos[:, 1] >= 0.) & (pos[:, 1] <= 1.),
                               torch.tensor([1.]), torch.tensor([0.])).byte()

        def area_circle(pos, **kwargs):
            return torch.where((torch.sqrt(pos[:, 0]**2 + pos[:, 1]**2) <= 0.5),
                               torch.tensor([1.]), torch.tensor([0.])).byte()
        
        # AABB made bigger on purpose in order to test the rejection
        aabb = im.Utilities.AABB(-1., 2., -1., 2.)

        filled = im.Utilities.fill_area_random(area, aabb, 20)

        self.assertIsInstance(filled, torch.Tensor)
        self.assertTrue(torch.all(area(filled)))

        filled_circle = im.Utilities.fill_area_random(area_circle, aabb, 100)

        self.assertTrue(torch.all(area_circle(filled_circle)))



