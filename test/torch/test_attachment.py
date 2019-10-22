import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + '..') * 2)

import unittest

import torch

import implicitmodules.torch as im

torch.set_default_dtype(torch.float64)


def make_test_attachment(dim):
    class TestAttachment(unittest.TestCase):
        def setUp(self):
            self.N = 10
            self.source = torch.rand(self.N, dim)
            self.source_weights = torch.rand(self.N)
            self.target = torch.rand(self.N, dim)
            self.target_weights = torch.rand(self.N)

        def test_energy_attachment(self):
            energy = im.Attachment.EnergyAttachment()
            energy.target = (self.source, self.source_weights)
            self.assertTrue(torch.allclose(energy((self.source, self.source_weights)), torch.tensor([0.])))
            energy.target = (self.target, self.target_weights)
            self.assertFalse(torch.allclose(energy((self.source, self.source_weights)), torch.tensor([0.])))

        def test_energy_attachment_gradcheck(self):
            def my_energy(source):
                energy = im.Attachment.EnergyAttachment()
                energy.target = (self.target, self.target_weights)
                return energy((source, self.source_weights))

            self.assertTrue(torch.autograd.gradcheck(my_energy, self.source.requires_grad_()))

        def test_l2_norm_attachment(self):
            l2_norm = im.Attachment.L2NormAttachment()
            l2_norm.target = self.source
            self.assertTrue(torch.allclose(l2_norm(self.source), torch.tensor([0.])))

            l2_norm.target = self.target
            self.assertFalse(torch.allclose(l2_norm(self.source), torch.tensor([0.])))

        def test_l2_norm_attachment_gradcheck(self):
            def my_l2_norm(source, target):
                l2_norm = im.Attachment.L2NormAttachment()
                l2_norm.target = target
                return l2_norm(source)

            self.assertTrue(torch.autograd.gradcheck(my_l2_norm, (self.source.requires_grad_(), self.target.requires_grad_())))

        def test_geomloss_attachment(self):
            pass

        def test_euclidean_attachment(self):
            euclidean = im.Attachment.EuclideanPointwiseDistanceAttachment()
            euclidean.target = self.source
            self.assertTrue(torch.allclose(euclidean(self.source), torch.tensor([0.])))

            euclidean.target = self.target
            self.assertFalse(torch.allclose(euclidean(self.source), torch.tensor([0.])))

        def test_euclidean_attachment_gradcheck(self):
            def my_euclidean(source, target):
                euclidean = im.Attachment.EuclideanPointwiseDistanceAttachment()
                euclidean.target = target
                return euclidean(source)

            self.assertTrue(torch.autograd.gradcheck(my_euclidean, (self.source.requires_grad_(), self.target.requires_grad_())))

    return TestAttachment


class TestAttachment2D(make_test_attachment(2)):
    pass


class TestAttachment3D(make_test_attachment(3)):
    pass


class TestAttachmentVarifold2D(unittest.TestCase):
    def test_attachment(self):
        N = 10
        source = torch.randn(N, 2)
        target = torch.randn(N, 2)

        varifold = im.Attachment.VarifoldAttachment(2, [1.5])
        varifold.target = source
        self.assertTrue(torch.allclose(varifold(source), torch.tensor([0.])))

        varifold.target = target
        self.assertFalse(torch.allclose(varifold(source), torch.tensor([0.])))

    def test_attachment_gradcheck(self):
        def my_varifold(source, target):
            varifold = im.Attachment.VarifoldAttachment(2, [1.5])
            varifold.target = target
            return varifold(source)

        source = torch.rand(5, 2)
        target = torch.rand(3, 2)

        self.assertTrue(torch.autograd.gradcheck(my_varifold, (source.requires_grad_(), target.requires_grad_())))
        

def make_test_attachment_varifold_3d(backend):
    class TestAttachementVarifold3D(unittest.TestCase):
        def setUp(self):
            self.source_vertices = torch.randn(3, 3)
            self.target_vertices = torch.randn(3, 3)
            self.source_faces = torch.LongTensor([[0, 1, 2]])
            self.target_faces = torch.LongTensor([[0, 1, 2]])

        def test_attachment(self):
            varifold = im.Attachment.VarifoldAttachment(3, [1.5], backend=backend)
            varifold.target = (self.source_vertices, self.source_faces)
            self.assertTrue(torch.allclose(varifold((self.source_vertices, self.source_faces)), torch.tensor([0.])))

            varifold.target = (self.target_vertices, self.target_faces)
            self.assertFalse(torch.allclose(varifold((self.source_vertices, self.source_faces)), torch.tensor([0.])))

        def test_attachment_gradcheck(self):
            def my_varifold(source_vertices, target_vertices):
                varifold = im.Attachment.VarifoldAttachment(3, [1.5], backend=backend)
                varifold.target = (target_vertices, self.target_faces)
                return varifold((source_vertices, self.source_faces))

            self.assertTrue(torch.autograd.gradcheck(my_varifold, (self.source_vertices.requires_grad_(), self.target_vertices.requires_grad_())))

    return TestAttachementVarifold3D


class TestAttachementVarifold3D_Torch(make_test_attachment_varifold_3d('torch')):
    pass


class TestAttachementVarifold3D_KeOps(make_test_attachment_varifold_3d('keops')):
    pass

