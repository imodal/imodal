from typing import Iterable

import torch

from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields import StructuredField_m, StructuredField_0
from implicitmodules.torch.StructuredFields.Abstract import CompoundStructuredField


class Stiefel(Manifold):
    def __init__(self, dim, nb_pts, gd=None, tan=None, cotan=None, device=None):
        assert (gd is None) or (isinstance(gd, Iterable) and (len(gd) == 2))
        assert (tan is None) or (isinstance(tan, Iterable) and (len(tan) == 2))
        assert (cotan is None) or (isinstance(cotan, Iterable) and (len(cotan) == 2))
        assert (gd is None) or ((gd[0].shape[0] == dim * nb_pts) and (gd[1].shape[0] == dim * dim * nb_pts))
        assert (tan is None) or ((tan[0].shape[0] == dim * nb_pts) and (tan[1].shape[0] == dim * dim * nb_pts))
        assert (cotan is None) or ((cotan[0].shape[0] == dim * nb_pts) and (cotan[1].shape[0] == dim * dim * nb_pts))

        self.__device = self.__find_device(gd, tan, cotan, device)

        self.__dim = dim
        self.__nb_pts = nb_pts

        self.__point_shape = torch.Size([self.__nb_pts, self.__dim])
        self.__mat_shape = torch.Size([self.__nb_pts, self.__dim, self.__dim])

        self.__numel_gd_points = self.__nb_pts * self.__dim
        self.__numel_gd_mat = self.__nb_pts * self.__dim * self.__dim
        self.__numel_gd = self.__numel_gd_points + self.__numel_gd_mat

        if gd is not None:
            self.fill_gd(gd, copy=False)
        else:
            self.__gd = (torch.zeros(self.__numel_gd_points, requires_grad=True, device=self.__device),
                         torch.zeros(self.__numel_gd_mat, requires_grad=True, device=self.__device))

        # TODO: tan is only rarely used. Maybe do something about this?
        if tan is not None:
            self.fill_tan(tan, copy=False)
        else:
            self.__tan = (torch.zeros(self.__numel_gd_points, requires_grad=True, device=self.__device),
                          torch.zeros(self.__numel_gd_mat, requires_grad=True, device=self.__device))

        if cotan is not None:
            self.fill_cotan(cotan, copy=False)
        else:
            self.__cotan = (torch.zeros(self.__numel_gd_points, requires_grad=True, device=self.__device),
                            torch.zeros(self.__numel_gd_mat, requires_grad=True, device=self.__device))

    def __find_device(self, gd, tan, cotan, device):
        if device is None:
            # Device is not specified, we need to get it from the tensors
            cur_device = None
            if gd is not None:
                if gd[0].device != gd[1].device:
                    raise RuntimeError("Stiefel.__init__(): gd[0] and gd[1] are not on the same device.")
                cur_device = gd[0].device
            elif tan is not None:
                if tan[0].device != tan[1].device:
                    raise RuntimeError("Stiefel.__init__(): tan[0] and tan[1] are not on the same device.")
                cur_device = tan.device
            elif cotan is not None:
                if cotan[0].device != cotan[1].device:
                    raise RuntimeError("Stiefel.__init__(): cotan[0] and cotan[1] are not on the same device.")
                cur_device = cotan.device
            else:
                return None

            # We now compare the device with the other tensors and see if it corresponds
            if ((gd is not None) and ((gd[0].device != cur_device) or (gd[1].device != cur_device))):
                raise RuntimeError("Stiefel.__init__(): gd is not on device" + str(device))
            if ((tan is not None) and ((tan[0].device != cur_device) or (tan[1].device != cur_device))):
                raise RuntimeError("Stiefel.__init__(): tan is not on device" + str(device))
            if ((cotan is not None) and ((cotan[0].device != cur_device) or (cotan[1].device != cur_device))):
                raise RuntimeError("Stiefel.__init__(): cotan is not on device" + str(device))

            return cur_device

        else:
            if gd is not None:
                gd[0].to(device=device)
                gd[1].to(device=device)
            if tan is not None:
                tan[0].to(device=device)
                tan[1].to(device=device)
            if cotan is not None:
                cotan[0].to(device=device)
                cotan[1].to(device=device)

            return device

    def to(self, device):
        self.__device = device
        self.__gd.to(device)
        self.__tan.to(device)
        self.__cotan.to(device)

    @property
    def device(self):
        return self.__device

    def copy(self, requires_grad=True):
        out = Stiefel(self.__dim, self.__nb_pts)
        out.fill(self, copy=True, requires_grad=requires_grad)
        return out

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def dim(self):
        return self.__dim

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def numel_gd_points(self):
        return self.__numel_gd_points

    @property
    def numel_gd_mat(self):
        return self.__numel_gd_mat

    @property
    def len_gd(self):
        return 2

    @property
    def dim_gd(self):
        return (self.__numel_gd_points, self.__numel_gd_mat)

    def unroll_gd(self):
        return [self.__gd[0], self.__gd[1]]

    def unroll_tan(self):
        return [self.__tan[0], self.__tan[1]]

    def unroll_cotan(self):
        return [self.__cotan[0], self.__cotan[1]]

    def roll_gd(self, l):
        return [l.pop(0), l.pop(0)]

    def roll_tan(self, l):
        return [l.pop(0), l.pop(0)]

    def roll_cotan(self, l):
        return [l.pop(0), l.pop(0)]

    def __get_gd(self):
        return self.__gd

    def __get_tan(self):
        return self.__tan

    def __get_cotan(self):
        return self.__cotan

    def fill(self, manifold, copy=False, requires_grad=True):
        assert isinstance(manifold, Stiefel)
        self.fill_gd(manifold.gd, copy=copy, requires_grad=requires_grad)
        self.fill_tan(manifold.tan, copy=copy, requires_grad=requires_grad)
        self.fill_cotan(manifold.cotan, copy=copy, requires_grad=requires_grad)

    def fill_gd(self, gd, copy=False, requires_grad=True):
        assert isinstance(gd, Iterable) and (len(gd) == 2) and (gd[0].numel() == self.__numel_gd_points) and (gd[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__gd = gd
        else:
            self.__gd = (gd[0].detach().clone().requires_grad_(requires_grad),
                         gd[1].detach().clone().requires_grad_(requires_grad))

    def fill_tan(self, tan, copy=False, requires_grad=True):
        assert isinstance(tan, Iterable) and (len(tan) == 2) and (tan[0].numel() == self.__numel_gd_points) and (tan[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__tan = tan
        else:
            self.__tan = (tan[0].detach().clone().requires_grad_(requires_grad),
                          tan[1].detach().clone().requires_grad_(requires_grad))

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        assert isinstance(cotan, Iterable) and (len(cotan) == 2) and (cotan[0].numel() == self.__numel_gd_points) and (cotan[1].numel() == self.__numel_gd_mat)
        if not copy:
            self.__cotan = cotan
        else:
            self.__cotan = (cotan[0].detach().clone().requires_grad_(requires_grad),
                            cotan[1].detach().clone().requires_grad_(requires_grad))

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        self.__gd = (self.__gd[0] + scale * gd[0], self.__gd[1] + scale * gd[1])

    def muladd_tan(self, tan, scale):
        self.__tan = (self.__tan[0] + scale * tan[0], self.__tan[1] + scale * tan[1])

    def muladd_cotan(self, cotan, scale):
        self.__cotan = (self.__cotan[0] + scale * cotan[0], self.__cotan[1] + scale * cotan[1])

    def negate_gd(self):
        self.__gd = (-self.__gd[0], -self.__gd[1])

    def negate_tan(self):
        self.__tan = (-self.__tan[0], -self.__tan[1])

    def negate_cotan(self):
        self.__cotan = (-self.__cotan[0], -self.__cotan[1])

    def cot_to_vs(self, sigma, backend=None):
        v0 = StructuredField_0(self.__gd[0].view(-1, self.__dim), self.__cotan[0].view(-1, self.__dim), sigma, backend=backend)
        R = torch.einsum('nik, njk->nij', self.__cotan[1].view(-1, self.__dim, self.__dim), self.__gd[1].view(-1, self.__dim, self.__dim))

        vm = StructuredField_m(self.__gd[0].view(-1, self.__dim), R, sigma, backend=backend)

        return CompoundStructuredField([v0, vm])

    def inner_prod_field(self, field):
        man = self.infinitesimal_action(field)

        return torch.dot(self.cotan[0].view(-1), man.tan[0].view(-1)) + \
            torch.einsum('nij, nij->', self.cotan[1].view(-1, self.__dim, self.__dim), man.tan[1].view(-1, self.__dim, self.__dim))

    def infinitesimal_action(self, field):
        """Applies the vector field generated by the module on the landmark."""
        vx = field(self.__gd[0].view(-1, self.__dim))
        d_vx = field(self.__gd[0].view(-1, self.__dim), k=1)

        S = 0.5 * (d_vx - torch.transpose(d_vx, 1, 2))
        vr = torch.bmm(S, self.__gd[1].view(-1, self.__dim, self.__dim))

        return Stiefel(self.__dim, self.__nb_pts, gd=self.__gd, tan=(vx.view(-1), vr.view(-1)))

