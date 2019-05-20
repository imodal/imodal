import torch
import torch.optim

from implicitmodules.torch.Manifolds import Landmarks
from .deformationmodules import SilentPoints, CompoundModule
from .hamiltonian import Hamiltonian
from .kernels import distances, scal
from .sampling import sample_from_greyscale, deformed_intensities
from .shooting import shoot
from .usefulfunctions import grid2vec, vec2grid, close_shape


def fidelity(a, b):
    """Energy Distance between two sampled probability measures."""
    x_i, a_i = a
    y_j, b_j = b
    K_xx = -distances(x_i, x_i)
    K_xy = -distances(x_i, y_j)
    K_yy = -distances(y_j, y_j)
    cost = .5*scal(a_i, torch.mm(K_xx, a_i.view(-1, 1))) - scal(a_i, torch.mm(K_xy, b_j.view(-1, 1))) + .5*scal(b_j, torch.mm(K_yy, b_j.view(-1, 1)))
    return cost


def L2_norm_fidelity(a, b):
    return torch.dist(a, b)


def cost_varifold(x, y, sigma):
    def dot_varifold(x, y, sigma):
        cx, cy = close_shape(x), close_shape(y)
        nx, ny = x.shape[0], y.shape[0]

        vx, vy = cx[1:nx + 1, :] - x, cy[1:ny + 1, :] - y
        mx, my = (cx[1:nx + 1, :] + x) / 2, (cy[1:ny + 1, :] + y) / 2

        xy = torch.tensordot(torch.transpose(torch.tensordot(mx, my, dims=0), 1, 2), torch.eye(2))

        d2 = torch.sum(mx * mx, dim=1).reshape(nx, 1).repeat(1, ny) + torch.sum(my * my, dim=1).repeat(nx, 1) - 2 * xy

        kxy = torch.exp(-d2 / (2 * sigma ** 2))

        vxvy = torch.tensordot(torch.transpose(torch.tensordot(vx, vy, dims=0), 1, 2), torch.eye(2)) ** 2

        nvx = torch.sqrt(torch.sum(vx * vx, dim=1))
        nvy = torch.sqrt(torch.sum(vy * vy, dim=1))

        mask = vxvy > 0

        cost = torch.sum(kxy[mask] * vxvy[mask] / (torch.tensordot(nvx, nvy, dims=0)[mask]))
        return cost

    return dot_varifold(x, x, sigma) + dot_varifold(y, y, sigma) - 2 * dot_varifold(x, y, sigma)


class Model():
    def __init__(self, attachement):
        self.__attachement = attachement

    @property
    def attachement(self):
        return self.__attachement

    def compute(self):
        raise NotImplementedError

    def __call__(self, reverse=False):
        raise NotImplementedError

    def transform_target(self, target):
        return target

    def fit(self, target, lr=1e-3, l=1., max_iter=100, tol=1e-7, log_interval=10):
        transformed_target = self.transform_target(target)

        optimizer = torch.optim.LBFGS(self.parameters, lr=lr, max_iter=4)
        self.nit = -1
        self.break_loop = False
        costs = []

        def closure():
            self.nit += 1
            optimizer.zero_grad()
            self.compute(transformed_target)
            attach = l*self.attach
            cost = attach + self.deformation_cost
            if(self.nit%log_interval == 0):
                print("It: %d, deformation cost: %.6f, attach: %.6f. Total cost: %.6f" % (self.nit, self.deformation_cost.detach().numpy(), attach.detach().numpy(), cost.detach().numpy()))

            costs.append(cost.item())

            if(len(costs) > 1 and abs(costs[-1] - costs[-2]) < tol) or self.nit >= max_iter:
                self.break_loop = True
            else:
                cost.backward(retain_graph=True)

            return cost

        for i in range(0, max_iter):
            optimizer.step(closure)

            if(self.break_loop):
                break

        print("End of the optimisation process.")
        return costs


class ModelCompound(Model):
    def __init__(self, modules, fixed, attachement):
        super().__init__(attachement)
        self.__modules = modules
        self.__fixed = fixed

        self.__init_manifold = CompoundModule(self.__modules).manifold.copy()

        self.__parameters = []

        for i in range(len(self.__modules)):
            self.__parameters.extend(self.__init_manifold[i].unroll_cotan())
            if(not self.__fixed[i]):
                self.__parameters.extend(self.__init_manifold[i].unroll_gd())

    @property
    def modules(self):
        return self.__modules

    @property
    def fixed(self):
        return self.__fixed

    @property
    def init_manifold(self):
        return self.__init_manifold

    @property
    def parameters(self):
        return self.__parameters

    @property
    def shot(self):
        return self.__shot

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=2, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentPoints(grid_landmarks)
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        intermediate = shoot(Hamiltonian([grid_silent, *compound]), 10, "torch_euler")

        return vec2grid(grid_landmarks.gd.view(-1, 2).detach(), grid_resolution[0], grid_resolution[1])


class ModelCompoundWithPointsRegistration(ModelCompound):
    def __init__(self, source, module_list, fixed, attachement):
        self.alpha = source[1]

        module_list.insert(0, SilentPoints(Landmarks(2, source[0].shape[0], gd=source[0].view(-1).requires_grad_())))
        fixed.insert(0, True)

        super().__init__(module_list, fixed, attachement)

    def compute(self, target):
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)
        h = Hamiltonian(compound)
        shoot(h, 10, "torch_euler")
        self.__shot_points = compound[0].manifold.gd.view(-1, 2)
        self.shot_manifold = compound.manifold.copy()
        self.deformation_cost = compound.cost()
        self.attach = self.attachement((self.__shot_points, self.alpha), target)

    def __call__(self):
        return self.__shot_points, self.alpha


class ModelCompoundImageRegistration(ModelCompound):
    def __init__(self, source_image, modules, fixed, lossFunc, img_transform=lambda x: x):
        self.__frame_res = source_image.shape
        self.__source = sample_from_greyscale(source_image.clone(), 0., centered=False, normalise_weights=False, normalise_position=False)
        self.__img_transform = img_transform
        super().__init__(modules, fixed, lossFunc)

    def transform_target(self, target):
        return self.__img_transform(target)

    def compute(self, target):
        # First, forward step shooting only the deformation modules
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)
        shoot(Hamiltonian(compound), it=4)
        self.shot_manifold = compound.manifold.copy()

        # Prepare for reverse shooting
        compound.manifold.negate_cotan()

        image_landmarks = Landmarks(2, self.__source[0].shape[0], gd=self.__source[0].view(-1))
        compound = CompoundModule([SilentPoints(image_landmarks), *compound])

        # Then, reverse shooting in order to get the final deformed image
        intermediate = shoot(Hamiltonian(compound), it=8)

        self.__output_image = deformed_intensities(compound[0].manifold.gd.view(-1, 2), self.__source[1].view(self.__frame_res)).clone()

        # Compute attach and deformation cost
        self.attach = self.loss(self.__output_image, target)
        self.deformation_cost = compound.cost()

    def __call__(self):
        return self.__output_image

