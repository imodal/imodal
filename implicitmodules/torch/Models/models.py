import copy

import torch
import torch.optim

from implicitmodules.torch.DeformationModules import CompoundModule, SilentLandmarks
from implicitmodules.torch.HamiltonianDynamic import Hamiltonian, shoot
from implicitmodules.torch.Manifolds import Landmarks
from implicitmodules.torch.Utilities.sampling import sample_from_greyscale, deformed_intensities
from implicitmodules.torch.Utilities.usefulfunctions import grid2vec, vec2grid


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
    def __init__(self, modules, fixed, attachement, parameters=[]):
        super().__init__(attachement)
        self.__modules = modules
        self.__fixed = fixed

        self.__init_manifold = CompoundModule(self.__modules).manifold.copy()
        self.__init_parameters = copy.copy(parameters)

        self.__parameters = []

        for i in range(len(self.__modules)):
            self.__parameters.extend(self.__init_manifold[i].unroll_cotan())
            if(not self.__fixed[i]):
                self.__parameters.extend(self.__init_manifold[i].unroll_gd())

        self.__parameters.extend(parameters)

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
    def init_parameters(self):
        return self.__init_parameters

    @property
    def parameters(self):
        return self.__parameters

    def compute_deformation_grid(self, grid_origin, grid_size, grid_resolution, it=2, intermediate=False):
        x, y = torch.meshgrid([
            torch.linspace(grid_origin[0], grid_origin[0]+grid_size[0], grid_resolution[0]),
            torch.linspace(grid_origin[1], grid_origin[1]+grid_size[1], grid_resolution[1])])

        gridpos = grid2vec(x, y)

        grid_landmarks = Landmarks(2, gridpos.shape[0], gd=gridpos.view(-1))
        grid_silent = SilentLandmarks(grid_landmarks)
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)

        intermediate_states, _ = shoot(Hamiltonian([grid_silent, *compound]), 10, "torch_euler")

        out = [vec2grid(inter[0].gd.detach().view(-1, 2), grid_resolution[0], grid_resolution[1]) for inter in intermediate_states]
        return out
        #return vec2grid(grid_landmarks.gd.view(-1, 2).detach(), grid_resolution[0], grid_resolution[1])


class ModelCompoundWithPointsRegistration(ModelCompound):
    def __init__(self, source, module_list, fixed, attachement, parameters=[]):
        if isinstance(source, list):
            self.__compound_fit = True
            self.__compound_fit_size = len(source)
            self.alpha = []
            for i in range(self.__compound_fit_size):
                self.alpha.insert(i, source[i][1])
                module_list.insert(i, SilentPoints(Landmarks(2, source[i][0].shape[0], gd=source[i][0].view(-1).requires_grad_())))
                fixed.insert(i, True)

        else:
            self.__compound_fit = False
            self.alpha = source[1]
            module_list.insert(0, SilentPoints(Landmarks(2, source[0].shape[0], gd=source[0].view(-1).requires_grad_())))
            fixed.insert(0, True)

        super().__init__(module_list, fixed, attachement, parameters=parameters)

    def compute(self, target):
        compound = CompoundModule(self.modules)
        compound.manifold.fill(self.init_manifold)
        h = Hamiltonian(compound)
        shoot(h, 10, "torch_euler")
        self.shot_manifold = compound.manifold.copy()
        self.deformation_cost = compound.cost()

        if self.__compound_fit:
            self.__shot_points = []
            attach_list = []
            for i in range(self.__compound_fit_size):
                self.__shot_points.append(compound[i].manifold.gd.view(-1, 2))
                attach_list.append(self.attachement[i]((compound[i].manifold.gd.view(-1, 2), self.alpha[i]), target[i]))
            self.attach = sum(attach_list)
        else:
            self.__shot_points = compound[0].manifold.gd.view(-1, 2)
            self.attach = self.attachement((self.__shot_points, self.alpha), target)

    def __call__(self):
        if self.__compound_fit:
            return list(zip(self.__shot_points, self.alpha))
        else:
            return (self.__shot_points, self.alpha)


class ModelCompoundImageRegistration(ModelCompound):
    def __init__(self, source_image, modules, fixed, lossFunc, img_transform=lambda x: x, parameters=[]):
        self.__frame_res = source_image.shape
        self.__source = sample_from_greyscale(source_image.clone(), 0., centered=False, normalise_weights=False, normalise_position=False)
        self.__img_transform = img_transform
        super().__init__(modules, fixed, lossFunc, parameters=parameters)

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
        compound = CompoundModule([SilentLandmarks(image_landmarks), *compound])

        # Then, reverse shooting in order to get the final deformed image
        intermediate = shoot(Hamiltonian(compound), it=8)

        self.__output_image = deformed_intensities(compound[0].manifold.gd.view(-1, 2), self.__source[1].view(self.__frame_res)).clone()

        # Compute attach and deformation cost
        self.attach = self.loss(self.__output_image, target)
        self.deformation_cost = compound.cost()

    def __call__(self):
        return self.__output_image

