from implicitmodules.torch.Manifolds.Abstract import Manifold
from implicitmodules.torch.StructuredFields.Abstract import CompoundStructuredField


class CompoundManifold(Manifold):
    def __init__(self, manifold_list):
        super().__init__()
        self.__manifold_list = manifold_list
        self.__dim = self.__manifold_list[0].dim
        self.__nb_pts = sum([m.nb_pts for m in self.__manifold_list])
        self.__numel_gd = sum([m.numel_gd for m in self.__manifold_list])
        self.__len_gd = sum([m.len_gd for m in self.__manifold_list])
        self.__dim_gd = tuple(sum((m.dim_gd for m in self.__manifold_list), ()))

    def copy(self):
        manifold_list = [m.copy() for m in self.__manifold_list]
        return CompoundManifold(manifold_list)

    @property
    def manifold_list(self):
        return self.__manifold_list

    @property
    def nb_manifold(self):
        return len(self.__manifold_list)

    def __getitem__(self, index):
        return self.__manifold_list[index]

    @property
    def dim(self):
        return self.__dim

    @property
    def nb_pts(self):
        return self.__nb_pts

    @property
    def numel_gd(self):
        return self.__numel_gd

    @property
    def len_gd(self):
        return self.__len_gd

    @property
    def dim_gd(self):
        return self.__dim_gd

    def unroll_gd(self):
        """Returns a flattened list of all gd tensors."""
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_gd())
        return l

    def unroll_cotan(self):
        l = []
        for man in self.__manifold_list:
            l.extend(man.unroll_cotan())
        return l

    def roll_gd(self, l):
        """Unflattens the list into one suitable for fill_gd() or all *_gd() numerical operations."""
        for man in self.__manifold_list:
            l.append(man.roll_gd(l))
        return l

    def roll_cotan(self, l):
        for man in self.__manifold_list:
            l.append(man.roll_cotan(l))
        return l

    def __get_gd(self):
        return [m.gd for m in self.__manifold_list]

    def __get_tan(self):
        return [m.tan for m in self.__manifold_list]

    def __get_cotan(self):
        return [m.cotan for m in self.__manifold_list]

    def fill(self, manifold, copy=False):
        self.fill_gd(manifold.gd, copy=copy)
        self.fill_tan(manifold.tan, copy=copy)
        self.fill_cotan(manifold.cotan, copy=copy)

    def fill_gd(self, gd, copy=False):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_gd(gd[i], copy=copy)

    def fill_tan(self, tan, copy=False):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_tan(tan[i], copy=copy)

    def fill_cotan(self, cotan, copy=False):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].fill_cotan(cotan[i], copy=copy)

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_gd(gd[i], scale)

    def muladd_tan(self, tan, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_tan(tan[i], scale)

    def muladd_cotan(self, cotan, scale):
        for i in range(len(self.__manifold_list)):
            self.__manifold_list[i].muladd_cotan(cotan[i], scale)

    def negate_gd(self):
        for m in self.__manifold_list:
            m.negate_gd()

    def negate_tan(self):
        for m in self.__manifold_list:
            m.negate_tan()

    def negate_cotan(self):
        for m in self.__manifold_list:
            m.negate_cotan()

    def cot_to_vs(self, sigma):
        return CompoundStructuredField([m.cot_to_vs(sigma) for m in self.__manifold_list])

    def inner_prod_field(self, field):
        return sum([m.inner_prod_field(field) for m in self.__manifold_list])

    def infinitesimal_action(self, field):
        actions = []
        for m in self.__manifold_list:
            actions.append(m.infinitesimal_action(field))

        return CompoundManifold(actions)

