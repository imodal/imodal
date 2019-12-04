from implicitmodules.torch.StructuredFields.Abstract import CompoundStructuredField


class CompoundManifold():
    def __init__(self, manifolds):
        super().__init__()
        self.__manifolds = manifolds

    def to_(self, device):
        [man.to_(device) for man in self.__manifolds]

    @property
    def device(self):
        return self.__manifolds[0].device

    def copy(self, requires_grad=True):
        manifolds = [m.copy(requires_grad=requires_grad) for m in self.__manifolds]
        return CompoundManifold(manifolds)

    @property
    def manifolds(self):
        return self.__manifolds

    @property
    def nb_manifold(self):
        return len(self.__manifolds)

    def __getitem__(self, index):
        return self.__manifolds[index]

    @property
    def dim(self):
        return self.__manifolds[0].dim

    @property
    def nb_pts(self):
        return sum([m.nb_pts for m in self.__manifolds])

    @property
    def len_gd(self):
        return sum([m.len_gd for m in self.__manifolds])

    @property
    def numel_gd(self):
        return tuple(sum((m.numel_gd for m in self.__manifolds), ()))

    def unroll_gd(self):
        """Returns a flattened list of all gd tensors."""
        l = []
        for man in self.__manifolds:
            l.extend(man.unroll_gd())
        return l

    def unroll_cotan(self):
        l = []
        for man in self.__manifolds:
            l.extend(man.unroll_cotan())
        return l

    def roll_gd(self, l):
        """Unflattens the list into one suitable for fill_gd() or all *_gd() numerical operations."""
        out = []
        for man in self.__manifolds:
            out.append(man.roll_gd(l))
        return out

    def roll_cotan(self, l):
        out = []
        for man in self.__manifolds:
            out.append(man.roll_cotan(l))
        return out

    def __get_gd(self):
        return [m.gd for m in self.__manifolds]

    def __get_tan(self):
        return [m.tan for m in self.__manifolds]

    def __get_cotan(self):
        return [m.cotan for m in self.__manifolds]

    def fill(self, manifold, copy=False, requires_grad=True):
        self.fill_gd(manifold.gd, copy=copy, requires_grad=requires_grad)
        self.fill_tan(manifold.tan, copy=copy, requires_grad=requires_grad)
        self.fill_cotan(manifold.cotan, copy=copy, requires_grad=requires_grad)

    def fill_gd(self, gd, copy=False, requires_grad=True):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].fill_gd(gd[i], copy=copy, requires_grad=requires_grad)

    def fill_tan(self, tan, copy=False, requires_grad=True):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].fill_tan(tan[i], copy=copy, requires_grad=requires_grad)

    def fill_cotan(self, cotan, copy=False, requires_grad=True):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].fill_cotan(cotan[i], copy=copy, requires_grad=requires_grad)

    gd = property(__get_gd, fill_gd)
    tan = property(__get_tan, fill_tan)
    cotan = property(__get_cotan, fill_cotan)

    def muladd_gd(self, gd, scale):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].muladd_gd(gd[i], scale)

    def muladd_tan(self, tan, scale):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].muladd_tan(tan[i], scale)

    def muladd_cotan(self, cotan, scale):
        for i in range(len(self.__manifolds)):
            self.__manifolds[i].muladd_cotan(cotan[i], scale)

    def negate_gd(self):
        for m in self.__manifolds:
            m.negate_gd()

    def negate_tan(self):
        for m in self.__manifolds:
            m.negate_tan()

    def negate_cotan(self):
        for m in self.__manifolds:
            m.negate_cotan()

    def cot_to_vs(self, sigma, backend=None):
        return CompoundStructuredField([m.cot_to_vs(sigma, backend=backend) for m in self.__manifolds])

    def inner_prod_field(self, field):
        return sum([m.inner_prod_field(field) for m in self.__manifolds])

    def infinitesimal_action(self, field):
        actions = []
        for m in self.__manifolds:
            actions.append(m.infinitesimal_action(field))

        return CompoundManifold(actions)

