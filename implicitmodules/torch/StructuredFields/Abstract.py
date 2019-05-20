class StructuredField:
    def __init__(self):
        pass

    def __call__(self, points, k=0):
        raise NotImplementedError


class SupportStructuredField(StructuredField):
    def __init__(self, support, moments):
        super().__init__()
        self.__support = support
        self.__moments = moments

    @property
    def support(self):
        return self.__support

    @property
    def moments(self):
        return self.__moments


class CompoundStructuredField(StructuredField):
    def __init__(self, fields):
        super().__init__()
        self.__fields = fields

    @property
    def fields(self):
        return self.__fields

    @property
    def nb_field(self):
        return len(self.__fields)

    def __getitem__(self, index):
        return self.__fields

    def __call__(self, points, k=0):
        return sum([field(points, k) for field in self.__fields])

