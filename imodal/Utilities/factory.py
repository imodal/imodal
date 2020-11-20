class ObjectFactory:
    def __init__(self, builders=None):
        self.__builders = {}
        if builders is not None:
            self.__builders.update(builders)

    def __call__(self, key, **kwargs):
        return spawn(key, kwargs)

    def register_builder(self, key, builder):
        self.__builders[key] = builder

    def spawn(self, key, **kwargs):
        builder = self.__builders.get(key)

        if not builder:
            raise KeyError(key)

        return builder(**kwargs)

