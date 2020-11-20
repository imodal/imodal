
__compute_backend = 'torch'
__compute_backends = ['keops', 'torch']


def set_compute_backend(backend):
    """ Set the current default kernel computation backend.

    Parameters
    ----------
    backend : str
       Kernel computation backend to use. Either `torch` or `keops`.
    """
    if backend != 'torch' and backend != 'keops':
        raise RuntimeError("Backend", backend, " not supported!")

    global __compute_backend
    __compute_backend = backend


def get_compute_backend():
    """ Returns current kernel computation backend.
    """
    return __compute_backend


def is_valid_backend(backend):
    return backend in __compute_backends

