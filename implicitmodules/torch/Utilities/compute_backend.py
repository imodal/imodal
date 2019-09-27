
__compute_backend = 'torch'
__compute_backends = ['keops', 'torch']


def set_compute_backend(backend):
    if backend != 'torch' and backend != 'keops':
        raise RuntimeError("Backend", backend, " not supported!")

    global __compute_backend
    __compute_backend = backend


def get_compute_backend():
    return __compute_backend


def is_valid_backend(backend):
    return backend in __compute_backends

