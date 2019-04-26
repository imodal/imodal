import numpy as np

from implicitmodules.numpy.Utilities import FunctionsEta as fun_eta


def my_xmy(x, y):
    return (np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)).reshape(-1, 2)


def my_vker(x, k, sig):  # tested
    """ Gaussian radial function and its derivatives.
    vectorized version
    x is a matrix containing positions
    k is the order (0 gives the function at locations x, k=1 its
    first derivatives and k=2 its hessian
    sig is the gaussian size.
    """
    
    x = x / sig
    h = np.asarray(np.exp(-np.sum(x ** 2 / 2, axis=1)))
    r = h  # order 0
    if k == 1:
        r = -np.tile(h.reshape((x.shape[0], 1)), (1, 2)) * x / sig
    elif k == 2:
        th = np.tile(h.reshape((x.shape[0], 1, 1)), (1, 2, 2))
        tI = np.tile(np.eye(2), (x.shape[0], 1, 1))
        r = th * (-tI + np.einsum('ki, kj->kij', x, x)) / sig ** 2
    elif k == 3:
        th = np.tile(h.reshape((x.shape[0], 1, 1)), (1, 2, 2))
        tI = np.tile(np.eye(2), (x.shape[0], 1, 1))
        r = th * (-tI + np.einsum('ki, kj->kij', x, x))
        tth = np.tile(h.reshape((x.shape[0], 1, 1, 1)), (1, 2, 2, 2))
        r = -np.einsum('kij, kl->kijl', r, x) + \
            tth * (np.swapaxes(np.tensordot(x, np.eye(2), axes=0), 1, 2) + \
                   np.tensordot(x, np.eye(2), axes=0))
        r = r / sig ** 3
    return r


def my_K(x, y, sigma, k):
    """ vectorized version of my_K(x,y,sig,k,l) for x (N,2) and k=l
    as need by SKS
    """
    
    N = x.shape[0]
    M = y.shape[0]
    if (k == 0):
        return np.moveaxis(np.einsum('ij, kl->ijkl', my_vker(my_xmy(x, y), 0, sigma).reshape(N, N), np.eye(2)), [0, 1, 2, 3], [0, 2, 1, 3]).reshape(2*M, 2*N)

    elif (k == 1):
        t = np.tensordot(-my_vker(my_xmy(x, x), 2, sigma), np.eye(2), axes=0)
        K = fun_eta.my_Keta(np.swapaxes(t, 2, 3))
        K = np.tensordot(K, fun_eta.my_eta(), axes=([1, 2], [0, 1]))
        return np.moveaxis(K.reshape(N, M, 3, 3), [0, 1, 2, 3], [0, 2, 1, 3]).reshape(3*N, 3*N)

    else:
        raise NotImplementedError

