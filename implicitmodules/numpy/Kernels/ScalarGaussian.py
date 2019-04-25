import numpy as np

from implicitmodules.numpy.Utilities import FunctionsEta as fun_eta


def my_tensordotaxes0(x, y):
    """ we assume here that y is (N,d)
    """
    
    if len(x.shape) == 2:
        dx = x.shape[1]
        dy = y.shape[1]
        
        z = np.empty((x.shape[0], dx, dy))
        for i in range(dx):
            for j in range(dy):
                z[:, i, j] = x[:, i] * y[:, j]
    elif len(x.shape) == 3:
        (dx, dpx, dy) = x.shape[1], x.shape[2], y.shape[1]
        z = np.empty((x.shape[0], dx, dpx, dy))
        for i in range(dx):
            for ip in range(dpx):
                for j in range(dy):
                    z[:, i, ip, j] = x[:, i, ip] * y[:, j]
    return z


def my_xmy(x, y):
    (n, d) = x.shape
    (m, d) = y.shape
    xmy = np.empty((n * m, d))
    for i in range(d):
        xmy[:, i] = (np.tile(x[:, i].reshape((n, 1)), (1, m)) -
                     np.tile(y[:, i].reshape((1, m)), (n, 1))).flatten()
    return xmy


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
        r = th * (-tI + my_tensordotaxes0(x, x)) / sig ** 2
    elif k == 3:
        th = np.tile(h.reshape((x.shape[0], 1, 1)), (1, 2, 2))
        tI = np.tile(np.eye(2), (x.shape[0], 1, 1))
        r = th * (-tI + my_tensordotaxes0(x, x))
        tth = np.tile(h.reshape((x.shape[0], 1, 1, 1)), (1, 2, 2, 2))
        r = -my_tensordotaxes0(r, x) + \
            tth * (np.swapaxes(np.tensordot(x, np.eye(2), axes=0), 1, 2)
                   + np.tensordot(x, np.eye(2), axes=0))
        r = r / sig ** 3
    return r
    
    # Main kernel dot product


def my_K(x, y, sig, k):  # tested
    """ vectorized version of my_K(x,y,sig,k,l) for x (N,2) and k=l
    as need by SKS
    """
    
    N = x.shape[0]
    M = y.shape[0]
    if (k == 0):
        K = np.zeros((N * M, 2, 2))
        r = my_vker(my_xmy(x, x), 0, sig)
        K[:, 0, 0], K[:, 1, 1] = r, r
        fK = K.flatten()
        K = np.zeros((2 * N, 2 * M))
        for i in range(2):
            for j in range(2):
                K[i::2, j::2] = fK[(j + 2 * i)::4].reshape((N, M))
    elif (k == 1):
        t = np.tensordot(-my_vker(my_xmy(x, x), 2, sig), np.eye(2), axes=0)
        K = fun_eta.my_Keta(np.swapaxes(t, 2, 3))
        K = np.tensordot(K, fun_eta.my_eta(), axes=([1, 2], [0, 1]))
        fK = K.flatten()
        K = np.zeros((3 * N, 3 * M))
        for i in range(3):
            for j in range(3):
                K[i::3, j::3] = fK[(j + 3 * i)::9].reshape((N, M))
    return K
