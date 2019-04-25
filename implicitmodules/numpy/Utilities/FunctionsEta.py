import numpy as np


def my_eta():  # symmetric case
    eta = np.zeros((2, 2, 3))
    c = 1 / np.sqrt(2)
    eta[0, 0, 0], eta[0, 1, 1], eta[1, 0, 1], eta[1, 1, 2] = 1., c, c, 1.
    return eta


def my_skew_eta():  # skew-symmetric case
    skew_eta = np.zeros((2, 2, 1))
    c = 1 / np.sqrt(2)
    skew_eta[0, 1, 0], skew_eta[1, 0, 0] = -c, +c
    return skew_eta


def my_Keta(K):  # transformation on the right (sym case)
    return np.tensordot(K, my_eta(), axes=2)


def my_Kskew_eta(K):  # same (skew-sym case)
    return np.tensordot(K, my_skew_eta(), axes=2)


def my_etaK(K):  # Now on the left
    return np.tensordot(my_eta().transpose(), K, axes=2)


def my_skew_etaK(K):  # Same for skew-sym case
    return np.tensordot(my_skew_eta().transpose(), K, axes=2)


def my_etaKeta(K):  # Now on both sides (sym case)
    return my_etaK(my_Keta(K))


def my_skew_etaKeta(K):  # Same mixed case
    return my_skew_etaK(my_Keta(K))
